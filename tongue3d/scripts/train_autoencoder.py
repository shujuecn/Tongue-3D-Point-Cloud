from __future__ import annotations

import math
import time

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tongue3d.config import (
    ensure_output_dir,
    load_autoencoder_config,
    resolve_config_path,
    resolve_device,
    save_config_json,
)
from tongue3d.data import TonguePointCloudDataset, save_splits_csv
from tongue3d.losses import (
    chamfer_with_neighbors,
    edge_length_regularizer,
    laplacian_smoothness_loss,
    normal_alignment_loss,
    repulsion_loss,
)
from tongue3d.models import TonguePointAutoEncoder
from tongue3d.scripts.common import (
    append_metrics_csv,
    build_splits,
    compute_train_normalization,
    create_run_dir,
    format_seconds,
    make_grad_scaler,
    make_loader,
    maybe_autocast,
    maybe_create_summary_writer,
    save_normalization_json,
)
from tongue3d.utils import (
    denormalize_points,
    has_matplotlib,
    save_autoencoder_visual,
    save_checkpoint,
    seed_everything,
    write_pointcloud_ply,
)


def compute_loss(
    points: torch.Tensor,
    normals: torch.Tensor,
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    cfg,
) -> dict[str, torch.Tensor]:
    chamfer, min_pred, min_gt, idx_pred_to_gt, idx_gt_to_pred = chamfer_with_neighbors(
        recon_points,
        points,
        chunk_size=cfg.loss.chamfer_chunk_size,
    )
    normal = normal_alignment_loss(
        pred_normals=recon_normals,
        gt_normals=normals,
        idx_pred_to_gt=idx_pred_to_gt,
        idx_gt_to_pred=idx_gt_to_pred,
    )
    lap = laplacian_smoothness_loss(recon_points)
    edge = edge_length_regularizer(recon_points)
    repel = repulsion_loss(recon_points)
    precision = (min_pred <= cfg.loss.fscore_threshold).float().mean()
    recall = (min_gt <= cfg.loss.fscore_threshold).float().mean()
    fscore = (2.0 * precision * recall) / (precision + recall + 1e-8)
    cd_l1 = min_pred.mean() + min_gt.mean()

    total = (
        cfg.loss.chamfer * chamfer
        + cfg.loss.normal * normal
        + cfg.loss.laplacian * lap
        + cfg.loss.edge * edge
        + cfg.loss.repulsion * repel
    )
    return {
        "total": total,
        "chamfer": chamfer,
        "normal": normal,
        "laplacian": lap,
        "edge": edge,
        "repulsion": repel,
        "cd_l1": cd_l1,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }


def run_epoch(model, loader, optimizer, scaler, device: str, cfg, is_train: bool):
    phase = "train" if is_train else "val"
    model.train(is_train)

    meter = {
        "total": 0.0,
        "chamfer": 0.0,
        "normal": 0.0,
        "laplacian": 0.0,
        "edge": 0.0,
        "repulsion": 0.0,
        "cd_l1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fscore": 0.0,
    }

    pbar = tqdm(loader, desc=phase, leave=False)
    grad_accum = int(cfg.grad_accum_steps)
    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar, start=1):
        points = batch["points"].to(device, non_blocking=True).float()
        normals = batch["normals"].to(device, non_blocking=True).float()

        with maybe_autocast(device=device, enabled=cfg.runtime.amp):
            _, recon_points, recon_normals = model(points)
            losses = compute_loss(points, normals, recon_points, recon_normals, cfg)

        if is_train:
            scaled_loss = losses["total"] / float(grad_accum)
            scaler.scale(scaled_loss).backward()
            should_step = (step % grad_accum == 0) or (step == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        bs = points.shape[0]
        for key in meter:
            meter[key] += float(losses[key].detach().item()) * bs

        running = meter["total"] / max(1, step * loader.batch_size)
        pbar.set_postfix(total=f"{running:.5f}")

    denom = len(loader.dataset)
    return {k: v / max(1, denom) for k, v in meter.items()}


def main() -> None:
    config_path = resolve_config_path("configs/autoencoder.yaml")
    cfg = load_autoencoder_config(config_path)

    ensure_output_dir(cfg.output_dir)
    run_dir = create_run_dir(cfg.output_dir, cfg.experiment_name)
    save_config_json(run_dir / "config.json", cfg)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.runtime.device)

    splits = build_splits(cfg.dataset, cfg.split)
    if len(splits["train"]) == 0 or len(splits["val"]) == 0:
        raise ValueError("Dataset split produced empty train or val set")

    save_splits_csv(splits, run_dir / "splits.csv")

    center, scale = compute_train_normalization(splits["train"])
    save_normalization_json(run_dir / "normalization.json", center, scale)

    train_ds = TonguePointCloudDataset(
        samples=splits["train"],
        dataset_cfg=cfg.dataset,
        center=center,
        scale=scale,
        preload_meshes=cfg.dataset.preload_meshes,
        deterministic_sampling=False,
    )
    val_ds = TonguePointCloudDataset(
        samples=splits["val"],
        dataset_cfg=cfg.dataset,
        center=center,
        scale=scale,
        preload_meshes=cfg.dataset.preload_meshes,
        deterministic_sampling=True,
    )

    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, runtime_cfg=cfg.runtime)
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, runtime_cfg=cfg.runtime)

    model = TonguePointAutoEncoder(
        latent_dim=cfg.model.latent_dim,
        num_points=cfg.dataset.num_points,
        decoder_hidden_dim=cfg.model.decoder_hidden_dim,
        dropout=cfg.model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = make_grad_scaler(device=device, enabled=cfg.runtime.amp)

    tb_writer = maybe_create_summary_writer(
        log_dir=run_dir / "tensorboard",
        enabled=cfg.logging.tensorboard,
    )
    metrics_csv_path = run_dir / "metrics.csv"

    best_val = math.inf
    train_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, cfg, is_train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, cfg, is_train=False)

        scheduler.step()
        epoch_sec = time.perf_counter() - epoch_start
        elapsed_sec = time.perf_counter() - train_start

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"lr={lr:.3e} "
            f"train_total={train_metrics['total']:.6f} "
            f"val_total={val_metrics['total']:.6f} "
            f"val_cd_l2={val_metrics['chamfer']:.6f} "
            f"val_cd_l1={val_metrics['cd_l1']:.6f} "
            f"val_f1={val_metrics['fscore']:.4f} "
            f"epoch={format_seconds(epoch_sec)} elapsed={format_seconds(elapsed_sec)}"
        )

        row = {
            "epoch": epoch,
            "lr": lr,
            "epoch_sec": round(epoch_sec, 4),
            "elapsed_sec": round(elapsed_sec, 4),
            "train_total": train_metrics["total"],
            "train_chamfer": train_metrics["chamfer"],
            "train_normal": train_metrics["normal"],
            "train_laplacian": train_metrics["laplacian"],
            "train_edge": train_metrics["edge"],
            "train_repulsion": train_metrics["repulsion"],
            "train_cd_l1": train_metrics["cd_l1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_fscore": train_metrics["fscore"],
            "val_total": val_metrics["total"],
            "val_chamfer": val_metrics["chamfer"],
            "val_normal": val_metrics["normal"],
            "val_laplacian": val_metrics["laplacian"],
            "val_edge": val_metrics["edge"],
            "val_repulsion": val_metrics["repulsion"],
            "val_cd_l1": val_metrics["cd_l1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_fscore": val_metrics["fscore"],
        }
        if cfg.logging.metrics_csv:
            append_metrics_csv(metrics_csv_path, row)

        if tb_writer is not None:
            for key in train_metrics.keys():
                tb_writer.add_scalars(f"loss/{key}", {"train": train_metrics[key], "val": val_metrics[key]}, epoch)
            tb_writer.add_scalar("lr/main", lr, epoch)
            tb_writer.add_scalar("time/epoch_sec", epoch_sec, epoch)
            tb_writer.add_scalar("time/elapsed_sec", elapsed_sec, epoch)
            tb_writer.flush()

        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "config": cfg.model_dump(mode="json"),
            "center": center.tolist(),
            "scale": float(scale),
            "model_kwargs": {
                "latent_dim": cfg.model.latent_dim,
                "num_points": cfg.dataset.num_points,
                "decoder_hidden_dim": cfg.model.decoder_hidden_dim,
                "dropout": cfg.model.dropout,
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
            },
            "run_dir": str(run_dir),
        }

        save_checkpoint(run_dir / "last.pt", payload)
        save_checkpoint(cfg.output_dir / "last.pt", payload)

        if epoch % cfg.checkpoint.save_every == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:03d}.pt", payload)

        if val_metrics["chamfer"] < best_val:
            best_val = val_metrics["chamfer"]
            save_checkpoint(run_dir / "best.pt", payload)
            save_checkpoint(cfg.output_dir / "best.pt", payload)

        if (
            cfg.visualization.enabled
            and epoch % cfg.visualization.every_n_epochs == 0
            and has_matplotlib()
        ):
            save_autoencoder_snapshot(
                model=model,
                val_loader=val_loader,
                device=device,
                run_dir=run_dir,
                epoch=epoch,
                center=center,
                scale=scale,
                num_samples=cfg.visualization.num_samples,
                max_points=cfg.visualization.max_points,
                save_ply=cfg.visualization.save_ply,
            )

    if tb_writer is not None:
        tb_writer.close()

    print(f"[done] run_dir={run_dir}")


def save_autoencoder_snapshot(
    model,
    val_loader,
    device: str,
    run_dir,
    epoch: int,
    center,
    scale: float,
    num_samples: int,
    max_points: int,
    save_ply: bool,
) -> None:
    model.eval()
    batch = next(iter(val_loader))
    points = batch["points"][:num_samples].to(device, non_blocking=True).float()
    sample_ids = batch["sample_id"][:num_samples]

    with torch.no_grad():
        _, pred_points, _ = model(points)

    vis_dir = run_dir / "visuals" / f"epoch_{epoch:03d}"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for i in range(points.shape[0]):
        sid = str(sample_ids[i])
        gt_i = points[i].detach().cpu()
        pred_i = pred_points[i].detach().cpu()

        save_autoencoder_visual(
            out_path=vis_dir / f"{sid}_ae.png",
            gt_points=gt_i,
            pred_points=pred_i,
            sample_id=sid,
            max_points=max_points,
        )

        if save_ply:
            gt_denorm = denormalize_points(gt_i.numpy(), center=center, scale=scale)
            pred_denorm = denormalize_points(pred_i.numpy(), center=center, scale=scale)
            write_pointcloud_ply(vis_dir / f"{sid}_gt.ply", gt_denorm)
            write_pointcloud_ply(vis_dir / f"{sid}_pred.ply", pred_denorm)


if __name__ == "__main__":
    main()
