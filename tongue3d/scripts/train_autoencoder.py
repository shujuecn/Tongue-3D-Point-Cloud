from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tongue3d.config import (
    ensure_output_dir,
    load_autoencoder_config,
    resolve_config_path,
    resolve_device,
    save_config_json,
)
from tongue3d.data import TonguePointCloudDataset
from tongue3d.losses import (
    chamfer_distance,
    edge_length_regularizer,
    laplacian_smoothness_loss,
    normal_alignment_loss,
)
from tongue3d.models import TonguePointAutoEncoder
from tongue3d.scripts.common import (
    build_splits,
    compute_train_normalization,
    make_loader,
    maybe_autocast,
    save_normalization_json,
)
from tongue3d.utils import save_checkpoint, seed_everything


def compute_loss(
    points: torch.Tensor,
    normals: torch.Tensor,
    recon_points: torch.Tensor,
    recon_normals: torch.Tensor,
    cfg,
) -> dict[str, torch.Tensor]:
    chamfer, idx_pred_to_gt, idx_gt_to_pred = chamfer_distance(
        recon_points,
        points,
        return_indices=True,
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

    total = (
        cfg.loss.chamfer * chamfer
        + cfg.loss.normal * normal
        + cfg.loss.laplacian * lap
        + cfg.loss.edge * edge
    )
    return {
        "total": total,
        "chamfer": chamfer,
        "normal": normal,
        "laplacian": lap,
        "edge": edge,
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
    }

    pbar = tqdm(loader, desc=phase, leave=False)

    for batch in pbar:
        points = batch["points"].to(device, non_blocking=True).float()
        normals = batch["normals"].to(device, non_blocking=True).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with maybe_autocast(device=device, enabled=cfg.runtime.amp):
            _, recon_points, recon_normals = model(points)
            losses = compute_loss(points, normals, recon_points, recon_normals, cfg)

        if is_train:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        bs = points.shape[0]
        for key in meter:
            meter[key] += float(losses[key].detach().item()) * bs

        running = meter["total"] / max(1, (pbar.n + 1) * loader.batch_size)
        pbar.set_postfix(total=f"{running:.5f}")

    denom = len(loader.dataset)
    return {k: v / max(1, denom) for k, v in meter.items()}


def main() -> None:
    config_path = resolve_config_path("configs/autoencoder.yaml")
    cfg = load_autoencoder_config(config_path)

    ensure_output_dir(cfg.output_dir)
    save_config_json(cfg.output_dir / "config.json", cfg)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.runtime.device)

    splits = build_splits(cfg.dataset, cfg.split)
    if len(splits["train"]) == 0 or len(splits["val"]) == 0:
        raise ValueError("Dataset split produced empty train or val set")

    center, scale = compute_train_normalization(splits["train"])
    save_normalization_json(cfg.output_dir / "normalization.json", center, scale)

    train_ds = TonguePointCloudDataset(
        samples=splits["train"],
        dataset_cfg=cfg.dataset,
        center=center,
        scale=scale,
        preload_meshes=cfg.dataset.preload_meshes,
    )
    val_ds = TonguePointCloudDataset(
        samples=splits["val"],
        dataset_cfg=cfg.dataset,
        center=center,
        scale=scale,
        preload_meshes=cfg.dataset.preload_meshes,
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
    scaler = GradScaler(enabled=(cfg.runtime.amp and device.startswith("cuda")))

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, cfg, is_train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, cfg, is_train=False)

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} "
            f"lr={lr:.3e} "
            f"train_total={train_metrics['total']:.6f} "
            f"val_total={val_metrics['total']:.6f} "
            f"val_cd={val_metrics['chamfer']:.6f}"
        )

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
        }

        save_checkpoint(cfg.output_dir / "last.pt", payload)

        if epoch % cfg.checkpoint.save_every == 0:
            save_checkpoint(cfg.output_dir / f"epoch_{epoch:03d}.pt", payload)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(cfg.output_dir / "best.pt", payload)


if __name__ == "__main__":
    main()
