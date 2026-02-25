from __future__ import annotations

import math
import numpy as np

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from tongue3d.config import (
    ensure_output_dir,
    load_image2shape_config,
    resolve_config_path,
    resolve_device,
    save_config_json,
)
from tongue3d.data import TongueImagePointDataset
from tongue3d.losses import (
    chamfer_distance,
    edge_length_regularizer,
    laplacian_smoothness_loss,
    normal_alignment_loss,
)
from tongue3d.models import TongueImageToShape, TonguePointAutoEncoder
from tongue3d.scripts.common import (
    build_splits,
    make_loader,
    maybe_autocast,
    save_normalization_json,
)
from tongue3d.utils import load_checkpoint, seed_everything, save_checkpoint


def compute_loss(
    pred_latent: torch.Tensor,
    target_latent: torch.Tensor,
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    cfg,
) -> dict[str, torch.Tensor]:
    chamfer, idx_pred_to_gt, idx_gt_to_pred = chamfer_distance(
        pred_points,
        gt_points,
        return_indices=True,
    )
    normal = normal_alignment_loss(
        pred_normals=pred_normals,
        gt_normals=gt_normals,
        idx_pred_to_gt=idx_pred_to_gt,
        idx_gt_to_pred=idx_gt_to_pred,
    )
    lap = laplacian_smoothness_loss(pred_points)
    edge = edge_length_regularizer(pred_points)
    latent = torch.mean((pred_latent - target_latent) ** 2)

    total = (
        cfg.loss.chamfer * chamfer
        + cfg.loss.normal * normal
        + cfg.loss.laplacian * lap
        + cfg.loss.edge * edge
        + cfg.loss.latent * latent
    )
    return {
        "total": total,
        "chamfer": chamfer,
        "normal": normal,
        "laplacian": lap,
        "edge": edge,
        "latent": latent,
    }


def run_epoch(model, ae, loader, optimizer, scaler, device: str, cfg, is_train: bool):
    phase = "train" if is_train else "val"
    model.train(is_train)
    if cfg.freeze_decoder:
        model.decoder.eval()

    meter = {
        "total": 0.0,
        "chamfer": 0.0,
        "normal": 0.0,
        "laplacian": 0.0,
        "edge": 0.0,
        "latent": 0.0,
    }

    pbar = tqdm(loader, desc=phase, leave=False)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True).float()
        points = batch["points"].to(device, non_blocking=True).float()
        normals = batch["normals"].to(device, non_blocking=True).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            target_latent = ae.encode(points)

        with maybe_autocast(device=device, enabled=cfg.runtime.amp):
            pred_latent, pred_points, pred_normals = model(images)
            losses = compute_loss(
                pred_latent=pred_latent,
                target_latent=target_latent,
                pred_points=pred_points,
                gt_points=points,
                pred_normals=pred_normals,
                gt_normals=normals,
                cfg=cfg,
            )

        if is_train:
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg.grad_clip_norm,
            )
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
    config_path = resolve_config_path("configs/image2shape.yaml")
    cfg = load_image2shape_config(config_path)

    ensure_output_dir(cfg.output_dir)
    save_config_json(cfg.output_dir / "config.json", cfg)

    seed_everything(cfg.seed)
    device = resolve_device(cfg.runtime.device)

    ae_ckpt = load_checkpoint(cfg.autoencoder_checkpoint, map_location="cpu")
    center = np.asarray(ae_ckpt.get("center"), dtype=np.float32)
    scale = float(ae_ckpt.get("scale", 1.0))

    if center.size != 3:
        raise ValueError("Autoencoder checkpoint does not include valid normalization center")

    save_normalization_json(cfg.output_dir / "normalization.json", center, scale)

    splits = build_splits(cfg.dataset, cfg.split)
    if len(splits["train"]) == 0 or len(splits["val"]) == 0:
        raise ValueError("Dataset split produced empty train or val set")

    train_ds = TongueImagePointDataset(
        samples=splits["train"],
        dataset_cfg=cfg.dataset,
        center=center,
        scale=scale,
        augment=cfg.dataset.augment,
        preload_meshes=cfg.dataset.preload_meshes,
    )
    val_ds = TongueImagePointDataset(
        samples=splits["val"],
        dataset_cfg=cfg.dataset,
        center=center,
        scale=scale,
        augment=False,
        preload_meshes=cfg.dataset.preload_meshes,
    )

    train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True, runtime_cfg=cfg.runtime)
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False, runtime_cfg=cfg.runtime)

    model_kwargs = ae_ckpt.get("model_kwargs", {})
    ae_model = TonguePointAutoEncoder(
        latent_dim=model_kwargs.get("latent_dim", cfg.model.latent_dim),
        num_points=model_kwargs.get("num_points", cfg.dataset.num_points),
        decoder_hidden_dim=model_kwargs.get("decoder_hidden_dim", cfg.model.decoder_hidden_dim),
        dropout=model_kwargs.get("dropout", cfg.model.dropout),
    )
    ae_model.load_state_dict(ae_ckpt["model_state"])
    ae_model = ae_model.to(device)
    ae_model.eval()
    for param in ae_model.parameters():
        param.requires_grad = False

    model = TongueImageToShape(
        latent_dim=cfg.model.latent_dim,
        num_points=cfg.dataset.num_points,
        decoder_hidden_dim=cfg.model.decoder_hidden_dim,
        dropout=cfg.model.dropout,
        pretrained_backbone=True,
        decoder=ae_model.decoder,
    ).to(device)

    if cfg.freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        optim_params,
        lr=cfg.optimizer.lr,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = GradScaler(enabled=(cfg.runtime.amp and device.startswith("cuda")))

    best_val = math.inf

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, ae_model, train_loader, optimizer, scaler, device, cfg, is_train=True)
        with torch.no_grad():
            val_metrics = run_epoch(model, ae_model, val_loader, optimizer, scaler, device, cfg, is_train=False)

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
            "autoencoder_checkpoint": str(cfg.autoencoder_checkpoint),
        }

        save_checkpoint(cfg.output_dir / "last.pt", payload)

        if epoch % cfg.checkpoint.save_every == 0:
            save_checkpoint(cfg.output_dir / f"epoch_{epoch:03d}.pt", payload)

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(cfg.output_dir / "best.pt", payload)


if __name__ == "__main__":
    main()
