from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tongue3d.config import DatasetConfig, RuntimeConfig, SplitConfig, resolve_device
from tongue3d.data import TongueImagePointDataset, collect_samples, load_splits_csv, split_samples
from tongue3d.losses import chamfer_with_neighbors, normal_alignment_loss
from tongue3d.models import TongueImageToShape
from tongue3d.scripts.common import make_loader
from tongue3d.utils import load_checkpoint


def parse_cli() -> tuple[Path, str, Path | None, Path | None]:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.evaluate <img2shape_checkpoint.pt> [split] [split_csv] [output_json]"
        )
    ckpt = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    split_csv = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    output_json = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    return ckpt, split, split_csv, output_json


def infer_split_csv_path(ckpt_path: Path, checkpoint: dict) -> Path | None:
    run_dir = checkpoint.get("run_dir")
    if run_dir:
        candidate = Path(str(run_dir)) / "splits.csv"
        if candidate.exists():
            return candidate

    candidate = ckpt_path.parent / "splits.csv"
    if candidate.exists():
        return candidate
    return None


def load_target_samples(
    split_name: str,
    dataset_cfg: DatasetConfig,
    split_cfg: SplitConfig,
    split_csv: Path | None,
    ckpt_path: Path,
    checkpoint: dict,
):
    effective_split_csv = split_csv if split_csv is not None else infer_split_csv_path(ckpt_path, checkpoint)
    if effective_split_csv is not None:
        splits = load_splits_csv(effective_split_csv)
    else:
        samples = collect_samples(dataset_cfg)
        splits = split_samples(samples, split_cfg)

    if split_name not in splits:
        raise ValueError(f"Unknown split '{split_name}'. Available: {list(splits.keys())}")

    target = splits[split_name]
    if len(target) == 0:
        raise ValueError(f"Split '{split_name}' is empty")

    return target, effective_split_csv


def main() -> None:
    ckpt_path, split_name, split_csv_arg, output_json = parse_cli()
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")

    cfg = checkpoint["config"]
    dataset_cfg = DatasetConfig.model_validate(cfg["dataset"])
    split_cfg = SplitConfig.model_validate(cfg["split"])
    runtime_cfg = RuntimeConfig.model_validate(cfg["runtime"])

    center = np.asarray(checkpoint["center"], dtype=np.float32)
    scale = float(checkpoint["scale"])

    target, used_split_csv = load_target_samples(
        split_name=split_name,
        dataset_cfg=dataset_cfg,
        split_cfg=split_cfg,
        split_csv=split_csv_arg,
        ckpt_path=ckpt_path,
        checkpoint=checkpoint,
    )

    ds = TongueImagePointDataset(
        samples=target,
        dataset_cfg=dataset_cfg,
        center=center,
        scale=scale,
        augment=False,
        preload_meshes=dataset_cfg.preload_meshes,
        deterministic_sampling=True,
    )
    loader = make_loader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        runtime_cfg=runtime_cfg,
    )

    model_kwargs = checkpoint["model_kwargs"]
    model = TongueImageToShape(
        latent_dim=int(model_kwargs["latent_dim"]),
        num_points=int(model_kwargs["num_points"]),
        decoder_hidden_dim=int(model_kwargs["decoder_hidden_dim"]),
        dropout=float(model_kwargs["dropout"]),
        pretrained_backbone=False,
        input_channels=int(model_kwargs.get("input_channels", 3)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = resolve_device("cuda")
    model = model.to(device)
    model.eval()

    threshold = float(cfg.get("loss", {}).get("fscore_threshold", 0.01))
    chunk_size = int(cfg.get("loss", {}).get("chamfer_chunk_size", 0))

    meter = {
        "chamfer_l2": 0.0,
        "chamfer_l1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "fscore": 0.0,
        "normal_loss": 0.0,
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{split_name}"):
            images = batch["image"].to(device, non_blocking=True).float()
            points = batch["points"].to(device, non_blocking=True).float()
            normals = batch["normals"].to(device, non_blocking=True).float()

            _, pred_points, pred_normals = model(images)
            cd_l2, min_pred, min_gt, idx_pred_to_gt, idx_gt_to_pred = chamfer_with_neighbors(
                pred_points,
                points,
                chunk_size=chunk_size,
            )
            cd_l1 = min_pred.mean() + min_gt.mean()
            precision = (min_pred <= threshold).float().mean()
            recall = (min_gt <= threshold).float().mean()
            fscore = (2.0 * precision * recall) / (precision + recall + 1e-8)
            normal_loss = normal_alignment_loss(
                pred_normals=pred_normals,
                gt_normals=normals,
                idx_pred_to_gt=idx_pred_to_gt,
                idx_gt_to_pred=idx_gt_to_pred,
            )

            bs = points.shape[0]
            meter["chamfer_l2"] += float(cd_l2.item()) * bs
            meter["chamfer_l1"] += float(cd_l1.item()) * bs
            meter["precision"] += float(precision.item()) * bs
            meter["recall"] += float(recall.item()) * bs
            meter["fscore"] += float(fscore.item()) * bs
            meter["normal_loss"] += float(normal_loss.item()) * bs

    total = len(ds)
    results = {k: v / max(1, total) for k, v in meter.items()}
    results.update(
        {
            "split": split_name,
            "samples": total,
            "threshold": threshold,
            "checkpoint": str(ckpt_path.resolve()),
            "split_csv": str(used_split_csv.resolve()) if used_split_csv is not None else "",
        }
    )

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved metrics JSON: {output_json}")

    print(
        f"Split={split_name} samples={total} "
        f"CD_L2={results['chamfer_l2']:.8f} "
        f"CD_L1={results['chamfer_l1']:.8f} "
        f"F1@{threshold:.4f}={results['fscore']:.6f} "
        f"P={results['precision']:.6f} R={results['recall']:.6f} "
        f"NLoss={results['normal_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
