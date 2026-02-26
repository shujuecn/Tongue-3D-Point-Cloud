from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tongue3d.config import DatasetConfig, RuntimeConfig, SplitConfig, resolve_device
from tongue3d.data import TongueImagePointDataset, collect_samples, split_samples
from tongue3d.losses import chamfer_distance
from tongue3d.models import TongueImageToShape
from tongue3d.scripts.common import make_loader
from tongue3d.utils import load_checkpoint


def parse_cli() -> tuple[Path, str]:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m tongue3d.scripts.evaluate <img2shape_checkpoint.pt> [split]")
    ckpt = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    return ckpt, split


def main() -> None:
    ckpt_path, split_name = parse_cli()
    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")

    cfg = checkpoint["config"]
    dataset_cfg = DatasetConfig.model_validate(cfg["dataset"])
    split_cfg = SplitConfig.model_validate(cfg["split"])
    runtime_cfg = RuntimeConfig.model_validate(cfg["runtime"])

    center = np.asarray(checkpoint["center"], dtype=np.float32)
    scale = float(checkpoint["scale"])

    samples = collect_samples(dataset_cfg)
    splits = split_samples(samples, split_cfg)
    if split_name not in splits:
        raise ValueError(f"Unknown split '{split_name}'. Available: {list(splits.keys())}")
    target = splits[split_name]
    if len(target) == 0:
        raise ValueError(f"Split '{split_name}' is empty")

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
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = resolve_device("cuda")
    model = model.to(device)
    model.eval()

    total_cd = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval-{split_name}"):
            images = batch["image"].to(device, non_blocking=True).float()
            points = batch["points"].to(device, non_blocking=True).float()
            _, pred_points, _ = model(images)
            cd = chamfer_distance(
                pred_points,
                points,
                chunk_size=int(cfg["loss"].get("chamfer_chunk_size", 0)),
            )
            total_cd += float(cd.item()) * points.shape[0]

    mean_cd = total_cd / len(ds)
    print(f"Split={split_name} samples={len(ds)} Chamfer={mean_cd:.8f}")


if __name__ == "__main__":
    main()
