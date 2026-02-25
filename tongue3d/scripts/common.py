from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tongue3d.config import DatasetConfig, RuntimeConfig, SplitConfig
from tongue3d.data import collect_samples, split_samples
from tongue3d.utils.mesh import compute_normalization_stats


def build_splits(dataset_cfg: DatasetConfig, split_cfg: SplitConfig) -> dict[str, list[Any]]:
    samples = collect_samples(dataset_cfg)
    splits = split_samples(samples, split_cfg)
    return splits


def compute_train_normalization(train_samples: list[Any]) -> tuple[np.ndarray, float]:
    mesh_paths = [sample.mesh_path for sample in train_samples]
    center, scale = compute_normalization_stats(mesh_paths)
    return center, float(scale)


def save_normalization_json(path: Path, center: np.ndarray, scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"center": center.tolist(), "scale": float(scale)}, f, indent=2)


def load_normalization_json(path: Path) -> tuple[np.ndarray, float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    center = np.asarray(raw["center"], dtype=np.float32)
    scale = float(raw["scale"])
    return center, scale


def make_loader(dataset, batch_size: int, shuffle: bool, runtime_cfg: RuntimeConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=runtime_cfg.num_workers,
        pin_memory=runtime_cfg.pin_memory,
        drop_last=shuffle,
    )


def maybe_autocast(device: str, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)

    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)
