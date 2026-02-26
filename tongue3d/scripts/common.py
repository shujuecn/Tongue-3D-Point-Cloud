from __future__ import annotations

import csv
import json
from datetime import datetime
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
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": runtime_cfg.num_workers,
        "pin_memory": runtime_cfg.pin_memory,
        "drop_last": shuffle,
    }
    if runtime_cfg.num_workers > 0:
        kwargs["persistent_workers"] = runtime_cfg.persistent_workers
        kwargs["prefetch_factor"] = runtime_cfg.prefetch_factor
    return DataLoader(**kwargs)


def maybe_autocast(device: str, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)

    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", enabled=True)
    return torch.autocast(device_type="cpu", enabled=False)


def make_grad_scaler(device: str, enabled: bool):
    scaler_enabled = bool(enabled and device.startswith("cuda"))
    scaler_device = "cuda" if device.startswith("cuda") else "cpu"

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device=scaler_device, enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def create_run_dir(output_root: Path, experiment_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{ts}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_file = output_root / "latest_run.txt"
    latest_file.write_text(str(run_dir.resolve()), encoding="utf-8")
    return run_dir


def maybe_create_summary_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_dir))
    except Exception:
        return None


def append_metrics_csv(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_seconds(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
