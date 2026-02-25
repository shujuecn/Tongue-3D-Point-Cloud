from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from tongue3d.config import DatasetConfig, SplitConfig


@dataclass(frozen=True)
class TongueSample:
    sample_id: str
    image_path: Path
    mesh_path: Path


def collect_samples(dataset_cfg: DatasetConfig) -> list[TongueSample]:
    image_dir = dataset_cfg.image_dir
    mesh_dir = dataset_cfg.mesh_dir

    image_map = {p.stem: p for p in sorted(image_dir.glob("*.png"))}
    mesh_map = {p.stem: p for p in sorted(mesh_dir.glob("*.obj"))}

    common = sorted(set(image_map).intersection(mesh_map))
    if not common:
        raise ValueError(
            f"No paired samples found between {image_dir} and {mesh_dir}."
        )

    samples = [
        TongueSample(sample_id=sid, image_path=image_map[sid], mesh_path=mesh_map[sid])
        for sid in common
    ]
    return samples


def split_samples(samples: list[TongueSample], split_cfg: SplitConfig) -> dict[str, list[TongueSample]]:
    ordered = list(samples)
    rng = random.Random(split_cfg.seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(n * split_cfg.train_ratio)
    n_val = int(n * split_cfg.val_ratio)

    train = ordered[:n_train]
    val = ordered[n_train : n_train + n_val]
    test = ordered[n_train + n_val :]

    return {"train": train, "val": val, "test": test}
