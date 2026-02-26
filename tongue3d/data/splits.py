from __future__ import annotations

import csv
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


def save_splits_csv(splits: dict[str, list[TongueSample]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "sample_id", "image_path", "mesh_path"])
        for split_name in ("train", "val", "test"):
            for sample in splits.get(split_name, []):
                writer.writerow(
                    [
                        split_name,
                        sample.sample_id,
                        str(sample.image_path.resolve()),
                        str(sample.mesh_path.resolve()),
                    ]
                )


def load_splits_csv(csv_path: Path) -> dict[str, list[TongueSample]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    splits: dict[str, list[TongueSample]] = {"train": [], "val": [], "test": []}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"split", "sample_id", "image_path", "mesh_path"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Invalid split CSV header in {csv_path}")

        for row in reader:
            split_name = str(row["split"]).strip()
            if split_name not in splits:
                continue
            sample = TongueSample(
                sample_id=str(row["sample_id"]).strip(),
                image_path=Path(str(row["image_path"]).strip()),
                mesh_path=Path(str(row["mesh_path"]).strip()),
            )
            splits[split_name].append(sample)
    return splits
