from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tongue3d.config import resolve_device
from tongue3d.models import TongueImageToShape
from tongue3d.utils import denormalize_points, load_checkpoint, write_pointcloud_ply
from tongue3d.data.dataset import build_image_transform


def parse_cli() -> tuple[Path, str, Path | None, Path | None]:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.infer_single <img2shape_checkpoint.pt> <image_path_or_sample_id> [output_ply] [split_csv]"
        )

    ckpt_path = Path(sys.argv[1])
    query = sys.argv[2]
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    split_csv = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    return ckpt_path, query, output_path, split_csv


def infer_split_csv(ckpt_path: Path, checkpoint: dict, provided: Path | None) -> Path | None:
    if provided is not None:
        return provided

    run_dir = checkpoint.get("run_dir")
    if run_dir:
        candidate = Path(str(run_dir)) / "splits.csv"
        if candidate.exists():
            return candidate

    candidate = ckpt_path.parent / "splits.csv"
    if candidate.exists():
        return candidate
    return None


def resolve_image_path(query: str, split_csv: Path | None) -> tuple[Path, str]:
    query_path = Path(query)
    if query_path.exists():
        return query_path, query_path.stem

    if split_csv is None or not split_csv.exists():
        raise FileNotFoundError(
            f"'{query}' is not a valid image path, and split CSV is unavailable. "
            f"Provide split CSV as the 4th arg to resolve sample_id."
        )

    preferred_order = {"val": 0, "test": 1, "train": 2}
    best_row = None
    best_rank = 999

    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if sample_id != query:
                continue
            rank = preferred_order.get(str(row.get("split", "")).strip(), 10)
            if rank < best_rank:
                best_rank = rank
                best_row = row

    if best_row is None:
        raise ValueError(f"sample_id '{query}' not found in {split_csv}")

    image_path = Path(str(best_row["image_path"]).strip())
    if not image_path.exists():
        raise FileNotFoundError(f"Image path from split CSV does not exist: {image_path}")

    return image_path, str(best_row["sample_id"]).strip()


def main() -> None:
    ckpt_path, query, output_path, split_csv_arg = parse_cli()

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model_kwargs = checkpoint.get("model_kwargs", {})
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))

    center = np.asarray(checkpoint["center"], dtype=np.float32)
    scale = float(checkpoint["scale"])

    split_csv = infer_split_csv(ckpt_path=ckpt_path, checkpoint=checkpoint, provided=split_csv_arg)
    image_path, sample_id = resolve_image_path(query=query, split_csv=split_csv)

    if output_path is None:
        output_path = Path("runs/predictions") / f"{sample_id}_pred.ply"

    latent_dim = int(model_kwargs.get("latent_dim", cfg.get("model", {}).get("latent_dim", 256)))
    num_points = int(model_kwargs.get("num_points", cfg.get("dataset", {}).get("num_points", 2048)))
    hidden_dim = int(model_kwargs.get("decoder_hidden_dim", cfg.get("model", {}).get("decoder_hidden_dim", 1024)))
    dropout = float(model_kwargs.get("dropout", cfg.get("model", {}).get("dropout", 0.1)))

    model = TongueImageToShape(
        latent_dim=latent_dim,
        num_points=num_points,
        decoder_hidden_dim=hidden_dim,
        dropout=dropout,
        pretrained_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)

    device = resolve_device("cuda")
    model = model.to(device)
    model.eval()

    transform = build_image_transform(image_size=image_size, augment=False)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, pred_points, pred_normals = model(image_tensor)

    points = pred_points[0].detach().cpu().numpy()
    normals = pred_normals[0].detach().cpu().numpy()
    points = denormalize_points(points, center=center, scale=scale)

    write_pointcloud_ply(output_path, points, normals)
    print(f"Saved reconstruction to: {output_path}")
    print(f"Input image: {image_path}")
    if split_csv is not None:
        print(f"Split CSV: {split_csv}")


if __name__ == "__main__":
    main()
