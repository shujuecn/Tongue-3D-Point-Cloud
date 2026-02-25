from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tongue3d.config import resolve_device
from tongue3d.models import TongueImageToShape
from tongue3d.utils import denormalize_points, load_checkpoint, write_pointcloud_ply
from tongue3d.data.dataset import build_image_transform


def parse_cli() -> tuple[Path, Path, Path]:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.infer_single <img2shape_checkpoint.pt> <image_path> [output_ply]"
        )

    ckpt_path = Path(sys.argv[1])
    image_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("runs/predictions") / f"{image_path.stem}.ply"
    return ckpt_path, image_path, output_path


def main() -> None:
    ckpt_path, image_path, output_path = parse_cli()

    checkpoint = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model_kwargs = checkpoint.get("model_kwargs", {})
    image_size = int(cfg.get("dataset", {}).get("image_size", 224))

    center = np.asarray(checkpoint["center"], dtype=np.float32)
    scale = float(checkpoint["scale"])

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


if __name__ == "__main__":
    main()
