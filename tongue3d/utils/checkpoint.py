from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_normalization(path: Path, center: list[float], scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"center": center, "scale": scale}, f, indent=2)


def load_normalization(path: Path) -> tuple[list[float], float]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    center = [float(v) for v in raw["center"]]
    scale = float(raw["scale"])
    return center, scale
