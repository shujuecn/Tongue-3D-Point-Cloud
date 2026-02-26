from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    root_dir: Path = Path("TongueDB")
    image_subdir: str = "images"
    mesh_subdir: str = "meshes"
    num_points: int = Field(default=2048, ge=256)
    image_size: int = Field(default=224, ge=128)
    preload_meshes: bool = True
    augment: bool = True

    @property
    def image_dir(self) -> Path:
        return self.root_dir / self.image_subdir

    @property
    def mesh_dir(self) -> Path:
        return self.root_dir / self.mesh_subdir


class SplitConfig(BaseModel):
    train_ratio: float = 0.9
    val_ratio: float = 0.1
    test_ratio: float = 0.0
    seed: int = 42

    @model_validator(mode="after")
    def validate_sum(self) -> "SplitConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + val_ratio + test_ratio must be 1.0, got {total}")
        return self


class ModelConfig(BaseModel):
    latent_dim: int = 256
    decoder_hidden_dim: int = 1024
    dropout: float = 0.1


class OptimConfig(BaseModel):
    lr: float = 1e-4
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999


class RuntimeConfig(BaseModel):
    device: str = "cuda"
    amp: bool = True
    num_workers: int = 8
    pin_memory: bool = True


class CheckpointConfig(BaseModel):
    save_every: int = 5


class AutoencoderLossConfig(BaseModel):
    chamfer: float = 1.0
    chamfer_chunk_size: int = Field(default=0, ge=0)
    normal: float = 0.1
    laplacian: float = 0.05
    edge: float = 0.02


class Image2ShapeLossConfig(BaseModel):
    chamfer: float = 1.2
    chamfer_chunk_size: int = Field(default=0, ge=0)
    normal: float = 0.05
    laplacian: float = 0.05
    edge: float = 0.02
    latent: float = 1.5


class AutoencoderTrainConfig(BaseModel):
    experiment_name: str = "ae_baseline"
    output_dir: Path = Path("runs/ae_baseline")
    seed: int = 42
    epochs: int = 160
    batch_size: int = 16
    grad_clip_norm: float = 1.0
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimConfig = Field(default_factory=OptimConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    loss: AutoencoderLossConfig = Field(default_factory=AutoencoderLossConfig)


class Image2ShapeTrainConfig(BaseModel):
    experiment_name: str = "img2shape_baseline"
    output_dir: Path = Path("runs/img2shape_baseline")
    seed: int = 42
    epochs: int = 120
    batch_size: int = 24
    grad_clip_norm: float = 1.0
    autoencoder_checkpoint: Path = Path("runs/ae_baseline/best.pt")
    freeze_decoder: bool = True
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    optimizer: OptimConfig = Field(default_factory=lambda: OptimConfig(lr=2e-4))
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    loss: Image2ShapeLossConfig = Field(default_factory=Image2ShapeLossConfig)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_autoencoder_config(path: Path) -> AutoencoderTrainConfig:
    raw = load_yaml(path)
    return AutoencoderTrainConfig.model_validate(raw)


def load_image2shape_config(path: Path) -> Image2ShapeTrainConfig:
    raw = load_yaml(path)
    return Image2ShapeTrainConfig.model_validate(raw)


def save_config_json(path: Path, cfg: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.model_dump(mode="json"), f, indent=2)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_config_path(default_path: str) -> Path:
    import os
    import sys

    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    env_path = os.environ.get("TONGUE3D_CONFIG")
    if env_path:
        return Path(env_path)
    return Path(default_path)


def resolve_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


ConfigKind = Literal["autoencoder", "image2shape"]
