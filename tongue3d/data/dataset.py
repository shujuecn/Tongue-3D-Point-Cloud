from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import warnings

try:
    from torchvision import transforms
    _HAS_TORCHVISION = True
except Exception:
    transforms = None
    _HAS_TORCHVISION = False

from tongue3d.config import DatasetConfig
from tongue3d.data.splits import TongueSample
from tongue3d.utils.mesh import load_obj, normalize_points, sample_points_from_mesh


class _MeshCache:
    storage: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def build_image_transform(image_size: int, augment: bool) -> transforms.Compose:
    if not _HAS_TORCHVISION:
        warnings.warn(
            "torchvision is not available. Falling back to basic resize+tensor transform without augmentation."
        )
        return _fallback_transform(image_size=image_size)

    ops: list[Any] = [transforms.Resize((image_size, image_size))]
    if augment:
        ops.extend(
            [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def _fallback_transform(image_size: int):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def _apply(image: Image.Image) -> torch.Tensor:
        image = image.resize((image_size, image_size), resample=Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = (tensor - mean) / std
        return tensor

    return _apply


class TonguePointCloudDataset(Dataset):
    def __init__(
        self,
        samples: list[TongueSample],
        dataset_cfg: DatasetConfig,
        center: np.ndarray,
        scale: float,
        preload_meshes: bool | None = None,
        deterministic_sampling: bool = False,
    ) -> None:
        self.samples = samples
        self.dataset_cfg = dataset_cfg
        self.center = center.astype(np.float32)
        self.scale = float(scale)
        self.preload_meshes = dataset_cfg.preload_meshes if preload_meshes is None else preload_meshes
        self.deterministic_sampling = deterministic_sampling

        if self.preload_meshes:
            iterator = self.samples
            try:
                from tqdm import tqdm

                iterator = tqdm(self.samples, desc="preload-meshes", leave=False)
            except Exception:
                pass

            for sample in iterator:
                key = str(sample.mesh_path)
                if key not in _MeshCache.storage:
                    _MeshCache.storage[key] = load_obj(sample.mesh_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mesh(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path)
        if self.preload_meshes:
            item = _MeshCache.storage.get(key)
            if item is None:
                item = load_obj(path)
                _MeshCache.storage[key] = item
            return item
        return load_obj(path)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        vertices, faces = self._load_mesh(sample.mesh_path)

        rng = None
        if self.deterministic_sampling:
            seed = stable_seed_from_string(sample.sample_id)
            rng = np.random.default_rng(seed)

        points, normals = sample_points_from_mesh(
            vertices,
            faces,
            self.dataset_cfg.num_points,
            rng=rng,
        )
        points = normalize_points(points, self.center, self.scale)

        return {
            "sample_id": sample.sample_id,
            "points": torch.from_numpy(points),
            "normals": torch.from_numpy(normals),
        }


class TongueImagePointDataset(TonguePointCloudDataset):
    def __init__(
        self,
        samples: list[TongueSample],
        dataset_cfg: DatasetConfig,
        center: np.ndarray,
        scale: float,
        augment: bool,
        preload_meshes: bool | None = None,
        deterministic_sampling: bool = False,
    ) -> None:
        super().__init__(
            samples,
            dataset_cfg,
            center,
            scale,
            preload_meshes=preload_meshes,
            deterministic_sampling=deterministic_sampling,
        )
        self.transform = build_image_transform(dataset_cfg.image_size, augment)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        out = super().__getitem__(idx)
        sample = self.samples[idx]

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)

        out["image"] = image_tensor
        return out


def stable_seed_from_string(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
