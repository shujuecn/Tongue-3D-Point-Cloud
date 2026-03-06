from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
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


@dataclass(frozen=True)
class InTheWildPairSample:
    sample_id: str
    color_path: Path
    segmented_path: Path


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
        self.mask_map = self._build_mask_map() if dataset_cfg.use_mask else {}

    def _build_mask_map(self) -> dict[str, Path]:
        mask_dir = self.dataset_cfg.mask_dir
        if not mask_dir.exists():
            warnings.warn(
                f"dataset.use_mask=True but mask directory does not exist: {mask_dir}. "
                "Fallback to raw images."
            )
            return {}

        mask_map = {p.stem: p for p in sorted(mask_dir.glob("*.png"))}
        if len(mask_map) == 0:
            warnings.warn(f"No mask files found in: {mask_dir}. Fallback to raw images.")
        return mask_map

    def _apply_mask_preprocess(self, image: Image.Image, sample_id: str) -> Image.Image:
        if not self.dataset_cfg.use_mask:
            return image

        mask_path = self.mask_map.get(sample_id)
        if mask_path is None or not mask_path.exists():
            return image

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, resample=Image.NEAREST)

            mask_np = np.asarray(mask, dtype=np.uint8) >= int(self.dataset_cfg.mask_threshold)
            if not np.any(mask_np):
                return image

            if self.dataset_cfg.mask_background_zero:
                image_np = np.asarray(image, dtype=np.uint8).copy()
                image_np[~mask_np] = 0
                image = Image.fromarray(image_np, mode="RGB")

            if self.dataset_cfg.mask_crop:
                ys, xs = np.where(mask_np)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                margin = int(max(image.width, image.height) * float(self.dataset_cfg.mask_margin_ratio))
                x0 = max(0, x0 - margin)
                y0 = max(0, y0 - margin)
                x1 = min(image.width - 1, x1 + margin)
                y1 = min(image.height - 1, y1 + margin)
                if x1 > x0 and y1 > y0:
                    image = image.crop((x0, y0, x1 + 1, y1 + 1))
        return image

    def __getitem__(self, idx: int) -> dict[str, Any]:
        out = super().__getitem__(idx)
        sample = self.samples[idx]

        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            image = self._apply_mask_preprocess(image=image, sample_id=sample.sample_id)
            image_tensor = self.transform(image)

        out["image"] = image_tensor
        return out


def load_in_the_wild_manifest(csv_path: Path) -> list[InTheWildPairSample]:
    if not csv_path.exists():
        raise FileNotFoundError(f"In-the-wild manifest not found: {csv_path}")

    samples: list[InTheWildPairSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "color_path", "segmented_path"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid manifest header in {csv_path}. "
                f"Expected columns: {sorted(required)}"
            )

        for row in reader:
            sid = str(row["sample_id"]).strip()
            color_path = Path(str(row["color_path"]).strip())
            segmented_path = Path(str(row["segmented_path"]).strip())
            if not sid or not color_path.exists() or not segmented_path.exists():
                continue
            samples.append(
                InTheWildPairSample(
                    sample_id=sid,
                    color_path=color_path,
                    segmented_path=segmented_path,
                )
            )
    if len(samples) == 0:
        raise ValueError(f"No valid in-the-wild samples found in manifest: {csv_path}")
    return samples


class TongueInTheWildPairDataset(Dataset):
    def __init__(
        self,
        samples: list[InTheWildPairSample],
        image_size: int,
        augment: bool = False,
        use_segmented_mask_preprocess: bool = True,
    ) -> None:
        self.samples = samples
        self.transform = build_image_transform(image_size=image_size, augment=augment)
        self.use_segmented_mask_preprocess = bool(use_segmented_mask_preprocess)

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _mask_from_segmented(segmented_rgb: np.ndarray, threshold: int = 5) -> np.ndarray:
        gray = segmented_rgb.max(axis=2)
        return gray >= threshold

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        with Image.open(sample.color_path) as color_im:
            color_im = color_im.convert("RGB")
            color_np = np.asarray(color_im, dtype=np.uint8)

        with Image.open(sample.segmented_path) as seg_im:
            seg_im = seg_im.convert("RGB")
            seg_np = np.asarray(seg_im, dtype=np.uint8)

        if seg_np.shape[:2] != color_np.shape[:2]:
            seg_np = np.asarray(
                Image.fromarray(seg_np, mode="RGB").resize(
                    (color_np.shape[1], color_np.shape[0]),
                    resample=Image.BILINEAR,
                ),
                dtype=np.uint8,
            )

        if self.use_segmented_mask_preprocess:
            tongue_mask = self._mask_from_segmented(seg_np)
            color_np = color_np.copy()
            color_np[~tongue_mask] = 0

        color_tensor = self.transform(Image.fromarray(color_np, mode="RGB"))
        segmented_tensor = self.transform(Image.fromarray(seg_np, mode="RGB"))
        return {
            "sample_id": sample.sample_id,
            "color_image": color_tensor,
            "segmented_image": segmented_tensor,
        }


def stable_seed_from_string(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
