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


def _resize_to_square_with_letterbox(
    image: Image.Image,
    image_size: int,
    resample: int,
) -> Image.Image:
    if image.size == (image_size, image_size):
        return image

    src_w, src_h = image.size
    scale = min(image_size / max(1, src_w), image_size / max(1, src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = image.resize((new_w, new_h), resample=resample)
    canvas = Image.new(image.mode, (image_size, image_size), color=0)
    left = (image_size - new_w) // 2
    top = (image_size - new_h) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _resize_rgb_hwc(
    rgb_hwc_uint8: np.ndarray,
    image_size: int,
    resize_mode: str,
    resample: int,
) -> np.ndarray:
    image = Image.fromarray(rgb_hwc_uint8, mode="RGB")
    if resize_mode == "direct":
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), resample=resample)
    elif resize_mode == "letterbox":
        image = _resize_to_square_with_letterbox(image, image_size=image_size, resample=resample)
    else:
        raise ValueError(f"Unsupported resize_mode='{resize_mode}'. Expected 'direct' or 'letterbox'.")
    return np.asarray(image, dtype=np.uint8)


def mask_from_segmented_rgb(segmented_rgb: np.ndarray, threshold: int = 16) -> np.ndarray:
    gray = segmented_rgb.max(axis=2)
    return gray >= int(threshold)


def prepare_in_the_wild_pair_arrays(
    color_rgb: np.ndarray,
    segmented_rgb: np.ndarray,
    image_size: int,
    use_segmented_mask_preprocess: bool,
    resize_mode: str = "letterbox",
    segmented_mask_threshold: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    if segmented_rgb.shape[:2] != color_rgb.shape[:2]:
        segmented_rgb = np.asarray(
            Image.fromarray(segmented_rgb, mode="RGB").resize(
                (color_rgb.shape[1], color_rgb.shape[0]),
                resample=Image.NEAREST,
            ),
            dtype=np.uint8,
        )

    color_np = _resize_rgb_hwc(
        color_rgb,
        image_size=image_size,
        resize_mode=resize_mode,
        resample=Image.BILINEAR,
    )
    seg_np = _resize_rgb_hwc(
        segmented_rgb,
        image_size=image_size,
        resize_mode=resize_mode,
        resample=Image.NEAREST,
    )

    if use_segmented_mask_preprocess:
        tongue_mask = mask_from_segmented_rgb(seg_np, threshold=segmented_mask_threshold)
        color_np = color_np.copy()
        color_np[~tongue_mask] = 0

    return color_np, seg_np


def apply_mask_preprocess_with_mask(
    image: Image.Image,
    mask: Image.Image,
    dataset_cfg: DatasetConfig,
) -> Image.Image:
    mask = mask.convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, resample=Image.NEAREST)

    mask_np = np.asarray(mask, dtype=np.uint8) >= int(dataset_cfg.mask_threshold)
    if not np.any(mask_np):
        return image

    if dataset_cfg.mask_background_zero:
        image_np = np.asarray(image, dtype=np.uint8).copy()
        image_np[~mask_np] = 0
        image = Image.fromarray(image_np, mode="RGB")

    if dataset_cfg.mask_crop:
        ys, xs = np.where(mask_np)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        margin = int(max(image.width, image.height) * float(dataset_cfg.mask_margin_ratio))
        x0 = max(0, x0 - margin)
        y0 = max(0, y0 - margin)
        x1 = min(image.width - 1, x1 + margin)
        y1 = min(image.height - 1, y1 + margin)
        if x1 > x0 and y1 > y0:
            image = image.crop((x0, y0, x1 + 1, y1 + 1))
    return image


def apply_mask_preprocess_with_mask_path(
    image: Image.Image,
    mask_path: Path,
    dataset_cfg: DatasetConfig,
) -> Image.Image:
    with Image.open(mask_path) as mask:
        return apply_mask_preprocess_with_mask(image=image, mask=mask, dataset_cfg=dataset_cfg)


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

        return apply_mask_preprocess_with_mask_path(
            image=image,
            mask_path=mask_path,
            dataset_cfg=self.dataset_cfg,
        )

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
        resize_mode: str = "letterbox",
        segmented_mask_threshold: int = 16,
    ) -> None:
        self.samples = samples
        self.image_size = int(image_size)
        self.transform = build_image_transform(image_size=image_size, augment=augment)
        self.use_segmented_mask_preprocess = bool(use_segmented_mask_preprocess)
        self.resize_mode = str(resize_mode)
        self.segmented_mask_threshold = int(segmented_mask_threshold)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        with Image.open(sample.color_path) as color_im:
            color_im = color_im.convert("RGB")
            color_np = np.asarray(color_im, dtype=np.uint8)

        with Image.open(sample.segmented_path) as seg_im:
            seg_im = seg_im.convert("RGB")
            seg_np = np.asarray(seg_im, dtype=np.uint8)

        color_np, seg_np = prepare_in_the_wild_pair_arrays(
            color_rgb=color_np,
            segmented_rgb=seg_np,
            image_size=self.image_size,
            use_segmented_mask_preprocess=self.use_segmented_mask_preprocess,
            resize_mode=self.resize_mode,
            segmented_mask_threshold=self.segmented_mask_threshold,
        )

        color_tensor = self.transform(Image.fromarray(color_np, mode="RGB"))
        segmented_tensor = self.transform(Image.fromarray(seg_np, mode="RGB"))
        return {
            "sample_id": sample.sample_id,
            "color_image": color_tensor,
            "segmented_image": segmented_tensor,
        }


class TongueInTheWildCacheDataset(Dataset):
    def __init__(
        self,
        cache_path: Path,
        image_size: int,
        augment: bool = False,
        expected_use_segmented_mask_preprocess: bool | None = None,
        expected_resize_mode: str | None = None,
        expected_segmented_mask_threshold: int | None = None,
    ) -> None:
        self.cache_path = cache_path
        pack = np.load(cache_path, allow_pickle=False)
        self.sample_ids = pack["sample_ids"]
        self.color = pack["color"]
        self.segmented = pack["segmented"]
        self.cache_image_size = int(pack["image_size"][0]) if "image_size" in pack else int(image_size)
        self.cache_use_segmented_mask_preprocess = (
            bool(int(pack["use_segmented_mask_preprocess"][0]))
            if "use_segmented_mask_preprocess" in pack
            else None
        )
        self.cache_resize_mode = str(pack["resize_mode"][0]) if "resize_mode" in pack else None
        self.cache_segmented_mask_threshold = (
            int(pack["segmented_mask_threshold"][0]) if "segmented_mask_threshold" in pack else None
        )

        if self.color.ndim != 4 or self.color.shape[-1] != 3:
            raise ValueError(f"Invalid color tensor shape in cache: {self.color.shape}")
        if self.segmented.shape != self.color.shape:
            raise ValueError(
                f"Cache color/segmented shape mismatch: {self.color.shape} vs {self.segmented.shape}"
            )
        if self.color.shape[1] != image_size or self.color.shape[2] != image_size:
            raise ValueError(
                f"Cache image_size mismatch: cache={self.color.shape[1]} requested={image_size}. "
                "Please rebuild cache with the target image_size."
            )

        if expected_use_segmented_mask_preprocess is not None:
            if self.cache_use_segmented_mask_preprocess is None:
                warnings.warn(
                    "Cache metadata missing 'use_segmented_mask_preprocess'; "
                    "consider rebuilding cache for strict config consistency."
                )
            elif bool(expected_use_segmented_mask_preprocess) != self.cache_use_segmented_mask_preprocess:
                raise ValueError(
                    "Cache metadata mismatch for use_segmented_mask_preprocess. "
                    f"cache={self.cache_use_segmented_mask_preprocess}, "
                    f"config={bool(expected_use_segmented_mask_preprocess)}. "
                    "Please rebuild cache via ./train.sh cache-wild ..."
                )

        if expected_resize_mode is not None:
            if self.cache_resize_mode is None:
                warnings.warn(
                    "Cache metadata missing 'resize_mode'; consider rebuilding cache "
                    "to avoid preprocessing mismatch."
                )
            elif str(expected_resize_mode) != self.cache_resize_mode:
                raise ValueError(
                    "Cache metadata mismatch for resize_mode. "
                    f"cache={self.cache_resize_mode}, config={expected_resize_mode}. "
                    "Please rebuild cache via ./train.sh cache-wild ..."
                )

        if expected_segmented_mask_threshold is not None:
            if self.cache_segmented_mask_threshold is None:
                warnings.warn(
                    "Cache metadata missing 'segmented_mask_threshold'; "
                    "consider rebuilding cache for strict config consistency."
                )
            elif int(expected_segmented_mask_threshold) != self.cache_segmented_mask_threshold:
                raise ValueError(
                    "Cache metadata mismatch for segmented_mask_threshold. "
                    f"cache={self.cache_segmented_mask_threshold}, "
                    f"config={int(expected_segmented_mask_threshold)}. "
                    "Please rebuild cache via ./train.sh cache-wild ..."
                )

        self.transform = build_image_transform(image_size=image_size, augment=augment)
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return int(self.color.shape[0])

    def _to_tensor_fast(self, image_hwc_uint8: np.ndarray) -> torch.Tensor:
        # Fast path for cache mode when augmentation is disabled.
        x = torch.from_numpy(image_hwc_uint8).permute(2, 0, 1).contiguous().float().div_(255.0)
        x = (x - self._mean) / self._std
        return x

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample_id = str(self.sample_ids[idx])
        color_np = self.color[idx]
        segmented_np = self.segmented[idx]

        if self.augment:
            color_tensor = self.transform(Image.fromarray(color_np, mode="RGB"))
            segmented_tensor = self.transform(Image.fromarray(segmented_np, mode="RGB"))
        else:
            color_tensor = self._to_tensor_fast(color_np)
            segmented_tensor = self._to_tensor_fast(segmented_np)

        return {
            "sample_id": sample_id,
            "color_image": color_tensor,
            "segmented_image": segmented_tensor,
        }


def stable_seed_from_string(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)
