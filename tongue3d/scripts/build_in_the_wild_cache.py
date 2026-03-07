from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tongue3d.data.dataset import InTheWildPairSample, load_in_the_wild_manifest


def parse_cli() -> tuple[Path, Path, int, bool]:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.build_in_the_wild_cache "
            "<manifest_csv> [output_npz] [image_size=224] [use_segmented_mask_preprocess=1]"
        )

    manifest_csv = Path(sys.argv[1])
    output_npz = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("TongueDB/in_the_wild_cache.npz")
    image_size = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    use_segmented_mask_preprocess = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    return manifest_csv, output_npz, image_size, use_segmented_mask_preprocess


def _load_resized_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (image_size, image_size):
            im = im.resize((image_size, image_size), resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def _mask_from_segmented(segmented_rgb: np.ndarray, threshold: int = 5) -> np.ndarray:
    gray = segmented_rgb.max(axis=2)
    return gray >= threshold


def _process_pair(
    sample: InTheWildPairSample,
    image_size: int,
    use_segmented_mask_preprocess: bool,
) -> tuple[np.ndarray, np.ndarray]:
    color_np = _load_resized_rgb(sample.color_path, image_size=image_size)
    seg_np = _load_resized_rgb(sample.segmented_path, image_size=image_size)

    if use_segmented_mask_preprocess:
        mask = _mask_from_segmented(seg_np)
        color_np = color_np.copy()
        color_np[~mask] = 0

    return color_np, seg_np


def main() -> None:
    manifest_csv, output_npz, image_size, use_segmented_mask_preprocess = parse_cli()
    samples = load_in_the_wild_manifest(manifest_csv)
    n = len(samples)

    color = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    segmented = np.empty((n, image_size, image_size, 3), dtype=np.uint8)
    sample_ids = np.empty((n,), dtype=f"<U{max(16, max(len(s.sample_id) for s in samples))}")

    iterator = samples
    try:
        from tqdm import tqdm

        iterator = tqdm(samples, desc="build-wild-cache", total=n)
    except Exception:
        pass

    for i, sample in enumerate(iterator):
        c, s = _process_pair(
            sample=sample,
            image_size=image_size,
            use_segmented_mask_preprocess=use_segmented_mask_preprocess,
        )
        color[i] = c
        segmented[i] = s
        sample_ids[i] = sample.sample_id

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        sample_ids=sample_ids,
        color=color,
        segmented=segmented,
        image_size=np.asarray([image_size], dtype=np.int32),
        use_segmented_mask_preprocess=np.asarray(
            [1 if use_segmented_mask_preprocess else 0],
            dtype=np.int32,
        ),
    )

    mb = output_npz.stat().st_size / (1024 * 1024)
    print(f"Saved cache: {output_npz}")
    print(f"Samples: {n}")
    print(f"Image size: {image_size}")
    print(f"Masked color: {use_segmented_mask_preprocess}")
    print(f"File size: {mb:.2f} MB")


if __name__ == "__main__":
    main()

