from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tongue3d.data.dataset import (
    InTheWildPairSample,
    load_in_the_wild_manifest,
    prepare_in_the_wild_pair_arrays,
)


def parse_cli() -> tuple[Path, Path, int, bool, str, int]:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.build_in_the_wild_cache "
            "<manifest_csv> [output_npz] [image_size=224] [use_segmented_mask_preprocess=1] "
            "[resize_mode=letterbox] [segmented_mask_threshold=16]"
        )

    manifest_csv = Path(sys.argv[1])
    output_npz = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("TongueDB/in_the_wild_cache.npz")
    image_size = int(sys.argv[3]) if len(sys.argv) > 3 else 224
    use_segmented_mask_preprocess = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    resize_mode = str(sys.argv[5]) if len(sys.argv) > 5 else "letterbox"
    segmented_mask_threshold = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    return (
        manifest_csv,
        output_npz,
        image_size,
        use_segmented_mask_preprocess,
        resize_mode,
        segmented_mask_threshold,
    )


def _process_pair(
    sample: InTheWildPairSample,
    image_size: int,
    use_segmented_mask_preprocess: bool,
    resize_mode: str,
    segmented_mask_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(sample.color_path) as color_im:
        color_np = np.asarray(color_im.convert("RGB"), dtype=np.uint8)
    with Image.open(sample.segmented_path) as seg_im:
        seg_np = np.asarray(seg_im.convert("RGB"), dtype=np.uint8)

    return prepare_in_the_wild_pair_arrays(
        color_rgb=color_np,
        segmented_rgb=seg_np,
        image_size=image_size,
        use_segmented_mask_preprocess=use_segmented_mask_preprocess,
        resize_mode=resize_mode,
        segmented_mask_threshold=segmented_mask_threshold,
    )


def main() -> None:
    (
        manifest_csv,
        output_npz,
        image_size,
        use_segmented_mask_preprocess,
        resize_mode,
        segmented_mask_threshold,
    ) = parse_cli()
    if resize_mode not in {"direct", "letterbox"}:
        raise ValueError(f"resize_mode must be 'direct' or 'letterbox', got: {resize_mode}")
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
            resize_mode=resize_mode,
            segmented_mask_threshold=segmented_mask_threshold,
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
        resize_mode=np.asarray([resize_mode]),
        segmented_mask_threshold=np.asarray([segmented_mask_threshold], dtype=np.int32),
    )

    mb = output_npz.stat().st_size / (1024 * 1024)
    print(f"Saved cache: {output_npz}")
    print(f"Samples: {n}")
    print(f"Image size: {image_size}")
    print(f"Masked color: {use_segmented_mask_preprocess}")
    print(f"Resize mode: {resize_mode}")
    print(f"Segmented mask threshold: {segmented_mask_threshold}")
    print(f"File size: {mb:.2f} MB")


if __name__ == "__main__":
    main()
