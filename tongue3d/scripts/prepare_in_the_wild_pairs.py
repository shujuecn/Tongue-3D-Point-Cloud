from __future__ import annotations

import csv
import sys
from pathlib import Path


def parse_cli() -> tuple[Path, Path, Path]:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.prepare_in_the_wild_pairs "
            "<color_dir> <segmented_dir> [output_csv]"
        )
    color_dir = normalize_input_path(sys.argv[1])
    segmented_dir = normalize_input_path(sys.argv[2])
    output_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("TongueDB/in_the_wild_pairs.csv")
    validate_mnt_path(color_dir, "color_dir")
    validate_mnt_path(segmented_dir, "segmented_dir")
    return color_dir, segmented_dir, output_csv


def normalize_input_path(raw: str) -> Path:
    s = raw.strip().strip('"').strip("'")
    s = s.replace("\\", "/")
    return Path(s)


def validate_mnt_path(path: Path, name: str) -> None:
    parts = path.as_posix().split("/")
    # Expected WSL mount style: /mnt/f/...
    if len(parts) < 4 or parts[1] != "mnt" or len(parts[2]) != 1 or not parts[2].isalpha():
        raise ValueError(
            f"{name} must use WSL mount path style '/mnt/<drive>/...'. Got: {path}"
        )


def collect_images(root: Path) -> dict[str, Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return {p.stem: p for p in sorted(files)}


def main() -> None:
    color_dir, segmented_dir, output_csv = parse_cli()
    if not color_dir.exists():
        raise FileNotFoundError(f"Color image directory not found: {color_dir}")
    if not segmented_dir.exists():
        raise FileNotFoundError(f"Segmented image directory not found: {segmented_dir}")

    color_map = collect_images(color_dir)
    segmented_map = collect_images(segmented_dir)
    common = sorted(set(color_map).intersection(segmented_map))
    if len(common) == 0:
        raise ValueError("No matched files by stem between color and segmented directories")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "color_path", "segmented_path"])
        for sid in common:
            writer.writerow([sid, str(color_map[sid]), str(segmented_map[sid])])

    print(f"Saved manifest: {output_csv}")
    print(f"Matched pairs: {len(common)}")
    print(f"Color dir: {color_dir}")
    print(f"Segmented dir: {segmented_dir}")


if __name__ == "__main__":
    main()
