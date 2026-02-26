from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tongue3d.utils import load_obj, read_pointcloud_ply


def parse_cli() -> tuple[Path, Path, int, float]:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.render_blue_splat <point_cloud.(ply|obj)> [output_png] [max_points] [size_scale]"
        )

    in_path = Path(sys.argv[1])
    out_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else Path("runs/renders") / f"{in_path.stem}_blue_splat.png"
    )
    max_points = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    size_scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    return in_path, out_path, max_points, size_scale


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".ply":
        return read_pointcloud_ply(path)
    if suffix == ".obj":
        verts, _ = load_obj(path)
        return verts
    raise ValueError(f"Unsupported file type: {path}")


def subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(20260226)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1e-5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_blue_splat(ax, points: np.ndarray, size_scale: float) -> None:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Three-layer splat to mimic paper-style dense blue Gaussian-like point render.
    ax.scatter(x, y, z, s=18.0 * size_scale, c="#7fb6e5", alpha=0.08, linewidths=0, depthshade=False)
    ax.scatter(x, y, z, s=7.0 * size_scale, c="#6eaadf", alpha=0.20, linewidths=0, depthshade=False)
    ax.scatter(x, y, z, s=1.2 * size_scale, c="#5d9ad0", alpha=0.98, linewidths=0, depthshade=False)

    set_axes_equal(ax, points)
    ax.axis("off")


def main() -> None:
    in_path, out_path, max_points, size_scale = parse_cli()

    points = load_points(in_path).astype(np.float32)
    points = subsample(points, max_points=max_points)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for render_blue_splat") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    views = [(18.0, -62.0), (10.0, 18.0), (90.0, -90.0)]
    titles = ["Perspective", "Side", "Top"]

    fig = plt.figure(figsize=(11, 4), dpi=260, facecolor="white")
    for i, ((elev, azim), title) in enumerate(zip(views, titles), start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        draw_blue_splat(ax, points, size_scale=size_scale)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)

    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, facecolor="white")
    plt.close(fig)

    print(f"Saved blue-splat render: {out_path}")


if __name__ == "__main__":
    main()
