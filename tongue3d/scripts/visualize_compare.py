from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from tongue3d.utils import load_obj, read_pointcloud_ply


def parse_cli() -> tuple[Path, Path, Path, int]:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: python -m tongue3d.scripts.visualize_compare <gt_obj> <pred_ply> [output_png] [max_points]"
        )

    gt_obj = Path(sys.argv[1])
    pred_ply = Path(sys.argv[2])
    output_png = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else Path("runs/compare") / f"{gt_obj.stem}_vs_{pred_ply.stem}.png"
    )
    max_points = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    return gt_obj, pred_ply, output_png, max_points


def subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(1234)
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


def main() -> None:
    gt_obj, pred_ply, output_png, max_points = parse_cli()

    gt_vertices, _ = load_obj(gt_obj)
    pred_points = read_pointcloud_ply(pred_ply)

    gt = subsample(gt_vertices.astype(np.float32), max_points=max_points)
    pred = subsample(pred_points.astype(np.float32), max_points=max_points)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualize_compare") from exc

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 4), dpi=220)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _scatter_splat(ax1, gt, color="#5d9ad0")
    set_axes_equal(ax1, gt)
    ax1.set_title(f"GT OBJ ({gt_obj.stem})")
    ax1.view_init(elev=20.0, azim=-65.0)
    ax1.axis("off")

    _scatter_splat(ax2, gt, color="#5d9ad0", alpha=0.18)
    _scatter_splat(ax2, pred, color="#f28b54", alpha=0.7)
    set_axes_equal(ax2, np.concatenate([gt, pred], axis=0))
    ax2.set_title("Overlay (GT blue / Pred orange)")
    ax2.view_init(elev=20.0, azim=-65.0)
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)

    print(f"Saved comparison image: {output_png}")


def _scatter_splat(ax, points: np.ndarray, color: str, alpha: float = 0.9) -> None:
    for size, scale_alpha in ((12.0, 0.10), (4.0, 0.28), (1.2, 1.0)):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=size,
            alpha=min(alpha * scale_alpha, 1.0),
            linewidths=0,
            depthshade=False,
        )


if __name__ == "__main__":
    main()
