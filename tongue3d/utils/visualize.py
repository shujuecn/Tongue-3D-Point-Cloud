from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def has_matplotlib() -> bool:
    return _HAS_MPL


def save_autoencoder_visual(
    out_path: Path,
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    sample_id: str,
    max_points: int = 4096,
) -> bool:
    if not _HAS_MPL:
        return False

    gt = _to_numpy_points(gt_points)
    pred = _to_numpy_points(pred_points)
    gt = _subsample(gt, max_points=max_points)
    pred = _subsample(pred, max_points=max_points)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 4), dpi=220)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    _plot_points(ax1, gt, title=f"GT ({sample_id})", color="#5d9ad0", style="splat")
    _plot_points(ax2, pred, title=f"Reconstruction ({sample_id})", color="#5d9ad0", style="splat")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def save_image2shape_visual(
    out_path: Path,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    sample_id: str,
    max_points: int = 4096,
) -> bool:
    if not _HAS_MPL:
        return False

    gt = _subsample(_to_numpy_points(gt_points), max_points=max_points)
    pred = _subsample(_to_numpy_points(pred_points), max_points=max_points)
    image_np = _to_numpy_image(image)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 4), dpi=220)
    ax_img = fig.add_subplot(1, 3, 1)
    ax_gt = fig.add_subplot(1, 3, 2, projection="3d")
    ax_overlay = fig.add_subplot(1, 3, 3, projection="3d")

    ax_img.imshow(image_np)
    ax_img.set_title(f"Input ({sample_id})")
    ax_img.axis("off")

    _plot_points(ax_gt, gt, title="GT Surface Points", color="#5d9ad0", style="splat")

    _plot_points(ax_overlay, gt, title="Overlay GT/Pred", color="#5d9ad0", alpha=0.18, style="splat")
    _scatter_points(ax_overlay, pred, color="#f28b54", alpha=0.75)
    _set_axes_equal(ax_overlay, np.concatenate([gt, pred], axis=0))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _to_numpy_points(points: torch.Tensor) -> np.ndarray:
    pts = points.detach().cpu().float().numpy()
    return np.asarray(pts, dtype=np.float32)


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD[None, None, :] + IMAGENET_MEAN[None, None, :]
    img = np.clip(img, 0.0, 1.0)
    return img


def _subsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(12345)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def _scatter_points(ax, points: np.ndarray, color: str, alpha: float = 0.9, style: str = "plain") -> None:
    if style == "splat":
        # Layered circles approximate dense Gaussian splat appearance for snapshots.
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=10.0,
            alpha=min(alpha, 0.12),
            linewidths=0,
            depthshade=False,
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=4.0,
            alpha=min(alpha + 0.1, 0.32),
            linewidths=0,
            depthshade=False,
        )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        s=1.2,
        alpha=alpha,
        linewidths=0,
        depthshade=False,
    )


def _plot_points(
    ax,
    points: np.ndarray,
    title: str,
    color: str,
    alpha: float = 0.9,
    style: str = "plain",
) -> None:
    _scatter_points(ax, points, color=color, alpha=alpha, style=style)
    _set_axes_equal(ax, points)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20.0, azim=-65.0)


def _set_axes_equal(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    radius = float(np.max(maxs - mins) * 0.55)
    radius = max(radius, 1e-5)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
