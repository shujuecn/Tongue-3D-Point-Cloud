from .checkpoint import load_checkpoint, load_normalization, save_checkpoint, save_normalization
from .mesh import (
    compute_normalization_stats,
    denormalize_points,
    load_obj,
    normalize_points,
    sample_points_from_mesh,
)
from .ply import read_pointcloud_ply, write_pointcloud_ply
from .seed import seed_everything


def has_matplotlib() -> bool:
    from .visualize import has_matplotlib as _has_matplotlib

    return _has_matplotlib()


def save_autoencoder_visual(*args, **kwargs):
    from .visualize import save_autoencoder_visual as _save_autoencoder_visual

    return _save_autoencoder_visual(*args, **kwargs)


def save_image2shape_visual(*args, **kwargs):
    from .visualize import save_image2shape_visual as _save_image2shape_visual

    return _save_image2shape_visual(*args, **kwargs)

__all__ = [
    "seed_everything",
    "load_obj",
    "sample_points_from_mesh",
    "compute_normalization_stats",
    "normalize_points",
    "denormalize_points",
    "write_pointcloud_ply",
    "read_pointcloud_ply",
    "save_checkpoint",
    "load_checkpoint",
    "save_normalization",
    "load_normalization",
    "has_matplotlib",
    "save_autoencoder_visual",
    "save_image2shape_visual",
]
