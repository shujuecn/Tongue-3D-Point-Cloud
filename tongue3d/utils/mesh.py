from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-8


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z, *_ = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                tokens = line.strip().split()[1:]
                if len(tokens) < 3:
                    continue
                indices = []
                for token in tokens:
                    head = token.split("/")[0]
                    if not head:
                        continue
                    indices.append(int(head) - 1)
                if len(indices) == 3:
                    faces.append(indices)
                elif len(indices) > 3:
                    base = indices[0]
                    for i in range(1, len(indices) - 1):
                        faces.append([base, indices[i], indices[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"Failed to parse mesh: {path}")

    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), EPS)
    return areas, normals


def sample_points_from_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    areas, face_normals = _triangle_areas(vertices, faces)
    valid = np.where(areas > EPS)[0]
    if len(valid) == 0:
        raise ValueError("Mesh has no valid triangle areas")

    valid_areas = areas[valid]
    probs = valid_areas / np.sum(valid_areas)
    sampled_face_ids = rng.choice(valid, size=num_points, p=probs)

    tris = vertices[faces[sampled_face_ids]]

    r1 = np.sqrt(rng.random(num_points, dtype=np.float32))
    r2 = rng.random(num_points, dtype=np.float32)
    a = 1.0 - r1
    b = r1 * (1.0 - r2)
    c = r1 * r2

    points = (
        tris[:, 0] * a[:, None]
        + tris[:, 1] * b[:, None]
        + tris[:, 2] * c[:, None]
    )
    normals = face_normals[sampled_face_ids]
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), EPS)

    return points.astype(np.float32), normals.astype(np.float32)


def compute_normalization_stats(mesh_paths: Iterable[Path]) -> tuple[np.ndarray, float]:
    sum_xyz = np.zeros(3, dtype=np.float64)
    count = 0

    mesh_paths = list(mesh_paths)
    for path in mesh_paths:
        vertices, _ = load_obj(path)
        sum_xyz += vertices.sum(axis=0)
        count += int(vertices.shape[0])

    if count == 0:
        raise ValueError("No vertices were found while computing normalization stats")

    center = (sum_xyz / float(count)).astype(np.float32)

    max_norm = 0.0
    for path in mesh_paths:
        vertices, _ = load_obj(path)
        centered = vertices - center[None, :]
        norms = np.linalg.norm(centered, axis=1)
        if norms.size > 0:
            max_norm = max(max_norm, float(norms.max()))

    scale = max(max_norm, 1e-6)
    return center, scale


def normalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (points - center[None, :]) / float(scale)


def denormalize_points(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return points * float(scale) + center[None, :]
