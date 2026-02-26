from __future__ import annotations

from pathlib import Path

import numpy as np


def write_pointcloud_ply(path: Path, points: np.ndarray, normals: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if normals is None:
        with path.open("w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
            f.write("end_header\n")
            for p, n in zip(points, normals):
                f.write(
                    f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                    f"{n[0]:.8f} {n[1]:.8f} {n[2]:.8f}\n"
                )


def read_pointcloud_ply(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        line = f.readline().strip()
        if line != "ply":
            raise ValueError(f"Not a PLY file: {path}")

        num_vertices = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            line = line.strip()
            if line.startswith("element vertex "):
                num_vertices = int(line.split()[-1])
            if line == "end_header":
                break

        if num_vertices is None:
            raise ValueError(f"PLY file missing vertex count: {path}")

        points = []
        for _ in range(num_vertices):
            row = f.readline()
            if not row:
                break
            vals = row.strip().split()
            if len(vals) < 3:
                continue
            points.append([float(vals[0]), float(vals[1]), float(vals[2])])

    if not points:
        raise ValueError(f"No points found in PLY: {path}")
    return np.asarray(points, dtype=np.float32)
