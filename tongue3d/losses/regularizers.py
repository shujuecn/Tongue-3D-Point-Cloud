from __future__ import annotations

import torch


EPS = 1e-8


def _batched_subset(points: torch.Tensor, sample_size: int) -> torch.Tensor:
    b, n, _ = points.shape
    if sample_size <= 0 or sample_size >= n:
        return points

    idx = torch.randint(0, n, (b, sample_size), device=points.device)
    idx = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, dim=1, index=idx)


def _knn_neighbors(base: torch.Tensor, ref: torch.Tensor, k: int) -> torch.Tensor:
    dist = torch.cdist(base, ref, p=2)
    k_query = min(k + 1, ref.shape[1])
    indices = torch.topk(dist, k=k_query, dim=2, largest=False).indices
    if k_query > 1:
        indices = indices[:, :, 1:]

    idx = indices.unsqueeze(-1).expand(-1, -1, -1, ref.shape[-1])
    expanded_ref = ref.unsqueeze(1).expand(-1, base.shape[1], -1, -1)
    neighbors = torch.gather(expanded_ref, dim=2, index=idx)
    return neighbors


def laplacian_smoothness_loss(points: torch.Tensor, k: int = 12, sample_size: int = 768) -> torch.Tensor:
    base = _batched_subset(points, sample_size=sample_size)
    neighbors = _knn_neighbors(base, points, k=k)
    mean_neighbors = neighbors.mean(dim=2)
    return ((base - mean_neighbors) ** 2).sum(dim=-1).mean()


def edge_length_regularizer(points: torch.Tensor, k: int = 12, sample_size: int = 768) -> torch.Tensor:
    base = _batched_subset(points, sample_size=sample_size)
    neighbors = _knn_neighbors(base, points, k=k)
    edge_lengths = torch.sqrt(((base.unsqueeze(2) - neighbors) ** 2).sum(dim=-1) + EPS)

    mean_length = edge_lengths.mean(dim=2)
    return mean_length.var(dim=1).mean() + edge_lengths.var(dim=2).mean()
