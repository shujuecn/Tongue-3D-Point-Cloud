from __future__ import annotations

import torch


def gather_by_index(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    idx = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, dim=1, index=idx)


def chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    return_indices: bool = False,
    chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    loss, _, _, idx_pred_to_gt, idx_gt_to_pred = chamfer_with_neighbors(
        pred_points=pred_points,
        gt_points=gt_points,
        chunk_size=chunk_size,
    )
    if return_indices:
        return loss, idx_pred_to_gt, idx_gt_to_pred
    return loss


def chamfer_with_neighbors(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    chunk_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if chunk_size and chunk_size > 0:
        return _chamfer_stats_chunked(
            pred_points=pred_points,
            gt_points=gt_points,
            chunk_size=chunk_size,
        )
    return _chamfer_stats_full(
        pred_points=pred_points,
        gt_points=gt_points,
    )


def _chamfer_stats_full(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = torch.cdist(pred_points, gt_points, p=2)
    min_pred, idx_pred_to_gt = torch.min(dist, dim=2)
    min_gt, idx_gt_to_pred = torch.min(dist, dim=1)

    loss = (min_pred.pow(2).mean()) + (min_gt.pow(2).mean())
    return loss, min_pred, min_gt, idx_pred_to_gt, idx_gt_to_pred


def _chamfer_stats_chunked(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, n_pred, _ = pred_points.shape
    _, n_gt, _ = gt_points.shape

    pred_mins: list[torch.Tensor] = []
    pred_indices: list[torch.Tensor] = []
    for start in range(0, n_pred, chunk_size):
        end = min(start + chunk_size, n_pred)
        pred_chunk = pred_points[:, start:end, :]
        dist = torch.cdist(pred_chunk, gt_points, p=2)
        vals, idx = torch.min(dist, dim=2)
        pred_mins.append(vals)
        pred_indices.append(idx)

    gt_mins: list[torch.Tensor] = []
    gt_indices: list[torch.Tensor] = []
    for start in range(0, n_gt, chunk_size):
        end = min(start + chunk_size, n_gt)
        gt_chunk = gt_points[:, start:end, :]
        dist = torch.cdist(gt_chunk, pred_points, p=2)
        vals, idx = torch.min(dist, dim=2)
        gt_mins.append(vals)
        gt_indices.append(idx)

    min_pred = torch.cat(pred_mins, dim=1)
    min_gt = torch.cat(gt_mins, dim=1)
    idx_pred_to_gt = torch.cat(pred_indices, dim=1)
    idx_gt_to_pred = torch.cat(gt_indices, dim=1)

    loss = (min_pred.pow(2).mean()) + (min_gt.pow(2).mean())
    return loss, min_pred, min_gt, idx_pred_to_gt, idx_gt_to_pred


def normal_alignment_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    idx_pred_to_gt: torch.Tensor,
    idx_gt_to_pred: torch.Tensor,
) -> torch.Tensor:
    matched_gt_normals = gather_by_index(gt_normals, idx_pred_to_gt)
    cos_pred = torch.abs((pred_normals * matched_gt_normals).sum(dim=-1)).mean()

    matched_pred_normals = gather_by_index(pred_normals, idx_gt_to_pred)
    cos_gt = torch.abs((gt_normals * matched_pred_normals).sum(dim=-1)).mean()

    return (1.0 - cos_pred) + (1.0 - cos_gt)
