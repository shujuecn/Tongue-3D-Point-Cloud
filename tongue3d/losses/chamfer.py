from __future__ import annotations

import torch


def gather_by_index(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    idx = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, dim=1, index=idx)


def chamfer_distance(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    return_indices: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    dist = torch.cdist(pred_points, gt_points, p=2)
    min_pred, idx_pred_to_gt = torch.min(dist, dim=2)
    min_gt, idx_gt_to_pred = torch.min(dist, dim=1)

    loss = (min_pred.pow(2).mean()) + (min_gt.pow(2).mean())

    if return_indices:
        return loss, idx_pred_to_gt, idx_gt_to_pred
    return loss


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
