from __future__ import annotations

import torch
from torch import nn


def soft_iou_loss(
    probs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    dims = tuple(range(1, probs.dim()))
    intersection = (probs * targets).sum(dim=dims)
    union = (probs + targets - probs * targets).sum(dim=dims)
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def bce_iou_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 1.0,
    iou_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
    )
    probs = torch.sigmoid(logits)
    iou = soft_iou_loss(probs, targets)
    loss = bce_weight * bce + iou_weight * iou
    return loss, {
        "bce": float(bce.detach().cpu().item()),
        "iou": float(iou.detach().cpu().item()),
        "total": float(loss.detach().cpu().item()),
    }
