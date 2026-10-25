"""Segmentation losses and confusion-matrix-based metrics."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def dice_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Combine all-pixel cross entropy with foreground-class Dice loss."""
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("Expected logits [N,C,H,W] and target [N,H,W]")
    if logits.shape[0] != target.shape[0] or logits.shape[-2:] != target.shape[-2:]:
        raise ValueError("Logit and target batch/spatial dimensions must match")

    cross_entropy = F.cross_entropy(logits, target)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2)
    one_hot = one_hot.to(dtype=probabilities.dtype)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)

    foreground_support = one_hot[:, 1:].sum(dim=(0, 2, 3))
    present = foreground_support > 0
    foreground_dice_loss = (
        1.0 - dice[1:][present].mean()
        if present.any()
        else cross_entropy.new_zeros(())
    )
    return cross_entropy + foreground_dice_loss


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    """Compute pixel accuracy and per-class/foreground Dice and IoU."""
    if not id_to_label or sorted(id_to_label) != list(range(len(id_to_label))):
        raise ValueError("Class IDs must be contiguous and start at zero")

    model.eval()
    class_count = len(id_to_label)
    confusion = torch.zeros(
        (class_count, class_count), dtype=torch.int64, device=device
    )
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        predictions = model(images).argmax(dim=1)
        if masks.min() < 0 or masks.max() >= class_count:
            raise ValueError("Target mask contains class IDs outside the configured map")
        encoded = masks.reshape(-1) * class_count + predictions.reshape(-1)
        confusion += torch.bincount(
            encoded, minlength=class_count * class_count
        ).reshape(class_count, class_count)

    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()
    dice = (2 * true_positive) / (support + predicted).clamp_min(1)
    iou = true_positive / (support + predicted - true_positive).clamp_min(1)
    foreground_present = (support[1:] > 0) | (predicted[1:] > 0)
    total_pixels = confusion.sum().clamp_min(1)
    mean_dice = (
        float(dice[1:][foreground_present].mean().cpu())
        if foreground_present.any()
        else None
    )
    mean_iou = (
        float(iou[1:][foreground_present].mean().cpu())
        if foreground_present.any()
        else None
    )

    return {
        "mean_dice_foreground": mean_dice,
        "mean_iou_foreground": mean_iou,
        "pixel_accuracy": float((true_positive.sum() / total_pixels).cpu()),
        "per_class": {
            id_to_label[class_id]: {
                "dice": float(dice[class_id].cpu()),
                "iou": float(iou[class_id].cpu()),
                "support": int(support[class_id].cpu()),
            }
            for class_id in range(class_count)
        },
    }
