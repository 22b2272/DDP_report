import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neurolens.metrics import dice_ce_loss, evaluate


def test_dice_ce_loss_is_finite_and_differentiable():
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8))

    loss = dice_ce_loss(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


class _FixedDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        mask = torch.tensor([[0, 1], [1, 0]])
        return {"image": mask.unsqueeze(0).float(), "mask": mask}


class _FixedModel(nn.Module):
    def forward(self, images):
        mask = images[:, 0].long()
        return torch.nn.functional.one_hot(mask, num_classes=2).permute(0, 3, 1, 2).float() * 10


def test_metrics_report_per_class_and_foreground_scores():
    results = evaluate(
        _FixedModel(),
        DataLoader(_FixedDataset()),
        torch.device("cpu"),
        {0: "background", 1: "region_1"},
    )

    assert results["pixel_accuracy"] == 1.0
    assert results["mean_dice_foreground"] == 1.0
    assert results["mean_iou_foreground"] == 1.0
    assert results["per_class"]["region_1"]["support"] == 2
