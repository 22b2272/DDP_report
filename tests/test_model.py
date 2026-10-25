from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neurolens.model import DinoV2Segmenter


class _FakeBackbone(nn.Module):
    config = SimpleNamespace(
        patch_size=14,
        num_register_tokens=0,
        hidden_size=8,
    )

    def forward(self, pixel_values, return_dict=True):
        batch = pixel_values.shape[0]
        return SimpleNamespace(last_hidden_state=torch.randn(batch, 17, 8))


def test_segmenter_outputs_full_resolution_logits():
    model = DinoV2Segmenter(
        num_classes=3,
        backbone=_FakeBackbone(),
        freeze_backbone=True,
    )

    logits = model(torch.randn(2, 3, 56, 56))

    assert logits.shape == (2, 3, 56, 56)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_segmenter_rejects_resolution_not_divisible_by_patch_size():
    model = DinoV2Segmenter(num_classes=3, backbone=_FakeBackbone())

    with pytest.raises(ValueError, match="divisible by patch size"):
        model(torch.randn(1, 3, 55, 56))
