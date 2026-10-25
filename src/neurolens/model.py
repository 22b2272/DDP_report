"""DINOv2 feature extractor and 2D semantic-segmentation head."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Dinov2Config, Dinov2Model

DEFAULT_MODEL_NAME = "facebook/dinov2-base"


class DinoV2Segmenter(nn.Module):
    """DINOv2 patch features with a compact convolutional segmentation head."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = DEFAULT_MODEL_NAME,
        freeze_backbone: bool = True,
        backbone: nn.Module | None = None,
        backbone_config: Dinov2Config | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must include background and foreground")
        if backbone is None:
            self.backbone = Dinov2Model.from_pretrained(model_name)
        else:
            self.backbone = backbone

        config = backbone_config or self.backbone.config
        self.patch_size = int(config.patch_size)
        self.register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        hidden_size = int(config.hidden_size)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def train(self, mode: bool = True) -> DinoV2Segmenter:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        height, width = pixel_values.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"Input height and width must be divisible by patch size "
                f"{self.patch_size}; got {(height, width)}"
            )

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        else:
            outputs = self.backbone(pixel_values=pixel_values, return_dict=True)

        patch_tokens = outputs.last_hidden_state[
            :, 1 + self.register_tokens :, :
        ]
        grid_height = height // self.patch_size
        grid_width = width // self.patch_size
        expected_tokens = grid_height * grid_width
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Patch count mismatch: got {patch_tokens.shape[1]}, "
                f"expected {expected_tokens}"
            )

        features = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], -1, grid_height, grid_width
        )
        logits = self.head(features)
        return F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
