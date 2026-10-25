import torch
import pytest
from transformers import Dinov2Config, Dinov2Model

from neurolens.labels import ID_TO_LABEL
from neurolens.model import DinoV2Segmenter
from neurolens.predict import load_model, predict


def test_checkpoint_round_trip_without_network_download(tmp_path):
    config = Dinov2Config(
        image_size=28,
        patch_size=14,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_register_tokens=0,
    )
    model = DinoV2Segmenter(
        num_classes=len(ID_TO_LABEL),
        backbone=Dinov2Model(config),
        backbone_config=config,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": "test-model",
            "backbone_config": config.to_dict(),
            "num_classes": len(ID_TO_LABEL),
            "image_size": 28,
            "id_to_label": ID_TO_LABEL,
            "freeze_backbone": True,
        },
        checkpoint_path,
    )

    loaded, labels, image_size = load_model(checkpoint_path, torch.device("cpu"))

    assert labels == ID_TO_LABEL
    assert image_size == 28
    assert loaded(torch.randn(1, 3, 28, 28)).shape == (1, len(ID_TO_LABEL), 28, 28)


def test_prediction_rejects_lossy_output_format(tmp_path):
    with pytest.raises(ValueError, match="lossless PNG"):
        predict(
            tmp_path / "missing-checkpoint.pt",
            tmp_path / "missing-image.png",
            tmp_path / "mask.jpg",
        )
