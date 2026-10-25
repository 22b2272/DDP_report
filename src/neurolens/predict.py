"""Run inference with a trained NeuroLens segmentation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Dinov2Config, Dinov2Model

from neurolens.data import _read_array, image_to_float_rgb
from neurolens.model import DinoV2Segmenter


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[
    DinoV2Segmenter, dict[int, str], int
]:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    config = Dinov2Config.from_dict(checkpoint["backbone_config"])
    backbone = Dinov2Model(config)
    model = DinoV2Segmenter(
        num_classes=int(checkpoint["num_classes"]),
        model_name=checkpoint["model_name"],
        freeze_backbone=bool(checkpoint["freeze_backbone"]),
        backbone=backbone,
        backbone_config=config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    id_to_label = {int(key): value for key, value in checkpoint["id_to_label"].items()}
    return model, id_to_label, int(checkpoint["image_size"])


def predict(
    checkpoint_path: Path,
    image_path: Path,
    output_path: Path,
    channel_index: str = "",
) -> None:
    if output_path.suffix.lower() != ".png":
        raise ValueError("Predicted class-ID masks must be saved as lossless PNG files")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, id_to_label, image_size = load_model(checkpoint_path, device)
    image_array = _read_array(image_path)
    image = image_to_float_rgb(image_array, channel_index)
    input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    input_tensor = torch.nn.functional.interpolate(
        input_tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).to(device)

    with torch.inference_mode():
        logits = model(input_tensor)
        logits = torch.nn.functional.interpolate(
            logits,
            size=image_array.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(output_path)
    pixel_counts = {
        id_to_label[class_id]: int((mask == class_id).sum())
        for class_id in sorted(id_to_label)
    }
    print(json.dumps({"mask": str(output_path), "pixel_counts": pixel_counts}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--channel-index",
        default="",
        help="Channel to select from a channels-last multi-channel image",
    )
    args = parser.parse_args()
    predict(args.checkpoint, args.image, args.output, args.channel_index)


if __name__ == "__main__":
    main()
