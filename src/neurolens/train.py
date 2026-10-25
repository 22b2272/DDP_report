"""Train and evaluate the 2D DINOv2 segmentation baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neurolens.data import BrainSliceDataset, read_manifest
from neurolens.labels import ID_TO_LABEL, NUM_CLASSES
from neurolens.metrics import dice_ce_loss, evaluate
from neurolens.model import DEFAULT_MODEL_NAME, DinoV2Segmenter


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 1:
        raise ValueError("epochs, batch_size, and image_size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = read_manifest(Path(args.manifest), Path(args.data_root))
    train_loader = DataLoader(
        BrainSliceDataset(splits["train"], args.image_size),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        BrainSliceDataset(splits["val"], args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = DinoV2Segmenter(
        num_classes=NUM_CLASSES,
        model_name=args.model_name,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_validation_dice = -1.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = dice_ce_loss(model(images), masks)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())

        validation_metrics = evaluate(
            model, validation_loader, device, ID_TO_LABEL
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": total_loss / len(train_loader),
                    "validation": validation_metrics,
                }
            )
        )
        score = validation_metrics["mean_dice_foreground"]
        if score is not None and score > best_validation_dice:
            best_validation_dice = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model_name,
                    "backbone_config": model.backbone.config.to_dict(),
                    "num_classes": NUM_CLASSES,
                    "image_size": args.image_size,
                    "id_to_label": ID_TO_LABEL,
                    "freeze_backbone": not args.unfreeze_backbone,
                    "best_validation_dice_foreground": best_validation_dice,
                    "seed": args.seed,
                },
                output,
            )

    if best_validation_dice < 0:
        raise RuntimeError(
            "No validation checkpoint was selected; check that validation "
            "contains foreground classes."
        )

    print(
        f"Best validation foreground Dice: {best_validation_dice:.4f}; "
        f"checkpoint: {output}"
    )

    if splits["test"]:
        checkpoint = torch.load(output, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loader = DataLoader(
            BrainSliceDataset(splits["test"], args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
        )
        print(
            json.dumps(
                {"test": evaluate(model, test_loader, device, ID_TO_LABEL)},
                indent=2,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="checkpoints/neurolens_dinov2.pt")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.image_size % 14 != 0:
        raise SystemExit("--image-size must be divisible by DINOv2-base patch size 14")
    train(args)


if __name__ == "__main__":
    main()
