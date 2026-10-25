import csv

import numpy as np
import pytest
import torch
from PIL import Image

from neurolens.data import BrainSliceDataset, image_to_float_rgb, read_manifest


def _write_sample(root, stem, subject, split, mask_values=(0, 1)):
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    image_path = image_dir / f"{stem}.png"
    mask_path = mask_dir / f"{stem}.png"
    Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8)).save(image_path)
    Image.fromarray(
        np.resize(np.asarray(mask_values, dtype=np.uint8), (8, 8))
    ).save(mask_path)
    return {
        "image_path": str(image_path.relative_to(root)),
        "mask_path": str(mask_path.relative_to(root)),
        "subject_id": subject,
        "split": split,
    }


def _write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["image_path", "mask_path", "subject_id", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_requires_disjoint_subjects_and_valid_paths(tmp_path):
    rows = [
        _write_sample(tmp_path, "train", "subject-a", "train"),
        _write_sample(tmp_path, "val", "subject-b", "val"),
    ]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    splits = read_manifest(manifest, tmp_path)

    assert len(splits["train"]) == 1
    assert len(splits["val"]) == 1


def test_manifest_rejects_subject_leakage(tmp_path):
    rows = [
        _write_sample(tmp_path, "train", "subject-a", "train"),
        _write_sample(tmp_path, "val", "subject-a", "val"),
    ]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match="Subject leakage"):
        read_manifest(manifest, tmp_path)


def test_dataset_resizes_image_and_mask_without_new_mask_values(tmp_path):
    rows = [
        _write_sample(tmp_path, "sample", "subject-a", "train"),
        _write_sample(tmp_path, "validation", "subject-b", "val"),
    ]
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rows)
    row = read_manifest(manifest, tmp_path)["train"][0]

    sample = BrainSliceDataset([row], image_size=14)[0]

    assert sample["image"].shape == (3, 14, 14)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].shape == (14, 14)
    assert set(sample["mask"].unique().tolist()) <= {0, 1}


def test_image_conversion_rejects_ambiguous_multichannel_input():
    with pytest.raises(ValueError, match="specify channel_index"):
        image_to_float_rgb(np.zeros((8, 8, 7), dtype=np.uint16))

    converted = image_to_float_rgb(
        np.zeros((8, 8, 7), dtype=np.uint16), channel_index="2"
    )
    assert converted.shape == (8, 8, 3)
