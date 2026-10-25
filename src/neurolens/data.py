"""Manifest validation and image/mask loading for 2D segmentation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from neurolens.labels import ID_TO_LABEL

SUPPORTED_SPLITS = {"train", "val", "test"}


class ManifestRow(TypedDict):
    image_path: str
    mask_path: str
    subject_id: str
    split: str
    channel_index: str


def _resolve_manifest_path(data_root: Path, value: str, field: str) -> Path:
    if not value.strip():
        raise ValueError(f"Manifest field {field!r} cannot be empty")
    path = (data_root / value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest {field} does not exist: {path}")
    return path


def read_manifest(manifest_path: Path, data_root: Path) -> dict[str, list[ManifestRow]]:
    """Read a CSV manifest and enforce subject-disjoint splits."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: dict[str, list[ManifestRow]] = {split: [] for split in SUPPORTED_SPLITS}
    subjects_by_split: dict[str, set[str]] = defaultdict(set)
    required = {"image_path", "mask_path", "subject_id", "split"}

    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = required - fields
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        for line_number, raw_row in enumerate(reader, start=2):
            split = (raw_row.get("split") or "").strip().lower()
            if split not in SUPPORTED_SPLITS:
                raise ValueError(
                    f"Invalid split {split!r} on manifest line {line_number}; "
                    "use train, val, or test"
                )

            subject_id = (raw_row.get("subject_id") or "").strip()
            if not subject_id:
                raise ValueError(f"Empty subject_id on manifest line {line_number}")

            image_path = _resolve_manifest_path(
                data_root, raw_row.get("image_path") or "", "image_path"
            )
            mask_path = _resolve_manifest_path(
                data_root, raw_row.get("mask_path") or "", "mask_path"
            )
            row: ManifestRow = {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "subject_id": subject_id,
                "split": split,
                "channel_index": (raw_row.get("channel_index") or "").strip(),
            }
            rows[split].append(row)
            subjects_by_split[split].add(subject_id)

    if not rows["train"] or not rows["val"]:
        raise ValueError("Manifest must contain non-empty train and val splits")

    overlaps: set[str] = set()
    split_names = sorted(SUPPORTED_SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps.update(subjects_by_split[left] & subjects_by_split[right])
    if overlaps:
        raise ValueError(f"Subject leakage across splits: {sorted(overlaps)}")

    return rows


def _read_array(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
        except ImportError as exc:
            raise ImportError(
                "TIFF input requires the optional dependency; install with "
                "`pip install -e \".[tiff]\"`"
            ) from exc
        return np.asarray(tifffile.imread(path))
    with Image.open(path) as image:
        return np.asarray(image)


def image_to_float_rgb(image: np.ndarray, channel_index: str = "") -> np.ndarray:
    """Convert a grayscale or RGB image to normalized HWC float data."""
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[-1] not in (3, 4):
        if not channel_index:
            raise ValueError(
                f"Image has {image.shape[-1]} channels; specify channel_index "
                "for a channels-last multi-channel image"
            )
        index = int(channel_index)
        if not 0 <= index < image.shape[-1]:
            raise ValueError(
                f"channel_index {index} is outside the image shape {image.shape}"
            )
        image = image[..., index]

    if image.ndim == 2:
        image = image.astype(np.float32)
        finite = np.isfinite(image)
        if not finite.all():
            raise ValueError("Grayscale image contains non-finite intensity values")
        foreground = finite & (image != 0)
        values = image[foreground] if foreground.any() else image[finite]
        low, high = np.percentile(values, [0.5, 99.5])
        if high <= low:
            high = low + 1.0
        image = np.clip((image - low) / (high - low), 0.0, 1.0)
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].astype(np.float32)
        finite = np.isfinite(image)
        if not finite.all():
            raise ValueError("RGB image contains non-finite intensity values")
        if image.max(initial=0.0) > 1.0:
            image /= 255.0
        image = np.clip(image, 0.0, 1.0)
    else:
        raise ValueError(
            f"Expected a 2D grayscale or channels-last RGB image; got {image.shape}"
        )

    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((image.astype(np.float32) - mean) / std).astype(np.float32)


class BrainSliceDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Load validated 2D images and integer-valued class masks."""

    def __init__(
        self,
        rows: list[ManifestRow],
        image_size: int = 448,
        id_to_label: dict[int, str] | None = None,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.rows = rows
        self.image_size = image_size
        self.id_to_label = ID_TO_LABEL if id_to_label is None else id_to_label

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image = image_to_float_rgb(
            _read_array(image_path), row.get("channel_index", "")
        )
        mask = _read_array(mask_path)
        if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"Mask must be a 2D integer label map: {mask_path}")
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"{image.shape[:2]} vs {mask.shape}"
            )

        class_ids = set(np.unique(mask).tolist())
        unexpected = class_ids - set(self.id_to_label)
        if unexpected:
            raise ValueError(
                f"Unexpected class IDs in {mask_path}: {sorted(unexpected)}"
            )

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = F.interpolate(
            image_tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask_tensor = F.interpolate(
            mask_tensor, size=(self.image_size, self.image_size), mode="nearest"
        )[0, 0].long()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "subject_id": row["subject_id"],
        }
