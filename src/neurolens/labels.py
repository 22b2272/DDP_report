"""Segmentation class IDs used by the baseline."""

ID_TO_LABEL = {
    0: "background",
    **{class_id: f"region_{class_id}" for class_id in range(1, 11)},
}

NUM_CLASSES = len(ID_TO_LABEL)
