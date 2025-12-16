"""
Project AFVID configuration.

This module stays model-agnostic by centralizing:
- model loading parameters
- class metadata (category, nation-of-origin, threat/friendly semantics)
- display rules (label formatting and colors)
- dataset definitions used for training custom AFVID weights

Class IDs and names are defined exclusively in src/schema.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

# Canonical schema (single source of truth for class IDs and metadata)
try:  # pragma: no cover - import shim for package vs. module usage
    from . import schema  # type: ignore
except ImportError:  # pragma: no cover
    import schema  # type: ignore

CategoryType = schema.CategoryType
ThreatStatusType = schema.ThreatStatusType
VehicleMetadata = schema.VehicleMetadata


@dataclass(frozen=True)
class ModelConfig:
    """Model loading configuration."""

    weights_path: str
    img_size: int
    confidence_threshold: float
    device: str | None = None  # e.g., "cpu", "cuda:0"


@dataclass(frozen=True)
class DatasetConfig:
    """
    YOLO dataset definition used for training a custom AFVID model.

    The paths are placeholders; point them to your dataset root when training.
    """

    train: str
    val: str
    names: Dict[int, str]


# --------------------------------------------------------------------------- #
# Model defaults
# --------------------------------------------------------------------------- #

MODEL = ModelConfig(
    weights_path="yolov8n.pt",  # placeholder COCO weights for immediate demo use
    img_size=640,
    confidence_threshold=0.25,
    device=None,
)


# --------------------------------------------------------------------------- #
# Class metadata
# --------------------------------------------------------------------------- #
# Canonical AFVID metadata (single source of truth is schema.py)
AFVID_CLASS_METADATA: Dict[str, VehicleMetadata] = schema.CLASS_METADATA

# COCO-aligned metadata preserved for demo compatibility with public weights.
BASELINE_COCO_METADATA: Dict[str, VehicleMetadata] = {
    "car": VehicleMetadata(
        model_label="car",
        display_name="Sedan",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
    "truck": VehicleMetadata(
        model_label="truck",
        display_name="Truck",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
    "bus": VehicleMetadata(
        model_label="bus",
        display_name="Bus",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
    "motorcycle": VehicleMetadata(
        model_label="motorcycle",
        display_name="Motorcycle",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
}

# Combined metadata used by the detector; canonical AFVID classes plus COCO demo support.
CLASS_METADATA: Dict[str, VehicleMetadata] = {**BASELINE_COCO_METADATA, **AFVID_CLASS_METADATA}

# --------------------------------------------------------------------------- #
# Display rules
# --------------------------------------------------------------------------- #
# Colors are BGR tuples (OpenCV default).
DISPLAY_COLOR_RULES: Dict[str, Tuple[int, int, int]] = {
    "Civilian": (46, 204, 113),  # green
    "Threat": (0, 0, 255),  # red
    "Friendly": (255, 140, 0),  # orange accent for friendly military
}

# Toggle to append confidence percentages to labels.
LABEL_INCLUDE_CONFIDENCE: bool = True


def get_color_for_metadata(metadata: VehicleMetadata) -> Tuple[int, int, int]:
    """
    Pick display color based on semantic intent.

    Civilian detections stay green.
    Military detections inherit threat/friendly color coding.
    """
    if metadata.category == "Civilian":
        return DISPLAY_COLOR_RULES["Civilian"]
    return DISPLAY_COLOR_RULES.get(metadata.threat_status, DISPLAY_COLOR_RULES["Threat"])


def format_label(metadata: VehicleMetadata, confidence: float | None = None) -> str:
    """
    Build the on-screen label string per AFVID display rules.

    Civilian: "Civilian: Truck"
    Threat:   "Threat: T-90 [Russia]"
    Friendly: "Friendly: Leopard 2 [Germany]"
    """
    if metadata.category == "Civilian":
        label = f"Civilian: {metadata.display_name}"
    else:
        prefix = "Friendly" if metadata.threat_status == "Friendly" else "Threat"
        label = f"{prefix}: {metadata.display_name} [{metadata.nation_of_origin}]"
    if LABEL_INCLUDE_CONFIDENCE and confidence is not None:
        label = f"{label} {confidence * 100:.1f}%"
    return label


# --------------------------------------------------------------------------- #
# Dataset config (placeholder values for custom AFVID training)
# --------------------------------------------------------------------------- #
DATASET = DatasetConfig(
    train="data/afvid/images/train",
    val="data/afvid/images/val",
    names=dict(schema.CLASS_ID_MAP),
)


def build_dataset_yaml(dataset: DatasetConfig = DATASET, dataset_root: str | Path | None = None) -> str:
    """
    Generate a YOLO-compatible dataset YAML string for training.

    Parameters
    ----------
    dataset:
        DatasetConfig describing train/val paths and class names.
    dataset_root:
        Optional path prefix for train/val. If provided, paths are rendered relative
        to that root so the YAML can travel with the dataset.
    """
    base_path = Path(dataset_root) if dataset_root is not None else Path(".")
    names_lines = "\n".join(
        f"  {idx}: {name}" for idx, name in sorted(dataset.names.items(), key=lambda item: item[0])
    )
    yaml_content = (
        f"# AFVID YOLO dataset definition\n"
        f"path: {base_path.as_posix()}\n"
        f"train: {Path(dataset.train).as_posix()}\n"
        f"val: {Path(dataset.val).as_posix()}\n\n"
        f"names:\n{names_lines}\n"
    )
    return yaml_content
