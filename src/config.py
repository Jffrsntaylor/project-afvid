"""
Project AFVID configuration.

This module is the single source of truth for:
- model loading parameters
- class metadata (category, nation-of-origin, threat/friendly semantics)
- display rules (label formatting and colors)
- dataset definitions used for training custom AFVID weights

All perception logic consumes this config so the system stays model-agnostic and
can be re-pointed to a custom-trained AFVID dataset without touching detector or UI code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Literal

# Semantic enums kept lightweight to remain easily serializable.
CategoryType = Literal["Civilian", "Military"]
ThreatStatusType = Literal["Threat", "Friendly"]


@dataclass(frozen=True)
class VehicleMetadata:
    """Semantic enrichment for a single model class."""

    model_label: str  # label as produced by the YOLO model
    display_name: str  # human-readable label for UI
    category: CategoryType
    nation_of_origin: str
    threat_status: ThreatStatusType


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
# Keys match model output labels. For COCO placeholder weights we map the vehicle
# classes that exist; additional AFV classes are included for custom AFVID weights.
CLASS_METADATA: Dict[str, VehicleMetadata] = {
    # Civilian traffic (COCO-aligned labels)
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
    # Military platforms (AFVID-focused classes; use with custom weights)
    "t90": VehicleMetadata(
        model_label="t90",
        display_name="T-90",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "t72": VehicleMetadata(
        model_label="t72",
        display_name="T-72",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "bmp2": VehicleMetadata(
        model_label="bmp2",
        display_name="BMP-2",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "btr80": VehicleMetadata(
        model_label="btr80",
        display_name="BTR-80",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "m2_bradley": VehicleMetadata(
        model_label="m2_bradley",
        display_name="M2 Bradley",
        category="Military",
        nation_of_origin="USA",
        threat_status="Friendly",
    ),
    "leopard2": VehicleMetadata(
        model_label="leopard2",
        display_name="Leopard 2",
        category="Military",
        nation_of_origin="Germany",
        threat_status="Friendly",
    ),
    "challenger2": VehicleMetadata(
        model_label="challenger2",
        display_name="Challenger 2",
        category="Military",
        nation_of_origin="UK",
        threat_status="Friendly",
    ),
    "puma_ifv": VehicleMetadata(
        model_label="puma_ifv",
        display_name="Puma IFV",
        category="Military",
        nation_of_origin="Germany",
        threat_status="Friendly",
    ),
    "cv90": VehicleMetadata(
        model_label="cv90",
        display_name="CV90",
        category="Military",
        nation_of_origin="Sweden",
        threat_status="Friendly",
    ),
}

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
    train="data/afvid/train/images",
    val="data/afvid/val/images",
    names={
        0: "civilian_truck",
        1: "sedan",
        2: "motorcycle",
        3: "t90",
        4: "t72",
        5: "bmp2",
        6: "btr80",
        7: "m2_bradley",
        8: "leopard2",
        9: "challenger2",
        10: "puma_ifv",
        11: "cv90",
    },
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
