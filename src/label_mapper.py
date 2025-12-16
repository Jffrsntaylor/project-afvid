"""
Mapping utilities for bridging COCO demo outputs to AFVID schema classes.

This module centralizes:
- automatic mode detection (COCO demo vs AFVID-trained)
- deterministic class mappings for COCO vehicle-like classes
"""

from __future__ import annotations

from typing import Dict, Iterable, Literal

try:  # pragma: no cover - shim for package vs module import
    from . import schema  # type: ignore
except ImportError:  # pragma: no cover
    import schema  # type: ignore

ModeLiteral = Literal["coco", "afvid"]

COCO_BASE_VEHICLE_MAP: Dict[str, str] = {
    "car": "civ_sedan",
    "truck": "civ_truck",
    "bus": "civ_truck",
    "train": "civ_truck",
    "motorcycle": "civ_motorcycle",
}

# Aliases/variants that occasionally surface depending on model zoo weights.
COCO_ALIASES: Dict[str, str] = {
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
}

AVIATION_CLASSES = {"airplane", "helicopter"}
BOAT_CLASSES = {"boat"}


def _iter_model_names(model_names: Dict[int, str] | Iterable[str]) -> Iterable[str]:
    """Iterate over raw class names regardless of the container shape."""
    if isinstance(model_names, dict):
        return model_names.values()
    return model_names


def detect_mode(model_names: Dict[int, str] | Iterable[str]) -> ModeLiteral:
    """
    Determine whether the model is AFVID-trained or COCO demo based on class names.

    If any class starts with "mil_", we treat the model as AFVID mode.
    """
    for name in _iter_model_names(model_names):
        if isinstance(name, str) and name.lower().startswith("mil_"):
            return "afvid"
    return "coco"


def map_detection(
    class_name: str,
    mode: ModeLiteral,
    *,
    include_aviation: bool = False,
    include_boats: bool = False,
) -> str | None:
    """
    Map a raw detector class name to an AFVID schema class name.

    Returns None if the class should be ignored.
    """
    normalized = class_name.strip().lower()

    if mode == "afvid":
        return normalized if normalized in schema.CLASS_LIST else None

    # COCO demo mode mappings
    normalized = COCO_ALIASES.get(normalized, normalized)
    if normalized in AVIATION_CLASSES:
        return "mil_rotary_unknown" if include_aviation else None
    if normalized in BOAT_CLASSES:
        return "civ_truck" if include_boats else None
    if normalized in COCO_BASE_VEHICLE_MAP:
        return COCO_BASE_VEHICLE_MAP[normalized]
    return None
