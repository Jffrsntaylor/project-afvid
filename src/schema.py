"""
Canonical AFVID dataset schema and class metadata.

This is the single source of truth for class IDs, names, and schema versioning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

SCHEMA_VERSION = "1.0.0"

CategoryType = Literal["Civilian", "Military"]
ThreatStatusType = Literal["Threat", "Friendly"]


@dataclass(frozen=True)
class VehicleMetadata:
    """Semantic enrichment for a single AFVID class."""

    model_label: str  # label as produced by the YOLO model
    display_name: str  # human-readable label for UI
    category: CategoryType
    nation_of_origin: str
    threat_status: ThreatStatusType


# Deterministic, stable ordering of classes. Never reorder once training begins.
CLASS_LIST: List[str] = [
    "civ_sedan",
    "civ_truck",
    "civ_motorcycle",
    "mil_t90",
    "mil_bmp3",
    "mil_btr80",
    "mil_ka52",
    "mil_leopard2",
    "mil_bradley",
    "mil_tank_unknown",
    "mil_ifv_unknown",
    "mil_apc_unknown",
    "mil_rotary_unknown",
]

CLASS_METADATA: Dict[str, VehicleMetadata] = {
    # Civilian traffic
    "civ_sedan": VehicleMetadata(
        model_label="civ_sedan",
        display_name="Sedan",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
    "civ_truck": VehicleMetadata(
        model_label="civ_truck",
        display_name="Civilian Truck",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
    "civ_motorcycle": VehicleMetadata(
        model_label="civ_motorcycle",
        display_name="Motorcycle",
        category="Civilian",
        nation_of_origin="Various",
        threat_status="Friendly",
    ),
    # Military - Russia
    "mil_t90": VehicleMetadata(
        model_label="mil_t90",
        display_name="T-90 MBT",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "mil_bmp3": VehicleMetadata(
        model_label="mil_bmp3",
        display_name="BMP-3 IFV",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "mil_btr80": VehicleMetadata(
        model_label="mil_btr80",
        display_name="BTR-80 APC",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    "mil_ka52": VehicleMetadata(
        model_label="mil_ka52",
        display_name="Ka-52 Attack Helicopter",
        category="Military",
        nation_of_origin="Russia",
        threat_status="Threat",
    ),
    # Military - NATO / Friendly
    "mil_leopard2": VehicleMetadata(
        model_label="mil_leopard2",
        display_name="Leopard 2 MBT",
        category="Military",
        nation_of_origin="Germany",
        threat_status="Friendly",
    ),
    "mil_bradley": VehicleMetadata(
        model_label="mil_bradley",
        display_name="M2 Bradley IFV",
        category="Military",
        nation_of_origin="USA",
        threat_status="Friendly",
    ),
    # Ambiguity-safe fallbacks
    "mil_tank_unknown": VehicleMetadata(
        model_label="mil_tank_unknown",
        display_name="Unknown Tank (Abstain)",
        category="Military",
        nation_of_origin="Unknown",
        threat_status="Threat",
    ),
    "mil_ifv_unknown": VehicleMetadata(
        model_label="mil_ifv_unknown",
        display_name="Unknown IFV (Abstain)",
        category="Military",
        nation_of_origin="Unknown",
        threat_status="Threat",
    ),
    "mil_apc_unknown": VehicleMetadata(
        model_label="mil_apc_unknown",
        display_name="Unknown APC (Abstain)",
        category="Military",
        nation_of_origin="Unknown",
        threat_status="Threat",
    ),
    "mil_rotary_unknown": VehicleMetadata(
        model_label="mil_rotary_unknown",
        display_name="Unknown Rotary Wing (Abstain)",
        category="Military",
        nation_of_origin="Unknown",
        threat_status="Threat",
    ),
}

# Deterministic mapping id -> name for YOLO training configs.
CLASS_ID_MAP: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_LIST)}
