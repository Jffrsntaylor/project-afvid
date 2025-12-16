"""
AFVID detector wrapper around Ultralytics YOLO.

Responsibilities:
- model loading
- running inference
- mapping raw detections to AFVID semantic metadata
- generating display-ready labels and colors
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Support both package import (src as a package) and direct module import (Streamlit path hack).
try:  # pragma: no cover - import shim
    from . import config  # type: ignore
    from . import label_mapper  # type: ignore
except ImportError:  # pragma: no cover
    import config  # type: ignore
    import label_mapper  # type: ignore


@dataclass(frozen=True)
class Detection:
    """Rich detection object consumed by UI layers."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    raw_class_name: str
    mapped_class_name: str
    metadata: config.VehicleMetadata
    label: str
    color: Tuple[int, int, int]  # BGR for OpenCV drawing
    mode: label_mapper.ModeLiteral


@dataclass(frozen=True)
class IgnoredDetection:
    """Detections filtered out by mapping rules (for optional logging/debug)."""

    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    raw_class_name: str
    reason: str
    mode: label_mapper.ModeLiteral


@dataclass(frozen=True)
class PredictionResult:
    """Batch of detections plus any ignored entries."""

    detections: List[Detection]
    ignored: List[IgnoredDetection]
    mode: label_mapper.ModeLiteral


class AFVIDDetector:
    """UI-agnostic detector wrapper."""

    def __init__(
        self,
        model_config: config.ModelConfig = config.MODEL,
        class_metadata: Dict[str, config.VehicleMetadata] = config.CLASS_METADATA,
    ) -> None:
        self.model_config = model_config
        self.class_metadata = class_metadata
        self.model = self._load_model(model_config.weights_path)
        self.model_names = self._normalize_model_names(self.model.model.names)
        self.mode: label_mapper.ModeLiteral = label_mapper.detect_mode(self.model_names)

    def _load_model(self, weights_path: str) -> YOLO:
        """
        Load YOLO weights.

        The model stays swappable so replacing with custom AFVID weights is a one-line change.
        """
        return YOLO(weights_path)

    def _normalize_model_names(self, model_names: Dict[int, str] | Iterable[str]) -> Dict[int, str]:
        """Normalize Ultralytics names into a dict[int, str] regardless of source shape."""
        if isinstance(model_names, dict):
            return {int(idx): name for idx, name in model_names.items()}
        return {idx: name for idx, name in enumerate(model_names)}

    def _lookup_metadata(self, class_name: str) -> config.VehicleMetadata:
        """
        Retrieve metadata for a class; default to a conservative civilian-friendly label.
        This keeps the pipeline tolerant to unknown classes without flagging false threats.
        """
        if class_name in self.class_metadata:
            return self.class_metadata[class_name]
        return config.VehicleMetadata(
            model_label=class_name,
            display_name=class_name,
            category="Civilian",
            nation_of_origin="Unknown",
            threat_status="Friendly",
        )

    def _ignore_reason(
        self, raw_class_name: str, *, include_aviation: bool, include_boats: bool
    ) -> str:
        normalized = raw_class_name.strip().lower()
        normalized = label_mapper.COCO_ALIASES.get(normalized, normalized)

        if self.mode == "coco":
            if normalized in label_mapper.AVIATION_CLASSES and not include_aviation:
                return "Aviation filtered (toggle off)"
            if normalized in label_mapper.BOAT_CLASSES and not include_boats:
                return "Boat filtered (toggle off)"
        return "No AFVID mapping"

    def predict_image(
        self,
        image_bgr: np.ndarray,
        *,
        include_aviation: bool = False,
        include_boats: bool = False,
        log_ignored: bool = False,
    ) -> PredictionResult:
        """
        Run inference on a single image and return enriched detections.

        Parameters
        ----------
        image_bgr:
            OpenCV BGR image.
        """
        results = self.model.predict(
            image_bgr,
            conf=self.model_config.confidence_threshold,
            imgsz=self.model_config.img_size,
            device=self.model_config.device,
            verbose=False,
        )
        if not results:
            return PredictionResult(detections=[], ignored=[], mode=self.mode)

        result = results[0]
        detections: List[Detection] = []
        ignored: List[IgnoredDetection] = []
        names_lookup = self._normalize_model_names(getattr(result, "names", self.model_names))
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            raw_class_name = names_lookup.get(class_id, f"class_{class_id}")
            mapped_class = label_mapper.map_detection(
                raw_class_name,
                self.mode,
                include_aviation=include_aviation,
                include_boats=include_boats,
            )
            if mapped_class is None:
                if log_ignored:
                    ignored.append(
                        IgnoredDetection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=confidence,
                            class_id=class_id,
                            raw_class_name=raw_class_name,
                            reason=self._ignore_reason(
                                raw_class_name,
                                include_aviation=include_aviation,
                                include_boats=include_boats,
                            ),
                            mode=self.mode,
                        )
                    )
                continue

            metadata = self._lookup_metadata(mapped_class)
            label = config.format_label(metadata, confidence)
            color = config.get_color_for_metadata(metadata)
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                    class_id=class_id,
                    raw_class_name=raw_class_name,
                    mapped_class_name=mapped_class,
                    metadata=metadata,
                    label=label,
                    color=color,
                    mode=self.mode,
                )
            )
        return PredictionResult(detections=detections, ignored=ignored, mode=self.mode)

    def annotate(self, image_bgr: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes and labels on an image.

        This function stays UI-agnostic so Streamlit, notebooks, and other consumers
        can reuse the same annotation utility.
        """
        output = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(output, (x1, y1), (x2, y2), det.color, thickness=2)
            label = det.label
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Text background box to preserve readability on bright/dark imagery
            text_origin = (x1, max(y1 - 5, text_h + 2))
            cv2.rectangle(
                output,
                (x1, text_origin[1] - text_h - baseline - 2),
                (x1 + text_w + 2, text_origin[1] + baseline),
                det.color,
                thickness=cv2.FILLED,
            )
            cv2.putText(
                output,
                label,
                (x1 + 1, text_origin[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
        return output
