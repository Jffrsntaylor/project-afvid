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
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# Support both package import (src as a package) and direct module import (Streamlit path hack).
try:  # pragma: no cover - import shim
    from . import config  # type: ignore
except ImportError:  # pragma: no cover
    import config  # type: ignore


@dataclass(frozen=True)
class Detection:
    """Rich detection object consumed by UI layers."""

    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixel coordinates)
    confidence: float
    class_id: int
    class_name: str
    metadata: config.VehicleMetadata
    label: str
    color: Tuple[int, int, int]  # BGR for OpenCV drawing


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

    def _load_model(self, weights_path: str) -> YOLO:
        """
        Load YOLO weights.

        The model stays swappable so replacing with custom AFVID weights is a one-line change.
        """
        return YOLO(weights_path)

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

    def predict_image(self, image_bgr: np.ndarray) -> List[Detection]:
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
            return []

        result = results[0]
        detections: List[Detection] = []
        names_lookup = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = names_lookup.get(class_id, f"class_{class_id}")
            metadata = self._lookup_metadata(class_name)
            label = config.format_label(metadata, confidence)
            color = config.get_color_for_metadata(metadata)
            detections.append(
                Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    metadata=metadata,
                    label=label,
                    color=color,
                )
            )
        return detections

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
