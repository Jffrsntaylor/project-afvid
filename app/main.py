"""
Streamlit demo for Project AFVID.

This UI is intentionally minimal and purpose-built for fieldable prototyping:
- upload an image or video
- run YOLO inference through the AFVID detector wrapper
- view annotated outputs side-by-side with a scrolling detection log
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np
import streamlit as st

# Ensure local src/ is importable without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import config
from detector import AFVIDDetector, PredictionResult


@st.cache_resource
def load_detector(weights_path: str, confidence_threshold: float) -> AFVIDDetector:
    model_cfg = config.ModelConfig(
        weights_path=weights_path,
        img_size=config.MODEL.img_size,
        confidence_threshold=confidence_threshold,
        device=config.MODEL.device,
    )
    return AFVIDDetector(model_config=model_cfg)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def decode_image(file_bytes: bytes) -> np.ndarray:
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image.")
    return image


def render_detection_log(result: PredictionResult, frame_idx: int | str, show_ignored: bool) -> List[dict]:
    log_rows = []
    for det in result.detections:
        log_rows.append(
            {
                "frame_or_time": frame_idx,
                "mode": det.mode,
                "raw_class": det.raw_class_name,
                "mapped_class": det.mapped_class_name,
                "object": det.metadata.display_name,
                "category": det.metadata.category,
                "nation": det.metadata.nation_of_origin,
                "threat_status": det.metadata.threat_status,
                "confidence": f"{det.confidence * 100:.1f}%",
                "status": "kept",
            }
        )
    if show_ignored:
        for det in result.ignored:
            log_rows.append(
                {
                    "frame_or_time": frame_idx,
                    "mode": det.mode,
                    "raw_class": det.raw_class_name,
                    "mapped_class": None,
                    "object": "",
                    "category": "",
                    "nation": "",
                    "threat_status": "",
                    "confidence": f"{det.confidence * 100:.1f}%",
                    "status": f"ignored: {det.reason}",
                }
            )
    return log_rows


def handle_image(
    detector: AFVIDDetector,
    file_bytes: bytes,
    *,
    include_aviation: bool,
    include_boats: bool,
    show_ignored: bool,
) -> None:
    image = decode_image(file_bytes)
    result = detector.predict_image(
        image,
        include_aviation=include_aviation,
        include_boats=include_boats,
        log_ignored=show_ignored,
    )
    annotated = detector.annotate(image, result.detections)

    st.subheader("Image Inference")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        st.image(bgr_to_rgb(image), channels="RGB", use_container_width=True)
    with col2:
        st.caption("Annotated")
        st.image(bgr_to_rgb(annotated), channels="RGB", use_container_width=True)

    log_rows = render_detection_log(result, datetime.utcnow().isoformat(), show_ignored)
    st.subheader("Detection Log")
    if log_rows:
        st.dataframe(log_rows, use_container_width=True, hide_index=True)
    else:
        st.info("No detections above threshold.")


def handle_video(
    detector: AFVIDDetector,
    file_bytes: bytes,
    *,
    include_aviation: bool,
    include_boats: bool,
    show_ignored: bool,
) -> None:
    st.subheader("Video Inference")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Unable to read video.")
        return

    FRAME_SAMPLE_RATE = 5  # process every Nth frame for responsiveness
    MAX_FRAMES = 150
    aggregated_logs: List[dict] = []
    annotated_preview = None
    preview_frame = None

    frame_idx = 0
    processed_frames = 0
    progress = st.progress(0.0, text="Processing video...")
    while processed_frames < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SAMPLE_RATE != 0:
            frame_idx += 1
            continue

        result = detector.predict_image(
            frame,
            include_aviation=include_aviation,
            include_boats=include_boats,
            log_ignored=show_ignored,
        )
        if annotated_preview is None:
            annotated_preview = detector.annotate(frame, result.detections)
            preview_frame = frame.copy()

        aggregated_logs.extend(render_detection_log(result, frame_idx, show_ignored))
        processed_frames += 1
        frame_idx += 1
        progress.progress(min(processed_frames / MAX_FRAMES, 1.0))

    cap.release()
    progress.empty()

    if annotated_preview is not None and preview_frame is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Representative Frame (Original)")
            st.image(bgr_to_rgb(preview_frame), channels="RGB", use_container_width=True)
        with col2:
            st.caption("Annotated Preview")
            st.image(bgr_to_rgb(annotated_preview), channels="RGB", use_container_width=True)
    else:
        st.info("No frames processed.")

    st.subheader("Detection Log")
    if aggregated_logs:
        st.dataframe(aggregated_logs, use_container_width=True, hide_index=True)
    else:
        st.info("No detections above threshold in sampled frames.")


def main() -> None:
    st.set_page_config(page_title="Project AFVID - Armored Fighting Vehicle ID", layout="wide")
    st.title("Project AFVID")
    st.caption("Armored Fighting Vehicle Identification perception demo")

    st.sidebar.header("Model Controls")
    weights_path = st.sidebar.text_input("YOLO Weights", value=config.MODEL.weights_path)
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", min_value=0.05, max_value=0.8, value=config.MODEL.confidence_threshold, step=0.05
    )
    st.sidebar.markdown(
        "- Use pretrained COCO weights for quick validation.\n"
        "- Swap in custom AFVID weights after training without code changes."
    )

    detector = load_detector(weights_path, conf_threshold)
    st.sidebar.subheader("Mode")
    st.sidebar.write(f"Auto-detected: `{detector.mode}`")
    include_aviation = st.sidebar.checkbox(
        "Include aviation classes in COCO demo mode",
        value=False,
        help="Maps airplane/helicopter to mil_rotary_unknown when enabled (COCO demo mode only).",
        disabled=detector.mode == "afvid",
    )
    include_boats = st.sidebar.checkbox(
        "Include boats in COCO demo mode",
        value=False,
        help="Maps boat to a civilian fallback when enabled (COCO demo mode only).",
        disabled=detector.mode == "afvid",
    )
    show_ignored = st.sidebar.checkbox(
        "Show ignored classes in log (debug)",
        value=False,
        help="Adds rows for detections filtered out by mapping rules.",
    )

    uploaded = st.file_uploader("Upload imagery", type=["jpg", "jpeg", "png", "bmp", "mp4", "mov", "avi"])
    if uploaded is None:
        st.info("Upload an image or video to begin.")
        return

    file_bytes = uploaded.read()
    if uploaded.type.startswith("image/"):
        handle_image(
            detector,
            file_bytes,
            include_aviation=include_aviation,
            include_boats=include_boats,
            show_ignored=show_ignored,
        )
    else:
        handle_video(
            detector,
            file_bytes,
            include_aviation=include_aviation,
            include_boats=include_boats,
            show_ignored=show_ignored,
        )


if __name__ == "__main__":
    main()
