# Project AFVID — Armored Fighting Vehicle Identification
Computer-vision replacement for traditional AFVID silhouette flashcards. Built as a defense-grade prototype to augment human operators with reliable, config-driven perception.

## Mission
Project AFVID is a prototype force protection / situational awareness module that distinguishes civilian traffic from armored fighting vehicles, enriches detections with nation-of-origin metadata, and surfaces threat vs. friendly context to reduce cognitive load and prevent misidentification.

## The “Why” — Built by a Former 13F
I built this project to solve a problem I lived personally.

As a former Forward Observer (13F) in the US Army's 75th Ranger Regiment, I spent weeks in Armored Fighting Vehicle Identification (AFVID) training, memorizing the silhouettes, turret geometries, thermal profiles, and track configurations of hundreds of vehicles using flashcards.

In the field, correctly distinguishing:

a friendly tank from a hostile tank,

a civilian truck from a technical,

or a reconnaissance vehicle from routine traffic

relies entirely on human cognitive performance under stress.

The Problem:
Human reliability degrades with fatigue. After 12+ hours on glass, pattern recognition degrades, reaction times slow, and misidentification risk increases.

The Solution:
Project AFVID (internally nicknamed Overwatch-13) offloads part of that cognitive burden. It acts as a second set of eyes that never gets tired, providing real-time probabilistic checks on:

Class: Civilian vs. Military

Origin: Vehicle identification to reduce fratricide risk (e.g., T-90 vs. Leopard 2)

Intent Baseline: Differentiating normal civilian traffic from anomalous military movement

The system does not replace the operator — it augments them.

## Architecture
- `src/config.py`: Single source of truth for class metadata (category, nation, threat/friendly), display rules, model parameters, and YOLO dataset definitions. No inference or UI code lives here.
- `src/detector.py`: Thin, UI-agnostic wrapper around Ultralytics YOLO for loading weights, running inference, enriching detections with semantics, and producing formatted labels/colors.
- `app/main.py`: Streamlit demo dashboard for rapid fieldable visualization (image/video upload, side-by-side views, scrolling detection log).

### Display semantics
- Civilian detections render green.
- Military threats render red.
- Military friendly render orange by default (distinct from civilian green and threat red).
- Labels:
  - `Civilian: Truck`
  - `Threat: T-90 [Russia]`
  - `Friendly: Leopard 2 [Germany]`

### Tech stack
- Python 3.10+
- Ultralytics YOLO (v8+)
- OpenCV
- Streamlit (demo/visualization only; perception logic stays decoupled)

## Quickstart
1. Install dependencies:
   ```bash
   pip install -U ultralytics opencv-python streamlit
   ```
2. Run the demo (uses COCO weights as placeholders):
   ```bash
   streamlit run app/main.py
   ```
3. Upload an image or video. The UI shows original vs. annotated outputs and a detection log (frame/time, category, origin, threat status).

## Pretrained weights (placeholder) vs. custom AFVID weights
- Default weights: `yolov8n.pt` (COCO) for immediate, out-of-the-box validation.
- Swap in custom AFVID weights after training by updating the "YOLO Weights" field in the Streamlit sidebar or by editing `config.MODEL.weights_path`. No other code changes are required—the metadata and display rules remain config-driven.

## Training-ready dataset config
`src/config.py` includes a YOLO-compatible dataset definition and helper to emit YAML:
```python
from src import config
print(config.build_dataset_yaml())  # copy to configs/afvid.yaml before training
```
Placeholder class mappings (extend or replace as you build your AFVID dataset):
```
0: civilian_truck
1: sedan
2: motorcycle
3: t90
4: t72
5: bmp2
6: btr80
7: m2_bradley
8: leopard2
9: challenger2
10: puma_ifv
11: cv90
```

## Why config-driven semantics matter
- Defense perception systems must adapt to changing OOB data and coalition rules; hardcoded threat logic is brittle and risky.
- All category, nation-of-origin, and threat/friendly semantics live in `config.py`, so swapping models or changing ROE-driven threat maps is a config change, not a code change.

## Rapid prototyping with Streamlit
Streamlit is used strictly as a demo layer for analysts and recruiters: quick iteration, clean UI, and zero coupling to core perception logic. The detector remains usable in headless services, notebooks, or ROS nodes without modification.

## Repository status
This is production-quality boilerplate intended for internal demos and future hardening. Extend the metadata, plug in custom AFVID weights, and integrate the detector into downstream systems as needed.
