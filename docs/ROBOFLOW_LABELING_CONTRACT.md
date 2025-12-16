# Project AFVID Roboflow Labeling Contract (Schema v1.0.0 - Immutable)

This contract is binding for all Roboflow labeling work on Project AFVID. The AFVID schema in `src/schema.py` (SCHEMA_VERSION 1.0.0) is authoritative and MUST NOT be changed, reordered, or remapped. Any deviation invalidates the dataset.

## A) Canonical Class Mapping (Authoritative)
AFVID classes, Roboflow labels, and YOLO IDs are identical and must remain stable.

| AFVID Class        | Roboflow Label    | YOLO Class ID | Notes                                  |
|--------------------|-------------------|---------------|----------------------------------------|
| civ_sedan          | civ_sedan         | 0             | Must not be renamed.                   |
| civ_truck          | civ_truck         | 1             | Includes pickup + box trucks.          |
| civ_motorcycle     | civ_motorcycle    | 2             | Two-wheel civilian motorcycles only.   |
| mil_t90            | mil_t90           | 3             | Never use generic “tank”.              |
| mil_bmp3           | mil_bmp3          | 4             | IFV; do not alias to “bmp”.            |
| mil_btr80          | mil_btr80         | 5             | 8x8 APC; do not alias to “btr”.        |
| mil_ka52           | mil_ka52          | 6             | Rotary; only Ka-52, no other rotorcraft.|
| mil_leopard2       | mil_leopard2      | 7             | NATO MBT; do not use “leo2”.           |
| mil_bradley        | mil_bradley       | 8             | IFV; do not use “brad”.                |
| mil_tank_unknown   | mil_tank_unknown  | 9             | Use only when tank is clear but model unknown. |
| mil_ifv_unknown    | mil_ifv_unknown   | 10            | Turreted IFV/APC unknown platform.     |
| mil_apc_unknown    | mil_apc_unknown   | 11            | Wheeled APC/4x4/8x8 unknown platform.  |
| mil_rotary_unknown | mil_rotary_unknown| 12            | Rotorcraft present, platform unknown.  |

Rules:
- Roboflow label names MUST match AFVID class names exactly.
- No aliases, no spaces, no capitalization changes, no reordering, no auto-mapping.

## B) Explicitly Forbidden Labels (DO NOT USE)
Use of these labels is a contract breach: tank; armored; military_vehicle; ifv; apc; helicopter; enemy_tank; friendly_vehicle; armored_car; mbt; generic_vehicle; unknown_vehicle. Reasons: schema pollution, loss of deterministic IDs, threat/friendly leakage, and unverifiable mappings during export.

## C) Ambiguity Handling Rules (Non-Negotiable)
- Use `mil_tank_unknown` only when the object is clearly a tank (tracks + main gun) but platform cannot be identified.
- Use `mil_ifv_unknown` for tracked turreted IFV/APC with cannon when platform is unclear.
- Use `mil_apc_unknown` for wheeled APC/4x4/8x8 armored hull when platform is unclear.
- Use `mil_rotary_unknown` only when a rotorcraft is visible but platform is unclear.
- Abstain (NO LABEL) when visibility is insufficient (<20% hull/fuselage), severe blur, heavy fog/smoke, or object <40px on short side.
- Turret-only views: label `mil_tank_unknown` only if gun tube base and turret ring are visible; otherwise abstain.
- Partial occlusion: require at least two consecutive road wheels per side for tracked vehicles or two axles for wheeled APC; otherwise abstain.
- Night/thermal: label only if platform cues are present; otherwise abstain or use appropriate unknown class if military is certain but platform is not.
- Escalate for review when ambiguity persists after applying the above rules; do not guess.

## D) Labeling Workflow in Roboflow (Step-by-Step)
1. Project creation: set project type to Object Detection; disable any auto-class creation or suggested classes; ensure class order is manual.
2. Class list import: manually enter the CLASS_LIST exactly in order (see checklist), no extras. Verify count == 13.
3. Image upload constraints: RGB EO imagery only; no synthetic/renders; no top-down/nadir unless a dedicated split is approved; ensure filenames are stable and deterministic.
4. Annotation rules: tight boxes around physical vehicle only; exclude shadows, reflections, muzzle flash, dust, searchlights, and rotor shadows; include full rotor disk for helicopters; include gun barrel to muzzle; do not hallucinate occluded geometry.
5. Quality control: perform a first-pass label, then a second-pass review (different reviewer when possible); resolve ambiguities per Section C; document escalations.
6. Export settings: select YOLOv8; disable class remapping/merging; preserve class order as entered; disable augmentations at export; ensure train/val/test splits are stable and non-overlapping.

## E) Export Configuration (Critical)
- Export format: YOLOv8.
- Class ordering: MUST preserve Roboflow class order identical to AFVID CLASS_LIST.
- No auto-remapping, no class merging, no relabeling.
- No augmentation at export time.

Export Checklist (all items must be confirmed and recorded):
- [ ] Class count equals 13 (len(CLASS_LIST)).
- [ ] Class names exactly match AFVID schema (see Section A).
- [ ] YOLO IDs match schema order 0-12.
- [ ] Splits (train/val/test) have no duplicate images.
- [ ] No empty label files are exported.

## F) Post-Export Verification Gate (Mandatory)
Run locally before merge:
```
python tools/validate_dataset.py --dataset-root data/afvid --strict
```
If this gate fails, the dataset is rejected and MUST NOT be used for training until fixed and revalidated.

## How to Export and Ingest into AFVID (Mandatory Sequence)
- Export from Roboflow as YOLOv8 with the exact class list and order from Section A (13 classes, no extras).
- Ingest and gate locally before any merge or training:
  - With a zip export: `python tools/gate_dataset.py --export-zip path/to/roboflow.zip`
  - With an unpacked export: `python tools/gate_dataset.py --export-dir path/to/export`
  - To re-validate an existing dataset: `python tools/gate_dataset.py --dataset-root data/afvid`
- If gate fails, the dataset is rejected until corrected and re-run through the gate.

Troubleshooting (fix before re-running gate):
- Class order mismatch: ensure Roboflow class list matches Section A exactly (name and index); re-export after correcting order.
- Roboflow “valid” split: allowed; gate normalizes to `val`. If absent, add a proper val split and re-export.
- Missing test split: permitted, but remain explicit; if provided, it must follow the same structure.
- Nested images/labels folders: flatten to `images/{train,val,test}/...` and `labels/{train,val,test}/...` before re-running gate.

## G) Versioning & Change Control
- Schema is immutable at SCHEMA_VERSION 1.0.0. Changes require a version bump in `src/schema.py`, a new Roboflow project version, and a new dataset version. Do not alter past datasets retroactively.
- Any schema change requires an updated contract and reissuance to labelers before work resumes.
- Training on datasets that do not match the active schema version is prohibited.
