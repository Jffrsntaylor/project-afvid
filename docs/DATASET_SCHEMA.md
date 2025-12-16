# Project AFVID Dataset Schema & Labeling Standard

Authoritative guidance for producing training-ready AFVID data with unambiguous classes, stable IDs, and reproducible splits. Threat/friendly status is **not** a training label; it remains config/context-derived at inference time.

## Scope & Assumptions
- Sensor modality: RGB EO imagery (8-bit). Thermal/IR is optional and out-of-scope unless explicitly flagged and kept in a separate domain-tagged split.
- Viewpoints: ground-level, mast, or low-oblique UAV (0-60 deg off-nadir). True top-down/nadir (>75 deg) is excluded unless a dedicated top-down split is created with identical class IDs. Synthetic renders are disallowed unless documented.
- Vehicle presence: label only physical vehicles (hull/cabin/turret/fuselage visible). Ignore clutter, posters, reflections, or toy/scale models. Decoys count only if visually indistinguishable from the real platform.
- Scene quality: do not label if motion blur, compression, or glare removes primary discriminators (tracks/wheels/turret geometry/rotor). Snow/dust/foliage camouflage is acceptable if the visibility rules below are met.

## Canonical Class List (Stable IDs)
Generic labels such as `tank`, `armored`, or `helicopter` are forbidden. Use the explicit, prefixed names below. Class IDs are stable and must not be reordered in future releases. Threat/friendly status is injected by config, not encoded in labels.

| id | class name          | vehicle type         | category  | visual discriminators |
|----|---------------------|----------------------|-----------|-----------------------|
| 0  | civ_sedan           | Sedan                | Civilian  | 2-box or 3-box silhouette, unarmored body panels, civilian wheels/tires |
| 1  | civ_truck           | Pickup/box truck     | Civilian  | Open or box bed, exposed axles, no armor plating/turret |
| 2  | civ_motorcycle      | Motorcycle           | Civilian  | Two wheels, handlebar geometry, rider-optional |
| 3  | mil_t90             | MBT (Russia)         | Military  | Low dome turret, 6 road wheels/side, Kontakt-5 ERA bricks, snorkel mount |
| 4  | mil_bmp3            | IFV (Russia)         | Military  | Low hull, 6 road wheels/side, rear engine deck, 100mm/30mm coax turret |
| 5  | mil_btr80           | APC (Russia)         | Military  | 8x8, boat-shaped bow, side troop hatches, roof-mounted turret |
| 6  | mil_ka52            | Attack helicopter    | Military  | Coaxial rotors, side-by-side cockpit, winglets with pylons |
| 7  | mil_leopard2        | MBT (NATO)           | Military  | Angular/wedge turret, 7 road wheels/side, side skirts, rear APU exhaust |
| 8  | mil_bradley         | IFV (NATO)           | Military  | 6 road wheels/side, tall turret, TOW launcher box, rear troop ramp |
| 9  | mil_tank_unknown    | MBT (unknown)        | Military  | Clearly a tank (tracks + main gun) but platform uncertain |
| 10 | mil_ifv_unknown     | IFV (unknown)        | Military  | Tracked turreted IFV/APC with cannon but platform uncertain |
| 11 | mil_apc_unknown     | APC (unknown)        | Military  | Wheeled APC/4x4/8x8 hull, no clear platform ID |
| 12 | mil_rotary_unknown  | Helicopter (unknown) | Military  | Rotorcraft present but platform uncertain |

All above classes are fully supported in the schema and validator. No other classes are allowed for training data.

## Ambiguity & Fallback Policy (The "13F Standard")
- Occlusion: do not label if <20% of hull/fuselage area is visible. For tracked vehicles, require at least two consecutive road wheels per side; for wheeled APCs, require at least two axles in view.
- Turret-only: if only the turret is visible, use `mil_tank_unknown` only when the gun tube base and turret ring are present; otherwise abstain.
- Partial running gear: if only a single road wheel/axle is visible, abstain. Do not infer tracks/wheels from shadows.
- Ambiguity handling:
  - Military but class unclear -> use the closest unknown class (`mil_tank_unknown`, `mil_ifv_unknown`, `mil_apc_unknown`, `mil_rotary_unknown`).
  - Nation cannot be determined -> do not guess; never substitute NATO/Russia based on camouflage alone.
  - Ground vs. rotary unclear -> abstain (do not force `mil_rotary_unknown`).
- Misidentification prevention (near-neighbors):
  - T-72 vs. T-90: T-90 has dome turret and prominent Shtora boxes; if uncertain, fallback to `mil_tank_unknown`.
  - Leopard 2 vs. Challenger-style wedges: look for 7 road wheels and rear exhaust layout; if uncertain, fallback.
  - Bradley vs. BMP: Bradley has 6 larger road wheels and TOW box; BMP-3 has lower profile and 6 smaller wheels; if uncertain, fallback to unknown IFV.
- Abstain instead of forcing a label when: severe motion blur, heavy fog/smoke, night/thermal silhouettes without platform cues, or when the vehicle is <40 pixels on the short side.

## Bounding Box Standards
- Annotation format: YOLO normalized `class x_center y_center width height` in `[0,1]`.
- Box tightness: wrap the physical vehicle (hull + turret/cabin). Exclude shadows, reflections, muzzle flash, dust clouds, and searchlights.
- Gun barrels: include to the muzzle if in-frame; keep width tight to the barrel, not to empty space beyond.
- Antennas: exclude unless they are rigid armor masts integral to silhouette. Do not chase whip antennas.
- Rotors: include the full rotor disk and stub wings for helicopters. Do not include rotor shadow.
- Occlusion: draw boxes around only the visible portion--do not hallucinate hidden geometry.
- Overlaps: each vehicle gets its own box even when overlapping; do not merge convoys or formations into a single box.

## Directory Structure (YOLOv8-Compatible)
```
data/
  afvid/
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
```
- Filenames must pair exactly: `images/<split>/<scene>/<name>.jpg` -> `labels/<split>/<scene>/<name>.txt`. Train/val data must not ship empty label files; drop truly empty scenes or keep them in a background-only set outside this schema instead of delivering blank annotations.
- Image/label counts must match per split; avoid duplicates across splits. Use a deterministic split seed and persist the split manifest to prevent drift.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`. Labels are UTF-8 `.txt`.
- Test split is optional for training but required for publishable evaluations; keep class IDs identical across splits and releases.

## Schema Stability Contract
- SCHEMA_VERSION: 1.0.0.
- Class IDs are stable and must never be reordered once training begins; add new classes by appending to the list and bumping the schema version.
- Training, validation, and reporting tools must all consume the class list from `src/schema.py`; hardcoding class IDs elsewhere is forbidden to prevent silent model breakage.
- Threat/friendly status remains context-driven and must not be encoded in labels; schema only defines platform identity.
