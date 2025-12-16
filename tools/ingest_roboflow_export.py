"""
Roboflow YOLO export ingestion into canonical AFVID layout.

- Validates class ordering against schema.CLASS_LIST (immutable).
- Normalizes split names (train/val/test) and copies images/labels.
- Skips duplicates by default; optional overwrite.
- Emits canonical data/afvid.yaml from schema/config.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

try:  # pragma: no cover - import shim for package vs module usage
    from src import schema, config  # type: ignore
except ImportError:  # pragma: no cover
    import schema  # type: ignore
    import config  # type: ignore

try:
    import yaml  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyyaml is required for ingest_roboflow_export.py") from exc

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_KEYS = [
    ("train", "train"),
    ("val", "val"),
    ("valid", "val"),
    ("validation", "val"),
    ("test", "test"),
]


class IngestError(RuntimeError):
    pass


def _load_yaml(path: Path) -> Mapping:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise IngestError(f"YAML at {path} is not a mapping")
    return data


def _find_data_yaml(root: Path) -> Path:
    candidates = []
    for name in ("data.yaml", "dataset.yaml"):
        candidates.extend(root.rglob(name))
    if not candidates:
        raise IngestError("No data.yaml or dataset.yaml found in export")
    # Prefer the shallowest path to avoid nested archives inside
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def _resolve_path(base: Path, raw: str) -> Path:
    p = Path(raw)
    return (base / p).resolve() if not p.is_absolute() else p


def _normalize_names(names: object) -> List[str]:
    if isinstance(names, list):
        return [str(n) for n in names]
    if isinstance(names, dict):
        try:
            ordered_keys = sorted(names.keys(), key=lambda k: int(k))
            return [str(names[k]) for k in ordered_keys]
        except Exception as exc:  # pragma: no cover - defensive
            raise IngestError(f"Unable to normalize names from dict: {exc}") from exc
    raise IngestError("names field must be a list or dict")


def _validate_class_order(export_names: List[str], schema_names: List[str]) -> None:
    if len(export_names) != len(schema_names):
        raise IngestError(
            f"Class count mismatch: export={len(export_names)} vs schema={len(schema_names)}"
        )
    for idx, (expected, got) in enumerate(zip(schema_names, export_names)):
        if expected != got:
            raise IngestError(
                f"Class mismatch at index {idx}: schema='{expected}' vs export='{got}'"
            )


def _guess_labels_dir(images_dir: Path) -> Path:
    if not images_dir.exists():
        raise IngestError(f"Images directory does not exist: {images_dir}")

    candidates = []
    if images_dir.name == "images":
        candidates.append(images_dir.parent / "labels")
    # Replace first occurrence of "images" in path
    parts = list(images_dir.parts)
    for i, part in enumerate(parts):
        if part == "images":
            candidate = Path(*parts[:i]) / "labels" / Path(*parts[i + 1 :])
            candidates.append(candidate)
            break
    candidates.append(images_dir.with_name("labels"))

    for cand in candidates:
        if cand.exists():
            return cand
    raise IngestError(f"Could not locate labels directory corresponding to {images_dir}")


def _gather_split_dirs(data_cfg: Mapping, yaml_path: Path) -> Dict[str, Tuple[Path, Path]]:
    yaml_dir = yaml_path.parent
    splits: Dict[str, Tuple[Path, Path]] = {}
    for key, normalized in SPLIT_KEYS:
        if key not in data_cfg:
            continue
        images_dir = _resolve_path(yaml_dir, str(data_cfg[key]))
        labels_dir = _guess_labels_dir(images_dir)
        splits[normalized] = (images_dir, labels_dir)
    if "train" not in splits:
        raise IngestError("Export is missing train split path")
    if "val" not in splits:
        raise IngestError("Export is missing val/valid split path")
    return splits


def _copy_tree(src_dir: Path, dst_dir: Path, *, overwrite: bool, exts: set[str] | None) -> Tuple[int, int]:
    copied = 0
    skipped = 0
    if not src_dir.exists():
        raise IngestError(f"Source directory missing: {src_dir}")
    for path in src_dir.rglob("*"):
        if path.is_dir():
            continue
        if exts is not None and path.suffix.lower() not in exts:
            continue
        rel = path.relative_to(src_dir)
        dest_path = dst_dir / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and not overwrite:
            skipped += 1
            continue
        shutil.copy2(path, dest_path)
        copied += 1
    return copied, skipped


def _write_dataset_yaml(dataset_root: Path) -> None:
    yaml_text = config.build_dataset_yaml(dataset_root=dataset_root)
    target = dataset_root.parent / "afvid.yaml"
    target.write_text(yaml_text, encoding="utf-8")


def ingest_export(export_path: Path, dataset_root: Path, overwrite: bool = False) -> None:
    dataset_root = dataset_root.resolve()
    dataset_images = dataset_root / "images"
    dataset_labels = dataset_root / "labels"
    dataset_images.mkdir(parents=True, exist_ok=True)
    dataset_labels.mkdir(parents=True, exist_ok=True)

    working_dir: Path
    if export_path.is_file():
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(export_path, "r") as zf:
                zf.extractall(tmpdir)
            working_dir = Path(tmpdir)
            _ingest_from_dir(working_dir, dataset_root, overwrite)
    else:
        working_dir = export_path
        _ingest_from_dir(working_dir, dataset_root, overwrite)



def _ingest_from_dir(source_dir: Path, dataset_root: Path, overwrite: bool) -> None:
    data_yaml = _find_data_yaml(source_dir)
    cfg = _load_yaml(data_yaml)

    names = _normalize_names(cfg.get("names"))
    _validate_class_order(names, schema.CLASS_LIST)

    splits = _gather_split_dirs(cfg, data_yaml)

    total_copied = 0
    total_skipped = 0
    for split, (img_dir, lbl_dir) in splits.items():
        dest_images = dataset_root / "images" / split
        dest_labels = dataset_root / "labels" / split
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        copied_i, skipped_i = _copy_tree(img_dir, dest_images, overwrite=overwrite, exts=IMAGE_EXTENSIONS)
        copied_l, skipped_l = _copy_tree(lbl_dir, dest_labels, overwrite=overwrite, exts={".txt"})
        logging.info(
            "Split %s: images copied=%d skipped=%d | labels copied=%d skipped=%d",
            split,
            copied_i,
            skipped_i,
            copied_l,
            skipped_l,
        )
        total_copied += copied_i + copied_l
        total_skipped += skipped_i + skipped_l

    _write_dataset_yaml(dataset_root)
    logging.info("Ingestion complete. Files copied: %d, skipped: %d", total_copied, total_skipped)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a Roboflow YOLO export into AFVID layout.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export-zip", type=Path, help="Path to Roboflow export .zip")
    group.add_argument("--export-dir", type=Path, help="Path to unpacked Roboflow export directory")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/afvid"), help="Destination dataset root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files instead of skipping")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(message)s")

    try:
        export_path = args.export_zip if args.export_zip else args.export_dir
        assert export_path is not None
        ingest_export(export_path, args.dataset_root, overwrite=args.overwrite)
    except IngestError as exc:
        logging.error("Ingestion failed: %s", exc)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Unexpected error during ingestion: %s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
