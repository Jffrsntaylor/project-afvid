"""
AFVID dataset validator.

Run this before training to enforce class consistency, bounding-box sanity, and dataset hygiene.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
from tqdm import tqdm

# Make src importable without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:  # pragma: no cover - import shim
    import schema  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    from src import schema  # type: ignore  # noqa: E402

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class ValidationSummary:
    errors: List[str]
    warnings: List[str]
    class_counts: Counter
    total_images: int
    total_labels: int


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def load_class_map() -> Dict[int, str]:
    """Return the authoritative AFVID class map (id -> name) from schema."""
    return dict(schema.CLASS_ID_MAP)


def collect_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    if not root.exists():
        return []
    extensions_lower = {ext.lower() for ext in extensions}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions_lower
    ]


def parse_label_file(label_path: Path, allowed_class_ids: set[int]) -> Tuple[List[Tuple[int, float, float, float, float]], List[str]]:
    """
    Parse a YOLO label file.

    Returns a tuple of (valid_entries, errors_for_file).
    """
    errors: List[str] = []
    entries: List[Tuple[int, float, float, float, float]] = []

    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return entries, [f"{label_path}: unable to read label file ({exc})"]

    stripped_lines = [line.strip() for line in lines if line.strip() != ""]
    if not stripped_lines:
        return entries, [f"{label_path}: empty label file"]

    for idx, line in enumerate(stripped_lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}: line {idx} should have 5 values (class x_center y_center width height)")
            continue
        try:
            class_id = int(parts[0])
            bbox = [float(v) for v in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}: line {idx} has non-numeric values")
            continue

        if class_id not in allowed_class_ids:
            errors.append(f"{label_path}: line {idx} uses invalid class id {class_id}")
            continue

        x_center, y_center, width, height = bbox
        if any(value < 0.0 or value > 1.0 for value in bbox):
            errors.append(f"{label_path}: line {idx} has coordinates outside [0,1]")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"{label_path}: line {idx} has non-positive width/height")
            continue

        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2
        if x_min < 0.0 or x_max > 1.0 or y_min < 0.0 or y_max > 1.0:
            errors.append(f"{label_path}: line {idx} box extends outside image bounds")
            continue

        entries.append((class_id, x_center, y_center, width, height))

    return entries, errors


def build_image_index(images_root: Path) -> Dict[str, Path]:
    """
    Build a mapping from relative stem (without extension) to image path.

    Example key: "train/convoy/image001"
    """
    index: Dict[str, Path] = {}
    for image_path in collect_files(images_root, IMAGE_EXTENSIONS):
        rel_stem = image_path.relative_to(images_root).with_suffix("")
        key = rel_stem.as_posix()
        index[key] = image_path
    return index


def read_image_safely(image_path: Path) -> bool:
    """Return True if the image can be decoded by OpenCV."""
    image = cv2.imread(str(image_path))
    return image is not None


def check_class_balance(class_counts: Counter, class_map: Dict[int, str]) -> List[str]:
    warnings: List[str] = []

    for class_id, class_name in class_map.items():
        if class_counts.get(class_id, 0) == 0:
            warnings.append(f"Class absent: {class_name} (id {class_id})")

    non_zero = {cid: count for cid, count in class_counts.items() if count > 0}
    if not non_zero:
        warnings.append("No labeled instances found.")
        return warnings
    if len(non_zero) == 1:
        only_id, count = next(iter(non_zero.items()))
        warnings.append(f"Single-class dataset: {class_map.get(only_id, str(only_id))} ({count})")
        return warnings

    max_id, max_count = max(non_zero.items(), key=lambda item: item[1])
    min_id, min_count = min(non_zero.items(), key=lambda item: item[1])
    if min_count > 0 and max_count / min_count >= 20:
        warnings.append(
            f"High imbalance: {class_map[max_id]} ({max_count}) vs {class_map[min_id]} ({min_count})"
        )
    return warnings


def validate_dataset(dataset_root: Path) -> ValidationSummary:
    labels_root = dataset_root / "labels"
    images_root = dataset_root / "images"
    class_map = load_class_map()
    allowed_class_ids = set(class_map.keys())

    errors: List[str] = []
    warnings: List[str] = []
    class_counts: Counter = Counter()

    if not labels_root.exists():
        errors.append(f"Missing labels directory: {labels_root}")
        return ValidationSummary(errors, warnings, class_counts, total_images=0, total_labels=0)
    if not images_root.exists():
        errors.append(f"Missing images directory: {images_root}")
        return ValidationSummary(errors, warnings, class_counts, total_images=0, total_labels=0)

    label_files = collect_files(labels_root, [".txt"])
    image_index = build_image_index(images_root)
    seen_image_keys: set[str] = set()

    for label_path in tqdm(label_files, desc="Validating label files", unit="file"):
        rel_key = label_path.relative_to(labels_root).with_suffix("").as_posix()
        image_path = image_index.get(rel_key)
        if image_path is None:
            errors.append(f"{label_path}: missing paired image for key '{rel_key}'")
        else:
            seen_image_keys.add(rel_key)
            if not read_image_safely(image_path):
                errors.append(f"{image_path}: unreadable or corrupt image")

        entries, file_errors = parse_label_file(label_path, allowed_class_ids)
        errors.extend(file_errors)
        for class_id, *_ in entries:
            class_counts[class_id] += 1

    # Images without labels
    for key, image_path in image_index.items():
        if key not in seen_image_keys:
            errors.append(f"{image_path}: missing paired label file for key '{key}'")

    warnings.extend(check_class_balance(class_counts, class_map))

    return ValidationSummary(
        errors=errors,
        warnings=warnings,
        class_counts=class_counts,
        total_images=len(image_index),
        total_labels=len(label_files),
    )


def render_report(summary: ValidationSummary, class_map: Dict[int, str]) -> None:
    print("\nDataset Health Report")
    print(f"  Total images scanned : {summary.total_images}")
    print(f"  Total label files    : {summary.total_labels}")
    print("  Per-class counts (desc):")
    for class_id, count in summary.class_counts.most_common():
        print(f"    {class_map.get(class_id, str(class_id))}: {count}")
    if summary.warnings:
        print("\nWarnings:")
        for msg in summary.warnings:
            print(f"  - {msg}")
    if summary.errors:
        print("\nErrors:")
        for msg in summary.errors:
            print(f"  - {msg}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AFVID dataset integrity before training.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "afvid",
        help="Path to dataset root containing images/ and labels/ (default: data/afvid)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as fatal (non-zero exit).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    setup_logging(verbose=args.verbose)
    logging.info(
        "AFVID schema version %s | known classes: %d",
        schema.SCHEMA_VERSION,
        len(schema.CLASS_LIST),
    )
    summary = validate_dataset(args.dataset_root)
    class_map = load_class_map()

    if summary.errors:
        logging.error("Validation failed with %d error(s).", len(summary.errors))
    else:
        logging.info("Validation completed without critical errors.")

    for warning in summary.warnings:
        logging.warning(warning)
    for error in summary.errors:
        logging.error(error)

    render_report(summary, class_map)

    if summary.errors:
        return 1
    if args.strict and summary.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
