"""
Single dataset gate: optional Roboflow ingestion + strict validation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:  # pragma: no cover - import shim
    from tools import validate_dataset
    from tools import ingest_roboflow_export as ingest
except ImportError:  # pragma: no cover
    import validate_dataset  # type: ignore
    import ingest_roboflow_export as ingest  # type: ignore


class GateError(RuntimeError):
    pass


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate AFVID dataset (ingest + strict validation).")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--export-zip", type=Path, help="Optional Roboflow export .zip to ingest before validation")
    group.add_argument("--export-dir", type=Path, help="Optional Roboflow export directory to ingest before validation")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/afvid"), help="Dataset root to validate")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite during ingest (default skips duplicates)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def _run_validation(dataset_root: Path) -> int:
    summary = validate_dataset.validate_dataset(dataset_root)
    class_map = validate_dataset.load_class_map()

    for warning in summary.warnings:
        logging.warning(warning)
    for error in summary.errors:
        logging.error(error)

    validate_dataset.render_report(summary, class_map)

    if summary.errors:
        logging.error("Validation failed with %d error(s).", len(summary.errors))
        return 1
    if summary.warnings:
        logging.error("Strict mode: warnings present (%d).", len(summary.warnings))
        return 1
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s | %(message)s")

    try:
        if args.export_zip or args.export_dir:
            export_path = args.export_zip if args.export_zip else args.export_dir
            assert export_path is not None
            logging.info("Ingesting Roboflow export from %s", export_path)
            ingest.ingest_export(export_path, args.dataset_root, overwrite=args.overwrite)
        else:
            logging.info("Skipping ingestion; validating existing dataset at %s", args.dataset_root)

        rc = _run_validation(args.dataset_root)
        if rc == 0:
            print("\nDATASET GATE: PASS")
        else:
            print("\nDATASET GATE: FAIL")
        return rc
    except ingest.IngestError as exc:
        logging.error("Ingestion failed: %s", exc)
        print("\nDATASET GATE: FAIL")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        logging.exception("Unexpected error: %s", exc)
        print("\nDATASET GATE: FAIL")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
