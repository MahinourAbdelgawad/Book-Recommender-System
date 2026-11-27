import argparse
import json
import logging
import os
from pathlib import Path
import re
import time

from recommender_app.ml.active_bundle import DEFAULT_POINTER, active_bundle_path
from recommender_app.ml.bundle import load_bundle
from .activation import (
    ActivationError,
    activate_candidate,
    check_active_service,
)
from .locking import exclusive_lock
from .prepare_snapshot import prepare_snapshot
from .train_candidate import run_candidate

ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "instance/library.db"
DATASET = ROOT / "preparation-and-design/dataset/cleaned_dataset.parquet"
OUTPUT_ROOT = ROOT / "instance/ml/candidates"
SNAPSHOT_DIR = ROOT / "instance/ml/retraining-snapshots"
LOGGER = logging.getLogger("recommender.retraining")


def next_candidate_version(output_root, active_version):
    versions = [int(match.group(1)) for value in [active_version]
                if (match := re.fullmatch(r"v([0-9]+)", value))]
    root = Path(output_root)
    if root.exists():
        versions.extend(
            int(match.group(1)) for child in root.iterdir()
            if (match := re.fullmatch(r"v([0-9]+)", child.name))
        )
    return f"v{max(versions, default=0) + 1:03d}"


def run_once(database, dataset, snapshot_dir, output_root, threshold, api_url, api_key,
             pointer=DEFAULT_POINTER, threads=4, book_artifacts=None, minimum_book_readers=3):
    if os.environ.get("MODEL_BUNDLE_DIR"):
        raise ActivationError(
            "Unset MODEL_BUNDLE_DIR to allow the scheduler to manage active bundles"
        )
    if not api_key:
        raise ActivationError("Set MODEL_API_KEY before starting the retraining scheduler")

    active_bundle = active_bundle_path(pointer=pointer)
    _, active_manifest = load_bundle(active_bundle)
    check_active_service(api_url, api_key, active_manifest["version"])

    snapshot_result = prepare_snapshot(
        database, active_bundle, snapshot_dir, threshold, dataset=dataset,
        book_artifacts=book_artifacts, minimum_book_readers=minimum_book_readers,
    )
    if not snapshot_result["written"]:
        return {
            "status": "below_threshold",
            "pending_ratings": snapshot_result["pending_ratings"],
            "threshold": threshold,
        }

    version = next_candidate_version(output_root, active_manifest["version"])
    candidate_report = run_candidate(
        dataset, snapshot_result["snapshot"], active_bundle,
        output_root, version, threshold, threads,
    )
    if not candidate_report["passed"]:
        return {
            "status": "candidate_rejected",
            "candidate_version": version,
            "candidate_report": str(Path(output_root) / version / "candidate_report.json"),
        }

    activation = activate_candidate(
        Path(output_root) / version / "bundle", output_root, api_url, api_key, pointer
    )
    return {
        "status": "activated",
        "candidate_version": version,
        "snapshot": str(snapshot_result["snapshot"]),
        "activation": activation,
    }


def run_scheduler(database, dataset, snapshot_dir, output_root, threshold,
                  api_url, api_key, interval_seconds, *, once=False,
                  pointer=DEFAULT_POINTER, threads=4, book_artifacts=None, minimum_book_readers=3):
    if interval_seconds <= 0:
        raise ValueError("Schedule interval must be positive")
    while True:
        try:
            with exclusive_lock(Path(pointer).with_suffix(".scheduler.lock")):
                result = run_once(
                    database, dataset, snapshot_dir, output_root, threshold,
                    api_url, api_key, pointer, threads, book_artifacts, minimum_book_readers,
                )
            LOGGER.info("Retraining cycle: %s", json.dumps(result, sort_keys=True))
        except Exception:
            LOGGER.exception("Retraining cycle failed; active bundle was not intentionally changed")
            if once:
                raise
        if once:
            return
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--interval-hours", type=float, default=24)
    parser.add_argument("--once", action="store_true",
                        help="perform one cycle instead of continuing on a schedule")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--snapshot-dir", type=Path, default=SNAPSHOT_DIR)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--book-artifacts", type=Path)
    parser.add_argument("--minimum-book-readers", type=int, default=3)
    args = parser.parse_args()
    if args.threshold < 1:
        parser.error("--threshold must be at least one")
    if args.interval_hours <= 0:
        parser.error("--interval-hours must be positive")
    if args.threads < 1:
        parser.error("--threads must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run_scheduler(
            args.database,
            args.dataset or (args.book_artifacts / "cleaned_dataset.parquet"
                             if args.book_artifacts else DATASET),
            args.snapshot_dir, args.output_root,
            args.threshold, args.api_url, os.environ.get("MODEL_API_KEY", ""),
            args.interval_hours * 3600, once=args.once, threads=args.threads,
            book_artifacts=args.book_artifacts, minimum_book_readers=args.minimum_book_readers,
        )
    except (ActivationError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
