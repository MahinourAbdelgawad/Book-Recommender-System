import argparse
from datetime import datetime, timezone
from pathlib import Path

from recommender_app.ml.bundle import load_bundle
from .export import (
    DEFAULT_BUNDLE,
    DEFAULT_DATABASE,
    catalog_book_ids,
    eligible_ratings,
    read_library_ratings,
)
from .status import check_threshold, included_ratings_from_manifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "instance/ml/retraining-snapshots"


def prepare_snapshot(database, bundle, output_dir, threshold):
    if threshold < 1:
        raise ValueError("Retraining threshold must be at least one")

    _, manifest = load_bundle(bundle)
    current = read_library_ratings(database)
    eligible = eligible_ratings(current, catalog_book_ids(bundle))
    previously_included = included_ratings_from_manifest(manifest)
    summary = check_threshold(eligible, previously_included, threshold)
    if not summary["threshold_met"]:
        return {**summary, "snapshot": None, "written": False}

    output_dir = Path(output_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_dir / f"ratings-{timestamp}.parquet"
    snapshot = eligible[["app_user_id", "book_id", "work_id", "rating", "updated_at"]].copy()
    snapshot["app_user_id"] = snapshot["app_user_id"].astype(str)
    snapshot["book_id"] = snapshot["book_id"].astype(str)

    output_dir.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as destination:
        snapshot.to_parquet(destination, index=False)
    return {**summary, "snapshot": output, "written": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, required=True,
                        help="new or changed eligible ratings needed before retraining")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = prepare_snapshot(
        args.database, args.bundle, args.output_dir, args.threshold,
    )
    print(f"New or changed eligible ratings: {result['pending_ratings']}")
    print(f"Threshold: {result['threshold']}")
    if result["written"]:
        print(f"Frozen snapshot saved: {result['snapshot']}")
    else:
        print("Threshold not reached; no snapshot was written.")


if __name__ == "__main__":
    main()
