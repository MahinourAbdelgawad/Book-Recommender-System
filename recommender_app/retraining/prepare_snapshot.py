import argparse
from datetime import datetime, timezone
from pathlib import Path

from recommender_app.ml.bundle import fingerprint, load_bundle
from .expansion import expanded_catalog
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


def prepare_snapshot(database, bundle, output_dir, threshold, *,
                     dataset=None, book_artifacts=None, minimum_book_readers=3):
    if threshold < 1:
        raise ValueError("Retraining threshold must be at least one")

    _, manifest = load_bundle(bundle)
    if book_artifacts is not None and (
        dataset is None or fingerprint(dataset) != manifest["dataset_sha256"]
    ):
        raise ValueError("Use an active baseline trained on the dataset exported with the book transforms")
    current = read_library_ratings(database)
    catalog, vectors = expanded_catalog(
        database, dataset, book_artifacts, current, catalog_book_ids(bundle), minimum_book_readers
    )
    eligible = eligible_ratings(current, catalog)
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
    if vectors:
        snapshot["static_features"] = snapshot["book_id"].map(vectors)

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
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--book-artifacts", type=Path)
    parser.add_argument("--minimum-book-readers", type=int, default=3)
    args = parser.parse_args()

    result = prepare_snapshot(
        args.database, args.bundle, args.output_dir, args.threshold, dataset=args.dataset,
        book_artifacts=args.book_artifacts, minimum_book_readers=args.minimum_book_readers,
    )
    print(f"New or changed eligible ratings: {result['pending_ratings']}")
    print(f"Threshold: {result['threshold']}")
    if result["written"]:
        print(f"Frozen snapshot saved: {result['snapshot']}")
    else:
        print("Threshold not reached; no snapshot was written.")


if __name__ == "__main__":
    main()
