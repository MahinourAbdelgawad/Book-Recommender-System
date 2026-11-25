import argparse
import hashlib
import json
import os
from pathlib import Path

from .export import (
    DEFAULT_BUNDLE,
    DEFAULT_DATABASE,
    MIN_READER_RATINGS,
    catalog_book_ids,
    eligible_ratings,
    read_library_ratings,
)

PROVENANCE_SCHEMA_VERSION = 1


def rating_signatures(ratings):
    """Map each reader/work to a digest of its current selected rating row."""
    required = {"app_user_id", "work_id", "book_id", "rating", "updated_at"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Rating provenance is missing columns: {sorted(missing)}")
    result = {}
    for row in ratings.itertuples(index=False):
        identity = f"{row.app_user_id}\x1f{row.work_id}"
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        contents = json.dumps(
            [str(row.book_id), int(row.rating), str(row.updated_at)],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        result[key] = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    return result


def check_threshold(current, previously_included, threshold):
    if threshold < 1:
        raise ValueError("threshold must be at least one")
    current_signatures = rating_signatures(current)
    pending = {
        key: signature for key, signature in current_signatures.items()
        if previously_included.get(key) != signature
    }
    return {
        "eligible_ratings": len(current_signatures),
        "previously_included_ratings": len(previously_included),
        "pending_ratings": len(pending),
        "threshold": threshold,
        "threshold_met": len(pending) >= threshold,
    }


def included_ratings_from_manifest(manifest):
    """Read inclusion metadata stored in the selected bundle itself."""
    metadata = manifest.get("retraining")
    if metadata is None:
        if manifest.get("version") == "v001":
            return {}
        raise ValueError(
            "The selected non-baseline bundle has no retraining provenance; "
            "refusing to guess which app ratings it contains"
        )
    if metadata.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("Unsupported bundle retraining provenance schema")
    included = metadata.get("included_ratings")
    if not isinstance(included, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in included.items()
    ):
        raise ValueError("Bundle retraining provenance has an invalid included_ratings map")
    return included


def bundle_retraining_metadata(ratings, snapshot_sha256=None):
    """Build private, non-identifying rating provenance for a candidate bundle."""
    included = rating_signatures(ratings)
    metadata = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "included_ratings": included,
    }
    if snapshot_sha256 is not None:
        metadata["snapshot_sha256"] = snapshot_sha256
    return metadata


def status(database, bundle, threshold, minimum_reader_ratings=MIN_READER_RATINGS):
    """Read the app DB and selected bundle and return a threshold summary."""
    from recommender_app.ml.bundle import load_bundle

    _, manifest = load_bundle(bundle)
    ratings = read_library_ratings(database)
    current = eligible_ratings(
        ratings, catalog_book_ids(bundle), minimum_reader_ratings
    )
    included = included_ratings_from_manifest(manifest)
    return check_threshold(current, included, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bundle", type=Path,
                        default=Path(os.environ.get("MODEL_BUNDLE_DIR", DEFAULT_BUNDLE)))
    parser.add_argument("--threshold", type=int, required=True,
                        help="new or changed eligible ratings needed before retraining")
    parser.add_argument("--minimum-reader-ratings", type=int, default=MIN_READER_RATINGS)
    args = parser.parse_args()

    summary = status(args.database, args.bundle, args.threshold,
                     args.minimum_reader_ratings)
    print(f"Eligible app ratings: {summary['eligible_ratings']}")
    print(f"Previously included in active bundle: {summary['previously_included_ratings']}")
    print(f"New or changed eligible ratings: {summary['pending_ratings']}")
    print(f"Retraining threshold: {summary['threshold']}")
    print("Threshold reached." if summary["threshold_met"] else "Threshold not reached.")
    print("Read-only check: no snapshot, state, or model files were written.")


if __name__ == "__main__":
    main()
