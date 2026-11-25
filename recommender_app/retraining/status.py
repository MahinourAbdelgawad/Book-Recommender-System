import argparse
import hashlib
import json
from pathlib import Path

from .export import (
    DEFAULT_BUNDLE,
    DEFAULT_DATABASE,
    MIN_READER_RATINGS,
    catalog_book_ids,
    eligible_ratings,
    read_library_ratings,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "instance/ml/retraining-state.json"


def rating_signatures(ratings):
    """Map each reader/work to a digest of its current selected rating row."""
    result = {}
    for row in ratings.itertuples(index=False):
        key = f"{row.app_user_id}\x1f{row.work_id}"
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


def read_included_state(path, active_version):
    path = Path(path)
    if not path.exists():
        if active_version != "v001":
            raise ValueError(
                "Missing retraining state for a non-baseline bundle; refusing to guess "
                "which app ratings it contains"
            )
        return {}
    state = json.loads(path.read_text())
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported retraining state schema")
    if state.get("active_bundle_version") != active_version:
        raise ValueError(
            "Retraining state does not match the selected active bundle; refusing to count ratings"
        )
    included = state.get("included_ratings")
    if not isinstance(included, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in included.items()
    ):
        raise ValueError("Retraining state has an invalid included_ratings map")
    return included


def status(database, bundle, state_path, threshold, minimum_reader_ratings=MIN_READER_RATINGS):
    """Read the app DB and activation state and return a threshold summary."""
    from recommender_app.ml.bundle import load_bundle

    _, manifest = load_bundle(bundle)
    ratings = read_library_ratings(database)
    current = eligible_ratings(
        ratings, catalog_book_ids(bundle), minimum_reader_ratings
    )
    included = read_included_state(state_path, manifest["version"])
    return check_threshold(current, included, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--threshold", type=int, required=True,
                        help="new or changed eligible ratings needed before retraining")
    parser.add_argument("--minimum-reader-ratings", type=int, default=MIN_READER_RATINGS)
    args = parser.parse_args()

    summary = status(args.database, args.bundle, args.state, args.threshold,
                     args.minimum_reader_ratings)
    print(f"Eligible app ratings: {summary['eligible_ratings']}")
    print(f"Previously included in active bundle: {summary['previously_included_ratings']}")
    print(f"New or changed eligible ratings: {summary['pending_ratings']}")
    print(f"Retraining threshold: {summary['threshold']}")
    print("Threshold reached." if summary["threshold_met"] else "Threshold not reached.")
    print("Read-only check: no snapshot, state, or model files were written.")


if __name__ == "__main__":
    main()
