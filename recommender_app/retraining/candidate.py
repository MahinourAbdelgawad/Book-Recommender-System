from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recommender_app.ml.bundle import fingerprint, save_bundle
from recommender_app.ml.features import Examples, FeatureData
from .dataset import app_reader_transform, app_training_rows
from .export import MIN_READER_RATINGS
from .status import bundle_retraining_metadata


@dataclass
class CandidateInputs:
    """Training inputs derived from one immutable app-rating snapshot."""

    data: FeatureData
    train: Examples
    ratings: pd.DataFrame
    provenance: dict
    snapshot_sha256: str


def prepare_candidate_inputs(dataset: Path, snapshot: Path) -> CandidateInputs:
    """Build training data and provenance from the same exported rating rows."""
    ratings = pd.read_parquet(snapshot)
    if "work_id" not in ratings.columns:
        raise ValueError(
            "This rating snapshot predates work-level provenance. Re-export it before "
            "preparing a candidate bundle."
        )
    required = {"app_user_id", "book_id", "work_id", "rating", "updated_at"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Rating snapshot is missing columns: {sorted(missing)}")
    if ratings.duplicated(["app_user_id", "work_id"]).any():
        raise ValueError("Rating snapshot must contain at most one selected edition per reader/work")
    values = pd.to_numeric(ratings["rating"], errors="raise")
    if (not values.between(1, 5).all()) or ((values % 1) != 0).any():
        raise ValueError("Snapshot ratings must be whole stars between 1 and 5")
    reader_counts = ratings.groupby("app_user_id")["work_id"].nunique()
    if not reader_counts.empty and (reader_counts < MIN_READER_RATINGS).any():
        raise ValueError(
            f"Every snapshot reader must have at least {MIN_READER_RATINGS} eligible ratings"
        )
    app_rows = app_training_rows(dataset, ratings)
    data = FeatureData(dataset, extra_train=app_rows)
    snapshot_hash = fingerprint(snapshot)
    return CandidateInputs(
        data=data,
        train=data.warm("train"),
        ratings=ratings,
        provenance=bundle_retraining_metadata(
            ratings, snapshot_hash, app_reader_transform(dataset)
        ),
        snapshot_sha256=snapshot_hash,
    )


def save_candidate_bundle(directory, model, config, inputs, selection, dataset_hash):
    """Export a trained candidate with provenance for exactly its input snapshot."""
    return save_bundle(
        directory, model, config, inputs.data, selection, dataset_hash,
        retraining=inputs.provenance,
    )
