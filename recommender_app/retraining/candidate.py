from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recommender_app.ml.bundle import save_bundle
from recommender_app.ml.features import Examples, FeatureData
from .dataset import app_training_rows
from .status import bundle_retraining_metadata


@dataclass
class CandidateInputs:
    """Training inputs derived from one immutable app-rating snapshot."""

    data: FeatureData
    train: Examples
    ratings: pd.DataFrame
    provenance: dict


def prepare_candidate_inputs(dataset: Path, snapshot: Path) -> CandidateInputs:
    """Build training data and provenance from the same exported rating rows."""
    ratings = pd.read_parquet(snapshot)
    if "work_id" not in ratings.columns:
        raise ValueError(
            "This rating snapshot predates work-level provenance. Re-export it before "
            "preparing a candidate bundle."
        )
    app_rows = app_training_rows(dataset, ratings)
    data = FeatureData(dataset, extra_train=app_rows)
    return CandidateInputs(
        data=data,
        train=data.warm("train"),
        ratings=ratings,
        provenance=bundle_retraining_metadata(ratings),
    )


def save_candidate_bundle(directory, model, config, inputs, selection, dataset_hash):
    """Export a trained candidate with provenance for exactly its input snapshot."""
    return save_bundle(
        directory, model, config, inputs.data, selection, dataset_hash,
        retraining=inputs.provenance,
    )
