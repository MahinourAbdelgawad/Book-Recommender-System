from pathlib import Path

import pandas as pd

from recommender_app.ml.bundle import fingerprint

ORIGINAL_DATASET = (Path(__file__).resolve().parents[2]
                    / "preparation-and-design/dataset/cleaned_dataset.parquet")


def comparison_dataset(dataset, manifest, original_dataset=ORIGINAL_DATASET):
    if fingerprint(dataset) == manifest["dataset_sha256"]:
        return dataset
    if manifest["version"] != "v001" or fingerprint(original_dataset) != manifest["dataset_sha256"]:
        raise ValueError("Candidate dataset differs from the active model's base dataset")
    columns = ["user_id", "book_id", "rating", "split"]
    original = pd.read_parquet(original_dataset, columns=columns)
    candidate = pd.read_parquet(dataset, columns=columns)
    if not original.equals(candidate):
        raise ValueError("Expansion export changed the original interactions or split order")
    return original_dataset
