import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from recommender_app.ml.features import FeatureData

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "preparation-and-design/dataset/cleaned_dataset.parquet"
DEFAULT_ROWS = ROOT / "instance/ml/app-training-rows.parquet"


def inspect(dataset, rows_path):
    original = FeatureData(dataset)
    app_rows = pd.read_parquet(rows_path)
    augmented = FeatureData(dataset, extra_train=app_rows)

    # Validation and test assignments must remain the same original examples.
    for split in ("val", "test"):
        before = original.warm_rows[split]
        after = augmented.warm_rows[split]
        if not np.array_equal(before, after):
            raise ValueError(f"Original warm {split} rows changed after adding app ratings")
        if not np.array_equal(original.ratings[before], augmented.ratings[after]):
            raise ValueError(f"Original warm {split} labels changed after adding app ratings")

    for split in ("val", "test"):
        if not np.array_equal(original.cold_users[split], augmented.cold_users[split]):
            raise ValueError(f"Original cold {split} reader cohort changed")

    train_before = len(original.warm_rows["train"])
    train_after = len(augmented.warm_rows["train"])
    if train_after - train_before != len(app_rows):
        raise ValueError("Not all app rows were added to the warm training split")

    print("Feature preview completed; no model was trained.")
    print(f"App rows added to training: {len(app_rows)}")
    print(f"Warm training rows: {train_before} -> {train_after}")
    print(f"Warm validation rows retained: {len(augmented.warm_rows['val'])}")
    print(f"Warm test rows retained: {len(augmented.warm_rows['test'])}")
    print("Original cold validation and test reader cohorts: unchanged")
    print(f"Feature columns: {len(augmented.columns)}")
    print("Validation/test feature transforms are rebuilt from training data only.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    args = parser.parse_args()
    inspect(args.dataset, args.rows)


if __name__ == "__main__":
    main()
