import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "preparation-and-design/dataset/cleaned_dataset.parquet"
DEFAULT_SNAPSHOT = ROOT / "instance/ml/retraining-ratings.parquet"
DEFAULT_OUTPUT = ROOT / "instance/ml/app-training-rows.parquet"
APP_USER_PREFIX = "app_user_"


def app_reader_transform(dataset):
    """Describe how app account IDs are encoded beyond the base reader IDs."""
    base = pd.read_parquet(dataset, columns=["user_id"])
    if pd.api.types.is_numeric_dtype(base["user_id"].dtype):
        return {"mode": "numeric_offset", "offset": int(base["user_id"].max()) + 1}
    return {"mode": "prefix", "prefix": APP_USER_PREFIX}


def app_training_rows(dataset, ratings, extra_book_ids=()):
    """Return rows with dataset-compatible IDs and a train split.

    App account IDs are mapped outside the original reader range. Books must
    exist in the base dataset or have static features supplied by the snapshot.
    """
    base = pd.read_parquet(dataset, columns=["user_id", "book_id"])
    if ratings.empty:
        return pd.DataFrame(columns=["user_id", "book_id", "rating", "split"])

    numeric_users = pd.api.types.is_numeric_dtype(base["user_id"].dtype)
    numeric_books = pd.api.types.is_numeric_dtype(base["book_id"].dtype)

    if numeric_users:
        app_ids = pd.to_numeric(ratings["app_user_id"], errors="raise").astype("int64")
        if (app_ids < 0).any():
            raise ValueError("App account IDs must be non-negative")
        users = int(base["user_id"].max()) + 1 + app_ids
    else:
        users = APP_USER_PREFIX + ratings["app_user_id"].astype(str)

    if numeric_books:
        book_ids = pd.to_numeric(ratings["book_id"], errors="raise").astype("int64")
    else:
        book_ids = ratings["book_id"].astype(str)

    known_books = set(base["book_id"].drop_duplicates())
    unknown = sorted(set(book_ids) - known_books - set(extra_book_ids), key=str)
    if unknown:
        raise ValueError(
            f"{len(unknown)} snapshot books are absent from the engineered dataset. "
            "Only books represented by the original model features can be added in this step."
        )

    collisions = set(users) & set(base["user_id"].drop_duplicates())
    if collisions:
        raise ValueError("Generated app reader IDs collide with original dataset reader IDs")

    return pd.DataFrame({
        "user_id": users.to_numpy(),
        "book_id": book_ids.to_numpy(),
        "rating": pd.to_numeric(ratings["rating"], errors="raise").astype("float32").to_numpy(),
        "split": "train",
    })


def prepare_rows(dataset, snapshot, output):
    ratings = pd.read_parquet(snapshot)
    rows = app_training_rows(dataset, ratings)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(output, index=False)
    return {
        "output": output,
        "app_training_rows": len(rows),
        "app_readers": int(rows["user_id"].nunique()) if len(rows) else 0,
        "books": int(rows["book_id"].nunique()) if len(rows) else 0,
        "split": "train",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = prepare_rows(args.dataset, args.snapshot, args.output)
    print(f"App training rows written to {summary['output']}")
    print(f"Rows: {summary['app_training_rows']}")
    print(f"App readers: {summary['app_readers']}")
    print(f"Books: {summary['books']}")
    print(f"Split: {summary['split']}")


if __name__ == "__main__":
    main()
