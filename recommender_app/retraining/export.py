import argparse
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from recommender_app.ml.bundle import load_bundle
from recommender_app.web.library import open_database

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = ROOT / "instance/library.db"
DEFAULT_BUNDLE = ROOT / "instance/ml/v001/bundle"
DEFAULT_OUTPUT = ROOT / "instance/ml/retraining-ratings.parquet"
MIN_READER_RATINGS = 5

RATING_QUERY = """
SELECT
    library.user_id AS app_user_id,
    books.goodreads_id AS book_id,
    COALESCE(NULLIF(books.work_id, ''), 'book:' || books.goodreads_id) AS work_id,
    library.rating AS rating,
    library.updated_at AS updated_at
FROM library
JOIN books ON books.id = library.book_id
WHERE library.user_id IS NOT NULL
  AND library.rating BETWEEN 1 AND 5
  AND books.goodreads_id IS NOT NULL
"""


def catalog_book_ids(bundle_directory):
    with np.load(bundle_directory / "catalog.npz", allow_pickle=False) as catalog:
        return {str(value) for value in catalog["book_ids"][catalog["catalog_indices"]]}


def read_library_ratings(database):
    with closing(open_database(database)) as connection:
        return pd.read_sql_query(RATING_QUERY, connection)


def eligible_ratings(ratings, catalog_ids, minimum_per_reader=MIN_READER_RATINGS):
    """Keep catalog ratings from sufficiently active readers, once per Goodreads work."""
    if minimum_per_reader < 1:
        raise ValueError("minimum_per_reader must be at least one")

    eligible = ratings[ratings["book_id"].astype(str).isin(catalog_ids)].copy()
    eligible = eligible.sort_values(["updated_at", "book_id"])
    eligible = eligible.drop_duplicates(["app_user_id", "work_id"], keep="last")
    counts = eligible.groupby("app_user_id")["rating"].transform("size")
    return eligible[counts >= minimum_per_reader].reset_index(drop=True)


def export_snapshot(database, bundle_directory, output, minimum_per_reader=MIN_READER_RATINGS):
    load_bundle(bundle_directory)
    ratings = read_library_ratings(database)
    catalog_ids = catalog_book_ids(bundle_directory)
    eligible = eligible_ratings(ratings, catalog_ids, minimum_per_reader)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = eligible[["app_user_id", "book_id", "work_id", "rating", "updated_at"]].copy()
    snapshot["app_user_id"] = snapshot["app_user_id"].astype(str)
    snapshot["book_id"] = snapshot["book_id"].astype(str)
    snapshot.to_parquet(output, index=False)

    catalog_ratings = int(ratings["book_id"].astype(str).isin(catalog_ids).sum())
    return {
        "output": output,
        "explicit_ratings": len(ratings),
        "catalog_ratings_before_work_deduplication": catalog_ratings,
        "eligible_ratings": len(eligible),
        "eligible_readers": int(eligible["app_user_id"].nunique()),
        "minimum_ratings_per_reader": minimum_per_reader,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-reader-ratings", type=int, default=MIN_READER_RATINGS)
    args = parser.parse_args()

    summary = export_snapshot(
        args.database, args.bundle, args.output, args.minimum_reader_ratings
    )
    print(f"Rating snapshot written to {summary['output']}")
    print(f"Explicit app ratings: {summary['explicit_ratings']}")
    print(f"Catalog ratings before edition grouping: "
          f"{summary['catalog_ratings_before_work_deduplication']}")
    print(f"Eligible ratings: {summary['eligible_ratings']}")
    print(f"Eligible readers: {summary['eligible_readers']}")
    print(f"Minimum ratings per reader: {summary['minimum_ratings_per_reader']}")


if __name__ == "__main__":
    main()
