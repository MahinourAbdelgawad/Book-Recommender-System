from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd

from recommender_app.ml.book_features import BookFeatureBuilder


def expanded_catalog(database, dataset, artifacts, ratings, active_ids, minimum_readers):
    if minimum_readers < 1:
        raise ValueError("New books need at least one distinct reader")
    if artifacts is None:
        return active_ids, {}
    builder = BookFeatureBuilder(artifacts, dataset)
    base_ids = set(pd.read_parquet(dataset, columns=["book_id"]).book_id.astype(str))
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        books = {row["goodreads_id"]: row for row in db.execute(
            "SELECT * FROM books WHERE goodreads_id IS NOT NULL"
        )}
    active_works = {books[key]["work_id"] for key in active_ids if key in books
                    and books[key]["work_id"]}
    selected_works = set(active_works)
    counts = ratings.groupby("book_id").app_user_id.nunique()
    vectors = {}
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], str(item[0]))):
        key = str(key)
        book = books.get(key)
        if not book or key in base_ids or not key.isascii() or not key.isdecimal():
            continue
        if str(int(key)) != key or int(key) > 2**63 - 1:
            continue
        work = book["work_id"]
        if key not in active_ids and (count < minimum_readers or work in selected_works):
            continue
        year = str(book["year"]).strip()
        valid_year = year.isascii() and year.isdecimal() and 1 <= int(year) <= 9999
        # Center unknown scaled fields at their training mean. Missing genres and
        # count metadata use zero, with an explicit flag for missing pages.
        metadata = {name: 0 for name in builder.spec["columns"]}
        for name, (mean, _) in builder.spec["scaler"].items():
            if name not in builder.spec["log_columns"]:
                metadata[name] = mean
        if "num_pages" in builder.spec["scaler"]:
            metadata["num_pages"] = max(0, math.expm1(builder.spec["scaler"]["num_pages"][0]))
        if valid_year:
            metadata["publication_year"] = int(year)
        metadata.update(description=book["description"].strip(),
                        pages_missing=1, year_missing=int(not valid_year), n_authors=1,
                        in_series=int(bool(book["series_key"])))
        vectors[key] = json.dumps(builder.transform(metadata).to_dict(), allow_nan=False)
        if work:
            selected_works.add(work)
    return (active_ids & base_ids) | vectors.keys(), vectors
