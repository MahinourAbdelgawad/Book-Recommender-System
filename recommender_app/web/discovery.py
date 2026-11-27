import json

import numpy as np

GENRES = {
    "fiction": "Fiction", "fantasy": "Fantasy", "mystery": "Mystery / Thriller",
    "young_adult": "Young Adult", "romance": "Romance", "history": "History",
    "nonfiction": "Nonfiction", "poetry": "Poetry", "children": "Children's",
    "comics": "Comics", "science_fiction": "Science Fiction",
}


def catalog_facets(bundle):
    manifest = bundle / "manifest.json"
    catalog_file = bundle / "catalog.npz"
    if not manifest.is_file() or not catalog_file.is_file():
        return {}
    columns = json.loads(manifest.read_text())["feature_names"]
    genres = [(key, columns.index(f"p_{key}")) for key in GENRES if f"p_{key}" in columns]
    popularity = columns.index("book_ratings_count") if "book_ratings_count" in columns else None
    with np.load(catalog_file, allow_pickle=False) as catalog:
        ids, features = catalog["book_ids"], catalog["book_features"]
        return {
            str(ids[i]): (float(features[i, popularity]) if popularity is not None else 0,
                          frozenset(key for key, col in genres if features[i, col] > 0))
            for i in catalog["catalog_indices"]
        }


def browse_rows(db, facets, work_key):
    ranked = sorted(facets, key=lambda key: (-facets[key][0], key))
    books = {row["goodreads_id"]: row for row in db.execute(
        "SELECT * FROM books WHERE goodreads_id IS NOT NULL AND bundle = 0"
    )}
    rows = []
    for genre in ("", "science_fiction", "fantasy", "young_adult", "mystery", "romance"):
        items, seen = [], set()
        for key in ranked:
            book = books.get(key)
            if book is None or (genre and genre not in facets[key][1]):
                continue
            work = work_key(book)
            if work in seen:
                continue
            seen.add(work)
            items.append(book)
            if len(items) == 16:
                break
        if items:
            rows.append((GENRES.get(genre, "See what’s popular"), items))
    return rows
