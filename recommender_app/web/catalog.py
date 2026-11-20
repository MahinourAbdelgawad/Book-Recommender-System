"""Build a small searchable catalog by streaming the local Goodreads files."""

from contextlib import closing
import json
from pathlib import Path
import re

import numpy as np

from .library import open_database, search_key

BOOK_ID = re.compile(rb'"book_id"\s*:\s*"([0-9]+)"')


def seed_catalog(database: Path, bundle: Path, raw: Path):
    with np.load(bundle / "catalog.npz", allow_pickle=False) as catalog:
        ids = {str(book).encode() for book in catalog["book_ids"][catalog["catalog_indices"]]}
    with (raw / "goodreads_book_authors.json").open("rb") as source:
        authors = {row["author_id"]: row["name"] for row in map(json.loads, source)}
    pending = set(ids)
    records = []
    with (raw / "goodreads_books.json").open("rb") as source:
        for line in source:
            found = BOOK_ID.search(line)
            if not found or found[1] not in pending:
                continue
            book = json.loads(line)
            title = book["title"].strip()
            names = [authors.get(entry["author_id"], "") for entry in book["authors"]]
            author = ", ".join(dict.fromkeys(name for name in names if name))
            cover = book.get("image_url", "")
            if not cover.startswith("https://"):
                cover = ""
            records.append((book["book_id"], title, author, search_key(title), search_key(author),
                            cover, book.get("description", ""), book.get("isbn", ""),
                            book.get("isbn13", ""), book.get("publication_year", "")))
            pending.remove(found[1])
            if not pending:
                break
    if not records:
        raise ValueError("No model catalog books were found in the Goodreads metadata")
    with closing(open_database(database)) as db, db:
        db.executemany("""
            INSERT INTO books (goodreads_id, title, author, title_key, author_key, cover_url,
                               description, isbn, isbn13, year)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(goodreads_id) DO UPDATE SET title = excluded.title, author = excluded.author,
                title_key = excluded.title_key, author_key = excluded.author_key,
                cover_url = excluded.cover_url, description = excluded.description,
                isbn = excluded.isbn, isbn13 = excluded.isbn13, year = excluded.year
        """, records)
    return len(records), len(pending)
