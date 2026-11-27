import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from recommender_app.ml.active_bundle import active_bundle_path

from .export import DEFAULT_DATABASE, catalog_book_ids


def preview(database, bundle, minimum_readers):
    if minimum_readers < 1:
        raise ValueError("Minimum readers must be positive")
    catalog = catalog_book_ids(bundle)
    uri = Path(database).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as db:
        db.row_factory = sqlite3.Row
        books = db.execute("SELECT id, goodreads_id, work_id FROM books").fetchall()
        rated = db.execute("""
            SELECT b.*, l.user_id
            FROM books AS b JOIN library AS l ON l.book_id = b.id
            WHERE l.user_id IS NOT NULL AND l.rating IN (1, 2, 3, 4, 5)
            ORDER BY b.id
        """).fetchall()

    def work_key(book):
        if book["work_id"]:
            return ("work", book["work_id"])
        if book["goodreads_id"]:
            return ("goodreads", book["goodreads_id"])
        return ("local", book["id"])

    active_works = {work_key(book) for book in books if book["goodreads_id"] in catalog}
    groups = {}
    for row in rated:
        key = work_key(row)
        if key not in active_works:
            editions, readers = groups.setdefault(key, ({}, set()))
            editions[row["id"]] = row
            readers.add(row["user_id"])

    works = []
    for editions, readers in groups.values():
        def missing(book):
            fields = ("goodreads_id", "title", "author", "description", "year")
            return [field for field in fields if not str(book[field] or "").strip()]

        best = min(editions.values(), key=lambda book: (
            not bool(book["goodreads_id"]), len(missing(book)), book["id"],
        ))
        blockers = missing(best)
        works.append({
            "title": best["title"],
            "goodreads_id": best["goodreads_id"],
            "rated_editions": len(editions),
            "distinct_readers": len(readers),
            "meets_reader_minimum": len(readers) >= minimum_readers,
            "missing_metadata": blockers,
            "passes_initial_screen": (
                len(readers) >= minimum_readers and bool(best["goodreads_id"])
                and best["goodreads_id"].isascii() and best["goodreads_id"].isdecimal()
            ),
        })
    works.sort(key=lambda work: (-work["distinct_readers"], work["title"]))
    return {
        "minimum_distinct_readers": minimum_readers,
        "rated_works_outside_catalog": len(works),
        "works_passing_initial_screen": sum(work["passes_initial_screen"] for work in works),
        "reader_work_ratings_passing_initial_screen": sum(
            work["distinct_readers"] for work in works if work["passes_initial_screen"]
        ),
        "works": works,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--minimum-readers", type=int, required=True)
    args = parser.parse_args()
    if args.minimum_readers < 1:
        parser.error("--minimum-readers must be positive")
    print(json.dumps(preview(args.database, args.bundle or active_bundle_path(),
                             args.minimum_readers), indent=2))


if __name__ == "__main__":
    main()
