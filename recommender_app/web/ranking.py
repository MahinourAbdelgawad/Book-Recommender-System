import re

from . import model_client
from .library import search_key

PAGE_SIZE = 10

VOLUME = re.compile(r"\((?P<name>[^()]+?),?\s+#(?P<number>\d+)(?P<suffix>[^()]*)\)", re.I)
BUNDLE = re.compile(r"\bbox\s*set\b|\bboxed set\b|\bomnibus\b|\bthe complete\b", re.I)
RANGE = re.compile(r"^\s*[-–—,&+]\s*#?\d")


def catalog_series(title, series, work_id):
    match = VOLUME.search(title)
    bundle = bool(BUNDLE.search(title) or (match and RANGE.match(match["suffix"])))
    position = int(match["number"]) if match and not match["suffix"].strip() else None
    key = f"series:{series[0]}" if series else ""
    if match and not key:
        key = f"title:{search_key(match['name'].rstrip(','))}"
    if not key and "calvin and hobbes" in title.casefold():
        key = "title:calvin and hobbes"
        if re.sub(r"[^a-z ]", "", title.casefold()).strip() == "calvin and hobbes":
            position = 1
    return str(work_id or ""), key, position, int(bundle)


def work_key(book):
    return book["work_id"] or f"local:{book['id']}"


def select_books(db, api_url, api_key, books, seen, dismissed, limit=PAGE_SIZE, reader_id=None):
    known = {book["goodreads_id"]: book for book in db.execute("""
        SELECT id, goodreads_id, work_id, series_key, series_position, bundle FROM books
        WHERE goodreads_id IS NOT NULL
    """)}
    blocked_works = {work_key(book) for book in books}
    blocked_works.update(work_key(book) for book in dismissed)
    blocked_works.update(work_key(book) for book in seen)
    blocked_series = {book["series_key"] for book in seen if book["series_key"]}
    excluded = {book_id for book_id, book in known.items()
                if book["bundle"] or work_key(book) in blocked_works
                or (book["series_key"] and (book["series_position"] != 1
                                            or book["series_key"] in blocked_series))}
    selected = []
    version = None
    while len(selected) < limit:
        version, candidates = model_client.recommend(
            api_url, api_key, books, 100, excluded, reader_id=reader_id
        )
        fresh = [item for item in candidates if item.book_id not in excluded]
        for item in fresh:
            excluded.add(item.book_id)
            book = known.get(item.book_id)
            if not book or work_key(book) in blocked_works or (
                book["series_key"] and book["series_key"] in blocked_series
            ):
                continue
            selected.append(item)
            blocked_works.add(work_key(book))
            if book["series_key"]:
                blocked_series.add(book["series_key"])
            if len(selected) == limit:
                break
        if len(candidates) < 100 or not fresh:
            break
    return version, selected
