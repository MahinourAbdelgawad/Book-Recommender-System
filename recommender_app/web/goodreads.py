"""Parse Goodreads exports, match editions, and save confirmed library rows."""

import csv
import io
import re

from .library import search_key

MAX_ROWS = 5000
REQUIRED_COLUMNS = {"Title", "Author", "My Rating", "Exclusive Shelf"}
ISBN_WRAPPER = re.compile(r'^="([^"]*)"$')


def isbn(value):
    match = ISBN_WRAPPER.fullmatch(value)
    return (match[1] if match else value).replace("-", "").replace(" ", "").upper()


def parse_export(contents):
    try:
        text = contents.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if not REQUIRED_COLUMNS.issubset(reader.fieldnames or ()):
            raise ValueError("Choose a Goodreads library export with title, author, rating, and shelf columns.")
        rows = []
        for raw in reader:
            if len(rows) == MAX_ROWS:
                raise ValueError(f"Import files with at most {MAX_ROWS:,} books at a time.")

            def field(name):
                return (raw.get(name) or "").strip()

            if None in raw or any(raw.get(name) is None for name in REQUIRED_COLUMNS):
                raise ValueError(f"CSV row ending at line {reader.line_num} has the wrong number of columns.")
            title, author = field("Title"), field("Author")
            if not title or len(title) > 1000 or len(author) > 1000:
                raise ValueError(f"Check the title and author near line {reader.line_num}.")
            rating = field("My Rating")
            if rating and not re.fullmatch(r"[0-5](?:\.0+)?", rating):
                raise ValueError(f"Invalid star rating near line {reader.line_num}.")
            goodreads_id = field("Book Id")
            if goodreads_id and not re.fullmatch(r"[0-9]{1,20}", goodreads_id):
                raise ValueError(f"Invalid Goodreads book ID near line {reader.line_num}.")
            rows.append(dict(title=title, author=author, goodreads_id=goodreads_id or None,
                             rating=int(rating[0]) if rating and rating[0] != "0" else None,
                             shelf=field("Exclusive Shelf")[:100] or None,
                             isbn=isbn(field("ISBN"))[:20], isbn13=isbn(field("ISBN13"))[:20],
                             year=field("Year Published")[:8]))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError("Could not read this file as a UTF-8 Goodreads CSV.") from error
    if not rows:
        raise ValueError("This export contains no books.")
    return rows


def match_books(db, row):
    identity = (search_key(row["title"]), search_key(row["author"]))
    if row["goodreads_id"]:
        exact = db.execute("SELECT * FROM books WHERE goodreads_id = ?", (row["goodreads_id"],)).fetchall()
        if exact:
            return exact, "id"
    for column in ("isbn13", "isbn"):
        if row[column]:
            matches = db.execute(f"SELECT * FROM books WHERE {column} = ? ORDER BY id",
                                 (row[column],)).fetchall()
            if matches:
                return matches, "isbn"
    matches = db.execute("""
        SELECT * FROM books WHERE title_key = ? AND author_key = ? ORDER BY id
    """, identity).fetchall()
    return matches, "title"


def add_imported_book(db, row):
    identity = (search_key(row["title"]), search_key(row["author"]))
    db.execute("""
        INSERT INTO books (goodreads_id, title, author, title_key, author_key, isbn, isbn13, year)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING
    """, (row["goodreads_id"], row["title"], row["author"], *identity,
          row["isbn"], row["isbn13"], row["year"]))
    if row["goodreads_id"]:
        return db.execute("SELECT id FROM books WHERE goodreads_id = ?", (row["goodreads_id"],)).fetchone()["id"]
    return db.execute("""SELECT id FROM books WHERE goodreads_id IS NULL AND title_key = ? AND author_key = ?""",
                      identity).fetchone()["id"]


def save_import_row(db, book_id, row):
    db.execute("""
        INSERT INTO library (book_id, rating, shelf) VALUES (?, ?, ?)
        ON CONFLICT(book_id) DO UPDATE SET
            rating = COALESCE(library.rating, excluded.rating),
            shelf = COALESCE(library.shelf, excluded.shelf)
    """, (book_id, row["rating"], row["shelf"]))
