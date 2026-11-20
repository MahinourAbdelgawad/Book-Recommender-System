"""SQLite storage for the local library and searchable book catalog."""

from contextlib import closing
import sqlite3
from unicodedata import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id INTEGER PRIMARY KEY,
    goodreads_id TEXT UNIQUE,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    title_key TEXT NOT NULL,
    author_key TEXT NOT NULL,
    cover_url TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    isbn TEXT NOT NULL DEFAULT '',
    isbn13 TEXT NOT NULL DEFAULT '',
    year TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS books_title ON books(title_key);
CREATE INDEX IF NOT EXISTS books_author ON books(author_key);
CREATE UNIQUE INDEX IF NOT EXISTS manual_book_identity ON books(title_key, author_key)
    WHERE goodreads_id IS NULL;
CREATE TABLE IF NOT EXISTS library (
    book_id INTEGER PRIMARY KEY REFERENCES books(id),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    shelf TEXT,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS books_isbn ON books(isbn);
CREATE INDEX IF NOT EXISTS books_isbn13 ON books(isbn13);
CREATE TABLE IF NOT EXISTS import_drafts (
    id TEXT PRIMARY KEY,
    rows_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    event_id TEXT PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id),
    model_version TEXT NOT NULL,
    predicted_rating REAL NOT NULL CHECK (predicted_rating BETWEEN 1 AND 5),
    actual_rating INTEGER NOT NULL CHECK (actual_rating BETWEEN 1 AND 5),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def search_key(value):
    return " ".join(normalize("NFKC", value).casefold().split())


def open_database(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def initialize(path):
    with closing(open_database(path)) as db, db:
        db.executescript(SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        if "shelf" not in {row["name"] for row in db.execute("PRAGMA table_info(library)")}:
            db.execute("ALTER TABLE library ADD COLUMN shelf TEXT")


def search_books(db, query):
    key = search_key(query)
    literal = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return db.execute("""
        SELECT b.*, l.rating, l.shelf, l.book_id AS saved_id
        FROM books b LEFT JOIN library l ON l.book_id = b.id
        WHERE b.title_key LIKE ? ESCAPE '\\' OR b.author_key LIKE ? ESCAPE '\\'
        ORDER BY CASE WHEN b.title_key = ? THEN 0 WHEN b.title_key LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END,
                 b.title_key, b.id
        LIMIT 30
    """, (f"%{literal}%", f"%{literal}%", key, f"{literal}%")).fetchall()


def library_books(db):
    return db.execute("""
        SELECT b.*, l.rating, l.shelf, l.book_id AS saved_id
        FROM library l JOIN books b ON b.id = l.book_id
        ORDER BY l.updated_at DESC, b.title_key, b.id
    """).fetchall()


def add_manual_book(db, title, author):
    identity = (search_key(title), search_key(author))
    existing = db.execute("""
        SELECT id FROM books WHERE title_key = ? AND author_key = ?
        ORDER BY goodreads_id IS NULL, id LIMIT 2
    """, identity).fetchall()
    if len(existing) == 1:
        return existing[0]["id"]
    db.execute("""
        INSERT INTO books (title, author, title_key, author_key) VALUES (?, ?, ?, ?)
        ON CONFLICT(title_key, author_key) WHERE goodreads_id IS NULL DO NOTHING
    """, (title, author, *identity))
    return db.execute("""
        SELECT id FROM books WHERE goodreads_id IS NULL AND title_key = ? AND author_key = ?
    """, identity).fetchone()["id"]


def save_book(db, book_id, rating, *, update):
    db.execute("INSERT INTO library (book_id, rating) VALUES (?, ?) ON CONFLICT(book_id) DO NOTHING",
               (book_id, rating))
    if update:
        db.execute("UPDATE library SET rating = ?, updated_at = CURRENT_TIMESTAMP WHERE book_id = ?",
                   (rating, book_id))


def save_feedback(db, event_id, book_id, version, prediction, rating):
    inserted = db.execute("""
        INSERT INTO recommendation_feedback (event_id, book_id, model_version, predicted_rating, actual_rating)
        VALUES (?, ?, ?, ?, ?) ON CONFLICT(event_id) DO NOTHING
    """, (event_id, book_id, version, prediction, rating))
    if inserted.rowcount:
        save_book(db, book_id, rating, update=True)
        db.execute("UPDATE library SET shelf = 'read' WHERE book_id = ?", (book_id,))


def books_by_goodreads_id(db, ids):
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = db.execute(f"SELECT * FROM books WHERE goodreads_id IN ({placeholders})", ids)
    return {book["goodreads_id"]: book for book in rows}
