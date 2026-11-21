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
    year TEXT NOT NULL DEFAULT '',
    work_id TEXT NOT NULL DEFAULT '',
    series_key TEXT NOT NULL DEFAULT '',
    series_position INTEGER,
    bundle INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS books_title ON books(title_key);
CREATE INDEX IF NOT EXISTS books_author ON books(author_key);
CREATE UNIQUE INDEX IF NOT EXISTS manual_book_identity ON books(title_key, author_key)
    WHERE goodreads_id IS NULL;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS library (
    user_id INTEGER REFERENCES users(id),
    book_id INTEGER NOT NULL REFERENCES books(id),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    shelf TEXT,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, book_id)
);
CREATE TABLE IF NOT EXISTS recommendation_pages (
    user_id INTEGER NOT NULL REFERENCES users(id),
    browsing_id TEXT NOT NULL,
    page INTEGER NOT NULL,
    predictions TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, browsing_id, page)
);
CREATE TABLE IF NOT EXISTS dismissed_recommendations (
    user_id INTEGER NOT NULL REFERENCES users(id),
    book_id INTEGER NOT NULL REFERENCES books(id),
    PRIMARY KEY (user_id, book_id)
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
    user_id INTEGER REFERENCES users(id),
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
        columns = lambda table: {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
        old_library = columns("library")
        if "user_id" not in old_library:
            db.execute("ALTER TABLE library RENAME TO legacy_library")
            db.execute("""
                CREATE TABLE library (
                    user_id INTEGER REFERENCES users(id),
                    book_id INTEGER NOT NULL REFERENCES books(id),
                    rating INTEGER CHECK (rating BETWEEN 1 AND 5), shelf TEXT,
                    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, book_id)
                )
            """)
            shelf = "shelf" if "shelf" in old_library else "NULL"
            db.execute(f"""INSERT INTO library (book_id, rating, shelf, added_at, updated_at)
                           SELECT book_id, rating, {shelf}, added_at, updated_at FROM legacy_library""")
            db.execute("DROP TABLE legacy_library")
        if "user_id" not in columns("recommendation_feedback"):
            db.execute("ALTER TABLE recommendation_feedback ADD COLUMN user_id INTEGER REFERENCES users(id)")
        for name, definition in (
            ("work_id", "TEXT NOT NULL DEFAULT ''"), ("series_key", "TEXT NOT NULL DEFAULT ''"),
            ("series_position", "INTEGER"), ("bundle", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns("books"):
                db.execute(f"ALTER TABLE books ADD COLUMN {name} {definition}")
        db.execute("CREATE INDEX IF NOT EXISTS books_work ON books(work_id)")


def search_books(db, user_id, query):
    key = search_key(query)
    literal = key.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return db.execute("""
        SELECT b.*, l.rating, l.shelf, l.book_id AS saved_id
        FROM books b LEFT JOIN library l ON l.user_id = ? AND l.book_id = (
            SELECT saved.id FROM books saved JOIN library mine ON mine.book_id = saved.id
            WHERE mine.user_id = ?
                AND (saved.id = b.id OR (b.work_id != '' AND saved.work_id = b.work_id))
            ORDER BY saved.id LIMIT 1
        )
        WHERE (b.goodreads_id IS NOT NULL OR l.book_id IS NOT NULL)
          AND (b.title_key LIKE ? ESCAPE '\\' OR b.author_key LIKE ? ESCAPE '\\')
        ORDER BY CASE WHEN b.title_key = ? THEN 0 WHEN b.title_key LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END,
                 b.title_key, b.id
        LIMIT 30
    """, (user_id, user_id, f"%{literal}%", f"%{literal}%", key, f"{literal}%")).fetchall()


def library_books(db, user_id):
    return db.execute("""
        SELECT b.*, l.rating, l.shelf, l.book_id AS saved_id
        FROM library l JOIN books b ON b.id = l.book_id
        WHERE l.user_id = ?
        ORDER BY l.updated_at DESC, b.title_key, b.id
    """, (user_id,)).fetchall()


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


def library_edition(db, user_id, book_id):
    match = db.execute("""
        SELECT saved.id FROM books chosen
        JOIN books saved ON (saved.id = chosen.id OR (chosen.work_id != '' AND saved.work_id = chosen.work_id))
        JOIN library l ON l.book_id = saved.id AND l.user_id = ?
        WHERE chosen.id = ? ORDER BY (saved.id = chosen.id) DESC, saved.id LIMIT 1
    """, (user_id, book_id)).fetchone()
    return match["id"] if match else book_id


def save_book(db, user_id, book_id, rating, *, update):
    book_id = library_edition(db, user_id, book_id)
    db.execute("INSERT INTO library (user_id, book_id, rating) VALUES (?, ?, ?) ON CONFLICT(user_id, book_id) DO NOTHING",
               (user_id, book_id, rating))
    if update:
        db.execute("UPDATE library SET rating = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ? AND book_id = ?",
                   (rating, user_id, book_id))


def save_to_read(db, user_id, book_id):
    book_id = library_edition(db, user_id, book_id)
    db.execute("""
        INSERT INTO library (user_id, book_id, shelf) VALUES (?, ?, 'to-read')
        ON CONFLICT(user_id, book_id) DO NOTHING
    """, (user_id, book_id))


def dismiss_book(db, user_id, book_id):
    db.execute("INSERT OR IGNORE INTO dismissed_recommendations (user_id, book_id) VALUES (?, ?)",
               (user_id, book_id))


def dismissed_books(db, user_id):
    return db.execute("""
        SELECT b.* FROM dismissed_recommendations d JOIN books b ON b.id = d.book_id
        WHERE d.user_id = ?
    """, (user_id,)).fetchall()


def books_by_goodreads_id(db, ids):
    result = {}
    for start in range(0, len(ids), 500):
        batch = ids[start:start + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = db.execute(f"SELECT * FROM books WHERE goodreads_id IN ({placeholders})", batch)
        result.update((book["goodreads_id"], book) for book in rows)
    return result


def distinct_rating_count(books):
    return len({book["work_id"] or f"local:{book['id']}"
                for book in books if book["rating"] is not None})
