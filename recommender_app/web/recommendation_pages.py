import json

from .model_client import Prediction


def load_page(db, user_id, browsing_id, page):
    row = db.execute("""
        SELECT predictions FROM recommendation_pages
        WHERE user_id = ? AND browsing_id = ? AND page = ?
    """, (user_id, browsing_id, page)).fetchone()
    return [Prediction(*item) for item in json.loads(row["predictions"])] if row else None


def last_page(db, user_id, browsing_id):
    return db.execute("""
        SELECT COALESCE(MAX(page), 0) FROM recommendation_pages
        WHERE user_id = ? AND browsing_id = ?
    """, (user_id, browsing_id)).fetchone()[0]


def save_page(db, user_id, browsing_id, page, ranked):
    db.execute("DELETE FROM recommendation_pages WHERE created_at < datetime('now', '-7 days')")
    db.execute("""
        INSERT OR IGNORE INTO recommendation_pages (user_id, browsing_id, page, predictions)
        VALUES (?, ?, ?, ?)
    """, (user_id, browsing_id, page, json.dumps([(item.book_id, item.rating) for item in ranked])))


def seen_ids(db, user_id, browsing_id, page):
    rows = db.execute("""
        SELECT predictions FROM recommendation_pages
        WHERE user_id = ? AND browsing_id = ? AND page < ?
    """, (user_id, browsing_id, page))
    return list(dict.fromkeys(book_id for row in rows for book_id, _ in json.loads(row["predictions"])))
