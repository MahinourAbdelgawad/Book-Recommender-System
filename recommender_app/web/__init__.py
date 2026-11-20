"""Flask application for building a local reading library."""

import hmac
import os
from pathlib import Path
import secrets
from urllib.parse import quote

import click
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for

from itsdangerous import BadSignature, URLSafeTimedSerializer

from . import model_client
from .imports import imports
from .library import (
    add_manual_book, books_by_goodreads_id, initialize, library_books, open_database,
    save_book, save_feedback, search_books,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "instance/ml/v001/bundle"
DEFAULT_RAW = ROOT / "preparation-and-design/dataset_original"
RATINGS_REQUIRED = 5


def create_app(config=None):
    app = Flask(__name__, instance_path=str(ROOT / "instance"))
    app.config.from_mapping(
        DATABASE=Path(os.environ.get("LIBRARY_DB", ROOT / "instance/library.db")),
        SECRET_KEY=os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_SAMESITE="Lax",
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        MODEL_API_URL=os.environ.get("MODEL_API_URL", "http://127.0.0.1:8000"),
        MODEL_API_KEY=os.environ.get("MODEL_API_KEY"),
    )
    app.config.update(config or {})
    initialize(app.config["DATABASE"])
    feedback_tokens = URLSafeTimedSerializer(app.secret_key, salt="recommendation-feedback")

    def database():
        if "db" not in g:
            g.db = open_database(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_database(_error):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.context_processor
    def form_token():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return {"csrf_token": session["csrf_token"]}

    @app.before_request
    def verify_form_token():
        if request.method == "POST":
            supplied = request.form.get("csrf_token", "").encode()
            expected = session.get("csrf_token", "").encode()
            if not expected or not hmac.compare_digest(supplied, expected):
                abort(400, "This form expired. Reload the library and try again.")

    def submitted_rating():
        value = request.form.get("rating", "")
        if value and value not in {"1", "2", "3", "4", "5"}:
            abort(400, "Choose a rating from 1 to 5 stars.")
        return int(value) if value else None

    @app.get("/")
    def library():
        query = request.args.get("q", "").strip()[:200]
        books = library_books(database())
        rated = sum(book["rating"] is not None for book in books)
        return render_template("library.html", books=books, query=query,
                               matches=search_books(database(), query) if query else [],
                               rated=rated, required=RATINGS_REQUIRED,
                               remaining=max(RATINGS_REQUIRED - rated, 0))

    @app.post("/library/books/<int:book_id>")
    def save_catalog_book(book_id):
        db = database()
        if db.execute("SELECT 1 FROM books WHERE id = ?", (book_id,)).fetchone() is None:
            abort(404)
        update = "rating" in request.form
        rating = submitted_rating()
        with db:
            save_book(db, book_id, rating, update=update)
        flash("Rating saved." if rating else "Library updated.")
        return redirect(url_for("library"), code=303)

    @app.post("/library/books/new")
    def save_manual_book():
        title = request.form.get("title", "").strip()
        author = request.form.get("author", "").strip()
        if not title or not author or max(len(title), len(author)) > 250:
            flash("Enter a title and author, up to 250 characters each.")
            return redirect(url_for("library", _anchor="manual"), code=303)
        rating = submitted_rating()
        db = database()
        with db:
            book_id = add_manual_book(db, title, author)
            save_book(db, book_id, rating, update=rating is not None)
        flash("Book saved to your library.")
        return redirect(url_for("library"), code=303)

    @app.get("/recommendations")
    def recommendations():
        db = database()
        books = library_books(db)
        remaining = max(RATINGS_REQUIRED - sum(book["rating"] is not None for book in books), 0)
        if remaining:
            return render_template("recommendations.html", remaining=remaining)

        try:
            version, ranked = model_client.recommend(
                app.config["MODEL_API_URL"], app.config["MODEL_API_KEY"], books)
        except model_client.ModelServiceError:
            app.logger.warning("Model service recommendation request failed", exc_info=True)
            return render_template("recommendations.html", unavailable=True), 503

        metadata = books_by_goodreads_id(db, [row.book_id for row in ranked])
        cards = []
        for row in ranked:
            book = metadata.get(row.book_id)
            if book is None:
                continue
            proof = feedback_tokens.dumps([secrets.token_urlsafe(16), book["id"], version, row.rating])
            cards.append(dict(book=book, predicted=row.rating, proof=proof,
                              goodreads_url=f"https://www.goodreads.com/book/show/{quote(row.book_id, safe='')}"))
        return render_template("recommendations.html", cards=cards,
                               missing_metadata=len(cards) < len(ranked))

    @app.post("/recommendations/feedback")
    def rate_recommendation():
        rating = submitted_rating()
        if rating is None:
            abort(400, "Choose a rating from 1 to 5 stars.")
        try:
            event_id, book_id, version, prediction = feedback_tokens.loads(
                request.form.get("proof", ""), max_age=86400)
        except BadSignature:
            flash("This recommendation expired. Open Recommendations again to rate it.")
            return redirect(url_for("recommendations"), code=303)

        db = database()
        with db:
            save_feedback(db, event_id, book_id, version, prediction, rating)
        flash("Rating saved to your library.")
        return redirect(url_for("library", _anchor=f"book-{book_id}"), code=303)

    @app.cli.command("seed-catalog")
    @click.option("--bundle", type=click.Path(path_type=Path, exists=True, file_okay=False),
                  default=DEFAULT_BUNDLE, show_default=True)
    @click.option("--raw", type=click.Path(path_type=Path, exists=True, file_okay=False),
                  default=DEFAULT_RAW, show_default=True)
    def seed(bundle, raw):
        """Load the model's book titles, authors and display metadata."""
        from .catalog import seed_catalog

        loaded, missing = seed_catalog(app.config["DATABASE"], bundle, raw)
        click.echo(f"Loaded {loaded:,} books. Missing Goodreads metadata for {missing:,} catalog IDs.")

    app.register_blueprint(imports)
    return app
