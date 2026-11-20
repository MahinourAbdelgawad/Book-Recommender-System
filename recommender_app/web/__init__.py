"""Flask application for building a local reading library."""

import hmac
import os
from pathlib import Path
import secrets

import click
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for

from .library import add_manual_book, initialize, library_books, open_database, save_book, search_books

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
    )
    app.config.update(config or {})
    initialize(app.config["DATABASE"])

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

    return app
