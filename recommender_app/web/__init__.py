import hmac
import os
from pathlib import Path
import secrets
from urllib.parse import quote

import click
from flask import Flask, abort, flash, g, redirect, render_template, request, session, url_for

from recommender_app.ml.active_bundle import active_bundle_path

from . import model_client
from . import recommendation_pages as pages
from .auth import auth
from .imports import imports
from .library import (
    add_manual_book, books_by_goodreads_id, distinct_rating_count, initialize, library_books, open_database,
    dismiss_book, dismissed_books, save_book, save_to_read, search_books,
)

from .ranking import PAGE_SIZE, select_books, work_key
from .discovery import GENRES, browse_rows, catalog_facets
from .library import manual_book_matches, search_key

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

    facets = catalog_facets(active_bundle_path())
    genres = {key: label for key, label in GENRES.items()
              if any(key in tags for _, tags in facets.values())}

    def book_genres(book):
        return facets.get(book["goodreads_id"], (0, frozenset()))[1]

    def selected_genre():
        genre = request.args.get("genre", "")
        return genre if genre in genres else ""

    def recommendation_scope(genre):
        return f'{session["browsing_id"]}:genre:{genre}' if genre else session["browsing_id"]

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
        if session.get("user_id") is None and request.endpoint not in {"auth.login", "auth.signup", "static"}:
            return redirect(url_for("auth.login"))
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
    def home():
        db = database()
        books = library_books(db, session["user_id"])
        rated = distinct_rating_count(books)
        rows = browse_rows(db, facets, work_key)
        if rated >= RATINGS_REQUIRED:
            try:
                _, picks = select_books(db, app.config["MODEL_API_URL"], app.config["MODEL_API_KEY"],
                                        books, (), dismissed_books(db, session["user_id"]),
                                        reader_id=f'app:{session["user_id"]}')
                metadata = books_by_goodreads_id(db, [pick.book_id for pick in picks])
                if metadata:
                    rows.insert(1, ("Picked for you", [metadata[pick.book_id] for pick in picks
                                                       if pick.book_id in metadata]))
            except model_client.ModelServiceError:
                app.logger.warning("Home recommendations unavailable", exc_info=True)
        return render_template("home.html", rows=rows,
                               remaining=max(RATINGS_REQUIRED - rated, 0))

    @app.get("/library")
    def library():
        query = request.args.get("q", "").strip()[:200]
        books = library_books(database(), session["user_id"])
        rated = distinct_rating_count(books)
        shelf = request.args.get("shelf", "")
        genre = selected_genre()
        sort = request.args.get("sort", "updated")
        shelves = sorted({book["shelf"] for book in books if book["shelf"]})
        visible = [book for book in books if (not shelf or book["shelf"] == shelf)
                   and (not genre or genre in book_genres(book))]
        if sort == "read":
            visible.sort(key=lambda book: book["date_read"] or "", reverse=True)
        elif sort == "added":
            visible.sort(key=lambda book: book["added_at"], reverse=True)
        elif sort == "title":
            visible.sort(key=lambda book: book["title_key"])
        return render_template("library.html", books=visible, query=query, shelf=shelf, shelves=shelves,
                               genres=genres, genre=genre, sort=sort,
                               matches=search_books(database(), session["user_id"], query) if query else [],
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
            save_book(db, session["user_id"], book_id, rating, update=update)
        flash("Rating saved." if rating else "Library updated.")
        return redirect(url_for("library"), code=303)

    @app.get("/library/book-suggestions")
    def book_suggestions():
        title = request.args.get("title", "").strip()[:250]
        matches = search_books(database(), session["user_id"], title)[:8] if len(title) >= 2 else []
        return {"books": [{key: book[key] for key in ("id", "title", "author", "year")}
                          for book in matches]}

    @app.post("/library/books/<int:book_id>/remove")
    def remove_library_book(book_id):
        db = database()
        with db:
            removed = db.execute("DELETE FROM library WHERE user_id = ? AND book_id = ?",
                                 (session["user_id"], book_id)).rowcount
        flash("Book removed from your library." if removed else "That book isn’t on your shelf.")
        return redirect(url_for("library"), code=303)

    @app.post("/library/books/new")
    def save_manual_book():
        title = request.form.get("title", "").strip()
        author = request.form.get("author", "").strip()
        chosen = request.form.get("existing_id", type=int)
        if chosen is None and (not title or not author or max(len(title), len(author)) > 250):
            flash("Enter a title and author, up to 250 characters each.")
            return redirect(url_for("library", _anchor="manual"), code=303)
        rating = submitted_rating()
        db = database()
        user_id = session["user_id"]
        with db:
            db.execute("BEGIN IMMEDIATE")
            exact = manual_book_matches(db, title, author)
            if any(book["id"] == chosen for book in exact):
                book_id = chosen
            elif chosen is not None:
                abort(400, "The selected book no longer matches the title and author.")
            else:
                key = search_key(title)
                matches = exact or [
                    book for book in search_books(db, user_id, title)
                    if book["title_key"] == key or book["title_key"].startswith(key + " (")
                ]
                if matches and (exact or request.form.get("different_book") != "yes"):
                    return render_template("manual_review.html", rating=rating, matches=matches,
                                           title=title, author=author, allow_new=not exact)
                book_id = add_manual_book(db, title, author)
            save_book(db, user_id, book_id, rating, update=rating is not None)
        flash("Book saved to your library.")
        return redirect(url_for("library"), code=303)

    @app.get("/recommendations")
    def recommendations():
        db = database()
        genre = selected_genre()
        user_id, browsing_id = session["user_id"], recommendation_scope(genre)
        books = library_books(db, user_id)
        remaining = max(RATINGS_REQUIRED - distinct_rating_count(books), 0)
        if remaining:
            return render_template("recommendations.html", remaining=remaining)

        page = max(1, request.args.get("page", 1, type=int))
        last = pages.last_page(db, user_id, browsing_id)
        if page > last + 1:
            return redirect(url_for("recommendations", page=last + 1, genre=genre))
        ranked = pages.load_page(db, user_id, browsing_id, page)
        dismissed = dismissed_books(db, user_id)
        if ranked is None:
            seen = books_by_goodreads_id(db, pages.seen_ids(db, user_id, browsing_id, page))
            try:
                _, ranked = select_books(db, app.config["MODEL_API_URL"], app.config["MODEL_API_KEY"],
                                         books, seen.values(), dismissed,
                                         reader_id=f"app:{user_id}",
                                         allowed_ids={key for key, (_, tags) in facets.items()
                                                      if genre in tags} if genre else None)
            except model_client.ModelServiceError:
                app.logger.warning("Model service recommendation request failed", exc_info=True)
                return render_template("recommendations.html", unavailable=True, page=page,
                                       genre=genre, genres=genres), 503
            with db:
                pages.save_page(db, user_id, browsing_id, page, ranked)
            ranked = pages.load_page(db, user_id, browsing_id, page)

        hidden = {work_key(book) for book in dismissed}
        hidden.update(work_key(book) for book in books
                      if book["shelf"] != "to-read" or book["rating"] is not None)
        to_read = {work_key(book) for book in books if book["shelf"] == "to-read"}
        metadata = books_by_goodreads_id(db, [row.book_id for row in ranked])
        cards = []
        for row in ranked:
            book = metadata.get(row.book_id)
            if book is not None and work_key(book) not in hidden:
                cards.append(dict(book=book, predicted=row.rating, to_read=work_key(book) in to_read,
                                  goodreads_url=f"https://www.goodreads.com/book/show/{quote(row.book_id, safe='')}"))
        return render_template("recommendations.html", cards=cards, page=page, genre=genre, genres=genres,
                               has_more=len(ranked) == PAGE_SIZE, matching_cards=bool(ranked))

    @app.post("/recommendations/<int:page>/books/<int:book_id>")
    def recommendation_action(page, book_id):
        db = database()
        genre = selected_genre()
        user_id, browsing_id = session["user_id"], recommendation_scope(genre)
        ranked = pages.load_page(db, user_id, browsing_id, page) or []
        book = db.execute("SELECT goodreads_id FROM books WHERE id = ?", (book_id,)).fetchone()
        if book is None or not any(item.book_id == book["goodreads_id"] for item in ranked):
            abort(404)
        action = request.form.get("action")
        with db:
            if action == "to-read":
                save_to_read(db, user_id, book_id)
                flash("Added to your To-read shelf.")
            elif action == "dismiss":
                dismiss_book(db, user_id, book_id)
                flash("This book will no longer be recommended.")
            else:
                abort(400)
        return redirect(url_for("recommendations", page=page, genre=genre), code=303)

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

    app.register_blueprint(auth)
    app.register_blueprint(imports)
    return app
