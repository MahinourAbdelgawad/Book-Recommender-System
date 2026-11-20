"""Preview and confirm imports without putting uploaded rows in cookies."""

from contextlib import closing
import json
import secrets

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for

from .goodreads import add_imported_book, match_books, parse_export, save_import_row
from .library import open_database

imports = Blueprint("imports", __name__, url_prefix="/import")


def current_draft(db):
    return db.execute("""
        SELECT rows_json FROM import_drafts WHERE id = ? AND created_at >= datetime('now', '-1 day')
    """, (session.get("goodreads_draft"),)).fetchone()


def preview_rows(db, rows):
    preview = []
    for row in rows:
        matches, method = match_books(db, row)
        different_edition = (method == "title" and row["goodreads_id"]
                             and any(book["goodreads_id"] for book in matches))
        if len(matches) > 1 or different_edition:
            status = "ambiguous"
        else:
            status = "matched" if matches else "new"
        preview.append(dict(row=row, matches=matches, status=status, method=method))
    return preview


@imports.get("/")
def index():
    with closing(open_database(current_app.config["DATABASE"])) as db:
        draft = current_draft(db)
        preview = preview_rows(db, json.loads(draft["rows_json"])) if draft else []
    return render_template("import.html", preview=preview, draft_id=session.get("goodreads_draft"),
                           counts={status: sum(item["status"] == status for item in preview)
                                   for status in ("matched", "new", "ambiguous")})


@imports.post("/preview")
def preview():
    upload = request.files.get("export")
    if not upload or not upload.filename:
        flash("Choose a Goodreads CSV file first.")
        return redirect(url_for("imports.index"), code=303)
    try:
        rows = parse_export(upload.read())
    except ValueError as error:
        flash(str(error))
        return redirect(url_for("imports.index"), code=303)
    draft_id = secrets.token_urlsafe(24)
    with closing(open_database(current_app.config["DATABASE"])) as db, db:
        db.execute("DELETE FROM import_drafts WHERE id = ? OR created_at < datetime('now', '-1 day')",
                   (session.get("goodreads_draft"),))
        db.execute("INSERT INTO import_drafts (id, rows_json) VALUES (?, ?)", (draft_id, json.dumps(rows)))
    session["goodreads_draft"] = draft_id
    return redirect(url_for("imports.index"), code=303)


@imports.post("/confirm")
def confirm():
    if request.form.get("draft_id") != session.get("goodreads_draft"):
        flash("This preview has changed. Review the current preview before confirming.")
        return redirect(url_for("imports.index"), code=303)
    imported = skipped = 0
    with closing(open_database(current_app.config["DATABASE"])) as db, db:
        db.execute("BEGIN IMMEDIATE")
        draft = current_draft(db)
        if not draft:
            flash("Upload the CSV again to create a fresh preview.")
            return redirect(url_for("imports.index"), code=303)
        entries = preview_rows(db, json.loads(draft["rows_json"]))
        for index, item in enumerate(entries):
            choice = request.form.get(f"row-{index}", "skip")
            matches = {str(book["id"]): book["id"] for book in item["matches"]}
            if choice == "skip":
                skipped += 1
                continue
            if choice != "new" and choice not in matches:
                db.rollback()
                flash("A match changed after preview. Review the updated choices and confirm again.")
                return redirect(url_for("imports.index"), code=303)
            row = item["row"]
            book_id = add_imported_book(db, row) if choice == "new" else matches[choice]
            save_import_row(db, session["user_id"], book_id, row)
            imported += 1
        db.execute("DELETE FROM import_drafts WHERE id = ?", (session["goodreads_draft"],))
    session.pop("goodreads_draft", None)
    flash(f"Imported {imported} {'row' if imported == 1 else 'rows'}. Skipped {skipped}.")
    return redirect(url_for("library"), code=303)
