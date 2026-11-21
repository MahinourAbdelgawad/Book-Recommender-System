from contextlib import closing
import secrets

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from .library import open_database

auth = Blueprint("auth", __name__)


@auth.route("/login", methods=("GET", "POST"), endpoint="login")
@auth.route("/signup", methods=("GET", "POST"), endpoint="signup")
def account():
    if session.get("user_id") is not None:
        return redirect(url_for("home"))
    signup = request.path == "/signup"
    if request.method == "POST":
        email = request.form.get("email", "").strip().casefold()
        password = request.form.get("password", "")
        if not email or "@" not in email or len(email) > 254 or len(password) > 128:
            flash("Enter a valid email and password.")
        elif signup and len(password) < 8:
            flash("Use at least eight characters for your password.")
        else:
            with closing(open_database(current_app.config["DATABASE"])) as db, db:
                if signup:
                    password_hash = generate_password_hash(password)
                    db.execute("BEGIN IMMEDIATE")
                    first = not db.execute("SELECT 1 FROM users LIMIT 1").fetchone()
                    result = db.execute("""
                        INSERT INTO users(email, password_hash) VALUES (?, ?) ON CONFLICT(email) DO NOTHING
                    """, (email, password_hash))
                    user_id = result.lastrowid if result.rowcount else None
                    if first and user_id:
                        db.execute("UPDATE library SET user_id = ? WHERE user_id IS NULL", (user_id,))
                        db.execute("UPDATE recommendation_feedback SET user_id = ? WHERE user_id IS NULL", (user_id,))
                else:
                    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
                    user_id = user["id"] if user and check_password_hash(user["password_hash"], password) else None
            if user_id:
                session.clear()
                session.update(user_id=user_id, browsing_id=secrets.token_urlsafe(24))
                return redirect(url_for("home"), code=303)
            flash("An account already uses this email." if signup else "Email or password is incorrect.")
    return render_template("account.html", signup=signup)


@auth.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"), code=303)
