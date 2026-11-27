"""Run a complete catalog expansion simulation in one isolated lab."""

import argparse
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

SRC = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
BASE = SRC / "instance/ml/v001/bundle"
ARTIFACTS = SRC / "instance/ml/catalog-features-v1"
READERS = (0, 10, 15, 0, 25)
NEW_BOOKS = json.loads(Path(__file__).with_name("books.json").read_text())
EXPANSION_BOOKS = NEW_BOOKS[:10]


def initialize(run):
    run.mkdir(parents=True)
    shutil.copytree(BASE, run / "v001/bundle")
    (run / "active.txt").write_text("v001/bundle\n")
    # Copy catalog metadata only. The source connection cannot write.
    from recommender_app.web.library import SCHEMA

    uri = (SRC / "instance/library.db").as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        with closing(sqlite3.connect(run / "library.db")) as db, db:
            db.executescript(SCHEMA)
            columns = [row[1] for row in db.execute("PRAGMA table_info(books)")]
            fields = ", ".join(columns)
            db.executemany(
                f"INSERT INTO books ({fields}) VALUES ({','.join('?' for _ in columns)})",
                source.execute(f"SELECT {fields} FROM books"),
            )


def seed(run, number):
    with np.load(run / "v001/bundle/catalog.npz", allow_pickle=False) as catalog:
        ids = {str(value) for value in catalog["book_ids"][catalog["catalog_indices"]]}
    with closing(sqlite3.connect(run / "library.db")) as db, db:
        works = {}
        for book, identifier, work in db.execute(
            "SELECT id, goodreads_id, work_id FROM books ORDER BY id"
        ):
            if identifier in ids:
                works.setdefault(work or f"book:{identifier}", book)
        if len(works) < 5:
            raise ValueError("Seed the source catalog before creating a lab")
        books = list(works.values())[:30]
        new_ids = []
        for identifier, title, author, year, description in NEW_BOOKS:
            if identifier in ids:
                raise ValueError(f"{title} is already in the baseline catalog")
            db.execute("""
                INSERT INTO books(goodreads_id, title, author, title_key, author_key,
                                  year, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goodreads_id) DO UPDATE SET
                    title=excluded.title, author=excluded.author,
                    title_key=excluded.title_key, author_key=excluded.author_key,
                    year=excluded.year, description=excluded.description,
                    bundle=0, series_key='', work_id=''
            """, (identifier, title, author, title.casefold(), author.casefold(), year,
                  description))
            new_ids.append(db.execute(
                "SELECT id FROM books WHERE goodreads_id=?", (identifier,)
            ).fetchone()[0])
        readers = READERS[number - 1]
        for reader in range(readers):
            account = db.execute(
                "INSERT INTO users(email, password_hash) VALUES (?, 'disabled')",
                (f"round{number}-{reader}@lab.invalid",),
            ).lastrowid
            entries = [new_ids[9 + account]] if account <= 30 else []
            if account <= 4:
                entries += [books[(account * 5 + i) % len(books)] for i in range(5)]
            db.executemany("INSERT INTO library(user_id, book_id, rating) VALUES (?, ?, ?)",
                           [(account, book, 3 + (i + account) % 3)
                            for i, book in enumerate(entries)])
        if number == len(READERS):
            for (account,) in db.execute("SELECT id FROM users").fetchall():
                base = [books[(account * 5 + i) % len(books)] for i in range(5)]
                supported = [new_ids[(account * 2 + i) % 10] for i in range(2)]
                db.executemany(
                    "INSERT OR IGNORE INTO library(user_id, book_id, rating) VALUES (?, ?, ?)",
                    [(account, book, 3 + (i + account) % 3)
                     for i, book in enumerate(base + supported)],
                )
        return {"readers_added": readers, "readers_total": db.execute(
            "SELECT COUNT(*) FROM users").fetchone()[0],
            "ratings_total": db.execute("SELECT COUNT(*) FROM library").fetchone()[0],
            "new_book_support": dict(db.execute("""
                SELECT books.goodreads_id, COUNT(library.user_id)
                FROM books LEFT JOIN library ON library.book_id=books.id
                WHERE books.goodreads_id IN ({placeholders}) GROUP BY books.goodreads_id
            """.format(placeholders=",".join("?" for _ in NEW_BOOKS)),
            [book[0] for book in NEW_BOOKS]))}


@contextmanager
def model_service(run, log):
    env = os.environ.copy()
    env.pop("MODEL_BUNDLE_DIR", None)
    env.update(MODEL_ACTIVE_BUNDLE_FILE=str(run / "active.txt"),
               MODEL_API_KEY=secrets.token_hex(32), PYTHONUNBUFFERED="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "recommender_app.model_service.api:app",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(60):
            if api.poll() is not None:
                raise RuntimeError("Lab API exited. See the round log.")
            try:
                with urlopen(url + "/health", timeout=1):
                    break
            except (URLError, TimeoutError):
                time.sleep(1)
        else:
            raise RuntimeError("Lab API did not start. See the round log.")
        yield url, env
    finally:
        if api.poll() is None:
            api.terminate()
            try:
                api.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api.kill()
                api.wait()


def probe(url, env):
    checks = {}
    for identifier, *_ in NEW_BOOKS:
        request = Request(
            url + "/v1/predict",
            data=json.dumps({"book_ids": [identifier], "ratings": {}}).encode(),
            headers={"Content-Type": "application/json", "X-API-Key": env["MODEL_API_KEY"]},
        )
        try:
            with urlopen(request, timeout=15) as response:
                checks[identifier] = {"status": response.status, "body": json.load(response)}
        except HTTPError as error:
            with error:
                checks[identifier] = {"status": error.code, "body": json.load(error)}
    return checks


def run_lab(run, interval):
    import pandas as pd

    summary = {"threshold": 21, "minimum_book_readers": 3, "interval_seconds": interval,
               "books": [{"id": item[0], "title": item[1], "author": item[2],
                          "year": item[3], "source": f"https://www.goodreads.com/book/show/{item[0]}"}
                         for item in NEW_BOOKS], "ticks": []}
    with (run / "api.log").open("x") as api_log, model_service(run, api_log) as (url, env):
        for number in range(1, len(READERS) + 1):
            if number > 1 and interval:
                time.sleep(interval)
            tick = {"number": number, "started_at": datetime.now(timezone.utc).isoformat()}
            summary["ticks"].append(tick)
            log_path = run / f"tick-{number}.log"
            print(f"Tick {number}/{len(READERS)}. Log: {log_path}", flush=True)
            try:
                tick.update(seed(run, number))
                tick["before"] = probe(url, env)
                with log_path.open("x") as log:
                    print(json.dumps(tick), file=log, flush=True)
                    subprocess.run([
                        sys.executable, "-m", "recommender_app.retraining.scheduler", "--once",
                        "--threshold", "21", "--minimum-book-readers", "3",
                        "--api-url", url, "--database", str(run / "library.db"),
                        "--book-artifacts", str(ARTIFACTS), "--snapshot-dir", str(run / "snapshots"),
                        "--output-root", str(run / "candidates"),
                    ], cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                decision = "Retraining cycle: "
                tick["scheduler"] = next(
                    json.loads(line.split(decision, 1)[1])
                    for line in reversed(log_path.read_text().splitlines()) if decision in line
                )
                tick["after"] = probe(url, env)
                tick["active_bundle"] = (run / "active.txt").read_text().strip()
                tick["candidate_reports"] = [
                    json.loads(path.read_text())
                    for path in sorted((run / "candidates").glob("*/candidate_report.json"))
                ]
                tick["snapshots"] = {}
                for path in (run / "snapshots").glob("*.parquet"):
                    ratings = pd.read_parquet(path, columns=["book_id"])
                    tick["snapshots"][path.name] = {
                        "rows": len(ratings),
                        "new_book_ratings": {identifier: int((ratings.book_id == identifier).sum())
                                             for identifier, *_ in NEW_BOOKS},
                    }
                expected = {"below_threshold"} if number < len(READERS) else {"activated", "candidate_rejected"}
                if tick["scheduler"]["status"] not in expected:
                    raise RuntimeError(f"Unexpected scheduler result in tick {number}")
                if number < len(READERS):
                    pending = 0 if number == 1 else 20
                    if tick["scheduler"]["pending_ratings"] != pending or tick["snapshots"]:
                        raise RuntimeError("Below-threshold tick did not preserve eligibility or snapshot rules")
                else:
                    snapshots = list(tick["snapshots"].values())
                    if snapshots != [{
                        "rows": 350,
                        "new_book_ratings": {identifier: (10 if index < 10 else 0)
                                             for index, (identifier, *_) in enumerate(NEW_BOOKS)},
                    }]:
                        raise RuntimeError("Final snapshot did not include the expected new-book ratings")
                activated = tick["scheduler"]["status"] == "activated"
                admitted = {book[0] for book in EXPANSION_BOOKS} if activated else set()
                tick["active_new_book_count"] = sum(
                    check["status"] == 200 for check in tick["after"].values()
                )
                if any(check["status"] != (200 if identifier in admitted else 404)
                       for identifier, check in tick["after"].items()):
                    raise RuntimeError(f"Unexpected new-book prediction result in tick {number}")
            except Exception as error:
                tick["error"] = str(error)
                raise
            finally:
                tick["finished_at"] = datetime.now(timezone.utc).isoformat()
                (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Name for this lab run")
    parser.add_argument("--interval-seconds", type=int, default=0,
                        help="Delay between ticks. Default: run back to back.")
    args = parser.parse_args()
    if not args.name.isascii() or not args.name.replace("-", "").isalnum():
        parser.error("Use letters, digits, and hyphens in the lab name")
    sys.path.insert(0, str(SRC))
    run = RUNS / args.name
    if run.exists():
        parser.error("Choose a fresh run name")
    if args.interval_seconds < 0:
        parser.error("Interval must be nonnegative")
    if not (ARTIFACTS / "cleaned_dataset.parquet").is_file():
        parser.error(f"Missing catalog feature export at {ARTIFACTS}")
    initialize(run)
    run_lab(run, args.interval_seconds)
    print(f"Complete. Report: {run / 'summary.json'}")


if __name__ == "__main__":
    main()
