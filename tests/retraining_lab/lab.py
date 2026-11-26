"""Run one isolated retraining round at a time from src/."""

import argparse
from contextlib import closing
import os
from pathlib import Path
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

SRC = Path(__file__).resolve().parents[2]
RUNS = Path(__file__).resolve().parent / "runs"
BASE = SRC / "instance/ml/v001/bundle"
DATASET = SRC / "preparation-and-design/dataset/cleaned_dataset.parquet"
READERS = (2, 3, 0, 5)


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
        books = list(works.values())[:5]
        if len(books) < 5:
            raise ValueError("Seed the source catalog before creating a lab")
        readers = READERS[number - 1]
        for reader in range(readers):
            account = db.execute(
                "INSERT INTO users(email, password_hash) VALUES (?, 'disabled')",
                (f"round{number}-{reader}@lab.invalid",),
            ).lastrowid
            db.executemany("INSERT INTO library(user_id, book_id, rating) VALUES (?, ?, ?)",
                           [(account, book, 4 + (i + reader) % 2)
                            for i, book in enumerate(books)])
            for i in range(12):
                title = f"Lab book {number}-{reader}-{i}"
                book = db.execute(
                    """INSERT INTO books(goodreads_id, title, author, title_key, author_key)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"lab-{number}-{reader}-{i}", title, "Lab Author",
                     title.casefold(), "lab author"),
                ).lastrowid
                db.execute("INSERT INTO library(user_id, book_id, rating) VALUES (?, ?, 4)",
                           (account, book))
    return readers


def cycle(run, log):
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
        subprocess.run(
            [sys.executable, "-m", "recommender_app.retraining.scheduler", "--once",
             "--threshold", "21", "--api-url", url, "--database", str(run / "library.db"),
             "--dataset", str(DATASET), "--snapshot-dir", str(run / "snapshots"),
             "--output-root", str(run / "candidates")],
            cwd=SRC, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    finally:
        if api.poll() is None:
            api.terminate()
            try:
                api.wait(timeout=10)
            except subprocess.TimeoutExpired:
                api.kill()
                api.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Name for this lab run")
    parser.add_argument("round", type=int, choices=range(1, 5))
    parser.add_argument("--skip-interval", action="store_true",
                        help="Run the next tick without waiting ten minutes")
    args = parser.parse_args()
    if not args.name.isascii() or not args.name.replace("-", "").isalnum():
        parser.error("Use letters, digits, and hyphens in the lab name")
    sys.path.insert(0, str(SRC))
    run = RUNS / args.name
    if not run.exists():
        if args.round != 1:
            parser.error("Start with round 1")
        initialize(run)
    if args.round > 1:
        previous = run / f"round-{args.round - 1}.done"
        if not previous.exists():
            parser.error("Complete the previous round first")
        remaining = 600 - (time.time() - previous.stat().st_mtime)
        if remaining > 0 and not args.skip_interval:
            parser.error(f"Next tick is in {remaining:.0f} seconds. Run again then.")
    log_path = run / f"round-{args.round}.log"
    print(f"Running round {args.round}. Follow {log_path}", flush=True)
    with log_path.open("x") as log:
        readers = seed(run, args.round)
        print(f"Added {readers} readers with {readers * 5} catalog ratings and "
              f"{readers * 12} outside-catalog ratings. Threshold: 21.",
              file=log, flush=True)
        cycle(run, log)
        print("Bundle after round:", (run / "active.txt").read_text().strip(),
              file=log, flush=True)
    (run / f"round-{args.round}.done").touch()


if __name__ == "__main__":
    main()
