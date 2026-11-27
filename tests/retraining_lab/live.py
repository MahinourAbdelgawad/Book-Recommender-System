"""Drive Flask usage while the real scheduler runs on a wall-clock interval."""

import argparse
from contextlib import closing
import json
import os
import queue
import re
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time

import httpx

from lab import ARTIFACTS, EXPANSION_BOOKS, READERS, RUNS, SRC, initialize, probe, seed


def endpoint():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = str(sock.getsockname()[1])
    return port, f"http://127.0.0.1:{port}"


def simulate(fixtures, number, flask_url, accounts, submitted, say):
    counts = seed(fixtures, number)
    with closing(sqlite3.connect(fixtures / "library.db")) as db:
        users = db.execute("SELECT id, email FROM users ORDER BY id").fetchall()
        ratings = db.execute("SELECT user_id, book_id, rating FROM library ORDER BY user_id, book_id").fetchall()
    password = "isolated-lab-password"
    for account, email in users:
        if account in accounts:
            continue
        client = httpx.Client(base_url=flask_url, timeout=20, trust_env=False)
        accounts[account] = client
        form = client.get("/signup")
        form.raise_for_status()
        token = csrf(form.text)
        response = client.post("/signup", data={
            "csrf_token": token, "email": email, "password": password,
        })
        if response.status_code != 303:
            raise RuntimeError(f"Signup failed for {email}: HTTP {response.status_code}")
    tokens = {}
    for account, book, rating in ratings:
        if (account, book) in submitted:
            continue
        client = accounts[account]
        if account not in tokens:
            page = client.get("/library")
            page.raise_for_status()
            tokens[account] = csrf(page.text)
        response = client.post(f"/library/books/{book}", data={
            "csrf_token": tokens[account], "rating": rating,
        })
        if response.status_code != 302 and response.status_code != 303:
            raise RuntimeError(f"Rating failed for reader {account}, book {book}: HTTP {response.status_code}")
        submitted.add((account, book))
    if users and len(submitted) >= 20:
        response = accounts[1].get("/recommendations")
        say("simulator", f"Reader 1 opened recommendations: HTTP {response.status_code}")
        response.raise_for_status()
    say("simulator", json.dumps({"stage": number, **counts, "ratings_submitted": len(submitted)}))


def csrf(html):
    match = re.search(r'name="csrf_token"\s+value="([^"]+)"', html)
    if not match:
        raise RuntimeError("Flask page did not contain a CSRF token")
    return match[1]


def run_live(run, interval, timeout):
    fixtures = run / "fixtures"
    fixtures.mkdir()
    (fixtures / "v001").symlink_to(run / "v001", target_is_directory=True)
    shutil.copy2(run / "library.db", fixtures / "library.db")
    seed(fixtures, 1)
    shutil.copy2(fixtures / "library.db", run / "library.db")
    api_port, api_url = endpoint()
    flask_port, flask_url = endpoint()
    env = os.environ.copy()
    env.pop("MODEL_BUNDLE_DIR", None)
    env.update(PYTHONUNBUFFERED="1", MODEL_API_KEY=secrets.token_hex(32),
               FLASK_SECRET_KEY=secrets.token_hex(32), MODEL_API_URL=api_url,
               MODEL_ACTIVE_BUNDLE_FILE=str(run / "active.txt"),
               LIBRARY_DB=str(run / "library.db"))
    events = queue.Queue()
    processes = []
    readers = []
    accounts = {}
    submitted = set()
    with (run / "live.log").open("x", buffering=1) as log:
        lock = threading.Lock()

        def say(component, message):
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{component}] {message}"
            with lock:
                print(line, flush=True)
                print(line, file=log)

        def start(component, *args):
            proc = subprocess.Popen([sys.executable, *args], cwd=SRC, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            processes.append(proc)

            def forward():
                with proc.stdout:
                    for line in proc.stdout:
                        say(component, line.rstrip())
                        if component == "scheduler":
                            if "Retraining cycle: " in line:
                                events.put(json.loads(line.split("Retraining cycle: ", 1)[1]))
                            elif "Retraining cycle failed" in line:
                                events.put({"status": "error"})
                events.put({"status": "exit", "component": component})

            thread = threading.Thread(target=forward, daemon=True)
            readers.append(thread)
            thread.start()

        def ready(url):
            for _ in range(120):
                try:
                    if httpx.get(url, timeout=1, trust_env=False).is_success:
                        return
                except httpx.RequestError:
                    pass
                time.sleep(0.5)
            raise RuntimeError(f"Service startup timed out: {url}")

        try:
            say("lab", f"Flask {flask_url}, FastAPI {api_url}, interval {interval}s")
            start("api", "-m", "uvicorn", "recommender_app.model_service.api:app",
                  "--host", "127.0.0.1", "--port", api_port)
            start("flask", "-m", "flask", "--app", "recommender_app.web", "run",
                  "--host", "127.0.0.1", "--port", flask_port, "--no-reload")
            ready(api_url + "/health")
            ready(flask_url + "/signup")
            start("scheduler", "-m", "recommender_app.retraining.scheduler",
                  "--interval-hours", str(interval / 3600), "--threshold", "21",
                  "--minimum-book-readers", "3", "--api-url", api_url,
                  "--database", str(run / "library.db"), "--book-artifacts", str(ARTIFACTS),
                  "--snapshot-dir", str(run / "snapshots"), "--output-root", str(run / "candidates"))
            for tick in range(1, len(READERS) + 1):
                result = events.get(timeout=timeout + interval)
                say("lab", json.dumps({"tick": tick, **result}))
                if tick < len(READERS):
                    if result.get("status") != "below_threshold":
                        raise RuntimeError(f"Unexpected outcome on tick {tick}: {result}")
                    expected = 0 if tick == 1 else 20
                    if result["pending_ratings"] != expected:
                        raise RuntimeError(f"Tick {tick} expected {expected} eligible ratings")
                    simulate(fixtures, tick + 1, flask_url, accounts, submitted, say)
                elif result.get("status") not in {"activated", "candidate_rejected"}:
                    raise RuntimeError(f"Final tick did not complete training: {result}")
            for report in sorted((run / "candidates").glob("*/candidate_report.json")):
                say("report", json.dumps(json.loads(report.read_text())))
            scored = probe(api_url, env)
            say("report", json.dumps({"prediction_probes": scored,
                                      "active_bundle": (run / "active.txt").read_text().strip()}))
            import pandas as pd

            snapshots = list((run / "snapshots").glob("*.parquet"))
            if len(snapshots) != 1:
                raise RuntimeError(f"Expected one snapshot, found {len(snapshots)}")
            ratings = pd.read_parquet(snapshots[0])
            supported = {book[0] for book in EXPANSION_BOOKS}
            counts = ratings[ratings.book_id.isin(scored)].book_id.value_counts().to_dict()
            say("report", json.dumps({"snapshot_rows": len(ratings), "new_book_ratings": counts,
                                      "active_new_books": sum(x["status"] == 200 for x in scored.values())}))
            if len(ratings) != 350 or counts != dict.fromkeys(supported, 10):
                raise RuntimeError("Final snapshot did not match the 350-rating fixture")
            admitted = supported if result["status"] == "activated" else set()
            if any(value["status"] != (200 if key in admitted else 404)
                   for key, value in scored.items()):
                raise RuntimeError("API did not serve the expected expanded catalog")
            say("lab", "Five live scheduler ticks completed")
        except BaseException as error:
            say("lab", f"Stopped: {type(error).__name__}: {error}")
            raise
        finally:
            for client in accounts.values():
                client.close()
            for proc in reversed(processes):
                if proc.poll() is None:
                    proc.terminate()
            for proc in reversed(processes):
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            for reader in readers:
                reader.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if not args.name.isascii() or not args.name.replace("-", "").isalnum():
        parser.error("Use letters, digits, and hyphens in the run name")
    if args.interval_seconds < 15 or args.timeout_seconds < 1:
        parser.error("Use an interval of at least 15 seconds and a positive timeout")
    run = RUNS / args.name
    if run.exists():
        parser.error("Choose a fresh run name")
    sys.path.insert(0, str(SRC))
    initialize(run)
    run_live(run, args.interval_seconds, args.timeout_seconds)


if __name__ == "__main__":
    main()
