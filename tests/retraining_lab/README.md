# Catalog expansion dry lab

## Live scheduler run

From `src/` with the virtual environment active:

```bash
python tests/retraining_lab/live.py live-50
```

This launches isolated Flask, FastAPI, and a continuously running scheduler.
The simulator signs up 50 accounts and submits their ratings through Flask.
The scheduler waits a real 60 seconds between cycles by default. Use
`--interval-seconds 600` for ten minutes. The final cycle also takes the time
needed for training. Use a fresh run name or a longer interval if your machine
cannot finish submitting a batch before the next tick.

All console output is also written to `runs/live-50/live.log`, with each line
labeled by process. The launcher stops its three services after five completed
ticks or on error. The log includes the final snapshot and new-book API checks.

## Fast simulation

From `src/`, activate the project virtual environment and run:

```bash
python tests/retraining_lab/lab.py expansion-01
```

Use a fresh name for each run. The lab needs the v001 bundle, the original
cleaned dataset, the catalog feature export, and the local book metadata.
It copies v001 and book metadata into a fresh database and starts one private
FastAPI process for the whole run. The accounts and ratings are synthetic.

The command drives five scheduler ticks with no delay by default. Add
`--interval-seconds 600` for a ten-minute delay between ticks. This exercises
the real scheduler's single-cycle path with a lab-controlled clock. The final
tick performs one real training attempt and may take several minutes.

1. **Empty library.** All 40 books from the Goodreads 2025 releases shelf are loaded with no ratings.
2. **Early feedback.** Ten readers join. Four submit complete catalog histories, giving 20 eligible ratings.
3. **Sparse feedback.** Fifteen more readers join. Their new-book ratings are too sparse to qualify.
4. **Idle tick.** The eligible count remains at 20, below the threshold of 21.
5. **Expansion attempt.** Another 25 readers join and all 50 complete their histories.
   The snapshot should contain 350 eligible ratings: 250 for existing books and 100
   for exactly ten new books. Each of the other 30 new books has one rating and
   stays below the three-reader minimum.

The lab checks all 40 books through the API. If validation passes, exactly ten
should become scoreable. If the candidate is rejected, all 40 should remain
outside the active catalog. Book IDs and titles come from the
[Goodreads 2025 releases shelf](https://www.goodreads.com/shelf/show/releases-2025).

## Results

Find the ignored output directory at `runs/expansion-01/` beside this guide.

- `summary.json` records each tick, fixture counts, scheduler outcome,
  per-book snapshot counts, validation reports, and prediction probes.
- `tick-*.log` captures scheduler output and training progress.
- `api.log` captures startup, health, prediction, and reload requests.
- `snapshots/`, `candidates/`, and `active.txt` retain the actual artifacts.

A rejected model should leave the new books unavailable for scoring. A
successful activation should make them scoreable. The run records which
outcome occurred. It does not force a passing model or inject a rollback fault.
