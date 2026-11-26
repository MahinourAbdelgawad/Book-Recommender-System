# Retraining lab

Run one scheduler cycle at a time and inspect the result before moving on. The first cycle runs immediately. Later cycles require ten minutes after the last cycle completed, matching the scheduler's delay between cycles. Use `--skip-interval` to test the flow without waiting.

## Start

From `src/`, use the project's Python environment with its dependencies installed. The lab needs the local v001 bundle, the original cleaned Parquet dataset, and a seeded book catalog in the local SQLite database.

```bash
python tests/retraining_lab/lab.py trial-01 1
```

The lab copies v001 and reads only book metadata from the source database into a separate SQLite file. All accounts and ratings in that file are synthetic. The cleaned dataset is shared for read access. Each cycle starts its own local API and stops it when the cycle finishes. Snapshots, models, logs, locks, and the active pointer live under the ignored `runs/trial-01/` directory.

## Four rounds

The scheduler threshold is 21 eligible new or changed ratings, meaning strictly above 20. Each reader adds five distinct catalog ratings and twelve outside the catalog. The latter exercise the eligibility filter and do not add new books to the model.

1. **Round 1** adds 10 eligible ratings and 24 outside the catalog. Expect no snapshot or training.
2. **Round 2** adds 15 eligible ratings and 36 outside the catalog. Expect a training attempt with 25 pending. Activation depends on validation and API checks.
3. **Round 3** adds nothing. If round 2 activated, expect zero pending. If it was rejected, the 25 ratings remain pending and the scheduler can retry.
4. **Round 4** adds 25 eligible ratings and 60 outside the catalog. Expect another training attempt. A successful first promotion makes this round compare against a retrained model.

After each command returns, send **next** with the run name. We can inspect the round log and any candidate report before you run the same command with the next round number. Training rounds can take longer than ten minutes.

## Read the results

- `round-N.log` includes fixture counts, API output, training progress, and the scheduler decision.
- `candidates/vNNN/candidate_report.json` records validation metrics and whether the candidate passed.
- `active.txt` identifies the lab bundle selected by the API. A rejected candidate leaves it unchanged.
- A nonzero exit is a failed round. Keep its logs for diagnosis. Use a new run name to start a clean trial.

These rounds exercise normal promotion and rejection if they occur. A quality rejection is a valid result. A forced reload failure is a separate check for the rollback path.
