import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import torch

from recommender_app.ml.active_bundle import active_bundle_path
from recommender_app.ml.bundle import fingerprint, load_bundle, write_json
from recommender_app.ml.evaluation import Holdout
from recommender_app.ml.features import FeatureData
from recommender_app.ml.model import ModelConfig
from recommender_app.ml.training import TrainingConfig, fit
from .candidate import prepare_candidate_inputs, save_candidate_bundle
from .export import MIN_READER_RATINGS, catalog_book_ids, eligible_ratings
from .status import check_threshold, included_ratings_from_manifest

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "preparation-and-design/dataset/cleaned_dataset.parquet"
ACTIVE_BUNDLE = active_bundle_path()
OUTPUT_ROOT = ROOT / "instance/ml/candidates"
MODEL_CONFIG = ModelConfig(hidden=(128, 64))
TRAINING_CONFIG = TrainingConfig()
SEED = 42
WARM_TOLERANCE = 0.002
COLD_TOLERANCE = 0.005
HR_TOLERANCE = 0.01


def candidate_run_path(output_root, version, active_bundle):
    if not re.fullmatch(r"v[0-9]+", version):
        raise ValueError("Candidate version must look like v002")
    run = (Path(output_root) / version).resolve()
    active = Path(active_bundle).resolve()
    if run == active or run in active.parents or active in run.parents:
        raise ValueError("Candidate output must be separate from the active model bundle")
    if run.exists():
        raise FileExistsError(f"Candidate run already exists: {run}")
    return run


def passes_validation(candidate, active, *, warm_tolerance=WARM_TOLERANCE,
                      cold_tolerance=COLD_TOLERANCE, hr_tolerance=HR_TOLERANCE):
    return (
        candidate["warm_rmse"] <= active["warm_rmse"] + warm_tolerance
        and candidate["cold5_rmse"] <= active["cold5_rmse"] + cold_tolerance
        and candidate["cold5_hr10"] >= active["cold5_hr10"] - hr_tolerance
    )


def run_candidate(dataset, snapshot, active_bundle, output_root, version, threshold,
                  threads=4, seed=SEED, on_epoch=print):
    run = candidate_run_path(output_root, version, active_bundle)
    if threshold < 1:
        raise ValueError("Retraining threshold must be at least one")
    if threads < 1:
        raise ValueError("Thread count must be positive")

    active_model, active_manifest = load_bundle(active_bundle)
    dataset_hash = fingerprint(dataset)
    if dataset_hash != active_manifest["dataset_sha256"]:
        raise ValueError("Dataset differs from the one used by the active bundle")

    active_included = included_ratings_from_manifest(active_manifest)
    inputs = prepare_candidate_inputs(dataset, snapshot)
    validated_snapshot = eligible_ratings(
        inputs.ratings, catalog_book_ids(active_bundle), MIN_READER_RATINGS
    )
    if len(validated_snapshot) != len(inputs.ratings):
        raise ValueError(
            "Snapshot contains duplicate, ineligible, or out-of-catalog ratings; "
            "create it with the threshold-gated snapshot command"
        )
    threshold_result = check_threshold(inputs.ratings, active_included, threshold)
    if not threshold_result["threshold_met"]:
        raise ValueError(
            f"Only {threshold_result['pending_ratings']} new or changed eligible ratings; "
            f"threshold is {threshold}. No candidate was trained or written."
        )

    torch.set_num_threads(threads)
    baseline_data = FeatureData(dataset)
    baseline_val_rows = baseline_data.warm_rows["val"]
    candidate_val_rows = inputs.data.warm_rows["val"]
    if not np.array_equal(baseline_val_rows, candidate_val_rows):
        raise ValueError("Candidate construction changed the original warm validation examples")
    if not np.array_equal(
        baseline_data.ratings[baseline_val_rows], inputs.data.ratings[candidate_val_rows]
    ):
        raise ValueError("Candidate construction changed the original validation labels")
    baseline_cold_val = baseline_data.user_ids[baseline_data.cold_users["val"]]
    candidate_cold_val = inputs.data.user_ids[inputs.data.cold_users["val"]]
    if not np.array_equal(baseline_cold_val, candidate_cold_val):
        raise ValueError("Candidate construction changed the original cold validation readers")
    candidate_validation = Holdout(inputs.data, "val")
    if active_manifest.get("retraining"):
        selection = active_manifest["selection"]
        if selection.get("validation_protocol") != 1:
            raise ValueError("Active candidate lacks comparable validation metrics")
        active_metrics = selection["candidate_validation"]
    else:
        active_metrics = Holdout(baseline_data, "val").metrics(active_model)

    run.mkdir(parents=True, exist_ok=False)
    protocol = {
        "candidate_version": version,
        "active_version": active_manifest["version"],
        "active_weights_sha256": active_manifest["weights_sha256"],
        "validation_protocol": 1,
        "dataset_sha256": dataset_hash,
        "snapshot_sha256": inputs.snapshot_sha256,
        "threshold": threshold,
        "threshold_result": threshold_result,
        "model": asdict(MODEL_CONFIG),
        "training": asdict(TRAINING_CONFIG),
        "seed": seed,
        "validation_only": True,
        "test_split_used": False,
    }
    write_json(run / "protocol.json", protocol)

    model, history, best_epoch = fit(
        MODEL_CONFIG, TRAINING_CONFIG, inputs.data, inputs.train,
        candidate_validation, seed, on_epoch=on_epoch,
    )
    candidate_metrics = candidate_validation.metrics(model)
    passed = passes_validation(candidate_metrics, active_metrics)
    report = {
        **protocol,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "active_validation": active_metrics,
        "candidate_validation": candidate_metrics,
        "gate": {
            "warm_rmse_max_increase": WARM_TOLERANCE,
            "cold5_rmse_max_increase": COLD_TOLERANCE,
            "cold5_hr10_max_decrease": HR_TOLERANCE,
        },
        "passed": passed,
    }
    if passed:
        save_candidate_bundle(
            run / "bundle", model, MODEL_CONFIG, inputs, report, dataset_hash
        )
    write_json(run / "candidate_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="new candidate version, e.g. v002")
    parser.add_argument("--threshold", type=int, required=True,
                        help="minimum number of new or changed eligible ratings")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="exported, threshold-qualified ratings snapshot")
    parser.add_argument("--active-bundle", type=Path, default=ACTIVE_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    try:
        report = run_candidate(
            args.dataset, args.snapshot, args.active_bundle, args.output_root,
            args.version, args.threshold, args.threads,
        )
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({
        "passed": report["passed"],
        "active_validation": report["active_validation"],
        "candidate_validation": report["candidate_validation"],
        "candidate_bundle": str(Path(args.output_root) / args.version / "bundle")
        if report["passed"] else None,
    }, indent=2))
    print("Candidate is not active. The model service bundle was not changed.")


if __name__ == "__main__":
    main()
