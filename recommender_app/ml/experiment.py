import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from .bundle import fingerprint, load_bundle, save_bundle, write_json
from .evaluation import Holdout
from .features import FeatureData
from .model import HybridModel, ModelConfig
from .training import TrainingConfig, fit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "preparation-and-design/dataset/cleaned_dataset.parquet"
RUNS = ROOT / "instance/ml"
DEPTHS = ((64,), (128, 64), (256, 128, 64))
SEEDS = (0, 1, 2)
SETTINGS = TrainingConfig()
WARM_TOLERANCE = 0.002
COLD_TOLERANCE = 0.005
HR_TOLERANCE = 0.01
OBJECTIVE_TIE = 0.001


def select(records):
    """Screen against the three-layer control, then favor simplicity within a tie."""
    summaries = []
    for depth in range(1, 4):
        rows = [row for row in records if row["depth"] == depth]
        metrics = {name: float(np.mean([row[name] for row in rows]))
                   for name in ("warm_rmse", "cold5_rmse", "cold5_hr10", "objective", "gap")}
        summaries.append(dict(depth=depth, parameters=rows[0]["parameters"], **metrics))
    control = summaries[-1]
    eligible = [row for row in summaries
                if row["warm_rmse"] <= control["warm_rmse"] + WARM_TOLERANCE
                and row["cold5_rmse"] <= control["cold5_rmse"] + COLD_TOLERANCE
                and row["cold5_hr10"] >= control["cold5_hr10"] - HR_TOLERANCE]
    best = min(row["objective"] for row in eligible)
    chosen = min((row for row in eligible if row["objective"] <= best + OBJECTIVE_TIE),
                 key=lambda row: (row["parameters"], abs(row["gap"])))
    candidates = [row for row in records if row["depth"] == chosen["depth"]]
    representative = min(candidates, key=lambda row: (abs(row["objective"] - chosen["objective"]), row["seed"]))
    return chosen, representative, summaries


def plot_curves(directory, rows, winning_depth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, layout="constrained")
    for depth, ax in enumerate(axes, 1):
        for seed in SEEDS:
            curve = [row for row in rows if row["depth"] == depth and row["seed"] == seed]
            epochs = [row["epoch"] for row in curve]
            for name, color in (("train_rmse", "tab:blue"), ("validation_rmse", "tab:orange")):
                ax.plot(epochs, [row[name] for row in curve], color=color, alpha=0.65,
                        label=name.replace("_", " ") if seed == SEEDS[0] else None)
        ax.set_title(f"{depth} hidden layer{'s' if depth > 1 else ''}" +
                     (" (selected)" if depth == winning_depth else ""))
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("RMSE in stars")
    axes[0].legend()
    fig.savefig(directory / "learning_curves.png", dpi=160)
    plt.close(fig)


def sweep(dataset, run):
    dataset_hash = fingerprint(dataset)
    data = FeatureData(dataset)
    train = data.warm("train")
    validation = Holdout(data, "val")
    run.mkdir(parents=True, exist_ok=False)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir()
    plan = {"depths": DEPTHS, "seeds": SEEDS, "training": asdict(SETTINGS),
            "warm_tolerance": WARM_TOLERANCE, "cold_tolerance": COLD_TOLERANCE,
            "hr_tolerance": HR_TOLERANCE, "objective_tie": OBJECTIVE_TIE}
    write_json(run / "protocol.json", plan)
    rows, records = [], []
    for depth, hidden in enumerate(DEPTHS, 1):
        config = ModelConfig(hidden=hidden)
        for seed in SEEDS:
            print(f"Training {hidden} with seed {seed}", flush=True)
            model, history, best_epoch = fit(
                config, SETTINGS, data, train, validation, seed,
                on_epoch=lambda row: print(row, flush=True))
            metrics = validation.metrics(model)
            parameters = sum(p.numel() for p in model.parameters())
            best = history[best_epoch - 1]
            records.append(dict(depth=depth, seed=seed, best_epoch=best_epoch,
                                parameters=parameters, train_rmse=best["train_rmse"],
                                gap=best["gap"], **metrics))
            rows.extend(dict(depth=depth, seed=seed, **epoch) for epoch in history)
            torch.save(model.state_dict(), checkpoints / f"depth-{depth}-seed-{seed}.pt")
            pd.DataFrame(rows).to_csv(run / "learning_curves.csv", index=False)
            pd.DataFrame(records).to_csv(run / "validation_runs.csv", index=False)

    chosen, representative, summaries = select(records)
    pd.DataFrame(summaries).to_csv(run / "depth_summary.csv", index=False)
    plot_curves(run, rows, chosen["depth"])
    config = ModelConfig(hidden=DEPTHS[chosen["depth"] - 1])
    model = HybridModel(config, len(data.columns), len(data.user_ids), len(data.book_ids), data.mean)
    weights = checkpoints / f"depth-{chosen['depth']}-seed-{representative['seed']}.pt"
    model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    save_bundle(run / "bundle", model, config, data, {
        "protocol": plan, "configuration": chosen, "checkpoint": representative,
    }, dataset_hash)
    print(f"Selected {config.hidden}; bundle saved to {run / 'bundle'}", flush=True)


def evaluate_test(dataset, run):
    report = run / "test_metrics.json"
    if report.exists():
        print(report.read_text(), end="")
        return
    model, manifest = load_bundle(run / "bundle")
    if fingerprint(dataset) != manifest["dataset_sha256"]:
        raise ValueError("Test dataset differs from the one used to select the bundle")
    data = FeatureData(dataset)
    scores = Holdout(data, "test").metrics(model)
    contents = {"version": manifest["version"], "weights_sha256": manifest["weights_sha256"],
                "metrics": scores}
    with report.open("x") as output:
        json.dump(contents, output, indent=2, allow_nan=False)
        output.write("\n")
    print(report.read_text(), end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sweep", "evaluate-test"))
    parser.add_argument("--version", required=True, help="A new run name, e.g. v001")
    parser.add_argument("--dataset", type=Path, default=DATA)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+", args.version):
        parser.error("--version must be v followed by digits")
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    run = RUNS / args.version
    command = sweep if args.command == "sweep" else evaluate_test
    command(args.dataset, run)


if __name__ == "__main__":
    main()
