import argparse
from pathlib import Path
import re

import torch

from recommender_app.ml.book_features import BookFeatureBuilder
from recommender_app.ml.bundle import fingerprint, save_bundle
from recommender_app.ml.evaluation import Holdout
from recommender_app.ml.features import FeatureData
from recommender_app.ml.training import fit
from .train_candidate import MODEL_CONFIG, SEED, TRAINING_CONFIG

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+", args.version) or args.version == "v001":
        parser.error("Choose a new version such as v100")
    run = ROOT / "instance/ml/catalog-baselines" / args.version
    dataset = args.artifacts / "cleaned_dataset.parquet"
    BookFeatureBuilder(args.artifacts, dataset)
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    data = FeatureData(dataset)
    validation = Holdout(data, "val")
    model, history, best_epoch = fit(
        MODEL_CONFIG, TRAINING_CONFIG, data, data.warm("train"), validation, SEED,
        on_epoch=lambda row: print(row, flush=True),
    )
    save_bundle(run / "bundle", model, MODEL_CONFIG, data, {
        "catalog_base": True,
        "best_epoch": best_epoch,
        "candidate_validation": validation.metrics(model),
        "learning_curves": history,
    }, fingerprint(dataset))
    print(run / "bundle")


if __name__ == "__main__":
    main()
