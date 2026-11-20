import json
import math
from dataclasses import dataclass
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evaluation import predict
from .experiment import DATA, ROOT
from .features import FeatureData, reader_features
from .model import HybridModel, ModelConfig

DESIGN = ROOT / "preparation-and-design/model_design"


def main():
    torch.set_num_threads(4)
    data = FeatureData(DATA)
    results = DESIGN / "phase4_results"
    with (results / "final_artifacts.pkl").open("rb") as source:
        artifacts = pickle.load(source)  # Trusted artifact produced by the local notebook.
    keep = artifacts["kept_idx"]
    assert data.columns == tuple(artifacts["feature_names"][i] for i in keep)
    assert dict(zip(data.user_ids, range(len(data.user_ids)))) == artifacts["user_ids"]
    assert dict(zip(data.book_ids, range(len(data.book_ids)))) == artifacts["book_ids"]

    expected_books = np.load(results / "book_feature_table.npy")[:, keep]
    np.testing.assert_allclose(data.book_table, expected_books, atol=1e-6, rtol=0)
    for name, pair in data.scaler.items():
        np.testing.assert_allclose(pair, artifacts["scaler"][name], atol=1e-10, rtol=0)

    # Execute only the archived feature-building and model-definition cells.
    notebook = json.loads((DESIGN / "phase4_model_design_experiments.ipynb").read_text())
    legacy = dict(np=np, pd=pd, torch=torch, nn=nn, math=math, dataclass=dataclass,
                  DATA_PATH=DATA, QUICK=False, COLD_MIN_RATINGS=30, COLD_FRAC=0.10,
                  SHRINK_B=0.0, EPOCHS=30,
                  SKIP_COLS=["u_mean_rating", "b_mean_rating", "u_std_rating"])
    for number in (2, 3, 6):
        exec("".join(notebook["cells"][number]["source"]), legacy)

    checkpoint = torch.load(results / "final_model.pt", weights_only=True, map_location="cpu")
    config = artifacts["cfg"]
    model = HybridModel(ModelConfig(config["hidden"], config["emb_dim"], config["dropout"]),
                        len(data.columns), len(data.user_ids), len(data.book_ids), data.mean)
    archived = legacy["HybridNet"](legacy["Cfg"](**config), data.columns).eval()
    archived.load_state_dict(checkpoint)
    model.load_state_dict({key: checkpoint[key] for key in model.state_dict()})

    for split in ("train", "val"):
        examples = data.warm(split)
        rows = np.linspace(0, len(examples.users) - 1, 2048, dtype=int)
        reference_rows = legacy["idx"][split][rows]
        np.testing.assert_array_equal(data.warm_rows[split][rows], reference_rows)
        expected = legacy["X_all"][np.ix_(reference_rows, keep)]
        np.testing.assert_allclose(examples.features[rows], expected, atol=1e-6, rtol=0)
        users, books = examples.users[rows], examples.books[rows]
        with torch.inference_mode():
            reference = archived(*(torch.from_numpy(a) for a in (expected, users, books))).numpy()
        predictions = predict(model, examples.features[rows], users, books)
        np.testing.assert_allclose(predictions, reference, atol=1e-5, rtol=0)
        print(f"{split}: {len(rows)} feature rows and saved-model predictions match")
        del examples

    ratings = np.array([1, 3, 4, 4, 5], dtype=np.float32)
    profile = reader_features(ratings, mean=data.mean, default_std=data.default_std, scaler=data.scaler)
    expected_profile = legacy["std_user"](
        np.array([[ratings.mean(), ratings.std(), np.log1p(len(ratings))]]))[0]
    np.testing.assert_allclose(profile, expected_profile, atol=1e-6, rtol=0)
    books = data.catalog_indices[:128]
    users = np.full(len(books), data.unknown_user)
    expected = expected_books[books]
    expected[:, data.user_columns] = expected_profile
    with torch.inference_mode():
        reference = archived(*(torch.from_numpy(a) for a in (expected, users, books))).numpy()
    np.testing.assert_allclose(predict(model, data.with_reader(books, profile), users, books),
                               reference, atol=1e-5, rtol=0)
    print("Five-rating reader profile and 128 cold-start predictions match")


if __name__ == "__main__":
    main()
