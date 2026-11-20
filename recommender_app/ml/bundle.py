from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .model import HybridModel, ModelConfig

SCHEMA_VERSION = 1


def fingerprint(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, contents):
    path.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")


def save_bundle(directory, model, config, data, selection, dataset_hash):
    directory.mkdir()
    torch.save(model.state_dict(), directory / "weights.pt")
    np.savez_compressed(directory / "catalog.npz", book_features=data.book_table,
                        user_ids=data.user_ids.astype(str), book_ids=data.book_ids.astype(str),
                        catalog_indices=data.catalog_indices)
    write_json(directory / "manifest.json", {
        "schema_version": SCHEMA_VERSION, "version": directory.parent.name,
        "model": asdict(config), "feature_names": data.columns,
        "user_columns": data.user_columns, "scaler": data.scaler, "mean_rating": data.mean,
        "default_reader_std": data.default_std, "dataset_sha256": dataset_hash,
        "n_users": len(data.user_ids), "n_books": len(data.book_ids),
        "selection": selection,
        "weights_sha256": fingerprint(directory / "weights.pt"),
    })


def load_bundle(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported model bundle schema")
    weights = directory / "weights.pt"
    if fingerprint(weights) != manifest["weights_sha256"]:
        raise ValueError("Bundle weights do not match the manifest")
    config = ModelConfig(**manifest["model"])
    model = HybridModel(config, len(manifest["feature_names"]), manifest["n_users"],
                        manifest["n_books"], manifest["mean_rating"])
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval(), manifest
