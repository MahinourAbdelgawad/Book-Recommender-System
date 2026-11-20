"""Index trained reader rows for bundles exported before the index was included."""

import argparse
import json
from pathlib import Path

import numpy as np

from recommender_app.ml.bundle import fingerprint, write_json
from recommender_app.ml.experiment import DATA, RUNS
from recommender_app.ml.features import FeatureData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=RUNS / "v001/bundle")
    parser.add_argument("--dataset", type=Path, default=DATA)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if fingerprint(args.dataset) != manifest["dataset_sha256"]:
        raise ValueError("The dataset must match the bundle's training dataset")
    data = FeatureData(args.dataset)
    with np.load(args.bundle / "catalog.npz", allow_pickle=False) as catalog:
        np.testing.assert_array_equal(catalog["user_ids"], data.user_ids.astype(str))
    trained = np.unique(data.users[data.warm_rows["train"]]).tolist()
    path = args.bundle / "trained_users.json"
    temporary = path.with_suffix(".tmp")
    write_json(temporary, {"weights_sha256": manifest["weights_sha256"], "indices": trained})
    temporary.replace(path)
    print(f"Indexed {len(trained)} trained readers in {path}")


if __name__ == "__main__":
    main()
