import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import fingerprint


class BookFeatureBuilder:
    def __init__(self, artifact_dir, dataset):
        from gensim.models.doc2vec import Doc2Vec

        directory = Path(artifact_dir)
        self.spec = json.loads((directory / "book_features.json").read_text())
        if self.spec["schema_version"] != 1:
            raise ValueError("Unsupported book feature schema")
        if self.spec["dataset_sha256"] != fingerprint(dataset):
            raise ValueError("Book feature transforms belong to a different base dataset")
        self.encoder = Doc2Vec.load(str(directory / "description.model"))

    def transform(self, metadata):
        """Return static features in dataset order from one preprocessed metadata mapping."""
        from gensim.utils import simple_preprocess

        values = dict(metadata)
        description = values.get("description") or ""
        tokens = simple_preprocess(description)
        vector = self.encoder.infer_vector(tokens) if tokens else np.zeros(self.encoder.vector_size)
        values.update({f"d2v_{i}": value for i, value in enumerate(vector)})
        values["desc_missing"] = int(not description)
        for field, categories in self.spec["categories"].items():
            category = values.get(field) or "unknown"
            if category not in categories:
                category = "other"
            for column in self.spec["columns"]:
                if column.startswith(f"{field}_"):
                    values[column] = int(column == f"{field}_{category}")

        columns = self.spec["columns"]
        missing = [column for column in columns if values.get(column) is None]
        if missing:
            raise ValueError(f"Missing preprocessed book fields: {', '.join(missing)}")
        for column, (mean, scale) in self.spec["scaler"].items():
            value = float(values[column])
            if column in self.spec["log_columns"]:
                if value < 0:
                    raise ValueError(f"{column} must be nonnegative")
                value = np.log1p(value)
            values[column] = (value - mean) / scale
        features = np.asarray([values[column] for column in columns], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Book features must be finite")
        return pd.Series(features, index=columns)
