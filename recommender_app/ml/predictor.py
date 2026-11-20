import json

import numpy as np

from .bundle import load_bundle
from .evaluation import predict
from .features import reader_features


class Predictor:
    def __init__(self, directory):
        self.model, manifest = load_bundle(directory)
        self.version = manifest["version"]
        self.manifest = manifest
        with np.load(directory / "catalog.npz", allow_pickle=False) as catalog:
            self.book_features = catalog["book_features"]
            self.book_ids = catalog["book_ids"]
            self.catalog_indices = catalog["catalog_indices"]
            user_ids = catalog["user_ids"]
            if "trained_user_indices" in catalog:
                trained = catalog["trained_user_indices"]
            else:
                index_path = directory / "trained_users.json"
                if not index_path.exists():
                    raise FileNotFoundError(
                        f"{index_path} is missing. Run python -m recommender_app.model_service.index_readers."
                    )
                index = json.loads(index_path.read_text())
                if index["weights_sha256"] != manifest["weights_sha256"]:
                    raise ValueError("Reader index belongs to a different checkpoint")
                trained = index["indices"]

        self.users = {str(user_ids[i]): int(i) for i in trained}
        self.books = {str(self.book_ids[i]): int(i) for i in self.catalog_indices}

    def score(self, book_ids, ratings, reader_id=None):
        indices = np.array([self.books[book_id] for book_id in book_ids], dtype=np.int64)
        user = self.users.get(reader_id, self.model.unknown_user)
        profile = reader_features(list(ratings.values()), mean=self.manifest["mean_rating"],
                                  default_std=self.manifest["default_reader_std"],
                                  scaler=self.manifest["scaler"])
        features = self.book_features[indices]
        features[:, self.manifest["user_columns"]] = profile
        scores = predict(self.model, features, np.full(len(indices), user, dtype=np.int64), indices)
        return np.clip(scores, 1, 5).tolist()

    def recommend(self, ratings, reader_id, excluded, top_n):
        excluded = set(excluded) | ratings.keys()
        candidates = [book for book in self.books if book not in excluded]
        if not candidates:
            return []
        scores = self.score(candidates, ratings, reader_id)
        ranked = sorted(zip(candidates, scores), key=lambda item: -item[1])
        return ranked[:top_n]
