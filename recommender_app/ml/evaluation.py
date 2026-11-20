import numpy as np
import torch


@torch.inference_mode()
def predict(model, features, users, books, batch_size=8192):
    model.eval()
    return np.concatenate([
        model(*(torch.from_numpy(a[start:start + batch_size]) for a in (features, users, books))).numpy()
        for start in range(0, len(users), batch_size)
    ])


def rmse(predictions, ratings):
    residuals = np.clip(predictions, 1, 5) - ratings
    return float(np.sqrt(np.mean(residuals**2)))


class Holdout:
    def __init__(self, data, split):
        self.warm = data.warm(split)
        self.cold, self.cold_users = data.cold(split)
        seed = 3 if split == "val" else 4
        rng = np.random.default_rng(seed)
        users, starts, counts = np.unique(self.cold_users, return_index=True, return_counts=True)
        self.rank_groups = [(start, count) for start, count in zip(starts, counts)]
        negatives = np.stack([
            rng.choice(np.setdiff1d(np.arange(len(data.book_ids)), data.books[data.user_rows[u]]),
                       100, replace=False)
            for u in users
        ]).ravel()
        self.negative_books = negatives
        self.negative_users = np.full(len(negatives), data.unknown_user)
        profiles = self.cold.features[starts][:, data.user_columns]
        self.negative_features = data.with_reader(negatives, np.repeat(profiles, 100, axis=0))

    def warm_rmse(self, model):
        warm = self.warm
        return rmse(predict(model, warm.features, warm.users, warm.books), warm.ratings)

    def metrics(self, model):
        cold = self.cold
        cold_scores = predict(model, cold.features, cold.users, cold.books)
        negatives = predict(model, self.negative_features, self.negative_users,
                            self.negative_books).reshape(-1, 100)
        hits = []
        for (start, count), scores in zip(self.rank_groups, negatives):
            rows = slice(start, start + count)
            positives = cold_scores[rows][cold.ratings[rows] >= 4]
            if len(positives):
                hits.append(((scores >= positives[:, None]).sum(axis=1) < 10).mean())
        warm_rmse = self.warm_rmse(model)
        cold_rmse = rmse(cold_scores, cold.ratings)
        return {"warm_rmse": warm_rmse, "cold5_rmse": cold_rmse,
                "cold5_hr10": float(np.mean(hits)), "objective": (warm_rmse + cold_rmse) / 2}
