from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

USER_COLUMNS = ("u_mean_rating", "u_std_rating", "u_n_train")
BOOK_COLUMNS = ("b_mean_rating", "b_n_train")
NON_FEATURES = ("user_id", "book_id", "rating", "split", "is_reviewed")


def standardize(values, columns, scaler):
    mean, scale = np.array([scaler[name] for name in columns]).T
    return ((values - mean) / scale).astype(np.float32)


def reader_features(ratings, *, mean, default_std, scaler):
    """Transform explicit reader ratings (including ratings of out-of-catalog books)."""
    ratings = np.asarray(ratings, dtype=np.float64)
    if ratings.ndim != 1 or not np.isfinite(ratings).all() or not ((ratings >= 1) & (ratings <= 5)).all():
        raise ValueError("Expected a sequence of ratings between 1 and 5")
    values = [ratings.mean() if ratings.size else mean,
              ratings.std() if ratings.size > 1 else default_std,
              np.log1p(ratings.size)]
    return standardize(values, USER_COLUMNS, scaler)


@dataclass
class Examples:
    features: np.ndarray
    users: np.ndarray
    books: np.ndarray
    ratings: np.ndarray


def moments(ids, ratings, size, fallback):
    counts = np.bincount(ids, minlength=size).astype(np.float64)
    sums = np.bincount(ids, weights=ratings, minlength=size)
    means = np.divide(sums, counts, out=np.full(size, fallback), where=counts > 0)
    return counts, sums, means


def leave_one_out(ids, ratings, counts, sums, fallback):
    others = counts[ids] - 1
    return np.divide(sums[ids] - ratings, others,
                     out=np.full(len(ids), fallback), where=others > 0)


class FeatureData:
    """Keep book features and user statistics separate until a split is materialized."""

    def __init__(self, path: Path, extra_train: pd.DataFrame | None = None, new_books=None):
        base = pd.read_parquet(path)
        new_books = {} if new_books is None else new_books
        if set(new_books) & set(base.book_id):
            raise ValueError("New book features cannot replace base book features")
        # Keep the original holdout cohort fixed when app readers are appended.
        # The model's validation/test comparison must use the same readers each time.
        base_user_ids, base_users = np.unique(base.user_id, return_inverse=True)
        base_counts = np.bincount(base_users)
        rng = np.random.default_rng(12345)
        eligible = np.flatnonzero(base_counts >= 30)
        cold_users = np.sort(rng.choice(eligible, round(0.10 * len(base_user_ids)), replace=False))
        shuffled = rng.permutation(cold_users)
        midpoint = len(shuffled) // 2
        cold_user_ids = {
            "val": base_user_ids[np.sort(shuffled[:midpoint])],
            "test": base_user_ids[np.sort(shuffled[midpoint:])],
        }

        if extra_train is None:
            frame = base
        else:
            required = {"user_id", "book_id", "rating", "split"}
            missing = required - set(extra_train.columns)
            if missing:
                raise ValueError(f"Extra training rows are missing columns: {sorted(missing)}")
            if not extra_train.empty:
                if not (extra_train.split.astype(str).str.lower() == "train").all():
                    raise ValueError("Extra app rows must all belong to the train split")
                ratings = pd.to_numeric(extra_train.rating, errors="raise")
                if not np.isfinite(ratings).all() or not ratings.between(1, 5).all():
                    raise ValueError("Extra app ratings must be between 1 and 5")
                if set(extra_train.user_id) & set(base.user_id):
                    raise ValueError("Extra app reader IDs must not collide with original readers")
                unknown_books = set(extra_train.book_id) - set(base.book_id) - new_books.keys()
                if unknown_books:
                    raise ValueError("Extra app rows contain books outside the original catalog")
            extra = extra_train.reindex(columns=base.columns)
            frame = pd.concat((base, extra), ignore_index=True)

        self.columns = tuple(name for name in frame if name not in NON_FEATURES)
        self.user_columns = [self.columns.index(name) for name in USER_COLUMNS]
        self.book_columns = [self.columns.index(name) for name in BOOK_COLUMNS]
        self.user_ids, self.users = np.unique(frame.user_id, return_inverse=True)
        self.book_ids, self.books = np.unique(frame.book_id, return_inverse=True)
        self.holdout_book_indices = np.flatnonzero(np.isin(self.book_ids, base.book_id.unique()))
        self.ratings = frame.rating.to_numpy(dtype=np.float32)
        self.split = frame.split.astype(str).str.lower().replace(
            {"valid": "val", "validation": "val"}).to_numpy()
        if set(self.split) != {"train", "val", "test"}:
            raise ValueError("Expected train, val and test split labels")

        order = np.argsort(self.users, kind="stable")
        counts = np.bincount(self.users)
        self.user_rows = np.split(order, np.cumsum(counts)[:-1])
        user_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.cold_users = {
            split: np.sort(np.asarray([user_index[user_id] for user_id in ids], dtype=np.int64))
            for split, ids in cold_user_ids.items()
        }
        self.cold_order = {u: rng.permutation(self.user_rows[u]) for u in cold_users}
        warm = ~np.isin(self.users[order], cold_users)
        self.warm_rows = {split: order[warm & (self.split[order] == split)]
                          for split in ("train", "val", "test")}

        rows = self.warm_rows["train"]
        users, books = self.users[rows], self.books[rows]
        ratings = self.ratings[rows].astype(np.float64)
        self.mean = float(ratings.mean())
        uc, us, um = moments(users, ratings, len(self.user_ids), self.mean)
        bc, bs, bm = moments(books, ratings, len(self.book_ids), self.mean)
        squares = np.bincount(users, weights=ratings**2, minlength=len(uc))
        std = np.sqrt(np.maximum(squares / np.maximum(uc, 1) - um**2, 0))
        self.default_std = float(std[uc > 0].mean())
        std[uc == 0] = self.default_std

        user_values = np.column_stack((um, std, np.log1p(uc)))
        book_values = np.column_stack((bm, np.log1p(bc)))
        self.train_user_mean = leave_one_out(users, ratings, uc, us, self.mean)
        self.train_book_mean = leave_one_out(books, ratings, bc, bs, self.mean)
        train_values = np.column_stack((self.train_user_mean, std[users], np.log1p(uc[users]),
                                        self.train_book_mean, np.log1p(bc[books])))
        self.scaler = {name: (float(values.mean()), float(values.std()) + 1e-8)
                       for name, values in zip(USER_COLUMNS + BOOK_COLUMNS, train_values.T)}
        self.user_table = standardize(user_values, USER_COLUMNS, self.scaler)
        self.book_table = np.zeros((len(self.book_ids), len(self.columns)), dtype=np.float32)
        self.catalog_indices = np.flatnonzero(bc > 0)
        static = [name for name in self.columns if name not in USER_COLUMNS + BOOK_COLUMNS]
        first = frame.drop_duplicates("book_id").set_index("book_id").loc[self.book_ids, static]
        for book_id, values in new_books.items():
            if set(values) != set(static) or book_id not in first.index:
                raise ValueError("New book features do not match the training schema")
            first.loc[book_id, static] = [values[name] for name in static]
        self.book_table[:, [self.columns.index(name) for name in static]] = first.to_numpy(np.float32)
        self.book_table[:, self.book_columns] = standardize(book_values, BOOK_COLUMNS, self.scaler)
        if not np.isfinite(self.book_table).all():
            raise ValueError("Book features contain missing or non-finite values")

    @property
    def unknown_user(self):
        return len(self.user_ids)

    def with_reader(self, books, reader):
        features = self.book_table[books]
        features[:, self.user_columns] = reader
        return features

    def warm(self, split):
        rows = self.warm_rows[split]
        users, books = self.users[rows], self.books[rows]
        features = self.with_reader(books, self.user_table[users])
        if split == "train":
            for name, index, means in (
                (USER_COLUMNS[0], self.user_columns[0], self.train_user_mean),
                (BOOK_COLUMNS[0], self.book_columns[0], self.train_book_mean),
            ):
                mean, scale = self.scaler[name]
                features[:, index] = (means - mean) / scale
        return Examples(features, users, books, self.ratings[rows])

    def cold(self, split):
        rows, profiles = [], []
        for user in self.cold_users[split]:
            order = self.cold_order[user]
            if split == "val":
                order = order[self.split[order] != "test"]
            context, evaluation = order[:5], order[20:]
            if not len(evaluation):
                continue
            profile = reader_features(self.ratings[context], mean=self.mean,
                                      default_std=self.default_std, scaler=self.scaler)
            rows.extend(evaluation)
            profiles.extend([profile] * len(evaluation))
        rows = np.asarray(rows, dtype=np.int64)
        books = self.books[rows]
        examples = Examples(self.with_reader(books, np.asarray(profiles)),
                            np.full(len(rows), self.unknown_user), books, self.ratings[rows])
        return examples, self.users[rows]
