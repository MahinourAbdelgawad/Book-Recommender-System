"""Shared helpers between all models"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEED = 42
DATA = Path("dataset/prepared")
FAST_FRAC = 0.05


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def out_dir(fast):
    root = Path("results_fast" if fast else "results")
    (root / "preds").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    return root


def load_data(fast=False):
    D = {}
    for s in ("train", "val", "test"):
        D[f"X_{s}"] = np.load(DATA / f"X_{s}.npy")
        D[f"y_{s}"] = np.load(DATA / f"y_{s}.npy")
        D[f"uid_{s}"] = np.load(DATA / f"uid_{s}.npy")
    alt = DATA / "X_train_plainmean.npy"
    D["X_train_plainmean"] = np.load(alt) if alt.exists() else None
    D["feature_names"] = json.load(open(DATA / "feature_names.json"))

    if fast:  # 5% random rows of every split, same train rows for the alt matrix
        rng = np.random.default_rng(SEED)
        for s in ("train", "val", "test"):
            n = len(D[f"y_{s}"])
            idx = np.sort(rng.choice(n, max(int(n * FAST_FRAC), 1000), replace=False))
            for k in ("X", "y", "uid"):
                D[f"{k}_{s}"] = D[f"{k}_{s}"][idx]
            if s == "train" and D["X_train_plainmean"] is not None:
                D["X_train_plainmean"] = D["X_train_plainmean"][idx]
    return D


def ndcg_at_k(uid, y, pred, k=10):
    """Per-user NDCG@k when re-ranking each user's held-out items by predicted rating.
    Gain = 2^rating - 1. Ties in the prediction are broken randomly (seeded)."""
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"u": uid, "y": np.asarray(y, float),
                       "p": np.asarray(pred, float) + rng.normal(0, 1e-9, len(pred))})
    df["gain"] = 2.0 ** df["y"] - 1
    df["r"] = df.groupby("u")["p"].rank(method="first", ascending=False) - 1
    df["ri"] = df.groupby("u")["y"].rank(method="first", ascending=False) - 1
    top = df[df["r"] < k]
    dcg = (top["gain"] / np.log2(top["r"] + 2)).groupby(top["u"]).sum()
    ide = df[df["ri"] < k]
    idcg = (ide["gain"] / np.log2(ide["ri"] + 2)).groupby(ide["u"]).sum()
    return float((dcg.reindex(idcg.index).fillna(0) / idcg).mean())


def evaluate(y, reg, uid, cls=None):
    y = np.asarray(y)
    reg = np.clip(np.asarray(reg, float), 1, 5)
    m = {"rmse": float(np.sqrt(np.mean((reg - y) ** 2))),
         "mae": float(np.mean(np.abs(reg - y))),
         "ndcg@10": ndcg_at_k(uid, y, reg)}
    if cls is not None:
        yi = y.astype(int)
        m.update(accuracy=float(accuracy_score(yi, cls)),
                 f1_macro=float(f1_score(yi, cls, average="macro", zero_division=0)),
                 precision_macro=float(precision_score(yi, cls, average="macro", zero_division=0)),
                 recall_macro=float(recall_score(yi, cls, average="macro", zero_division=0)))
    return m