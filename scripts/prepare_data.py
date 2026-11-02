"""
Prepare the data into model ready data

Usage:  python prepare_data.py path/to/final_dataset.parquet  

Writes dataset/: X_{train,val,test}.npy (float32), y_*.npy, uid_*.npy, bid_*.npy,
feature_names.json, audit.json, and X_train_plainmean.npy (leakage ablation, see below).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS, LABEL, SPLIT = ["user_id", "book_id"], "rating", "split"

from common import DATA
out = DATA
out.mkdir(parents=True, exist_ok=True)

path = sys.argv[1]
df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
df[SPLIT] = df[SPLIT].astype(str).str.lower().replace({"validation": "val", "valid": "val"})
print("split values:", df[SPLIT].value_counts().to_dict())
assert set(df[SPLIT].unique()) == {"train", "val", "test"}, "unexpected split labels"

feats = [c for c in df.columns if c not in ID_COLS + [LABEL, SPLIT]]
assert not df[feats].isna().any().any(), "NaNs in features"
assert df[LABEL].between(1, 5).all(), "rating outside 1..5"
df["uid"] = pd.factorize(df["user_id"])[0].astype(np.int32)
df["bid"] = pd.factorize(df["book_id"])[0].astype(np.int32)
parts = {s: df[df[SPLIT] == s] for s in ("train", "val", "test")}

tr_books = set(parts["train"]["bid"])
audit = {"n_features": len(feats), "n_rows": {s: len(p) for s, p in parts.items()},
         "n_users": {s: int(p["uid"].nunique()) for s, p in parts.items()},
         "n_books": {s: int(p["bid"].nunique()) for s, p in parts.items()},
         "label_distribution_train": parts["train"][LABEL].value_counts(normalize=True).sort_index().round(4).to_dict(),
         "constant_features": [c for c in feats if parts["train"][c].nunique() <= 1]}

for s in ("val", "test"):  # cold-start check: books with NO training ratings
    unseen = ~parts[s]["bid"].isin(tr_books)
    audit[f"{s}_rows_with_book_unseen_in_train"] = int(unseen.sum())
    audit[f"{s}_users_unseen_in_train"] = int((~parts[s]["uid"].isin(set(parts["train"]["uid"]))).sum())

print(json.dumps(audit, indent=2))
json.dump(audit, open(out / "audit.json", "w"), indent=2)
if audit["val_rows_with_book_unseen_in_train"] or audit["test_rows_with_book_unseen_in_train"]:
    print("\n!! Some val/test books have no training ratings: they need a global-mean fallback + is_new_book flag.")

for s, p in parts.items():
    np.save(out / f"X_{s}.npy", p[feats].to_numpy(np.float32))
    np.save(out / f"y_{s}.npy", p[LABEL].to_numpy(np.int8))
    np.save(out / f"uid_{s}.npy", p["uid"].to_numpy())
    np.save(out / f"bid_{s}.npy", p["bid"].to_numpy())
json.dump(feats, open(out / "feature_names.json", "w"))


# ---------------- plain-mean ablation matrix ----------------
# Train rows currently carry LEAVE-ONE-OUT means (mean of the user's/book's OTHER ratings).
# Within one user that feature is a deterministic decreasing function of the row's own rating,
# which flexible models can decode.  Val/test rows carry the plain full-train mean.
# Here we rebuild the train column as the plain train mean (same value on every row of the group),
# mapped into the already-standardised space by an affine fit against the val rows.
def plain_group_mean(key, col):
    tr, va = parts["train"], parts["val"]
    raw = tr.groupby(key)[LABEL].mean()
    std_val = va.groupby(key)[col].first()
    common = raw.index.intersection(std_val.index)
    a, b = np.polyfit(raw[common].to_numpy(), std_val[common].to_numpy(), 1)
    resid = np.abs(a * raw[common].to_numpy() + b - std_val[common].to_numpy()).max()
    return a, b, raw, resid


Xp = parts["train"][feats].to_numpy(np.float32).copy()
ok = True
for key, col in (("uid", "u_mean_rating"), ("bid", "b_mean_rating")):
    if col not in feats:
        print(f"{col} not found, skipping plain-mean variant"); ok = False; break
    a, b, raw, resid = plain_group_mean(key, col)
    print(f"plain-mean {col}: affine fit max residual {resid:.2e}")
    if resid > 1e-3:
        print(f"!! {col}: val values are not an affine map of the plain train mean; skipping variant"); ok = False; break
    Xp[:, feats.index(col)] = (parts["train"][key].map(raw).to_numpy() * a + b).astype(np.float32)
if ok:
    np.save(out / "X_train_plainmean.npy", Xp)
print("\nwrote", sorted(p.name for p in out.iterdir()))