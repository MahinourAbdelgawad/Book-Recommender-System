import time
from functools import partial

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (BaggingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .common import SEED, log

N_JOBS = 6  
SITE_WIDE = ["book_ratings_count", "book_text_reviews_count", "author_avg_rating",
             "author_ratings_count", "author_text_reviews_count"]


def _predict(model, X, kind):
    """Returns (rating-scale prediction, class prediction or None)."""
    if kind == "reg":
        return model.predict(X), None
    cls = model.predict(X)
    if hasattr(model, "predict_proba"):  # expected rating = sum_k p_k * k
        return model.predict_proba(X) @ model.classes_.astype(float), cls
    return cls.astype(float), cls


def _fit_eval(D, make, grid, kind, keep=None, drop=None, n_train=None, train_key="X_train"):
    Xtr, ytr = D[train_key], D["y_train"]
    if Xtr is None:
        raise RuntimeError(f"{train_key} missing - run prepare_data.py (plain-mean variant failed?)")
    Xva, Xte = D["X_val"], D["X_test"]
    names = D["feature_names"]
    if keep is not None or drop is not None:
        cols = [i for i, n in enumerate(names) if (keep is None or n in keep) and (drop is None or n not in drop)]
        Xtr, Xva, Xte = Xtr[:, cols], Xva[:, cols], Xte[:, cols]
    if n_train and len(ytr) > n_train:
        idx = np.random.default_rng(SEED).choice(len(ytr), n_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    yfit = ytr.astype(int) if kind == "cls" else ytr.astype(float)

    best, trials = None, []
    for cfg in grid:
        log(f"fitting {cfg} on {Xtr.shape}")
        model = make(**cfg)
        t = time.time(); model.fit(Xtr, yfit); fit_s = time.time() - t
        log(f"  fit done in {fit_s:.0f}s, predicting val")
        t = time.time(); reg_v, cls_v = _predict(model, Xva, kind); pred_s = time.time() - t
        rmse = float(np.sqrt(np.mean((np.clip(reg_v, 1, 5) - D["y_val"]) ** 2)))
        log(f"  val rmse {rmse:.4f} (predict {pred_s:.0f}s)")
        trials.append({"config": cfg, "val_rmse": rmse, "fit_s": fit_s, "predict_val_s": pred_s})
        if best is None or rmse < best["rmse"]:
            best = dict(model=model, cfg=cfg, rmse=rmse, reg_v=reg_v, cls_v=cls_v, fit_s=fit_s, pred_s=pred_s)
    log(f"best config {best['cfg']}, predicting test")
    reg_t, cls_t = _predict(best["model"], Xte, kind)
    return {"params": {"selected": best["cfg"], "trials": trials,
                       "n_train": int(len(ytr)), "n_features": int(Xtr.shape[1])},
            "fit_time_s": best["fit_s"], "predict_time_s": best["pred_s"],
            "val": (best["reg_v"], best["cls_v"]), "test": (reg_t, cls_t)}


MODELS = {}


def add(name, family, kind, make, grid=({},), **kw):
    MODELS[name] = (family, kind, partial(_fit_eval, make=make, grid=list(grid), kind=kind, **kw))


#1) BASELINES AND LINEAR REGRESSION AND LOGISTIC REGRESSION AND NAIVE BAYES AND SVM
add("baseline_global_mean", "baseline", "reg", DummyRegressor)
add("lin_user_book_mean", "linear", "reg", LinearRegression, keep=["u_mean_rating", "b_mean_rating"])
add("linreg", "linear", "reg", LinearRegression)
add("ridge", "linear", "reg", Ridge, grid=[{"alpha": a} for a in (1, 10, 100, 1000)])
add("gnb", "naive_bayes", "cls", GaussianNB)  # Gaussian assumption is wrong for one-hots; baseline only
add("perceptron", "perceptron", "cls", partial(Perceptron, max_iter=50, tol=1e-3, random_state=SEED))
add("logreg", "logistic", "cls", partial(LogisticRegression, max_iter=300),
    grid=[{"C": c} for c in (0.1, 1.0)])
add("linsvr", "svm", "reg",
    partial(LinearSVR, loss="squared_epsilon_insensitive", dual=False, epsilon=0.0, max_iter=2000, random_state=SEED),
    grid=[{"C": c} for c in (0.01, 0.1)])
add("linsvc", "svm", "cls", partial(LinearSVC, dual=False, max_iter=1000, random_state=SEED),
    grid=[{"C": c} for c in (0.1,)])

# ========================================
# 2 DECISION TREE
add("dtree_reg", "tree", "reg", partial(DecisionTreeRegressor, random_state=SEED),
    grid=[{"max_depth": d, "min_samples_leaf": l} for d in (6, 10, 14) for l in (50, 200)])
add("dtree_clf", "tree", "cls", partial(DecisionTreeClassifier, random_state=SEED),
    grid=[{"max_depth": d, "min_samples_leaf": 100} for d in (6, 10)])

# ---------- leakage ablations (same fixed HGB config, only the feature set changes) ----------
_hgb_fixed = partial(HistGradientBoostingRegressor, max_iter=400, learning_rate=0.1, max_leaf_nodes=31,
                     early_stopping=True, n_iter_no_change=20, random_state=SEED)
add("abl_hgb_full", "ablation", "reg", _hgb_fixed)
add("abl_hgb_nosite", "ablation", "reg", _hgb_fixed, drop=SITE_WIDE)
add("abl_hgb_plainmean", "ablation", "reg", _hgb_fixed, train_key="X_train_plainmean")  # no leave-one-out on train

# ===================================================
# 3) BOOSTING, BAGGING, AND RANDOM FOREST
add("hgb_reg", "boosting", "reg",
    partial(HistGradientBoostingRegressor, max_iter=600, early_stopping=True, n_iter_no_change=25, random_state=SEED),
    grid=[{"learning_rate": lr, "max_leaf_nodes": n} for lr in (0.05, 0.1) for n in (31, 63)])
add("hgb_clf", "boosting", "cls",
    partial(HistGradientBoostingClassifier, max_iter=300, learning_rate=0.1, early_stopping=True,
            n_iter_no_change=15, random_state=SEED))
add("bagging_dt", "bagging", "reg",
    partial(BaggingRegressor, estimator=DecisionTreeRegressor(max_depth=15, min_samples_leaf=20),
            n_estimators=30, max_samples=0.5, n_jobs=N_JOBS, random_state=SEED))
add("rf_reg", "random_forest", "reg",
    partial(RandomForestRegressor, n_estimators=200, min_samples_leaf=20, max_features=0.33,
            n_jobs=N_JOBS, random_state=SEED))
add("rf_clf", "random_forest", "cls",
    partial(RandomForestClassifier, n_estimators=150, min_samples_leaf=20, max_features="sqrt",
            n_jobs=N_JOBS, random_state=SEED))

# ==============================================
# KNN AND RBF SVM
add("knn_reg", "knn", "reg",
    partial(KNeighborsRegressor, weights="distance", algorithm="brute", n_jobs=N_JOBS),
    grid=[{"n_neighbors": k} for k in (20, 50)], n_train=150_000)
add("svr_rbf", "svm", "reg", partial(SVR, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale", cache_size=2000),
    n_train=30_000)