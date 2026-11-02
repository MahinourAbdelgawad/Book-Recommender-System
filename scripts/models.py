import time
from functools import partial
 
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, BaggingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
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


def make_adaboost_regressor(max_depth, n_estimators=50, learning_rate=1.0):
    return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=max_depth),
                             n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED)


def make_adaboost_classifier(max_depth, n_estimators=50, learning_rate=1.0):
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth),
                              n_estimators=n_estimators, learning_rate=learning_rate, random_state=SEED)


MODELS = {}


def add(name, family, kind, make, grid=({},), **kw):
    MODELS[name] = (family, kind, partial(_fit_eval, make=make, grid=list(grid), kind=kind, **kw))


_random_forest = partial(RandomForestRegressor, n_estimators=200, min_samples_leaf=20, max_features=0.33,
                         n_jobs=N_JOBS, random_state=SEED)


#1) BASELINES AND LINEARS
add("baseline_global_mean", "baseline", "reg", DummyRegressor)

add("linear_regression_user_and_book_mean_only", "linear", "reg", LinearRegression,
    keep=["u_mean_rating", "b_mean_rating"])

add("linear_regression", "linear", "reg", LinearRegression)

add("ablation_linear_regression_no_site_stats", "ablation", "reg", LinearRegression, drop=SITE_WIDE)

add("ablation_linear_regression_plain_means", "ablation", "reg", LinearRegression,
    train_key="X_train_plainmean")  

add("naive_bayes_gaussian", "naive_bayes", "cls", GaussianNB) 

add("perceptron", "perceptron", "cls", partial(Perceptron, max_iter=50, tol=1e-3, random_state=SEED))

add("logistic_regression", "logistic", "cls", partial(LogisticRegression, max_iter=300),
    grid=[{"C": c} for c in (0.01, 0.1, 1.0)])

add("svm_linear_regressor", "svm", "reg",
    partial(LinearSVR, loss="squared_epsilon_insensitive", dual=False, epsilon=0.0, max_iter=2000, random_state=SEED),
    grid=[{"C": c} for c in (0.01, 0.1)])

add("svm_linear_classifier", "svm", "cls", partial(LinearSVC, dual=False, max_iter=1000, random_state=SEED),
    grid=[{"C": c} for c in (0.1,)])

# ========================================
# 2 TREES AND ENSEMBLES
add("decision_tree_regressor", "tree", "reg", partial(DecisionTreeRegressor, random_state=SEED),
    grid=[{"max_depth": d, "min_samples_leaf": l} for d in (3, 4, 6, 10) for l in (50, 200)])

add("decision_tree_classifier", "tree", "cls", partial(DecisionTreeClassifier, random_state=SEED),
    grid=[{"max_depth": d, "min_samples_leaf": 100} for d in (6, 10)])

add("bagging_decision_trees_regressor", "bagging", "reg",
    partial(BaggingRegressor, estimator=DecisionTreeRegressor(max_depth=15, min_samples_leaf=20),
            n_estimators=30, max_samples=0.5, n_jobs=N_JOBS, random_state=SEED))

add("adaboost_regressor", "boosting", "reg", make_adaboost_regressor,
    grid=[{"max_depth": d} for d in (3, 5)])

add("adaboost_classifier", "boosting", "cls", make_adaboost_classifier,
    grid=[{"max_depth": d} for d in (3, 5)])

add("random_forest_regressor", "random_forest", "reg", _random_forest)

add("ablation_random_forest_no_site_stats", "ablation", "reg", _random_forest, drop=SITE_WIDE)

add("ablation_random_forest_plain_means", "ablation", "reg", _random_forest, train_key="X_train_plainmean")

add("random_forest_classifier", "random_forest", "cls",
    partial(RandomForestClassifier, n_estimators=150, min_samples_leaf=20, max_features="sqrt",
            n_jobs=N_JOBS, random_state=SEED))


# ==============================================
# KNN AND RBF SVM
add("knn_regressor", "knn", "reg",
    partial(KNeighborsRegressor, weights="distance", algorithm="brute", n_jobs=N_JOBS),
    grid=[{"n_neighbors": k} for k in (20, 50)])


add("svm_rbf_kernel_regressor", "svm", "reg",
    partial(SVR, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale", cache_size=2000, verbose=True))