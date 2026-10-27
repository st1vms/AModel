"""Model definitions: baselines + gradient boosting for 1X2 and goals.

Classifiers (1X2)
-----------------
- ``base_rate``:  train-set class frequencies (naive baseline).
- ``logistic``:   StandardScaler + LogisticRegression (scaled non-tree baseline).
- ``xgboost`` / ``lightgbm`` / ``catboost``: gradient boosting.

Regressors (home & away goals)
------------------------------
- ``mean``:       train-set mean (naive baseline).
- ``poisson``:    StandardScaler + PoissonRegressor (baseline).
- ``xgboost`` / ``lightgbm`` / ``catboost``: gradient boosting (clipped at 0).

Class imbalance (draws ~25%) is handled with inverse-frequency weights:
``sample_weight`` for XGBoost/LightGBM, ``class_weights`` for CatBoost,
``class_weight`` for LogisticRegression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import catboost as cb
import lightgbm as lgb
import xgboost as xgb

from src.ml.data import OUTCOME_AWAY, OUTCOME_DRAW, OUTCOME_HOME

N_CLASSES = 3
RANDOM_STATE = 42

# Light tuning grid (fit on train only; see tune_* below).
TUNING_GRID = {
    "n_estimators": [500, 1000],
    "learning_rate": [0.03, 0.05],
    "max_depth": [4, 6],
    "subsample": [0.8, 1.0],
}
EARLY_STOPPING_ROUNDS = 50


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def inverse_class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights: w_c = N / (n_classes * count_c)."""
    y = np.asarray(y)
    counts = np.bincount(y, minlength=N_CLASSES)
    total = counts.sum()
    return {c: total / (N_CLASSES * n) for c, n in enumerate(counts) if n > 0}


def sample_weights(y: np.ndarray, weights: dict[int, float]) -> np.ndarray:
    """Per-sample weights from a class-weight map."""
    return np.array([weights[int(v)] for v in y], dtype=float)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class BaseRateClassifier:
    """Predicts the training-set class frequencies for every match."""

    def __init__(self) -> None:
        self.probs_: np.ndarray | None = None

    def fit(self, X, y) -> "BaseRateClassifier":
        counts = np.bincount(np.asarray(y), minlength=N_CLASSES)
        self.probs_ = counts / counts.sum()
        return self

    def predict_proba(self, X) -> np.ndarray:
        n = len(X)
        return np.tile(self.probs_, (n, 1))

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class MeanRegressor:
    """Predicts the training-set mean target for every match."""

    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, X, y) -> "MeanRegressor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), self.mean_)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def make_classifier(
    name: str,
    class_weights: dict[int, float] | None = None,
    **params,
):
    """Build an untrained classifier by name.

    ``params`` overrides defaults (used by the tuner). CatBoost receives
    ``class_weights`` in its constructor; XGBoost/LightGBM apply weights as
    ``sample_weight`` at fit time (see ``fit_classifier``).
    """
    if name == "base_rate":
        return BaseRateClassifier()
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, C=1.0, random_state=RANDOM_STATE,
                class_weight=class_weights,
            )),
        ])
    if name == "xgboost":
        p = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=1.0, min_child_weight=3,
                 reg_lambda=1.0, tree_method="hist",
                 objective="multi:softprob", num_class=N_CLASSES,
                 eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=4,
                 early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        p.update(params)
        return xgb.XGBClassifier(**p)
    if name == "lightgbm":
        p = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                 subsample=0.8, subsample_freq=1, colsample_bytree=1.0,
                 min_child_samples=20, reg_lambda=1.0, num_leaves=31,
                 objective="multiclass", num_class=N_CLASSES,
                 random_state=RANDOM_STATE, n_jobs=4, verbose=-1)
        p.update(params)
        return lgb.LGBMClassifier(**p)
    if name == "catboost":
        p = dict(iterations=1000, learning_rate=0.05, depth=6,
                 subsample=0.8, bootstrap_type="Bernoulli", l2_leaf_reg=3.0,
                 loss_function="MultiClass", eval_metric="MultiClass",
                 random_seed=RANDOM_STATE, task_type="CPU",
                 class_weights=class_weights, allow_writing_files=False)
        return cb.CatBoostClassifier(**_catboost_params(p, params))
    raise ValueError(f"unknown classifier: {name!r}")


def make_regressor(name: str, **params):
    """Build an untrained goal regressor by name."""
    if name == "mean":
        return MeanRegressor()
    if name == "poisson":
        return Pipeline([
            ("scale", StandardScaler()),
            ("reg", PoissonRegressor(alpha=1.0, max_iter=1000)),
        ])
    if name == "xgboost":
        p = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                 subsample=0.8, colsample_bytree=1.0, min_child_weight=3,
                 reg_lambda=1.0, tree_method="hist",
                 objective="reg:squarederror", eval_metric="rmse",
                 random_state=RANDOM_STATE, n_jobs=4,
                 early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        p.update(params)
        return xgb.XGBRegressor(**p)
    if name == "lightgbm":
        p = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                 subsample=0.8, subsample_freq=1, colsample_bytree=1.0,
                 min_child_samples=20, reg_lambda=1.0, num_leaves=31,
                 random_state=RANDOM_STATE, n_jobs=4, verbose=-1)
        p.update(params)
        return lgb.LGBMRegressor(**p)
    if name == "catboost":
        p = dict(iterations=1000, learning_rate=0.05, depth=6,
                 subsample=0.8, bootstrap_type="Bernoulli", l2_leaf_reg=3.0,
                 loss_function="RMSE", eval_metric="RMSE",
                 random_seed=RANDOM_STATE, task_type="CPU",
                 allow_writing_files=False)
        return cb.CatBoostRegressor(**_catboost_params(p, params))
    raise ValueError(f"unknown regressor: {name!r}")


def _catboost_params(base: dict, params: dict) -> dict:
    """Merge params into CatBoost defaults, translating sklearn-style names.

    CatBoost rejects having both ``iterations`` and ``n_estimators`` set, and
    uses ``depth`` (not ``max_depth``), so the shared tuning grid's
    ``n_estimators``/``max_depth`` are translated.
    """
    p = dict(base)
    p.update(params)
    if "n_estimators" in p:
        p["iterations"] = p.pop("n_estimators")
    if "max_depth" in p:
        p["depth"] = p.pop("max_depth")
    return p


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_classifier(model, X: pd.DataFrame, y: np.ndarray,
                   class_weights: dict[int, float] | None = None,
                   eval_set=None, early_stopping: bool = False):
    """Fit a classifier, applying class weights per framework.

    ``eval_set`` = (X_eval, y_eval) enables native early stopping for the
    boosting frameworks (tuning only).
    """
    if isinstance(model, (BaseRateClassifier,)):
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        model.fit(X, y)
        return model
    sw = sample_weights(y, class_weights) if class_weights else None
    if isinstance(model, xgb.XGBClassifier):
        if early_stopping and eval_set is not None:
            model.fit(X, y, sample_weight=sw,
                      eval_set=[eval_set],
                      verbose=False)
        else:
            # XGBoost 3.x: early_stopping_rounds in the ctor requires an
            # eval set, so disable it for plain fits.
            model.early_stopping_rounds = None
            model.fit(X, y, sample_weight=sw)
        return model
    if isinstance(model, lgb.LGBMClassifier):
        if early_stopping and eval_set is not None:
            X_eval, y_eval = eval_set
            model.fit(X, y, sample_weight=sw,
                      eval_X=X_eval, eval_y=y_eval,
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                    verbose=False)])
        else:
            model.fit(X, y, sample_weight=sw)
        return model
    if isinstance(model, cb.CatBoostClassifier):
        if early_stopping and eval_set is not None:
            model.fit(X, y, eval_set=eval_set,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      verbose=False)
        else:
            model.fit(X, y, verbose=False)
        return model
    raise TypeError(f"unsupported classifier type: {type(model)!r}")


def fit_regressor(model, X: pd.DataFrame, y: np.ndarray,
                  eval_set=None, early_stopping: bool = False):
    """Fit a goal regressor (early stopping for boosting frameworks)."""
    if isinstance(model, MeanRegressor):
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        model.fit(X, y)
        return model
    if isinstance(model, xgb.XGBRegressor):
        if early_stopping and eval_set is not None:
            model.fit(X, y, eval_set=[eval_set], verbose=False)
        else:
            # XGBoost 3.x: early_stopping_rounds in the ctor requires an
            # eval set, so disable it for plain fits.
            model.early_stopping_rounds = None
            model.fit(X, y)
        return model
    if isinstance(model, lgb.LGBMRegressor):
        if early_stopping and eval_set is not None:
            X_eval, y_eval = eval_set
            model.fit(X, y, eval_X=X_eval, eval_y=y_eval,
                      callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS,
                                                    verbose=False)])
        else:
            model.fit(X, y)
        return model
    if isinstance(model, cb.CatBoostRegressor):
        if early_stopping and eval_set is not None:
            model.fit(X, y, eval_set=eval_set,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                      verbose=False)
        else:
            model.fit(X, y, verbose=False)
        return model
    raise TypeError(f"unsupported regressor type: {type(model)!r}")


def predict_goals(model, X: pd.DataFrame) -> np.ndarray:
    """Regressor predictions clipped at 0 (goals are non-negative)."""
    pred = np.asarray(model.predict(X), dtype=float)
    return np.clip(pred, 0.0, None)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(fitted_model, X_cal: pd.DataFrame, y_cal: np.ndarray):
    """Isotonic calibration of a fitted classifier on a held-out slice.

    The base model is the one already fitted on the main train (wrapped in
    ``FrozenEstimator`` — the sklearn 1.9+ replacement for ``cv="prefit"``);
    only the isotonic mapping is learned on ``X_cal``.
    """
    from sklearn.frozen import FrozenEstimator
    return CalibratedClassifierCV(FrozenEstimator(fitted_model),
                                  method="isotonic").fit(X_cal, y_cal)


# ---------------------------------------------------------------------------
# Light tuning
# ---------------------------------------------------------------------------

def _grid_combos() -> list[dict]:
    """Cartesian product of TUNING_GRID (16 combos)."""
    keys = list(TUNING_GRID)
    combos = [{}]
    for k in keys:
        combos = [{**c, k: v} for c in combos for v in TUNING_GRID[k]]
    return combos


def tune_classifier(name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                    X_eval: pd.DataFrame, y_eval: np.ndarray,
                    class_weights: dict[int, float] | None = None) -> dict:
    """Grid-search TUNING_GRID with early stopping on the eval slice.

    Returns the best params (with ``n_estimators`` set to the best
    early-stopped iteration count).
    """
    best: dict | None = None
    best_score = np.inf
    for combo in _grid_combos():
        model = make_classifier(name, class_weights=class_weights, **combo)
        fit_classifier(model, X_train, y_train, class_weights=class_weights,
                       eval_set=(X_eval, y_eval), early_stopping=True)
        proba = model.predict_proba(X_eval)
        score = log_loss(y_eval, proba, labels=[0, 1, 2])
        if score < best_score:
            best_score = score
            iters = _best_iterations(model)
            best = {**combo, "n_estimators": iters}
    if best is None:
        raise RuntimeError(f"tuning produced no result for {name!r}")
    return best


def tune_regressor(name: str, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_eval: pd.DataFrame, y_eval: np.ndarray) -> dict:
    """Grid-search TUNING_GRID with early stopping; score by MAE."""
    best: dict | None = None
    best_score = np.inf
    for combo in _grid_combos():
        model = make_regressor(name, **combo)
        fit_regressor(model, X_train, y_train,
                      eval_set=(X_eval, y_eval), early_stopping=True)
        score = mean_absolute_error(y_eval, predict_goals(model, X_eval))
        if score < best_score:
            best_score = score
            iters = _best_iterations(model)
            best = {**combo, "n_estimators": iters}
    if best is None:
        raise RuntimeError(f"tuning produced no result for {name!r}")
    return best


def _best_iterations(model) -> int:
    """Number of trees actually used (early-stopped) by a fitted booster.

    XGBoost ``best_iteration`` is 0-based (-1 if early stopping never
    triggered); LightGBM ``best_iteration_`` is already a 1-based tree count;
    CatBoost ``get_best_iteration()`` is 0-based.
    """
    if isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
        bi = int(model.best_iteration)
        return bi + 1 if bi >= 0 else int(model.n_estimators)
    if isinstance(model, (lgb.LGBMClassifier, lgb.LGBMRegressor)):
        return int(model.best_iteration_)
    if isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRegressor)):
        return int(model.get_best_iteration()) + 1
    raise TypeError(f"no iteration info for {type(model)!r}")
