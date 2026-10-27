"""Training orchestrator: load -> split -> tune -> train -> evaluate -> save.

Usage
-----
    python -m src.ml.train [--include-ht] [--no-xg] [--walk-forward]
                           [--skip-tuning] [--out-dir PATH]

Single split (default): train on seasons <= 2023-24, validate on
2024-25..2026-27 (metrics reported with and without the partial 2026-27).
``--walk-forward`` adds an expanding-window evaluation over 2022-23..2025-26.

Outputs (under --out-dir, default data/export/ml):
    results/*.csv   comparison tables (single split + walk-forward)
    results/*.png   confusion matrix, feature importance, calibration
    models/*.joblib best classifier + best home/away goal regressors
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from src.ml import data, evaluate, models, plots
from src.ml.splits import (
    DEFAULT_CUTOFF,
    Split,
    time_split,
    walk_forward_splits,
)

CLASSIFIERS = ["base_rate", "logistic", "xgboost", "lightgbm", "catboost"]
REGRESSORS = ["mean", "poisson", "xgboost", "lightgbm", "catboost"]
BOOSTING = ["xgboost", "lightgbm", "catboost"]
CAT_FEATURES = ["home_team", "away_team"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _X(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return df[features]


def _y(df: pd.DataFrame, col: str) -> np.ndarray:
    return df[col].to_numpy()


def _valid_slice(valid: pd.DataFrame, exclude_in_progress: bool,
                 league: str = config.DEFAULT_LEAGUE) -> pd.DataFrame:
    if exclude_in_progress:
        return valid[valid["season"] != config.LEAGUES[league]["in_progress"]]
    return valid


def _tuning_slice(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the train frame into (train_minus_last, last_season) for tuning.

    The last season present in ``train`` (2023-24 for the default cutoff) is
    held out as the early-stopping / calibration slice.
    """
    last = train["season"].max()
    return train[train["season"] != last], train[train["season"] == last]


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_all(
    train: pd.DataFrame,
    features: list[str],
) -> dict[str, dict]:
    """Light grid search for each boosting model (classifier + 2 regressors).

    Returns {model_key: best_params} where model_key is one of
    ``clf:<name>``, ``reg_home:<name>``, ``reg_away:<name>``.
    """
    train_main, train_eval = _tuning_slice(train)
    X_main, X_eval = _X(train_main, features), _X(train_eval, features)
    y_out = _y(train_main, "outcome")
    y_eval_out = _y(train_eval, "outcome")
    cw = models.inverse_class_weights(y_out)

    tuned: dict[str, dict] = {}
    for name in BOOSTING:
        t0 = time.time()
        tuned[f"clf:{name}"] = models.tune_classifier(
            name, X_main, y_out, X_eval, y_eval_out, class_weights=cw)
        print(f"  tuned clf:{name} in {time.time() - t0:.1f}s -> "
              f"{tuned[f'clf:{name}']}")
    for target in ("home_goals", "away_goals"):
        y_main = _y(train_main, target)
        y_eval_t = _y(train_eval, target)
        for name in BOOSTING:
            t0 = time.time()
            tuned[f"reg_{target}:{name}"] = models.tune_regressor(
                name, X_main, y_main, X_eval, y_eval_t)
            print(f"  tuned reg_{target}:{name} in {time.time() - t0:.1f}s -> "
                  f"{tuned[f'reg_{target}:{name}']}")
    return tuned


# ---------------------------------------------------------------------------
# Single-split training & evaluation
# ---------------------------------------------------------------------------

def run_classifiers_single_split(
    split: Split,
    features: list[str],
    tuned: dict[str, dict],
    league: str = config.DEFAULT_LEAGUE,
) -> tuple[list[dict], dict[str, object]]:
    """Train & evaluate all classifiers on the single split.

    Returns (results, artifacts) where artifacts holds the fitted models and
    the best model's predictions for plotting.
    """
    train, valid = split.train, split.valid
    X_train, X_valid = _X(train, features), _X(valid, features)
    y_train = _y(train, "outcome")
    cw = models.inverse_class_weights(y_train)

    results: list[dict] = []
    fitted: dict[str, object] = {}
    for name in CLASSIFIERS:
        params = tuned.get(f"clf:{name}", {})
        model = models.make_classifier(name, class_weights=cw, **params)
        models.fit_classifier(model, X_train, y_train, class_weights=cw)
        fitted[name] = model
        proba = model.predict_proba(X_valid)
        for tag, vdf in (("all", valid),
                         ("no_in_progress", _valid_slice(valid, True, league))):
            m = evaluate.classification_metrics(
                _y(vdf, "outcome"), model.predict_proba(_X(vdf, features)))
            results.append({
                "model": name, "split": "single", "valid": tag,
                **{k: v for k, v in m.items() if k != "confusion_matrix"},
            })

    # Best model by log loss on the full validation set.
    full = [r for r in results if r["valid"] == "all"]
    best_name = min(full, key=lambda r: r["log_loss"])["model"]
    best = fitted[best_name]
    cm = evaluate.classification_metrics(
        _y(valid, "outcome"), best.predict_proba(X_valid))["confusion_matrix"]
    for r in results:
        if r["model"] == best_name and r["valid"] == "all":
            r["confusion_matrix"] = cm

    artifacts = {
        "fitted": fitted,
        "best_name": best_name,
        "best": best,
        "best_cm": cm,
        "best_proba_valid": best.predict_proba(X_valid),
        "y_valid": _y(valid, "outcome"),
        "class_weights": cw,
    }
    return results, artifacts


def run_regressors_single_split(
    split: Split,
    features: list[str],
    tuned: dict[str, dict],
    league: str = config.DEFAULT_LEAGUE,
) -> tuple[list[dict], dict[str, object]]:
    """Train & evaluate all goal regressors on the single split."""
    train, valid = split.train, split.valid
    X_train, X_valid = _X(train, features), _X(valid, features)

    results: list[dict] = []
    fitted: dict[str, object] = {}
    for target in ("home_goals", "away_goals"):
        y_train = _y(train, target)
        for name in REGRESSORS:
            params = tuned.get(f"reg_{target}:{name}", {})
            model = models.make_regressor(name, **params)
            models.fit_regressor(model, X_train, y_train)
            fitted[f"{target}:{name}"] = model
            for tag, vdf in (("all", valid),
                             ("no_in_progress", _valid_slice(valid, True, league))):
                pred = models.predict_goals(model, _X(vdf, features))
                m = evaluate.regression_metrics(_y(vdf, target), pred)
                results.append({
                    "model": name, "target": target, "split": "single",
                    "valid": tag, **m,
                })

    # Best per target by MAE on the full validation set.
    best: dict[str, str] = {}
    for target in ("home_goals", "away_goals"):
        rows = [r for r in results
                if r["target"] == target and r["valid"] == "all"]
        best[target] = min(rows, key=lambda r: r["mae"])["model"]

    artifacts = {"fitted": fitted, "best": best}
    return results, artifacts


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_walk_forward(
    splits: list[Split],
    features: list[str],
    tuned: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Expanding-window evaluation for all classifiers & regressors."""
    clf_rows: list[dict] = []
    reg_rows: list[dict] = []
    for sp in splits:
        X_train, X_valid = _X(sp.train, features), _X(sp.valid, features)
        y_train = _y(sp.train, "outcome")
        cw = models.inverse_class_weights(y_train)
        for name in CLASSIFIERS:
            params = tuned.get(f"clf:{name}", {})
            model = models.make_classifier(name, class_weights=cw, **params)
            models.fit_classifier(model, X_train, y_train, class_weights=cw)
            m = evaluate.classification_metrics(
                _y(sp.valid, "outcome"), model.predict_proba(X_valid))
            clf_rows.append({
                "model": name, "fold": sp.name,
                **{k: v for k, v in m.items() if k != "confusion_matrix"},
            })
        for target in ("home_goals", "away_goals"):
            y_tr = _y(sp.train, target)
            for name in REGRESSORS:
                params = tuned.get(f"reg_{target}:{name}", {})
                model = models.make_regressor(name, **params)
                models.fit_regressor(model, X_train, y_tr)
                pred = models.predict_goals(model, X_valid)
                m = evaluate.regression_metrics(_y(sp.valid, target), pred)
                reg_rows.append({
                    "model": name, "target": target, "fold": sp.name, **m,
                })
    return clf_rows, reg_rows


# ---------------------------------------------------------------------------
# Calibration (best classifier)
# ---------------------------------------------------------------------------

def run_calibration(
    split: Split,
    features: list[str],
    best_name: str,
) -> dict | None:
    """Isotonic calibration of the best classifier on the 2023-24 slice.

    The base model is retrained on train minus the last season; the isotonic
    mapping is learned on the last season. Returns a result row (or None if
    the best model is not calibratable, e.g. base_rate).
    """
    train, valid = split.train, split.valid
    train_main, train_cal = _tuning_slice(train)
    X_main = _X(train_main, features)
    X_cal = _X(train_cal, features)
    y_main = _y(train_main, "outcome")
    y_cal = _y(train_cal, "outcome")
    cw = models.inverse_class_weights(y_main)

    base = models.make_classifier(best_name, class_weights=cw)
    models.fit_classifier(base, X_main, y_main, class_weights=cw)
    calibrated = models.calibrate(base, X_cal, y_cal)
    m = evaluate.classification_metrics(
        _y(valid, "outcome"), calibrated.predict_proba(_X(valid, features)))
    return {
        "model": f"{best_name}+isotonic", "split": "single", "valid": "all",
        **{k: v for k, v in m.items() if k != "confusion_matrix"},
    }


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_models(
    models_dir: Path,
    clf_artifacts: dict,
    reg_artifacts: dict,
    features: list[str],
    labels: dict[int, str],
    tuned: dict[str, dict],
    include_ht: bool,
    no_xg: bool,
    league: str = config.DEFAULT_LEAGUE,
) -> dict[str, Path]:
    """Persist the best models + metadata as joblib artifacts."""
    models_dir.mkdir(parents=True, exist_ok=True)
    best_name = clf_artifacts["best_name"]
    best_clf = clf_artifacts["best"]
    best_home = reg_artifacts["fitted"][f"home_goals:{reg_artifacts['best']['home_goals']}"]
    best_away = reg_artifacts["fitted"][f"away_goals:{reg_artifacts['best']['away_goals']}"]

    pipeline = {
        "league": league,
        "classifier": best_clf,
        "classifier_name": best_name,
        "regressor_home": best_home,
        "regressor_home_name": reg_artifacts["best"]["home_goals"],
        "regressor_away": best_away,
        "regressor_away_name": reg_artifacts["best"]["away_goals"],
        "features": features,
        "team_labels": labels,
        "class_weights": clf_artifacts["class_weights"],
        "tuned_params": tuned,
        "include_ht": include_ht,
        "no_xg": no_xg,
        "saved_at": pd.Timestamp.now().isoformat(),
    }
    path = models_dir / "pipeline.joblib"
    joblib.dump(pipeline, path)
    return {"pipeline": path}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_training(
    league: str = config.DEFAULT_LEAGUE,
    include_ht: bool = False,
    no_xg: bool = False,
    walk_forward: bool = False,
    skip_tuning: bool = False,
    out_dir: str | Path | None = None,
    log=print,
) -> dict:
    """Run the full training pipeline and return structured results.

    Same flow as the CLI (load -> split -> tune -> train -> evaluate ->
    plots -> save), but returns a dict instead of only printing, so the
    Streamlit UI (``src/ui/app.py``) can render the results. ``log`` is a
    callable used for progress messages (default: ``print``).

    Returns
    -------
    dict with keys:
      league                        league key trained on
      clf_frame, reg_frame          single-split results (DataFrames)
      wf_clf_frame, wf_reg_frame    walk-forward per-fold results (or None)
      clf_agg_frame, reg_agg_frame  walk-forward aggregates (or None)
      best_name                     best classifier (by validation log loss)
      results_dir, models_dir       output directories (Path)
      saved                         {artifact: path} from save_models
      elapsed                       wall-clock seconds
      n_rows, n_teams, n_features   data summary
      season_counts                 {season: n_matches}
      split_info                    train/valid sizes and season ranges
    """
    out_dir = Path(out_dir or config.ml_dir(league))
    results_dir = out_dir / "results"
    models_dir = out_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log(f"Loading data for {config.LEAGUES[league]['name']} "
        f"(include_ht={include_ht}, no_xg={no_xg}) ...")
    df, labels, features = data.load_data(
        league, include_ht=include_ht, no_xg=no_xg)
    log(f"  {len(df)} rows, {len(labels)} teams, {len(features)} features")
    season_counts = data.season_counts(df)
    log(f"  seasons: {season_counts}")

    split = time_split(df, cutoff=DEFAULT_CUTOFF)
    split_info = {
        "train_rows": len(split.train),
        "valid_rows": len(split.valid),
        "train_seasons": f"{split.train['season'].min()}..{split.train['season'].max()}",
        "valid_seasons": f"{split.valid['season'].min()}..{split.valid['season'].max()}",
    }
    log(f"\nSingle split: train {split_info['train_rows']} "
        f"({split_info['train_seasons']}), "
        f"valid {split_info['valid_rows']} ({split_info['valid_seasons']})")

    # --- tuning ---
    tuned: dict[str, dict] = {}
    if not skip_tuning:
        log("\nTuning (light grid, early stopping on 2023-24) ...")
        tuned = tune_all(split.train, features)

    # --- single split ---
    log("\nTraining classifiers (single split) ...")
    clf_results, clf_art = run_classifiers_single_split(
        split, features, tuned, league)
    log("Training regressors (single split) ...")
    reg_results, reg_art = run_regressors_single_split(
        split, features, tuned, league)

    # --- calibration of the best classifier ---
    cal_row = run_calibration(split, features, clf_art["best_name"])
    if cal_row:
        clf_results.append(cal_row)

    clf_frame = evaluate.results_to_frame(clf_results)
    reg_frame = evaluate.results_to_frame(reg_results)

    log("\n=== Classification (single split) ===")
    log(evaluate.table_to_string(clf_frame))
    log("\n=== Goal regression (single split) ===")
    log(evaluate.table_to_string(reg_frame))

    clf_frame.to_csv(results_dir / "classification_single_split.csv", index=False)
    reg_frame.to_csv(results_dir / "regression_single_split.csv", index=False)

    # --- walk-forward ---
    wf_clf_frame = wf_reg_frame = None
    clf_agg_frame = reg_agg_frame = None
    if walk_forward:
        log("\nWalk-forward (expanding window) ...")
        wf_splits = walk_forward_splits(df)
        wf_clf_rows, wf_reg_rows = run_walk_forward(wf_splits, features, tuned)
        wf_clf_frame = pd.DataFrame(wf_clf_rows)
        wf_reg_frame = pd.DataFrame(wf_reg_rows)
        log("\n=== Classification (walk-forward, per fold) ===")
        log(evaluate.table_to_string(wf_clf_frame))
        log("\n=== Goal regression (walk-forward, per fold) ===")
        log(evaluate.table_to_string(wf_reg_frame))
        wf_clf_frame.to_csv(results_dir / "classification_walkforward.csv",
                            index=False)
        wf_reg_frame.to_csv(results_dir / "regression_walkforward.csv",
                            index=False)

        # Aggregated mean +/- std per model.
        for frame, kind in ((wf_clf_frame, "clf"), (wf_reg_frame, "reg")):
            agg_rows = []
            keys = (["model"] if kind == "clf" else ["model", "target"])
            for key, grp in frame.groupby(keys, sort=True):
                if not isinstance(key, tuple):
                    key = (key,)
                agg = evaluate.aggregate_folds(grp.to_dict("records"))
                row = dict(zip(keys, key))
                for metric, ms in agg.items():
                    row[f"{metric}_mean"] = ms["mean"]
                    row[f"{metric}_std"] = ms["std"]
                agg_rows.append(row)
            agg_frame = pd.DataFrame(agg_rows)
            log(f"\n=== {kind} walk-forward aggregate (mean +/- std) ===")
            log(evaluate.table_to_string(agg_frame))
            agg_frame.to_csv(
                results_dir / f"{kind}_walkforward_aggregate.csv", index=False)
            if kind == "clf":
                clf_agg_frame = agg_frame
            else:
                reg_agg_frame = agg_frame

    # --- plots (best classifier, full validation set) ---
    log("\nSaving plots ...")
    plots.plot_confusion_matrix(
        clf_art["best_cm"],
        results_dir / "confusion_matrix.png",
        title=f"Confusion matrix — {clf_art['best_name']} (validation)",
    )
    try:
        plots.plot_feature_importance(
            clf_art["best"], features,
            results_dir / "feature_importance.png",
            title=f"Feature importance — {clf_art['best_name']}",
        )
    except TypeError as e:
        log(f"  (no feature importance plot: {e})")
    plots.plot_calibration(
        clf_art["y_valid"], clf_art["best_proba_valid"],
        results_dir / "calibration_curve.png",
        title=f"Calibration — {clf_art['best_name']} (validation)",
    )

    # --- save models ---
    saved = save_models(models_dir, clf_art, reg_art, features, labels,
                        tuned, include_ht, no_xg, league)
    for k, p in saved.items():
        log(f"  saved {k}: {p}")

    elapsed = time.time() - t0
    log(f"\nDone in {elapsed:.1f}s. Results in {results_dir}")
    return {
        "league": league,
        "clf_frame": clf_frame,
        "reg_frame": reg_frame,
        "wf_clf_frame": wf_clf_frame,
        "wf_reg_frame": wf_reg_frame,
        "clf_agg_frame": clf_agg_frame,
        "reg_agg_frame": reg_agg_frame,
        "best_name": clf_art["best_name"],
        "results_dir": results_dir,
        "models_dir": models_dir,
        "saved": saved,
        "elapsed": elapsed,
        "n_rows": len(df),
        "n_teams": len(labels),
        "n_features": len(features),
        "season_counts": season_counts,
        "split_info": split_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league", default=config.DEFAULT_LEAGUE,
                    choices=list(config.LEAGUES),
                    help="league to train on")
    ap.add_argument("--include-ht", action="store_true",
                    help="add halftime goals as features (leakage for pre-match)")
    ap.add_argument("--no-xg", action="store_true",
                    help="drop xG features (strict pre-match variant)")
    ap.add_argument("--walk-forward", action="store_true",
                    help="also run the expanding-window walk-forward")
    ap.add_argument("--skip-tuning", action="store_true",
                    help="use default hyperparameters (no grid search)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: data/export/ml/<league>)")
    args = ap.parse_args()

    run_training(
        league=args.league,
        include_ht=args.include_ht,
        no_xg=args.no_xg,
        walk_forward=args.walk_forward,
        skip_tuning=args.skip_tuning,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
