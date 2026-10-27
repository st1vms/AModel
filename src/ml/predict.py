"""Inference CLI: predict 1X2 probabilities & expected goals for fixtures.

Usage
-----
    python -m src.ml.predict --matches fixtures.csv
    python -m src.ml.predict --demo

``--matches``: CSV with the same schema as matches_clean.csv minus the
targets (date, home_team, away_team, shots_*, xg_*, month, day_of_week,
matchday, form features). ``home_team``/``away_team`` may be integer codes
or team names (decoded via the saved team_labels map).

``--demo``: re-predicts the most recent matchday in the training data as a
sanity check (probabilities vs actual results).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config
from src.ml import data, models

# Legacy flat path (default league; new code should use the per-league dirs).
PIPELINE_PATH = config.ml_models_dir(config.DEFAULT_LEAGUE) / "pipeline.joblib"


def load_pipeline(
    league: str = config.DEFAULT_LEAGUE,
    path: str | Path | None = None,
) -> dict:
    """Load the saved inference pipeline (models + metadata) for a league."""
    if path is None and league not in config.LEAGUES:
        raise ValueError(
            f"unknown league {league!r} — expected one of "
            f"{sorted(config.LEAGUES)}, or pass an explicit path=..."
        )
    return joblib.load(path or config.ml_models_dir(league) / "pipeline.joblib")


def prepare_fixtures(
    df: pd.DataFrame,
    pipeline: dict,
) -> pd.DataFrame:
    """Validate & normalize a fixtures frame against the pipeline metadata.

    - Decodes team names to codes when needed.
    - Checks that all pipeline features are present.
    """
    df = df.copy()
    labels: dict[int, str] = pipeline["team_labels"]
    name_to_code = {v: k for k, v in labels.items()}

    for col in ("home_team", "away_team"):
        if df[col].dtype == object:
            unknown = set(df[col]) - set(name_to_code)
            if unknown:
                raise ValueError(f"unknown team name(s) in {col}: {sorted(unknown)}")
            df[col] = df[col].map(name_to_code)
        df[col] = df[col].astype(int)

    features = pipeline["features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"fixtures missing feature column(s): {missing}")
    # Return the full decoded frame (not just features): predict() selects
    # the feature columns itself and also needs date/home/away for output.
    return df


def predict(
    fixtures: pd.DataFrame,
    pipeline: dict,
) -> pd.DataFrame:
    """Run the saved models on prepared fixtures.

    Returns a frame with: date, home_team, away_team, p_home, p_draw,
    p_away, outcome (H/D/A), expected_home_goals, expected_away_goals.
    """
    X = fixtures[pipeline["features"]].copy()
    # Pre-match fixtures have no HT score yet, so HT feature columns are NaN.
    # Impute 0 — the natural "no goals scored yet" state and the modal HT
    # result — so estimators that don't accept NaN natively (LogisticRegression)
    # don't crash. GBMs would tolerate the NaN, but 0 is a defensible prior.
    ht_cols = [c for c in X.columns if c.startswith("ht_")]
    if ht_cols:
        X[ht_cols] = X[ht_cols].fillna(0.0)
    proba = pipeline["classifier"].predict_proba(X)
    exp_home = models.predict_goals(pipeline["regressor_home"], X)
    exp_away = models.predict_goals(pipeline["regressor_away"], X)

    out = pd.DataFrame({
        "date": fixtures["date"].dt.date if hasattr(fixtures["date"], "dt")
                else fixtures["date"],
        "home_team": fixtures["home_team"],
        "away_team": fixtures["away_team"],
        "p_home": proba[:, 0],
        "p_draw": proba[:, 1],
        "p_away": proba[:, 2],
        "outcome": np.array(["H", "D", "A"])[np.argmax(proba, axis=1)],
        "expected_home_goals": np.round(exp_home, 2),
        "expected_away_goals": np.round(exp_away, 2),
    })
    return out


def demo_fixtures(
    league: str = config.DEFAULT_LEAGUE,
    n: int = 10,
) -> pd.DataFrame:
    """Build a demo fixtures frame: the last `n` matches in the clean data.

    Used as a sanity check — the model re-predicts recent results.
    """
    df, _, _ = data.load_data(league)
    return df.tail(n).copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league", default=config.DEFAULT_LEAGUE,
                    choices=list(config.LEAGUES),
                    help="league the pipeline was trained on")
    ap.add_argument("--matches", default=None,
                    help="fixtures CSV (matches_clean.csv schema minus targets)")
    ap.add_argument("--demo", action="store_true",
                    help="re-predict the last 10 matches in the training data")
    ap.add_argument("--pipeline", default=None,
                    help="path to pipeline.joblib (default: per-league path)")
    ap.add_argument("--out", default=None,
                    help="write predictions to this CSV (default: stdout only)")
    args = ap.parse_args()

    if not args.demo and not args.matches:
        ap.error("provide --matches PATH or --demo")

    pipeline = load_pipeline(args.league, args.pipeline)
    if args.demo:
        fixtures = demo_fixtures(args.league)
        print(f"Demo: re-predicting the last {len(fixtures)} matches "
              f"(classifier: {pipeline['classifier_name']}, "
              f"goal models: {pipeline['regressor_home_name']}/"
              f"{pipeline['regressor_away_name']})")
    else:
        fixtures = pd.read_csv(args.matches, parse_dates=["date"])

    X = prepare_fixtures(fixtures, pipeline)
    preds = predict(X, pipeline)

    # Attach actual results for the demo (sanity check).
    if args.demo:
        actual = {
            "actual": [
                "H" if h > a else ("D" if h == a else "A")
                for h, a in zip(fixtures["home_goals"], fixtures["away_goals"])
            ],
            "actual_score": [
                f"{h}-{a}" for h, a in zip(fixtures["home_goals"],
                                           fixtures["away_goals"])
            ],
        }
        # Reset the index so the concat aligns row-by-row (preds keeps the
        # original data index from the fixtures frame).
        preds = pd.concat([preds.reset_index(drop=True),
                           pd.DataFrame(actual)], axis=1)

    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.3f}"):
        print(preds.to_string(index=False))

    if args.out:
        preds.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
