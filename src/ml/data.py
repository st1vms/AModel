"""Data loading & target/feature construction for the ML models.

Reads the modeling-ready frame produced by ``src/preprocess/clean.py``
(``data/export/ml/matches_clean.csv``) plus the team code map
(``team_labels.csv``), derives the missing ``season`` column from ``date``,
and builds the classification/regression targets and feature matrix.

Season derivation
-----------------
The clean CSV drops ``season`` (metadata). We re-derive it from ``date``:
a match in month >= 7 belongs to the season *starting* that year
(Aug 2016 -> 2016-17); a match in month <= 6 belongs to the season
*ending* that year (May 2017 -> 2016-17).

**Pandemic special case**: the 2019-20 season ran to Aug 2020, so matches
dated 2020-06-01 .. 2020-08-31 must map to 2019-20, not 2020-21
(2020-21 started in Sep 2020).
"""
from __future__ import annotations

import pandas as pd

import config

MATCHES_CSV = config.ML_DIR / "matches_clean.csv"
TEAM_LABELS_CSV = config.ML_DIR / "team_labels.csv"

# Outcome labels (1X2): 0 = Home win, 1 = Draw, 2 = Away win
OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2
OUTCOME_NAMES = {OUTCOME_HOME: "H", OUTCOME_DRAW: "D", OUTCOME_AWAY: "A"}

# Base feature set (18). Team codes are kept as-is: tree models treat them
# as high-cardinality categoricals natively (CatBoost via cat_features).
FEATURES: list[str] = [
    "home_team", "away_team",
    "month", "day_of_week", "matchday",
    "shots_home", "shots_away",
    "xg_home", "xg_away",
    "shots_diff", "xg_diff",
    "home_form_pts", "home_form_gf", "home_form_ga",
    "away_form_pts", "away_form_gf", "away_form_ga",
    "form_pts_diff",
]

# Optional halftime features (known only at HT -> leakage for pre-match use)
HT_FEATURES: list[str] = ["ht_home_goals", "ht_away_goals"]

# Pandemic window: 2019-20 season matches played Jun-Aug 2020.
_PANDEMIC_START = pd.Timestamp("2020-06-01")
_PANDEMIC_END = pd.Timestamp("2020-08-31")


def derive_season(date: pd.Timestamp) -> str:
    """Map a match date to its season string (e.g. '2016-17').

    month >= 7 -> season starting that year; month <= 6 -> season ending
    that year. Dates in the 2020 pandemic window map to 2019-20.
    """
    if _PANDEMIC_START <= date <= _PANDEMIC_END:
        return "2019-20"
    if date.month >= 7:
        return f"{date.year}-{str(date.year + 1)[-2:]}"
    return f"{date.year - 1}-{str(date.year)[-2:]}"


def load_team_labels(path: str | None = None) -> dict[int, str]:
    """code -> team name map from team_labels.csv."""
    df = pd.read_csv(path or TEAM_LABELS_CSV)
    return dict(zip(df["code"], df["team"]))


def load_data(
    path: str | None = None,
    include_ht: bool = False,
    no_xg: bool = False,
) -> tuple[pd.DataFrame, dict[int, str], list[str]]:
    """Load the clean frame, derive season, build targets and features.

    Returns
    -------
    (df, team_labels, features)
      df: sorted by date, with columns: date, season, home_team, away_team,
          home_goals, away_goals, outcome, + feature columns.
      team_labels: code -> team name (for reporting/decoding).
      features: the active feature list (FEATURES, optionally + HT,
          optionally minus xG columns).
    """
    df = pd.read_csv(path or MATCHES_CSV, parse_dates=["date"])
    labels = load_team_labels()

    df["season"] = df["date"].map(derive_season)
    df = df.sort_values(["date", "home_team"], kind="mergesort").reset_index(drop=True)

    # --- targets ---
    df["outcome"] = (
        (df["home_goals"] > df["away_goals"]) * OUTCOME_HOME
        + (df["home_goals"] == df["away_goals"]) * OUTCOME_DRAW
        + (df["home_goals"] < df["away_goals"]) * OUTCOME_AWAY
    ).astype(int)

    # --- feature selection ---
    feats = list(FEATURES)
    if include_ht:
        feats += HT_FEATURES
    if no_xg:
        feats = [f for f in feats if f not in ("xg_home", "xg_away", "xg_diff")]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing from data: {missing}")

    return df, labels, feats


def season_counts(df: pd.DataFrame) -> dict[str, int]:
    """{season: n_matches} for quick sanity reporting."""
    return df["season"].value_counts().sort_index().to_dict()
