"""Data loading & target/feature construction for the ML models.

Reads the modeling-ready frame produced by ``src/preprocess/clean.py``
(``data/export/ml/<league>/matches_clean.csv``) plus the team code map
(``team_labels.csv``), derives the missing ``season`` column from ``date``,
and builds the classification/regression targets and feature matrix.

Season derivation
-----------------
The clean CSV drops ``season`` (metadata). We re-derive it from ``date``:
a match in month >= 7 belongs to the season *starting* that year
(Aug 2016 -> 2016-17); a match in month <= 6 belongs to the season
*ending* that year (May 2017 -> 2016-17).

**Pandemic special case**: for leagues whose 2019-20 season ran to
Aug 2020 (see ``config.LEAGUES[league]["pandemic_window"]``), matches
dated inside that window must map to 2019-20, not 2020-21
(2020-21 started in Sep 2020).
"""
from __future__ import annotations

import pandas as pd

import config

# Legacy flat paths (default league; new code should use the per-league dirs).
MATCHES_CSV = config.ml_dir(config.DEFAULT_LEAGUE) / "matches_clean.csv"
TEAM_LABELS_CSV = config.ml_dir(config.DEFAULT_LEAGUE) / "team_labels.csv"

# Outcome labels (1X2): 0 = Home win, 1 = Draw, 2 = Away win
OUTCOME_HOME, OUTCOME_DRAW, OUTCOME_AWAY = 0, 1, 2
OUTCOME_NAMES = {OUTCOME_HOME: "H", OUTCOME_DRAW: "D", OUTCOME_AWAY: "A"}

# Base feature set (34). Team codes are kept as-is: tree models treat them
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
    # temporal
    "rest_days_home", "rest_days_away", "is_midweek",
    # rolling xG & finishing luck
    "xg_diff_rolling", "finishing_luck_home", "finishing_luck_away",
    # Elo
    "elo_home", "elo_away", "elo_diff",
    # strength-of-estimate weighted form & rolling PPG
    "home_form_soe", "away_form_soe",
    "ppg_rolling_home", "ppg_rolling_away",
    # market (de-vigged B365 opening odds)
    "mkt_prob_home", "mkt_prob_draw", "mkt_prob_away",
]

# xG-derived features dropped together with the raw xG columns when no_xg
XG_FEATURES: list[str] = [
    "xg_home", "xg_away", "xg_diff",
    "xg_diff_rolling", "finishing_luck_home", "finishing_luck_away",
]

# Optional halftime features (known only at HT -> leakage for pre-match use)
HT_FEATURES: list[str] = ["ht_home_goals", "ht_away_goals"]

def _pandemic_window(league: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Pandemic window for a league, or None if it had no such case."""
    win = config.LEAGUES[league].get("pandemic_window")
    if not win:
        return None
    return pd.Timestamp(win[0]), pd.Timestamp(win[1])


def derive_season(date: pd.Timestamp, league: str = config.DEFAULT_LEAGUE) -> str:
    """Map a match date to its season string (e.g. '2016-17').

    month >= 7 -> season starting that year; month <= 6 -> season ending
    that year. Dates inside the league's pandemic window (if any) map to
    2019-20.
    """
    win = _pandemic_window(league)
    if win and win[0] <= date <= win[1]:
        return "2019-20"
    if date.month >= 7:
        return f"{date.year}-{str(date.year + 1)[-2:]}"
    return f"{date.year - 1}-{str(date.year)[-2:]}"


def load_team_labels(
    league: str = config.DEFAULT_LEAGUE,
    path: str | None = None,
) -> dict[int, str]:
    """code -> team name map from team_labels.csv (per league)."""
    df = pd.read_csv(path or config.ml_dir(league) / "team_labels.csv")
    return dict(zip(df["code"], df["team"]))


def load_data(
    league: str = config.DEFAULT_LEAGUE,
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
    df = pd.read_csv(path or config.ml_dir(league) / "matches_clean.csv",
                     parse_dates=["date"])
    labels = load_team_labels(league, path=None)

    df["season"] = df["date"].map(lambda d: derive_season(d, league))
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
        feats = [f for f in feats if f not in XG_FEATURES]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing from data: {missing}")

    return df, labels, feats


def season_counts(df: pd.DataFrame) -> dict[str, int]:
    """{season: n_matches} for quick sanity reporting."""
    return df["season"].value_counts().sort_index().to_dict()
