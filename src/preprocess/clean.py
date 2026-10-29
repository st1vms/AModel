"""Data cleaning & preprocessing for ML/AI modeling.

Transforms the raw merged match frame (SQLite DB or export CSV) into a
modeling-ready DataFrame.

Cleaning steps
--------------
1. Missing values
   - ``referee`` is null for every match (football-data no longer publishes it)
     -> dropped.
   - ``flag`` / ``source`` / ``xg_source`` / ``season`` are metadata, not
     predictive per match -> dropped (``keep_metadata`` to retain them).
   - ``score_mismatch`` rows (the two sources disagree on the score) ->
     dropped by default (``keep_flagged`` to retain).
   - Rows with null HT goals (e.g. Sassuolo-Pescara 2016-08-28) -> dropped
     together with the flagged row.
2. Types
   - ``date`` -> ``datetime64[ns]``.
3. Encoding (``encode_teams``)
   - ``home_team`` / ``away_team`` -> label | onehot | target.
4. Feature engineering (``add_features``)
   - temporal: ``month``, ``day_of_week``, ``matchday`` (dense rank in season)
   - differentials: ``shots_diff``, ``xg_diff``
   - recent form: last-5-match averages of points / goals for / goals against,
     computed leakage-free (``shift(1)``: only strictly earlier matches).

Usage
-----
    python -m src.preprocess.clean [--encoding label|onehot|target]
                                   [--keep-flagged] [--keep-metadata]
                                   [--out PATH]

With ``--encoding label`` a ``team_labels.csv`` (code -> team) is written next
to the output so the integer codes can be decoded.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

import config

# Columns that are metadata, not per-match predictive features.
METADATA_COLS = ["season", "source", "xg_source", "flag"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matches(source: str | Path | None = None, league: str | None = None) -> pd.DataFrame:
    """Load the merged match frame for a league from the DB (default) or a CSV export."""
    if source is None:
        if league is None:
            league = config.DEFAULT_LEAGUE
        conn = sqlite3.connect(config.DB_PATH)
        try:
            df = pd.read_sql_query(
                """
                SELECT m.league, m.season, m.date,
                       t1.name AS home_team, t2.name AS away_team,
                       m.home_goals, m.away_goals,
                       m.ht_home_goals, m.ht_away_goals,
                       m.shots_home, m.shots_away,
                       m.shots_on_target_home, m.shots_on_target_away,
                       m.xg_home, m.xg_away,
                       m.b365h, m.b365d, m.b365a,
                       m.b365ch, m.b365cd, m.b365ca,
                       m.referee, m.source, m.xg_source, m.flag
                FROM matches m
                JOIN teams t1 ON m.home_team_id = t1.team_id
                JOIN teams t2 ON m.away_team_id = t2.team_id
                WHERE m.league = ?
                """,
                conn,
                params=(league,),
            )
        finally:
            conn.close()
        return df
    return pd.read_csv(source)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    keep_flagged: bool = False,
    keep_metadata: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply missing-value and type cleaning. Returns (frame, notes)."""
    df = df.copy()
    notes: list[str] = []

    # --- date -> datetime ---
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # --- drop rows where the sources disagree on the score ---
    flagged = df["flag"].fillna("").str.contains("score_mismatch")
    if flagged.any():
        notes.append(
            f"dropped {int(flagged.sum())} score_mismatch row(s): "
            + ", ".join(
                f"{r.home_team}-{r.away_team} {r.date.date()}"
                for r in df[flagged].itertuples()
            )
        )
        df = df[~flagged]
    elif not keep_flagged:
        notes.append("no score_mismatch rows found")

    # --- drop rows with null HT goals (only Sassuolo-Pescara 2016-08-28) ---
    ht_null = df["ht_home_goals"].isna() | df["ht_away_goals"].isna()
    if ht_null.any():
        notes.append(
            f"dropped {int(ht_null.sum())} row(s) with null HT goals: "
            + ", ".join(
                f"{r.home_team}-{r.away_team} {r.date.date()}"
                for r in df[ht_null].itertuples()
            )
        )
        df = df[~ht_null]

    # --- drop non-predictive columns ---
    # NOTE: `season` is kept here even when keep_metadata=False because the
    # feature-engineering step needs it (matchday, per-season form); prepare()
    # drops it at the very end.
    drop_cols = ["referee"]
    if not keep_metadata:
        drop_cols += [c for c in METADATA_COLS if c in df.columns and c != "season"]
    df = df.drop(columns=drop_cols)
    notes.append(f"dropped columns: {drop_cols}")

    # --- numeric coercion (defensive: exports may come back as object) ---
    for col in [
        "home_goals", "away_goals", "ht_home_goals", "ht_away_goals",
        "shots_home", "shots_away",
        "shots_on_target_home", "shots_on_target_away",
        "xg_home", "xg_away",
        "b365h", "b365d", "b365a",
        "b365ch", "b365cd", "b365ca",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    return df, notes


# ---------------------------------------------------------------------------
# Team encoding
# ---------------------------------------------------------------------------

def label_codes(df: pd.DataFrame) -> dict[str, int]:
    """Deterministic team -> integer code map (teams sorted by name).

    This is the single source of truth for label encoding: both
    ``encode_teams`` and the exported ``team_labels.csv`` use it, so the
    mapping file always matches the codes in the data.
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    return {t: i for i, t in enumerate(teams)}


def encode_teams(
    df: pd.DataFrame,
    method: str = "label",
    target_col: str | None = None,
) -> pd.DataFrame:
    """Encode home_team/away_team.

    method:
      - "label":  deterministic integer codes (sorted by team name).
      - "onehot": one-hot dummies with home_/away_ prefixes.
      - "target": mean goals scored by the team in that role (home -> mean
        home_goals as home, away -> mean away_goals as away); unknown teams
        fall back to the global mean.
    """
    df = df.copy()
    if method == "label":
        codes = label_codes(df)
        df["home_team"] = df["home_team"].map(codes)
        df["away_team"] = df["away_team"].map(codes)
    elif method == "onehot":
        df = pd.get_dummies(df, columns=["home_team", "away_team"],
                            prefix=["home", "away"], dtype=int)
    elif method == "target":
        tgt = target_col or "home_goals"
        home_mean = df.groupby("home_team")["home_goals"].mean()
        away_mean = df.groupby("away_team")["away_goals"].mean()
        df["home_team"] = df["home_team"].map(home_mean).fillna(df["home_goals"].mean())
        df["away_team"] = df["away_team"].map(away_mean).fillna(df["away_goals"].mean())
    else:
        raise ValueError(f"unknown encoding method: {method!r}")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _form_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Leakage-free last-N-match form per team.

    For each match, a team's form is the mean over its strictly earlier
    matches in the same season (shift(1) before rolling). First match of a
    season has no history -> 0.0.
    """
    home_pts = (
        3 * (df["home_goals"] > df["away_goals"])
        + 1 * (df["home_goals"] == df["away_goals"])
    )
    away_pts = (
        3 * (df["away_goals"] > df["home_goals"])
        + 1 * (df["home_goals"] == df["away_goals"])
    )

    has_league = "league" in df.columns
    long = pd.DataFrame(
        {
            "season": pd.concat([df["season"], df["season"]], ignore_index=True),
            "date": pd.concat([df["date"], df["date"]], ignore_index=True),
            "team": pd.concat([df["home_team"], df["away_team"]], ignore_index=True),
            "pts": pd.concat([home_pts, away_pts], ignore_index=True).astype(float),
            "gf": pd.concat([df["home_goals"], df["away_goals"]], ignore_index=True).astype(float),
            "ga": pd.concat([df["away_goals"], df["home_goals"]], ignore_index=True).astype(float),
        }
    )
    if has_league:
        long["league"] = pd.concat([df["league"], df["league"]], ignore_index=True)
    grp = ["league", "season", "team"] if has_league else ["season", "team"]
    sort_cols = (["league", "season", "team", "date"] if has_league
                 else ["season", "team", "date"])
    long = long.sort_values(sort_cols, kind="mergesort")
    g = long.groupby(grp, sort=False)
    for src, dst in (("pts", "form_pts"), ("gf", "form_gf"), ("ga", "form_ga")):
        long[dst] = g[src].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    long = long.fillna({c: 0.0 for c in ("form_pts", "form_gf", "form_ga")})

    # Map back: (season, date, team) is unique within a single-league frame
    # (one match per team per day).
    home = long.set_index(["season", "date", "team"])[["form_pts", "form_gf", "form_ga"]]
    out = pd.DataFrame(index=df.index)
    out["home_form_pts"] = [
        home.loc[(s, d, t), "form_pts"]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["home_form_gf"] = [
        home.loc[(s, d, t), "form_gf"]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["home_form_ga"] = [
        home.loc[(s, d, t), "form_ga"]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["away_form_pts"] = [
        home.loc[(s, d, t), "form_pts"]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    out["away_form_gf"] = [
        home.loc[(s, d, t), "form_gf"]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    out["away_form_ga"] = [
        home.loc[(s, d, t), "form_ga"]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    return out


def _xg_eff(df: pd.DataFrame) -> pd.DataFrame:
    """Effective xG per side: real xG when available, else shots_on_target * 0.33."""
    out = pd.DataFrame(index=df.index)
    sot_h = df.get("shots_on_target_home")
    sot_a = df.get("shots_on_target_away")
    if sot_h is not None:
        out["xg_eff_home"] = df["xg_home"].fillna(sot_h * 0.33)
    else:
        out["xg_eff_home"] = df["xg_home"].fillna(0.0)
    if sot_a is not None:
        out["xg_eff_away"] = df["xg_away"].fillna(sot_a * 0.33)
    else:
        out["xg_eff_away"] = df["xg_away"].fillna(0.0)
    return out


def _rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Days since last match (same season, same team). First match -> 7.0."""
    has_league = "league" in df.columns
    long = pd.DataFrame(
        {
            "season": pd.concat([df["season"], df["season"]], ignore_index=True),
            "date": pd.concat([df["date"], df["date"]], ignore_index=True),
            "team": pd.concat([df["home_team"], df["away_team"]], ignore_index=True),
        }
    )
    if has_league:
        long["league"] = pd.concat([df["league"], df["league"]], ignore_index=True)
    grp = ["league", "season", "team"] if has_league else ["season", "team"]
    sort_cols = (["league", "season", "team", "date"] if has_league
                 else ["season", "team", "date"])
    long = long.sort_values(sort_cols, kind="mergesort")
    g = long.groupby(grp, sort=False)
    long["rest_days"] = g["date"].transform(lambda s: s.diff().dt.days).fillna(7.0)

    home = long.set_index(["season", "date", "team"])["rest_days"]
    out = pd.DataFrame(index=df.index)
    out["rest_days_home"] = [
        home.loc[(s, d, t)]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["rest_days_away"] = [
        home.loc[(s, d, t)]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    out["is_midweek"] = ((out["rest_days_home"] <= 4) | (out["rest_days_away"] <= 4)).astype(int)
    return out


def _xg_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Rolling xG differential and finishing luck (goals - xg_eff), leakage-free."""
    xg_eff = _xg_eff(df)
    has_league = "league" in df.columns
    long = pd.DataFrame(
        {
            "season": pd.concat([df["season"], df["season"]], ignore_index=True),
            "date": pd.concat([df["date"], df["date"]], ignore_index=True),
            "team": pd.concat([df["home_team"], df["away_team"]], ignore_index=True),
            "xg_for": pd.concat([xg_eff["xg_eff_home"], xg_eff["xg_eff_away"]], ignore_index=True),
            "xg_against": pd.concat([xg_eff["xg_eff_away"], xg_eff["xg_eff_home"]], ignore_index=True),
            "goals_for": pd.concat([df["home_goals"], df["away_goals"]], ignore_index=True).astype(float),
        }
    )
    if has_league:
        long["league"] = pd.concat([df["league"], df["league"]], ignore_index=True)
    grp = ["league", "season", "team"] if has_league else ["season", "team"]
    sort_cols = (["league", "season", "team", "date"] if has_league
                 else ["season", "team", "date"])
    long = long.sort_values(sort_cols, kind="mergesort")
    g = long.groupby(grp, sort=False)

    roll_for = g["xg_for"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    roll_against = g["xg_against"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["roll_xg_diff"] = roll_for - roll_against
    long["luck"] = long["goals_for"] - long["xg_for"]
    long["roll_luck"] = g["luck"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long = long.fillna({"roll_xg_diff": 0.0, "roll_luck": 0.0})

    home = long.set_index(["season", "date", "team"])[["roll_xg_diff", "roll_luck"]]
    out = pd.DataFrame(index=df.index)
    out["xg_diff_rolling"] = [
        home.loc[(s, d, th), "roll_xg_diff"] - home.loc[(s, d, ta), "roll_xg_diff"]
        for s, d, th, ta in zip(df["season"], df["date"], df["home_team"], df["away_team"])
    ]
    out["finishing_luck_home"] = [
        home.loc[(s, d, t), "roll_luck"]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["finishing_luck_away"] = [
        home.loc[(s, d, t), "roll_luck"]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    return out


def _elo_ratings(df: pd.DataFrame, k: float = 20.0) -> pd.DataFrame:
    """Pre-match Elo ratings, chronological across all seasons (persistent).

    Ratings are recorded *before* each match (leakage-free) and updated
    afterwards. They persist across seasons so a team's strength carries over.
    """
    order = df.sort_values(["date", "home_team"], kind="mergesort")
    elo: dict[str, float] = {}
    elo_home: list[float] = []
    elo_away: list[float] = []
    for _, r in order.iterrows():
        h, a = r["home_team"], r["away_team"]
        eh = elo.get(h, 1500.0)
        ea = elo.get(a, 1500.0)
        elo_home.append(eh)
        elo_away.append(ea)
        expected = 1.0 / (1.0 + 10 ** ((ea - eh) / 400.0))
        if pd.notna(r["home_goals"]) and pd.notna(r["away_goals"]):
            if r["home_goals"] > r["away_goals"]:
                sh = 1.0
            elif r["home_goals"] < r["away_goals"]:
                sh = 0.0
            else:
                sh = 0.5
            elo[h] = eh + k * (sh - expected)
            elo[a] = ea + k * ((1 - sh) - (1 - expected))
    out = pd.DataFrame(index=df.index)
    out["elo_home"] = pd.Series(elo_home, index=order.index).reindex(df.index)
    out["elo_away"] = pd.Series(elo_away, index=order.index).reindex(df.index)
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out


def _soe_form_features(df: pd.DataFrame, elo: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Opponent-Elo-weighted rolling form and rolling PPG, leakage-free."""
    has_league = "league" in df.columns
    home_pts = (
        3 * (df["home_goals"] > df["away_goals"])
        + 1 * (df["home_goals"] == df["away_goals"])
    )
    away_pts = (
        3 * (df["away_goals"] > df["home_goals"])
        + 1 * (df["away_goals"] == df["home_goals"])
    )
    long = pd.DataFrame(
        {
            "season": pd.concat([df["season"], df["season"]], ignore_index=True),
            "date": pd.concat([df["date"], df["date"]], ignore_index=True),
            "team": pd.concat([df["home_team"], df["away_team"]], ignore_index=True),
            "pts": pd.concat([home_pts, away_pts], ignore_index=True).astype(float),
            "opp_elo": pd.concat([elo["elo_away"], elo["elo_home"]], ignore_index=True),
        }
    )
    if has_league:
        long["league"] = pd.concat([df["league"], df["league"]], ignore_index=True)
    grp = ["league", "season", "team"] if has_league else ["season", "team"]
    sort_cols = (["league", "season", "team", "date"] if has_league
                 else ["season", "team", "date"])
    long = long.sort_values(sort_cols, kind="mergesort")
    g = long.groupby(grp, sort=False)

    # SoE-weighted form: weighted rolling mean of pts * (opp_elo/1500)
    long["soe_weight"] = long["opp_elo"] / 1500.0
    long["weighted_pts"] = long["pts"] * long["soe_weight"]
    roll_wpts = g["weighted_pts"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    roll_w = g["soe_weight"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["form_soe"] = (roll_wpts / roll_w.replace(0, pd.NA)).fillna(0.0)

    # Rolling PPG
    long["ppg_rolling"] = g["pts"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).fillna(0.0)

    home = long.set_index(["season", "date", "team"])[["form_soe", "ppg_rolling"]]
    out = pd.DataFrame(index=df.index)
    out["home_form_soe"] = [
        home.loc[(s, d, t), "form_soe"]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["away_form_soe"] = [
        home.loc[(s, d, t), "form_soe"]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    out["ppg_rolling_home"] = [
        home.loc[(s, d, t), "ppg_rolling"]
        for s, d, t in zip(df["season"], df["date"], df["home_team"])
    ]
    out["ppg_rolling_away"] = [
        home.loc[(s, d, t), "ppg_rolling"]
        for s, d, t in zip(df["season"], df["date"], df["away_team"])
    ]
    return out


def _market_features(df: pd.DataFrame) -> pd.DataFrame:
    """De-vigged market probabilities from B365 opening odds. Missing -> 1/3."""
    out = pd.DataFrame(index=df.index)
    inv_h = 1.0 / df["b365h"]
    inv_d = 1.0 / df["b365d"]
    inv_a = 1.0 / df["b365a"]
    overround = inv_h + inv_d + inv_a
    out["mkt_prob_home"] = (inv_h / overround).fillna(1.0 / 3.0)
    out["mkt_prob_draw"] = (inv_d / overround).fillna(1.0 / 3.0)
    out["mkt_prob_away"] = (inv_a / overround).fillna(1.0 / 3.0)
    return out


def add_features(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Add temporal, differential, form, xG, Elo, SoE, and market features."""
    df = df.copy()

    # --- temporal ---
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_of_week  # Mon=0 .. Sun=6
    if "league" in df.columns:
        df["matchday"] = (
            df.groupby(["league", "season"])["date"].rank(method="dense").astype(int)
        )
    else:
        df["matchday"] = (
            df.groupby("season")["date"].rank(method="dense").astype(int)
        )

    # --- differentials (home - away) ---
    df["shots_diff"] = df["shots_home"] - df["shots_away"]
    df["xg_diff"] = df["xg_home"] - df["xg_away"]

    # --- recent form (last `form_window` matches, leakage-free) ---
    form = _form_features(df, window=form_window)
    df = pd.concat([df, form], axis=1)

    # --- form differential ---
    df["form_pts_diff"] = df["home_form_pts"] - df["away_form_pts"]

    # --- rest / midweek ---
    rest = _rest_features(df)
    df = pd.concat([df, rest], axis=1)

    # --- rolling xG & finishing luck ---
    xg_roll = _xg_rolling_features(df, window=form_window)
    df = pd.concat([df, xg_roll], axis=1)

    # --- Elo ratings (before SoE, which depends on Elo) ---
    elo = _elo_ratings(df)
    df = pd.concat([df, elo], axis=1)

    # --- SoE-weighted form & rolling PPG ---
    soe = _soe_form_features(df, elo, window=form_window)
    df = pd.concat([df, soe], axis=1)

    # --- market probabilities ---
    mkt = _market_features(df)
    df = pd.concat([df, mkt], axis=1)

    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def prepare(
    df: pd.DataFrame,
    encoding: str = "label",
    keep_flagged: bool = False,
    keep_metadata: bool = False,
    form_window: int = 5,
) -> tuple[pd.DataFrame, list[str], dict[str, int] | None]:
    """clean -> features -> encode.

    Returns (frame, notes, label_map). ``label_map`` is the team -> code
    mapping when ``encoding == "label"`` (else None); it is computed on the
    same frame that gets encoded, so it always matches the output codes.
    """
    notes: list[str] = []
    cleaned, clean_notes = clean(df, keep_flagged=keep_flagged, keep_metadata=keep_metadata)
    notes += clean_notes
    # Features first: form features key on the raw team names, which
    # one-hot/target encoding would have replaced.
    out = add_features(cleaned, form_window=form_window)
    label_map = label_codes(out) if encoding == "label" else None
    out = encode_teams(out, method=encoding)
    if not keep_metadata and "season" in out.columns:
        out = out.drop(columns=["season"])
        notes.append("dropped column: season (metadata)")
    notes.append(f"team encoding: {encoding}")
    return out, notes, label_map


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=None,
                    help="CSV to load instead of the DB (default: data/amodel.db)")
    ap.add_argument("--league", default=config.DEFAULT_LEAGUE,
                    help="league registry key (default: %(default)s)")
    ap.add_argument("--encoding", choices=["label", "onehot", "target"],
                    default="label")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="keep score_mismatch rows")
    ap.add_argument("--keep-metadata", action="store_true",
                    help="keep season/source/xg_source/flag columns")
    ap.add_argument("--form-window", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="output CSV (default: per-league ml dir)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else config.ml_dir(args.league) / "matches_clean.csv"
    raw = load_matches(args.source, league=args.league)
    out, notes, label_map = prepare(
        raw,
        encoding=args.encoding,
        keep_flagged=args.keep_flagged,
        keep_metadata=args.keep_metadata,
        form_window=args.form_window,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if label_map is not None:
        map_path = out_path.with_name("team_labels.csv")
        # label_map is team -> code; emit (code, team) rows so the "code"
        # column holds the int and "team" the name (matches load_team_labels).
        pd.DataFrame(
            [(code, team) for team, code in sorted(label_map.items(),
                                                   key=lambda kv: kv[1])],
            columns=["code", "team"],
        ).to_csv(map_path, index=False)
        notes.append(f"team label map saved: {map_path}")

    print(f"input rows:  {len(raw)}")
    print(f"output rows: {len(out)}")
    for n in notes:
        print(f"  [note] {n}")
    print(f"\nnulls per column:")
    nulls = out.isna().sum()
    print(nulls[nulls > 0].to_string() if nulls.any() else "  (none)")
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
