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

DEFAULT_OUT = config.EXPORT_DIR / "ml" / "matches_clean.csv"

# Columns that are metadata, not per-match predictive features.
METADATA_COLS = ["season", "source", "xg_source", "flag"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matches(source: str | Path | None = None) -> pd.DataFrame:
    """Load the merged match frame from the DB (default) or a CSV export."""
    if source is None:
        conn = sqlite3.connect(config.DB_PATH)
        try:
            df = pd.read_sql_query(
                """
                SELECT m.season, m.date,
                       t1.name AS home_team, t2.name AS away_team,
                       m.home_goals, m.away_goals,
                       m.ht_home_goals, m.ht_away_goals,
                       m.shots_home, m.shots_away,
                       m.xg_home, m.xg_away,
                       m.referee, m.source, m.xg_source, m.flag
                FROM matches m
                JOIN teams t1 ON m.home_team_id = t1.team_id
                JOIN teams t2 ON m.away_team_id = t2.team_id
                """,
                conn,
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
        "shots_home", "shots_away", "xg_home", "xg_away",
    ]:
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
    long = long.sort_values(["season", "team", "date"], kind="mergesort")
    g = long.groupby(["season", "team"], sort=False)
    for src, dst in (("pts", "form_pts"), ("gf", "form_gf"), ("ga", "form_ga")):
        long[dst] = g[src].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
    long = long.fillna({c: 0.0 for c in ("form_pts", "form_gf", "form_ga")})

    # Map back: (season, date, team) is unique (one match per team per day).
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


def add_features(df: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Add temporal, differential, and recent-form features."""
    df = df.copy()

    # --- temporal ---
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.day_of_week  # Mon=0 .. Sun=6
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
    ap.add_argument("--encoding", choices=["label", "onehot", "target"],
                    default="label")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="keep score_mismatch rows")
    ap.add_argument("--keep-metadata", action="store_true",
                    help="keep season/source/xg_source/flag columns")
    ap.add_argument("--form-window", type=int, default=5)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    raw = load_matches(args.source)
    out, notes, label_map = prepare(
        raw,
        encoding=args.encoding,
        keep_flagged=args.keep_flagged,
        keep_metadata=args.keep_metadata,
        form_window=args.form_window,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if label_map is not None:
        map_path = out_path.with_name("team_labels.csv")
        pd.DataFrame(
            sorted(label_map.items(), key=lambda kv: kv[1]),
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
