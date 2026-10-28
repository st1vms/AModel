"""Auto-fixture builder: pull scheduled (unplayed) matches from the raw
Understat JSON and turn them into a frame the saved pipeline can predict on.

The merged data (DB / matches_clean.csv) only contains *played* matches —
``src/clean/merge.py`` drops Understat entries with ``isResult: false``.
The raw JSON files (``data/raw/understat/{year}.json``) still hold the full
season schedule, so scheduled fixtures can be recovered from there.

Scheduled matches have no shots/xG yet, so those are filled with each
team's own historical role-specific averages (home team's mean home
shots/xG, away team's mean away shots/xG), falling back to the league
average for teams with no history. Form features are computed with the
same leakage-free routine the clean pipeline uses
(``src/preprocess/clean.add_features``) on *played* matches only — fixture
rows have NaN goals, which would otherwise count as 0-point / 0-goal
results and pollute the rolling window of later fixtures — and each team's
latest real-match form is mapped onto all of its scheduled fixtures.

Usage
-----
    from src.ml.fixtures import build_fixtures
    fixtures = build_fixtures("2026-27")            # all scheduled matches
    fixtures = build_fixtures("2026-27", n_matchdays=1)  # next matchday only
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from src.clean.normalize import canonical_team
from src.ml.data import FEATURES, derive_season
from src.preprocess.clean import add_features, label_codes, load_matches


def load_scheduled(
    season: str,
    league: str = config.DEFAULT_LEAGUE,
) -> pd.DataFrame:
    """Scheduled (unplayed) matches for ``season`` from the raw Understat JSON.

    Returns a frame with columns ``date`` (datetime), ``home_team``,
    ``away_team`` (canonical names), sorted by date. Empty if the raw file
    is missing or has no scheduled entries.
    """
    path = config.league_dir("understat", league) / f"{config.understat_year(season)}.json"
    if not path.exists():
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for m in data.get("dates") or data.get("matches") or []:
        if m.get("isResult") is not False:
            continue
        dt = m.get("datetime")
        if not dt:
            continue
        home = canonical_team(m["h"]["title"], league)
        away = canonical_team(m["a"]["title"], league)
        rows.append({
            "date": pd.Timestamp(dt.split(" ")[0]),
            "home_team": home,
            "away_team": away,
        })
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team"])
    if df.empty:
        return df
    # Keep only matches that actually belong to the requested season
    # (defensive: the file is keyed by start year, but verify by date).
    df = df[df["date"].map(lambda d: derive_season(d, league)) == season]
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


def scheduled_matchdays(
    season: str,
    league: str = config.DEFAULT_LEAGUE,
) -> pd.DataFrame:
    """One row per scheduled (unplayed) matchday of ``season``.

    Returns a frame with columns ``date`` (datetime, sorted ascending) and
    ``matchday`` (int). ``matchday`` is the dense rank of the date among ALL
    distinct dates of the season (played + scheduled) — the same convention
    as the ``matchday`` feature in ``src/preprocess/clean.add_features`` — so
    the labels match what the trained model sees. The first row is the
    current (next) matchday. Empty if there are no scheduled matches.
    """
    scheduled = load_scheduled(season, league)
    if scheduled.empty:
        return pd.DataFrame(columns=["date", "matchday"])

    sched_dates = scheduled["date"].drop_duplicates().sort_values()

    # Played dates of the season (from the DB) so the dense rank counts the
    # whole season, not just the remaining fixtures.
    played = load_matches(league=league)
    played_dates = pd.Series(
        pd.to_datetime(played.loc[played["season"] == season, "date"])
    ).drop_duplicates()

    all_dates = (
        pd.concat([sched_dates, played_dates], ignore_index=True)
        .drop_duplicates()
        .sort_values()
    )
    ranks = all_dates.rank(method="dense").astype(int)
    return pd.DataFrame({
        "date": sched_dates,
        "matchday": sched_dates.map(dict(zip(all_dates, ranks))).astype(int),
    }).reset_index(drop=True)


def _team_priors(historical: pd.DataFrame) -> dict[str, pd.Series]:
    """Role-specific shots/xG priors per team, from played matches.

    Returns a dict with four Series indexed by team name:
    ``home_shots`` / ``home_xg`` (mean as a home side) and
    ``away_shots`` / ``away_xg`` (mean as an away side). Teams with no
    history in a role are absent from the Series; callers fall back to
    the league-wide average for those.
    """
    out: dict[str, pd.Series] = {}
    for role, col in (("home", "shots_home"), ("home", "xg_home"),
                      ("away", "shots_away"), ("away", "xg_away")):
        team_col = f"{role}_team"
        grouped = (
            pd.to_numeric(historical[col], errors="coerce")
            .groupby(historical[team_col])
            .mean()
        )
        out[f"{role}_{col.split('_')[0]}"] = grouped
    return out


def _league_averages(historical: pd.DataFrame) -> dict[str, float]:
    """League-wide mean shots and xG per side, from played matches."""
    out = {}
    for col in ("shots_home", "shots_away", "xg_home", "xg_away"):
        vals = pd.to_numeric(historical[col], errors="coerce").dropna()
        out[col] = float(vals.mean()) if len(vals) else 0.0
    return out


def build_fixtures(
    season: str,
    league: str = config.DEFAULT_LEAGUE,
    n_matchdays: int | None = None,
    matchday: pd.Timestamp | str | None = None,
    labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build a prediction-ready frame of scheduled matches for ``season``.

    Parameters
    ----------
    season: e.g. "2026-27". Scheduled matches are read from the raw
        Understat JSON for that season's start year.
    league: league key (see ``config.LEAGUES``).
    n_matchdays: if given, keep only the first N matchdays (distinct dates,
        earliest first). None = all scheduled matches. Ignored when
        ``matchday`` is given.
    matchday: if given (a date), keep only the scheduled matches on that
        single date. Takes precedence over ``n_matchdays``. Raises
        ``ValueError`` if no scheduled matches fall on that date.
    labels: optional code -> team map (e.g. ``pipeline["team_labels"]``).
        When given, teams are encoded with the saved pipeline's own map so
        the codes are guaranteed consistent with the trained models; an
        unknown team raises ``ValueError``. When None, the deterministic
        ``label_codes`` mapping is used (same as the clean pipeline).

    Returns
    -------
    DataFrame with ``date`` (datetime), ``home_team``/``away_team`` as
    integer codes, and every column in ``src.ml.data.FEATURES``. Directly
    compatible with ``src.ml.predict.prepare_fixtures`` / ``predict``.
    """
    scheduled = load_scheduled(season, league)
    if scheduled.empty:
        raise ValueError(
            f"no scheduled matches found for {season} "
            f"(check {config.league_dir('understat', league) / f'{config.understat_year(season)}.json'})"
        )
    if matchday is not None:
        # Single-matchday selection (takes precedence over n_matchdays).
        scheduled = scheduled[scheduled["date"] == pd.Timestamp(matchday)]
        if scheduled.empty:
            raise ValueError(
                f"no scheduled matches on {pd.Timestamp(matchday).date()} "
                f"for {season}"
            )
    elif n_matchdays is not None:
        first_dates = scheduled["date"].drop_duplicates().sort_values().head(n_matchdays)
        scheduled = scheduled[scheduled["date"].isin(first_dates)]

    historical = load_matches(league=league)  # DB, team names, played only
    historical = historical.copy()
    historical["date"] = pd.to_datetime(historical["date"])
    # Match the clean pipeline: drop rows with null HT goals (e.g.
    # understat_only rows). Otherwise a team can appear twice on the same
    # date (once per source, possibly with home/away reversed), which breaks
    # _form_features' assumption that (season, date, team) is unique and
    # yields object-dtype form columns that LightGBM rejects.
    ht_null = historical["ht_home_goals"].isna() | historical["ht_away_goals"].isna()
    if ht_null.any():
        historical = historical[~ht_null]
    avgs = _league_averages(historical)
    priors = _team_priors(historical)

    # Fixture rows: no goals yet; shots/xG filled with each team's own
    # role-specific historical averages (league average as fallback).
    fixtures = scheduled.copy()
    fixtures["league"] = league
    fixtures["season"] = season
    fixtures["home_goals"] = float("nan")
    fixtures["away_goals"] = float("nan")
    fixtures["ht_home_goals"] = float("nan")
    fixtures["ht_away_goals"] = float("nan")
    fixtures["shots_home"] = fixtures["home_team"].map(priors["home_shots"]).fillna(avgs["shots_home"])
    fixtures["shots_away"] = fixtures["away_team"].map(priors["away_shots"]).fillna(avgs["shots_away"])
    fixtures["xg_home"] = fixtures["home_team"].map(priors["home_xg"]).fillna(avgs["xg_home"])
    fixtures["xg_away"] = fixtures["away_team"].map(priors["away_xg"]).fillna(avgs["xg_away"])
    # Unknown at fixture time: shots on target and opening odds. NaN lets
    # xg_eff fall back to 0 and market probs fall back to 1/3.
    fixtures["shots_on_target_home"] = float("nan")
    fixtures["shots_on_target_away"] = float("nan")
    fixtures["b365h"] = float("nan")
    fixtures["b365d"] = float("nan")
    fixtures["b365a"] = float("nan")
    fixtures["_is_fixture"] = True

    hist = historical[["league", "season", "date", "home_team", "away_team",
                       "home_goals", "away_goals",
                       "shots_home", "shots_away",
                       "shots_on_target_home", "shots_on_target_away",
                       "xg_home", "xg_away",
                       "b365h", "b365d", "b365a"]].copy()
    hist["_is_fixture"] = False
    combined = pd.concat([hist, fixtures], ignore_index=True)
    combined = combined.sort_values(["season", "date", "home_team"],
                                    kind="mergesort").reset_index(drop=True)

    # Feature engineering on the combined frame: fixture rows (latest dates
    # in their season) get month/day_of_week/matchday and the shots/xG
    # diffs from their own rows.
    combined = add_features(combined)

    # Rolling-feature fix: fixture rows have NaN goals, which the rolling
    # features (_form_features, _xg_rolling_features, _soe_form_features)
    # treat as 0 points / 0 goals, so each team's own *future scheduled
    # matches* would pollute the rolling window of later fixtures (form
    # decays to 0 by mid-season and the model falls back to base-rate
    # draws). The values on *played* rows are already correct — fixture
    # dates are all in the future, so they cannot enter a played match's
    # strictly-earlier window — so take each team's latest real-match
    # values and map them onto all of its fixtures.
    #
    # Per-team rolling features to carry forward (home/away role columns
    # map onto the same per-team quantity):
    per_team = {
        "form_pts": ("home_form_pts", "away_form_pts"),
        "form_gf": ("home_form_gf", "away_form_gf"),
        "form_ga": ("home_form_ga", "away_form_ga"),
        "form_soe": ("home_form_soe", "away_form_soe"),
        "ppg_rolling": ("ppg_rolling_home", "ppg_rolling_away"),
        "roll_luck": ("finishing_luck_home", "finishing_luck_away"),
    }
    real = combined[~combined["_is_fixture"]]
    long_parts = []
    for feat, (hcol, acol) in per_team.items():
        h = real[["season", "date", "home_team", hcol]].rename(
            columns={"home_team": "team", hcol: feat})
        a = real[["season", "date", "away_team", acol]].rename(
            columns={"away_team": "team", acol: feat})
        long_parts.append(pd.concat([h, a], ignore_index=True))
    latest = (
        pd.concat(long_parts, ignore_index=True)
        .sort_values(["season", "team", "date"], kind="mergesort")
        .groupby(["season", "team"], as_index=False)
        .last()
        .drop(columns=["date"])
        .rename(columns={f: f"_latest_{f}" for f in per_team})
    )
    combined = (
        combined
        .merge(latest.rename(columns={"team": "home_team"}),
               on=["season", "home_team"], how="left", suffixes=("", "_home"))
        .merge(latest.rename(columns={"team": "away_team"}),
               on=["season", "away_team"], how="left", suffixes=("", "_away"))
    )
    for feat, (hcol, acol) in per_team.items():
        combined[hcol] = combined[f"_latest_{feat}"].fillna(0.0)
        combined[acol] = combined[f"_latest_{feat}_away"].fillna(0.0)
    combined["form_pts_diff"] = combined["home_form_pts"] - combined["away_form_pts"]
    # xg_diff_rolling is a home-away differential of a per-team rolling xg
    # gap that is not stored separately, so it cannot be carried forward
    # per team. Use 0.0 (neutral) for fixtures: the rolling xg gap is a
    # small-magnitude feature and 0 is a safe neutral prior.
    combined["xg_diff_rolling"] = 0.0
    combined = combined.drop(columns=[
        f"_latest_{f}" for f in per_team
    ] + [f"_latest_{f}_away" for f in per_team])

    # Rest days: for a fixture, the true rest is the gap since the team's
    # last *played* match (fixture rows have no goals, so the date-diff
    # computed on the combined frame would measure gap-to-previous-fixture
    # for a team's second+ fixture). Recompute from the last real date.
    last_real = pd.concat([
        real[["season", "date", "home_team"]].rename(columns={"home_team": "team"}),
        real[["season", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True)
    last_real = (
        last_real
        .sort_values(["season", "team", "date"], kind="mergesort")
        .groupby(["season", "team"], as_index=False)
        .last()
        .rename(columns={"date": "last_date"})
    )
    fx = combined[combined["_is_fixture"]]
    rest_h = fx.merge(
        last_real.rename(columns={"team": "home_team"}),
        on=["season", "home_team"], how="left",
    )
    rest_a = fx.merge(
        last_real.rename(columns={"team": "away_team"}),
        on=["season", "away_team"], how="left",
    )
    # rest_h/rest_a carry a fresh RangeIndex; align positionally with fx.
    combined.loc[fx.index, "rest_days_home"] = (
        (fx["date"].to_numpy() - pd.to_datetime(rest_h["last_date"]).to_numpy())
        / np.timedelta64(1, "D")
    )
    combined.loc[fx.index, "rest_days_away"] = (
        (fx["date"].to_numpy() - pd.to_datetime(rest_a["last_date"]).to_numpy())
        / np.timedelta64(1, "D")
    )
    combined["rest_days_home"] = combined["rest_days_home"].fillna(7.0)
    combined["rest_days_away"] = combined["rest_days_away"].fillna(7.0)
    combined["is_midweek"] = (
        (combined["rest_days_home"] <= 4) | (combined["rest_days_away"] <= 4)
    ).astype(int)

    out = combined[combined["_is_fixture"]].drop(columns=["_is_fixture"])

    # Encode teams. When the caller supplies the saved pipeline's label map,
    # use it so codes match the trained models exactly; otherwise fall back
    # to the deterministic label_codes mapping (same as the clean pipeline).
    if labels is not None:
        # Only fixture teams matter (historical rows are dropped below).
        unknown = (
            set(fixtures["home_team"]) | set(fixtures["away_team"])
        ) - set(labels.values())
        if unknown:
            raise ValueError(f"Teams not in pipeline label map: {sorted(unknown)}")
        name_to_code = {name: code for code, name in labels.items()}
        out["home_team"] = out["home_team"].map(name_to_code).astype(int)
        out["away_team"] = out["away_team"].map(name_to_code).astype(int)
    else:
        codes = label_codes(combined)
        out["home_team"] = out["home_team"].map(codes).astype(int)
        out["away_team"] = out["away_team"].map(codes).astype(int)

    # FEATURES already includes home_team/away_team; add the non-feature
    # columns (date, HT goals) without duplicating.
    keep = ["date", "ht_home_goals", "ht_away_goals"] + list(FEATURES)
    return out[keep].reset_index(drop=True)
