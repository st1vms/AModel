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

import pandas as pd

import config
from src.clean.normalize import canonical_team
from src.ml.data import FEATURES, derive_season
from src.preprocess.clean import add_features, label_codes, load_matches


def load_scheduled(season: str) -> pd.DataFrame:
    """Scheduled (unplayed) matches for ``season`` from the raw Understat JSON.

    Returns a frame with columns ``date`` (datetime), ``home_team``,
    ``away_team`` (canonical names), sorted by date. Empty if the raw file
    is missing or has no scheduled entries.
    """
    path = config.UNDERSTAT_DIR / f"{config.understat_year(season)}.json"
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
        home = canonical_team(m["h"]["title"])
        away = canonical_team(m["a"]["title"])
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
    df = df[df["date"].map(derive_season) == season]
    return df.sort_values("date", kind="mergesort").reset_index(drop=True)


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
    n_matchdays: int | None = None,
    labels: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Build a prediction-ready frame of scheduled matches for ``season``.

    Parameters
    ----------
    season: e.g. "2026-27". Scheduled matches are read from the raw
        Understat JSON for that season's start year.
    n_matchdays: if given, keep only the first N matchdays (distinct dates,
        earliest first). None = all scheduled matches.
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
    scheduled = load_scheduled(season)
    if scheduled.empty:
        raise ValueError(
            f"no scheduled matches found for {season} "
            f"(check {config.UNDERSTAT_DIR / f'{config.understat_year(season)}.json'})"
        )
    if n_matchdays is not None:
        first_dates = scheduled["date"].drop_duplicates().sort_values().head(n_matchdays)
        scheduled = scheduled[scheduled["date"].isin(first_dates)]

    historical = load_matches()  # DB, team names, played matches only
    historical = historical.copy()
    historical["date"] = pd.to_datetime(historical["date"])
    avgs = _league_averages(historical)
    priors = _team_priors(historical)

    # Fixture rows: no goals yet; shots/xG filled with each team's own
    # role-specific historical averages (league average as fallback).
    fixtures = scheduled.copy()
    fixtures["season"] = season
    fixtures["home_goals"] = float("nan")
    fixtures["away_goals"] = float("nan")
    fixtures["ht_home_goals"] = float("nan")
    fixtures["ht_away_goals"] = float("nan")
    fixtures["shots_home"] = fixtures["home_team"].map(priors["home_shots"]).fillna(avgs["shots_home"])
    fixtures["shots_away"] = fixtures["away_team"].map(priors["away_shots"]).fillna(avgs["shots_away"])
    fixtures["xg_home"] = fixtures["home_team"].map(priors["home_xg"]).fillna(avgs["xg_home"])
    fixtures["xg_away"] = fixtures["away_team"].map(priors["away_xg"]).fillna(avgs["xg_away"])
    fixtures["_is_fixture"] = True

    hist = historical[["season", "date", "home_team", "away_team",
                       "home_goals", "away_goals",
                       "shots_home", "shots_away", "xg_home", "xg_away"]].copy()
    hist["_is_fixture"] = False
    combined = pd.concat([hist, fixtures], ignore_index=True)
    combined = combined.sort_values(["season", "date", "home_team"],
                                    kind="mergesort").reset_index(drop=True)

    # Feature engineering on the combined frame: fixture rows (latest dates
    # in their season) get month/day_of_week/matchday and the shots/xG
    # diffs from their own rows.
    combined = add_features(combined)

    # Form fix: fixture rows have NaN goals, which _form_features treats as
    # 0 points / 0 goals, so each team's own *future scheduled matches*
    # would pollute the rolling window of later fixtures (form decays to 0
    # by mid-season and the model falls back to base-rate draws). The form
    # on *played* rows is already correct — fixture dates are all in the
    # future, so they cannot enter a played match's strictly-earlier
    # window — so take each team's latest real-match form and map it onto
    # all of its fixtures.
    real = combined[~combined["_is_fixture"]]
    home_long = real[["season", "date", "home_team",
                      "home_form_pts", "home_form_gf", "home_form_ga"]].rename(
        columns={"home_team": "team", "home_form_pts": "form_pts",
                 "home_form_gf": "form_gf", "home_form_ga": "form_ga"})
    away_long = real[["season", "date", "away_team",
                      "away_form_pts", "away_form_gf", "away_form_ga"]].rename(
        columns={"away_team": "team", "away_form_pts": "form_pts",
                 "away_form_gf": "form_gf", "away_form_ga": "form_ga"})
    latest = (
        pd.concat([home_long, away_long], ignore_index=True)
        .sort_values(["season", "team", "date"], kind="mergesort")
        .groupby(["season", "team"], as_index=False)
        .last()
        .drop(columns=["date"])
    )
    combined = (
        combined
        .merge(latest.rename(columns={"team": "home_team"}),
               on=["season", "home_team"], how="left", suffixes=("", "_home"))
        .merge(latest.rename(columns={"team": "away_team"}),
               on=["season", "away_team"], how="left", suffixes=("", "_away"))
    )
    combined["home_form_pts"] = combined["form_pts"].fillna(0.0)
    combined["home_form_gf"] = combined["form_gf"].fillna(0.0)
    combined["home_form_ga"] = combined["form_ga"].fillna(0.0)
    combined["away_form_pts"] = combined["form_pts_away"].fillna(0.0)
    combined["away_form_gf"] = combined["form_gf_away"].fillna(0.0)
    combined["away_form_ga"] = combined["form_ga_away"].fillna(0.0)
    combined["form_pts_diff"] = combined["home_form_pts"] - combined["away_form_pts"]
    combined = combined.drop(columns=[
        "form_pts", "form_gf", "form_ga",
        "form_pts_away", "form_gf_away", "form_ga_away",
    ])

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
