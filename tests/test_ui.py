"""Tests for the UI support code: fixture builder, run_training, app import."""
from __future__ import annotations

import pandas as pd
import pytest

import config
from src.ml import data, evaluate, fixtures, train


# ---------------------------------------------------------------------------
# load_scheduled
# ---------------------------------------------------------------------------

def test_load_scheduled_in_progress_season():
    df = fixtures.load_scheduled(config.IN_PROGRESS_SEASON)
    assert not df.empty
    # All scheduled (unplayed) matches of the season.
    assert len(df) > 0
    assert set(df.columns) == {"date", "home_team", "away_team"}
    # Sorted by date, no duplicates on (date, home, away).
    assert df["date"].is_monotonic_increasing
    assert not df.duplicated(subset=["date", "home_team", "away_team"]).any()
    # Team names are canonical (no Understat suffixes like "FC").
    assert df["home_team"].notna().all()
    assert df["away_team"].notna().all()


def test_load_scheduled_finished_season_empty():
    # 2016-17 is fully played -> no isResult=false rows remain.
    df = fixtures.load_scheduled("2016-17")
    assert df.empty
    assert list(df.columns) == ["date", "home_team", "away_team"]


def test_load_scheduled_missing_file():
    df = fixtures.load_scheduled("2099-00")
    assert df.empty


# ---------------------------------------------------------------------------
# build_fixtures
# ---------------------------------------------------------------------------

def test_build_fixtures_columns_and_dtypes():
    out = fixtures.build_fixtures(config.IN_PROGRESS_SEASON, n_matchdays=1)
    assert not out.empty
    # Superset of FEATURES + date + HT goal columns.
    for col in ["date", *data.FEATURES]:
        assert col in out.columns, f"missing column {col}"
    # Team columns are integer codes.
    assert pd.api.types.is_integer_dtype(out["home_team"])
    assert pd.api.types.is_integer_dtype(out["away_team"])
    # No NaN in any feature column (shots/xG filled with league averages,
    # form computed from strictly-earlier history).
    assert not out[data.FEATURES].isna().any().any()
    # Goals are still unknown for scheduled matches.
    assert out["ht_home_goals"].isna().all()
    # Only one matchday (distinct date) requested.
    assert out["date"].nunique() == 1


def test_build_fixtures_all_scheduled():
    out = fixtures.build_fixtures(config.IN_PROGRESS_SEASON)
    scheduled = fixtures.load_scheduled(config.IN_PROGRESS_SEASON)
    assert len(out) == len(scheduled)


def test_build_fixtures_with_pipeline_labels():
    # Real teams of the first scheduled matchday of the in-progress season.
    md1 = fixtures.load_scheduled(config.IN_PROGRESS_SEASON)
    first_date = md1["date"].iloc[0]
    teams = sorted(
        set(md1.loc[md1["date"] == first_date, "home_team"])
        | set(md1.loc[md1["date"] == first_date, "away_team"]))
    labels = {i: t for i, t in enumerate(teams)}
    out = fixtures.build_fixtures(
        config.IN_PROGRESS_SEASON, n_matchdays=1, labels=labels)
    # Only the labelled teams can appear, encoded with the supplied map.
    assert set(out["home_team"]) | set(out["away_team"]) <= set(labels)
    # Unknown team -> ValueError.
    with pytest.raises(ValueError, match="not in pipeline label map"):
        fixtures.build_fixtures(
            config.IN_PROGRESS_SEASON, n_matchdays=1,
            labels={i: t for i, t in enumerate(teams[:-1])})


def test_build_fixtures_no_scheduled_raises():
    with pytest.raises(ValueError, match="no scheduled matches"):
        fixtures.build_fixtures("2016-17")


def test_build_fixtures_form_does_not_decay():
    # Regression: fixture rows have NaN goals, which _form_features treats
    # as 0 pts / 0 goals. When form was computed on the combined
    # (played + scheduled) frame, each team's own future scheduled matches
    # polluted the rolling window of later fixtures and form decayed to 0
    # by mid-season, making the model predict base-rate draws.
    out = fixtures.build_fixtures(config.IN_PROGRESS_SEASON)
    dates = sorted(out["date"].unique())
    # Form on the last scheduled matchday must still carry real signal:
    # not all teams at 0 points (the polluted-form signature).
    last = out[out["date"] == dates[-1]]
    assert last["home_form_pts"].max() > 0
    assert last["away_form_pts"].max() > 0
    # And form must be constant per team across the season (each fixture
    # gets the team's latest real-match form, not a decaying window).
    for col, team_col in (("home_form_pts", "home_team"),
                          ("away_form_pts", "away_team")):
        per_team = out.groupby(team_col)[col].nunique()
        assert (per_team <= 1).all()


def test_build_fixtures_shots_xg_are_team_specific():
    # Regression: filling every fixture with the league-wide averages made
    # shots_diff / xg_diff identical for all matches, removing the
    # team-strength signal from those features.
    out = fixtures.build_fixtures(config.IN_PROGRESS_SEASON)
    assert out["xg_diff"].nunique() > 1
    assert out["shots_diff"].nunique() > 1
    assert out["xg_home"].std() > 0
    assert out["xg_away"].std() > 0


# ---------------------------------------------------------------------------
# evaluate.table_to_string
# ---------------------------------------------------------------------------

def test_table_to_string():
    frame = pd.DataFrame({"model": ["a", "b"], "acc": [0.5, 0.6]})
    text = evaluate.table_to_string(frame)
    assert "model" in text and "acc" in text
    assert "a" in text and "0.5" in text


# ---------------------------------------------------------------------------
# run_training (fast path: no tuning, no walk-forward)
# ---------------------------------------------------------------------------

def test_run_training_returns_structured_results(tmp_path):
    res = train.run_training(
        skip_tuning=True, walk_forward=False, out_dir=tmp_path)
    # Result frames non-empty.
    assert len(res["clf_frame"]) > 0
    assert len(res["reg_frame"]) > 0
    assert res["wf_clf_frame"] is None
    assert res["best_name"]
    # Artifacts written under out_dir.
    assert (tmp_path / "results" / "classification_single_split.csv").exists()
    assert (tmp_path / "results" / "regression_single_split.csv").exists()
    pipeline_path = res["saved"]["pipeline"]
    assert pipeline_path.exists()
    assert pipeline_path.name == "pipeline.joblib"
    # Metadata sanity.
    assert res["n_rows"] == 3879
    assert res["n_teams"] == 34
    assert res["n_features"] == len(data.FEATURES)
    assert res["elapsed"] > 0


# ---------------------------------------------------------------------------
# Streamlit app module import (bare mode, no server)
# ---------------------------------------------------------------------------

def test_app_module_imports():
    import importlib
    app = importlib.import_module("src.ui.app")
    for fn in ("main", "sidebar", "tab_update", "tab_train", "tab_predict"):
        assert callable(getattr(app, fn))
