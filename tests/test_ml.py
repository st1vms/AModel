"""Tests for the ML package (src/ml)."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

import config
from src.ml import data, evaluate, models, predict, splits


# ---------------------------------------------------------------------------
# derive_season
# ---------------------------------------------------------------------------

def test_derive_season_normal_years():
    assert data.derive_season(pd.Timestamp("2016-08-20")) == "2016-17"
    assert data.derive_season(pd.Timestamp("2017-05-28")) == "2016-17"
    assert data.derive_season(pd.Timestamp("2024-08-17")) == "2024-25"
    assert data.derive_season(pd.Timestamp("2025-05-25")) == "2024-25"
    assert data.derive_season(pd.Timestamp("2026-10-25")) == "2026-27"


def test_derive_season_pandemic_window():
    # 2019-20 season ran to Aug 2020: these must NOT become 2020-21.
    assert data.derive_season(pd.Timestamp("2020-06-01")) == "2019-20"
    assert data.derive_season(pd.Timestamp("2020-07-15")) == "2019-20"
    assert data.derive_season(pd.Timestamp("2020-08-02")) == "2019-20"
    # Sep 2020 onward is 2020-21.
    assert data.derive_season(pd.Timestamp("2020-09-20")) == "2020-21"


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_shapes_and_targets():
    df, labels, feats = data.load_data()
    assert len(df) == 3879
    assert len(labels) == 34
    assert len(feats) == 34
    assert df["outcome"].isin([0, 1, 2]).all()
    # Outcome consistent with goals.
    assert ((df["outcome"] == 0) == (df["home_goals"] > df["away_goals"])).all()
    assert ((df["outcome"] == 1) == (df["home_goals"] == df["away_goals"])).all()
    assert ((df["outcome"] == 2) == (df["home_goals"] < df["away_goals"])).all()
    # Chronological ordering.
    assert df["date"].is_monotonic_increasing
    # Season counts: 2019-20 must be 380 despite the pandemic dates.
    counts = data.season_counts(df)
    assert counts["2019-20"] == 380
    assert counts["2020-21"] == 380
    assert counts["2026-27"] == 80


def test_load_data_feature_variants():
    _, _, feats_ht = data.load_data(include_ht=True)
    assert "ht_home_goals" in feats_ht and "ht_away_goals" in feats_ht
    _, _, feats_no_xg = data.load_data(no_xg=True)
    assert "xg_home" not in feats_no_xg and "xg_diff" not in feats_no_xg


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def test_time_split_counts_and_leakage():
    df, _, _ = data.load_data()
    sp = splits.time_split(df)
    assert sp.train["season"].max() == "2023-24"
    assert sp.valid["season"].min() == "2024-25"
    assert len(sp.train) == 3039
    assert len(sp.valid) == 840
    assert sp.valid["date"].min() > sp.train["date"].max()


def test_walk_forward_folds():
    df, _, _ = data.load_data()
    wf = splits.walk_forward_splits(df)
    assert [s.valid["season"].iloc[0] for s in wf] == [
        "2022-23", "2023-24", "2024-25", "2025-26"]
    for s in wf:
        assert len(s.valid) == 380
        assert s.valid["date"].min() > s.train["date"].max()
        # Expanding window: each fold's train contains the previous fold's.
    assert len(wf[1].train) > len(wf[0].train)
    assert len(wf[3].train) > len(wf[2].train)


def test_split_leakage_guard_raises():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2023-12-31"]),
        "season": ["2023-24", "2023-24"],
    })
    # train = later date, valid = earlier date -> leakage.
    with pytest.raises(ValueError, match="leakage"):
        splits._check_leakage(df.iloc[:1], df.iloc[1:], "test")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_classification_metrics_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    proba = np.zeros((6, 3))
    proba[np.arange(6), y] = 1.0
    m = evaluate.classification_metrics(y, proba)
    assert m["log_loss"] < 1e-6
    assert m["accuracy"] == 1.0
    assert m["roc_auc_macro"] == pytest.approx(1.0)
    assert m["confusion_matrix"].diagonal().tolist() == [2, 2, 2]


def test_regression_metrics_perfect():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    m = evaluate.regression_metrics(y, y)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0


def test_inverse_class_weights():
    y = np.array([0, 0, 0, 1, 2])
    w = models.inverse_class_weights(y)
    # w_c = N / (3 * count_c): H=5/9, D=5/3, A=5/3
    assert w[0] == pytest.approx(5 / 9)
    assert w[1] == pytest.approx(5 / 3)
    assert w[2] == pytest.approx(5 / 3)


# ---------------------------------------------------------------------------
# Smoke test: tiny end-to-end training
# ---------------------------------------------------------------------------

def test_tiny_end_to_end():
    """Train tiny models on a 2-season slice; check prediction shapes."""
    df, _, feats = data.load_data()
    df = df[df["season"].isin(["2023-24", "2024-25"])]
    train = df[df["season"] == "2023-24"]
    valid = df[df["season"] == "2024-25"]
    X_tr, X_va = train[feats], valid[feats]
    y_tr = train["outcome"].to_numpy()
    cw = models.inverse_class_weights(y_tr)

    for name in ("xgboost", "lightgbm", "catboost"):
        clf = models.make_classifier(name, class_weights=cw,
                                     n_estimators=20, learning_rate=0.1)
        models.fit_classifier(clf, X_tr, y_tr, class_weights=cw)
        proba = clf.predict_proba(X_va)
        assert proba.shape == (len(valid), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
        m = evaluate.classification_metrics(valid["outcome"].to_numpy(), proba)
        assert 0.0 <= m["log_loss"] < 2.0
        assert 0.5 <= m["roc_auc_macro"] <= 1.0

    for name in ("xgboost", "lightgbm", "catboost"):
        reg = models.make_regressor(name, n_estimators=20, learning_rate=0.1)
        models.fit_regressor(reg, X_tr, train["home_goals"].to_numpy())
        pred = models.predict_goals(reg, X_va)
        assert pred.shape == (len(valid),)
        assert (pred >= 0).all()


# ---------------------------------------------------------------------------
# load_pipeline: signature regression (positional path vs league)
# ---------------------------------------------------------------------------

def test_load_pipeline_keyword_path(tmp_path):
    """The UI calls load_pipeline(path=...); a path string must not be
    mistaken for a league key (which doubled the path via pathlib's
    absolute-join rule and raised NotADirectoryError)."""
    fake = {"classifier_name": "xgboost", "features": ["a"]}
    p = tmp_path / "pipeline.joblib"
    joblib.dump(fake, p)

    # keyword call (what src/ui/app.py does)
    loaded = predict.load_pipeline(path=str(p))
    assert loaded["classifier_name"] == "xgboost"

    # positional path lands in the `league` slot -> clear ValueError,
    # not a confusing NotADirectoryError
    with pytest.raises(ValueError, match="unknown league"):
        predict.load_pipeline(str(p))


def test_predict_imputes_nan_ht_features(tmp_path):
    """Pre-match fixtures have NaN HT goals; predict() must impute them
    (as 0) so LogisticRegression — which rejects NaN — doesn't crash.

    Regression: PL/LL/L1 pipelines trained with include_ht=True crashed in
    the Predict tab with 'Input X contains NaN'.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "ht_home_goals": rng.integers(0, 3, size=n).astype(float),
        "ht_away_goals": rng.integers(0, 3, size=n).astype(float),
    })
    # 3 classes (H/D/A) — predict() indexes proba[:, 2]
    y = (X["x1"] > 0).astype(int) + (X["ht_home_goals"] > 1).astype(int)
    clf = LogisticRegression().fit(X, y)
    reg = LogisticRegression().fit(X, (X["x1"] > -1).astype(int))

    pipeline = {
        "features": ["x1", "ht_home_goals", "ht_away_goals"],
        "classifier": clf,
        "regressor_home": reg,
        "regressor_away": reg,
        "team_labels": {0: "A", 1: "B"},
    }
    fixtures = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-01", "2026-11-02"]),
        "home_team": [0, 1],
        "away_team": [1, 0],
        "x1": [0.5, -0.5],
        "ht_home_goals": [np.nan, np.nan],
        "ht_away_goals": [np.nan, np.nan],
    })
    prepared = predict.prepare_fixtures(fixtures, pipeline)
    out = predict.predict(prepared, pipeline)
    assert len(out) == 2
    assert out[["p_home", "p_draw", "p_away"]].notna().all().all()
    assert np.allclose(out[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# build_fixtures: form columns must be numeric (not object)
# ---------------------------------------------------------------------------

def test_build_fixtures_form_columns_numeric():
    """build_fixtures must return numeric form columns.

    Regression: Ligue 1's 2026-08-23 PSG-Rennes match existed in the DB
    twice (a football-data row and a reversed understat_only row), so PSG
    and Rennes each appeared twice on the same date. That broke
    _form_features' (season, date, team)-uniqueness assumption and produced
    object-dtype form columns that LightGBM rejected with
    'pandas dtypes must be int, float or bool'. build_fixtures now drops
    null-HT rows (matching the clean pipeline) so the duplicate is gone.
    """
    from src.ml import fixtures as ml_fixtures

    for league in ("premier_league", "la_liga", "ligue_1"):
        out = ml_fixtures.build_fixtures(season="2026-27", league=league)
        form_cols = [c for c in out.columns if "form" in c]
        assert form_cols, f"no form columns for {league}"
        for c in form_cols:
            assert pd.api.types.is_numeric_dtype(out[c]), (
                f"{league}: form column {c!r} is {out[c].dtype}, expected numeric"
            )
