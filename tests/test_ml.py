"""Tests for the ML package (src/ml)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml import data, evaluate, models, splits


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
    assert len(feats) == 18
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
        m = evaluate.regression_metrics(valid["home_goals"].to_numpy(), pred)
        assert m["mae"] < 1.5
