"""Tests for the backtest / CLV module (src/ml/backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from src.ml import backtest, data, models


# ---------------------------------------------------------------------------
# Market math
# ---------------------------------------------------------------------------

def test_implied_probs_raw():
    out = backtest.implied_probs(
        pd.Series([2.0]), pd.Series([3.0]), pd.Series([3.0])
    )
    assert out["home"].iloc[0] == pytest.approx(0.5)
    assert out["draw"].iloc[0] == pytest.approx(1 / 3)
    assert out["away"].iloc[0] == pytest.approx(1 / 3)


def test_devig_probs_sum_to_one():
    out = backtest.devig_probs(
        pd.Series([2.0, 1.5]), pd.Series([3.0, 3.5]), pd.Series([3.0, 4.0])
    )
    assert out.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    # De-vigging shrinks every probability (overround > 1) but preserves
    # the ranking: home stays the favourite.
    assert out["home"].iloc[0] < 0.5
    assert out["home"].iloc[0] > out["draw"].iloc[0]


def test_devig_probs_missing_falls_back_to_third():
    out = backtest.devig_probs(
        pd.Series([2.0, np.nan]), pd.Series([3.0, 3.0]), pd.Series([3.0, 3.0])
    )
    assert out.iloc[1].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_has_closing_mask():
    df = pd.DataFrame(
        {
            "b365ch": [2.0, np.nan, 2.0],
            "b365cd": [3.0, 3.0, np.nan],
            "b365ca": [3.0, 3.0, 3.0],
        }
    )
    assert backtest.has_closing_mask(df).tolist() == [True, False, False]


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def test_log_loss_perfect_and_uniform():
    y = np.array([0, 1, 2])
    perfect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert backtest.log_loss(y, perfect) == pytest.approx(0.0, abs=1e-12)
    uniform = np.full((3, 3), 1 / 3)
    assert backtest.log_loss(y, uniform) == pytest.approx(np.log(3))


def test_brier_score_perfect_and_uniform():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)
    assert backtest.brier_score(y, perfect) == pytest.approx(0.0, abs=1e-12)
    uniform = np.full((3, 3), 1 / 3)
    # Each row: sum of (1/3 - onehot)^2 = 2*(1/3)^2 + (2/3)^2 = 2/3.
    assert backtest.brier_score(y, uniform) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# ROI
# ---------------------------------------------------------------------------

def test_roi_flat_hand_computed():
    # One win @ 2.0 and one loss: gross ROI = (1 - 1)/2 = 0.
    wins = pd.Series([True, False])
    odds = pd.Series([2.0, 2.0])
    assert backtest.roi_flat(wins, odds) == pytest.approx(0.0)
    # With 3% vig: (2*0.97 - 1 - 1)/2 = -0.03.
    assert backtest.roi_flat(wins, odds, vig=0.03) == pytest.approx(-0.03)


def test_roi_kelly_no_edge_is_zero():
    # p*odds == 1 -> f = 0 -> bankroll untouched.
    p = pd.Series([0.5, 0.5])
    odds = pd.Series([2.0, 2.0])
    wins = pd.Series([True, False])
    assert backtest.roi_kelly(p, odds, wins) == pytest.approx(0.0)


def test_roi_kelly_positive_edge_grows():
    # p=0.6 @ 2.0: f = 0.2, quarter-Kelly = 0.05.
    # Win then loss: 1.05 * 0.95 = 0.9975 -> ROI -0.0025.
    p = pd.Series([0.6, 0.6])
    odds = pd.Series([2.0, 2.0])
    wins = pd.Series([True, False])
    assert backtest.roi_kelly(p, odds, wins, fraction=0.25) == pytest.approx(
        1.05 * 0.95 - 1.0
    )


# ---------------------------------------------------------------------------
# Betting frame
# ---------------------------------------------------------------------------

def _tiny_preds() -> pd.DataFrame:
    """Minimal per-match frame for build_betting_frame (4 matches)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-08-22", "2020-08-22", "2020-08-29", "2020-08-29"]
            ),
            "season": ["2020-21"] * 4,
            "home_team": ["A", "B", "C", "D"],
            "away_team": ["B", "A", "D", "C"],
            "outcome": [0, 0, 1, 2],
            "b365h": [2.0, 3.0, 2.5, 2.0],
            "b365d": [3.0, 3.0, 3.0, 3.0],
            "b365a": [3.0, 2.0, 3.0, 3.0],
            "b365ch": [1.9, np.nan, 2.4, 1.8],
            "b365cd": [3.1, np.nan, 3.1, 3.2],
            "b365ca": [3.2, np.nan, 3.2, 3.4],
            "p_home": [0.55, 0.20, 0.30, 0.60],
            "p_draw": [0.25, 0.30, 0.40, 0.20],
            "p_away": [0.20, 0.50, 0.30, 0.20],
        }
    )


def test_build_betting_frame_picks_and_clv():
    df = backtest.build_betting_frame(_tiny_preds())
    # argmax picks: home, away, draw, home.
    assert df["pick"].tolist() == [0, 2, 1, 0]
    # p_pick is the probability of the picked outcome.
    assert df["p_pick"].tolist() == pytest.approx([0.55, 0.50, 0.40, 0.60])
    # Confidence bin on the favourite side.
    assert df["confidence_bin"].tolist() == [
        "0.50-0.70", "0.50-0.70", "<0.50", "0.50-0.70"
    ]
    # Closing odds of the model pick (NaN where no closing).
    assert df["pick_odds_closing"].iloc[0] == pytest.approx(1.9)
    assert np.isnan(df["pick_odds_closing"].iloc[1])
    assert df["pick_odds_closing"].iloc[2] == pytest.approx(3.1)
    assert df["pick_odds_closing"].iloc[3] == pytest.approx(1.8)
    # Wins: match 0 home win (True), match 2 draw (True), others False.
    assert df["win"].tolist() == [True, False, True, False]
    # CLV: p_pick / p_closing_devigged - 1, finite where closing exists.
    assert np.isfinite(df["clv_pick"].iloc[0])
    assert np.isnan(df["clv_pick"].iloc[1])
    # Baseline: market favourite at closing (opening fallback pre-2019-20).
    # Match 2 closing 2.4/3.1/3.2 -> home is the favourite.
    assert df["base_pick"].tolist() == [0, 2, 0, 0]
    assert df["base_odds"].iloc[1] == pytest.approx(2.0)  # opening fallback


def test_build_betting_frame_baseline_uses_closing_when_present():
    df = backtest.build_betting_frame(_tiny_preds())
    # Match 0: closing fav is home (1.9) -> stake at closing 1.9.
    assert df["base_odds"].iloc[0] == pytest.approx(1.9)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def test_build_summary_segments_and_scopes():
    df = backtest.build_betting_frame(_tiny_preds())
    summary = backtest.build_summary(df, vig=0.03, kelly_fraction=0.25)
    scopes = set(summary["scope"])
    assert scopes == {"model", "baseline"}
    segments = set(summary["segment"])
    assert segments == {
        "all",
        "has_closing",
        "bin_<0.50",
        "bin_0.50-0.70",
        "bin_>=0.70",
        "odds_1.00-1.50",
        "odds_1.50-1.90",
        "odds_1.90-2.50",
        "odds_>2.50",
        "odds_>1.90",
    }
    # Model rows carry CLV/loss columns; baseline rows do not.
    model_all = summary[
        (summary["scope"] == "model") & (summary["segment"] == "all")
    ].iloc[0]
    assert model_all["n"] == 4
    assert np.isfinite(model_all["mean_clv"])
    base_all = summary[
        (summary["scope"] == "baseline") & (summary["segment"] == "all")
    ].iloc[0]
    assert np.isnan(base_all["mean_clv"])
    # ROI defined on all rows for the baseline (opening fallback),
    # but only on has_closing rows for the model.
    assert np.isfinite(base_all["roi_flat_gross"])
    model_hc = summary[
        (summary["scope"] == "model") & (summary["segment"] == "has_closing")
    ].iloc[0]
    assert model_hc["n"] == 3
    assert np.isfinite(model_hc["roi_flat_gross"])


# ---------------------------------------------------------------------------
# Odds Stratification
# ---------------------------------------------------------------------------

def test_assign_odds_bracket():
    # Test scalars
    assert backtest.assign_odds_bracket(1.25) == "1.00-1.50"
    assert backtest.assign_odds_bracket(1.50) == "1.50-1.90"
    assert backtest.assign_odds_bracket(1.85) == "1.50-1.90"
    assert backtest.assign_odds_bracket(1.90) == "1.90-2.50"
    assert backtest.assign_odds_bracket(2.50) == "1.90-2.50"
    assert backtest.assign_odds_bracket(2.51) == ">2.50"
    assert backtest.assign_odds_bracket(5.00) == ">2.50"
    assert backtest.assign_odds_bracket(np.nan) is None
    assert backtest.assign_odds_bracket(0.8) is None

    # Test Series
    s = pd.Series([1.10, 1.70, 2.10, 3.50, np.nan])
    res = backtest.assign_odds_bracket(s)
    assert res.iloc[0] == "1.00-1.50"
    assert res.iloc[1] == "1.50-1.90"
    assert res.iloc[2] == "1.90-2.50"
    assert res.iloc[3] == ">2.50"
    assert pd.isna(res.iloc[4])


def test_build_betting_frame_odds_bracket():
    df = backtest.build_betting_frame(_tiny_preds())
    assert "odds_bracket" in df.columns
    assert "odds_bin" in df.columns
    # Match 0: 1.9 -> 1.90-2.50
    # Match 1: NaN -> NaN
    # Match 2: 3.1 -> >2.50
    # Match 3: 1.8 -> 1.50-1.90
    assert df["odds_bracket"].iloc[0] == "1.90-2.50"
    assert pd.isna(df["odds_bracket"].iloc[1])
    assert df["odds_bracket"].iloc[2] == ">2.50"
    assert df["odds_bracket"].iloc[3] == "1.50-1.90"


def test_build_odds_summary():
    df = backtest.build_betting_frame(_tiny_preds())
    summary = backtest.build_odds_summary(df, vig=0.03)
    assert len(summary) == 6
    brackets = list(summary["bracket"])
    assert brackets == ["1.00-1.50", "1.50-1.90", "1.90-2.50", ">2.50", ">1.90", "All"]

    # 1.00-1.50 has 0 matches
    row_low = summary[summary["bracket"] == "1.00-1.50"].iloc[0]
    assert row_low["n_bets"] == 0
    assert np.isnan(row_low["model_accuracy"])
    assert np.isnan(row_low["model_roi_gross"])

    # 1.50-1.90 has 1 match (Match 3, loss)
    row_mid = summary[summary["bracket"] == "1.50-1.90"].iloc[0]
    assert row_mid["n_bets"] == 1
    assert row_mid["model_accuracy"] == 0.0
    assert row_mid["model_roi_gross"] == pytest.approx(-1.0)

    # 1.90-2.50 has 1 match (Match 0, win @ 1.9)
    row_high = summary[summary["bracket"] == "1.90-2.50"].iloc[0]
    assert row_high["n_bets"] == 1
    assert row_high["model_accuracy"] == 1.0
    assert row_high["model_roi_gross"] == pytest.approx(0.9)

    # >2.50 has 1 match (Match 2, win @ 3.1)
    row_top = summary[summary["bracket"] == ">2.50"].iloc[0]
    assert row_top["n_bets"] == 1
    assert row_top["model_accuracy"] == 1.0
    assert row_top["model_roi_gross"] == pytest.approx(2.1)

    # >1.90 has 2 matches (Match 0 and Match 2, both wins)
    row_hv = summary[summary["bracket"] == ">1.90"].iloc[0]
    assert row_hv["n_bets"] == 2
    assert row_hv["model_accuracy"] == 1.0
    assert row_hv["model_roi_gross"] == pytest.approx((0.9 + 2.1) / 2)


def test_format_odds_summary_markdown():
    df = backtest.build_betting_frame(_tiny_preds())
    summary = backtest.build_odds_summary(df, vig=0.03)
    md = backtest.format_odds_summary_markdown(summary, league="serie_a")

    assert "### Backtest Odds Stratification Summary (Serie A)" in md
    assert "| Odds Bracket | Bets (N) | Model Acc | Baseline Acc | Model ROI (Gross) | Model ROI (Net) | Baseline ROI | Mean CLV |" in md
    assert "| **> 1.90 (High-Value)** |" in md
    assert "High-Value Bracket Highlights (> 1.90)" in md


def test_export_odds_summary_markdown(tmp_path):
    from src.export import export_odds_summary_markdown
    df = backtest.build_betting_frame(_tiny_preds())
    summary = backtest.build_odds_summary(df, vig=0.03)
    out_file = tmp_path / "odds_report.md"
    md = export_odds_summary_markdown(summary, out_path=out_file, league="serie_a")
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8") == md


# ---------------------------------------------------------------------------
# End-to-end (small league, injected fast model)
# ---------------------------------------------------------------------------

def test_run_backtest_end_to_end(tmp_path):
    def fast_factory(X_train, y_train):
        cw = models.inverse_class_weights(y_train)
        model = models.make_classifier("logistic", class_weights=cw)
        models.fit_classifier(model, X_train, y_train, class_weights=cw)
        return model

    out = backtest.run_backtest(
        league="bundesliga",
        start="2019-20",
        end="2021-22",
        out_dir=tmp_path,
        model_factory=fast_factory,
        log=lambda *a, **k: None,
    )
    per = out["per_match"]
    # 3 validation seasons x 306 matches.
    assert len(per) == 3 * 306
    assert per["has_closing"].all()  # 2019-20 onward has closing odds
    assert "odds_bracket" in per.columns
    assert out["per_match_path"].exists()
    assert out["summary_path"].exists()
    assert out["odds_summary_path"].exists()
    assert out["odds_md_path"].exists()
    summary = out["summary"]
    assert len(summary) == 2 * 10  # 2 scopes x 10 segments
    odds_summary = out["odds_summary"]
    assert len(odds_summary) == 6
    assert "Odds Bracket" in out["odds_markdown"]
    model_all = summary[
        (summary["scope"] == "model") & (summary["segment"] == "all")
    ].iloc[0]
    assert model_all["n"] == 3 * 306
    assert np.isfinite(model_all["roi_flat_gross"])
    assert np.isfinite(model_all["mean_clv"])
