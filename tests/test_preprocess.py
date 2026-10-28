"""Tests for the preprocessing module (src/preprocess/clean.py)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import clean as pc


def _sample() -> pd.DataFrame:
    """A tiny 2-team, 2-season frame with known scores.

    2016-17:
      08-20  A 1-0 B
      08-27  B 2-1 A
      09-03  A 3-0 B
    2017-18:
      08-19  A 1-1 B
    """
    return pd.DataFrame(
        {
            "season": ["2016-17"] * 3 + ["2017-18"],
            "date": ["2016-08-20", "2016-08-27", "2016-09-03", "2017-08-19"],
            "home_team": ["A", "B", "A", "A"],
            "away_team": ["B", "A", "B", "B"],
            "home_goals": [1, 2, 3, 1],
            "away_goals": [0, 1, 0, 1],
            "ht_home_goals": [1, 1, 1, 1],
            "ht_away_goals": [0, 0, 0, 1],
            "shots_home": [10, 8, 15, 9],
            "shots_away": [5, 7, 4, 6],
            "shots_on_target_home": [4, 3, 6, 3],
            "shots_on_target_away": [2, 3, 1, 2],
            "xg_home": [1.2, 1.8, 2.5, 1.1],
            "xg_away": [0.4, 0.9, 0.3, 0.8],
            "b365h": [1.8, 3.4, 1.5, 2.1],
            "b365d": [3.3, 3.2, 3.6, 3.3],
            "b365a": [4.5, 2.1, 5.0, 3.4],
            "referee": [None, None, None, None],
            "source": ["football-data"] * 4,
            "xg_source": ["understat"] * 4,
            "flag": [None, None, None, None],
        }
    )


def test_clean_drops_flagged_and_ht_null_rows():
    df = _sample()
    df.loc[0, "flag"] = "score_mismatch"
    df.loc[1, "ht_home_goals"] = None
    out, notes = pc.clean(df)
    # row 0 (flagged) and row 1 (HT null) dropped -> 2 rows left
    assert len(out) == 2
    assert "referee" not in out.columns
    assert "season" in out.columns  # kept for feature engineering
    assert "source" not in out.columns
    assert out["date"].dtype.kind == "M"


def test_clean_keep_metadata_retains_columns():
    df = _sample()
    out, _ = pc.clean(df, keep_metadata=True)
    for col in ["season", "source", "xg_source", "flag"]:
        assert col in out.columns


def test_encode_label_is_deterministic_sorted():
    df = _sample()
    out, _ = pc.clean(df)
    enc = pc.encode_teams(out, method="label")
    # sorted: A=0, B=1
    assert enc["home_team"].tolist() == [0, 1, 0, 0]
    assert enc["away_team"].tolist() == [1, 0, 1, 1]


def test_encode_onehot_expands_columns():
    df = _sample()
    out, _ = pc.clean(df)
    enc = pc.encode_teams(out, method="onehot")
    assert "home_team" not in enc.columns
    assert "home_A" in enc.columns and "away_B" in enc.columns
    assert enc["home_A"].sum() == 3  # A is home in 3 of 4 rows


def test_encode_target_uses_role_means():
    df = _sample()
    out, _ = pc.clean(df)
    enc = pc.encode_teams(out, method="target")
    # A as home: goals [1,3,1] -> mean 5/3
    assert enc["home_team"].iloc[0] == pytest.approx(5 / 3)
    # B as away: goals [0,1,0] -> mean 1/3
    assert enc["away_team"].iloc[0] == pytest.approx(1 / 3)


def test_form_features_leakage_free():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)

    # First match of 2016-17: no history -> form 0.0
    first = feat.iloc[0]
    assert first["home_form_pts"] == 0.0
    assert first["away_form_pts"] == 0.0

    # Second match (08-27, B home vs A):
    #   B's only prior match is 08-20 (B away, lost 0-1) -> 0 pts
    #   A's only prior match is 08-20 (A home, won 1-0) -> 3 pts
    second = feat.iloc[1]
    assert second["home_form_pts"] == 0.0   # B
    assert second["away_form_pts"] == 3.0   # A

    # Third match (09-03, A home vs B):
    #   A's prior: 08-20 win (3), 08-27 loss (0) -> mean 1.5
    #   B's prior: 08-20 loss (0), 08-27 win (3) -> mean 1.5
    third = feat.iloc[2]
    assert third["home_form_pts"] == pytest.approx(1.5)
    assert third["away_form_pts"] == pytest.approx(1.5)

    # New season (2017-18): form resets to 0.0 (no history in this season)
    fourth = feat.iloc[3]
    assert fourth["home_form_pts"] == 0.0
    assert fourth["away_form_pts"] == 0.0


def test_rest_days_leakage_free():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)

    # First match of the season: no history -> 7.0 for both sides.
    first = feat.iloc[0]
    assert first["rest_days_home"] == 7.0
    assert first["rest_days_away"] == 7.0
    assert first["is_midweek"] == 0

    # Second match (08-27): 7 days after 08-20 for both teams.
    second = feat.iloc[1]
    assert second["rest_days_home"] == 7.0
    assert second["rest_days_away"] == 7.0

    # Third match (09-03): A last played 08-27 (7 days), B last played 08-27 (7 days).
    third = feat.iloc[2]
    assert third["rest_days_home"] == 7.0
    assert third["rest_days_away"] == 7.0

    # New season (2017-18): rest resets to 7.0 (no history in this season).
    fourth = feat.iloc[3]
    assert fourth["rest_days_home"] == 7.0
    assert fourth["rest_days_away"] == 7.0


def test_market_probs_devigged_and_sum_to_one():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)

    # All three probabilities present, in (0, 1), and sum to 1.
    for col in ("mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"):
        assert col in feat.columns
        assert (feat[col] > 0).all() and (feat[col] < 1).all()
    total = feat["mkt_prob_home"] + feat["mkt_prob_draw"] + feat["mkt_prob_away"]
    assert (total - 1.0).abs().max() < 1e-9

    # De-vigging: implied probs from raw odds are scaled down by the overround,
    # so each market prob is <= the raw implied prob.
    raw_h = 1.0 / out["b365h"]
    assert (feat["mkt_prob_home"] <= raw_h + 1e-9).all()


def test_market_probs_fallback_when_missing():
    df = _sample()
    out, _ = pc.clean(df)
    out["b365h"] = None
    out["b365d"] = None
    out["b365a"] = None
    feat = pc.add_features(out)
    for col in ("mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"):
        assert (feat[col] == pytest.approx(1.0 / 3.0)).all()


def test_elo_pre_match_and_persistent():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)

    # First match: both teams at the initial 1500.
    first = feat.iloc[0]
    assert first["elo_home"] == 1500.0
    assert first["elo_away"] == 1500.0
    assert first["elo_diff"] == 0.0

    # After A's 1-0 win, A's rating rises above 1500 and B's falls below.
    # The second match's pre-match ratings reflect that update.
    second = feat.iloc[1]
    assert second["elo_home"] < 1500.0   # B (lost)
    assert second["elo_away"] > 1500.0   # A (won)

    # Elo persists across seasons: 2017-18 ratings are not reset to 1500.
    fourth = feat.iloc[3]
    assert fourth["elo_home"] != 1500.0
    assert fourth["elo_away"] != 1500.0


def test_xg_rolling_and_finishing_luck_leakage_free():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)

    # First match: no history -> rolling values 0.0.
    first = feat.iloc[0]
    assert first["xg_diff_rolling"] == 0.0
    assert first["finishing_luck_home"] == 0.0
    assert first["finishing_luck_away"] == 0.0

    # Second match: A's prior luck = goals(1) - xg(1.2) = -0.2; B's = 0 - 0.4 = -0.4.
    second = feat.iloc[1]
    assert second["finishing_luck_home"] == pytest.approx(-0.4)  # B
    assert second["finishing_luck_away"] == pytest.approx(-0.2)  # A


def test_xg_eff_falls_back_to_sot_proxy():
    df = _sample()
    out, _ = pc.clean(df)
    out["xg_home"] = None
    out["xg_away"] = None
    feat = pc.add_features(out)
    # With xG missing, xg_eff = shots_on_target * 0.33, so finishing luck is
    # non-trivial (goals - SoT*0.33) rather than all zeros.
    assert feat["finishing_luck_home"].abs().sum() > 0
    assert feat["xg_diff_rolling"].abs().sum() > 0


def test_soe_form_and_ppg_rolling_leakage_free():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)

    # First match: no history -> 0.0.
    first = feat.iloc[0]
    assert first["home_form_soe"] == 0.0
    assert first["away_form_soe"] == 0.0
    assert first["ppg_rolling_home"] == 0.0
    assert first["ppg_rolling_away"] == 0.0

    # Second match: A's prior PPG = 3.0 (one win), B's = 0.0 (one loss).
    second = feat.iloc[1]
    assert second["ppg_rolling_home"] == pytest.approx(0.0)   # B
    assert second["ppg_rolling_away"] == pytest.approx(3.0)   # A
    # SoE form is on a 0-3 scale and bounded.
    assert 0.0 <= second["home_form_soe"] <= 3.0
    assert 0.0 <= second["away_form_soe"] <= 3.0


def test_differentials():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)
    assert (feat["shots_diff"] == feat["shots_home"] - feat["shots_away"]).all()
    assert (feat["xg_diff"] == feat["xg_home"] - feat["xg_away"]).all()
    assert (feat["form_pts_diff"] == feat["home_form_pts"] - feat["away_form_pts"]).all()


def test_temporal_features():
    df = _sample()
    out, _ = pc.clean(df)
    feat = pc.add_features(out)
    assert feat["month"].iloc[0] == 8
    assert feat["day_of_week"].iloc[0] == 5  # 2016-08-20 was a Saturday
    # matchday is a dense rank of dates within each season:
    # 2016-17 -> 1,2,3 ; 2017-18 restarts at 1
    assert feat["matchday"].tolist() == [1, 2, 3, 1]


def test_prepare_drops_season_at_end():
    df = _sample()
    out, notes, label_map = pc.prepare(df, encoding="label")
    assert "season" not in out.columns
    assert "home_form_pts" in out.columns
    assert "shots_diff" in out.columns
    assert any("label" in n for n in notes)


def test_prepare_keep_metadata_keeps_season():
    df = _sample()
    out, _, _ = pc.prepare(df, encoding="label", keep_metadata=True)
    assert "season" in out.columns


def test_label_map_matches_encoded_values():
    df = _sample()
    out, _, label_map = pc.prepare(df, encoding="label")
    # A=0, B=1 (sorted)
    assert label_map == {"A": 0, "B": 1}
    # every encoded value in the frame is a valid code
    assert set(out["home_team"]) | set(out["away_team"]) <= set(label_map.values())
    # inverse lookup reproduces the original names
    inv = {v: k for k, v in label_map.items()}
    assert out["home_team"].map(inv).tolist() == ["A", "B", "A", "A"]


def test_label_map_none_for_other_encodings():
    df = _sample()
    for method in ("onehot", "target"):
        _, _, label_map = pc.prepare(df, encoding=method)
        assert label_map is None
