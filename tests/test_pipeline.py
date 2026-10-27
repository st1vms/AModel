"""Smoke tests: normalize map, merge alignment, upsert idempotency."""
import sqlite3

import pandas as pd
import pytest

from src.clean import normalize
from src.clean.merge import merge
from src.db.upsert import get_conn, load


# ---------- normalize ----------

def test_canonical_team_known_aliases():
    assert normalize.canonical_team("Inter") == "Inter"
    assert normalize.canonical_team("Inter Milan") == "Inter"
    assert normalize.canonical_team("Milan") == "AC Milan"
    assert normalize.canonical_team("AC Milan") == "AC Milan"
    assert normalize.canonical_team("Hellas Verona") == "Hellas Verona"
    assert normalize.canonical_team("Juventus") == "Juventus"
    assert normalize.canonical_team("  Napoli  ") == "Napoli"


def test_canonical_team_bundesliga_aliases():
    """football-data short names must align with Understat full names.

    Regression: an empty Bundesliga map left 234 matches/season unaligned
    (understat_only), so their xG never merged in and LogisticRegression
    hit NaNs during training.
    """
    pairs = [
        ("Hertha", "Hertha Berlin"),
        ("Ein Frankfurt", "Eintracht Frankfurt"),
        ("M'gladbach", "Borussia M.Gladbach"),
        ("Dortmund", "Borussia Dortmund"),
        ("Leverkusen", "Bayer Leverkusen"),
        ("Hamburg", "Hamburger SV"),
        ("Hannover", "Hannover 96"),
        ("Mainz", "Mainz 05"),
        ("Nurnberg", "Nuernberg"),
        ("RB Leipzig", "RasenBallsport Leipzig"),
        ("St Pauli", "St. Pauli"),
        ("Stuttgart", "VfB Stuttgart"),
        ("FC Koln", "FC Cologne"),
        ("Heidenheim", "FC Heidenheim"),
        ("Bielefeld", "Arminia Bielefeld"),
        ("Greuther Furth", "Greuther Fuerth"),
        ("Fortuna Dusseldorf", "Fortuna Duesseldorf"),
    ]
    for fd_name, us_name in pairs:
        assert normalize.canonical_team(fd_name, "bundesliga") == \
            normalize.canonical_team(us_name, "bundesliga")


def test_canonical_team_premier_league_aliases():
    """football-data short names must align with Understat full names."""
    pairs = [
        ("Man City", "Manchester City"),
        ("Man United", "Manchester United"),
        ("Newcastle", "Newcastle United"),
        ("Nott'm Forest", "Nottingham Forest"),
        ("West Brom", "West Bromwich Albion"),
        ("Wolves", "Wolverhampton Wanderers"),
    ]
    for fd_name, us_name in pairs:
        assert normalize.canonical_team(fd_name, "premier_league") == \
            normalize.canonical_team(us_name, "premier_league")


def test_canonical_team_la_liga_aliases():
    """football-data short names must align with Understat full names."""
    pairs = [
        ("Ath Bilbao", "Athletic Club"),
        ("Ath Madrid", "Atletico Madrid"),
        ("Betis", "Real Betis"),
        ("Celta", "Celta Vigo"),
        ("Espanol", "Espanyol"),
        ("Huesca", "SD Huesca"),
        ("La Coruna", "Deportivo La Coruna"),
        ("Oviedo", "Real Oviedo"),
        ("Santander", "Racing Santander"),
        ("Sociedad", "Real Sociedad"),
        ("Sp Gijon", "Sporting Gijon"),
        ("Vallecano", "Rayo Vallecano"),
        ("Valladolid", "Real Valladolid"),
    ]
    for fd_name, us_name in pairs:
        assert normalize.canonical_team(fd_name, "la_liga") == \
            normalize.canonical_team(us_name, "la_liga")


def test_canonical_team_ligue_1_aliases():
    """football-data short names must align with Understat full names."""
    pairs = [
        ("Bastia", "SC Bastia"),
        ("Clermont", "Clermont Foot"),
        ("Paris SG", "Paris Saint Germain"),
        ("St Etienne", "Saint-Etienne"),
    ]
    for fd_name, us_name in pairs:
        assert normalize.canonical_team(fd_name, "ligue_1") == \
            normalize.canonical_team(us_name, "ligue_1")


def test_canonical_team_unknown_falls_back():
    assert normalize.canonical_team("Some New Club") == "Some New Club"
    assert normalize.canonical_team(None) == "UNKNOWN"


def test_unknown_teams():
    assert normalize.unknown_teams(["Inter", "Milan", "Brand New FC"]) == {"Brand New FC"}


def test_parse_date_formats():
    assert normalize.parse_date("27/10/2026") == "2026-10-27"
    assert normalize.parse_date("2026-10-27") == "2026-10-27"
    assert normalize.parse_date("garbage") is None
    assert normalize.parse_date(None) is None


def test_season_from_date():
    assert normalize.season_from_date("2026-08-15") == "2026-27"
    assert normalize.season_from_date("2026-10-27") == "2026-27"
    assert normalize.season_from_date("2027-05-02") == "2026-27"
    assert normalize.season_from_date("2026-01-10") == "2025-26"


def test_numeric_coercion():
    assert normalize.to_int_or_none("3") == 3
    assert normalize.to_int_or_none("") is None
    assert normalize.to_int_or_none(None) is None
    assert normalize.to_float_or_none("1.23") == 1.23
    assert normalize.to_float_or_none("nan") is None


# ---------- merge ----------

def test_merge_alignment_fixture(tmp_path, monkeypatch):
    """A football-data row and an Understat row for the same match must align."""
    import config
    import json

    # merge() reads config.league_dir(source, league) = RAW_DIR/source/league
    # and iterates config.LEAGUES[league]["seasons"].
    (tmp_path / "football_data" / "serie_a").mkdir(parents=True)
    (tmp_path / "understat" / "serie_a").mkdir(parents=True)

    csv_text = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HTR,Referee,HS,AS\n"
        "27/10/2026,Inter,Juventus,4,4,2,2,D,Ref A,15,12\n"
    )
    (tmp_path / "football_data" / "serie_a" / "2026-27.csv").write_text(
        csv_text, encoding="utf-8")

    us_data = {
        "dates": [
            {
                "id": "1",
                "isResult": True,
                "h": {"id": "1", "title": "Inter"},
                "a": {"id": "2", "title": "Juventus"},
                "goals": {"h": "4", "a": "4"},
                "xG": {"h": "2.1", "a": "1.9"},
                "datetime": "2026-10-27 20:45:00",
            }
        ]
    }
    (tmp_path / "understat" / "serie_a" / "2026.json").write_text(
        json.dumps(us_data), encoding="utf-8")

    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setitem(config.LEAGUES["serie_a"], "seasons", ["2026-27"])

    df = merge()
    assert len(df) == 1  # aligned into a single match, not two
    row = df.iloc[0]
    assert row["home_team"] == "Inter"
    assert row["away_team"] == "Juventus"
    assert row["home_goals"] == 4
    assert row["xg_home"] == 2.1
    assert row["source"] == "football-data.co.uk"
    assert row["flag"] is None or row["flag"] != "understat_only"


def test_merge_xg_fallback(tmp_path, monkeypatch):
    """When the CSV lacks xG (2016-17), Understat xG must be used."""
    import config
    import json

    (tmp_path / "football_data" / "serie_a").mkdir(parents=True)
    (tmp_path / "understat" / "serie_a").mkdir(parents=True)

    # 2016-17 CSV has no HxG/AxG columns
    csv_text = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HTR,Referee,HS,AS\n"
        "14/08/2016,Juventus,Napoli,2,1,1,0,H,Ref B,10,8\n"
    )
    (tmp_path / "football_data" / "serie_a" / "2016-17.csv").write_text(
        csv_text, encoding="utf-8")

    us_data = {
        "dates": [
            {
                "id": "1",
                "isResult": True,
                "h": {"id": "1", "title": "Juventus"},
                "a": {"id": "2", "title": "Napoli"},
                "goals": {"h": "2", "a": "1"},
                "xG": {"h": "1.5", "a": "0.8"},
                "datetime": "2016-08-14 18:00:00",
            }
        ]
    }
    (tmp_path / "understat" / "serie_a" / "2016.json").write_text(
        json.dumps(us_data), encoding="utf-8")

    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setitem(config.LEAGUES["serie_a"], "seasons", ["2016-17"])

    df = merge()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["xg_home"] == 1.5
    assert row["xg_away"] == 0.8
    assert row["xg_source"] == "understat"


def test_understat_legacy_format():
    """The legacy flat `matches` format must still parse."""
    from src.clean.merge import _understat_match_fields

    legacy = {
        "date": "2016-08-14",
        "team": "Juventus",
        "opponent": "Napoli",
        "goalsFor": 2,
        "opponentGoalsFor": 1,
        "xG": 1.5,
        "opponentxG": 0.8,
        "shots": 10,
        "opponentShots": 8,
    }
    f = _understat_match_fields(legacy)
    assert f["home"] == "Juventus"
    assert f["xg_home"] == 1.5
    assert f["date"] == "2016-08-14"

    modern = {
        "h": {"title": "Roma"},
        "a": {"title": "Udinese"},
        "goals": {"h": "4", "a": "0"},
        "xG": {"h": "5.7", "a": "0.5"},
        "datetime": "2016-08-20 20:00:00",
    }
    f2 = _understat_match_fields(modern)
    assert f2["home"] == "Roma"
    assert f2["date"] == "2016-08-20"
    assert f2["goals_home"] == "4"


# ---------- upsert idempotency ----------

def test_upsert_idempotent(tmp_path, monkeypatch):
    import config
    import json

    (tmp_path / "football_data" / "serie_a").mkdir(parents=True)
    (tmp_path / "understat" / "serie_a").mkdir(parents=True)
    (tmp_path / "football_data" / "serie_a" / "2026-27.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,HTHG,HTAG,HTR,Referee,HS,AS\n"
        "27/10/2026,Inter,Juventus,4,4,2,2,D,Ref A,15,12\n",
        encoding="utf-8",
    )
    (tmp_path / "understat" / "serie_a" / "2026.json").write_text(
        json.dumps(
            {
                "dates": [
                    {
                        "id": "1",
                        "isResult": True,
                        "h": {"id": "1", "title": "Inter"},
                        "a": {"id": "2", "title": "Juventus"},
                        "goals": {"h": "4", "a": "4"},
                        "xG": {"h": "2.1", "a": "1.9"},
                        "datetime": "2026-10-27 20:45:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    db = tmp_path / "test.db"
    monkeypatch.setattr(config, "DB_PATH", db)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setitem(config.LEAGUES["serie_a"], "seasons", ["2026-27"])

    n1 = load()
    n2 = load()
    assert n1 == n2 > 0

    conn = sqlite3.connect(db)
    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert total == n1  # second load added no duplicates
    leagues = conn.execute(
        "SELECT DISTINCT league FROM matches").fetchall()
    assert leagues == [("serie_a",)]
    conn.close()
