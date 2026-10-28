"""Tests for the incremental fetch behaviour of the two fetch modules.

The in-progress season is the only raw file that gains new matches (the
sources update it in place), so it must always be re-downloaded even with
``force=False``; completed seasons stay cached. ``force=True`` re-downloads
everything. Both ``src.fetch.football_data_csv`` and ``src.fetch.understat``
implement the same rule, so both are covered.
"""
from __future__ import annotations

import json
import shutil
import uuid

import pytest

import config
from src.fetch import football_data_csv as fd
from src.fetch import understat as us

FAKE_CSV = "I1,Date,HomeTeam,AwayTeam\nI1,2016-08-19,TeamA,TeamB\n"
FAKE_JSON = json.dumps({
    "dates": [
        {"isResult": False, "h": {"title": "TeamA"}, "a": {"title": "TeamB"},
         "datetime": "2026-08-15 18:30:00"},
    ]
})


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """Redirect config.league_dir to a throwaway dir under the workspace.

    Under the workspace so ``out_path.relative_to(config.WORKSPACE)`` in the
    fetch modules' print statements keeps working.
    """
    base = config.TMP_DIR / "test_fetch"
    base.mkdir(parents=True, exist_ok=True)
    d = base / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "league_dir", lambda source, league: d / source / league)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# football-data.co.uk
# ---------------------------------------------------------------------------

def _fd_path(raw_dir, season):
    return raw_dir / "football_data" / "serie_a" / f"{season}.csv"


def test_fd_completed_cached_skip(raw_dir, monkeypatch):
    season = "2016-17"  # completed
    out = _fd_path(raw_dir, season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FAKE_CSV, encoding="utf-8")
    calls = []
    monkeypatch.setattr(fd, "get_text", lambda url: calls.append(url) or FAKE_CSV)
    result = fd.fetch_season(season, league="serie_a", force=False)
    assert result == str(out)
    assert calls == []  # cached -> not re-downloaded


def test_fd_in_progress_refresh(raw_dir, monkeypatch):
    season = config.IN_PROGRESS_SEASON  # in progress
    out = _fd_path(raw_dir, season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FAKE_CSV, encoding="utf-8")
    calls = []
    monkeypatch.setattr(fd, "get_text", lambda url: calls.append(url) or FAKE_CSV)
    result = fd.fetch_season(season, league="serie_a", force=False)
    assert result == str(out)
    assert len(calls) == 1  # in progress -> re-downloaded


def test_fd_force_refresh(raw_dir, monkeypatch):
    season = "2016-17"  # completed, but force=True
    out = _fd_path(raw_dir, season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FAKE_CSV, encoding="utf-8")
    calls = []
    monkeypatch.setattr(fd, "get_text", lambda url: calls.append(url) or FAKE_CSV)
    result = fd.fetch_season(season, league="serie_a", force=True)
    assert result == str(out)
    assert len(calls) == 1  # force -> re-downloaded


def test_fd_404_returns_none(raw_dir, monkeypatch):
    season = "2016-17"

    def fake_get_text(url):
        raise FileNotFoundError("404 Not Found")

    monkeypatch.setattr(fd, "get_text", fake_get_text)
    result = fd.fetch_season(season, league="serie_a", force=True)
    assert result is None


# ---------------------------------------------------------------------------
# Understat
# ---------------------------------------------------------------------------

def _us_path(raw_dir, season):
    return raw_dir / "understat" / "serie_a" / f"{config.understat_year(season)}.json"


def test_us_completed_cached_skip(raw_dir, monkeypatch):
    season = "2016-17"  # completed
    out = _us_path(raw_dir, season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FAKE_JSON, encoding="utf-8")
    calls = []
    monkeypatch.setattr(us, "get_text", lambda url, **kw: calls.append(url) or FAKE_JSON)
    result = us.fetch_season(season, league="serie_a", force=False)
    assert result == str(out)
    assert calls == []  # cached -> not re-downloaded


def test_us_in_progress_refresh(raw_dir, monkeypatch):
    season = config.IN_PROGRESS_SEASON  # in progress
    out = _us_path(raw_dir, season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FAKE_JSON, encoding="utf-8")
    calls = []
    monkeypatch.setattr(us, "get_text", lambda url, **kw: calls.append(url) or FAKE_JSON)
    result = us.fetch_season(season, league="serie_a", force=False)
    assert result == str(out)
    assert len(calls) == 1  # in progress -> re-downloaded


def test_us_force_refresh(raw_dir, monkeypatch):
    season = "2016-17"  # completed, but force=True
    out = _us_path(raw_dir, season)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(FAKE_JSON, encoding="utf-8")
    calls = []
    monkeypatch.setattr(us, "get_text", lambda url, **kw: calls.append(url) or FAKE_JSON)
    result = us.fetch_season(season, league="serie_a", force=True)
    assert result == str(out)
    assert len(calls) == 1  # force -> re-downloaded


def test_us_404_returns_none(raw_dir, monkeypatch):
    season = "2016-17"

    def fake_get_text(url, **kw):
        raise FileNotFoundError("404 Not Found")

    monkeypatch.setattr(us, "get_text", fake_get_text)
    result = us.fetch_season(season, league="serie_a", force=True)
    assert result is None
