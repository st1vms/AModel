"""Fetch Understat league data and save the raw JSON.

Understat loads its data via an AJAX endpoint (the league HTML page no longer
embeds `var data = {...}`). The endpoint is:

    https://understat.com/getLeagueData/{slug}/{start_year}

which returns JSON with keys: teams, players, dates. `dates` is a list of
match dicts: {id, isResult, h:{title}, a:{title}, goals:{h,a}, xG:{h,a},
datetime:"YYYY-MM-DD HH:MM:SS", forecast}.

Files are written to data/raw/understat/{league}/{year}.json.

Usage: python -m src.fetch.understat [--league KEY] [--force]
"""
import json
import sys

import config
from src.fetch.http import get_text, polite_sleep


def _data_url(league: str) -> str:
    slug = config.LEAGUES[league]["understat_slug"]
    return f"https://understat.com/getLeagueData/{slug}/{{year}}"


def _xhr_headers(league: str) -> dict:
    # The AJAX endpoint rejects plain requests (404) unless these are present.
    slug = config.LEAGUES[league]["understat_slug"]
    return {
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{slug}",
    }


def extract_matches(data: dict) -> list[dict]:
    """Return the list of match dicts from an Understat payload.

    Handles both the current `dates` list format and the legacy `matches`
    format for robustness.
    """
    if isinstance(data.get("dates"), list):
        return data["dates"]
    if isinstance(data.get("matches"), list):
        return data["matches"]
    return []


def fetch_season(season: str, league: str = config.DEFAULT_LEAGUE, force: bool = False) -> str | None:
    """Fetch one season's Understat JSON for a league. Returns path, or None on 404."""
    spec = config.LEAGUES[league]
    out_dir = config.league_dir("understat", league)
    out_dir.mkdir(parents=True, exist_ok=True)
    year = config.understat_year(season)
    out_path = out_dir / f"{year}.json"
    # The in-progress season is the only file that gains new matches (the
    # source updates it in place), so it is always re-downloaded even when
    # force=False; completed seasons stay cached. force=True re-downloads all.
    is_in_progress = season == spec.get("in_progress")
    if out_path.exists() and out_path.stat().st_size > 0 and not force and not is_in_progress:
        print(f"[skip] {league} {season} (cached)")
        return str(out_path)
    if is_in_progress and not force:
        print(f"[refresh] {league} {season} (in progress)")

    url = _data_url(league).format(year=year)
    try:
        text = get_text(url, extra_headers=_xhr_headers(league))
    except FileNotFoundError:
        print(f"[404]  {league} {season} — not published yet")
        return None

    data = json.loads(text)
    n = len(extract_matches(data))
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[ok]   {league} {season} ({n} matches) -> {out_path.relative_to(config.WORKSPACE)}")
    return str(out_path)


def fetch_all(league: str = config.DEFAULT_LEAGUE, force: bool = False):
    for i, season in enumerate(config.LEAGUES[league]["seasons"]):
        if i > 0:
            polite_sleep()
        fetch_season(season, league=league, force=force)


if __name__ == "__main__":
    force = "--force" in sys.argv
    league = config.DEFAULT_LEAGUE
    if "--league" in sys.argv:
        league = sys.argv[sys.argv.index("--league") + 1]
    fetch_all(league=league, force=force)
