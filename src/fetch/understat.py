"""Fetch Understat Serie A league data and save the raw JSON.

Understat loads its data via an AJAX endpoint (the league HTML page no longer
embeds `var data = {...}`). The endpoint is:

    https://understat.com/getLeagueData/Serie_A/{start_year}

which returns JSON with keys: teams, players, dates. `dates` is a list of
match dicts: {id, isResult, h:{title}, a:{title}, goals:{h,a}, xG:{h,a},
datetime:"YYYY-MM-DD HH:MM:SS", forecast}.

Usage: python -m src.fetch.understat
"""
import json
import sys

import config
from src.fetch.http import get_text, polite_sleep

LEAGUE_SLUG = "Serie_A"
DATA_URL = f"https://understat.com/getLeagueData/{LEAGUE_SLUG}/{{year}}"

# The AJAX endpoint rejects plain requests (404) unless these are present.
XHR_HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"https://understat.com/league/{LEAGUE_SLUG}",
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


def fetch_season(season: str, force: bool = False) -> str | None:
    """Fetch one season's Understat JSON. Returns path, or None on 404."""
    config.UNDERSTAT_DIR.mkdir(parents=True, exist_ok=True)
    year = config.understat_year(season)
    out_path = config.UNDERSTAT_DIR / f"{year}.json"
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"[skip] {season} (cached)")
        return str(out_path)

    url = DATA_URL.format(year=year)
    try:
        text = get_text(url, extra_headers=XHR_HEADERS)
    except FileNotFoundError:
        print(f"[404]  {season} — not published yet")
        return None

    data = json.loads(text)
    n = len(extract_matches(data))
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[ok]   {season} ({n} matches) -> {out_path.relative_to(config.WORKSPACE)}")
    return str(out_path)


def fetch_all(force: bool = False):
    for i, season in enumerate(config.SEASONS):
        if i > 0:
            polite_sleep()
        fetch_season(season, force=force)


if __name__ == "__main__":
    force = "--force" in sys.argv
    fetch_all(force=force)
