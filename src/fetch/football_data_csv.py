"""Download per-season CSVs from football-data.co.uk for a league.

Usage: python -m src.fetch.football_data_csv [--league KEY] [--force]
Files are written to data/raw/football_data/{league}/{season}.csv.
"""
import sys

import config
from src.fetch.http import get_text, polite_sleep


def fetch_season(season: str, league: str = config.DEFAULT_LEAGUE, force: bool = False) -> str | None:
    """Download one season's CSV for a league. Returns path, or None if not available (404)."""
    spec = config.LEAGUES[league]
    out_dir = config.league_dir("football_data", league)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{season}.csv"
    # The in-progress season is the only file that gains new matches (the
    # source updates it in place), so it is always re-downloaded even when
    # force=False; completed seasons stay cached. force=True re-downloads all.
    is_in_progress = season == spec.get("in_progress")
    if out_path.exists() and out_path.stat().st_size > 0 and not force and not is_in_progress:
        print(f"[skip] {league} {season} (cached)")
        return str(out_path)
    if is_in_progress and not force:
        print(f"[refresh] {league} {season} (in progress)")

    url = config.FOOTBALL_DATA_URL.format(code=config.football_data_code(season), div=spec["fd_div"])
    try:
        text = get_text(url)
    except FileNotFoundError:
        print(f"[404]  {league} {season} — not published yet")
        return None

    # Sanity check: the first column of the first DATA row must be the division code.
    lines = [ln for ln in text.split("\n") if ln.strip()]
    div = lines[1].split(",")[0].lstrip("\ufeff") if len(lines) > 1 else "?"
    if div != spec["fd_div"]:
        print(f"[warn] {league} {season}: unexpected Div '{div}' (expected {spec['fd_div']}) — check URL")

    out_path.write_text(text, encoding="utf-8")
    print(f"[ok]   {league} {season} (Div={div}) -> {out_path.relative_to(config.WORKSPACE)}")
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
