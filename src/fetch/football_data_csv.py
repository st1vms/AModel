"""Download Serie A (SA0.csv) per season from football-data.co.uk.

Usage: python -m src.fetch.football_data_csv
"""
import sys

import config
from src.fetch.http import get_text, polite_sleep


def fetch_season(season: str, force: bool = False) -> str | None:
    """Download one season's CSV. Returns path, or None if not available (404)."""
    config.FOOTBALL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.FOOTBALL_DATA_DIR / f"{season}.csv"
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        print(f"[skip] {season} (cached)")
        return str(out_path)

    url = config.FOOTBALL_DATA_URL.format(code=config.football_data_code(season))
    try:
        text = get_text(url)
    except FileNotFoundError:
        print(f"[404]  {season} — not published yet")
        return None

    # Sanity check: the first column of the first DATA row must be "I1" (Italy tier 1).
    lines = [ln for ln in text.split("\n") if ln.strip()]
    div = lines[1].split(",")[0].lstrip("\ufeff") if len(lines) > 1 else "?"
    if div != "I1":
        print(f"[warn] {season}: unexpected Div '{div}' (expected I1) — check URL")

    out_path.write_text(text, encoding="utf-8")
    print(f"[ok]   {season} (Div={div}) -> {out_path.relative_to(config.WORKSPACE)}")
    return str(out_path)


def fetch_all(force: bool = False):
    for i, season in enumerate(config.SEASONS):
        if i > 0:
            polite_sleep()
        fetch_season(season, force=force)


if __name__ == "__main__":
    force = "--force" in sys.argv
    fetch_all(force=force)
