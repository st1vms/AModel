"""Export matches from SQLite to CSV (full + per-season, per-league).

Usage: python -m src.export.csv_export [--league KEY]

Layout:
- data/export/matches_full.csv            all leagues (league column included)
- data/export/{league}/matches_full.csv   one league
- data/export/{league}/{season}.csv       one league, one season
"""
import argparse
import sqlite3

import pandas as pd

import config
from src.db.upsert import get_conn

QUERY = """
SELECT
    m.league,
    m.season,
    m.date,
    th.name AS home_team,
    ta.name AS away_team,
    m.home_goals,
    m.away_goals,
    m.ht_home_goals,
    m.ht_away_goals,
    m.shots_home,
    m.shots_away,
    m.xg_home,
    m.xg_away,
    m.referee,
    m.source,
    m.xg_source,
    m.flag
FROM matches m
JOIN teams th ON th.team_id = m.home_team_id
JOIN teams ta ON ta.team_id = m.away_team_id
ORDER BY m.league, m.date, th.name
"""


def export(league: str | None = None) -> dict[str, int]:
    """Write full + per-season CSVs. Returns {path: row_count}.

    league=None exports every league in the DB; otherwise only that league.
    """
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    df = pd.read_sql_query(QUERY, conn)
    conn.close()
    if league is not None:
        df = df[df["league"] == league]

    counts: dict[str, int] = {}
    full = config.EXPORT_DIR / "matches_full.csv"
    df.to_csv(full, index=False)
    counts[str(full)] = len(df)

    for lg, lg_df in df.groupby("league"):
        lg_dir = config.EXPORT_DIR / lg
        lg_dir.mkdir(parents=True, exist_ok=True)
        lg_full = lg_dir / "matches_full.csv"
        lg_df.to_csv(lg_full, index=False)
        counts[str(lg_full)] = len(lg_df)
        for season, sub in lg_df.groupby("season"):
            path = lg_dir / f"{season}.csv"
            sub.to_csv(path, index=False)
            counts[str(path)] = len(sub)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league", default=None,
                    help="league registry key (default: all leagues in the DB)")
    args = ap.parse_args()
    for path, n in export(args.league).items():
        print(f"{n:4d}  {path}")
