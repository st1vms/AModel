"""Export matches from SQLite to CSV (full + per-season).

Usage: python -m src.export.csv_export
"""
import sqlite3

import pandas as pd

import config
from src.db.upsert import get_conn

QUERY = """
SELECT
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
ORDER BY m.date, th.name
"""


def export() -> dict[str, int]:
    """Write full + per-season CSVs. Returns {path: row_count}."""
    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    df = pd.read_sql_query(QUERY, conn)
    conn.close()

    counts: dict[str, int] = {}
    full = config.EXPORT_DIR / "matches_full.csv"
    df.to_csv(full, index=False)
    counts[str(full)] = len(df)

    for season, sub in df.groupby("season"):
        path = config.EXPORT_DIR / f"{season}.csv"
        sub.to_csv(path, index=False)
        counts[str(path)] = len(sub)
    return counts


if __name__ == "__main__":
    for path, n in export().items():
        print(f"{n:4d}  {path}")
