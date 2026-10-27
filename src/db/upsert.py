"""Idempotent load of the merged match frame into SQLite.

Usage: python -m src.db.upsert
"""
import sqlite3

import config
from src.clean.merge import merge

SCHEMA_PATH = config.WORKSPACE / "src" / "db" / "schema.sql"


def get_conn() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _upsert_teams(conn: sqlite3.Connection, names: list[str]) -> dict[str, int]:
    """Ensure teams exist; return name -> team_id."""
    ids: dict[str, int] = {}
    for name in names:
        row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (name,)).fetchone()
        if row is None:
            cur = conn.execute("INSERT INTO teams(name) VALUES (?)", (name,))
            row = (cur.lastrowid,)
        ids[name] = row[0]
    conn.commit()
    return ids


def load() -> int:
    """Merge sources and upsert into the DB. Returns number of matches loaded."""
    df = merge()
    if df.empty:
        print("no matches to load")
        return 0

    conn = get_conn()
    team_ids = _upsert_teams(conn, sorted(set(df["home_team"]) | set(df["away_team"])))

    rows = [
        (
            r.season,
            r.date,
            team_ids[r.home_team],
            team_ids[r.away_team],
            r.home_goals,
            r.away_goals,
            r.ht_home_goals,
            r.ht_away_goals,
            r.shots_home,
            r.shots_away,
            r.xg_home,
            r.xg_away,
            r.referee,
            r.source,
            r.xg_source,
            r.flag,
        )
        for r in df.itertuples()
    ]

    conn.executemany(
        """
        INSERT INTO matches (
            season, date, home_team_id, away_team_id,
            home_goals, away_goals, ht_home_goals, ht_away_goals,
            shots_home, shots_away, xg_home, xg_away,
            referee, source, xg_source, flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season, date, home_team_id, away_team_id) DO UPDATE SET
            home_goals = excluded.home_goals,
            away_goals = excluded.away_goals,
            ht_home_goals = excluded.ht_home_goals,
            ht_away_goals = excluded.ht_away_goals,
            shots_home = excluded.shots_home,
            shots_away = excluded.shots_away,
            xg_home = excluded.xg_home,
            xg_away = excluded.xg_away,
            referee = excluded.referee,
            source = excluded.source,
            xg_source = excluded.xg_source,
            flag = excluded.flag
        """,
        rows,
    )

    # Refresh seasons summary table (champion = top points, completed seasons only)
    conn.execute("DELETE FROM seasons")
    for s in sorted(df["season"].unique()):
        start = conn.execute(
            "SELECT MIN(date) FROM matches WHERE season = ?", (s,)
        ).fetchone()[0]
        end = conn.execute(
            "SELECT MAX(date) FROM matches WHERE season = ?", (s,)
        ).fetchone()[0]
        played = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE season = ?", (s,)
        ).fetchone()[0]
        champion = None
        if s != config.IN_PROGRESS_SEASON and played == 380:
            champion = conn.execute(
                """
                SELECT t.name FROM teams t
                WHERE t.team_id IN (
                    SELECT home_team_id FROM matches WHERE season = ?
                    UNION SELECT away_team_id FROM matches WHERE season = ?
                )
                ORDER BY (
                    SELECT SUM(
                        CASE
                            WHEN m.home_team_id = t.team_id AND m.home_goals > m.away_goals THEN 3
                            WHEN m.away_team_id = t.team_id AND m.away_goals > m.home_goals THEN 3
                            WHEN m.home_team_id = t.team_id AND m.home_goals = m.away_goals THEN 1
                            WHEN m.away_team_id = t.team_id AND m.home_goals = m.away_goals THEN 1
                            ELSE 0
                        END
                    ) FROM matches m WHERE m.season = ?
                ) DESC,
                (
                    SELECT SUM(
                        CASE
                            WHEN m.home_team_id = t.team_id THEN m.home_goals - m.away_goals
                            WHEN m.away_team_id = t.team_id THEN m.away_goals - m.home_goals
                            ELSE 0
                        END
                    ) FROM matches m WHERE m.season = ?
                ) DESC
                LIMIT 1
                """,
                (s, s, s, s),
            ).fetchone()[0]
        conn.execute(
            "INSERT INTO seasons (season, start_date, end_date, matches_played, champion) "
            "VALUES (?, ?, ?, ?, ?)",
            (s, start, end, played, champion),
        )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    conn.close()
    print(f"loaded {len(rows)} matches ({n} total in DB)")
    return len(rows)


if __name__ == "__main__":
    load()
