"""Idempotent load of the merged match frame into SQLite.

Usage: python -m src.db.upsert [--league KEY]
"""
import sqlite3

import config
from src.clean.merge import merge

SCHEMA_PATH = config.WORKSPACE / "src" / "db" / "schema.sql"


def _migrate(conn: sqlite3.Connection) -> None:
    """Migrate a legacy single-league DB to the multi-league schema.

    Detects the old schema (no `league` column on teams) and rebuilds the
    affected tables, backfilling league='serie_a'. No-op for new or already
    migrated DBs.
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(teams)").fetchall()]
    if not cols or "league" in cols:
        return  # no teams table yet (schema.sql creates it) or already migrated

    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute("BEGIN")
    try:
        # --- teams ---
        conn.execute(
            """
            CREATE TABLE teams_new (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                league  TEXT NOT NULL,
                name    TEXT NOT NULL,
                aliases TEXT,
                UNIQUE(league, name)
            )
            """
        )
        conn.execute(
            "INSERT INTO teams_new (team_id, league, name, aliases) "
            "SELECT team_id, 'serie_a', name, aliases FROM teams"
        )
        conn.execute("DROP TABLE teams")
        conn.execute("ALTER TABLE teams_new RENAME TO teams")

        # --- matches ---
        conn.execute(
            """
            CREATE TABLE matches_new (
                match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                league          TEXT NOT NULL,
                season          TEXT NOT NULL,
                date            TEXT NOT NULL,
                home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
                away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
                home_goals      INTEGER,
                away_goals      INTEGER,
                ht_home_goals   INTEGER,
                ht_away_goals   INTEGER,
                shots_home      INTEGER,
                shots_away      INTEGER,
                xg_home         REAL,
                xg_away         REAL,
                referee         TEXT,
                source          TEXT,
                xg_source       TEXT,
                flag            TEXT,
                UNIQUE(season, league, date, home_team_id, away_team_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO matches_new (
                match_id, league, season, date, home_team_id, away_team_id,
                home_goals, away_goals, ht_home_goals, ht_away_goals,
                shots_home, shots_away, xg_home, xg_away,
                referee, source, xg_source, flag
            )
            SELECT
                match_id, 'serie_a', season, date, home_team_id, away_team_id,
                home_goals, away_goals, ht_home_goals, ht_away_goals,
                shots_home, shots_away, xg_home, xg_away,
                referee, source, xg_source, flag
            FROM matches
            """
        )
        conn.execute("DROP TABLE matches")
        conn.execute("ALTER TABLE matches_new RENAME TO matches")
        conn.execute("CREATE INDEX idx_matches_league_season ON matches(league, season)")
        conn.execute("CREATE INDEX idx_matches_date ON matches(date)")

        # --- seasons ---
        conn.execute(
            """
            CREATE TABLE seasons_new (
                league         TEXT NOT NULL,
                season         TEXT NOT NULL,
                start_date     TEXT,
                end_date       TEXT,
                matches_played INTEGER,
                champion       TEXT,
                PRIMARY KEY(league, season)
            )
            """
        )
        conn.execute(
            "INSERT INTO seasons_new (league, season, start_date, end_date, matches_played, champion) "
            "SELECT 'serie_a', season, start_date, end_date, matches_played, champion FROM seasons"
        )
        conn.execute("DROP TABLE seasons")
        conn.execute("ALTER TABLE seasons_new RENAME TO seasons")

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys = ON")


def get_conn() -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    _migrate(conn)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return conn


def _upsert_teams(conn: sqlite3.Connection, league: str, names: list[str]) -> dict[str, int]:
    """Ensure teams exist for a league; return name -> team_id."""
    ids: dict[str, int] = {}
    for name in names:
        row = conn.execute(
            "SELECT team_id FROM teams WHERE league = ? AND name = ?", (league, name)
        ).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO teams(league, name) VALUES (?, ?)", (league, name)
            )
            row = (cur.lastrowid,)
        ids[name] = row[0]
    conn.commit()
    return ids


def load(league: str = config.DEFAULT_LEAGUE) -> int:
    """Merge sources for a league and upsert into the DB. Returns matches loaded."""
    df = merge(league)
    if df.empty:
        print(f"no matches to load for {league}")
        return 0

    conn = get_conn()
    team_ids = _upsert_teams(conn, league, sorted(set(df["home_team"]) | set(df["away_team"])))

    rows = [
        (
            league,
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
            league, season, date, home_team_id, away_team_id,
            home_goals, away_goals, ht_home_goals, ht_away_goals,
            shots_home, shots_away, xg_home, xg_away,
            referee, source, xg_source, flag
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season, league, date, home_team_id, away_team_id) DO UPDATE SET
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
    spec = config.LEAGUES[league]
    conn.execute("DELETE FROM seasons WHERE league = ?", (league,))
    for s in sorted(df["season"].unique()):
        start = conn.execute(
            "SELECT MIN(date) FROM matches WHERE league = ? AND season = ?", (league, s)
        ).fetchone()[0]
        end = conn.execute(
            "SELECT MAX(date) FROM matches WHERE league = ? AND season = ?", (league, s)
        ).fetchone()[0]
        played = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE league = ? AND season = ?", (league, s)
        ).fetchone()[0]
        champion = None
        if s != spec["in_progress"] and played == spec["expected_matches"]:
            champion = conn.execute(
                """
                SELECT t.name FROM teams t
                WHERE t.league = ? AND t.team_id IN (
                    SELECT home_team_id FROM matches WHERE league = ? AND season = ?
                    UNION SELECT away_team_id FROM matches WHERE league = ? AND season = ?
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
                    ) FROM matches m WHERE m.league = ? AND m.season = ?
                ) DESC,
                (
                    SELECT SUM(
                        CASE
                            WHEN m.home_team_id = t.team_id THEN m.home_goals - m.away_goals
                            WHEN m.away_team_id = t.team_id THEN m.away_goals - m.home_goals
                            ELSE 0
                        END
                    ) FROM matches m WHERE m.league = ? AND m.season = ?
                ) DESC
                LIMIT 1
                """,
                (league, league, s, league, s, league, s, league, s),
            ).fetchone()[0]
        conn.execute(
            "INSERT INTO seasons (league, season, start_date, end_date, matches_played, champion) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (league, s, start, end, played, champion),
        )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM matches WHERE league = ?", (league,)).fetchone()[0]
    conn.close()
    print(f"loaded {len(rows)} matches for {league} ({n} total in DB for league)")
    return len(rows)


if __name__ == "__main__":
    import sys

    league = config.DEFAULT_LEAGUE
    if "--league" in sys.argv:
        league = sys.argv[sys.argv.index("--league") + 1]
    load(league)
