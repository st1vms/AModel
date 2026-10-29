-- AModel SQLite schema (multi-league, 2016-17 -> present)
--
-- Every table is scoped by `league` (registry key, e.g. 'serie_a'). Old
-- single-league DBs are migrated in place by src.db.upsert._migrate, which
-- backfills league='serie_a'.

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    league  TEXT NOT NULL,
    name    TEXT NOT NULL,
    aliases TEXT,                            -- comma-separated known aliases
    UNIQUE(league, name)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    league          TEXT NOT NULL,
    season          TEXT NOT NULL,
    date            TEXT NOT NULL,           -- ISO YYYY-MM-DD
    home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals      INTEGER,
    away_goals      INTEGER,
    ht_home_goals   INTEGER,
    ht_away_goals   INTEGER,
    shots_home      INTEGER,
    shots_away      INTEGER,
    shots_on_target_home INTEGER,
    shots_on_target_away INTEGER,
    xg_home         REAL,
    xg_away         REAL,
    b365h           REAL,
    b365d           REAL,
    b365a           REAL,
    b365ch          REAL,                    -- Bet365 closing (2019-20+ only)
    b365cd          REAL,
    b365ca          REAL,
    referee         TEXT,
    source          TEXT,                    -- football-data.co.uk | understat
    xg_source       TEXT,                    -- football-data.co.uk | understat
    flag            TEXT,                    -- anomaly flags, e.g. xg_mismatch_0.21
    UNIQUE(season, league, date, home_team_id, away_team_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_league_season ON matches(league, season);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);

CREATE TABLE IF NOT EXISTS seasons (
    league         TEXT NOT NULL,
    season         TEXT NOT NULL,
    start_date     TEXT,
    end_date       TEXT,
    matches_played INTEGER,
    champion       TEXT,
    PRIMARY KEY(league, season)
);

-- Phase 5 (FBref enrichment)
CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    match_id  INTEGER NOT NULL REFERENCES matches(match_id),
    player_id INTEGER NOT NULL REFERENCES players(player_id),
    team_id   INTEGER NOT NULL REFERENCES teams(team_id),
    goals     INTEGER,
    assists   INTEGER,
    xg        REAL,
    rating    REAL,
    minutes   INTEGER,
    UNIQUE(match_id, player_id)
);
