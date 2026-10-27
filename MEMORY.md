# AModel — Project Memory

Persistent notes for the Serie A match-data pipeline (2016-17 → 2026-27).
Keep this file updated as the project evolves (see AGENTS.md).

## What this is

Python pipeline that extracts Serie A match data from two hybrid sources,
normalizes it, and loads it into SQLite (`data/amodel.db`) with CSV exports
and validation. Plan: `implementation_plan.md`.

- **Primary source**: football-data.co.uk CSVs — scores, HT, shots, referee.
- **Secondary source**: Understat — xG (and cross-checks).
- Seasons: 2016-17 … 2026-27 (11). `2026-27` is in progress (partial, 80 matches as of 2026-10-27).

## Environment

- Python **3.12.15** in `.venv/` (created with `uv`; system Python 3.14.4 lacks ensurepip/pip).
- **Always run as `.venv/bin/python ...`** (space-prefixed per AGENTS.md).
- Deps in `requirements.txt`: pandas, requests, beautifulsoup4, lxml, pytest.
- Temp files go in `.tmp/` only. No git.

## Data sources — critical quirks (learned the hard way)

### football-data.co.uk
- Correct URL: `https://www.football-data.co.uk/mmz4281/{YYMM}/I1.csv`
  - Serie A = **`I1`** (Italy tier 1). The plan's original `SA0` is actually the **Scottish Premiership** — this caused 608 rows/season + Scottish teams until fixed.
  - `{YYMM}` = start-year last-2-digits + end-year last-2-digits (e.g. `1617` for 2016-17). See `config.football_data_code()`.
- Plain `requests` works with a browser User-Agent.
- **BOM quirk**: files from **2024-25 onward** carry a *double-encoded* BOM (raw bytes `c3af c2bb c2bf` = literal "ï»¿") that corrupts the first column name (`Div`). Handled in `src/clean/merge.py` via `encoding="utf-8-sig"` + `_clean_columns()` (strips `\ufeff` and "ï»¿" from column names).
- **xG availability**: `HxG`/`AxG` columns exist **only in 2026-27**. No xG columns in 2016-17 … 2025-26. So xG comes from Understat for all seasons except 2026-27.
- Column layout varies by season (64 cols in 2016-17, 105+ in later years) — always read defensively.

### Understat
- Data loads via **AJAX endpoint** `https://understat.com/getLeagueData/Serie_A/{start_year}` — **NOT** the HTML page (which no longer embeds `var data =`).
- League slug is **`Serie_A`** (not `Italy`).
- Requires headers beyond UA: `X-Requested-With: XMLHttpRequest` and `Referer: https://understat.com/league/Serie_A` (UA alone → 404). See `src/fetch/understat.py` `XHR_HEADERS`.
- JSON shape: `{teams, players, dates}`. `dates` is a list of match dicts:
  `{id, isResult, h:{id,title,short_title}, a:{...}, goals:{h,a}, xG:{h,a}, datetime:"YYYY-MM-DD HH:MM:SS", forecast:{w,d,l}}`.
- **`isResult` quirk**: `dates` contains **380 entries per season INCLUDING scheduled (future) matches**. Filter with `isResult` (skip when `isResult is False`). All 300 `isResult=False` entries are in the 2026 file (scheduled 2026-27 fixtures).
- Files are keyed by **season-start year** (`2019.json` = 2019-20 season). The 2019-20 file runs to 2020-08-02 (pandemic delay).

## Pipeline architecture

```
config.py                 # seasons, paths, URLs, fetch settings (single source of truth)
src/fetch/http.py         # get_text() with retry/backoff, polite_sleep()
src/fetch/football_data_csv.py  # fetch_season/fetch_all -> data/raw/football_data/{season}.csv
src/fetch/understat.py    # fetch -> data/raw/understat/{year}.json
src/clean/normalize.py    # canonical_team(), parse_date(), season_from_date(), to_int/float_or_none
src/clean/merge.py        # load_football_data(), load_understat(), merge() -> unified DataFrame
src/db/schema.sql         # DDL: teams, matches, seasons, players, player_match_stats
src/db/upsert.py          # get_conn(), load() -> idempotent upsert + seasons summary
src/export/csv_export.py  # export() -> data/export/matches_full.csv + per-season CSVs
src/validate/checks.py    # run_checks() -> .tmp/validation_report.txt, exit 1 on hard errors
src/preprocess/clean.py   # ML preprocessing: clean/encode/feature-engineer -> data/export/ml/matches_clean.csv
tests/test_pipeline.py    # 10 pytest tests
tests/test_preprocess.py  # 12 pytest tests
```

### Key merge decisions
- **Alignment key = `(season_file, home, away)`**, NOT `(date, home, away)`.
  - Both sources are organized by season file, so the file's season is authoritative.
  - A date heuristic (`season_from_date`) misassigns the **pandemic 2019-20** season: its final matches were played in **Aug 2020** (month ≥ 7 → would become 2020-21). Using the file's season fixes this (2019-20 = 380, 2020-21 = 380).
  - The two sources occasionally disagree on the match date by a day (a few matches per season), which would break a date-based key.
- **Priority**: football-data for scores/shots/referee/HT; xG from football-data when present (2026-27) else Understat.
- **Flags**: `xg_mismatch_{diff}` (cross-source xG diff > 0.15, only 2026-27), `score_mismatch`, `understat_only`.
- **Canonical team map** (`normalize.py` `_CANONICAL`, ~142 aliases): maps both sources' name variants to one canonical name. Notable: `Parma Calcio 1913` (Understat) → `Parma` (football-data). Unknown names are logged, not dropped.

## Database schema (data/amodel.db)

- `teams(team_id PK, name UNIQUE, aliases)`
- `matches(match_id PK, season, date, home_team_id, away_team_id, home_goals, away_goals, ht_home_goals, ht_away_goals, shots_home, shots_away, xg_home, xg_away, referee, source, xg_source, flag, UNIQUE(season,date,home_team_id,away_team_id))` — upsert via `ON CONFLICT`.
- `seasons(season PK, start_date, end_date, matches_played, champion)` — champion = top points (3W/1D), goal-diff tiebreak, completed seasons only.
- `players`, `player_match_stats` — Phase 5 (FBref enrichment), not yet populated.

## How to run

```
.venv/bin/python -m src.fetch.football_data_csv [--force]   # fetch CSVs
.venv/bin/python -m src.fetch.understat                     # fetch xG JSON
.venv/bin/python -m src.clean.merge                         # merge (prints per-season counts)
.venv/bin/python -m src.db.upsert                           # load into SQLite
.venv/bin/python -m src.export.csv_export                   # CSV exports
.venv/bin/python -m src.validate.checks                     # validation report
.venv/bin/python -m src.preprocess.clean [--encoding label|onehot|target]
        [--keep-flagged] [--keep-metadata] [--form-window 5] [--out PATH]
.venv/bin/python -m pytest tests/ -q                        # tests
```

## Preprocessing (ML-ready frame)

`src/preprocess/clean.py` turns the merged frame (DB or export CSV) into a
modeling-ready CSV at `data/export/ml/matches_clean.csv`.

- **Cleaning**: `date` → datetime; drops `score_mismatch` rows (Sassuolo-Pescara
  2016-08-28) and HT-null rows (same row); drops `referee` (null in **all** 3880
  rows — football-data no longer publishes it) and metadata `source`/`xg_source`/
  `flag`/`season` (season is kept internally for features, dropped at the end).
  Result: **3879 rows, 0 nulls**.
- **Encoding** (`--encoding`): `label` (default, deterministic sorted codes),
  `onehot` (home_/away_ dummies), `target` (mean goals in role).
  - With `label`, a **`team_labels.csv`** (code → team, 34 rows) is written next
    to the output so codes can be decoded. `label_codes()` is the single source
    of truth: the map and the encoded frame always agree.
- **Features**: `month`, `day_of_week`, `matchday` (dense rank of dates within
  season — a season spans ~100-135 unique dates, so max ≈ 135, not 38),
  `shots_diff`, `xg_diff`, and leakage-free last-5 form per team
  (`home/away_form_pts/gf/ga`, `form_pts_diff`) via `shift(1).rolling(5)` on
  strictly earlier same-season matches (first match of a season → 0.0).
- **Ordering matters**: features are computed **before** encoding (form keys on
  raw team names) and `season` is dropped **after** features.
- pandas 3.1: use `dt.day_of_week` (not `dt.dayofweek`).

## Current status (2026-10-27)

- **3880 matches** in DB: 380 × 10 completed seasons + 80 (in-progress 2026-27).
- **34 teams**, 11 seasons.
- **Validation: 0 hard errors**, 4 legitimate warnings:
  - 2019-20: 114 matches with month outside Aug–May (pandemic — season ran to Aug 2020).
  - 2022-23: 10 matches in June (season ended 2023-06-04).
  - 2023-24: 1 match in June (ended 2024-06-02).
  - 2026-27: 52 cross-source xG mismatches > 0.15 (football-data vs Understat xG models differ).
- **22/22 tests pass** (10 pipeline + 12 preprocessing).
- Spot-checks verified: 2024-25 champion **Napoli** (82 pts), Bologna 5-0 Lazio 2025-03-16, Inter 4-4 Juventus 2024-10-27.
- Champions: 2016-17/17-18/18-19/19-20 Juventus, 2020-21 Inter, 2021-22 AC Milan, 2022-23 Napoli, 2023-24 Inter, 2024-25 Napoli, 2025-26 Inter.

## Known data quirks (accepted, flagged)

- **Sassuolo 0-3 Pescara (2016-08-28)**: football-data says 0-3, Understat says 2-1. Genuine source discrepancy → `score_mismatch` flag.
- **Inter 6-2 Pisa (2026-01-23)**: xG 6.24 — legitimate for a 6-goal match; xG sanity bound relaxed to [0, 8].

## Next (Phase 5)

- FBref player-level enrichment → `players` + `player_match_stats` (Playwright MCP, optional `playwright` dep).
