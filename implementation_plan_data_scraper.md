# Plan: Serie A Historical Data Extractor (2016-17 → present)

## TL;DR
Build a Python pipeline in `/home/steve/Scrivania/AModel` that extracts Serie A match data for seasons **2016-17 through 2026-27 (in progress)** using a **hybrid source strategy**: football-data.co.uk CSVs (results, shots, referee, xG from 2017-18) + Understat JSON (xG from 2014-15, covers 2016-17 gap). Data is normalized and loaded into **SQLite** (primary), with **CSV exports** as a supplementary format. An optional later phase enriches with **FBref** player-level stats (goals, assists, xG, ratings) via Playwright.

## User Decisions (from alignment)
- **Season range:** 2016-17 → present (10 seasons; robust sample, avoids outdated tactics/rule noise).
- **Data depth:** Maximum precision — results + xG + (later) player ratings/advanced stats à la FBref.
- **Storage:** SQLite primary; CSV only as supplementary export for training/sharing.
- **Source strategy:** Hybrid football-data.co.uk CSV + Understat xG (FBref as enrichment/fallback).

## Verified Source Facts
- **football-data.co.uk:** `https://www.football-data.co.uk/mmz4281/{YYMM}/SA0.csv` (e.g. `1617`, `2627`). Plain CSV, no anti-bot. Columns: `Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, HTHG, HTAG, HTR, Referee, HS, AS` + `HxG, AxG` (xG present from 2017-18 onward).
- **Understat:** `https://understat.com/league/Italy?season={YYYY}` (season = start year, 2016–2026). JSON embedded in a `<script>` tag (`var data = {...}`). Provides per-match xG, shots, big chances, xG timeline. Covers 2014-15+, so fills the 2016-17 xG gap.
- **FBref:** `https://fbref.com/en/comps/11/scores/{YYYY}-{YYYY}/...` — returns **HTTP 403 to plain requests** (Cloudflare). Needs Playwright (per AGENTS.md: Playwright MCP for browser tasks) or header spoofing. Player stats pages: `https://fbref.com/en/comps/11/stats/{YYYY}-{YYYY}-Serie-A-Stats`.
- **Wikipedia** season pages: reliable fallback for league tables/results (verified working).

## AGENTS.md Constraints to Honor
- Temp files only in `.tmp/`, cleaned up after use.
- No git commands / `.git` access.
- Keep `MEMORY.md` updated with project knowledge after each phase.
- Playwright (MCP) for browser automation; Context7 MCP for library docs.
- All writes stay inside workspace root.

## Project Structure (to create)
```
AModel/
├── requirements.txt          # requests, pandas, beautifulsoup4, lxml, playwright (optional)
├── config.py                 # season list, paths, delays, source toggles
├── src/
│   ├── __init__.py
│   ├── fetch/
│   │   ├── __init__.py
│   │   ├── football_data_csv.py   # download SA0.csv per season → data/raw/
│   │   ├── understat.py           # fetch + parse embedded JSON → data/raw/
│   │   └── fbref.py               # (Phase 5) Playwright-based player stats scraper
│   ├── clean/
│   │   ├── __init__.py
│   │   ├── normalize.py           # team-name canonical map, date parsing, season labels
│   │   └── merge.py               # join CSV + Understat into unified match frame
│   ├── db/
│   │   ├── __init__.py
│   │   ├── schema.sql             # SQLite DDL
│   │   └── upsert.py              # idempotent load into data/amodel.db
│   ├── export/
│   │   ├── __init__.py
│   │   └── csv_export.py          # per-season + full CSV exports
│   └── validate/
│       ├── __init__.py
│       └── checks.py              # completeness + anomaly validation
├── data/
│   ├── raw/                     # cached raw CSV/JSON (gitignored)
│   └── amodel.db                # SQLite DB (gitignored)
├── .tmp/                        # temp artifacts (gitignored)
└── tests/
    └── test_pipeline.py         # pytest smoke tests
```

## SQLite Schema
- `teams(team_id INTEGER PK, name TEXT UNIQUE, aliases TEXT)` — canonical names + alias list.
- `matches(match_id INTEGER PK, season TEXT, date TEXT, home_team_id, away_team_id, home_goals, away_goals, ht_home_goals, ht_away_goals, shots_home, shots_away, xg_home REAL, xg_away REAL, referee TEXT, source TEXT, UNIQUE(season, date, home_team_id, away_team_id))`.
- `seasons(season TEXT PK, start_date, end_date, matches_played, champion TEXT)`.
- (Phase 5) `players(player_id PK, name)`, `player_match_stats(match_id, player_id, team_id, goals, assists, xg, rating, minutes, UNIQUE(match_id, player_id))`.

## Steps

### Phase 1 — Scaffolding (no dependencies)
1. Create directory structure above; add `requirements.txt` (requests, pandas, beautifulsoup4, lxml; playwright commented as optional).
2. Create `config.py`: SEASONS = 2016-17…2026-27, raw/data paths, request delay (2-5s random), user-agent, source toggles.
3. Update `.gitignore`: add `data/`, `.tmp/`, `*.db`.

### Phase 2 — Core Fetchers (parallel: 4 and 5)
4. `src/fetch/football_data_csv.py`: for each season, compute `YYMM` code, download `SA0.csv` to `data/raw/football_data/{season}.csv` (skip if cached — idempotent), polite delay + retry with backoff, log HTTP errors.
5. `src/fetch/understat.py`: for each season (start-year 2016–2026), fetch `https://understat.com/league/Italy?season={year}`, extract the `var data = {...}` JSON via regex/BeautifulSoup, save to `data/raw/understat/{year}.json`.
6. `src/clean/normalize.py`: canonical team-name map (e.g. "Inter"→"Inter Milan", "Milan"→"AC Milan", "Hellas Verona"→"Hellas Verona", "Juventus" consistent), parse `Date` (DD/MM/YYYY), derive `season` label from date.

### Phase 3 — Merge + Load (depends on 4, 5, 6)
7. `src/clean/merge.py`: load both sources, align on (date, home, away); prefer football-data.co.uk for scores/shots/referee; take xG from Understat when CSV lacks it (2016-17), else prefer CSV xG; flag mismatches.
8. `src/db/schema.sql` + `src/db/upsert.py`: create tables, idempotent upsert into `data/amodel.db`.

### Phase 4 — Export + Validation (depends on 8)
9. `src/export/csv_export.py`: export `data/export/matches_full.csv` + per-season `data/export/{season}.csv`.
10. `src/validate/checks.py`: per completed season expect 380 matches; no duplicate (date, home, away); FT ≥ HT consistency; xG in [0, ~6]; cross-source xG agreement within 0.15; log anomalies to `.tmp/validation_report.txt` (then summarize into MEMORY.md).
11. `tests/test_pipeline.py`: pytest smoke tests — normalize map, merge alignment on a fixture, upsert idempotency.

### Phase 5 — FBref Player Enrichment (optional, depends on 8; heaviest lift)
12. `src/fetch/fbref.py`: Playwright-based fetch of FBref season scores + player stats pages (403 workaround), parse player goals/assists/xG/ratings, upsert into `player_match_stats`. Run per season with delays; cache HTML in `data/raw/fbref/`.

### Phase 6 — Housekeeping
13. Update `MEMORY.md`: sources, URL patterns, schema summary, known quirks (FBref 403, xG availability by season), run instructions.

## Critical Files
- `config.py` — single source of truth for seasons/paths/delays.
- `src/fetch/football_data_csv.py`, `src/fetch/understat.py` — the two core fetchers.
- `src/clean/normalize.py` — team-name canonical map (the main data-quality risk).
- `src/db/schema.sql` + `src/db/upsert.py` — storage contract.
- `src/validate/checks.py` — quality gate.

## Verification
1. Run fetchers: `python -m src.fetch.football_data_csv` and `python -m src.fetch.understat` → confirm 10 CSVs + 10 JSONs in `data/raw/`.
2. Run load: `python -m src.db.upsert` → `SELECT COUNT(*) FROM matches WHERE season='2024-25'` returns **380**; total matches ≈ 380×9 + partial 2026-27.
3. Run `python -m src.validate.checks` → 0 hard errors; review `.tmp/validation_report.txt`.
4. Spot-check known facts in SQLite: 2024-25 champion Napoli (82 pts derivable), Bologna 5-0 Lazio on 2025-03-16, Inter 4-4 Juventus on 2024-10-27.
5. `pytest tests/` passes.
6. Confirm CSV exports exist and row counts match DB.

## Decisions / Scope
- **In scope:** 2016-17 → 2026-27; results, HT/FT, shots, referee, xG; SQLite + CSV export; validation.
- **Out of scope (for now):** pre-2016-17 history, lineups, transfers, market values, live/in-season auto-updates (manual re-run suffices), FBref player stats (Phase 5, optional).
- **Assumption:** football-data.co.uk keeps the `mmz4281/{YYMM}/SA0.csv` pattern; Understat keeps embedded-JSON format. Both verified reachable as of 2026-10-27.
- 2026-27 is in progress (date 2026-10-27) → validation treats it as partial (no 380-match check).

## Further Considerations
1. If FBref 403 proves too flaky for Phase 5, fallback: use Understat's per-player xG (already in its JSON) + Wikipedia top-scorer tables for goals/assists instead of FBref.
2. Consider adding `pyarrow`/Parquet export later if model training needs columnar reads — CSV is enough for now.
3. Rate-limit budget: ~20 requests for core (10 CSV + 10 Understat) — trivial; FBref phase is the only one needing careful throttling.
