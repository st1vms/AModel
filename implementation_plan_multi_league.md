# Implementation Plan: Multi-League Support

Status: **implemented** (2026-10-28)
Builds on: `implementation_plan_data_scraper.md` (data pipeline),
`implementation_plan_ml_models.md` (ML models), `src/ui/app.py` (Streamlit UI)

## TL;DR

Generalize the Serie A (`I1`) pipeline to **any registered league** via a
`config.LEAGUES` registry, with per-league data directories, per-league DB
rows, per-league ML artifacts, and a **League selector** in the Streamlit
sidebar. Existing Serie A data is auto-migrated; no code change is needed to
add a new league (one registry entry + fetch/train).

## Design decisions

- **Registry-driven**: `config.LEAGUES` is the single source of truth.
  Each entry: `{name, fd_div, understat_slug, seasons, in_progress,
  expected_matches, pandemic_window}`.
  Registered: `premier_league` (E0), `serie_a` (I1, default), `la_liga` (SP1),
  `bundesliga` (D1, 306 matches — 18 teams), `ligue_1` (F1).
- **Per-league models**: each league trains its own `pipeline.joblib`
  (team label maps, features, and class priors are league-specific).
- **Per-league data layout**:
  - raw: `data/raw/{football_data,understat}/{league}/...`
  - ML: `data/export/ml/{league}/{matches_clean.csv, team_labels.csv, models/, results/}`
  - CSV export: `data/export/{league}/{season}.csv` + `matches_full.csv`
    (all leagues, with a `league` column)
- **DB**: `teams` and `matches` gain a `league` column;
  `UNIQUE(league, name)` / `UNIQUE(season, league, date, home_team_id, away_team_id)`.
  Legacy single-league DBs auto-migrate on next `get_conn()`
  (`upsert._migrate` backfills `league='serie_a'`).
- **Per-league season derivation**: the 2019-20 pandemic window is a
  registry field (`pandemic_window`); `data.derive_season(date, league)`
  applies it only for leagues that had it (Serie A, La Liga, Ligue 1).
- **League-scoped features**: form/matchday features group by
  `(league, season, team)` when a `league` column is present
  (`has_league` guards in `src/preprocess/clean.py`), so single-league
  frames keep working unchanged.

## Path helpers (config.py)

- `league_dir(source, league)` → `data/raw/{source}/{league}`
- `ml_dir(league)` → `data/export/ml/{league}`
- `ml_results_dir(league)`, `ml_models_dir(league)`
- `max_matchdays(league)` → teams − 1, derived from `expected_matches`
  (380 → 38, 306 → 34)

## League-aware signatures

| Module | Function |
|---|---|
| `src/fetch/football_data_csv.py` | `fetch_season(season, league)`, `fetch_all(league, force)` |
| `src/fetch/understat.py` | `fetch_season(season, league)`, `fetch_all(league, force)` |
| `src/clean/merge.py` | `load_football_data(league)`, `load_understat(league)`, `merge(league)` |
| `src/db/upsert.py` | `load(league)` (auto-migrates legacy DB) |
| `src/export/csv_export.py` | `export(league=None)` (None = all leagues) |
| `src/validate/checks.py` | `run_checks(league)` |
| `src/preprocess/clean.py` | `load_matches(source, league)`, `--league` CLI |
| `src/ml/data.py` | `load_data(league)`, `load_team_labels(league)`, `derive_season(date, league)` |
| `src/ml/train.py` | `run_training(league=...)`, `--league` CLI |
| `src/ml/predict.py` | `load_pipeline(league)`, `demo_fixtures(league)`, `--league` CLI |
| `src/ml/fixtures.py` | `load_scheduled(season, league)`, `build_fixtures(season, league, ...)` |
| `src/ui/app.py` | sidebar League selector → `st.session_state["league"]` → all tabs |

## Migration (legacy flat layout → per-league)

1. `data/raw/football_data/{season}.csv` → `data/raw/football_data/serie_a/{season}.csv`
2. `data/raw/understat/{year}.json` → `data/raw/understat/serie_a/{year}.json`
3. `data/export/ml/{matches_clean.csv, team_labels.csv, models/, results/}`
   → `data/export/ml/serie_a/...`
4. DB: auto-migrated by `upsert._migrate` on next `get_conn()`
   (verified: 3880 matches, 34 teams, all `league='serie_a'`).
5. Legacy flat per-season CSVs in `data/export/` removed (regenerated under
   `data/export/serie_a/` by the league-aware `csv_export`).

## Verification (2026-10-28)

- **45/45 tests pass** (`pytest tests/ -q`).
- Serie A regression: 3879 clean rows, 34 teams, 18 features; season counts
  2019-20 = 380, 2020-21 = 380, 2026-27 = 80; time split 3039/840;
  walk-forward folds 2022-23…2025-26 (380 each).
- EPL smoke test: fetched 2024-25 (380 matches) → DB → clean CSV (380 rows,
  24 cols) end-to-end.
- `csv_export` writes `data/export/matches_full.csv` (3880 + 380 rows,
  league column) + per-league/per-season CSVs.
- `predict --league serie_a --demo` runs against the migrated
  `data/export/ml/serie_a/models/pipeline.joblib`.

## Adding a new league

1. Add an entry to `config.LEAGUES` (fd_div, understat_slug, seasons,
   in_progress, expected_matches, pandemic_window).
2. Fetch: `python -m src.fetch.football_data_csv --league KEY` and
   `python -m src.fetch.understat --league KEY` (or UI tab 1).
3. Load: `python -m src.db.upsert --league KEY`.
4. Clean: `python -m src.preprocess.clean --league KEY`.
5. Train: `python -m src.ml.train --league KEY` (or UI tab 2).
6. Predict: `python -m src.ml.predict --league KEY --demo` (or UI tab 3).

No code changes required.
