# AModel — Project Memory

Persistent notes for the multi-league soccer match-data + ML pipeline
(2016-17 → 2026-27). Keep this file updated as the project evolves (see AGENTS.md).

## What this is

Python pipeline that extracts soccer match data for **any registered league**
from two hybrid sources, normalizes it, loads it into SQLite
(`data/amodel.db`), and trains/evaluates 1X2 + goal models per league.
Plans: `implementation_plan_data_scraper.md`, `implementation_plan_ml_models.md`,
`implementation_plan_multi_league.md`.

- **Primary source**: football-data.co.uk CSVs — scores, HT, shots, referee.
- **Secondary source**: Understat — xG (and cross-checks).
- Seasons: 2016-17 … 2026-27 (11). `2026-27` is in progress (partial, 80 matches as of 2026-10-27).

## Multi-league architecture (2026-10)

- **`config.LEAGUES`** is the single source of truth: registry key →
  `{name, fd_div, understat_slug, seasons, in_progress, expected_matches, pandemic_window}`.
  Registered: `premier_league` (E0), `serie_a` (I1, default), `la_liga` (SP1),
  `bundesliga` (D1, 306 matches/season — 18 teams), `ligue_1` (F1).
  `DEFAULT_LEAGUE = "serie_a"`.
- **Per-league data layout** (legacy flat files migrated 2026-10):
  - raw: `data/raw/{football_data,understat}/{league}/...`
  - ML: `data/export/ml/{league}/{matches_clean.csv, team_labels.csv, models/, results/}`
  - CSV export: `data/export/{league}/{season}.csv` + `matches_full.csv` (all leagues, league column)
- **Per-league models**: each league trains its own `pipeline.joblib` under
  `data/export/ml/{league}/models/`. The UI sidebar has a **League selector**
  (first control); every tab threads `opts["league"]` through.
- **DB**: `teams` and `matches` carry a `league` column;
  `UNIQUE(league, name)` / `UNIQUE(season, league, date, home_team_id, away_team_id)`.
  Legacy single-league DBs auto-migrate on next `get_conn()`
  (`upsert._migrate` backfills `league='serie_a'`).
- **Path helpers** (use these, not the legacy constants):
  `config.league_dir(source, league)`, `config.ml_dir(league)`,
  `config.ml_results_dir(league)`, `config.ml_models_dir(league)`,
  `config.max_matchdays(league)` (380→38, 306→34).
- **League-aware signatures**: `merge(league)`, `upsert.load(league)`,
  `fetch_all(league, force)`, `clean.load_matches(source, league)`,
  `data.load_data(league)`, `data.derive_season(date, league)` (per-league
  pandemic window), `train.run_training(league=...)`,
  `predict.load_pipeline(league)`, `fixtures.build_fixtures(season, league)`,
  `csv_export.export(league=None)`.
- **Form/matchday features** are computed per `(league, season, team)` when a
  `league` column is present (guarded by `has_league` in `clean.py`).
- **Adding a league**: add an entry to `config.LEAGUES`, then fetch + upsert +
  clean + train for it. No code changes needed.

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
config.py                 # LEAGUES registry, paths, URLs, fetch settings (single source of truth)
src/fetch/http.py         # get_text() with retry/backoff, polite_sleep()
src/fetch/football_data_csv.py  # fetch_season/fetch_all(league) -> data/raw/football_data/{league}/{season}.csv
src/fetch/understat.py    # fetch(league) -> data/raw/understat/{league}/{year}.json
src/clean/normalize.py    # canonical_team(), parse_date(), season_from_date(), to_int/float_or_none
src/clean/merge.py        # load_football_data(league), load_understat(league), merge(league)
src/db/schema.sql         # DDL: teams(league), matches(league), seasons(league), players, player_match_stats
src/db/upsert.py          # get_conn() (auto-migrates legacy DB), load(league) -> idempotent upsert
src/export/csv_export.py  # export(league=None) -> data/export/matches_full.csv + data/export/{league}/...
src/validate/checks.py    # run_checks(league) -> .tmp/validation_report.txt, exit 1 on hard errors
src/preprocess/clean.py   # ML preprocessing: clean/encode/feature-engineer -> data/export/ml/{league}/matches_clean.csv
src/ml/data.py            # load_data(league), derive_season(date, league), targets/features
src/ml/splits.py          # time_split (cutoff 2023-24) + walk_forward_splits (2022-23..2025-26)
src/ml/models.py          # baselines + XGB/LGBM/CatBoost classifiers & goal regressors
src/ml/evaluate.py        # log loss, ROC-AUC, accuracy, MAE/RMSE, aggregation
src/ml/plots.py           # confusion matrix, feature importance, calibration curve
src/ml/train.py           # CLI: tune -> train -> evaluate -> save models/plots (per league)
                          #   + run_training(league=...) (structured dict for the UI)
src/ml/predict.py         # CLI: inference with saved pipeline.joblib (per league)
src/ml/backtest.py        # CLI: walk-forward CLV/ROI backtest vs closing odds (per league)
src/ml/fixtures.py        # build_fixtures(season, league): scheduled (unplayed) matches ->
                          #   prediction-ready frame (auto from raw Understat JSON)
src/ui/app.py             # Streamlit UI: League selector + update data / train / predict (3 tabs)
tests/test_pipeline.py    # 10 pytest tests
tests/test_preprocess.py  # 12 pytest tests
tests/test_ml.py          # ML tests (season derivation, splits, metrics, smoke)
tests/test_ui.py          # UI tests (fixtures, run_training, app import)
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

All CLIs accept `--league KEY` (default: `serie_a`).

```
.venv/bin/python -m src.fetch.football_data_csv --league KEY [--force]   # fetch CSVs
.venv/bin/python -m src.fetch.understat --league KEY                     # fetch xG JSON
.venv/bin/python -m src.clean.merge --league KEY                         # merge (prints per-season counts)
.venv/bin/python -m src.db.upsert --league KEY                           # load into SQLite
.venv/bin/python -m src.export.csv_export [--league KEY]                 # CSV exports
.venv/bin/python -m src.validate.checks --league KEY                     # validation report
.venv/bin/python -m src.preprocess.clean --league KEY [--encoding label|onehot|target]
        [--keep-flagged] [--keep-metadata] [--form-window 5] [--out PATH]
.venv/bin/python -m src.ml.train --league KEY [--include-ht] [--no-xg] [--walk-forward]
        [--skip-tuning] [--out-dir PATH]
.venv/bin/python -m src.ml.predict --league KEY --demo | --matches fixtures.csv
.venv/bin/python -m src.ml.backtest --league KEY [--start 2017-18] [--end 2026-26]
        [--vig 0.03] [--kelly-fraction 0.25]                # CLV/ROI backtest
.venv/bin/python -m streamlit run src/ui/app.py             # Streamlit UI (League selector in sidebar)
.venv/bin/python -m pytest tests/ -q                        # tests
```

## Preprocessing (ML-ready frame)

`src/preprocess/clean.py` turns the merged frame (DB or export CSV) into a
modeling-ready CSV at `data/export/ml/{league}/matches_clean.csv`.

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

## ML models (src/ml)

Plan: `implementation_plan_ml_models.md`. Trains 1X2 outcome classifiers and
goal regressors on `matches_clean.csv` (3,879 rows, 34 teams, 11 seasons).

- **Targets**: `outcome` (0=H, 1=D, 2=A; draws ~25% = imbalanced class),
  `home_goals`/`away_goals` regression.
- **Features**: 18 (team codes, month, day_of_week, matchday, shots, xG,
  diffs, 5-match form pts/gf/ga). `--include-ht` adds HT goals (20 features),
  `--no-xg` drops xG (14 features).
- **Validation**: `time_split` cutoff 2023-24 (train 3,039 / valid 840,
  includes 2026-27) + `walk_forward_splits` (4 folds, 2022-23..2025-26,
  expanding window). `_check_leakage` guards date ordering.
- **Models**: base_rate, logistic, xgboost, lightgbm, catboost (classification);
  mean, poisson, xgboost, lightgbm, catboost (regression). Class imbalance via
  inverse class weights (sample_weight / class_weights).
- **Tuning**: light 16-combo grid (n_estimators, learning_rate, max_depth,
  subsample) on the last train season (2023-24) held out from the fit;
  `--skip-tuning` uses defaults.
- **Calibration**: isotonic on the 2023-24 tuning slice, applied to the best
  GBM before final evaluation.
- **Outputs** (per league): `data/export/ml/{league}/results/*.csv`
  (single_split, walk_forward, tuning), `confusion_matrix.png`,
  `feature_importance.png`, `calibration_curve.png`,
  `data/export/ml/{league}/models/pipeline.joblib` (carries a `league` key).
- **Season derivation** (`data.derive_season`): month >= 7 -> season starting
  that year; month <= 6 -> season ending that year; pandemic window
  (2020-06-01..2020-08-31) -> 2019-20.

### Results (2026-10-27, full run ~32s)

Walk-forward aggregate (mean over 4 folds, 2022-23..2025-26):

| model | log_loss | AUC (macro) | accuracy |
|---|---|---|---|
| base_rate | 1.0875 | 0.5000 | 0.4072 |
| logistic | 0.8613 | 0.7783 | 0.6086 |
| xgboost | 0.8627 | 0.7695 | 0.5993 |
| lightgbm | 0.8627 | 0.7677 | 0.6007 |
| catboost | **0.8605** | 0.7757 | 0.6039 |

Single split (train ≤2023-24, test 2024-25..2026-27, 840 rows): catboost
log_loss 0.8675 / AUC 0.7714 / acc 0.5976. All GBMs beat base rate by a wide
margin; logistic is competitive (small-data regime).

Goal regression (walk-forward MAE): best ~0.70 (away, lightgbm) / ~0.76
(home, xgboost/catboost) vs mean baseline ~0.99-1.00. RMSE ~1.1-1.2.

- **Caveat**: isotonic calibration on the 380-row 2023-24 slice *hurt* log loss
  (catboost+isotonic 1.1122 vs 0.8675 uncalibrated) — overfit on the small
  slice. Pipeline saves the uncalibrated best GBM; calibration is reported
  separately in the results CSVs.
- **Framework quirks** (XGBoost 3.4 / LightGBM 4.7 / CatBoost 1.2 / sklearn
  1.9): XGB `early_stopping_rounds` in ctor (set `None` before plain fits);
  LGBM `eval_X`/`eval_y` (not `eval_set`), `best_iteration_` is a 1-based tree
  count (no +1); CatBoost `eval_metric="MultiClass"` (Logloss is binary-only),
  `iterations`/`depth` param names (mapped in `_catboost_params`);
  sklearn `CalibratedClassifierCV` needs `FrozenEstimator` (cv="prefit" gone);
  `roc_auc_score` per-class = binary call, no `average`; `calibration_curve`
  is binary-only (binarize y_true).

## Backtest / CLV (2026-10-29)

`src/ml/backtest.py` answers: *does the model have real betting edge?* Walk-forward
(one fold per validation season, expanding window, 2017-18..2026-26), each fold
retrained with the saved pipeline's classifier + tuned params (no re-tuning).

- **Closing odds plumbed through the pipeline** (2026-10-29): `b365ch/b365cd/b365ca`
  added to `merge.py` record dicts, `schema.sql`, `upsert.py` (incl. `_migrate`),
  `clean.py` SELECT + coercion. **Deliberately NOT in `FEATURES`** — closing odds
  are the benchmark, not a feature (would leak). Clean CSVs are now 48 columns.
  Availability: B365C* present **2019-20 onward, zero nulls, all 5 leagues**
  (12,852 matches total); 2016-18..2018-19 lack them → `has_closing=False`,
  counted for accuracy but excluded from CLV/ROI.
- **Protocol**: model pick = argmax of model probs (ties → home). CLV per bet =
  `p_pick / p_closing_devigged - 1` (de-vig = `inv/sum(inv)`, NaN → 1/3, same as
  `_market_features`; `devig_probs` uses `sum(skipna=False)` so one missing odds
  NaNs the whole row). ROI: flat 1-unit + fractional Kelly (compounding bankroll,
  default ¼ Kelly), gross (raw closing odds already carry the book margin) and
  net of an extra 3% flat commission. Baseline = market favourite (higher
  de-vigged closing prob, ties → home; opening-odds fallback pre-2019-20).
  Confidence bins on `p_fav = max(p_home, p_away)`: `<0.50`, `0.50-0.70`, `>=0.70`.
- **Artifacts**: `data/export/ml/{league}/results/clv_backtest.csv` (per-match) +
  `clv_backtest_summary.csv` (scope ∈ {model, baseline} × segment ∈
  {all, has_closing, bin_<0.50, bin_0.50-0.70, bin_>=0.70, odds_1.00-1.50, odds_1.50-1.90, odds_1.90-2.50, odds_>2.50, odds_>1.90}) +
  `clv_odds_summary.csv` (bracket breakdown) + `clv_odds_summary.md` (Reddit/README markdown summary table).
- **Odds stratification (2026-10-31)**:
  - Added decimal odds binning logic (`assign_odds_bracket`, `MUTUAL_ODDS_BINS`, `ODDS_BINS`: `1.00-1.50`, `1.50-1.90`, `1.90-2.50`, `>2.50`, `>1.90`).
  - Added `odds_bracket` and `odds_bin` columns to `build_betting_frame`.
  - Added `build_odds_summary` and `format_odds_summary_markdown` in `src/ml/backtest.py` with export helper `export_odds_summary_markdown` in `src/export/backtest_report.py`.
  - Headline Serie A odds results (2,740 matches with closing odds):
    - `1.00 - 1.50`: 507 bets, Acc 78.9% (Base 78.9%), Model ROI gross +4.5% / net +1.4%, Mean CLV +14.4%.
    - `1.50 - 1.90`: 617 bets, Acc 69.0% (Base 69.0%), Model ROI gross +14.7% / net +11.2%, Mean CLV +29.7%.
    - `1.90 - 2.50`: 544 bets, Acc 61.2% (Base 61.2%), Model ROI gross +31.4% / net +27.4%, Mean CLV +53.2%.
    - `> 2.50`: 1,072 bets, Acc 42.4% (Base 32.2%), Model ROI gross +48.9% / net +44.4% vs Baseline -31.0%, Mean CLV +112.3%.
    - **`> 1.90 (High-Value)`**: 1,616 bets, Acc 48.7% (Base 42.0%), Model ROI gross +43.0% / net +38.7% vs Baseline -10.0%, Mean CLV +92.4%.
- **Headline results (2026-10-29, has_closing segment, flat ROI gross / net)**:

| league | model acc | model ROI g/n | baseline ROI g/n | mean CLV | ≥0.70 bin: acc / ROI g |
|---|---|---|---|---|---|
| premier_league | 0.610 | +0.417 / +0.375 | −0.037 / −0.066 | +0.573 | 0.853 / +0.502 |
| serie_a | 0.589 | +0.295 / +0.256 | −0.018 / −0.047 | +0.639 | 0.759 / +0.389 |
| la_liga | 0.595 | +0.398 / +0.356 | −0.009 / −0.039 | +0.585 | 0.825 / +0.513 |
| bundesliga | 0.619 | +0.487 / +0.442 | −0.056 / −0.084 | +0.654 | 0.834 / +0.557 |
| ligue_1 | 0.607 | +0.453 / +0.409 | −0.039 / −0.068 | +0.616 | 0.818 / +0.584 |

  Model log-loss beats the de-vigged closing line's own log-loss in every league
  (e.g. PL 0.843 vs 0.966). **The model beats the market favourite in every
  league and every confidence bin**; the ≥0.70 bin is the strongest (acc 0.76-0.85,
  ROI +0.39..+0.58). Kelly ROI is astronomically large (10¹⁵+) — mathematically
  correct for a real edge compounded over ~2,700 bets, but treat as "huge", not
  literal. Edge drivers: the model bets draws ~20% of the time (win rate 36% vs
  ~29% market-implied) and backs away sides more often than the market does.
  Caveat: walk-forward folds are in-sample-ish for the *features* (form/xG are
  pre-match, but the 2019-20+ closing line is the true benchmark — CLV is the
  honest number).

## UI (src/ui — Streamlit)

Streamlit app (only new dep: `streamlit>=1.30`) to update data, train, and
predict without the CLI. Run: `.venv/bin/python -m streamlit run src/ui/app.py`.

- **Sidebar**: **League selector first** (from `config.LEAGUE_ORDER`, stored in
  `st.session_state["league"]`), then season (from the league's `seasons`),
  matchdays (1–`config.max_matchdays(league)`), include_ht / no_xg /
  walk_forward / skip_tuning. Page title shows the selected league name.
- **Tab 1 — Update data**: `football_data_csv.fetch_all(league, force)` →
  `understat.fetch_all(league, force)` → `upsert.load(league)` → rebuild
  `data/export/ml/{league}/matches_clean.csv` + `team_labels.csv`
  (mirrors `preprocess.clean.main`). Shows DB/clean-CSV metrics (cached via
  `st.cache_data`, cleared after update).
- **Tab 2 — Train**: calls `train.run_training(league=..., ...)` (new function
  extracted from the old `main()`; same flow, returns a dict with result
  frames, `best_name`, `saved` paths, `elapsed`, etc. and takes a `log`
  callable). Renders result tables + the 3 PNGs; falls back to last saved
  results CSVs.
- **Tab 3 — Predict**: `fixtures.build_fixtures(season, league, n_matchdays,
  labels=pipeline["team_labels"])` → `predict.prepare_fixtures` →
  `predict.predict`. Decodes codes→names for display, offers CSV download
  (`predictions_{league}_{season}.csv`).

### src/ml/fixtures.py — auto fixtures
- `load_scheduled(season, league)`: reads
  `data/raw/understat/{league}/{start_year}.json`,
  keeps `isResult is False` rows (the merged DB drops them), filters by
  `derive_season(date, league) == season`, returns `date, home_team, away_team`
  (canonical names), sorted by date.
- `build_fixtures(season, n_matchdays=None, labels=None)`:
  - shots/xG filled with **league averages** from played matches (best prior).
  - form features via `add_features` on a combined historical+fixtures frame —
    leakage-free because fixture dates are the latest in the season.
  - team encoding: with `labels` (the saved pipeline's `team_labels`,
    code→name) uses that map so codes match the trained models exactly
    (unknown team → `ValueError`); without it, deterministic `label_codes`.
  - returns `date, ht_*_goals (NaN)` + all `FEATURES`; team cols are int codes.

### team_labels.csv column-order bug (fixed 2026-10-27)
`label_map` is **team→code**, but the CSV writer wrote it with
`columns=["code","team"]` — swapping the columns, so `load_team_labels()`
returned `{name: code}` instead of the documented `{code: name}`. The saved
pipeline therefore carried a name→code map. Fixed in **both** writers
(`src/preprocess/clean.py` and `src/ui/app.py._rebuild_clean_csv`) to emit
`(code, team)` rows; regenerated `team_labels.csv` and retrained so
`pipeline.joblib` now carries `{code: name}`.

### Bundesliga canonical-map bug (fixed 2026-10-27)
`CANONICAL_BY_LEAGUE["bundesliga"]` was **empty**, so football-data short
names (`Hertha`, `Ein Frankfurt`, `M'gladbach`, `Dortmund`, …) never aligned
with Understat full names (`Hertha Berlin`, `Eintracht Frankfurt`,
`Borussia M.Gladbach`, `Borussia Dortmund`, …) on the merge key
`(season_file, home, away)`. Result: 234 matches/season became
`understat_only` duplicates (DB had 540 rows for a 306-match season, 48 team
rows instead of 31); the clean step dropped those duplicates (null HT),
leaving only the football-data rows — **all without xG** (2,476/3,122 rows
NaN). GBMs tolerate NaN natively, so tuning passed, but
`LogisticRegression` raised `Input X contains NaN` during training.
Fix: full 31-team Bundesliga map in `normalize.py` (canonical names = the
Understat-style names already in the DB, so the upsert reuses team rows);
cleared + re-upserted Bundesliga (3,123 matches, 31 teams, 0 xG nulls,
0 understat_only) and rebuilt `matches_clean.csv`. Regression test:
`tests/test_pipeline.py::test_canonical_team_bundesliga_aliases`.
**Lesson**: when adding a league, populate its canonical map *before*
fetching — an empty map silently produces `understat_only` duplicates and
xG-less rows. Same bug fixed 2026-10-27 for the other three leagues
(`premier_league` 6 name pairs, `la_liga` 13, `ligue_1` 4) with full
canonical maps + regression tests (`test_canonical_team_{premier_league,la_liga,ligue_1}_aliases`)
and a clear/re-upsert/rebuild: PL 3,880 matches / 35 teams, La Liga 3,899 / 32,
Ligue 1 3,550 / 33 — all 0 xG nulls, 0 `understat_only` (Ligue 1 keeps 1
legit `understat_only`: 2026-08-23 PSG-Rennes, sources disagree on home/away;
plus 1 football-data row with genuinely missing HT, Bastia-Lyon 2017-04-16).
All 5 leagues now have full canonical maps.

### load_pipeline positional-arg bug (fixed 2026-10-28)
The migration changed the signature to `load_pipeline(league=..., path=None)`
(league first), but `app.py`'s `load_pipeline_cached` still called it with a
bare positional path — so the path string bound to `league`, `path` stayed
`None`, and `ml_models_dir(<abs path>)` collapsed via pathlib's absolute-join
rule (`Path("/a") / "/abs"` → `/abs`), producing a doubled path and
`NotADirectoryError`. Fixed: `app.py` now calls `load_pipeline(path=path)`
(keyword), and `load_pipeline` raises a clear `ValueError` when `path is None`
and `league` is not in `config.LEAGUES`. Regression test:
`tests/test_ml.py::test_load_pipeline_keyword_path`.
Note: the on-disk `pipeline.joblib` predates the migration and has **no
`league` key** — nothing in `app.py`/`predict.py` reads it, so it's compatible.

### Predict-tab NaN crash for HT pipelines (fixed 2026-10-28)
PL/LL/L1 pipelines were saved with `include_ht=True` (verification runs used
`--include-ht`), but pre-match fixtures have NaN HT goals by design →
`ValueError: Input X contains NaN` in `LogisticRegression.predict_proba`
(GBMs tolerate NaN, so only the logistic pipelines crashed). Fix:
`src/ml/predict.py::predict` imputes `ht_*` columns with 0.0 ("no goals
scored yet" — the modal HT state) before scoring; UI warning updated to
explain the imputation. Regression test:
`tests/test_ml.py::test_predict_imputes_nan_ht_features`. Note: retraining
without **Include HT goals** still gives the cleanest pre-match predictions.

### build_fixtures object-dtype form columns (fixed 2026-10-28)
Ligue 1's 2026-08-23 PSG-Rennes match exists in the DB twice (football-data
row + reversed `understat_only` row), so PSG/Rennes each appeared twice on
the same date in `build_fixtures`' historical frame → duplicate
`(season, date, team)` keys → `_form_features`' `home.loc` returned a Series
→ object-dtype form columns → LightGBM `ValueError: pandas dtypes must be
int, float or bool`. Fix: `build_fixtures` drops null-HT rows from
`historical` (matching the clean pipeline, which is what the models were
trained on). Regression test:
`tests/test_ml.py::test_build_fixtures_form_columns_numeric`.

## Current status (2026-10-29)

- **Multi-league migration done**: legacy flat Serie A files moved to
  `data/raw/{source}/serie_a/` and `data/export/ml/serie_a/`; DB auto-migrated
  (all rows `league='serie_a'`). 5 leagues registered in `config.LEAGUES`.
- **Serie A**: 3880 matches in DB (380 × 10 completed + 80 in-progress 2026-27),
  34 teams, 11 seasons.
- **All 5 leagues fully fetched** (11 seasons each): PL 3,880 matches / 35
  teams, La Liga 3,899 / 32, Ligue 1 3,550 / 33, Bundesliga 3,123 / 31 —
  all with 0 xG nulls after the canonical-map fixes (2026-10-27).
- **Validation: 0 hard errors**, 4 legitimate warnings:
  - 2019-20: 114 matches with month outside Aug–May (pandemic — season ran to Aug 2020).
  - 2022-23: 10 matches in June (season ended 2023-06-04).
  - 2023-24: 1 match in June (ended 2024-06-02).
  - 2026-27: 52 cross-source xG mismatches > 0.15 (football-data vs Understat xG models differ).
- **89/89 tests pass** (14 pipeline + 19 preprocessing + 14 ML + 16 UI + 8 fetch + 18 backtest).
- **Backtest/CLV suite done** (2026-10-29, updated 2026-10-31): closing odds in DB + clean CSVs,
  `src/ml/backtest.py` run for all 5 leagues, odds stratification breakdown + Markdown report generation added,
  `tests/test_backtest.py` (18 tests).
- Spot-checks verified: 2024-25 champion **Napoli** (82 pts), Bologna 5-0 Lazio 2025-03-16, Inter 4-4 Juventus 2024-10-27.
- Champions: 2016-17/17-18/18-19/19-20 Juventus, 2020-21 Inter, 2021-22 AC Milan, 2022-23 Napoli, 2023-24 Inter, 2024-25 Napoli, 2025-26 Inter.

## Known data quirks (accepted, flagged)

- **Sassuolo 0-3 Pescara (2016-08-28)**: football-data says 0-3, Understat says 2-1. Genuine source discrepancy → `score_mismatch` flag.
- **Inter 6-2 Pisa (2026-01-23)**: xG 6.24 — legitimate for a 6-goal match; xG sanity bound relaxed to [0, 8].

## Next (Phase 5)

- FBref player-level enrichment → `players` + `player_match_stats` (Playwright MCP, optional `playwright` dep).

## Done

- **Backtest / CLV & Odds Stratification** (2026-10-29, updated 2026-10-31) — closing odds (B365C*) plumbed through
  merge/schema/upsert/clean; `src/ml/backtest.py` (walk-forward CLV, flat +
  Kelly ROI, market-favourite baseline, confidence slicing, decimal odds stratification);
  Markdown reporting and README documentation added; `tests/test_backtest.py` (18 tests).
- **Multi-league support** (2026-10-28) — `config.LEAGUES` registry (5 leagues),
  per-league raw/ML/DB layout, per-league models, UI League selector,
  legacy Serie A data migrated, league-aware `csv_export`. Plan:
  `implementation_plan_multi_league.md`.
- **Streamlit UI** (`src/ui/app.py`) — update data / train / predict, 3 tabs.
  Auto-fixture builder (`src/ml/fixtures.py`) recovers scheduled matches from
  the raw Understat JSON. `train.run_training()` extracted for the UI.
  Fixed the `team_labels.csv` column-order bug (see UI section).
