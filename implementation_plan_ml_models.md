# Implementation Plan: Serie A 1X2 & Goal Prediction Models

Status: **approved for implementation** (2026-10-27)
Builds on: `implementation_plan_data_scraper.md` (data pipeline), `src/preprocess/clean.py` (ML-ready frame)

## TL;DR

New `src/ml/` package on top of the existing `data/export/ml/matches_clean.csv`
(3,879 rows, 11 seasons, 0 nulls, label-encoded teams + leakage-free form
features). Compare baselines vs XGBoost/LightGBM/CatBoost for:

- **1X2 classification** (Home / Draw / Away)
- **Goal regression** (home & away goals; GBM primary + Poisson baseline)

Evaluated with a single time-based split (train ≤ 2023-24 → validate
2024-25…2026-27) plus an expanding-window walk-forward (2022-23 → 2025-26).
Deliverables: metrics (log loss, ROC-AUC, accuracy, MAE, RMSE),
confusion-matrix + feature-importance plots, and a saved joblib inference
pipeline.

## Key data facts (verified)

- `matches_clean.csv` columns: `date`, `home_team`/`away_team` (int codes,
  34 teams, decoded via `team_labels.csv`), `home_goals`, `away_goals`,
  `ht_*_goals`, `shots_*`, `xg_*` (Understat, match-level), `month`,
  `day_of_week`, `matchday`, `shots_diff`, `xg_diff`,
  `home/away_form_pts/gf/ga`, `form_pts_diff`.
- **No `season` column** in the CSV (dropped as metadata) — derive from
  `date`.
- **Season derivation quirk**: 2019-20 pandemic matches played Jun–Aug 2020
  (file runs to 2020-08-02) must map to 2019-20, not 2020-21 (2020-21 started
  Sep 2020). Rule: month ≥ 7 → season starting that year, EXCEPT dates in
  [2020-06-01, 2020-08-31] → 2019-20.
- 2026-27 is in progress (80 matches, through 2026-10-25).
- venv (`.venv/`, Python 3.12) has NO ML libraries yet: needs scikit-learn,
  xgboost, lightgbm, catboost, matplotlib, seaborn, scipy, joblib.
- Conventions: run via `.venv/bin/python`, temp files in `.tmp/`, no git,
  update `MEMORY.md`, tests in `tests/`.

## Steps

### Phase 0 — Dependencies (blocks everything)

1. Add to `requirements.txt`: `scikit-learn>=1.4`, `xgboost>=2.0`,
   `lightgbm>=4.0`, `catboost>=1.2`, `matplotlib>=3.8`, `seaborn>=0.13`,
   `scipy>=1.11`, `joblib>=1.3`.
2. Install into venv: `.venv/bin/python -m pip install -r requirements.txt`
   (CatBoost wheel is large; if install fails, note it and make CatBoost
   import-guarded so the rest still runs).

### Phase 1 — Data & splits (`src/ml/data.py`, `src/ml/splits.py`)

3. `src/ml/data.py`:
   - `load_data()` → read `matches_clean.csv` + `team_labels.csv` (decode
     codes to names for reporting only; keep codes as features).
   - `derive_season(date)` with the 2020 pandemic special case; add `season`
     column.
   - Targets: `outcome` ∈ {0=H, 1=D, 2=A} from goals; `home_goals` /
     `away_goals` as regression targets.
   - `FEATURES` list (18): `home_team`, `away_team`, `month`, `day_of_week`,
     `matchday`, `shots_home`, `shots_away`, `xg_home`, `xg_away`,
     `shots_diff`, `xg_diff`, `home_form_pts`, `home_form_gf`,
     `home_form_ga`, `away_form_pts`, `away_form_gf`, `away_form_ga`,
     `form_pts_diff`.
   - `--include-ht` flag adds `ht_home_goals`, `ht_away_goals` (halftime
     variant); `--no-xg` drops the two xG columns (strict pre-match variant).
     Default: xG in, HT out.
4. `src/ml/splits.py`:
   - `time_split(df, cutoff="2023-24")` → train (season ≤ cutoff) / validate
     (2024-25, 2025-26, 2026-27).
   - `walk_forward_splits(df, start="2022-23", end="2025-26")` → expanding
     window: 4 folds, fold k trains on all seasons before validation season k
     (2022-23, 2023-24, 2024-25, 2025-26). 2026-27 excluded from walk-forward
     (partial season).
   - Sort by `date` everywhere; assert no validation date ≤ max train date
     (leakage guard).

### Phase 2 — Models (`src/ml/models.py`)

5. Classification (1X2), per framework:
   - **Baselines**: (a) historical base-rate predictor (train-set class
     frequencies); (b) `LogisticRegression` with `StandardScaler` in a
     `Pipeline` (scaling for non-tree models).
   - **XGBoost** `XGBClassifier`, **LightGBM** `LGBMClassifier`, **CatBoost**
     `CatBoostClassifier` (with `cat_features=[home_team, away_team]` for
     CatBoost; numeric for the others).
   - **Imbalance**: inverse-frequency class weights → `sample_weight`
     (XGB/LGBM) and `class_weights` (CatBoost).
   - **Calibration**: fit `CalibratedClassifierCV` (isotonic) on a
     calibration slice = last train season (2023-24) held out of the main
     train fit; report calibrated vs raw log loss.
6. Goal regression (home & away, 2 targets each):
   - **Primary**: XGBoost/LightGBM/CatBoost regression.
   - **Baseline**: Poisson regression
     (`sklearn.linear_model.PoissonRegressor` on scaled features) + naive
     mean baseline.
   - Clip GBM predictions at 0 for the non-negative guarantee.
7. **Tuning (light)**: small grid on train only — `n_estimators` {500, 1000},
   `learning_rate` {0.03, 0.05}, `max_depth` {4, 6}, `subsample` {0.8, 1.0};
   early stopping on a 10% time-based holdout of the train tail (2023-24).
   Tune once per model type; reuse params across walk-forward folds.

### Phase 3 — Evaluation & plots (`src/ml/evaluate.py`, `src/ml/plots.py`)

8. `evaluate.py`:
   - Classification: log loss, ROC-AUC (macro OvR + per-class), accuracy,
     `classification_report`, confusion matrix.
   - Regression: MAE + RMSE per target (home/away).
   - Aggregate walk-forward folds (mean ± std) and the single-split
     validation (with and without 2026-27).
   - Print a comparison table (all models × metrics) to stdout and save as
     CSV to `data/export/ml/results/`.
9. `plots.py` (matplotlib/seaborn, saved to `data/export/ml/results/`):
   - Confusion-matrix heatmap (best model, validation set).
   - Feature-importance plot (best model, top 15, horizontal bars).
   - Bonus: calibration curve (reliability diagram) for the best classifier.

### Phase 4 — Orchestration & inference (`src/ml/train.py`, `src/ml/predict.py`)

10. `src/ml/train.py` (CLI, `python -m src.ml.train`):
    - Flags: `--include-ht`, `--no-xg`, `--walk-forward`, `--skip-tuning`,
      `--out-dir`.
    - Runs: load → split → train+evaluate all classifiers & regressors
      (single split + walk-forward) → save results CSV + plots → save best
      models via joblib to `data/export/ml/models/` (best classifier, best
      home/away goal regressors, plus feature list, scaler, class weights,
      team labels).
11. `src/ml/predict.py` (CLI, `python -m src.ml.predict --matches PATH`):
    - Load saved artifacts; accept a CSV of future fixtures (same schema as
      `matches_clean.csv` minus targets, or raw names decoded via
      `team_labels.csv`); output predicted 1X2 probabilities, predicted
      outcome, and expected home/away goals.
    - Demo: predict the next unplayed 2026-27 fixtures from the current data.

### Phase 5 — Tests & docs

12. `tests/test_ml.py`:
    - `derive_season` unit tests incl. the 2020-07/08 → 2019-20 case and a
      normal 2024-08 → 2024-25 case.
    - Split tests: row counts, no date leakage, walk-forward fold ordering.
    - Metrics sanity: perfect predictions → log loss ≈ 0, AUC = 1, MAE = 0.
    - Smoke test: train tiny XGBoost (n_estimators=20) on a 2-season slice,
      assert shapes of predictions.
13. Update `MEMORY.md` (new `src/ml/` section, how-to-run, results summary)
    and `requirements.txt` (done in step 1).

## Parallelism

- Phase 0 blocks all. Phase 1 blocks 2–4.
- Within Phase 2, classifier and regressor work is independent (parallel).
- Steps 8–9 depend on 5–7. Step 10 depends on 5–9. Step 11 depends on 10.
- Step 12 can start after 3–4 (data/splits tests) and finish after 10.

## Critical files

- `data/export/ml/matches_clean.csv`, `data/export/ml/team_labels.csv` —
  inputs (read-only).
- `src/preprocess/clean.py` — reference for conventions (`label_codes`,
  leakage-free form via `shift(1).rolling(5)`); do NOT modify.
- `config.py` — add `ML_DIR = EXPORT_DIR / "ml"`, `ML_RESULTS_DIR`,
  `ML_MODELS_DIR` (single source of truth for paths).
- New: `src/ml/{__init__,data,splits,models,evaluate,plots,train,predict}.py`,
  `tests/test_ml.py`.
- Modify: `requirements.txt`, `config.py`, `MEMORY.md`.

## Verification

1. `.venv/bin/python -m pytest tests/ -q` → all existing 22 + new ML tests
   pass.
2. `.venv/bin/python -m src.ml.train` → prints comparison table; confirm:
   best GBM log loss < base-rate baseline; AUC > 0.65 (OvR macro); goal MAE
   < 1.1 (typical Serie A ~1.0–1.2); no NaNs; 2026-27 included in validation
   metrics.
3. Check artifacts exist: `data/export/ml/results/*.csv`, `*.png`
   (confusion matrix, feature importance, calibration),
   `data/export/ml/models/*.joblib`.
4. `.venv/bin/python -m src.ml.predict --matches <demo fixtures>` → sane
   probabilities (sum ≈ 1) and non-negative goals.
5. Manual: open the confusion-matrix and feature-importance PNGs;
   sanity-check that `xg_diff`/`form_pts_diff`/team codes rank among top
   importances.

## Decisions

- **Validation**: single split train ≤ 2023-24 → validate 2024-25 + 2025-26
  + 2026-27 (metrics reported with and without 2026-27). Walk-forward
  expanding window over 2022-23 → 2025-26 (2026-27 excluded, partial).
- **Goal models**: hybrid — GBM regression (XGB/LGBM/CatBoost) primary +
  Poisson regression baseline; GBM outputs clipped at 0.
- **Tuning**: light grid on key params, fit on train only, early stopping on
  the 2023-24 tail.
- **HT goals excluded by default** (known only at halftime → leakage for
  pre-match prediction); `--include-ht` for a halftime variant. **xG
  included by default** (prompt explicitly cites `xg_diff`); `--no-xg` for
  the strict pre-match variant. Note: Understat xG is match-level
  (shot-based), so the strict pre-match variant is the `--no-xg` one.
- **Team encoding**: keep label codes as features (trees handle natively);
  CatBoost gets them as `cat_features`. No one-hot needed.
- **Calibration**: isotonic on the 2023-24 slice; report calibrated vs raw
  log loss.
- **Scope excluded**: player-level features (Phase 5 FBref), deep learning,
  exact-score (scoreline) prediction beyond expected goals, live/in-play
  updates, hyperparameter tuning beyond the light grid.
