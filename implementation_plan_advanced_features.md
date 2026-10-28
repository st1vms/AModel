# Implementation Plan: Advanced Feature Families (Temporal, xG, Elo, SoE, Market)

Status: **implemented & verified** (2026-10-28)
Builds on: `implementation_plan_ml_models.md` (ML package), `src/preprocess/clean.py` (feature engineering)

## TL;DR

Extend the 18-feature ML frame with **16 new features** across five families,
bringing `FEATURES` to **34**, while preserving the multi-league architecture,
data hygiene, and chronological (leakage-free) validation:

| Family | Features | Source |
|---|---|---|
| Temporal | `rest_days_home`, `rest_days_away`, `is_midweek` | match dates |
| xG | `xg_diff_rolling`, `finishing_luck_home`, `finishing_luck_away` | xG (SoT×0.33 fallback) |
| Elo | `elo_home`, `elo_away`, `elo_diff` | computed from results |
| Strength-of-estimate | `home_form_soe`, `away_form_soe`, `ppg_rolling_home`, `ppg_rolling_away` | results + Elo |
| Market | `mkt_prob_home`, `mkt_prob_draw`, `mkt_prob_away` | B365 opening odds |

All rolling features use the canonical leakage pattern
`shift(1).rolling(window, min_periods=1).mean()` grouped by
`(league, season, team)` — strictly earlier same-season matches only.
Elo is computed chronologically over the whole league history and records
**pre-match** ratings (leakage-free), persistent across seasons.

## Scope decisions (approved)

- **Serie B dropped** entirely (not registered in `config.py`).
- **Opening odds only** (B365H/D/A) — no closing odds.
- **Rolling Elo** for strength-of-estimate weighting.
- **SoT × 0.33** as xG proxy when xG is missing (pre-2015-16 seasons).

## Key data facts (verified)

- Serie A: 3,880 matches in DB → 3,879 after clean (1 null-HT row dropped),
  34 teams.
- `b365h/d/a` have exactly 3 nulls each in the clean frame; market probs fall
  back to 1/3 there.
- `rest_days` range 3.0–116.0 (min 3 = midweek, max ~115 = winter break).
- `mkt_prob_home + mkt_prob_draw + mkt_prob_away == 1.0` on every row.
- Elo range ~1333–1815; `elo_diff` range ~-400 to +402.

## Steps

### Phase 1 — Data plumbing (merge → DB → clean)

1. `src/clean/merge.py`: football-data branch emits `shots_home/away` (HS/AS),
   `shots_on_target_home/away` (HST/AST), `xg_home/away` (HxG/AxG),
   `b365h/d/a` (B365H/D/A), `referee`. Understat-only rows set the 5 new
   numeric cols to `None`.
2. `src/db/schema.sql`: add the 5 new columns to `CREATE TABLE matches`.
3. `src/db/upsert.py`: idempotent `ALTER TABLE` migration for existing DBs;
   legacy `matches_new` rebuild + main INSERT/DO UPDATE carry the new cols.
4. `src/preprocess/clean.py`: `load_matches` SELECTs the new cols; coercion
   loop handles them (guarded by `if col in df.columns`).

### Phase 2 — Feature engineering (`src/preprocess/clean.py`)

New helpers, wired into `add_features` in order:

- `_xg_eff(df)` — `xg_eff_* = xg_*.fillna(shots_on_target_* × 0.33)`.
- `_rest_features(df)` — days since last same-season match per team
  (7.0 when no history); `is_midweek = (rest ≤ 4)` for either side.
- `_xg_rolling_features(df, window=5)` — rolling xG for/against/diff and
  `finishing_luck = rolling(goals − xg_eff)` (positive = overperforming).
- `_elo_ratings(df, k=20.0)` — chronological Elo over full league history;
  records pre-match values; persistent across seasons.
- `_soe_form_features(df, elo, window=5)` — form weighted by opponent Elo
  (`opp_elo/1500`), 0–3 scale; plus plain `ppg_rolling_*`.
- `_market_features(df)` — inverse-odds de-vigged by the overround;
  `fillna(1/3)` when odds missing.

### Phase 3 — ML registration + fixtures

5. `src/ml/data.py`: `FEATURES` 18 → 34; new `XG_FEATURES` list so `no_xg`
   drops all xG-derived features together.
6. `src/ml/fixtures.py`: fixture rows carry the 5 new raw cols as NaN;
   per-team rolling quantities (form, SoE form, ppg_rolling, luck) are
   carried forward from each team's latest real match (renamed
   `_latest_*` to avoid merge-suffix collisions); `xg_diff_rolling = 0.0`
   (neutral prior); `rest_days` computed from last real-match date
   (fillna 7.0) and `is_midweek` recomputed.

### Phase 4 — Tests

7. `tests/test_preprocess.py`: `_sample()` extended with the 5 new raw cols;
   7 new tests: rest-days leakage, market de-vigging + sum-to-1 + 1/3
   fallback, Elo pre-match & persistence, xG rolling/finishing-luck
   leakage, xG→SoT fallback, SoE form & ppg_rolling leakage.
8. `tests/test_ml.py`: feature count 18 → 34.
9. Regenerate `data/export/ml/serie_a/matches_clean.csv` (tests read it).

### Phase 5 — Run & verify

10. `.venv/bin/python -m pytest tests/ -q` → **59 passed** (was 52).
11. `.venv/bin/python -m src.ml.train --league serie_a --skip-tuning --walk-forward`
12. `.venv/bin/python -m src.ml.predict --league serie_a --demo`

## Results (Serie A, 34 features, walk-forward 2022-23…2025-26)

| Model | log_loss | AUC macro | accuracy |
|---|---|---|---|
| **logistic** | **0.8495** | **0.7844** | **0.6145** |
| catboost | 0.8736 | 0.7693 | 0.6026 |
| xgboost | 1.0383 | 0.7636 | 0.6039 |
| lightgbm | 1.1505 | 0.7603 | 0.5947 |

Baseline (18 features): best was catboost 0.8605 / 0.7757 / 0.6039.
The new market + Elo features are strong **linear** signals: logistic
regression becomes the best walk-forward classifier, ahead of the baseline
best on all three metrics. Single-split model selection (log_loss on the
2023-24 slice) still picks catboost (0.8476 vs 0.8515), so the saved
pipeline keeps catboost — consistent with existing selection behavior.

Goal regression (walk-forward, catboost): home MAE 0.745 / RMSE 0.954,
away MAE 0.701 / RMSE 0.890 — in line with baseline.

## Open considerations

1. **Elo persistence across seasons** — currently persistent (no reset).
   A per-season reset would be an alternative; persistent is the standard
   for cross-season strength tracking.
2. **`ppg_rolling_*` redundancy** — near-identical to `form_*_pts`; candidate
   for removal if feature pruning is ever desired.
3. **`xg_diff_rolling = 0.0` on fixtures** — neutral prior; per-team rolling
   xG gap is not stored separately, so the fixture value can't be composed.
4. **Model selection** — single-split log_loss picks catboost; walk-forward
   favors logistic. If walk-forward becomes the selection criterion, the
   saved pipeline would switch to logistic.
