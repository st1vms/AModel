# Agent Instruction Prompt: Model Accuracy Enhancement & Advanced Feature Engineering

You are an expert AI software engineer and machine learning specialist collaborating on **AModel**, a multi-league soccer match-data and prediction pipeline. 

Your objective is to guide the integration of **new advanced features** to push the predictive ceiling of our 1X2 classification and goal regression models, while strictly preserving our multi-league architecture, data hygiene, and chronological validation standards.

---

## 1. Scope of New Features to Implement

You must engineer the following features using **existing data sources** (`football-data.co.uk` and `Understat`) stored in SQLite (`data/amodel.db`) and processed in `src/preprocess/clean.py` and `src/ml/data.py`:

1. **Temporal & Fatigue Features (`rest_days`, `is_midweek`)**:
   - Calculate exact days elapsed since each team's last official match within the same season.
   - Flag whether a match occurs within $\le 4$ days of the previous fixture.
2. **Advanced xG Efficiency & Rolling Trends (`xg_diff_rolling`, `finishing_luck`)**:
   - 5-match rolling averages of $xG$ for ($xGF$) and $xG$ against ($xGA$).
   - *Finishing Luck:* Rolling difference between actual goals scored and expected goals ($Goals - xG$) over the last 5 matches to capture regression-to-the-mean effects.
3. **Strength of Schedule (SoE-weighted Form)**:
   - Weight team form points not just raw (3 for win, 1 for draw), but scaled by the opponent's current league rank or Elo/rolling strength at the time of the match.
4. **Market-Derived Implied Probabilities (De-vigghed Odds)**:
   - Extract opening/closing odds from `football-data.co.uk` (e.g., `B365H`, `B365D`, `B365A`).
   - Remove bookmaker overround (vig) to calculate clean pre-match market probabilities as features.

---

## 2. Multi-League Compatibility & Source Awareness (`config.LEAGUES`)

Before implementing features, verify availability across all registered leagues in `config.LEAGUES`:
* **Top-5 Leagues (`premier_league`, `serie_a`, `la_liga`, `bundesliga`, `ligue_1`)**: Full support for both `football-data` metrics and `Understat` xG data.
* **Serie B (`serie_b`)**: **No Understat coverage (`understat_slug = None`)**. 
  * *Critical Rule:* All xG-based rolling features or finishing luck features **must fail gracefully or fall back to shot-based proxies** (e.g., total shots on target differential) when processing Serie B to prevent `NaN` or unhandled exceptions during preprocessing and prediction.

---

## 3. Implementation Steps & Bug Prevention Rules

To integrate these features without introducing bugs or breaking existing tests, follow this strict protocol:

### Step A: Preprocessing (`src/preprocess/clean.py`)
* **Order of Operations:** Compute rolling features, rest days, and market probabilities *before* encoding team labels or dropping metadata columns.
* **Leakage Prevention:** All rolling features (form, xG rolling, rest days) must use `.shift(1).rolling(...)` restricted strictly to **past matches within the same season**. Never include data from the current match or future matches.
* **No NaN Policy:** Ensure the final `matches_clean.csv` contains 0 null values. Handle early-season matches (matchday 1-5 where rolling averages don't have 5 prior games) by imputing zeros or league-average defaults safely.

### Step B: Fixture Prediction Integration (`src/ml/fixtures.py`)
* When building pre-match fixtures for unplayed matches (`isResult is False`), ensure historical rolling features correctly append the upcoming fixture date as the latest timestamp without duplicating keys or generating object-dtype columns.
* Impute missing market odds or unreleased line-up stats cleanly if building fixtures ahead of time.

---

## 4. Feature Validation Protocol (Mandatory Before Training)

To prevent training crashes, silent metric degradation, or data leakage, execute and pass the following validation checks before running `src/ml/train.py`:

1. **The Temporal Ordering Check:**
   - Write a unit test or assert statement verifying that for any match $M$ at date $T$, the rolling features (`form`, `rest_days`, `xg_diff`) depend *only* on matches with dates $< T$.
2. **The Null & Infinity Check:**
   - Run a programmatic check over `matches_clean.csv` to ensure no `NaN`, `inf`, or `-inf` values exist in newly added feature columns across any league.
3. **The Serie B Graceful Fallback Check:**
   - Verify that when executing pipeline scripts with `--league serie_b`, missing xG sources do not raise `KeyError` or crash training/prediction scripts.
4. **Walk-Forward Validation Integrity:**
   - Confirm that adding the new features improves or maintains the walk-forward evaluation metrics (`log_loss`, `accuracy`) across the 4-fold temporal split without inflating ROC-AUC to unrealistic levels ($> 0.85$ is a red flag for data leakage).

---