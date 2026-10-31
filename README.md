# AModel

AModel is an advanced machine learning and data-driven football (soccer) prediction engine designed to forecast match outcomes, analyze team performance, and process sports data seamlessly through automated setup and execution pipelines.

---

## Table of Contents
- [Project Architecture & How It Works](#project-architecture--how-it-works)
- [Backtest Results & Predictive Edge](#backtest-results--predictive-edge)
- [Data Sources](#data-sources)
- [Supported Leagues](#supported-leagues)
- [Predictive Features](#predictive-features)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)

---

## Project Architecture & How It Works

AModel operates through a structured pipeline consisting of data ingestion, feature engineering, model training, and inference:

1. **Data Ingestion:** Automatically fetches historical and live fixtures, match statistics, odds, and team sheets from reliable sports data APIs and repositories.
2. **Feature Engineering:** Transforms raw match data into robust statistical indicators, calculating rolling averages, Elo ratings, and head-to-head metrics.
3. **Model Training:** Utilizes machine learning classifiers (such as Gradient Boosting / Random Forests) trained on historical match outcomes (Home Win, Draw, Away Win, Over/Under markets).
4. **Inference & Prediction:** Evaluates upcoming fixtures using the trained models to output probability distributions for match results.

---

## Backtest Results & Predictive Edge

To rigorously evaluate whether AModel possesses a genuine predictive edge against sports betting markets, we conduct out-of-sample walk-forward backtests across historical seasons benchmarked against real **closing market odds** (Bet365 closing prices).

Closing lines are widely acknowledged as the sharpest, most mathematically efficient odds available. Model performance is tested under flat 1-unit staking (both gross and net of a standard 3% flat commission) and benchmarked against a **market favourite baseline** (always backing the closing favourite).

### 1. Headline League Results (Walk-Forward Benchmark)

Summary across major European leagues on matches with closing odds available (2019-20 onward):

| League | Total Bets ($N$) | Model Accuracy | Baseline Accuracy | Model ROI (Gross) | Model ROI (Net 3%) | Baseline ROI | Mean CLV |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **Bundesliga** | 2,142 | 61.9% | 54.2% | **+48.7%** | **+44.2%** | -5.6% | +65.4% |
| **Ligue 1** | 2,470 | 60.7% | 53.8% | **+45.3%** | **+40.9%** | -3.9% | +61.6% |
| **Premier League** | 2,740 | 61.0% | 55.4% | **+41.7%** | **+37.5%** | -3.7% | +57.3% |
| **La Liga** | 2,760 | 59.5% | 54.1% | **+39.8%** | **+35.6%** | -0.9% | +58.5% |
| **Serie A** | 2,740 | 58.9% | 54.9% | **+29.5%** | **+25.6%** | -1.8% | +63.9% |

---

### 2. Breakdown by Odds Range (Odds Stratification)

To see where the predictive advantage lies, selections are stratified across standard decimal odds brackets:

| Odds Bracket | Bets ($N$) | Model Win Rate | Baseline Win Rate | Model ROI (Gross) | Model ROI (Net 3%) | Baseline ROI | Mean CLV |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **1.00 – 1.50** | 507 | 78.9% | 78.9% | +4.5% | +1.4% | +4.5% | +14.4% |
| **1.50 – 1.90** | 617 | 69.0% | 69.0% | +14.7% | +11.2% | +14.7% | +29.7% |
| **1.90 – 2.50** | 544 | 61.2% | 61.2% | +31.4% | +27.4% | +31.4% | +53.2% |
| **> 2.50** | 1,072 | 42.4% | 32.2% | +48.9% | +44.4% | -31.0% | +112.3% |
| **> 1.90 (High-Value)** | **1,616** | **48.7%** | **42.0%** | **+43.0%** | **+38.7%** | **-10.0%** | **+92.4%** |
| *All Matches* | *2,740* | *58.9%* | *54.9%* | *+29.5%* | *+25.6%* | *-1.8%* | *+63.9%* |

---

### 3. Understanding the Model's Edge

- **Beating the Favourite Trap:**
  Blindly backing the market favourite loses money across every single league ($-0.9\%$ to $-5.6\%$ flat ROI) due to bookmaker vigorish. In contrast, AModel's probability distributions consistently beat the closing line by double digits.
- **High-Value Edge ($> 1.90$ Odds):**
  In the high-value bracket ($\text{odds} \ge 1.90$), the model achieves a **48.7% win rate** and **+43.0% gross ROI** (+38.7% net) vs. the market favourite's $-10.0\%$ ROI. The model generates this advantage by identifying underpriced away wins and draws (~20% draw pick rate with a 36% win rate vs. ~29% market implied).
- **Longshot Value ($> 2.50$ Odds):**
  At prices $> 2.50$, the baseline favourite strategy drops to a $-31.0\%$ ROI, whereas the model delivers **+48.9% gross ROI** (+44.4% net) with a 42.4% hit rate.
- **Closing Line Value (CLV):**
  Closing Line Value evaluates whether our predicted probabilities systematically find edge against de-vigged closing market prices:
  $$\text{CLV} = \frac{p_{\text{model}}}{p_{\text{closing, devigged}}} - 1$$
  An average CLV of **+63.9%** overall and **+92.4%** in the $> 1.90$ range demonstrates persistent statistical edge rather than short-term luck.

#### Reproduce the Backtest
Run the backtest CLI locally for any registered league:
```bash
# Run walk-forward backtest and print stratified tables
python -m src.ml.backtest --league serie_a
python -m src.ml.backtest --league premier_league
```
Summary reports and Markdown tables are automatically saved to `data/export/ml/<league>/results/clv_odds_summary.md`.

---

## Data Sources

AModel integrates multiple high-quality data providers to ensure accuracy and statistical depth:
- **Football-Data.org / API-Football:** For fixture lists, live scores, live match events, and historical results.
- **Understat / FBref (Scraped metrics):** For advanced metrics such as Expected Goals ($xG$), Expected Assists ($xA$), and deep defensive/offensive pressure indicators.
- **Bookmaker Odds APIs:** For market sentiment, implied probabilities, and closing odds benchmarking.

---

## Supported Leagues

The framework is configured to support major domestic leagues and international competitions, including:
- **English Premier League** (England)
- **La Liga** (Spain)
- **Serie A** (Italy)
- **Bundesliga** (Germany)
- **Ligue 1** (France)
- **UEFA Champions League & Europa League** (Europe)

---

## Predictive Features

The prediction engine relies on a rich feature set to capture team form and contextual variables:
- **Recent Form & Momentum:** Rolling averages of points, goals scored, and goals conceded over the last 5 and 10 matches.
- **Expected Goals ($xG$) Metrics:** Differential between $xG$ for and $xG$ against to evaluate true performance independent of finishing variance.
- **Elo Ratings:** Dynamic strength ratings adjusted for home-field advantage and opponent difficulty.
- **Head-to-Head (H2H) History:** Historical outcomes between specific opponents.
- **Contextual Factors:** Rest days between matches, travel distance, and key player availability.

---

## Setup & Installation

To set up the project environment locally, run the automated setup script.

### Prerequisites
- Python 3.9 or higher
- `pip` and `virtualenv`

### Quick Setup
Run the provided setup script in your terminal:

```bash
# Make the setup script executable and run it
chmod +x setup.sh
./setup.sh
```

*(Alternatively, manual setup can be performed by running:)*
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Project

Once the environment is configured, you can execute the data pipeline, train models, and generate match predictions using the run script.

### Execute Pipeline
```bash
# Make the run script executable and run it
chmod +x run.sh
./run.sh
```

This script launches the interactive Streamlit UI:
```bash
./run.sh  # or .venv/bin/python -m streamlit run src/ui/app.py
```
You can also run pipeline modules directly via CLI:
1. Data fetching and merging (`python -m src.fetch.football_data_csv --league serie_a`, `python -m src.fetch.understat --league serie_a`, `python -m src.clean.merge --league serie_a`)
2. Database upsert (`python -m src.db.upsert --league serie_a`)
3. Preprocessing (`python -m src.preprocess.clean --league serie_a`)
4. Model training and evaluation (`python -m src.ml.train --league serie_a`)
5. Backtest & CLV evaluation (`python -m src.ml.backtest --league serie_a`)
6. Matchday prediction (`python -m src.ml.predict --league serie_a --demo`)

