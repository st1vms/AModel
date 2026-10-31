"""Backtest & CLV evaluation: does the model have real betting edge?

Walk-forward backtest (one fold per validation season, expanding window) that
scores each match with a classifier retrained on strictly earlier seasons,
then evaluates the picks against **closing** market prices:

- **CLV** (closing line value): per-bet ``p_pick / p_closing_devigged - 1``;
  positive mean => the model systematically finds value vs the closing line.
  Also log-loss / Brier of model probs vs de-vigged closing probs.
- **ROI**: flat 1-unit staking and fractional Kelly (compounding bankroll),
  gross (raw closing odds, which already carry the book margin) and net of an
  extra flat commission (default 3%).
- **Baseline**: market favourite — back whichever side has the higher de-vigged
  closing implied probability (ties -> home), staked at closing odds.
- **Confidence slicing**: performance by model confidence bin on the favourite
  side (``p_fav = max(p_home, p_away)``): ``<0.50``, ``0.50-0.70``, ``>=0.70``.

Closing odds (B365) exist only from 2019-20 onward; earlier seasons are kept
for accuracy but flagged ``has_closing=False`` and excluded from CLV/ROI.

Usage
-----
    python -m src.ml.backtest --league serie_a [--start 2017-18] [--end 2026-26]
                              [--vig 0.03] [--kelly-fraction 0.25]

Artifacts (per league, under ``config.ml_results_dir(league)``):
    clv_backtest.csv          per-match log
    clv_backtest_summary.csv  one row per (scope, segment)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from src.ml import data, evaluate, models, predict, splits

# ---------------------------------------------------------------------------
# Market math
# ---------------------------------------------------------------------------

def implied_probs(
    odds_h: pd.Series, odds_d: pd.Series, odds_a: pd.Series
) -> pd.DataFrame:
    """Raw implied probabilities ``1/odds`` (NaN-safe)."""
    return pd.DataFrame(
        {"home": 1.0 / odds_h, "draw": 1.0 / odds_d, "away": 1.0 / odds_a}
    )


def devig_probs(
    odds_h: pd.Series, odds_d: pd.Series, odds_a: pd.Series
) -> pd.DataFrame:
    """De-vigged market probabilities (``inv / sum(inv)``).

    Same convention as ``preprocess.clean._market_features``: a row with any
    missing odds falls back to 1/3 on every outcome.
    """
    raw = implied_probs(odds_h, odds_d, odds_a)
    # skipna=False so a single missing odds NaNs the whole row (matching
    # preprocess.clean._market_features), then falls back to 1/3 everywhere.
    total = raw.sum(axis=1, skipna=False)
    return raw.div(total, axis=0).fillna(1.0 / 3.0)


def has_closing_mask(df: pd.DataFrame) -> pd.Series:
    """True where all three closing odds are present."""
    return (
        df["b365ch"].notna() & df["b365cd"].notna() & df["b365ca"].notna()
    )


# ---------------------------------------------------------------------------
# Loss / scoring helpers
# ---------------------------------------------------------------------------

def log_loss(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multi-class log loss (lower is better)."""
    eps = 1e-15
    p = np.clip(proba, eps, 1.0 - eps)
    return float(-np.mean(np.log(p[np.arange(len(y_true)), y_true])))


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multi-class Brier score (lower is better)."""
    onehot = np.eye(proba.shape[1])[y_true]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


# ---------------------------------------------------------------------------
# Betting metrics
# ---------------------------------------------------------------------------

def roi_flat(wins: pd.Series, odds: pd.Series, vig: float = 0.0) -> float:
    """ROI under flat 1-unit staking.

    Win: ``odds * (1 - vig) - 1``; loss: ``-1``. Returns mean profit per unit
    staked (ROI). ``vig`` is an *extra* flat commission on top of the odds.
    """
    if len(wins) == 0:
        return float("nan")
    profit = np.where(wins.to_numpy(), odds.to_numpy() * (1.0 - vig) - 1.0, -1.0)
    return float(profit.mean())


def roi_kelly(
    p: pd.Series,
    odds: pd.Series,
    wins: pd.Series,
    fraction: float = 0.25,
    vig: float = 0.0,
) -> float:
    """ROI under fractional Kelly staking with a compounding bankroll (start 1.0).

    Per-bet Kelly fraction ``f = (p*odds - 1) / (odds - 1)`` (clipped at 0),
    scaled by ``fraction``. Returns bankroll growth (ROI).
    """
    if len(wins) == 0:
        return float("nan")
    o = odds.to_numpy()
    f = np.where(o > 1.0, (p.to_numpy() * o - 1.0) / np.maximum(o - 1.0, 1e-9), 0.0)
    f = np.clip(f, 0.0, None) * fraction
    bankroll = 1.0
    for w, od, s in zip(wins.to_numpy(), o, f):
        if w:
            bankroll *= 1.0 + s * (od * (1.0 - vig) - 1.0)
        else:
            bankroll *= 1.0 - s
    return bankroll - 1.0


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def _default_factory(pipeline: dict):
    """Build a per-fold model factory from the saved pipeline metadata.

    Reuses the saved classifier name + tuned params (no grid search), so the
    backtest evaluates the same model family the pipeline ships.
    """
    name = pipeline["classifier_name"]
    params = pipeline["tuned_params"].get(f"clf:{name}", {})

    def factory(X_train: pd.DataFrame, y_train: pd.Series) -> object:
        cw = models.inverse_class_weights(y_train)
        model = models.make_classifier(name, class_weights=cw, **params)
        models.fit_classifier(model, X_train, y_train, class_weights=cw)
        return model

    return factory


def generate_predictions(
    df: pd.DataFrame,
    features: list[str],
    start: str,
    end: str,
    model_factory,
    log=print,
) -> pd.DataFrame:
    """Walk-forward predictions: one fold per validation season in [start, end].

    ``model_factory(X_train, y_train)`` must return a fitted classifier with
    ``predict_proba``. Returns a per-match frame with model probabilities and
    the raw odds columns.
    """
    frames = []
    for sp in splits.walk_forward_splits(df, start=start, end=end):
        X_train, y_train = sp.train[features], sp.train["outcome"]
        X_valid = sp.valid[features]
        model = model_factory(X_train, y_train)
        proba = model.predict_proba(X_valid)
        out = sp.valid[
            ["date", "season", "home_team", "away_team", "outcome",
             "b365h", "b365d", "b365a", "b365ch", "b365cd", "b365ca"]
        ].copy()
        out["p_home"] = proba[:, 0]
        out["p_draw"] = proba[:, 1]
        out["p_away"] = proba[:, 2]
        out["fold"] = sp.name
        frames.append(out)
        log(
            f"  fold {sp.name}: {len(out)} matches "
            f"(train {len(sp.train)})"
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Betting frame assembly
# ---------------------------------------------------------------------------

CONFIDENCE_BINS = [
    ("<0.50", lambda p: p < 0.50),
    ("0.50-0.70", lambda p: (p >= 0.50) & (p < 0.70)),
    (">=0.70", lambda p: p >= 0.70),
]

MUTUAL_ODDS_BINS: list[tuple[str, Any]] = [
    ("1.00-1.50", lambda o: (o >= 1.0) & (o < 1.50)),
    ("1.50-1.90", lambda o: (o >= 1.50) & (o < 1.90)),
    ("1.90-2.50", lambda o: (o >= 1.90) & (o <= 2.50)),
    (">2.50", lambda o: o > 2.50),
]

ODDS_BINS: list[tuple[str, Any]] = MUTUAL_ODDS_BINS + [
    (">1.90", lambda o: o >= 1.90),
]


def assign_odds_bracket(
    odds: pd.Series | float | int | None,
) -> pd.Series | str | None:
    """Assign decimal odds to a mutual bracket string:
    '1.00-1.50', '1.50-1.90', '1.90-2.50', '>2.50' (or NaN/None if missing/invalid).
    """
    if isinstance(odds, pd.Series):
        res = pd.Series(index=odds.index, dtype=object)
        for name, fn in MUTUAL_ODDS_BINS:
            # Drop NaN/invalids before indexing
            valid = odds.notna()
            mask = valid & fn(odds)
            res.loc[mask] = name
        return res
    if odds is None or pd.isna(odds):
        return None
    val = float(odds)
    for name, fn in MUTUAL_ODDS_BINS:
        if fn(val):
            return name
    return None


def _pick_and_odds(probs: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """argmax pick (ties -> home) and the decimal odds of the picked outcome."""
    pick = probs.to_numpy().argmax(axis=1)
    pick_s = pd.Series(pick, index=probs.index)
    odds_arr = odds.to_numpy()
    pick_odds = pd.Series(
        odds_arr[np.arange(len(probs)), pick], index=probs.index
    )
    return pick_s, pick_odds


def build_betting_frame(preds: pd.DataFrame) -> pd.DataFrame:
    """Attach picks, closing probs, CLV, and the market-favourite baseline."""
    df = preds.copy()

    # --- model pick ---
    model_probs = pd.DataFrame(
        {"home": df["p_home"], "draw": df["p_draw"], "away": df["p_away"]}
    )
    df["pick"] = model_probs.to_numpy().argmax(axis=1)
    df["p_pick"] = model_probs.to_numpy()[np.arange(len(df)), df["pick"].to_numpy()]

    # --- confidence on the favourite side ---
    df["p_fav"] = df[["p_home", "p_away"]].max(axis=1)
    df["confidence_bin"] = np.select(
        [fn(df["p_fav"]) for _, fn in CONFIDENCE_BINS],
        [name for name, _ in CONFIDENCE_BINS],
        default="?",
    )

    # --- closing market (de-vigged) ---
    df["has_closing"] = has_closing_mask(df)
    closing = devig_probs(df["b365ch"], df["b365cd"], df["b365ca"])
    df["cp_home"] = closing["home"]
    df["cp_draw"] = closing["draw"]
    df["cp_away"] = closing["away"]

    # --- model CLV & closing-odds stake ---
    cp_pick = closing.to_numpy()[np.arange(len(df)), df["pick"].to_numpy()]
    df["clv_pick"] = df["p_pick"] / cp_pick - 1.0
    # CLV is only meaningful where closing odds exist (the 1/3 fallback
    # would otherwise dominate the mean on pre-2019-20 rows).
    df.loc[~df["has_closing"], "clv_pick"] = np.nan
    closing_odds = pd.DataFrame(
        {
            "home": df["b365ch"],
            "draw": df["b365cd"],
            "away": df["b365ca"],
        }
    )
    # Decimal closing odds of the model's picked outcome (NaN pre-2019-20).
    df["pick_odds_closing"] = closing_odds.to_numpy()[
        np.arange(len(df)), df["pick"].to_numpy()
    ]
    df["odds_bracket"] = assign_odds_bracket(df["pick_odds_closing"])
    df["odds_bin"] = df["odds_bracket"]
    df["win"] = df["pick"] == df["outcome"]

    # --- baseline: market favourite (closing; opening fallback pre-2019-20) ---
    opening = devig_probs(df["b365h"], df["b365d"], df["b365a"])
    base_probs = closing.where(df["has_closing"], opening)
    base_odds = pd.DataFrame(
        {
            "home": df["b365ch"].fillna(df["b365h"]),
            "draw": df["b365cd"].fillna(df["b365d"]),
            "away": df["b365ca"].fillna(df["b365a"]),
        }
    )
    df["base_pick"], df["base_odds"] = _pick_and_odds(base_probs, base_odds)
    df["base_p"] = base_probs.to_numpy()[
        np.arange(len(df)), df["base_pick"].to_numpy()
    ]
    df["base_win"] = df["base_pick"] == df["outcome"]
    return df


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _segment_metrics(
    df: pd.DataFrame, scope: str, vig: float, kelly_fraction: float
) -> dict:
    """Metrics for one (scope, segment) slice. scope in {model, baseline}."""
    if scope == "model":
        pick, p, odds, win = df["pick"], df["p_pick"], df["pick_odds_closing"], df["win"]
        # ROI is only defined where closing odds exist (stake is at closing).
        stake = df[odds.notna()]
        p, odds, win = p.loc[stake.index], odds.loc[stake.index], win.loc[stake.index]
    else:
        pick, p, odds, win = df["base_pick"], df["base_p"], df["base_odds"], df["base_win"]
        stake = df

    row = {
        "scope": scope,
        "n": int(len(df)),
        "n_staked": int(len(stake)),
        "accuracy": float((pick == df["outcome"]).mean()) if len(df) else np.nan,
        "roi_flat_gross": roi_flat(win, odds, vig=0.0) if len(win) else np.nan,
        "roi_flat_net": roi_flat(win, odds, vig=vig) if len(win) else np.nan,
        "roi_kelly_gross": roi_kelly(p, odds, win, kelly_fraction, 0.0) if len(win) else np.nan,
        "roi_kelly_net": roi_kelly(p, odds, win, kelly_fraction, vig) if len(win) else np.nan,
    }

    if scope == "model":
        probs = df[["p_home", "p_draw", "p_away"]].to_numpy()
        y = df["outcome"].to_numpy()
        row["log_loss"] = log_loss(y, probs) if len(df) else np.nan
        row["brier"] = brier_score(y, probs) if len(df) else np.nan
        clv = df["clv_pick"].dropna()
        row["mean_clv"] = float(clv.mean()) if len(clv) else np.nan
        row["median_clv"] = float(clv.median()) if len(clv) else np.nan
        # closing-line benchmark (only meaningful where closing odds exist)
        if df["has_closing"].any():
            cp = df.loc[df["has_closing"], ["cp_home", "cp_draw", "cp_away"]].to_numpy()
            yc = df.loc[df["has_closing"], "outcome"].to_numpy()
            row["log_loss_closing"] = log_loss(yc, cp)
            row["brier_closing"] = brier_score(yc, cp)
        else:
            row["log_loss_closing"] = np.nan
            row["brier_closing"] = np.nan
    else:
        row["mean_clv"] = np.nan
        row["median_clv"] = np.nan
        row["log_loss"] = np.nan
        row["brier"] = np.nan
        row["log_loss_closing"] = np.nan
        row["brier_closing"] = np.nan
    return row


def build_summary(
    df: pd.DataFrame, vig: float, kelly_fraction: float
) -> pd.DataFrame:
    """One row per (scope, segment): all / has_closing / confidence bins / odds bins."""
    segments: list[tuple[str, pd.DataFrame]] = [
        ("all", df),
        ("has_closing", df[df["has_closing"]]),
    ]
    for name, fn in CONFIDENCE_BINS:
        segments.append((f"bin_{name}", df[fn(df["p_fav"])]))

    for name, fn in ODDS_BINS:
        segments.append(
            (f"odds_{name}", df[df["has_closing"] & fn(df["pick_odds_closing"])])
        )

    rows = []
    for scope in ("model", "baseline"):
        for seg_name, seg in segments:
            row = _segment_metrics(seg, scope, vig, kelly_fraction)
            row["segment"] = seg_name
            rows.append(row)
    return pd.DataFrame(rows)


def build_odds_summary(
    df: pd.DataFrame, vig: float = 0.03, kelly_fraction: float = 0.25
) -> pd.DataFrame:
    """Stratified backtest summary table by odds bracket comparing model vs baseline."""
    closing_df = df[df["has_closing"]].copy()

    brackets = [
        ("1.00-1.50", lambda o: (o >= 1.0) & (o < 1.50)),
        ("1.50-1.90", lambda o: (o >= 1.50) & (o < 1.90)),
        ("1.90-2.50", lambda o: (o >= 1.90) & (o <= 2.50)),
        (">2.50", lambda o: o > 2.50),
        (">1.90", lambda o: o >= 1.90),
        ("All", lambda o: pd.Series(True, index=o.index)),
    ]

    rows = []
    for name, fn in brackets:
        seg = closing_df[fn(closing_df["pick_odds_closing"])]
        n = len(seg)
        if n > 0:
            m_win = seg["win"]
            m_odds = seg["pick_odds_closing"]
            m_acc = float((seg["pick"] == seg["outcome"]).mean())
            m_roi_gross = roi_flat(m_win, m_odds, vig=0.0)
            m_roi_net = roi_flat(m_win, m_odds, vig=vig)

            b_win = seg["base_win"]
            b_odds = seg["base_odds"]
            b_acc = float((seg["base_pick"] == seg["outcome"]).mean())
            b_roi_gross = roi_flat(b_win, b_odds, vig=0.0)
            b_roi_net = roi_flat(b_win, b_odds, vig=vig)

            clv = seg["clv_pick"].dropna()
            mean_clv = float(clv.mean()) if len(clv) else np.nan
            median_clv = float(clv.median()) if len(clv) else np.nan
        else:
            m_acc = np.nan
            b_acc = np.nan
            m_roi_gross = np.nan
            m_roi_net = np.nan
            b_roi_gross = np.nan
            b_roi_net = np.nan
            mean_clv = np.nan
            median_clv = np.nan

        rows.append(
            {
                "bracket": name,
                "n_bets": int(n),
                "model_accuracy": m_acc,
                "base_accuracy": b_acc,
                "model_roi_gross": m_roi_gross,
                "model_roi_net": m_roi_net,
                "base_roi_gross": b_roi_gross,
                "base_roi_net": b_roi_net,
                "mean_clv": mean_clv,
                "median_clv": median_clv,
            }
        )
    return pd.DataFrame(rows)


def format_odds_summary_markdown(
    summary: pd.DataFrame,
    league: str | None = None,
) -> str:
    """Format odds summary DataFrame into a clean Markdown table with high-value highlight."""
    def _fmt_pct(v: float, signed: bool = False) -> str:
        if pd.isna(v):
            return "N/A"
        pct = v * 100.0
        prefix = "+" if signed and pct > 0 else ""
        return f"{prefix}{pct:.1f}%"

    def _fmt_bracket_label(b: str) -> str:
        mapping = {
            "1.00-1.50": "1.00 - 1.50",
            "1.50-1.90": "1.50 - 1.90",
            "1.90-2.50": "1.90 - 2.50",
            ">2.50": "> 2.50",
            ">1.90": "> 1.90 (High-Value)",
            "All": "All Matches",
        }
        return mapping.get(b, b)

    lines = []
    if league:
        title = league.replace("_", " ").title()
        lines.append(f"### Backtest Odds Stratification Summary ({title})\n")
    else:
        lines.append("### Backtest Odds Stratification Summary\n")

    lines.append("| Odds Bracket | Bets (N) | Model Acc | Baseline Acc | Model ROI (Gross) | Model ROI (Net) | Baseline ROI | Mean CLV |")
    lines.append("|:---|---:|---:|---:|---:|---:|---:|---:|")

    high_val_row = None
    for _, r in summary.iterrows():
        b = str(r["bracket"])
        label = _fmt_bracket_label(b)
        n = f"{int(r['n_bets']):,}"
        m_acc = _fmt_pct(r["model_accuracy"])
        b_acc = _fmt_pct(r["base_accuracy"])
        m_gross = _fmt_pct(r["model_roi_gross"], signed=True)
        m_net = _fmt_pct(r["model_roi_net"], signed=True)
        b_gross = _fmt_pct(r["base_roi_gross"], signed=True)
        clv = _fmt_pct(r["mean_clv"], signed=True)

        if b == ">1.90":
            high_val_row = r
            lines.append(
                f"| **{label}** | **{n}** | **{m_acc}** | **{b_acc}** | **{m_gross}** | **{m_net}** | **{b_gross}** | **{clv}** |"
            )
        elif b == "All":
            lines.append(
                f"| *{label}* | *{n}* | *{m_acc}* | *{b_acc}* | *{m_gross}* | *{m_net}* | *{b_gross}* | *{clv}* |"
            )
        else:
            lines.append(
                f"| {label} | {n} | {m_acc} | {b_acc} | {m_gross} | {m_net} | {b_gross} | {clv} |"
            )

    if high_val_row is not None and int(high_val_row["n_bets"]) > 0:
        hv_n = f"{int(high_val_row['n_bets']):,}"
        hv_acc = _fmt_pct(high_val_row["model_accuracy"])
        hv_gross = _fmt_pct(high_val_row["model_roi_gross"], signed=True)
        hv_net = _fmt_pct(high_val_row["model_roi_net"], signed=True)
        hv_base_gross = _fmt_pct(high_val_row["base_roi_gross"], signed=True)
        hv_clv = _fmt_pct(high_val_row["mean_clv"], signed=True)
        lines.append("\n> **High-Value Bracket Highlights (> 1.90):**")
        lines.append(f"> - **Selections**: {hv_n} bets placed at odds ≥ 1.90.")
        lines.append(f"> - **Model Win Rate**: {hv_acc} vs baseline {hv_base_gross} ROI.")
        lines.append(f"> - **ROI**: {hv_gross} flat gross ({hv_net} net of 3% flat commission).")
        lines.append(f"> - **Closing Line Value (CLV)**: {hv_clv} mean edge vs de-vigged closing odds.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_backtest(
    league: str = config.DEFAULT_LEAGUE,
    start: str = "2017-18",
    end: str = "2026-26",
    vig: float = 0.03,
    kelly_fraction: float = 0.25,
    out_dir: str | Path | None = None,
    model_factory=None,
    log=print,
) -> dict:
    """Full backtest for one league. Returns {league, per_match, summary, paths}."""
    log(f"[{league}] loading data ...")
    df, labels, feats = data.load_data(league)
    log(f"[{league}] {len(df)} matches, {len(feats)} features")

    if model_factory is None:
        pipeline = predict.load_pipeline(league)
        model_factory = _default_factory(pipeline)
        log(f"[{league}] classifier: {pipeline['classifier_name']}")

    log(f"[{league}] walk-forward {start} -> {end} ...")
    preds = generate_predictions(df, feats, start, end, model_factory, log=log)
    betting = build_betting_frame(preds)
    summary = build_summary(betting, vig, kelly_fraction)
    odds_summary = build_odds_summary(betting, vig=vig, kelly_fraction=kelly_fraction)
    odds_md = format_odds_summary_markdown(odds_summary, league=league)

    out = Path(out_dir) if out_dir else config.ml_results_dir(league)
    out.mkdir(parents=True, exist_ok=True)
    per_match_path = out / "clv_backtest.csv"
    summary_path = out / "clv_backtest_summary.csv"
    odds_summary_path = out / "clv_odds_summary.csv"
    odds_md_path = out / "clv_odds_summary.md"

    betting.to_csv(per_match_path, index=False)
    summary.to_csv(summary_path, index=False)
    odds_summary.to_csv(odds_summary_path, index=False)
    odds_md_path.write_text(odds_md, encoding="utf-8")

    log(f"[{league}] wrote {per_match_path}")
    log(f"[{league}] wrote {summary_path}")
    log(f"[{league}] wrote {odds_summary_path}")
    log(f"[{league}] wrote {odds_md_path}")
    log(f"[{league}] summary:")
    log(evaluate.table_to_string(summary))
    log(f"[{league}] odds stratification summary:\n{odds_md}")
    return {
        "league": league,
        "per_match": betting,
        "summary": summary,
        "odds_summary": odds_summary,
        "odds_markdown": odds_md,
        "per_match_path": per_match_path,
        "summary_path": summary_path,
        "odds_summary_path": odds_summary_path,
        "odds_md_path": odds_md_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--league", default=config.DEFAULT_LEAGUE,
                    choices=sorted(config.LEAGUES))
    ap.add_argument("--start", default="2017-18",
                    help="first validation season (default 2017-18)")
    ap.add_argument("--end", default="2026-26",
                    help="last validation season (default 2026-26; 2026-27 is partial)")
    ap.add_argument("--vig", type=float, default=0.03,
                    help="extra flat commission for the 'net' ROI (default 0.03)")
    ap.add_argument("--kelly-fraction", type=float, default=0.25,
                    help="fraction of full Kelly (default 0.25)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default data/export/ml/{league}/results)")
    args = ap.parse_args()

    run_backtest(
        league=args.league,
        start=args.start,
        end=args.end,
        vig=args.vig,
        kelly_fraction=args.kelly_fraction,
        out_dir=args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
