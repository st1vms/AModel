"""Streamlit UI: update data, train models, predict fixtures.

Usage
-----
    .venv/bin/python -m streamlit run src/ui/app.py

Three tabs:
  1. **Update data** — fetch football-data.co.uk + Understat (network),
     merge, upsert into SQLite, rebuild ``matches_clean.csv``.
  2. **Train** — run the full training pipeline (tune -> train -> evaluate
     -> save ``pipeline.joblib``) and show results tables + plots.
  3. **Predict** — auto-pull the scheduled (unplayed) matches of the
     selected season from the raw Understat JSON and show 1X2
     probabilities + expected goals from the saved pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the workspace root importable regardless of the launch directory
# (Streamlit puts the script's own directory on sys.path, not the root).
WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import pandas as pd
import streamlit as st

import config
from src.clean.merge import merge
from src.db import upsert
from src.fetch import football_data_csv, understat
from src.ml import data as ml_data
from src.ml import fixtures as ml_fixtures
from src.ml import predict as ml_predict
from src.ml import train as ml_train
from src.preprocess import clean as preprocess

def _current_league() -> str:
    """League selected in the previous run (for the page title)."""
    return st.session_state.get("league", config.DEFAULT_LEAGUE)


st.set_page_config(
    page_title=f"AModel — {config.LEAGUES[_current_league()]['name']}",
    page_icon="⚽", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_pipeline_cached(path: str):
    # Pass `path` by keyword: load_pipeline's first positional param is
    # `league`, so a bare positional call would treat the path as a league key
    # and double the path (ml_models_dir(<path>) / "pipeline.joblib").
    return ml_predict.load_pipeline(path=path)


@st.cache_data(show_spinner=False)
def db_summary(league: str) -> dict:
    """Quick status of the SQLite DB for a league (count, latest, seasons)."""
    import sqlite3
    if not config.DB_PATH.exists():
        return {}
    conn = sqlite3.connect(config.DB_PATH)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM matches WHERE league = ?", (league,)
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(date) FROM matches WHERE league = ?", (league,)
        ).fetchone()[0]
        seasons = dict(conn.execute(
            "SELECT season, COUNT(*) FROM matches "
            "WHERE league = ? GROUP BY season", (league,)
        ).fetchall())
    finally:
        conn.close()
    return {"n_matches": n, "latest_date": latest, "seasons": seasons}


@st.cache_data(show_spinner=False)
def clean_csv_info(league: str) -> dict:
    """Status of the ML-ready CSV for a league (rows, latest date)."""
    path = config.ml_dir(league) / "matches_clean.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, parse_dates=["date"], usecols=["date"])
    return {"n_rows": len(df), "latest_date": str(df["date"].max().date()),
            "mtime": path.stat().st_mtime}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar() -> dict:
    league = st.sidebar.selectbox(
        "League", config.LEAGUE_ORDER,
        format_func=lambda k: config.LEAGUES[k]["name"],
        index=config.LEAGUE_ORDER.index(_current_league()),
        help="League to train on and predict. Each league has its own raw "
             "data, clean CSV, results and saved pipeline.")
    st.session_state["league"] = league
    spec = config.LEAGUES[league]

    st.sidebar.title(f"⚽ AModel — {spec['name']}")
    st.sidebar.caption(
        f"{spec['seasons'][0]} → {spec['seasons'][-1]} · "
        "football-data.co.uk + Understat"
    )

    season = st.sidebar.selectbox(
        "Season (for predictions)", spec["seasons"],
        index=spec["seasons"].index(spec["in_progress"]),
        help="Season whose scheduled fixtures will be predicted.")
    max_md = config.max_matchdays(league)
    n_matchdays = st.sidebar.number_input(
        "Matchdays to predict", min_value=1, max_value=max_md, value=1,
        step=1,
        help=f"How many upcoming matchdays (distinct dates) to predict. "
             f"{max_md} = the whole remaining season.")

    st.sidebar.divider()
    st.sidebar.subheader("Training options")
    include_ht = st.sidebar.checkbox(
        "Include HT goals", value=False,
        help="Adds halftime goals as features (leakage for pre-match use).")
    no_xg = st.sidebar.checkbox(
        "Drop xG features", value=False,
        help="Strict pre-match variant (no xG columns).")
    walk_forward = st.sidebar.checkbox(
        "Walk-forward evaluation", value=False,
        help="Also run the expanding-window evaluation (2022-23..2025-26).")
    skip_tuning = st.sidebar.checkbox(
        "Skip tuning (faster)", value=False,
        help="Use default hyperparameters instead of the light grid search.")

    st.sidebar.divider()
    st.sidebar.caption(
        f"DB: {config.DB_PATH.name} · models: "
        f"{config.ml_models_dir(league).name}/pipeline.joblib"
    )
    return {
        "league": league,
        "season": season,
        "n_matchdays": int(n_matchdays),
        "include_ht": include_ht,
        "no_xg": no_xg,
        "walk_forward": walk_forward,
        "skip_tuning": skip_tuning,
    }


# ---------------------------------------------------------------------------
# Tab 1 — Update data
# ---------------------------------------------------------------------------

def _rebuild_clean_csv(league: str) -> tuple[int, list[str]]:
    """DB -> clean/encode/feature-engineer -> matches_clean.csv + labels.

    Mirrors ``src/preprocess/clean.main`` (label encoding, default options).
    Returns (n_rows, notes).
    """
    raw = preprocess.load_matches(league=league)  # from the DB
    out, notes, label_map = preprocess.prepare(raw, encoding="label")
    out_path = config.ml_dir(league) / "matches_clean.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    if label_map is not None:
        # label_map is team -> code; emit (code, team) rows so the "code"
        # column holds the int and "team" the name (matches load_team_labels).
        pd.DataFrame(
            [(code, team) for team, code in sorted(label_map.items(),
                                                   key=lambda kv: kv[1])],
            columns=["code", "team"],
        ).to_csv(out_path.with_name("team_labels.csv"), index=False)
    return len(out), notes


def tab_update(opts: dict) -> None:
    league = opts["league"]
    st.subheader(f"Update data — {config.LEAGUES[league]['name']}")
    st.caption(
        "Fetches the latest football-data.co.uk CSVs and Understat JSON "
        "(network, ~1–2 min with polite delays), merges both sources, "
        "upserts into SQLite, and rebuilds `matches_clean.csv`."
    )

    force = st.checkbox(
        "Force re-download (ignore local cache)", value=True,
        help="Off = skip seasons whose raw files already exist.")

    if st.button("⬇️ Fetch latest & rebuild", type="primary"):
        log_lines: list[str] = []

        def log(msg: str) -> None:
            log_lines.append(str(msg))

        import contextlib
        import io

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                with st.status("Updating data…", expanded=True) as status:
                    status.write("**1/4** Fetching football-data.co.uk …")
                    football_data_csv.fetch_all(league, force=force)
                    status.write("**2/4** Fetching Understat …")
                    understat.fetch_all(league, force=force)
                    status.write("**3/4** Merging & upserting into SQLite …")
                    n_loaded = upsert.load(league)
                    status.write(f"**4/4** Rebuilding `matches_clean.csv` …")
                    n_rows, notes = _rebuild_clean_csv(league)
            status.update(label="Data updated", state="complete")
        except Exception as e:  # noqa: BLE001 — surface any stage failure
            st.error(f"Update failed: {e}")
            with st.expander("Log"):
                st.code("\n".join(log_lines) + "\n" + buf.getvalue())
            return

        st.cache_data.clear()
        st.success(
            f"Done — {n_loaded} matches in DB, {n_rows} rows in "
            f"`matches_clean.csv`."
        )
        with st.expander("Pipeline log"):
            st.code(buf.getvalue().strip() or "(no output)")
        for n in notes:
            st.caption(f"[note] {n}")

    # Current status.
    db = db_summary(league)
    cc = clean_csv_info(league)
    if db:
        c1, c2, c3 = st.columns(3)
        c1.metric("Matches in DB", db["n_matches"])
        c2.metric("Latest match date", db["latest_date"])
        c3.metric("Seasons", len(db["seasons"]))
        st.caption(
            "Per season: "
            + " · ".join(f"{s}: {n}" for s, n in sorted(db["seasons"].items()))
        )
    if cc:
        st.caption(
            f"`matches_clean.csv`: {cc['n_rows']} rows, latest "
            f"{cc['latest_date']}"
        )
    else:
        st.info("No `matches_clean.csv` yet — run the update above.")


# ---------------------------------------------------------------------------
# Tab 2 — Train
# ---------------------------------------------------------------------------

def tab_train(opts: dict) -> None:
    league = opts["league"]
    st.subheader(f"Train models — {config.LEAGUES[league]['name']}")
    st.caption(
        "Tune (light grid) → train all classifiers & goal regressors → "
        "evaluate (single split"
        + (" + walk-forward" if opts["walk_forward"] else "")
        + ") → save `pipeline.joblib`. ~30 s with tuning, a few seconds "
        "without."
    )
    cc = clean_csv_info(league)
    if not cc:
        st.info("Run **Update data** first (needs `matches_clean.csv`).")
        return

    if st.button("🏋️ Train models", type="primary"):
        lines: list[str] = []

        def log(msg: str) -> None:
            lines.append(str(msg))

        with st.spinner("Training… (this can take ~30 s with tuning)"):
            try:
                res = ml_train.run_training(
                    league=league,
                    include_ht=opts["include_ht"],
                    no_xg=opts["no_xg"],
                    walk_forward=opts["walk_forward"],
                    skip_tuning=opts["skip_tuning"],
                    log=log,
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Training failed: {e}")
                with st.expander("Log"):
                    st.code("\n".join(lines))
                return

        st.cache_data.clear()  # pipeline.joblib changed
        st.success(
            f"Trained in {res['elapsed']:.1f}s — best classifier: "
            f"**{res['best_name']}** (saved to "
            f"`{res['saved']['pipeline'].name}`)."
        )

        st.markdown("#### Classification (single split)")
        st.dataframe(res["clf_frame"], width="stretch")
        st.markdown("#### Goal regression (single split)")
        st.dataframe(res["reg_frame"], width="stretch")

        if res["wf_clf_frame"] is not None:
            st.markdown("#### Walk-forward aggregate (mean ± std)")
            st.dataframe(res["clf_agg_frame"], width="stretch")
            st.dataframe(res["reg_agg_frame"], width="stretch")

        st.markdown("#### Plots")
        p1, p2, p3 = st.columns(3)
        for col, name, title in (
            (p1, "confusion_matrix.png", "Confusion matrix"),
            (p2, "feature_importance.png", "Feature importance"),
            (p3, "calibration_curve.png", "Calibration"),
        ):
            path = res["results_dir"] / name
            if path.exists():
                col.image(str(path), caption=title)

        with st.expander("Training log"):
            st.code("\n".join(lines))
    else:
        # Show the most recent saved results, if any.
        results_dir = config.ml_results_dir(league)
        if (results_dir / "classification_single_split.csv").exists():
            st.markdown("#### Last saved results")
            st.dataframe(
                pd.read_csv(results_dir / "classification_single_split.csv"),
                width="stretch")
            st.dataframe(
                pd.read_csv(results_dir / "regression_single_split.csv"),
                width="stretch")
            p1, p2, p3 = st.columns(3)
            for col, name, title in (
                (p1, "confusion_matrix.png", "Confusion matrix"),
                (p2, "feature_importance.png", "Feature importance"),
                (p3, "calibration_curve.png", "Calibration"),
            ):
                path = results_dir / name
                if path.exists():
                    col.image(str(path), caption=title)
        else:
            st.info("No results yet — press **Train models**.")


# ---------------------------------------------------------------------------
# Tab 3 — Predict
# ---------------------------------------------------------------------------

def tab_predict(opts: dict) -> None:
    league = opts["league"]
    st.subheader(f"Predict fixtures — {config.LEAGUES[league]['name']}")
    st.caption(
        "Pulls the scheduled (unplayed) matches of the selected season from "
        "the raw Understat JSON, fills shots/xG with each team's historical "
        "averages, computes leakage-free form features, and runs the saved "
        "pipeline. Fair decimal odds (1 / probability) are shown alongside "
        "the 1X2 probabilities."
    )

    pipeline_path = config.ml_models_dir(league) / "pipeline.joblib"
    if not pipeline_path.exists():
        st.info("No saved pipeline yet — run **Train** first.")
        return

    pipeline = load_pipeline_cached(str(pipeline_path))
    st.caption(
        f"Pipeline: classifier **{pipeline['classifier_name']}**, goal models "
        f"{pipeline['regressor_home_name']} / "
        f"{pipeline['regressor_away_name']} "
        f"(saved {str(pipeline.get('saved_at', '?'))[:16]})."
    )
    if pipeline.get("include_ht"):
        st.warning(
            "This pipeline uses HT goals as features — for pre-match "
            "fixtures those are unknown, so they are imputed as 0 "
            "(no goals scored yet). For the cleanest pre-match "
            "predictions, train without **Include HT goals**."
        )

    if st.button("🔮 Predict upcoming fixtures", type="primary"):
        try:
            # Encode teams with the saved pipeline's own label map so the
            # codes are guaranteed consistent with the trained models.
            fixtures = ml_fixtures.build_fixtures(
                opts["season"], league=league,
                n_matchdays=opts["n_matchdays"],
                labels=pipeline["team_labels"])
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:  # noqa: BLE001
            st.error(f"Fixture build failed: {e}")
            return

        prepared = ml_predict.prepare_fixtures(fixtures, pipeline)
        preds = ml_predict.predict(prepared, pipeline)

        # Decode team codes -> names for display.
        labels: dict[int, str] = pipeline["team_labels"]
        preds = preds.copy()
        preds["home_team"] = preds["home_team"].map(
            lambda c: labels.get(int(c), f"team {c}"))
        preds["away_team"] = preds["away_team"].map(
            lambda c: labels.get(int(c), f"team {c}"))
        preds["date"] = pd.to_datetime(preds["date"]).dt.date

        # Fair decimal betting odds from the model probabilities (odds = 1/p,
        # no bookmaker margin).
        for pcol, ocol in (("p_home", "odds_home"), ("p_draw", "odds_draw"),
                           ("p_away", "odds_away")):
            preds[ocol] = (1.0 / preds[pcol].clip(lower=1e-6)).round(2)
        preds = preds[[
            "date", "home_team", "away_team",
            "p_home", "odds_home", "p_draw", "odds_draw",
            "p_away", "odds_away",
            "outcome", "expected_home_goals", "expected_away_goals",
        ]]

        st.success(f"{len(preds)} fixtures predicted.")
        st.caption(
            "Odds are fair decimal quotes (1 / probability) with no "
            "bookmaker margin."
        )
        st.dataframe(preds, width="stretch")
        st.download_button(
            "Download CSV", preds.to_csv(index=False).encode(),
            file_name=f"predictions_{league}_{opts['season']}.csv",
            mime="text/csv")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    opts = sidebar()
    t1, t2, t3 = st.tabs(["📥 Update data", "🏋️ Train", "🔮 Predict"])
    with t1:
        tab_update(opts)
    with t2:
        tab_train(opts)
    with t3:
        tab_predict(opts)


if __name__ == "__main__":
    main()
