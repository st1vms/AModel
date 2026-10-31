"""Export backtest reports and markdown tables for Reddit/README presentation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.ml.backtest import (
    build_odds_summary,
    format_odds_summary_markdown,
)


def export_odds_summary_markdown(
    summary: pd.DataFrame,
    out_path: str | Path | None = None,
    league: str | None = None,
) -> str:
    """Format an odds summary frame into markdown and optionally write to file."""
    md = format_odds_summary_markdown(summary, league=league)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    return md


def print_odds_summary_markdown(
    summary: pd.DataFrame,
    league: str | None = None,
) -> None:
    """Print the markdown odds summary table to stdout."""
    print(format_odds_summary_markdown(summary, league=league))
