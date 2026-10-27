"""Time-based and walk-forward splits (leakage-free by construction).

All splits key on the derived ``season`` column and are guarded by a date
assertion: no validation date may be <= the latest training date.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Default single-split cutoff: train on seasons <= 2023-24,
# validate on 2024-25, 2025-26, 2026-27.
DEFAULT_CUTOFF = "2023-24"
# Walk-forward validation seasons (2026-27 excluded: partial season).
WALK_FORWARD_START = "2022-23"
WALK_FORWARD_END = "2025-26"


@dataclass(frozen=True)
class Split:
    name: str
    train: pd.DataFrame
    valid: pd.DataFrame


def _check_leakage(train: pd.DataFrame, valid: pd.DataFrame, name: str) -> None:
    if len(train) == 0 or len(valid) == 0:
        raise ValueError(f"split {name!r}: empty side (train={len(train)}, valid={len(valid)})")
    if valid["date"].min() <= train["date"].max():
        raise ValueError(
            f"split {name!r}: date leakage — earliest valid date "
            f"{valid['date'].min().date()} <= latest train date {train['date'].max().date()}"
        )


def time_split(
    df: pd.DataFrame,
    cutoff: str = DEFAULT_CUTOFF,
) -> Split:
    """Single time-based split: train on seasons <= cutoff, validate after."""
    train = df[df["season"] <= cutoff]
    valid = df[df["season"] > cutoff]
    _check_leakage(train, valid, f"time_split(cutoff={cutoff})")
    return Split(name=f"time_split(cutoff={cutoff})", train=train, valid=valid)


def _seasons_between(start: str, end: str) -> list[str]:
    """Inclusive list of season strings from start to end (both inclusive)."""
    if start > end:
        raise ValueError(f"start season {start} after end season {end}")
    y0 = int(start.split("-")[0])
    y1 = int(end.split("-")[0])
    return [f"{y}-{str(y + 1)[-2:]}" for y in range(y0, y1 + 1)]


def walk_forward_splits(
    df: pd.DataFrame,
    start: str = WALK_FORWARD_START,
    end: str = WALK_FORWARD_END,
) -> list[Split]:
    """Expanding-window walk-forward: one fold per validation season.

    Fold for season S trains on ALL seasons strictly before S and validates
    on S. Seasons outside [start, end] are ignored (2026-27 is partial).
    """
    splits: list[Split] = []
    for val_season in _seasons_between(start, end):
        train = df[df["season"] < val_season]
        valid = df[df["season"] == val_season]
        _check_leakage(train, valid, f"walk_forward(val={val_season})")
        splits.append(Split(name=f"walk_forward(val={val_season})", train=train, valid=valid))
    return splits
