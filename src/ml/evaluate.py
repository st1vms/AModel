"""Evaluation metrics & result aggregation.

Classification: log loss, ROC-AUC (macro one-vs-rest + per-class),
accuracy, classification report, confusion matrix.
Regression: MAE and RMSE per goal target.

Results are plain dicts so they can be aggregated across walk-forward
folds (mean +/- std) and dumped to CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from src.ml.data import OUTCOME_NAMES

LABELS = [0, 1, 2]


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray | None = None,
) -> dict:
    """All classification metrics for a 1X2 prediction."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)

    # Per-class one-vs-rest ROC-AUC (guard: a class absent from y_true
    # yields a degenerate curve -> NaN, reported as such).
    per_class = {}
    for c in LABELS:
        try:
            # Binary one-vs-rest AUC for class c (no `average` arg: y is binary).
            per_class[OUTCOME_NAMES[c]] = float(
                roc_auc_score((y_true == c).astype(int), y_prob[:, c])
            )
        except ValueError:
            per_class[OUTCOME_NAMES[c]] = float("nan")

    try:
        macro_auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr",
                                        average="macro", labels=LABELS))
    except ValueError:
        macro_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=LABELS)),
        "roc_auc_macro": macro_auc,
        "roc_auc_H": per_class["H"],
        "roc_auc_D": per_class["D"],
        "roc_auc_A": per_class["A"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": cm,
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE and RMSE for a goal target."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def aggregate_folds(rows: list[dict]) -> dict:
    """Aggregate per-fold metric rows into mean +/- std.

    ``rows``: list of dicts with a common set of numeric metric keys.
    Returns {metric: {"mean": x, "std": y}}.
    """
    if not rows:
        return {}
    out: dict[str, dict] = {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))
            and not isinstance(rows[0][k], bool)]
    for k in keys:
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))
                and not isinstance(r.get(k), bool)]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    """Flatten result dicts into a tidy DataFrame (one row per entry).

    Each result dict may carry a ``confusion_matrix`` (stored as a
    space-separated string) so the frame stays CSV-friendly.
    """
    rows = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "confusion_matrix"}
        if "confusion_matrix" in r:
            row["confusion_matrix"] = " ".join(str(v) for v in
                                               r["confusion_matrix"].ravel())
        rows.append(row)
    return pd.DataFrame(rows)


def table_to_string(frame: pd.DataFrame) -> str:
    """Format a results frame as a fixed-width string (no printing)."""
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda v: f"{v:.4f}"):
        return frame.to_string(index=False)


def print_table(frame: pd.DataFrame) -> None:
    """Pretty-print a results frame to stdout."""
    print(table_to_string(frame))
