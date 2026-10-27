"""Plots: confusion matrix, feature importance, calibration curve.

All plots are saved as PNG (150 dpi) to the results directory.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve

from src.ml.data import OUTCOME_NAMES

sns.set_theme(style="whitegrid")


def plot_confusion_matrix(
    cm: np.ndarray,
    path: Path,
    title: str = "Confusion matrix (1X2)",
) -> Path:
    """Heatmap of a 3x3 confusion matrix (rows = true, cols = predicted)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["H", "D", "A"], yticklabels=["H", "D", "A"])
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _importances(model, feature_names: list[str]) -> np.ndarray | None:
    """Feature importances for the supported model types (else None)."""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "get_feature_importance"):
        imp = np.asarray(model.get_feature_importance(), dtype=float)
    else:
        return None
    if imp.shape[0] != len(feature_names):
        return None
    return imp


def plot_feature_importance(
    model,
    feature_names: list[str],
    path: Path,
    top_n: int = 15,
    title: str = "Feature importance",
) -> Path:
    """Horizontal bar chart of the top-N feature importances."""
    imp = _importances(model, feature_names)
    if imp is None:
        raise TypeError(f"no importances available for {type(model)!r}")
    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    values = imp[order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * top_n)))
    # hue=y + legend=False: seaborn 0.13 deprecates `palette` without `hue`.
    sns.barplot(x=values, y=names, hue=names, ax=ax, palette="viridis",
                legend=False)
    ax.set(xlabel="Importance", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    path: Path,
    title: str = "Calibration curve (1X2)",
) -> Path:
    """Reliability diagram: predicted probability vs observed frequency."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    for c, color in zip([0, 1, 2], ["tab:blue", "tab:green", "tab:red"]):
        # calibration_curve is binary-only: binarize y_true for class c.
        frac, mean = calibration_curve((y_true == c).astype(int), y_prob[:, c],
                                       pos_label=1, n_bins=10)
        ax.plot(frac, mean, marker="o", ms=4, color=color,
                label=OUTCOME_NAMES[c])
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency",
           title=title, ylim=(-0.02, 1.02))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
