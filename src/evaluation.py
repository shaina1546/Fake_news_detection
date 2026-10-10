"""
Model Evaluation and Metrics Visualization Module.

Generates reproducible evaluation metrics, confusion matrix visualizations,
and performance reports.
"""

from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Sequence

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for servers and CLI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data_loader import LABEL_REAL, LABEL_FAKE, LABEL_NAMES

logger = logging.getLogger(__name__)


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    output_dir: str | Path = "reports",
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Computes classification evaluation metrics and saves summary reports.

    Positive Class (1): Fake / Unreliable
    Negative Class (0): Real / Reliable

    Parameters:
    -----------
    y_true : Sequence[int]
        Ground-truth labels (0 or 1).
    y_pred : Sequence[int]
        Predicted labels (0 or 1).
    output_dir : str | Path, default="reports"
        Directory to save metrics JSON and confusion matrix plot.
    save_artifacts : bool, default=True
        Whether to write metrics.json and confusion_matrix.png to disk.

    Returns:
    --------
    Dict[str, Any]
        Dictionary of computed metrics and distribution statistics.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    total_samples = len(y_true_arr)
    if total_samples == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    # Core Performance Metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=LABEL_FAKE, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=LABEL_FAKE, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=LABEL_FAKE, zero_division=0))

    # Confusion Matrix:
    # rows: true [Real (0), Fake (1)], columns: pred [Real (0), Fake (1)]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[LABEL_REAL, LABEL_FAKE])
    tn, fp, fn, tp = cm.ravel()

    # Class distributions in evaluation slice
    real_true_count = int((y_true_arr == LABEL_REAL).sum())
    fake_true_count = int((y_true_arr == LABEL_FAKE).sum())

    report_data: Dict[str, Any] = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "positive_class": "Fake (1)",
        "negative_class": "Real (0)",
        "total_test_samples": total_samples,
        "class_distribution": {
            "real_samples": real_true_count,
            "fake_samples": fake_true_count,
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_fake": round(prec, 4),
            "recall_fake": round(rec, 4),
            "f1_score_fake": round(f1, 4),
        },
        "confusion_matrix": {
            "true_negatives_real_as_real": int(tn),
            "false_positives_real_as_fake": int(fp),
            "false_negatives_fake_as_real": int(fn),
            "true_positives_fake_as_fake": int(tp),
        },
        "notes": (
            "Evaluation on holdout test set from training dataset. "
            "Offline benchmark performance does not guarantee accuracy on breaking news, "
            "cross-domain articles, or out-of-distribution real-world reporting."
        ),
    }

    if save_artifacts:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save JSON metrics report
        metrics_file = out_path / "metrics_summary.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Evaluation metrics saved to {metrics_file}")

        # 2. Save Confusion Matrix visualization
        cm_image_path = out_path / "confusion_matrix.png"
        plot_confusion_matrix(cm, save_path=cm_image_path)
        logger.info(f"Confusion matrix plot saved to {cm_image_path}")

    return report_data


def plot_confusion_matrix(cm: np.ndarray, save_path: str | Path) -> None:
    """
    Renders and saves a clean, styled confusion matrix figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

    fig.colorbar(cax, fraction=0.046, pad=0.04)

    classes = [LABEL_NAMES[LABEL_REAL], LABEL_NAMES[LABEL_FAKE]]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")

    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, labelpad=10)
    ax.set_title("Holdout Test Confusion Matrix", fontsize=12, pad=15, fontweight="bold")

    # Annotate counts and percentages inside each cell
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total * 100) if total > 0 else 0
            color = "white" if count > (total * 0.4) else "#1A202C"
            ax.text(
                j,
                i,
                f"{count:,}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
                fontweight="bold",
            )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def load_latest_metrics(reports_dir: str | Path = "reports") -> Optional[Dict[str, Any]]:
    """Loads metrics_summary.json if it exists, otherwise returns None."""
    path = Path(reports_dir) / "metrics_summary.json"
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read metrics summary file: {e}")
            return None
    return None
