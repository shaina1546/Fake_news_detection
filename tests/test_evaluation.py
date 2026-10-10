"""
Unit tests for model evaluation and metrics generation.
"""

from pathlib import Path
import pytest
from src.evaluation import evaluate_predictions, load_latest_metrics


def test_evaluate_predictions_calculation(tmp_path: Path):
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]  # 1 real-as-real, 1 real-as-fake, 1 fake-as-real, 1 fake-as-fake

    out_dir = tmp_path / "reports"
    report = evaluate_predictions(y_true, y_pred, output_dir=out_dir, save_artifacts=True)

    # 2 correct out of 4 -> 50% accuracy
    assert report["metrics"]["accuracy"] == 0.5
    # Fake class: TP = 1, FP = 1 -> Precision = 1 / 2 = 0.5
    assert report["metrics"]["precision_fake"] == 0.5
    # Fake class: TP = 1, FN = 1 -> Recall = 1 / 2 = 0.5
    assert report["metrics"]["recall_fake"] == 0.5

    assert (out_dir / "metrics_summary.json").exists()
    assert (out_dir / "confusion_matrix.png").exists()

    loaded = load_latest_metrics(reports_dir=out_dir)
    assert loaded is not None
    assert loaded["metrics"]["accuracy"] == 0.5


def test_evaluate_predictions_empty():
    with pytest.raises(ValueError, match="Cannot evaluate empty predictions"):
        evaluate_predictions([], [])


def test_load_latest_metrics_missing(tmp_path: Path):
    empty_dir = tmp_path / "empty_reports"
    empty_dir.mkdir()
    assert load_latest_metrics(reports_dir=empty_dir) is None
