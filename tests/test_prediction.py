"""
Unit tests for the predict module.
"""

from pathlib import Path
import pytest
from sklearn.pipeline import Pipeline

from src.train_model import build_pipeline
from src.predict import (
    predict_news,
    load_model_pipeline,
    PredictionResult,
    ETHICAL_DISCLAIMER,
)


@pytest.fixture
def trained_test_pipeline() -> Pipeline:
    """Creates and fits a fast in-memory pipeline on synthetic strings."""
    texts = [
        "Official government economic report indicates stable growth and rising employment.",
        "Senate passes bilateral infrastructure and road construction act.",
        "Scientists publish peer reviewed study confirming atmospheric trends.",
        "SHOCKING BOMBSHELL: Aliens confirmed controlling high-ranking officials in secret bunker!",
        "UNBELIEVABLE: Miracle cure discovered that big pharma is hiding from you!",
        "EXCLUSIVE: Secret plot uncovered to secretly replace water supply with chemicals!",
    ]
    labels = [0, 0, 0, 1, 1, 1]  # 0: Real, 1: Fake
    pipeline = build_pipeline(max_features=50, random_state=42)
    pipeline.fit(texts, labels)
    return pipeline


def test_predict_news_valid_synthetic_model(trained_test_pipeline: Pipeline):
    text = "Exclusive bombshell revelation about alien bunker operations!"
    res = predict_news(text, model=trained_test_pipeline)

    assert isinstance(res, PredictionResult)
    assert res.predicted_label in (0, 1)
    assert res.label_name in ("Real", "Fake")
    assert "Real" in res.probabilities
    assert "Fake" in res.probabilities
    assert res.disclaimer == ETHICAL_DISCLAIMER
    assert res.cleaned_text == "exclusive bombshell revelation about alien bunker operations!"


def test_predict_news_empty_text(trained_test_pipeline: Pipeline):
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict_news("", model=trained_test_pipeline)

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict_news("    \n\t   ", model=trained_test_pipeline)


def test_predict_news_none_input(trained_test_pipeline: Pipeline):
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        predict_news(None, model=trained_test_pipeline)


def test_predict_news_missing_model_file():
    with pytest.raises(FileNotFoundError, match="Trained model artifact not found"):
        predict_news("Some news headline", model_path="models/nonexistent_model_file.joblib")


def test_load_model_pipeline_missing():
    with pytest.raises(FileNotFoundError):
        load_model_pipeline("invalid/path/to/model.joblib")
