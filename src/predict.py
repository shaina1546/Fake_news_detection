"""
Prediction Module for Fake News Detection System.

Encapsulates model inference, input validation, probability extraction,
and ethical disclaimers, separated cleanly from user interface code.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
from sklearn.pipeline import Pipeline

from src.data_loader import LABEL_REAL, LABEL_FAKE, LABEL_NAMES
from src.text_preprocessor import clean_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models") / "fake_news_model.joblib"

ETHICAL_DISCLAIMER = (
    "NOTICE: This assessment is an automated statistical prediction based on stylistic, "
    "lexical, and structural patterns present in the training dataset. It does NOT constitute "
    "independent investigative journalism or definitive factual verification of truth."
)


@dataclass
class PredictionResult:
    """Structured result returned by the prediction pipeline."""
    predicted_label: int
    label_name: str
    probabilities: Dict[str, float]
    raw_text: str
    cleaned_text: str
    model_source: str
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_label": self.predicted_label,
            "label_name": self.label_name,
            "probabilities": self.probabilities,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "model_source": self.model_source,
            "disclaimer": self.disclaimer,
        }


def load_model_pipeline(model_path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """
    Safely loads the trained scikit-learn pipeline from a local file.

    Raises FileNotFoundError if the trained model does not exist.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model artifact not found at '{path.resolve()}'.\n"
            f"Please ensure the model has been trained by executing:\n"
            f"    python -m src.train_model"
        )

    try:
        pipeline = joblib.load(path)
        if not hasattr(pipeline, "predict"):
            raise TypeError(f"Loaded object from '{path}' is not a valid scikit-learn model.")
        return pipeline
    except Exception as exc:
        raise RuntimeError(f"Error loading model from '{path}': {exc}") from exc


def predict_news(
    text: str,
    model: Optional[Pipeline] = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> PredictionResult:
    """
    Predicts whether a given news headline or article resembles Real or Fake news.

    Parameters:
    -----------
    text : str
        Input news headline or article content.
    model : Optional[Pipeline], default=None
        Preloaded model instance (if provided, avoids re-reading disk).
    model_path : str | Path, default=DEFAULT_MODEL_PATH
        Path to load model from if 'model' is None.

    Returns:
    --------
    PredictionResult
        Structured classification result with probabilities and disclaimer.
    """
    if text is None or not isinstance(text, str) or not text.strip():
        raise ValueError("Input text cannot be empty. Please provide a news headline or article body.")

    # Validate or load model
    pipeline = model if model is not None else load_model_pipeline(model_path)
    resolved_source = str(Path(model_path).resolve()) if model is None else "In-Memory Pipeline"

    # Clean input text
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("Input text contained only whitespace or non-printable characters.")

    # Model inference
    # Note: Pipeline's TfidfVectorizer preprocessor will also run clean_text,
    # ensuring identical feature representation.
    prediction = int(pipeline.predict([cleaned])[0])
    label_name = LABEL_NAMES.get(prediction, f"Class {prediction}")

    # Estimate class probabilities if supported
    probabilities: Dict[str, float] = {}
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba([cleaned])[0]
            # Mapping class indexes to class labels
            classes = list(getattr(pipeline, "classes_", [LABEL_REAL, LABEL_FAKE]))
            for idx, cls_val in enumerate(classes):
                c_name = LABEL_NAMES.get(int(cls_val), f"Class {cls_val}")
                probabilities[c_name] = round(float(probs[idx]), 4)
        except Exception as e:
            logger.warning(f"Could not compute prediction probabilities: {e}")

    return PredictionResult(
        predicted_label=prediction,
        label_name=label_name,
        probabilities=probabilities,
        raw_text=text,
        cleaned_text=cleaned,
        model_source=resolved_source,
        disclaimer=ETHICAL_DISCLAIMER,
    )
