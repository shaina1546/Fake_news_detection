"""
Model Training Module for Fake News Detection System.

Builds a reproducible scikit-learn Pipeline (TF-IDF + Logistic Regression),
performs stratified holdout evaluation, saves performance reports, and exports
the trained model pipeline with joblib.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_loader import load_dataset, LABEL_REAL, LABEL_FAKE, LABEL_NAMES
from src.text_preprocessor import clean_text
from src.evaluation import evaluate_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_FILENAME = "fake_news_model.joblib"
DEFAULT_METADATA_FILENAME = "model_metadata.json"


def build_pipeline(
    max_features: int = 10000,
    random_state: int = 42,
    C: float = 1.0,
) -> Pipeline:
    """
    Constructs the end-to-end Machine Learning Pipeline.

    Components:
    1. TfidfVectorizer:
       - Uses our custom clean_text preprocessor
       - Unigrams and bigrams (ngram_range=(1, 2))
       - English stop words filtered
       - sublinear_tf=True for dampened term frequency scaling
    2. LogisticRegression:
       - Balanced class weighting
       - Fixed random seed for complete reproducibility
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                max_features=max_features,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                C=C,
                class_weight="balanced",
                solver="lbfgs",
            ),
        ),
    ])


def train_and_evaluate(
    data_path: str | Path = "data",
    model_dir: str | Path = "models",
    reports_dir: str | Path = "reports",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 10000,
) -> Dict[str, Any]:
    """
    Orchestrates dataset loading, train-test splitting, pipeline fitting,
    holdout evaluation, and artifact saving.
    """
    logger.info("Starting model training workflow...")

    # 1. Load and validate dataset
    df, data_report = load_dataset(data_path=data_path)
    logger.info(f"Loaded {len(df)} validated samples.")

    # 2. Stratified train-test split
    # Ensures both training and test sets maintain identical class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    logger.info(
        f"Train set size: {len(X_train)} samples | Test set size: {len(X_test)} samples."
    )

    # 3. Construct and fit pipeline
    pipeline = build_pipeline(max_features=max_features, random_state=random_state)
    logger.info("Fitting TF-IDF Vectorizer and Logistic Regression Pipeline...")
    pipeline.fit(X_train, y_train)

    # 4. Evaluate exclusively on holdout test set (no data leakage)
    logger.info("Evaluating pipeline on holdout test data...")
    y_pred = pipeline.predict(X_test)
    eval_results = evaluate_predictions(
        y_true=y_test,
        y_pred=y_pred,
        output_dir=reports_dir,
        save_artifacts=True,
    )

    # 5. Save model artifacts only after successful verification
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model_filepath = model_dir_path / DEFAULT_MODEL_FILENAME

    logger.info(f"Saving trained model pipeline to {model_filepath}...")
    joblib.dump(pipeline, model_filepath)

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_file": DEFAULT_MODEL_FILENAME,
        "pipeline_stages": ["TfidfVectorizer", "LogisticRegression"],
        "hyperparameters": {
            "max_features": max_features,
            "ngram_range": [1, 2],
            "random_state": random_state,
            "test_size": test_size,
        },
        "dataset_summary": data_report.to_dict(),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": eval_results["metrics"],
        "labels": {
            str(LABEL_REAL): LABEL_NAMES[LABEL_REAL],
            str(LABEL_FAKE): LABEL_NAMES[LABEL_FAKE],
        },
    }

    metadata_path = model_dir_path / DEFAULT_METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to {metadata_path}")

    # Log summary to console
    print("\n" + "=" * 60)
    print("[SUCCESS] TRAINING AND EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total Dataset Samples: {data_report.total_samples:,}")
    print(f"Train / Test Split:    {len(X_train):,} / {len(X_test):,} (stratified)")
    print(f"Accuracy:              {eval_results['metrics']['accuracy'] * 100:.2f}%")
    print(f"Precision (Fake):      {eval_results['metrics']['precision_fake'] * 100:.2f}%")
    print(f"Recall (Fake):         {eval_results['metrics']['recall_fake'] * 100:.2f}%")
    print(f"F1-Score (Fake):       {eval_results['metrics']['f1_score_fake'] * 100:.2f}%")
    print(f"Model saved to:        {model_filepath}")
    print(f"Reports saved to:      {reports_dir}/")
    print("=" * 60 + "\n")

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Fake News Detection Logistic Regression Pipeline."
    )
    parser.add_argument(
        "--data-path",
        default="data",
        help="Path to folder containing Fake.csv/True.csv or a single CSV file (default: data).",
    )
    parser.add_argument(
        "--model-dir",
        default="models",
        help="Directory where the trained pipeline will be saved (default: models).",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory where metrics and plots will be saved (default: reports).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum vocabulary features for TF-IDF (default: 10000).",
    )

    args = parser.parse_args()

    try:
        train_and_evaluate(
            data_path=args.data_path,
            model_dir=args.model_dir,
            reports_dir=args.reports_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            max_features=args.max_features,
        )
    except FileNotFoundError as fnf_err:
        logger.error(f"\n[DATASET ERROR]:\n{fnf_err}")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"\n[TRAINING FAILED]: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
