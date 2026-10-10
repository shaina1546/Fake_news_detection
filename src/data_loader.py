"""
Dataset Loading and Validation Module.

Provides robust loading, validation, label normalization, and data-leakage
detection for the Fake News Detection System.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard Binary Class Definitions:
# 0 -> Real / Reliable News
# 1 -> Fake / Unreliable News
LABEL_REAL = 0
LABEL_FAKE = 1
LABEL_NAMES = {LABEL_REAL: "Real", LABEL_FAKE: "Fake"}

LABEL_MAPPING = {
    0: LABEL_REAL,
    "0": LABEL_REAL,
    "real": LABEL_REAL,
    "true": LABEL_REAL,
    "reliable": LABEL_REAL,
    1: LABEL_FAKE,
    "1": LABEL_FAKE,
    "fake": LABEL_FAKE,
    "false": LABEL_FAKE,
    "unreliable": LABEL_FAKE,
}


@dataclass
class DatasetReport:
    """Diagnostic report generated during dataset loading."""
    total_samples: int
    real_count: int
    fake_count: int
    missing_texts_dropped: int
    duplicate_articles_removed: int
    cross_class_duplicates_dropped: int
    leakage_warning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "real_count": self.real_count,
            "fake_count": self.fake_count,
            "missing_texts_dropped": self.missing_texts_dropped,
            "duplicate_articles_removed": self.duplicate_articles_removed,
            "cross_class_duplicates_dropped": self.cross_class_duplicates_dropped,
            "leakage_warning": self.leakage_warning,
        }


def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Finds a matching column name case-insensitively."""
    lower_cols = {col.lower().strip(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def normalize_label(label: Any) -> int:
    """
    Normalizes a dataset label into binary 0 (Real) or 1 (Fake).

    Raises ValueError if label cannot be mapped.
    """
    if pd.isna(label):
        raise ValueError("Encountered NaN or missing label value.")

    if isinstance(label, str):
        cleaned = label.strip().lower()
        if cleaned in LABEL_MAPPING:
            return LABEL_MAPPING[cleaned]
    elif label in LABEL_MAPPING:
        return LABEL_MAPPING[label]

    raise ValueError(
        f"Invalid label value '{label}'. Expected labels representing Real (0, 'real', 'true') "
        f"or Fake (1, 'fake', 'false')."
    )


def load_dataset(
    data_path: str | Path = "data",
    min_samples: int = 10,
    combine_title_and_text: bool = True,
) -> Tuple[pd.DataFrame, DatasetReport]:
    """
    Loads, cleans, validates, and normalizes news dataset.

    Supports:
    1. Directory with Kaggle's 'Fake.csv' and 'True.csv' files.
    2. Single unified CSV file containing text and label columns.

    Parameters:
    -----------
    data_path : str | Path
        Path to data folder or a single CSV file.
    min_samples : int, default=10
        Minimum acceptable labeled examples.
    combine_title_and_text : bool, default=True
        Whether to concatenate headline and body when both are available.

    Returns:
    --------
    Tuple[pd.DataFrame, DatasetReport]
        - Cleaned DataFrame with ['text', 'label'] columns.
        - DatasetReport with diagnostic statistics.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset path not found: '{path.resolve()}'.\n"
            f"Please download the 'Fake and Real News Dataset' from Kaggle:\n"
            f"https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset\n"
            f"and place 'Fake.csv' and 'True.csv' inside the 'data/' directory."
        )

    # 1. Determine files to load
    raw_df: pd.DataFrame
    if path.is_dir():
        # Look for Fake.csv and True.csv (case-insensitive)
        files = {f.name.lower(): f for f in path.glob("*.csv")}
        has_fake = "fake.csv" in files
        has_true = "true.csv" in files

        if has_fake and has_true:
            logger.info("Loading Kaggle two-file format: Fake.csv and True.csv")
            fake_df = pd.read_csv(files["fake.csv"])
            true_df = pd.read_csv(files["true.csv"])

            fake_df["label"] = LABEL_FAKE
            true_df["label"] = LABEL_REAL

            raw_df = pd.concat([fake_df, true_df], ignore_index=True)
        else:
            # Check for any single CSV file in the directory
            csv_files = list(path.glob("*.csv"))
            if len(csv_files) == 1:
                logger.info(f"Loading single CSV file found in directory: {csv_files[0].name}")
                raw_df = pd.read_csv(csv_files[0])
            else:
                raise FileNotFoundError(
                    f"Directory '{path.resolve()}' must contain either both 'Fake.csv' and 'True.csv' "
                    f"or a single dataset CSV file. Found: {[f.name for f in csv_files]}"
                )
    else:
        # Direct CSV file path
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, but got '{path.name}'.")
        raw_df = pd.read_csv(path)

    if raw_df.empty:
        raise ValueError("Loaded dataset is empty (0 rows).")

    # 2. Extract and format article text
    title_col = _find_column(raw_df, ["title", "headline"])
    text_col = _find_column(raw_df, ["text", "article", "content", "body", "statement"])

    if not text_col and not title_col:
        raise ValueError(
            f"Dataset missing required text columns. Columns available: {list(raw_df.columns)}."
        )

    if combine_title_and_text and title_col and text_col:
        # Concatenate title and text for stronger feature representation
        extracted_text = (
            raw_df[title_col].fillna("").astype(str).str.strip()
            + " "
            + raw_df[text_col].fillna("").astype(str).str.strip()
        )
    elif text_col:
        extracted_text = raw_df[text_col].fillna("").astype(str).str.strip()
    else:
        extracted_text = raw_df[title_col].fillna("").astype(str).str.strip()

    # 3. Extract and normalize labels
    label_col = _find_column(raw_df, ["label", "class", "target", "category"])
    if not label_col:
        raise ValueError(
            f"Dataset missing a label column. Available columns: {list(raw_df.columns)}."
        )

    processed_df = pd.DataFrame({
        "text": extracted_text,
        "raw_label": raw_df[label_col],
    })

    # Drop empty or whitespace-only texts
    initial_count = len(processed_df)
    non_empty_mask = processed_df["text"].str.strip() != ""
    processed_df = processed_df[non_empty_mask].copy()
    missing_texts_dropped = initial_count - len(processed_df)

    # Normalize labels safely
    try:
        processed_df["label"] = processed_df["raw_label"].apply(normalize_label)
    except Exception as exc:
        raise ValueError(f"Label normalization error: {exc}") from exc

    processed_df = processed_df[["text", "label"]].copy()

    # 4. Check for within-class exact duplicates
    before_dedup = len(processed_df)
    processed_df = processed_df.drop_duplicates(subset=["text", "label"]).copy()
    duplicate_articles_removed = before_dedup - len(processed_df)

    # 5. Check for Cross-Class duplicates (potential data leakage & conflicting labels)
    text_counts_per_class = processed_df.groupby("text")["label"].nunique()
    conflicting_texts = text_counts_per_class[text_counts_per_class > 1].index

    cross_class_duplicates_dropped = len(conflicting_texts)
    leakage_warning = None
    if cross_class_duplicates_dropped > 0:
        leakage_warning = (
            f"CRITICAL DATA LEAKAGE WARNING: {cross_class_duplicates_dropped} unique articles "
            f"were found labeled as BOTH Real and Fake! Dropping conflicting articles to prevent "
            f"label corruption."
        )
        logger.warning(leakage_warning)
        processed_df = processed_df[~processed_df["text"].isin(conflicting_texts)].copy()

    # 6. Validate class counts and minimum threshold
    real_count = int((processed_df["label"] == LABEL_REAL).sum())
    fake_count = int((processed_df["label"] == LABEL_FAKE).sum())
    total_valid = len(processed_df)

    if total_valid < min_samples:
        raise ValueError(
            f"Dataset contains only {total_valid} valid examples, below the minimum required "
            f"threshold of {min_samples}."
        )

    if real_count == 0 or fake_count == 0:
        raise ValueError(
            f"Dataset must contain both classes. Found {real_count} Real and {fake_count} Fake examples."
        )

    report = DatasetReport(
        total_samples=total_valid,
        real_count=real_count,
        fake_count=fake_count,
        missing_texts_dropped=missing_texts_dropped,
        duplicate_articles_removed=duplicate_articles_removed,
        cross_class_duplicates_dropped=cross_class_duplicates_dropped,
        leakage_warning=leakage_warning,
    )

    logger.info(
        f"Dataset loaded successfully: {total_valid} total samples "
        f"({real_count} Real, {fake_count} Fake)."
    )

    return processed_df, report
