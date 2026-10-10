"""
Unit tests for data_loader module using small synthetic fixtures.

Note: These fixtures are strictly synthetic test cases and do not contain
or represent real-world dataset articles.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.data_loader import (
    load_dataset,
    normalize_label,
    LABEL_REAL,
    LABEL_FAKE,
)


@pytest.fixture
def synthetic_kaggle_folder(tmp_path: Path) -> Path:
    """Creates a temporary synthetic Kaggle directory with Fake.csv and True.csv."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    fake_data = pd.DataFrame({
        "title": [f"Synthetic Fake Title {i}" for i in range(10)],
        "text": [f"Synthetic sensational text content {i}" for i in range(10)],
        "subject": ["News"] * 10,
        "date": ["January 1, 2026"] * 10,
    })
    true_data = pd.DataFrame({
        "title": [f"Synthetic True Title {i}" for i in range(10)],
        "text": [f"Synthetic factual press release content {i}" for i in range(10)],
        "subject": ["Politics"] * 10,
        "date": ["January 1, 2026"] * 10,
    })

    fake_data.to_csv(data_dir / "Fake.csv", index=False)
    true_data.to_csv(data_dir / "True.csv", index=False)
    return data_dir


@pytest.fixture
def synthetic_single_csv(tmp_path: Path) -> Path:
    """Creates a temporary single CSV file with text and label."""
    csv_file = tmp_path / "news_sample.csv"
    data = pd.DataFrame({
        "text": [f"Synthetic sample article {i}" for i in range(12)],
        "label": [0, 1] * 6,
    })
    data.to_csv(csv_file, index=False)
    return csv_file


def test_normalize_label_valid_cases():
    assert normalize_label(0) == LABEL_REAL
    assert normalize_label("0") == LABEL_REAL
    assert normalize_label("real") == LABEL_REAL
    assert normalize_label("True") == LABEL_REAL
    assert normalize_label(1) == LABEL_FAKE
    assert normalize_label("1") == LABEL_FAKE
    assert normalize_label("fake") == LABEL_FAKE
    assert normalize_label("False") == LABEL_FAKE


def test_normalize_label_invalid_cases():
    with pytest.raises(ValueError):
        normalize_label("unknown_label")

    with pytest.raises(ValueError):
        normalize_label(None)

    with pytest.raises(ValueError):
        normalize_label(99)


def test_load_dataset_from_kaggle_folder(synthetic_kaggle_folder: Path):
    df, report = load_dataset(data_path=synthetic_kaggle_folder, min_samples=5)
    assert len(df) == 20
    assert report.total_samples == 20
    assert report.real_count == 10
    assert report.fake_count == 10
    assert set(df.columns) == {"text", "label"}
    assert report.leakage_warning is None


def test_load_dataset_from_single_csv(synthetic_single_csv: Path):
    df, report = load_dataset(data_path=synthetic_single_csv, min_samples=5)
    assert len(df) == 12
    assert report.real_count == 6
    assert report.fake_count == 6


def test_load_dataset_missing_path():
    with pytest.raises(FileNotFoundError):
        load_dataset(data_path="nonexistent_folder_path_12345")


def test_load_dataset_insufficient_samples(tmp_path: Path):
    tiny_csv = tmp_path / "tiny.csv"
    pd.DataFrame({"text": ["Article 1", "Article 2"], "label": [0, 1]}).to_csv(tiny_csv, index=False)
    with pytest.raises(ValueError, match="below the minimum required threshold"):
        load_dataset(data_path=tiny_csv, min_samples=10)


def test_load_dataset_data_leakage_detection(tmp_path: Path):
    """Verifies that identical articles labeled under conflicting classes are detected & dropped."""
    leak_csv = tmp_path / "leak.csv"
    data = pd.DataFrame({
        "text": [
            "Identical duplicate article across both classes",
            "Identical duplicate article across both classes",
            "Valid Real Article A",
            "Valid Real Article B",
            "Valid Fake Article C",
            "Valid Fake Article D",
        ],
        "label": [0, 1, 0, 0, 1, 1],
    })
    data.to_csv(leak_csv, index=False)
    df, report = load_dataset(data_path=leak_csv, min_samples=2)

    assert report.cross_class_duplicates_dropped == 1
    assert report.leakage_warning is not None
    assert "Identical duplicate article across both classes" not in df["text"].values
