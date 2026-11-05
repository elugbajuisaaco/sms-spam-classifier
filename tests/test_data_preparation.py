import hashlib
from pathlib import Path

import pandas as pd

from src.prepare_data import (
    RAW_DATA_PATH,
    load_dataset,
    prepare_data,
    remove_exact_duplicates,
    validate_dataset,
)


def test_dataset_loading_returns_expected_schema() -> None:
    data = load_dataset()

    assert data.columns.tolist() == ["label", "message"]
    assert data.shape == (5574, 2)


def test_original_dataset_has_expected_valid_labels() -> None:
    data = load_dataset()

    assert set(data["label"].unique()) == {"ham", "spam"}
    assert data["label"].value_counts().to_dict() == {"ham": 4827, "spam": 747}
    validate_dataset(data)


def test_exact_duplicate_removal_keeps_first_occurrence() -> None:
    data = pd.DataFrame(
        {
            "label": ["ham", "spam", "ham"],
            "message": ["repeated text", "repeated text", "repeated text"],
        }
    )

    result = remove_exact_duplicates(data)

    assert result.to_dict("records") == [
        {"label": "ham", "message": "repeated text"},
        {"label": "spam", "message": "repeated text"},
    ]
    assert len(data) == 3


def test_original_dataset_is_not_modified(tmp_path: Path) -> None:
    before = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()

    prepare_data(output_path=tmp_path / "prepared.csv")

    after = hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest()
    assert after == before


def test_processed_dataset_is_created_successfully(tmp_path: Path) -> None:
    output_path = tmp_path / "processed" / "sms_spam_cleaned.csv"

    before_count, after_count = prepare_data(output_path=output_path)

    assert output_path.is_file()
    assert (before_count, after_count) == (5574, 5171)
    prepared = pd.read_csv(output_path, encoding="utf-8")
    assert prepared.columns.tolist() == ["label", "message"]
    assert prepared.shape == (5171, 2)
    assert prepared["label"].value_counts().to_dict() == {"ham": 4518, "spam": 653}
    expected = remove_exact_duplicates(load_dataset())
    pd.testing.assert_frame_equal(prepared, expected, check_dtype=False)
