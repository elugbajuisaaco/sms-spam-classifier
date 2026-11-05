from pathlib import Path

import pandas as pd
import pytest

from src.split_data import (
    check_no_message_overlap,
    create_split,
    load_dataset,
    save_split,
    split_dataset,
)


def make_balanced_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["ham"] * 20 + ["spam"] * 10,
            "message": [f"ham message {number}" for number in range(20)]
            + [f"spam message {number}" for number in range(10)],
        }
    )


def test_split_has_expected_proportions_and_stratified_classes() -> None:
    train, test = split_dataset(make_balanced_dataset())

    assert len(train) == 24
    assert len(test) == 6
    assert train["label"].value_counts().to_dict() == {"ham": 16, "spam": 8}
    assert test["label"].value_counts().to_dict() == {"ham": 4, "spam": 2}


def test_split_is_reproducible() -> None:
    data = make_balanced_dataset()

    first_train, first_test = split_dataset(data)
    second_train, second_test = split_dataset(data)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_preserves_original_message_text() -> None:
    data = make_balanced_dataset()
    data.loc[0, "message"] = "  Keep punctuation!!! £5 https://example.com  "

    train, test = split_dataset(data)
    combined_messages = set(train["message"]) | set(test["message"])

    assert "  Keep punctuation!!! £5 https://example.com  " in combined_messages


def test_no_identical_message_text_overlaps_between_splits() -> None:
    train, test = split_dataset(make_balanced_dataset())

    assert set(train["message"]).isdisjoint(set(test["message"]))
    check_no_message_overlap(train, test)


def test_overlap_is_reported_instead_of_silently_accepted() -> None:
    train = pd.DataFrame({"label": ["ham", "spam"], "message": ["same", "train"]})
    test = pd.DataFrame({"label": ["ham", "spam"], "message": ["same", "test"]})

    with pytest.raises(ValueError, match="Train/test leakage detected"):
        check_no_message_overlap(train, test)


def test_saved_files_have_expected_columns(tmp_path: Path) -> None:
    train, test = split_dataset(make_balanced_dataset())

    train_path, test_path = save_split(train, test, tmp_path)

    assert train_path.name == "train.csv"
    assert test_path.name == "test.csv"
    assert pd.read_csv(train_path).columns.tolist() == ["label", "message"]
    assert pd.read_csv(test_path).columns.tolist() == ["label", "message"]


def test_create_split_uses_temporary_input_and_writes_outputs(tmp_path: Path) -> None:
    source_path = tmp_path / "input.csv"
    output_directory = tmp_path / "output"
    make_balanced_dataset().to_csv(source_path, index=False)

    train, test = create_split(source_path, output_directory)

    assert (output_directory / "train.csv").is_file()
    assert (output_directory / "test.csv").is_file()
    assert len(train) + len(test) == 30


def test_processed_dataset_is_loadable_and_has_no_overlap() -> None:
    data = load_dataset()

    assert data.shape == (5171, 2)
    train, test = split_dataset(data)
    assert set(train["message"]).isdisjoint(set(test["message"]))
