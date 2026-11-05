"""Validate and deduplicate the SMS Spam Collection without changing the raw file."""

import csv
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SMSSpamCollection"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "sms_spam_cleaned.csv"
)
COLUMN_NAMES = ["label", "message"]
VALID_LABELS = {"ham", "spam"}


def load_dataset(dataset_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original headerless, tab-separated dataset without quote parsing."""
    return pd.read_csv(
        dataset_path,
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        dtype={"label": "string", "message": "string"},
        na_values=[""],
    )


def validate_dataset(data: pd.DataFrame) -> None:
    """Raise a clear error if the dataset schema or values are invalid."""
    if data.columns.tolist() != COLUMN_NAMES:
        raise ValueError(f"Expected columns {COLUMN_NAMES}; got {data.columns.tolist()}")
    if data.empty:
        raise ValueError("Dataset must contain at least one record.")
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Dataset contains missing values: {missing[missing > 0].to_dict()}")

    labels = set(data["label"].unique())
    invalid_labels = labels - VALID_LABELS
    if invalid_labels:
        raise ValueError(f"Dataset contains invalid labels: {sorted(invalid_labels)}")


def remove_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each identical row, preserving row order."""
    return data.drop_duplicates(keep="first", ignore_index=True)


def prepare_data(
    source_path: Path = RAW_DATA_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
) -> tuple[int, int]:
    """Validate, exactly deduplicate, and save a processed copy of the dataset."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Processed output path must not overwrite the raw dataset.")

    original = load_dataset(source_path)
    validate_dataset(original)
    prepared = remove_exact_duplicates(original)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")

    duplicate_count = len(original) - len(prepared)
    print(f"Input records: {len(original)}")
    print(f"Exact duplicate records removed: {duplicate_count}")
    print(f"Processed records: {len(prepared)}")
    print("Processed class counts:")
    print(prepared["label"].value_counts().reindex(["ham", "spam"], fill_value=0).to_string())
    print(f"Processed dataset saved to: {output_path}")
    return len(original), len(prepared)


def main() -> None:
    """Prepare the project's processed dataset."""
    prepare_data()


if __name__ == "__main__":
    main()
