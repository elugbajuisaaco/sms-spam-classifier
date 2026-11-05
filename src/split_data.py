"""Create a reproducible, stratified train/test split without fitting transforms."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "sms_spam_cleaned.csv"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "processed"
COLUMN_NAMES = ["label", "message"]
VALID_LABELS = {"ham", "spam"}
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_dataset(dataset_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the prepared CSV while preserving text values such as literal 'NA'."""
    data = pd.read_csv(
        dataset_path,
        encoding="utf-8",
        dtype={"label": "string", "message": "string"},
        keep_default_na=False,
    )
    validate_dataset(data)
    return data


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate required columns, labels, and message values before splitting."""
    if data.columns.tolist() != COLUMN_NAMES:
        raise ValueError(f"Expected columns {COLUMN_NAMES}; got {data.columns.tolist()}")
    if data.empty:
        raise ValueError("Dataset must contain at least one record.")
    if data.isna().any().any():
        missing = data.isna().sum()
        raise ValueError(f"Dataset contains missing values: {missing[missing > 0].to_dict()}")

    invalid_labels = set(data["label"].unique()) - VALID_LABELS
    if invalid_labels:
        raise ValueError(f"Dataset contains invalid labels: {sorted(invalid_labels)}")
    if not data["message"].map(lambda message: isinstance(message, str)).all():
        raise TypeError("Every message must be a string.")

    label_counts = data["label"].value_counts()
    if len(label_counts) != len(VALID_LABELS) or (label_counts < 2).any():
        raise ValueError("Each class must contain at least two records for stratification.")


def check_no_message_overlap(
    train: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Raise if an identical original message occurs in both subsets."""
    overlap = set(train["message"]) & set(test["message"])
    if overlap:
        raise ValueError(
            f"Train/test leakage detected: {len(overlap)} identical message "
            "text(s) appear in both subsets."
        )


def split_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a deterministic 80/20 stratified split of original messages."""
    validate_dataset(data)
    features = data["message"]
    labels = data["label"]
    train_messages, test_messages, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    train = pd.DataFrame(
        {"label": train_labels, "message": train_messages},
        columns=COLUMN_NAMES,
    ).reset_index(drop=True)
    test = pd.DataFrame(
        {"label": test_labels, "message": test_messages},
        columns=COLUMN_NAMES,
    ).reset_index(drop=True)
    check_no_message_overlap(train, test)
    return train, test


def save_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> tuple[Path, Path]:
    """Write validated split files after confirming there is no text overlap."""
    validate_dataset(train)
    validate_dataset(test)
    check_no_message_overlap(train, test)

    output_directory = Path(output_directory)
    train_path = output_directory / "train.csv"
    test_path = output_directory / "test.csv"
    output_directory.mkdir(parents=True, exist_ok=True)
    train.to_csv(train_path, index=False, encoding="utf-8", lineterminator="\n")
    test.to_csv(test_path, index=False, encoding="utf-8", lineterminator="\n")
    return train_path, test_path


def create_split(
    dataset_path: Path = DATA_PATH,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, split, validate, and save data without fitting any transformation."""
    data = load_dataset(dataset_path)
    train, test = split_dataset(data)
    save_split(train, test, output_directory)
    return train, test


def main() -> None:
    """Create the project's train and test CSV files."""
    train, test = create_split()
    print(f"Training records: {len(train)}")
    print(f"Test records: {len(test)}")
    for name, subset in (("Training", train), ("Test", test)):
        counts = subset["label"].value_counts().reindex(["ham", "spam"], fill_value=0)
        spam_percentage = counts["spam"] / len(subset) * 100
        print(
            f"{name} classes: ham={counts['ham']}, spam={counts['spam']} "
            f"({spam_percentage:.2f}% spam)"
        )
    print("Overlapping messages: 0")
    print(f"Saved splits to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
