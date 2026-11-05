"""Print a read-only summary of the original SMS Spam Collection dataset."""

import csv
from pathlib import Path

import pandas as pd


DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "SMSSpamCollection"
)
COLUMN_NAMES = ["label", "message"]


def main() -> None:
    """Load the original dataset and display basic data-quality information."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(
        DATA_PATH,
        sep="\t",
        header=None,
        names=COLUMN_NAMES,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
        dtype={"label": "string", "message": "string"},
        na_values=[""],
    )

    print(f"Dataset: {DATA_PATH}")
    print("\nColumns:")
    print(data.columns.tolist())
    print(f"\nShape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nFirst five records:")
    print(data.head().to_string(index=False))

    label_counts = data["label"].value_counts(dropna=False)
    print("\nClass counts:")
    print(label_counts.to_string())
    print("\nClass percentages:")
    print((data["label"].value_counts(normalize=True, dropna=False) * 100).round(2).to_string())

    print("\nMissing values by column:")
    print(data.isna().sum().to_string())

    message_counts = data["message"].value_counts(dropna=False)
    repeated_messages = message_counts[message_counts > 1]
    print(f"\nRepeated message-text groups: {len(repeated_messages)}")
    print(f"Duplicate message texts after first occurrence: {data['message'].duplicated().sum()}")
    print(f"Duplicate full records: {data.duplicated().sum()}")
    labels_per_message = data.groupby("message")["label"].nunique()
    conflicting_messages = labels_per_message[labels_per_message > 1]
    print(f"Messages with conflicting labels: {len(conflicting_messages)}")

    print("\nExample messages by class:")
    for label in ("ham", "spam"):
        print(f"\n{label}:")
        examples = data.loc[data["label"].eq(label), "message"].head(3)
        if examples.empty:
            print("  No messages found.")
        else:
            for message in examples:
                print(f"  - {message}")

    message_lengths = data["message"].str.len()
    print("\nMessage character-length distribution:")
    print(
        message_lengths.describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        ).to_string()
    )


if __name__ == "__main__":
    main()
