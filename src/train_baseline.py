"""Train and evaluate a TF-IDF / Multinomial Naive Bayes baseline."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "naive_bayes_baseline.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_model_report.md"
EXPECTED_COLUMNS = ["label", "message"]
LABELS = ["ham", "spam"]
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42


@dataclass
class BaselineResult:
    """Fitted pipeline and metrics from the held-out training validation set."""

    pipeline: Pipeline
    training_size: int
    validation_size: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion: list[list[int]]
    classification: str


def build_pipeline() -> Pipeline:
    """Create the baseline pipeline with the requested vectorizer settings."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB()),
        ]
    )


def load_training_data(train_path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """Load and validate only the training CSV; the final test CSV is never read."""
    data = pd.read_csv(
        train_path,
        encoding="utf-8",
        dtype={"label": "string", "message": "string"},
        keep_default_na=False,
    )
    if data.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Expected columns {EXPECTED_COLUMNS}; got {data.columns.tolist()}"
        )
    if data.empty:
        raise ValueError("Training data must contain at least one record.")
    if data.isna().any().any():
        raise ValueError("Training data contains missing values.")
    if not data["label"].isin(LABELS).all():
        invalid = sorted(set(data.loc[~data["label"].isin(LABELS), "label"]))
        raise ValueError(f"Training data contains invalid labels: {invalid}")
    if not data["message"].map(lambda message: isinstance(message, str)).all():
        raise TypeError("Every training message must be a string.")
    if data["message"].str.strip().eq("").any():
        raise ValueError("Training messages must not be empty.")
    if data["label"].value_counts().reindex(LABELS, fill_value=0).lt(2).any():
        raise ValueError("Each class must contain at least two records for stratification.")
    return data


def train_and_evaluate(
    train_path: Path = TRAIN_DATA_PATH,
    model_path: Path = MODEL_PATH,
    report_path: Path = REPORT_PATH,
) -> BaselineResult:
    """Split the training CSV, fit only its training subset, evaluate, and save."""
    data = load_training_data(train_path)
    train_messages, validation_messages, train_labels, validation_labels = (
        train_test_split(
            data["message"],
            data["label"],
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=data["label"],
        )
    )
    overlap = set(train_messages) & set(validation_messages)
    if overlap:
        raise ValueError(
            f"Validation leakage detected: {len(overlap)} identical messages "
            "appear in both subsets."
        )

    pipeline = build_pipeline()
    pipeline.fit(train_messages, train_labels)
    predictions = pipeline.predict(validation_messages)

    result = BaselineResult(
        pipeline=pipeline,
        training_size=len(train_messages),
        validation_size=len(validation_messages),
        accuracy=accuracy_score(validation_labels, predictions),
        precision=precision_score(
            validation_labels, predictions, pos_label="spam", zero_division=0
        ),
        recall=recall_score(
            validation_labels, predictions, pos_label="spam", zero_division=0
        ),
        f1_score=f1_score(
            validation_labels, predictions, pos_label="spam", zero_division=0
        ),
        confusion=confusion_matrix(
            validation_labels, predictions, labels=LABELS
        ).tolist(),
        classification=classification_report(
            validation_labels,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
            digits=4,
        ),
    )

    model_path = Path(model_path)
    report_path = Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: BaselineResult) -> str:
    """Format the validation results and baseline interpretation as Markdown."""
    true_ham, false_spam = result.confusion[0]
    false_ham, true_spam = result.confusion[1]
    return f"""# SMS Spam Classifier — Baseline Model Report

## Model architecture

The baseline is a scikit-learn `Pipeline` containing a `TfidfVectorizer`
followed by `MultinomialNB` with its default smoothing parameter (`alpha=1.0`).
Keeping vectorization and classification in one pipeline ensures the vocabulary
and inverse document frequencies are learned only from the model-training subset.

### TF-IDF configuration

- `lowercase=True`
- `strip_accents="unicode"`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.95`
- `sublinear_tf=True`

## Validation strategy

Only `data/processed/train.csv` was used. It was split with an 80/20
stratified `train_test_split` using `random_state=42`. The pipeline was fit
only on the model-training subset; the validation subset was used only for
prediction and evaluation. `data/processed/test.csv` was not loaded or used.

- Model-training subset: **{result.training_size:,}** records
- Validation subset: **{result.validation_size:,}** records
- Positive evaluation class: **spam**

## Validation metrics

| Metric (spam positive class) | Score |
|---|---:|
| Accuracy | {result.accuracy:.4f} |
| Precision | {result.precision:.4f} |
| Recall | {result.recall:.4f} |
| F1-score | {result.f1_score:.4f} |

### Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual \\ Predicted | ham | spam |
|---|---:|---:|
| ham | {true_ham} | {false_spam} |
| spam | {false_ham} | {true_spam} |

### Classification report

```text
{result.classification.rstrip()}
```

## Interpretation

- **True positive (TP):** spam correctly predicted as spam ({true_spam}).
- **True negative (TN):** ham correctly predicted as ham ({true_ham}).
- **False positive (FP):** ham incorrectly predicted as spam ({false_spam}); legitimate messages may be flagged.
- **False negative (FN):** spam incorrectly predicted as ham ({false_ham}); these spam messages may reach the recipient as legitimate mail.

False negatives deserve particular attention because they represent spam that
the classifier misses. Recall measures the fraction of actual spam found;
precision measures how often a spam prediction is correct. The F1-score
summarizes their balance. The measured scores describe this validation split
only, not the untouched final test set or future messages.

## Limitations

- This is a simple bag-of-words baseline; it does not understand message
  meaning, context, or sender behavior.
- The dataset has substantially more ham than spam, so accuracy alone can
  obscure weak performance on spam.
- A single random validation split can give a noisy estimate.
- Real-world messages may differ from this historical dataset, and false
  positives can prevent legitimate messages from being delivered.

## Recommendations

- Compare spam precision, recall, and F1-score, with special consideration of
  false negatives and the cost of false positives.
- Use cross-validation on training data for a more stable comparison.
- Review misclassified validation examples and assess whether additional
  message-level signals are appropriate.
- Keep the final test set untouched until model selection is complete.
"""


def main() -> None:
    """Train the baseline and print its held-out validation results."""
    result = train_and_evaluate()
    print(f"Model-training subset: {result.training_size}")
    print(f"Validation subset: {result.validation_size}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Precision (spam): {result.precision:.4f}")
    print(f"Recall (spam): {result.recall:.4f}")
    print(f"F1-score (spam): {result.f1_score:.4f}")
    print("Confusion matrix (rows=actual [ham, spam], columns=predicted [ham, spam]):")
    print(result.confusion)
    print("\nClassification report:")
    print(result.classification)
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
