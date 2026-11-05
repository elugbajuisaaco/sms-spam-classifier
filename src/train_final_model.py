"""Fit the selected final pipeline and evaluate it once on the held-out test set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MANIFEST_PATH = (
    PROJECT_ROOT
    / "models"
    / "experiments"
    / "candidate_configurations.json"
)
FINAL_DIRECTORY = PROJECT_ROOT / "models" / "final"
ARTIFACT_PATH = FINAL_DIRECTORY / "final_model.joblib"
METADATA_PATH = FINAL_DIRECTORY / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "final_model_evaluation.md"
PLOT_PATH = PROJECT_ROOT / "reports" / "plots" / "final_test_confusion_matrix.png"
LABELS = ["ham", "spam"]
MODEL_VERSION = "1.0.0"


@dataclass
class FinalEvaluation:
    """Test metrics and information needed to build final deliverables."""

    training_count: int
    test_count: int
    threshold: float
    metrics: dict[str, Any]
    classification: str
    training_class_counts: dict[str, int]
    test_class_counts: dict[str, int]
    training_hash: str
    test_hash: str
    model_timestamp: str
    pipeline: Pipeline
    vectorizer_config: dict[str, Any]
    classifier_config: dict[str, Any]
    threshold_origin: str


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict[str, Any]:
    """Load and validate the chosen candidate and its OOF threshold."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    candidates = manifest.get("candidates", {})
    candidate = candidates.get("logistic_regression_balanced")
    if candidate is None:
        raise ValueError("Manifest is missing logistic_regression_balanced.")
    if candidate.get("algorithm") != "LogisticRegression":
        raise ValueError("Selected candidate is not LogisticRegression.")
    classifier_config = candidate.get("parameters", {})
    if classifier_config.get("class_weight") != "balanced":
        raise ValueError("Selected Logistic Regression is not class-weight balanced.")
    vectorizer_config = manifest.get("tfidf")
    if not isinstance(vectorizer_config, dict):
        raise ValueError("Manifest is missing TF-IDF configuration.")
    threshold_name = "OOF best F1"
    thresholds = candidate.get("oof_selected_thresholds", {})
    if threshold_name not in thresholds:
        raise ValueError("Manifest is missing its OOF best-F1 threshold.")
    threshold = float(thresholds[threshold_name])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid probability decision threshold: {threshold}")
    return {
        "vectorizer": vectorizer_config,
        "classifier": classifier_config,
        "threshold": threshold,
        "threshold_name": threshold_name,
        "validation_metrics": candidate.get(
            "validation_metrics_at_oof_f1_threshold"
        ),
        "manifest_split": manifest.get("split"),
        "manifest_oof_selection": manifest.get("oof_threshold_selection"),
    }


def build_final_pipeline(
    vectorizer_config: dict[str, Any],
    classifier_config: dict[str, Any],
) -> Pipeline:
    """Build the selected TF-IDF and balanced Logistic Regression pipeline."""
    config = dict(vectorizer_config)
    if isinstance(config.get("ngram_range"), list):
        config["ngram_range"] = tuple(config["ngram_range"])
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**config)),
            ("classifier", LogisticRegression(**classifier_config)),
        ]
    )


def _load_csv(path: Path) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"label": "string", "message": "string"},
        keep_default_na=False,
    )
    if data.columns.tolist() != ["label", "message"]:
        raise ValueError(f"{path} must contain exactly label and message columns.")
    if data.empty:
        raise ValueError(f"{path} contains no records.")
    if data.isna().any().any():
        raise ValueError(f"{path} contains missing values.")
    if set(data["label"].unique()) != set(LABELS):
        raise ValueError(f"{path} must contain only ham and spam labels.")
    if not data["message"].map(lambda message: isinstance(message, str)).all():
        raise TypeError(f"{path} contains non-string messages.")
    if data["message"].str.strip().eq("").any():
        raise ValueError(f"{path} contains empty messages.")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_test_separation(
    train: pd.DataFrame, test: pd.DataFrame
) -> None:
    overlap = set(train["message"]) & set(test["message"])
    if overlap:
        raise ValueError(
            f"Train/test message leakage detected: {len(overlap)} identical "
            "message text(s) occur in both datasets."
        )


def _make_confusion_plot(matrix: np.ndarray, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["ham", "spam"])
    axis.set_yticks([0, 1], ["ham", "spam"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_title("Final model — independent test set")
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
            )
    figure.colorbar(image, ax=axis, label="Record count")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def render_report(result: FinalEvaluation) -> str:
    """Create the final evaluation report from the single test evaluation."""
    matrix = result.metrics["confusion_matrix"]
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    validation = result.metrics.get("validation_metrics")
    validation_section = ""
    if validation:
        validation_section = f"""| Accuracy | {validation['accuracy']:.4f} | {result.metrics['accuracy']:.4f} |
| Spam precision | {validation['precision']:.4f} | {result.metrics['precision']:.4f} |
| Spam recall | {validation['recall']:.4f} | {result.metrics['recall']:.4f} |
| Spam F1-score | {validation['f1']:.4f} | {result.metrics['f1']:.4f} |
| False positives | {validation['false_positives']} | {fp} |
| False negatives | {validation['false_negatives']} | {fn} |"""
    else:
        validation_section = "The manifest did not contain validation metrics."

    return f"""# Final SMS Spam Classifier — Independent Test Evaluation

## Selected model and justification

The selected candidate is **TF-IDF + Logistic Regression with
`class_weight="balanced"`**. It was chosen provisionally using training-only
out-of-fold threshold selection and exploratory validation results in
`reports/model_comparison_report.md`. No competing model was evaluated on the
test set, and no model or threshold decisions were changed after test
evaluation.

## Training dataset

- Source: `data/processed/train.csv`
- Records used for final fitting: **{result.training_count:,}**
- Class counts: ham={result.training_class_counts['ham']:,},
  spam={result.training_class_counts['spam']:,}
- SHA-256: `{result.training_hash}`
- The complete training dataset was used to fit the final pipeline.

## Decision threshold

- Threshold: **{result.threshold:.16g}**
- Origin: Task 07 manifest key
  `candidates.logistic_regression_balanced.oof_selected_thresholds.{result.threshold_origin}`.
- This probability threshold was selected using 5-fold stratified
  out-of-fold predictions on the Task 07 model-training subset only. It was
  not selected or adjusted using test labels.

## Test dataset and independent evaluation

- Source: `data/processed/test.csv`
- Records evaluated: **{result.test_count:,}**
- Class counts: ham={result.test_class_counts['ham']:,},
  spam={result.test_class_counts['spam']:,}
- SHA-256: `{result.test_hash}`
- Test labels were used once for the evaluation below. The test data was not
  passed to `fit`, used to choose the threshold, or used to change
  configuration.
- Positive class: **spam**

## Test metrics

| Metric | Independent test result |
|---|---:|
| Accuracy | {result.metrics['accuracy']:.4f} |
| Spam precision | {result.metrics['precision']:.4f} |
| Spam recall | {result.metrics['recall']:.4f} |
| Spam F1-score | {result.metrics['f1']:.4f} |
| Macro F1-score | {result.metrics['macro_f1']:.4f} |
| Average Precision (PR-AUC summary) | {result.metrics['average_precision']:.4f} |
| ROC-AUC | {result.metrics['roc_auc']:.4f} |
| False positives | {fp} |
| False negatives | {fn} |

### Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual \\ Predicted | ham | spam |
|---|---:|---:|
| ham | {tn} | {fp} |
| spam | {fn} | {tp} |

![Independent test confusion matrix](plots/final_test_confusion_matrix.png)

### Classification report

```text
{result.classification.rstrip()}
```

## False-positive and false-negative analysis

- **False positives ({fp}):** legitimate ham messages classified as spam.
  These may be incorrectly flagged or filtered.
- **False negatives ({fn}):** spam messages classified as ham. These are
  missed spam and are important to review when assessing the operating
  trade-off.

## Validation versus final test

The validation figures below are from exploratory Task 07 validation and are
not the independent final test result. The test column is the single
evaluation performed for this task.

| Metric | Validation (exploratory) | Final test (independent) |
|---|---:|---:|
{validation_section}

Validation and test metrics can differ because they contain different
messages. The test results are reported as measured and were not used for
further selection.

## Model configuration and saved artifacts

- Vectorizer: `TfidfVectorizer` with
  `lowercase={result.vectorizer_config['lowercase']}`,
  `strip_accents="{result.vectorizer_config['strip_accents']}"`,
  `ngram_range={tuple(result.vectorizer_config['ngram_range'])}`,
  `min_df={result.vectorizer_config['min_df']}`,
  `max_df={result.vectorizer_config['max_df']}`,
  `sublinear_tf={result.vectorizer_config['sublinear_tf']}`.
- Classifier: `LogisticRegression` with
  `class_weight="{result.classifier_config['class_weight']}"`,
  `max_iter={result.classifier_config['max_iter']}`,
  `random_state={result.classifier_config['random_state']}`.
- Pipeline plus threshold and metadata: `models/final/final_model.joblib`
  (Joblib serialization; bundle keys are `pipeline`, `threshold`, and
  `metadata`).
- Configuration sidecar: `models/final/model_metadata.json`.
- Model version: `{MODEL_VERSION}`
- Created (UTC): `{result.model_timestamp}`

## Limitations and application-readiness

This evaluation represents one held-out sample from the same collection, not
all future SMS traffic. The dataset is historically sourced and class
imbalanced; message content and spam tactics can shift. Average Precision and
ROC-AUC summarize ranking and do not replace the chosen operating threshold.
False positives and false negatives have different costs and should be
monitored in use.

The fitted pipeline and threshold are reproducibly packaged, so the project
can proceed to application development using the documented bundle format.
Treat the model as a baseline, disclose its limitations, and avoid representing
the test result as a guarantee of real-world performance.
"""


def train_and_evaluate_once(
    train_path: Path = TRAIN_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    manifest_path: Path = MANIFEST_PATH,
    final_directory: Path = FINAL_DIRECTORY,
    report_path: Path = REPORT_PATH,
    plot_path: Path = PLOT_PATH,
) -> FinalEvaluation:
    """Fit on train.csv, evaluate test.csv once, and save final artifacts."""
    train_path = Path(train_path)
    test_path = Path(test_path)
    final_directory = Path(final_directory)
    report_path = Path(report_path)
    plot_path = Path(plot_path)

    existing = [
        path
        for path in (
            final_directory / "final_model.joblib",
            final_directory / "model_metadata.json",
            report_path,
            plot_path,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite final evaluation artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    selection = load_manifest(manifest_path)
    train = _load_csv(train_path)
    training_hash = _sha256(train_path)

    pipeline = build_final_pipeline(
        selection["vectorizer"], selection["classifier"]
    )
    pipeline.fit(train["message"], train["label"])

    # Read and score the held-out test data once, after fitting is complete.
    test = _load_csv(test_path)
    _validate_test_separation(train, test)
    test_hash = _sha256(test_path)
    probabilities = pipeline.predict_proba(test["message"])
    classes = list(pipeline.named_steps["classifier"].classes_)
    spam_probability = probabilities[:, classes.index("spam")]
    predictions = np.where(
        spam_probability >= selection["threshold"], "spam", "ham"
    )
    matrix = confusion_matrix(test["label"], predictions, labels=LABELS)
    actual_spam = test["label"].eq("spam").to_numpy()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(test["label"], predictions)),
        "precision": float(
            precision_score(
                test["label"], predictions, pos_label="spam", zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                test["label"], predictions, pos_label="spam", zero_division=0
            )
        ),
        "f1": float(
            f1_score(test["label"], predictions, pos_label="spam", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(test["label"], predictions, average="macro", zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(actual_spam, spam_probability)
        ),
        "roc_auc": float(roc_auc_score(actual_spam, spam_probability)),
        "confusion_matrix": matrix.tolist(),
        "false_positives": int(matrix[0, 1]),
        "false_negatives": int(matrix[1, 0]),
        "classification_report": classification_report(
            test["label"],
            predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
            digits=4,
        ),
    }
    metrics["validation_metrics"] = selection["validation_metrics"]

    model_timestamp = datetime.now(timezone.utc).isoformat()
    result = FinalEvaluation(
        training_count=len(train),
        test_count=len(test),
        threshold=selection["threshold"],
        metrics=metrics,
        classification=metrics["classification_report"],
        training_class_counts={
            label: int(train["label"].eq(label).sum()) for label in LABELS
        },
        test_class_counts={
            label: int(test["label"].eq(label).sum()) for label in LABELS
        },
        training_hash=training_hash,
        test_hash=test_hash,
        model_timestamp=model_timestamp,
        pipeline=pipeline,
        vectorizer_config=selection["vectorizer"],
        classifier_config=selection["classifier"],
        threshold_origin=selection["threshold_name"],
    )

    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "created_at_utc": model_timestamp,
        "model_name": "tfidf_logistic_regression_balanced",
        "serialization": "joblib",
        "pipeline_steps": ["tfidf", "classifier"],
        "vectorizer": selection["vectorizer"],
        "classifier": selection["classifier"],
        "decision_threshold": selection["threshold"],
        "threshold_type": "spam predict_proba threshold",
        "threshold_origin": {
            "source": "models/experiments/candidate_configurations.json",
            "candidate": "logistic_regression_balanced",
            "key": selection["threshold_name"],
            "selection": "5-fold stratified out-of-fold best F1 on training subset",
            "value": selection["threshold"],
        },
        "training_data": {
            "path": str(train_path),
            "record_count": len(train),
            "class_counts": result.training_class_counts,
            "sha256": training_hash,
        },
        "independent_test": {
            "path": str(test_path),
            "record_count": len(test),
            "class_counts": result.test_class_counts,
            "sha256": test_hash,
            "used_for": "one independent evaluation only; never fitted",
        },
        "validation_metrics_from_task_07": selection["validation_metrics"],
    }

    final_directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "threshold": selection["threshold"], "metadata": metadata},
        final_directory / "final_model.joblib",
    )
    (final_directory / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _make_confusion_plot(matrix, plot_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    """Run the final fit and single independent evaluation."""
    result = train_and_evaluate_once()
    print(f"Training records: {result.training_count}")
    print(f"Independent test records: {result.test_count}")
    print(f"Decision threshold: {result.threshold:.16g}")
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "average_precision",
        "roc_auc",
        "false_positives",
        "false_negatives",
    ):
        print(f"{metric}: {result.metrics[metric]}")
    print("Confusion matrix [ham, spam]:")
    print(result.metrics["confusion_matrix"])
    print("Classification report:")
    print(result.classification)
    print(f"Saved final model: {FINAL_DIRECTORY / 'final_model.joblib'}")
    print(f"Saved metadata: {FINAL_DIRECTORY / 'model_metadata.json'}")
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved confusion matrix: {PLOT_PATH}")


if __name__ == "__main__":
    main()
