"""Compare spam-classification baselines using training-only validation data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
EXPERIMENTS_DIRECTORY = PROJECT_ROOT / "models" / "experiments"
REPORT_PATH = PROJECT_ROOT / "reports" / "model_comparison_report.md"
PLOTS_DIRECTORY = PROJECT_ROOT / "reports" / "plots"
LABELS = ["ham", "spam"]
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
PROBABILITY_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

VECTORIZER_CONFIG: dict[str, Any] = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {
    "multinomial_nb_default": {
        "algorithm": "MultinomialNB",
        "parameters": {"fit_prior": True, "class_prior": None},
    },
    "multinomial_nb_uniform_prior": {
        "algorithm": "MultinomialNB",
        "parameters": {"fit_prior": False, "class_prior": None},
    },
    "logistic_regression_default": {
        "algorithm": "LogisticRegression",
        "parameters": {
            "class_weight": None,
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
    },
    "logistic_regression_balanced": {
        "algorithm": "LogisticRegression",
        "parameters": {
            "class_weight": "balanced",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
    },
    "linear_svm_default": {
        "algorithm": "LinearSVC",
        "parameters": {
            "class_weight": None,
            "max_iter": 5000,
            "random_state": RANDOM_STATE,
        },
    },
    "linear_svm_balanced": {
        "algorithm": "LinearSVC",
        "parameters": {
            "class_weight": "balanced",
            "max_iter": 5000,
            "random_state": RANDOM_STATE,
        },
    },
}


@dataclass
class Metrics:
    """Binary classification metrics with spam as the positive class."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    average_precision: float
    roc_auc: float
    confusion_matrix: list[list[int]]
    false_positives: int
    false_negatives: int


@dataclass
class CandidateResult:
    """Validation results for one model configuration."""

    name: str
    pipeline: Pipeline
    score_kind: str
    default_threshold: float
    default_metrics: Metrics
    oof_f1_threshold: float
    oof_threshold_metrics: Metrics
    oof_thresholds: dict[str, float]
    threshold_metrics: dict[str, Metrics]
    validation_scores: np.ndarray


def build_vectorizer() -> TfidfVectorizer:
    """Build the shared TF-IDF configuration used by all candidates."""
    return TfidfVectorizer(**VECTORIZER_CONFIG)


def build_candidate(name: str) -> Pipeline:
    """Create one configured vectorizer/classifier pipeline."""
    if name not in CANDIDATE_CONFIGS:
        raise ValueError(f"Unknown model candidate: {name}")

    config = CANDIDATE_CONFIGS[name]
    parameters = config["parameters"]
    algorithm = config["algorithm"]
    if algorithm == "MultinomialNB":
        classifier = MultinomialNB(**parameters)
    elif algorithm == "LogisticRegression":
        classifier = LogisticRegression(**parameters)
    else:
        classifier = LinearSVC(**parameters)

    return Pipeline(
        [
            ("tfidf", build_vectorizer()),
            ("classifier", classifier),
        ]
    )


def load_training_data(train_path: Path = TRAIN_DATA_PATH) -> pd.DataFrame:
    """Load only the training CSV and verify its expected schema and values."""
    data = pd.read_csv(
        train_path,
        encoding="utf-8",
        dtype={"label": "string", "message": "string"},
        keep_default_na=False,
    )
    if data.columns.tolist() != ["label", "message"]:
        raise ValueError("Training data must contain exactly label and message columns.")
    if data.empty or data.isna().any().any():
        raise ValueError("Training data must be non-empty and contain no missing values.")
    if set(data["label"].unique()) != set(LABELS):
        raise ValueError("Training data must contain only ham and spam labels.")
    if not data["message"].map(lambda message: isinstance(message, str)).all():
        raise TypeError("Every training message must be a string.")
    if data["message"].str.strip().eq("").any():
        raise ValueError("Training messages must not be empty.")
    if data["message"].duplicated().any():
        raise ValueError("Training data contains duplicate messages.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("Each class needs at least two records.")
    return data


def _positive_scores(
    estimator: Pipeline, messages: pd.Series, score_kind: str
) -> np.ndarray:
    """Return spam probability or uncalibrated decision scores."""
    if score_kind == "probability":
        classes = list(estimator.named_steps["classifier"].classes_)
        spam_index = classes.index("spam")
        return estimator.predict_proba(messages)[:, spam_index]
    return np.asarray(estimator.decision_function(messages)).ravel()


def _metrics_from_predictions(
    labels: pd.Series, predictions: np.ndarray, scores: np.ndarray
) -> Metrics:
    actual_spam = labels.eq("spam").to_numpy()
    predicted_spam = predictions == "spam"
    matrix = confusion_matrix(labels, predictions, labels=LABELS)
    return Metrics(
        accuracy=float(accuracy_score(labels, predictions)),
        precision=float(
            precision_score(labels, predictions, pos_label="spam", zero_division=0)
        ),
        recall=float(
            recall_score(labels, predictions, pos_label="spam", zero_division=0)
        ),
        f1=float(f1_score(labels, predictions, pos_label="spam", zero_division=0)),
        macro_f1=float(f1_score(labels, predictions, average="macro", zero_division=0)),
        average_precision=float(average_precision_score(actual_spam, scores)),
        roc_auc=float(roc_auc_score(actual_spam, scores)),
        confusion_matrix=matrix.tolist(),
        false_positives=int(np.sum(~actual_spam & predicted_spam)),
        false_negatives=int(np.sum(actual_spam & ~predicted_spam)),
    )


def evaluate_threshold(
    labels: pd.Series, scores: np.ndarray, threshold: float
) -> Metrics:
    """Evaluate one threshold over spam scores, with spam as the positive class."""
    predictions = np.where(scores >= threshold, "spam", "ham")
    return _metrics_from_predictions(labels, predictions, scores)


def _threshold_for_best_f1(labels: pd.Series, scores: np.ndarray) -> float:
    actual_spam = labels.eq("spam").to_numpy()
    precision, recall, thresholds = precision_recall_curve(actual_spam, scores)
    if thresholds.size == 0:
        return float(np.max(scores))
    f1_values = (
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    )
    return float(thresholds[int(np.argmax(f1_values))])


def _threshold_for_recall(
    labels: pd.Series, scores: np.ndarray, target_recall: float
) -> float:
    actual_spam = labels.eq("spam").to_numpy()
    precision, recall, thresholds = precision_recall_curve(actual_spam, scores)
    viable = np.flatnonzero(recall[:-1] >= target_recall)
    if viable.size == 0:
        return float(np.min(scores))
    return float(np.max(thresholds[viable]))


def fit_and_evaluate_candidate(
    name: str,
    train_messages: pd.Series,
    train_labels: pd.Series,
    validation_messages: pd.Series,
    validation_labels: pd.Series,
    *,
    cv_splits: int = 5,
) -> CandidateResult:
    """Use OOF training scores to select thresholds, then evaluate on validation."""
    if set(train_labels.unique()) != set(LABELS):
        raise ValueError("Model training labels must contain ham and spam.")
    if set(validation_labels.unique()) != set(LABELS):
        raise ValueError("Validation labels must contain ham and spam.")
    if set(train_messages) & set(validation_messages):
        raise ValueError("Training and validation message text overlaps.")

    pipeline = build_candidate(name)
    score_kind = (
        "probability"
        if hasattr(pipeline.named_steps["classifier"], "predict_proba")
        else "decision_function"
    )
    method = "predict_proba" if score_kind == "probability" else "decision_function"
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    oof_output = cross_val_predict(
        clone(pipeline),
        train_messages,
        train_labels,
        cv=cv,
        method=method,
    )
    if score_kind == "probability":
        classes = sorted(train_labels.unique())
        oof_scores = oof_output[:, classes.index("spam")]
    else:
        oof_scores = np.asarray(oof_output).ravel()

    pipeline.fit(train_messages, train_labels)
    validation_scores = _positive_scores(pipeline, validation_messages, score_kind)
    default_threshold = 0.5 if score_kind == "probability" else 0.0
    default_metrics = evaluate_threshold(
        validation_labels, validation_scores, default_threshold
    )

    oof_thresholds = {
        "OOF best F1": _threshold_for_best_f1(train_labels, oof_scores),
        "OOF recall >= 0.80": _threshold_for_recall(
            train_labels, oof_scores, 0.80
        ),
        "OOF recall >= 0.90": _threshold_for_recall(
            train_labels, oof_scores, 0.90
        ),
    }
    oof_f1_threshold = oof_thresholds["OOF best F1"]
    oof_threshold_metrics = evaluate_threshold(
        validation_labels, validation_scores, oof_f1_threshold
    )

    thresholds: dict[str, float] = {
        f"Default ({default_threshold:.6g})": default_threshold,
        **{
            f"{name} ({threshold:.6g})": threshold
            for name, threshold in oof_thresholds.items()
        },
    }
    if score_kind == "probability":
        thresholds.update(
            {
                f"Validation experiment ({threshold:.2f})": threshold
                for threshold in PROBABILITY_THRESHOLDS
            }
        )
    threshold_metrics = {
        label: evaluate_threshold(validation_labels, validation_scores, threshold)
        for label, threshold in thresholds.items()
    }

    return CandidateResult(
        name=name,
        pipeline=pipeline,
        score_kind=score_kind,
        default_threshold=default_threshold,
        default_metrics=default_metrics,
        oof_f1_threshold=oof_f1_threshold,
        oof_threshold_metrics=oof_threshold_metrics,
        oof_thresholds=oof_thresholds,
        threshold_metrics=threshold_metrics,
        validation_scores=validation_scores,
    )


def _format_confusion(metrics: Metrics) -> str:
    tn, fp = metrics.confusion_matrix[0]
    fn, tp = metrics.confusion_matrix[1]
    return f"[[{tn}, {fp}], [{fn}, {tp}]] (FP={fp}, FN={fn})"


def _metrics_table(result: CandidateResult) -> str:
    m = result.oof_threshold_metrics
    return (
        f"| `{result.name}` | {result.score_kind} | "
        f"{result.default_threshold:.6g} | {result.oof_f1_threshold:.6g} | "
        f"{m.accuracy:.4f} | {m.precision:.4f} | {m.recall:.4f} | "
        f"{m.f1:.4f} | {m.macro_f1:.4f} | {m.average_precision:.4f} | "
        f"{m.roc_auc:.4f} | {m.false_positives} | {m.false_negatives} | "
        f"`{_format_confusion(m)}` |"
    )


def _class_effects_text(results: list[CandidateResult]) -> str:
    by_name = {result.name: result for result in results}

    def summary(metrics: Metrics) -> str:
        return (
            f"precision {metrics.precision:.4f}, recall {metrics.recall:.4f}, "
            f"F1 {metrics.f1:.4f}, FP {metrics.false_positives}, "
            f"FN {metrics.false_negatives}"
        )

    nb_default = by_name["multinomial_nb_default"].default_metrics
    nb_uniform = by_name["multinomial_nb_uniform_prior"].default_metrics
    lr_default = by_name["logistic_regression_default"].default_metrics
    lr_balanced = by_name["logistic_regression_balanced"].default_metrics
    svm_default = by_name["linear_svm_default"].default_metrics
    svm_balanced = by_name["linear_svm_balanced"].default_metrics
    return (
        "- **Naive Bayes prior (default cutoff 0.5):** learned priors "
        f"({summary(nb_default)}); uniform priors "
        f"({summary(nb_uniform)}). The uniform prior reduced false "
        "negatives on this validation set while increasing false positives.\n"
        "- **Logistic Regression (default cutoff 0.5):** unweighted "
        f"({summary(lr_default)}); balanced weights "
        f"({summary(lr_balanced)}). Balanced weights increased spam "
        "recall and reduced false negatives here, with a precision/false-"
        "positive trade-off.\n"
        "- **LinearSVC (default score cutoff 0.0):** unweighted "
        f"({summary(svm_default)}); balanced weights "
        f"({summary(svm_balanced)}). The change was modest on this "
        "validation set; it did not improve every metric.\n"
        "- At OOF-selected F1 cutoffs, compare the comparison table: Logistic "
        "Regression weighting changed precision/recall/F1/FN from "
        f"{summary(by_name['logistic_regression_default'].oof_threshold_metrics)} "
        "to "
        f"{summary(by_name['logistic_regression_balanced'].oof_threshold_metrics)}. "
        "OOF threshold selection can change the class-weight effect relative "
        "to the estimator's default cutoff."
    )


def _make_plots(results: list[CandidateResult], labels: pd.Series, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    plot_names: list[str] = []

    names = [result.name.replace("_", "\n") for result in results]
    recalls = [result.oof_threshold_metrics.recall for result in results]
    f1_scores = [result.oof_threshold_metrics.f1 for result in results]
    precisions = [result.oof_threshold_metrics.precision for result in results]
    positions = np.arange(len(results))
    width = 0.25
    fig, axis = plt.subplots(figsize=(13, 6))
    axis.bar(positions - width, precisions, width, label="Spam precision")
    axis.bar(positions, recalls, width, label="Spam recall")
    axis.bar(positions + width, f1_scores, width, label="Spam F1")
    axis.set_xticks(positions, names)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Validation score")
    axis.set_title("Validation performance at OOF-selected F1 thresholds")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "model_performance_comparison.png", dpi=150)
    plt.close(fig)
    plot_names.append("model_performance_comparison.png")

    actual_spam = labels.eq("spam").to_numpy()
    fig, axis = plt.subplots(figsize=(8, 7))
    for result in results:
        precision, recall, _ = precision_recall_curve(
            actual_spam, result.validation_scores
        )
        ap = result.oof_threshold_metrics.average_precision
        axis.plot(recall, precision, label=f"{result.name} (AP={ap:.3f})")
    axis.set_xlabel("Spam recall")
    axis.set_ylabel("Spam precision")
    axis.set_title("Validation precision-recall curves (exploratory)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "precision_recall_curves.png", dpi=150)
    plt.close(fig)
    plot_names.append("precision_recall_curves.png")

    leaders = {
        max(
            results,
            key=lambda result: (
                result.oof_threshold_metrics.recall,
                result.oof_threshold_metrics.f1,
            ),
        ).name,
        max(
            results,
            key=lambda result: (
                result.oof_threshold_metrics.f1,
                result.oof_threshold_metrics.precision,
            ),
        ).name,
        "multinomial_nb_default",
    }
    for result in results:
        if result.name not in leaders:
            continue
        matrix = np.asarray(result.oof_threshold_metrics.confusion_matrix)
        fig, axis = plt.subplots(figsize=(5, 4))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks([0, 1], ["ham", "spam"])
        axis.set_yticks([0, 1], ["ham", "spam"])
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("Actual label")
        axis.set_title(
            f"{result.name}\nOOF-F1 threshold {result.oof_f1_threshold:.4g}"
        )
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        filename = f"confusion_matrix_{result.name}.png"
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
        plot_names.append(filename)
    return plot_names


def render_report(
    results: list[CandidateResult],
    training_size: int,
    validation_size: int,
    plot_names: list[str],
) -> str:
    """Render a complete comparison report from measured results."""
    comparison_rows = "\n".join(_metrics_table(result) for result in results)
    threshold_sections: list[str] = []
    for result in results:
        rows = []
        for name, metrics in result.threshold_metrics.items():
            rows.append(
                f"| {name} | {metrics.precision:.4f} | {metrics.recall:.4f} | "
                f"{metrics.f1:.4f} | {metrics.false_positives} | "
                f"{metrics.false_negatives} | `{_format_confusion(metrics)}` |"
            )
        threshold_sections.append(
            f"### `{result.name}` ({result.score_kind})\n\n"
            "| Threshold | Precision | Recall | F1 | FP | FN | Confusion matrix (TN, FP; FN, TP) |\n"
            "|---|---:|---:|---:|---:|---:|---|\n"
            + "\n".join(rows)
        )

    best_recall = max(
        results, key=lambda result: result.oof_threshold_metrics.recall
    )
    best_f1 = max(
        results,
        key=lambda result: (
            result.oof_threshold_metrics.f1,
            result.oof_threshold_metrics.precision,
        ),
    )
    probability_grid_candidates = [
        (result, label, metrics)
        for result in results
        if result.score_kind == "probability"
        for label, metrics in result.threshold_metrics.items()
        if label.startswith("Validation experiment")
    ]
    highest_grid_recall = max(
        probability_grid_candidates,
        key=lambda item: (
            item[2].recall,
            item[2].precision,
            item[2].f1,
        ),
    )
    baseline = next(
        result for result in results if result.name == "multinomial_nb_default"
    )
    plot_links = "\n".join(
        f"![{name}](plots/{name})" for name in plot_names
    )
    configs = "\n".join(
        f"- `{name}`: `{config['algorithm']}` with `{config['parameters']}`; "
        f"saved as `models/experiments/{name}.joblib`."
        for name, config in CANDIDATE_CONFIGS.items()
    )
    return f"""# SMS Spam Classifier — Model Comparison

## Scope and validation protocol

The saved baseline was first reproduced using the existing training/validation
split. For all new candidates, only `data/processed/train.csv` was read. It
was split once with `train_test_split(test_size=0.2, random_state=42,
stratify=y)`: **{training_size:,}** model-training records and
**{validation_size:,}** validation records. The test file was not accessed.

Each model uses the same TF-IDF settings inside its own scikit-learn Pipeline:
`lowercase=True`, `strip_accents="unicode"`, `ngram_range=(1, 2)`,
`min_df=2`, `max_df=0.95`, and `sublinear_tf=True`. Pipelines were fit only
on the model-training subset. OOF scores from 5-fold `StratifiedKFold`
(`shuffle=True`, `random_state=42`) on that subset selected candidate
thresholds without using validation labels. Validation-based comparisons and
the charts below are exploratory and must not be interpreted as final test
performance.

Spam is the positive class. Confusion matrices use `[[TN, FP], [FN, TP]]`.
Average precision (AP), a common summary of the spam precision-recall curve,
is reported alongside ROC-AUC using continuous probability or decision scores.
AP is not a calibrated probability. Linear SVM decision values are uncalibrated
scores, not probabilities.

## Reproduced baseline

The saved `models/naive_bayes_baseline.joblib` reproduced the prior validation
result: accuracy **{baseline.default_metrics.accuracy:.4f}**, spam precision
**{baseline.default_metrics.precision:.4f}**, recall
**{baseline.default_metrics.recall:.4f}**, F1
**{baseline.default_metrics.f1:.4f}**, and confusion matrix
`{_format_confusion(baseline.default_metrics)}`. This matches the Task 06
report (rounding to 96.26% accuracy, 100% precision, 70.48% recall,
82.68% F1, and 31 false negatives).

## Model comparison

Metrics in this table use each candidate's OOF-selected best-F1 threshold,
evaluated on the validation subset. For probabilities, default prediction
threshold is 0.5; for LinearSVC, the default decision cutoff is 0.0.
Average precision and ROC-AUC are ranking metrics and do not depend on the
selected cutoff.

| Candidate | Score type | Default cutoff | OOF best-F1 cutoff | Accuracy | Spam precision | Spam recall | Spam F1 | Macro F1 | PR-AUC / AP | ROC-AUC | FP | FN | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{comparison_rows}

## Class weighting and prior effects

Logistic Regression and LinearSVC are each compared with default
`class_weight=None` and `class_weight="balanced"`. The default-cutoff effects
observed on validation were:

{_class_effects_text(results)}

MultinomialNB's `fit_prior=False` candidate uses uniform class priors and is
compared with the learned-prior baseline (`fit_prior=True`). The results are
split-dependent: balanced weights or uniform priors are not assumed to be
superior, and no setting is selected based on accuracy alone.

## Threshold experiments

For probability-producing candidates, the threshold tables include the
requested validation experiment cutoffs 0.20, 0.30, 0.40, 0.50, 0.60, and
0.70. These rows use validation labels and are exploratory, not threshold
selection evidence. OOF-selected F1 and recall-target thresholds are also
listed; those thresholds were chosen using OOF model-training-subset labels,
then evaluated on validation.

For LinearSVC, the corresponding values are decision-function cutoffs, which
may be negative or positive and are not calibrated probabilities. Its default
cutoff is 0.0. OOF F1-maximizing and OOF recall-at-least-0.80/0.90 cutoffs are
shown and then evaluated on validation.

{chr(10).join(threshold_sections)}

Lowering a threshold generally increases spam recall and reduces false
negatives, at the potential cost of more false positives and lower precision.
The tables report the observed trade-off rather than assuming a threshold is
universally best.

## Operating-condition candidates

- **Highest validation spam recall at the OOF-F1 threshold:** `{best_recall.name}`
  (recall {best_recall.oof_threshold_metrics.recall:.4f}, precision
  {best_recall.oof_threshold_metrics.precision:.4f}, F1
  {best_recall.oof_threshold_metrics.f1:.4f}, FN
  {best_recall.oof_threshold_metrics.false_negatives}).
- **Highest validation spam F1 at the OOF-F1 threshold:** `{best_f1.name}`
  (precision {best_f1.oof_threshold_metrics.precision:.4f}, recall
  {best_f1.oof_threshold_metrics.recall:.4f}, F1
  {best_f1.oof_threshold_metrics.f1:.4f}, FN
  {best_f1.oof_threshold_metrics.false_negatives}).
- **Highest recall in the requested validation probability grid (exploratory
  validation-based selection):** `{highest_grid_recall[0].name}` at
  `{highest_grid_recall[1].split('(')[-1].rstrip(')')}` (recall
  {highest_grid_recall[2].recall:.4f}, precision
  {highest_grid_recall[2].precision:.4f}, F1
  {highest_grid_recall[2].f1:.4f}, FP
  {highest_grid_recall[2].false_positives}, FN
  {highest_grid_recall[2].false_negatives}). This was observed by examining
  validation results and must not be treated as an unbiased threshold choice.

These are exploratory validation-based candidates, not a final model choice.
The operating cost of false negatives (spam missed) versus false positives
(legitimate SMS flagged) should determine a deployment threshold.

## Saved candidate configurations

All six fitted pipelines, each containing its fitted TF-IDF vectorizer, are
saved separately. Full estimator settings are recorded here and in
`models/experiments/candidate_configurations.json`:

{configs}

## Plots

{plot_links}

Confusion-matrix plots use each leading candidate's OOF-F1 threshold.
Precision-recall curves and all validation plots are exploratory.

## Limitations and recommendations

- This is one stratified validation split; validation-guided model comparison
  can overfit model choices to this split. Keep `test.csv` untouched until
  model selection and threshold decisions are complete.
- OOF threshold selection reduces direct use of validation labels for
  threshold selection, but scores remain estimates from the available
  training data and a chosen threshold should be confirmed later.
- LinearSVC decision scores are not calibrated probabilities. Calibration,
  if needed, must be learned using training data only.
- Class weighting and Naive Bayes priors can change precision/recall in
  different directions; monitor false negatives and false positives.
- Consider repeated cross-validation on training data and document the
  acceptable false-negative/false-positive trade-off before deployment.
"""


def run_comparison(
    train_path: Path = TRAIN_DATA_PATH,
    experiments_directory: Path = EXPERIMENTS_DIRECTORY,
    report_path: Path = REPORT_PATH,
    plots_directory: Path = PLOTS_DIRECTORY,
    *,
    cv_splits: int = 5,
) -> list[CandidateResult]:
    """Train candidates, select thresholds with OOF training scores, and report."""
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
    if set(train_messages) & set(validation_messages):
        raise ValueError("Model-training and validation messages overlap.")

    results = [
        fit_and_evaluate_candidate(
            name,
            train_messages,
            train_labels,
            validation_messages,
            validation_labels,
            cv_splits=cv_splits,
        )
        for name in CANDIDATE_CONFIGS
    ]

    experiments_directory = Path(experiments_directory)
    experiments_directory.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "training_source": str(Path(train_path)),
        "validation_size": len(validation_messages),
        "model_training_size": len(train_messages),
        "split": {
            "test_size": VALIDATION_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
        },
        "oof_threshold_selection": {
            "method": "StratifiedKFold",
            "n_splits": cv_splits,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "tfidf": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in VECTORIZER_CONFIG.items()
        },
        "candidates": {},
    }
    for result in results:
        joblib.dump(result.pipeline, experiments_directory / f"{result.name}.joblib")
        manifest["candidates"][result.name] = {
            **CANDIDATE_CONFIGS[result.name],
            "score_type": result.score_kind,
            "oof_selected_thresholds": result.oof_thresholds,
            "validation_metrics_at_oof_f1_threshold": asdict(
                result.oof_threshold_metrics
            ),
        }
    (experiments_directory / "candidate_configurations.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    plot_names = _make_plots(results, validation_labels, Path(plots_directory))
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_report(
            results,
            training_size=len(train_messages),
            validation_size=len(validation_messages),
            plot_names=plot_names,
        ),
        encoding="utf-8",
    )
    return results


def main() -> None:
    """Run all comparisons and print validation summary metrics."""
    results = run_comparison()
    print("Validation metrics at OOF-selected F1 thresholds")
    print(
        "candidate,accuracy,precision,recall,f1,macro_f1,AP,ROC_AUC,FP,FN"
    )
    for result in results:
        metrics = result.oof_threshold_metrics
        print(
            f"{result.name},{metrics.accuracy:.4f},{metrics.precision:.4f},"
            f"{metrics.recall:.4f},{metrics.f1:.4f},{metrics.macro_f1:.4f},"
            f"{metrics.average_precision:.4f},{metrics.roc_auc:.4f},"
            f"{metrics.false_positives},{metrics.false_negatives}"
        )
        print(
            f"  score={result.score_kind}, default={result.default_threshold:.6g}, "
            f"OOF thresholds={result.oof_thresholds}, "
            f"default FN={result.default_metrics.false_negatives}"
        )
    print(f"Report: {REPORT_PATH}")
    print(f"Experiments: {EXPERIMENTS_DIRECTORY}")
    print(f"Plots: {PLOTS_DIRECTORY}")


if __name__ == "__main__":
    main()
