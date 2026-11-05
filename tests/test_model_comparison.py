from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src import compare_models
from src.compare_models import (
    CANDIDATE_CONFIGS,
    build_candidate,
    evaluate_threshold,
    fit_and_evaluate_candidate,
    run_comparison,
)


def make_dataset() -> pd.DataFrame:
    ham = [
        f"ordinary family meeting lunch schedule note number {i}"
        for i in range(60)
    ]
    spam = [
        f"claim free prize winner call now offer number {i}"
        for i in range(30)
    ]
    return pd.DataFrame(
        {"label": ["ham"] * len(ham) + ["spam"] * len(spam), "message": ham + spam}
    )


def make_train_validation_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = make_dataset()
    train, validation = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True)


@pytest.mark.parametrize(
    ("name", "classifier_type"),
    [
        ("multinomial_nb_default", MultinomialNB),
        ("multinomial_nb_uniform_prior", MultinomialNB),
        ("logistic_regression_default", LogisticRegression),
        ("logistic_regression_balanced", LogisticRegression),
        ("linear_svm_default", LinearSVC),
        ("linear_svm_balanced", LinearSVC),
    ],
)
def test_each_candidate_pipeline_has_expected_steps(
    name: str, classifier_type: type
) -> None:
    pipeline = build_candidate(name)

    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["classifier"], classifier_type)
    assert CANDIDATE_CONFIGS[name]["algorithm"] == classifier_type.__name__


@pytest.mark.parametrize("name", list(CANDIDATE_CONFIGS))
def test_each_algorithm_trains_and_predicts_valid_labels(name: str) -> None:
    train, validation = make_train_validation_data()
    result = fit_and_evaluate_candidate(
        name,
        train.message,
        train.label,
        validation.message,
        validation.label,
        cv_splits=3,
    )

    predictions = result.pipeline.predict(validation.message)
    assert set(predictions).issubset({"ham", "spam"})
    assert result.default_metrics.confusion_matrix


@pytest.mark.parametrize(
    "name",
    [
        "multinomial_nb_default",
        "multinomial_nb_uniform_prior",
        "logistic_regression_default",
        "logistic_regression_balanced",
    ],
)
def test_probability_outputs_are_valid(name: str) -> None:
    train, validation = make_train_validation_data()
    result = fit_and_evaluate_candidate(
        name,
        train.message,
        train.label,
        validation.message,
        validation.label,
        cv_splits=3,
    )

    probabilities = result.pipeline.predict_proba(validation.message)
    assert result.score_kind == "probability"
    assert probabilities.shape == (len(validation), 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", ["linear_svm_default", "linear_svm_balanced"])
def test_svm_decision_scores_are_generated(name: str) -> None:
    train, validation = make_train_validation_data()
    result = fit_and_evaluate_candidate(
        name,
        train.message,
        train.label,
        validation.message,
        validation.label,
        cv_splits=3,
    )

    scores = result.pipeline.decision_function(validation.message)
    assert result.score_kind == "decision_function"
    assert scores.shape == (len(validation),)
    assert np.isfinite(scores).all()


def test_threshold_evaluation_reports_metrics_and_confusion() -> None:
    labels = pd.Series(["ham", "ham", "spam", "spam"])
    scores = np.array([0.1, 0.7, 0.4, 0.8])

    metrics = evaluate_threshold(labels, scores, 0.5)

    assert metrics.confusion_matrix == [[1, 1], [1, 1]]
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)


def test_run_comparison_serializes_candidates_and_never_reads_test_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.csv"
    forbidden_test_path = tmp_path / "test.csv"
    make_dataset().to_csv(train_path, index=False)
    pd.DataFrame(
        {"label": ["spam"], "message": ["test data must not be read"]}
    ).to_csv(forbidden_test_path, index=False)

    read_paths: list[Path] = []
    original_read_csv = compare_models.pd.read_csv

    def recording_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read_paths.append(Path(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(compare_models.pd, "read_csv", recording_read_csv)
    experiment_directory = tmp_path / "experiments"
    results = run_comparison(
        train_path,
        experiment_directory,
        tmp_path / "report.md",
        tmp_path / "plots",
        cv_splits=3,
    )

    assert len(results) == 6
    assert read_paths == [train_path]
    assert forbidden_test_path not in read_paths
    for name in CANDIDATE_CONFIGS:
        path = experiment_directory / f"{name}.joblib"
        loaded = joblib.load(path)
        assert isinstance(loaded.named_steps["tfidf"], TfidfVectorizer)
        assert loaded.predict(["claim free prize now"])[0] in {"ham", "spam"}
    assert (experiment_directory / "candidate_configurations.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "plots" / "precision_recall_curves.png").is_file()


def test_threshold_experiment_grid_is_configured() -> None:
    assert compare_models.PROBABILITY_THRESHOLDS == (
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
    )
