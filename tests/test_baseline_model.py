from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src import train_baseline
from src.train_baseline import build_pipeline, load_training_data, train_and_evaluate


def make_training_data() -> pd.DataFrame:
    ham_messages = [
        f"meeting schedule lunch office project update {number}"
        for number in range(50)
    ]
    spam_messages = [
        f"claim free prize winner call now offer {number}"
        for number in range(20)
    ]
    return pd.DataFrame(
        {
            "label": ["ham"] * len(ham_messages) + ["spam"] * len(spam_messages),
            "message": ham_messages + spam_messages,
        }
    )


def test_pipeline_contains_configured_tfidf_and_multinomial_nb() -> None:
    pipeline = build_pipeline()

    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["classifier"], MultinomialNB)
    vectorizer = pipeline.named_steps["tfidf"]
    assert vectorizer.lowercase is True
    assert vectorizer.strip_accents == "unicode"
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.max_df == 0.95
    assert vectorizer.sublinear_tf is True
    assert pipeline.named_steps["classifier"].alpha == 1.0


def test_training_completes_and_returns_metrics(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    make_training_data().to_csv(train_path, index=False)

    result = train_and_evaluate(
        train_path,
        tmp_path / "model.joblib",
        tmp_path / "report.md",
    )

    assert result.training_size + result.validation_size == 70
    assert result.training_size == 56
    assert result.validation_size == 14
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1_score <= 1.0
    assert len(result.confusion) == 2
    assert all(len(row) == 2 for row in result.confusion)


def test_predictions_and_probabilities_are_valid(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    make_training_data().to_csv(train_path, index=False)
    result = train_and_evaluate(
        train_path,
        tmp_path / "model.joblib",
        tmp_path / "report.md",
    )

    messages = ["free prize claim now", "meeting lunch schedule"]
    predictions = result.pipeline.predict(messages)
    probabilities = result.pipeline.predict_proba(messages)

    assert set(predictions).issubset({"ham", "spam"})
    assert probabilities.shape == (2, 2)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_model_can_be_saved_loaded_and_classify_new_message(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    model_path = tmp_path / "nested" / "baseline.joblib"
    make_training_data().to_csv(train_path, index=False)
    train_and_evaluate(train_path, model_path, tmp_path / "report.md")

    loaded = joblib.load(model_path)
    prediction = loaded.predict(["claim your free prize now"])

    assert isinstance(loaded.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(loaded.named_steps["classifier"], MultinomialNB)
    assert prediction[0] in {"ham", "spam"}


def test_training_reads_only_training_input_never_test_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    make_training_data().to_csv(train_path, index=False)
    pd.DataFrame(
        {"label": ["spam"], "message": ["test-only secret example"]}
    ).to_csv(test_path, index=False)

    read_paths: list[Path] = []
    original_read_csv = train_baseline.pd.read_csv

    def recording_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read_paths.append(Path(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(train_baseline.pd, "read_csv", recording_read_csv)
    result = train_and_evaluate(
        train_path,
        tmp_path / "model.joblib",
        tmp_path / "report.md",
    )

    assert result.training_size + result.validation_size == len(make_training_data())
    assert read_paths == [train_path]
    assert test_path not in read_paths


def test_training_data_validation_rejects_invalid_labels(tmp_path: Path) -> None:
    train_path = tmp_path / "invalid.csv"
    pd.DataFrame(
        {"label": ["ham", "spam", "other"], "message": ["a", "b", "c"]}
    ).to_csv(train_path, index=False)

    with pytest.raises(ValueError, match="invalid labels"):
        load_training_data(train_path)


def test_report_contains_metric_and_false_negative_explanations(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "train.csv"
    report_path = tmp_path / "report.md"
    make_training_data().to_csv(train_path, index=False)

    train_and_evaluate(train_path, tmp_path / "model.joblib", report_path)
    report = report_path.read_text(encoding="utf-8")

    assert "Accuracy" in report
    assert "Precision" in report
    assert "Recall" in report
    assert "F1-score" in report
    assert "False negative (FN)" in report
    assert "data/processed/test.csv" in report
