import hashlib
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import predict
from src.predict import ModelBundle, PredictionResult, load_model_bundle, predict_message


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "final" / "final_model.joblib"
METADATA_PATH = ROOT / "models" / "final" / "model_metadata.json"
EXPECTED_THRESHOLD = 0.5434265976102519


class MockPipeline:
    classes_ = np.array(["ham", "spam"])

    def __init__(self, spam_probability: float) -> None:
        self.spam_probability = spam_probability
        self.calls: list[list[str]] = []

    def predict_proba(self, messages: list[str]) -> np.ndarray:
        self.calls.append(messages)
        return np.array([[1.0 - self.spam_probability, self.spam_probability]])


def mock_bundle(probability: float, threshold: float = EXPECTED_THRESHOLD) -> ModelBundle:
    return ModelBundle(
        pipeline=MockPipeline(probability),  # type: ignore[arg-type]
        threshold=threshold,
        metadata={"decision_threshold": threshold},
    )


def test_actual_final_bundle_loads_with_expected_pipeline_and_threshold() -> None:
    bundle = load_model_bundle(MODEL_PATH)

    assert isinstance(bundle.pipeline, Pipeline)
    assert isinstance(bundle.pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(bundle.pipeline.named_steps["classifier"], LogisticRegression)
    assert bundle.pipeline.named_steps["classifier"].class_weight == "balanced"
    assert bundle.threshold == EXPECTED_THRESHOLD
    assert bundle.metadata["decision_threshold"] == EXPECTED_THRESHOLD


def test_bundle_loading_is_cached_for_reuse() -> None:
    predict._load_model_bundle_cached.cache_clear()

    first = load_model_bundle(MODEL_PATH)
    second = load_model_bundle(MODEL_PATH)

    assert first is second
    assert predict._load_model_bundle_cached.cache_info().hits >= 1


@pytest.mark.parametrize(
    "message",
    [
        "Congratulations! You won a prize. Call 09012345678 to claim.",
        "Free reward available now: https://example.com/reward",
        "URGENT: verify your account and send your personal details today.",
        "Hi, are we still meeting at 6?",
        "Reminder: your dentist appointment is tomorrow at 10:30.",
        "Your delivery is out for delivery. Track it in the usual app.",
    ],
)
def test_realistic_sms_examples_return_valid_predictions(message: str) -> None:
    result = predict_message(message)

    assert result.message == message
    assert result.prediction in {"ham", "spam"}
    assert 0.0 <= result.spam_probability <= 1.0
    assert result.threshold == EXPECTED_THRESHOLD


def test_result_contains_required_fields_and_preserves_input() -> None:
    message = "  Keep punctuation!!! £5 https://example.test/a?b=2  "

    result = predict_message(message, mock_bundle(0.7))

    assert isinstance(result, PredictionResult)
    assert result.message == message
    assert result.prediction == "spam"
    assert result.spam_probability == pytest.approx(0.7)
    assert result.threshold == EXPECTED_THRESHOLD


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(EXPECTED_THRESHOLD, "spam"), (0.9, "spam"), (0.1, "ham")],
)
def test_prediction_uses_saved_threshold_rule(
    probability: float, expected: str
) -> None:
    result = predict_message("sample SMS", mock_bundle(probability))

    assert result.prediction == expected
    assert result.threshold == EXPECTED_THRESHOLD


@pytest.mark.parametrize("message", ["", " ", "\t\n  "])
def test_empty_or_whitespace_only_input_is_rejected(message: str) -> None:
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        predict_message(message)


@pytest.mark.parametrize("message", [None, 42, 3.14, ["SMS"], object()])
def test_non_string_input_is_rejected_without_coercion(message: object) -> None:
    with pytest.raises(TypeError, match="Message must be a string"):
        predict_message(message)  # type: ignore[arg-type]


def test_short_and_special_character_messages_are_handled() -> None:
    messages = [
        "Hi",
        "£5!!! https://example.test/?x=42",
        "Call 12345!!!",
        "你好 😊 — £€¥",
        "Reminder: " + ("your appointment details are available. " * 100),
    ]

    for message in messages:
        result = predict_message(message)
        assert result.message == message
        assert result.prediction in {"ham", "spam"}
        assert 0.0 <= result.spam_probability <= 1.0
        assert result.threshold == EXPECTED_THRESHOLD


def test_repeated_classification_is_consistent() -> None:
    message = "Your delivery is out for delivery. Track it in the usual app."

    first = predict_message(message)
    second = predict_message(message)

    assert first == second


def test_threshold_and_probability_validation_for_controlled_pipeline() -> None:
    invalid_probability_bundle = mock_bundle(1.2)
    with pytest.raises(ValueError, match="invalid spam probability"):
        predict_message("message", invalid_probability_bundle)

    invalid_threshold_bundle = mock_bundle(0.6, threshold=1.1)
    with pytest.raises(ValueError, match="Invalid classification threshold"):
        predict_message("message", invalid_threshold_bundle)


def test_prediction_does_not_fit_or_modify_saved_model() -> None:
    before = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    bundle = load_model_bundle(MODEL_PATH)
    original_fit = Pipeline.fit
    fit = Mock(side_effect=AssertionError("Prediction must not fit a model"))
    Pipeline.fit = fit  # type: ignore[method-assign]
    try:
        result = predict_message("A message for inference only", bundle)
    finally:
        Pipeline.fit = original_fit  # type: ignore[method-assign]
    after = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    assert result.prediction in {"ham", "spam"}
    assert fit.call_count == 0
    assert after == before


def test_default_artifact_path_is_independent_of_current_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict._load_model_bundle_cached.cache_clear()
    monkeypatch.chdir(ROOT.parent)

    bundle = load_model_bundle()

    assert bundle.threshold == EXPECTED_THRESHOLD
    assert bundle.pipeline.named_steps["classifier"].class_weight == "balanced"


def test_model_files_are_not_rewritten_by_prediction_calls() -> None:
    paths = [MODEL_PATH, METADATA_PATH]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    predict_message("Free entry! Claim your prize now.")

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before
