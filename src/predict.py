"""Reusable SMS prediction interface for the saved final model bundle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final" / "final_model.joblib"
VALID_LABELS = {"ham", "spam"}


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model pipeline, its saved threshold, and provenance metadata."""

    pipeline: Pipeline
    threshold: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PredictionResult:
    """Prediction details; spam_probability is an estimate, not a guarantee."""

    message: str
    prediction: str
    spam_probability: float
    threshold: float


@lru_cache(maxsize=4)
def _load_model_bundle_cached(model_path: str) -> ModelBundle:
    """Load and validate a bundle once per resolved artifact path."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Final model bundle not found: {path}")

    raw_bundle = joblib.load(path)
    if not isinstance(raw_bundle, dict):
        raise ValueError("Final model artifact must contain a bundle dictionary.")
    if not {"pipeline", "threshold", "metadata"}.issubset(raw_bundle):
        raise ValueError(
            "Final model bundle must contain pipeline, threshold, and metadata."
        )

    pipeline = raw_bundle["pipeline"]
    if not isinstance(pipeline, Pipeline):
        raise ValueError("The model bundle's pipeline is not a scikit-learn Pipeline.")
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError("The saved pipeline does not support predict_proba.")
    classes = list(getattr(pipeline, "classes_", []))
    if not VALID_LABELS.issubset(classes):
        raise ValueError("The saved pipeline must support both ham and spam classes.")

    threshold_value = raw_bundle["threshold"]
    if isinstance(threshold_value, bool) or not isinstance(
        threshold_value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("The saved classification threshold must be numeric.")
    threshold = float(threshold_value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid saved probability threshold: {threshold_value!r}")

    metadata = raw_bundle["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("The model bundle metadata must be a dictionary.")
    metadata_threshold = metadata.get("decision_threshold")
    if metadata_threshold is not None and float(metadata_threshold) != threshold:
        raise ValueError(
            "Bundle threshold does not match the threshold recorded in metadata."
        )

    return ModelBundle(pipeline=pipeline, threshold=threshold, metadata=metadata)


def load_model_bundle(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> ModelBundle:
    """Load the final model bundle, caching it by its resolved file path.

    The default artifact location is derived from this module's path, so it
    works independently of the caller's current working directory.
    """
    resolved_path = Path(model_path).expanduser().resolve()
    return _load_model_bundle_cached(str(resolved_path))


def predict_message(
    message: str,
    model_bundle: ModelBundle | None = None,
) -> PredictionResult:
    """Classify one SMS using the saved pipeline and threshold.

    The original message is passed directly into the fitted pipeline. The
    function rejects empty input and does not coerce arbitrary values to text.
    """
    if not isinstance(message, str):
        raise TypeError("Message must be a string.")
    if not message.strip():
        raise ValueError("Message must not be empty or whitespace-only.")

    bundle = model_bundle if model_bundle is not None else load_model_bundle()
    if not hasattr(bundle.pipeline, "predict_proba"):
        raise ValueError("The supplied model pipeline does not support predict_proba.")

    classes = list(getattr(bundle.pipeline, "classes_", []))
    if "spam" not in classes:
        raise ValueError("The supplied model pipeline has no spam class.")
    probabilities = np.asarray(bundle.pipeline.predict_proba([message]))
    if probabilities.shape != (1, len(classes)):
        raise ValueError("The model returned an unexpected probability array shape.")

    spam_probability = float(probabilities[0, classes.index("spam")])
    if not np.isfinite(spam_probability) or not 0.0 <= spam_probability <= 1.0:
        raise ValueError("The model returned an invalid spam probability.")
    threshold = float(bundle.threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid classification threshold: {bundle.threshold!r}")

    prediction = "spam" if spam_probability >= threshold else "ham"
    return PredictionResult(
        message=message,
        prediction=prediction,
        spam_probability=spam_probability,
        threshold=threshold,
    )
