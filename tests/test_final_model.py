import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import train_final_model
from src.train_final_model import (
    build_final_pipeline,
    load_manifest,
    train_and_evaluate_once,
)


def make_dataset() -> pd.DataFrame:
    ham = [
        f"family lunch meeting schedule note number {i}" for i in range(30)
    ]
    spam = [
        f"claim free prize winner call now offer number {i}" for i in range(15)
    ]
    return pd.DataFrame(
        {"label": ["ham"] * len(ham) + ["spam"] * len(spam), "message": ham + spam}
    )


def write_manifest(path: Path) -> float:
    threshold = 0.5434265976102519
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tfidf": {
                    "lowercase": True,
                    "strip_accents": "unicode",
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_df": 0.95,
                    "sublinear_tf": True,
                },
                "candidates": {
                    "logistic_regression_balanced": {
                        "algorithm": "LogisticRegression",
                        "parameters": {
                            "class_weight": "balanced",
                            "max_iter": 1000,
                            "random_state": 42,
                        },
                        "oof_selected_thresholds": {
                            "OOF best F1": threshold,
                        },
                        "validation_metrics_at_oof_f1_threshold": {
                            "accuracy": 0.9794685990338164,
                            "precision": 0.9230769230769231,
                            "recall": 0.9142857142857143,
                            "f1": 0.9186602870813397,
                            "false_positives": 8,
                            "false_negatives": 9,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return threshold


def test_manifest_threshold_and_configuration_are_loaded_exactly(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    expected_threshold = write_manifest(manifest_path)

    selection = load_manifest(manifest_path)
    pipeline = build_final_pipeline(
        selection["vectorizer"], selection["classifier"]
    )

    assert selection["threshold"] == expected_threshold
    assert selection["threshold_name"] == "OOF best F1"
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert pipeline.named_steps["classifier"].class_weight == "balanced"
    assert pipeline.named_steps["tfidf"].ngram_range == (1, 2)


def test_invalid_manifest_threshold_is_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidates"]["logistic_regression_balanced"][
        "oof_selected_thresholds"
    ]["OOF best F1"] = 1.5
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid probability"):
        load_manifest(manifest_path)


def test_final_training_uses_train_only_and_test_for_one_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    manifest_path = tmp_path / "experiments" / "candidate_configurations.json"
    final_directory = tmp_path / "final"
    report_path = tmp_path / "reports" / "final.md"
    plot_path = tmp_path / "plots" / "confusion.png"
    train = make_dataset()
    test = pd.DataFrame(
        {
            "label": ["ham"] * 8 + ["spam"] * 4,
            "message": [f"held out family note {i}" for i in range(8)]
            + [f"urgent free claim prize {i}" for i in range(4)],
        }
    )
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    threshold = write_manifest(manifest_path)

    read_paths: list[Path] = []
    original_read_csv = train_final_model.pd.read_csv

    def recording_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read_paths.append(Path(path))
        return original_read_csv(path, *args, **kwargs)

    fit_inputs: list[tuple[list[str], list[str]]] = []
    original_fit = Pipeline.fit

    def recording_fit(
        self: Pipeline, X: object, y: object = None, **kwargs: object
    ) -> Pipeline:
        fit_inputs.append((list(X), list(y)))
        return original_fit(self, X, y, **kwargs)

    predict_proba_inputs: list[list[str]] = []
    original_predict_proba = Pipeline.predict_proba

    def recording_predict_proba(self: Pipeline, X: object, **kwargs: object) -> np.ndarray:
        predict_proba_inputs.append(list(X))
        return original_predict_proba(self, X, **kwargs)

    monkeypatch.setattr(train_final_model.pd, "read_csv", recording_read_csv)
    monkeypatch.setattr(Pipeline, "fit", recording_fit)
    monkeypatch.setattr(Pipeline, "predict_proba", recording_predict_proba)

    result = train_and_evaluate_once(
        train_path,
        test_path,
        manifest_path,
        final_directory,
        report_path,
        plot_path,
    )

    assert read_paths == [train_path, test_path]
    assert fit_inputs == [(train.message.tolist(), train.label.tolist())]
    assert predict_proba_inputs == [test.message.tolist()]
    assert result.threshold == threshold
    assert result.training_count == len(train)
    assert result.test_count == len(test)
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["precision"] <= 1.0
    assert 0.0 <= result.metrics["recall"] <= 1.0
    assert 0.0 <= result.metrics["f1"] <= 1.0
    assert 0.0 <= result.metrics["macro_f1"] <= 1.0
    assert 0.0 <= result.metrics["average_precision"] <= 1.0
    assert 0.0 <= result.metrics["roc_auc"] <= 1.0

    bundle = joblib.load(final_directory / "final_model.joblib")
    assert bundle["threshold"] == threshold
    assert isinstance(bundle["pipeline"], Pipeline)
    assert isinstance(bundle["pipeline"].named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(
        bundle["pipeline"].named_steps["classifier"], LogisticRegression
    )
    assert bundle["pipeline"].predict(["claim free prize"])[0] in {"ham", "spam"}
    probabilities = bundle["pipeline"].predict_proba(["claim free prize"])
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()
    assert np.isfinite(bundle["threshold"]) and 0.0 <= bundle["threshold"] <= 1.0
    metadata = json.loads(
        (final_directory / "model_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["threshold_origin"]["value"] == threshold
    assert metadata["training_data"]["sha256"] == hashlib.sha256(
        train_path.read_bytes()
    ).hexdigest()
    assert report_path.is_file()
    assert plot_path.is_file()


def test_test_input_is_not_used_for_pipeline_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    manifest_path = tmp_path / "manifest.json"
    train = make_dataset()
    test = pd.DataFrame(
        {
            "label": ["ham", "spam"],
            "message": ["test only ham content", "test only spam content"],
        }
    )
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    write_manifest(manifest_path)

    fit_inputs: list[tuple[list[str], list[str]]] = []
    original_fit = Pipeline.fit

    def recording_fit(
        self: Pipeline, X: object, y: object = None, **kwargs: object
    ) -> Pipeline:
        fit_inputs.append((list(X), list(y)))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", recording_fit)
    train_and_evaluate_once(
        train_path,
        test_path,
        manifest_path,
        tmp_path / "final",
        tmp_path / "report.md",
        tmp_path / "plot.png",
    )

    assert len(fit_inputs) == 1
    assert fit_inputs[0] == (train.message.tolist(), train.label.tolist())
    assert set(test.message).isdisjoint(set(fit_inputs[0][0]))


def test_existing_final_artifacts_are_not_overwritten(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    manifest_path = tmp_path / "manifest.json"
    train = make_dataset()
    train.to_csv(train_path, index=False)
    train.iloc[:4].to_csv(test_path, index=False)
    write_manifest(manifest_path)
    final_directory = tmp_path / "final"
    final_directory.mkdir()
    existing = final_directory / "final_model.joblib"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train_and_evaluate_once(
            train_path,
            test_path,
            manifest_path,
            final_directory,
            tmp_path / "report.md",
            tmp_path / "plot.png",
        )

    assert existing.read_text(encoding="utf-8") == "keep"
