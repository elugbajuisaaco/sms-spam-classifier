from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest

from src import predict


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_imports_and_renders_main_interface() -> None:
    app = AppTest.from_file(str(APP_PATH)).run()

    assert not app.exception
    assert app.title[0].value == "SMS Spam Classifier"
    assert app.text_area[0].label == "SMS message"
    assert app.button[0].label == "Classify Message"


def test_empty_message_shows_helpful_validation() -> None:
    prediction = Mock(side_effect=ValueError("Message must not be empty or whitespace-only."))
    from src import predict as predict_module

    original = predict_module.predict_message
    predict_module.predict_message = prediction
    try:
        app = AppTest.from_file(str(APP_PATH)).run()
        app.button[0].click().run()
    finally:
        predict_module.predict_message = original

    prediction.assert_called_once_with("")
    assert any("Please enter an SMS message" in item.value for item in app.error)
    assert not app.exception


def test_valid_message_calls_prediction_engine_and_displays_result(
    monkeypatch,
) -> None:
    prediction = Mock(
        return_value=SimpleNamespace(
            prediction="spam",
            spam_probability=0.8765,
            threshold=0.5434265976102519,
            message="Claim a prize!",
        )
    )
    monkeypatch.setattr(predict, "predict_message", prediction)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("Claim a prize!").run()
    app.button[0].click().run()

    prediction.assert_called_once_with("Claim a prize!")
    assert any("SPAM" in item.value for item in app.error)
    assert app.metric[0].label == "Estimated spam probability"
    assert app.metric[0].value == "87.65%"
    assert any("not a guarantee" in item.value for item in app.caption)
    assert not app.exception


def test_ham_prediction_is_displayed_and_result_persists_on_rerun(
    monkeypatch,
) -> None:
    prediction = Mock(
        return_value=SimpleNamespace(
            prediction="ham",
            spam_probability=0.12,
            threshold=0.5434265976102519,
            message="See you soon.",
        )
    )
    monkeypatch.setattr(predict, "predict_message", prediction)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("See you soon.").run()
    app.button[0].click().run()

    assert any("HAM" in item.value for item in app.success)
    assert app.metric[0].value == "12.00%"
    app.run()
    assert any("HAM" in item.value for item in app.success)
    assert app.metric[0].value == "12.00%"
    prediction.assert_called_once_with("See you soon.")


def test_unexpected_prediction_error_is_logged_and_hidden_from_ui(
    monkeypatch, caplog
) -> None:
    prediction = Mock(side_effect=RuntimeError("internal detail"))
    monkeypatch.setattr(predict, "predict_message", prediction)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("Any message").run()
    with caplog.at_level("ERROR"):
        app.button[0].click().run()

    assert any("could not be classified" in item.value for item in app.error)
    assert not any("internal detail" in item.value for item in app.error)
    assert "internal detail" in caplog.text
    assert not app.exception


def test_whitespace_message_shows_validation_error(monkeypatch) -> None:
    prediction = Mock(side_effect=ValueError("Message must not be empty or whitespace-only."))
    monkeypatch.setattr(predict, "predict_message", prediction)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value(" \t ").run()
    app.button[0].click().run()

    prediction.assert_called_once_with(" \t ")
    assert any("Please enter an SMS message" in item.value for item in app.error)
    assert not app.exception
