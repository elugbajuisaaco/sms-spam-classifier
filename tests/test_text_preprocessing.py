import pandas as pd
import pytest

from src.text_preprocessing import normalize_message


def test_normalization_only_trims_and_collapses_whitespace() -> None:
    message = "  Claim £500 NOW!!! Visit https://example.com/a?b=2 \n today.  "

    normalized = normalize_message(message)

    assert normalized == "Claim £500 NOW!!! Visit https://example.com/a?b=2 today."


def test_normalization_does_not_modify_original_message() -> None:
    original = "  Free entry!!! £100  "

    normalize_message(original)

    assert original == "  Free entry!!! £100  "


@pytest.mark.parametrize("message", [None, float("nan"), pd.NA, 42, ["message"]])
def test_normalization_rejects_missing_or_non_string_values(message: object) -> None:
    with pytest.raises(TypeError, match="Message must be a string"):
        normalize_message(message)


def test_normalization_handles_empty_or_whitespace_only_strings() -> None:
    assert normalize_message("") == ""
    assert normalize_message(" \t\n ") == ""
