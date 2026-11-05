"""Minimal message validation and whitespace normalization utilities."""


def normalize_message(message: object) -> str:
    """Trim a message and collapse whitespace without altering its content.

    Punctuation, numbers, URLs, currency symbols, capitalization, and stopwords
    are deliberately preserved. The caller should keep the original message
    separately when normalized text is needed for inspection.
    """
    if not isinstance(message, str):
        raise TypeError("Message must be a string.")

    return " ".join(message.split())
