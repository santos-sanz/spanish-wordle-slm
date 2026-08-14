import pytest

from wordle_slm.core import (
    decode_feedback,
    encode_feedback,
    feedback_digits,
    normalize_word,
)


def test_normalization_preserves_enye_and_removes_accents() -> None:
    assert normalize_word("CAÑÓN") == "cañon"
    assert normalize_word("ca├▒on") == "cañon"
    assert normalize_word("ACRO-") is None
    assert normalize_word("cuatro") is None


def test_duplicate_letters_are_consumed_once() -> None:
    assert feedback_digits("cacao", "anana") == (1, 0, 1, 0, 0)
    assert feedback_digits("perro", "error") == (1, 1, 2, 1, 0)


def test_feedback_encoding_round_trip() -> None:
    for code in range(243):
        assert encode_feedback(decode_feedback(code)) == code


def test_invalid_word_is_rejected() -> None:
    with pytest.raises(ValueError):
        feedback_digits("casa", "perro")
