from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

WORD_LENGTH = 5
ALL_GREEN = 242  # 2 + 2*3 + 2*9 + 2*27 + 2*81
ALPHABET = frozenset("abcdefghijklmnñopqrstuvwxyz")


def normalize_word(value: str) -> str | None:
    """Normalize the historical unaccented Wordle ES representation."""
    value = value.strip().lower().replace("├▒", "ñ").replace("ã±", "ñ")
    value = unicodedata.normalize("NFC", value)
    placeholder = "\u0000"
    value = value.replace("ñ", placeholder)
    value = "".join(
        char
        for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    ).replace(placeholder, "ñ")
    value = unicodedata.normalize("NFC", value)
    if len(value) != WORD_LENGTH or any(char not in ALPHABET for char in value):
        return None
    return value


def feedback_digits(target: str, guess: str) -> tuple[int, ...]:
    """Return gray/yellow/green digits using Wordle's duplicate-letter rules."""
    if normalize_word(target) != target or normalize_word(guess) != guess:
        raise ValueError("target and guess must be normalized five-letter Spanish words")
    result = [0] * WORD_LENGTH
    remaining: Counter[str] = Counter()
    for index, (target_char, guess_char) in enumerate(zip(target, guess, strict=True)):
        if target_char == guess_char:
            result[index] = 2
        else:
            remaining[target_char] += 1
    for index, guess_char in enumerate(guess):
        if result[index] == 0 and remaining[guess_char] > 0:
            result[index] = 1
            remaining[guess_char] -= 1
    return tuple(result)


def encode_feedback(digits: Sequence[int]) -> int:
    if len(digits) != WORD_LENGTH or any(digit not in (0, 1, 2) for digit in digits):
        raise ValueError("feedback must contain exactly five ternary digits")
    return sum(digit * (3**index) for index, digit in enumerate(digits))


def decode_feedback(code: int) -> tuple[int, ...]:
    if not 0 <= code < 3**WORD_LENGTH:
        raise ValueError("feedback code must be in [0, 242]")
    digits: list[int] = []
    for _ in range(WORD_LENGTH):
        digits.append(code % 3)
        code //= 3
    return tuple(digits)


def feedback_code(target: str, guess: str) -> int:
    return encode_feedback(feedback_digits(target, guess))


def feedback_text(target: str, guess: str) -> str:
    return "".join(str(value) for value in feedback_digits(target, guess))


def history_text(history: Iterable[tuple[str, int]]) -> str:
    rows = [f"{guess} -> {''.join(map(str, decode_feedback(code)))}" for guess, code in history]
    return "Sin intentos previos." if not rows else "\n".join(rows)
