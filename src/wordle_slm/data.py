from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .core import feedback_code, normalize_word

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
TRAINING_DIR = ROOT / "data" / "training"

ANSWERS_COMMIT = "093fe54b5425beb8ab8e4599256f6aee2e405938"
GUESSES_COMMIT = "c85f31b94b0805cace081b51c983315bc66302ee"
ANSWERS_URL = (
    "https://raw.githubusercontent.com/adrian154/blog/"
    f"{ANSWERS_COMMIT}/public/blogposts/wordle-es-wordlist/wordle-words.txt"
)
GUESSES_URL = (
    "https://raw.githubusercontent.com/cjsaavedra76/WORDLE-ES-resolver_csr/"
    f"{GUESSES_COMMIT}/wordle-es-resolver_csr.py"
)
SPLIT_SEED = 20260814


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "spanish-wordle-slm/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _ordered_unique(words: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(words))


def parse_answers(payload: bytes) -> tuple[list[str], int]:
    text = payload.decode("utf-8")
    raw = [line.strip() for line in text.splitlines() if line.strip()]
    normalized = [word for item in raw if (word := normalize_word(item)) is not None]
    return _ordered_unique(normalized), len(raw)


def parse_guesses(payload: bytes) -> tuple[list[str], int]:
    text = payload.decode("utf-8")
    match = re.search(r"\n\s*string\s*=\s*('(?:[^'\\]|\\.)*')", text, re.DOTALL)
    if not match:
        raise ValueError("could not locate the embedded historical guess list")
    raw_string = ast.literal_eval(match.group(1))
    raw = raw_string.split()
    normalized = [word for item in raw if (word := normalize_word(item)) is not None]
    return _ordered_unique(normalized), len(raw)


def deterministic_split(answers: list[str]) -> dict[str, list[str]]:
    shuffled = answers.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)
    train_end = int(len(shuffled) * 0.70)
    valid_end = train_end + int(len(shuffled) * 0.10)
    return {
        "train": sorted(shuffled[:train_end]),
        "valid": sorted(shuffled[train_end:valid_end]),
        "test": sorted(shuffled[valid_end:]),
    }


def build_matrix(guesses: list[str], answers: list[str]) -> np.ndarray:
    matrix = np.empty((len(guesses), len(answers)), dtype=np.uint8)
    for guess_index, guess in enumerate(guesses):
        matrix[guess_index] = np.fromiter(
            (feedback_code(answer, guess) for answer in answers),
            dtype=np.uint8,
            count=len(answers),
        )
    return matrix


def prepare_data() -> dict[str, object]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    answer_payload = _download(ANSWERS_URL)
    guess_payload = _download(GUESSES_URL)
    answers, raw_answer_count = parse_answers(answer_payload)
    guesses, raw_guess_count = parse_guesses(guess_payload)
    if raw_answer_count != 617:
        raise ValueError(f"expected 617 raw answers, found {raw_answer_count}")
    if raw_guess_count != 11180:
        raise ValueError(f"expected 11180 raw guesses, found {raw_guess_count}")
    guesses = sorted(set(guesses).union(answers))
    answers = sorted(answers)
    splits = deterministic_split(answers)
    matrix = build_matrix(guesses, answers)

    (PROCESSED_DIR / "answers.txt").write_text("\n".join(answers) + "\n", encoding="utf-8")
    (PROCESSED_DIR / "guesses.txt").write_text("\n".join(guesses) + "\n", encoding="utf-8")
    (PROCESSED_DIR / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    np.save(PROCESSED_DIR / "feedback.npy", matrix, allow_pickle=False)
    provenance: dict[str, object] = {
        "answers": {
            "url": ANSWERS_URL,
            "commit": ANSWERS_COMMIT,
            "sha256": _sha256(answer_payload),
            "raw_count": raw_answer_count,
            "normalized_count": len(answers),
        },
        "guesses": {
            "url": GUESSES_URL,
            "commit": GUESSES_COMMIT,
            "sha256": _sha256(guess_payload),
            "raw_count": raw_guess_count,
            "normalized_union_count": len(guesses),
        },
        "split_seed": SPLIT_SEED,
        "split_counts": {name: len(words) for name, words in splits.items()},
        "matrix_shape": list(matrix.shape),
        "matrix_sha256": _sha256((PROCESSED_DIR / "feedback.npy").read_bytes()),
    }
    (PROCESSED_DIR / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def load_processed() -> tuple[list[str], list[str], dict[str, list[str]], np.ndarray]:
    answers = (PROCESSED_DIR / "answers.txt").read_text(encoding="utf-8").splitlines()
    guesses = (PROCESSED_DIR / "guesses.txt").read_text(encoding="utf-8").splitlines()
    splits = json.loads((PROCESSED_DIR / "splits.json").read_text(encoding="utf-8"))
    matrix = np.load(PROCESSED_DIR / "feedback.npy", allow_pickle=False)
    return answers, guesses, splits, matrix
