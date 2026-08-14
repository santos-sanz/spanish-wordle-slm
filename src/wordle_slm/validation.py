from __future__ import annotations

import hashlib
import json
from typing import Any

from transformers import AutoTokenizer

from .core import feedback_code, normalize_word
from .data import PROCESSED_DIR, ROOT, TRAINING_DIR, load_processed
from .solver import WordleSolver


def validate_all(*, oracle: bool = True) -> dict[str, Any]:
    answers, guesses, splits, matrix = load_processed()
    # The historical source contains 617 rows, with "apoyo" duplicated once.
    if len(answers) != 616:
        raise AssertionError(f"expected 616 unique normalized answers, got {len(answers)}")
    if len(guesses) != len(set(guesses)) or len(answers) != len(set(answers)):
        raise AssertionError("word lists contain duplicates")
    if not set(answers).issubset(guesses):
        raise AssertionError("every answer must be a valid guess")
    if any(normalize_word(word) != word for word in [*answers, *guesses]):
        raise AssertionError("word lists contain non-normalized entries")
    split_sets = {name: set(words) for name, words in splits.items()}
    if (split_sets["train"] & split_sets["valid"]) or (split_sets["train"] & split_sets["test"]) or (split_sets["valid"] & split_sets["test"]):
        raise AssertionError("data splits overlap")
    if set().union(*split_sets.values()) != set(answers):
        raise AssertionError("data splits do not cover the answer set")
    if matrix.shape != (len(guesses), len(answers)):
        raise AssertionError("feedback matrix shape is invalid")
    for guess_index, guess in enumerate(guesses):
        for answer_index, answer in enumerate(answers):
            if int(matrix[guess_index, answer_index]) != feedback_code(answer, guess):
                raise AssertionError(f"matrix mismatch for {answer}/{guess}")
    result: dict[str, Any] = {
        "answers": len(answers),
        "guesses": len(guesses),
        "splits": {name: len(words) for name, words in splits.items()},
        "matrix_sha256": hashlib.sha256((PROCESSED_DIR / "feedback.npy").read_bytes()).hexdigest(),
        "matrix_exhaustive_check": True,
    }
    if oracle:
        oracle_result = WordleSolver().validate_oracle(
            ROOT / "artifacts" / "oracle-validation.json"
        )
        if oracle_result["failures"]:
            raise AssertionError(f"oracle failed: {oracle_result['failures']}")
        result["oracle"] = oracle_result
    model_dir = ROOT / "models" / "LFM2.5-2.6B-MLX-6bit"
    if model_dir.joinpath("tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenization: dict[str, Any] = {}
        for split in ("train", "valid"):
            zero_supervised = 0
            over_limit = 0
            longest = 0
            records = 0
            with TRAINING_DIR.joinpath(f"{split}.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    messages = record["messages"]
                    tools = record.get("tools")
                    tokens = tokenizer.apply_chat_template(
                        messages, tools=tools, return_dict=False
                    )
                    offset = len(
                        tokenizer.apply_chat_template(
                            messages[:-1],
                            tools=tools,
                            add_generation_prompt=messages[-1].get("role") == "assistant",
                            return_dict=False,
                        )
                    )
                    supervised = max(0, min(len(tokens), 512) - min(offset, 512))
                    zero_supervised += supervised == 0
                    over_limit += len(tokens) > 512
                    longest = max(longest, len(tokens))
                    records += 1
            if zero_supervised or over_limit:
                raise AssertionError(
                    f"{split} has {zero_supervised} zero-loss or {over_limit} over-limit records"
                )
            tokenization[split] = {
                "records": records,
                "zero_supervised": zero_supervised,
                "over_512": over_limit,
                "longest_tokens": longest,
            }
        result["training_tokenization"] = tokenization
    output = ROOT / "artifacts" / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
