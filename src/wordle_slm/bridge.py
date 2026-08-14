from __future__ import annotations

import json
import sys
from typing import Any

from .core import feedback_code, feedback_text, normalize_word
from .solver import WordleSolver


def _history(value: list[dict[str, Any]]) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for row in value:
        guess = normalize_word(str(row["guess"]))
        if guess is None:
            raise ValueError("history contains an invalid guess")
        raw_feedback = row["feedback"]
        if isinstance(raw_feedback, str):
            if len(raw_feedback) != 5 or any(char not in "012" for char in raw_feedback):
                raise ValueError("feedback text must contain five ternary digits")
            code = sum(int(char) * (3**index) for index, char in enumerate(raw_feedback))
        else:
            code = int(raw_feedback)
        parsed.append((guess, code))
    return parsed


class Bridge:
    def __init__(self) -> None:
        self.solver = WordleSolver()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "feedback":
            target = normalize_word(str(request.get("target", "")))
            guess = normalize_word(str(request.get("guess", "")))
            if target not in self.solver.answer_index or guess not in self.solver.guess_index:
                raise ValueError("target or guess is outside the pinned lexicon")
            return {
                "code": feedback_code(target, guess),
                "feedback": feedback_text(target, guess),
            }
        if operation == "validate_word":
            word = normalize_word(str(request.get("word", "")))
            return {"valid": word in self.solver.guess_index if word else False, "word": word}
        history = _history(request.get("history", []))
        candidates = self.solver.candidates_from_history(history)
        if operation == "get_candidates":
            words = [self.solver.answers[index] for index in candidates]
            return {"count": len(words), "candidates": words}
        if operation == "best_guess":
            remaining = max(1, 6 - len(history))
            return {"guess": self.solver.best_guess(candidates, remaining)}
        raise ValueError(f"unsupported operation: {operation}")


def main() -> None:
    bridge = Bridge()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = {"id": request.get("id"), "ok": True, "result": bridge.handle(request)}
        except Exception as error:  # noqa: BLE001 - protocol returns errors instead of crashing
            result = {"id": request.get("id") if "request" in locals() else None, "ok": False, "error": str(error)}
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
