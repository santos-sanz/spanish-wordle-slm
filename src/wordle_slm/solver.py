from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .core import ALL_GREEN
from .data import load_processed


class WordleSolver:
    def __init__(self) -> None:
        self.answers, self.guesses, self.splits, self.matrix = load_processed()
        self.answer_index = {word: index for index, word in enumerate(self.answers)}
        self.guess_index = {word: index for index, word in enumerate(self.guesses)}
        self._all_candidates = tuple(range(len(self.answers)))
        self._best_guess_cache: dict[tuple[tuple[int, ...], int], str] = {}

    @property
    def all_candidates(self) -> tuple[int, ...]:
        return self._all_candidates

    def apply_feedback(
        self, candidates: Iterable[int], guess: str, feedback: int
    ) -> tuple[int, ...]:
        row = self.matrix[self.guess_index[guess]]
        return tuple(index for index in candidates if int(row[index]) == feedback)

    def candidates_from_history(self, history: Iterable[tuple[str, int]]) -> tuple[int, ...]:
        candidates = self._all_candidates
        for guess, feedback in history:
            candidates = self.apply_feedback(candidates, guess, feedback)
        return candidates

    def candidate_words(self, history: Iterable[tuple[str, int]]) -> list[str]:
        return [self.answers[index] for index in self.candidates_from_history(history)]

    def best_guess(self, candidates: tuple[int, ...], remaining_turns: int = 6) -> str:
        cache_key = (candidates, remaining_turns)
        if cache_key in self._best_guess_cache:
            return self._best_guess_cache[cache_key]
        if not candidates:
            raise ValueError("candidate set is empty")
        if len(candidates) == 1:
            return self.answers[candidates[0]]
        candidate_set = set(candidates)
        best_key: tuple[float, ...] | None = None
        best_word = ""
        for guess_index, guess in enumerate(self.guesses):
            counts = np.bincount(self.matrix[guess_index, list(candidates)], minlength=243)
            nonzero = counts[counts > 0].astype(np.float64)
            probabilities = nonzero / len(candidates)
            entropy = float(-np.sum(probabilities * np.log2(probabilities)))
            worst = int(nonzero.max())
            expected = float(np.sum(nonzero * nonzero) / len(candidates))
            is_candidate = self.answer_index.get(guess) in candidate_set
            if remaining_turns <= 2:
                key = (worst, expected, -float(is_candidate), -entropy)
            else:
                key = (-entropy, worst, expected, -float(is_candidate))
            if best_key is None or key < best_key or (key == best_key and guess < best_word):
                best_key = key
                best_word = guess
        if len(self._best_guess_cache) >= 100_000:
            self._best_guess_cache.pop(next(iter(self._best_guess_cache)))
        self._best_guess_cache[cache_key] = best_word
        return best_word

    def play_oracle(self, target: str, max_turns: int = 6) -> list[tuple[str, int]]:
        target_index = self.answer_index[target]
        candidates = self._all_candidates
        history: list[tuple[str, int]] = []
        for turn in range(max_turns):
            guess = self.best_guess(candidates, max_turns - turn)
            code = int(self.matrix[self.guess_index[guess], target_index])
            history.append((guess, code))
            if code == ALL_GREEN:
                break
            candidates = self.apply_feedback(candidates, guess, code)
        return history

    def validate_oracle(self, output: Path | None = None) -> dict[str, object]:
        failures: list[str] = []
        turns: list[int] = []
        for answer in self.answers:
            game = self.play_oracle(answer)
            solved = bool(game and game[-1][1] == ALL_GREEN)
            if not solved:
                failures.append(answer)
            turns.append(len(game) if solved else 7)
        result: dict[str, object] = {
            "answers": len(self.answers),
            "solved": len(self.answers) - len(failures),
            "failures": failures,
            "mean_turns": sum(turns) / len(turns),
            "max_turns": max(turns),
        }
        if output:
            import json

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def entropy_for_counts(counts: Iterable[int]) -> float:
    values = [value for value in counts if value > 0]
    total = sum(values)
    return -sum((value / total) * math.log2(value / total) for value in values)
