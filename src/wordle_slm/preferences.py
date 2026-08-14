from __future__ import annotations

import json
import random
from typing import Any

import numpy as np

from .data import ROOT
from .dataset import PURE_SYSTEM, _games_for_target, state_prompt
from .solver import WordleSolver

PREFERENCE_DIR = ROOT / "data" / "preferences"


def _solver_key(
    solver: WordleSolver, guess: str, candidates: tuple[int, ...], remaining_turns: int
) -> tuple[float, ...]:
    guess_index = solver.guess_index[guess]
    counts = np.bincount(solver.matrix[guess_index, list(candidates)], minlength=243)
    nonzero = counts[counts > 0].astype(np.float64)
    probabilities = nonzero / len(candidates)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    worst = int(nonzero.max())
    expected = float(np.sum(nonzero * nonzero) / len(candidates))
    is_candidate = solver.answer_index.get(guess) in set(candidates)
    if remaining_turns <= 2:
        return (worst, expected, -float(is_candidate), -entropy)
    return (-entropy, worst, expected, -float(is_candidate))


def _rejected_guess(
    solver: WordleSolver,
    candidates: tuple[int, ...],
    chosen: str,
    remaining_turns: int,
    seed: str,
) -> tuple[str, tuple[float, ...]]:
    rng = random.Random(seed)
    candidate_words = [solver.answers[index] for index in candidates]
    pool = candidate_words if len(candidate_words) <= 64 else rng.sample(candidate_words, 64)
    other_guesses = [word for word in solver.guesses if word != chosen]
    pool.extend(rng.sample(other_guesses, min(128, len(other_guesses))))
    pool = sorted(set(pool) - {chosen})
    if not pool:
        raise RuntimeError("could not construct a rejected Wordle action")
    ranked = [(_solver_key(solver, word, candidates, remaining_turns), word) for word in pool]
    rejected_key, rejected = max(ranked)
    return rejected, rejected_key


def _preference_records(solver: WordleSolver, targets: list[str]) -> list[dict[str, Any]]:
    states: dict[tuple[tuple[str, int], ...], None] = {}
    for target in targets:
        for game in _games_for_target(solver, target):
            history: list[tuple[str, int]] = []
            for guess, feedback in game:
                states.setdefault(tuple(history), None)
                history.append((guess, feedback))

    records: list[dict[str, Any]] = []
    for history_tuple in states:
        history = list(history_tuple)
        candidates = solver.candidates_from_history(history)
        remaining_turns = max(1, 6 - len(history))
        chosen = solver.best_guess(candidates, remaining_turns)
        seed = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
        rejected, rejected_key = _rejected_guess(
            solver, candidates, chosen, remaining_turns, seed
        )
        chosen_key = _solver_key(solver, chosen, candidates, remaining_turns)
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": PURE_SYSTEM},
                    {
                        "role": "user",
                        "content": state_prompt(history, len(history) + 1),
                    },
                ],
                "chosen": json.dumps({"guess": chosen}, ensure_ascii=False),
                "rejected": json.dumps({"guess": rejected}, ensure_ascii=False),
                "candidates": len(candidates),
                "chosen_solver_key": chosen_key,
                "rejected_solver_key": rejected_key,
            }
        )
    rng = random.Random(20260814)
    rng.shuffle(records)
    return records


def generate_preference_data() -> dict[str, int]:
    solver = WordleSolver()
    PREFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    records_by_split = {
        split: _preference_records(solver, solver.splits[split])
        for split in ("train", "valid")
    }
    train_prompts = {
        json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True)
        for record in records_by_split["train"]
    }
    records_by_split["valid"] = [
        record
        for record in records_by_split["valid"]
        if json.dumps(record["prompt"], ensure_ascii=False, sort_keys=True) not in train_prompts
    ]
    counts: dict[str, int] = {}
    for split, records in records_by_split.items():
        path = PREFERENCE_DIR / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        counts[split] = len(records)
    manifest = {
        "method": "offline_direct_preference_optimization",
        "seed": 20260814,
        "source_splits": ["train", "valid"],
        "hidden_test_used": False,
        "train_validation_prompt_overlap": 0,
        "counts": counts,
        "chosen": "deterministic Wordle solver action",
        "rejected": "valid action with a worse deterministic solver key",
    }
    (PREFERENCE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return counts
