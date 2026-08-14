from __future__ import annotations

import json
import random
from typing import Any

from .core import ALL_GREEN, history_text
from .data import TRAINING_DIR
from .solver import WordleSolver

GET_CANDIDATES_TOOL = {
    "type": "function",
    "function": {
        "name": "get_candidates",
        "description": "Return all Spanish Wordle answers compatible with the current history.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}
BEST_GUESS_TOOL = {
    "type": "function",
    "function": {
        "name": "best_guess",
        "description": "Return the deterministic solver's best next Spanish Wordle guess.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}

PURE_SYSTEM = (
    "Juegas Wordle en español. La palabra objetivo tiene cinco letras y dispones de seis "
    "intentos. 0=gris, 1=amarillo, 2=verde. Respeta letras repetidas. Responde únicamente "
    'con JSON válido: {"guess":"palabra"}.'
)
AGENT_SYSTEM = (
    PURE_SYSTEM
    + " Puedes usar get_candidates una vez por turno para consultar soluciones compatibles."
)
ORACLE_SYSTEM = PURE_SYSTEM + " Usa best_guess para obtener la jugada del solver."


def state_prompt(history: list[tuple[str, int]], turn: int) -> str:
    return f"Turno {turn}/6. Historial:\n{history_text(history)}\nElige el siguiente intento."


def _tool_call(name: str, call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def _tool_result(name: str, call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
    }


def _pure_record(history: list[tuple[str, int]], turn: int, guess: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": PURE_SYSTEM},
            {"role": "user", "content": state_prompt(history, turn)},
            {"role": "assistant", "content": json.dumps({"guess": guess})},
        ]
    }


def _tool_records(
    *,
    mode: str,
    history: list[tuple[str, int]],
    turn: int,
    guess: str,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    if mode == "agent":
        system, tool = AGENT_SYSTEM, GET_CANDIDATES_TOOL
    else:
        system, tool = ORACLE_SYSTEM, BEST_GUESS_TOOL
    name = tool["function"]["name"]
    call_id = f"{mode}_{turn}"
    prefix = [
        {"role": "system", "content": system},
        {"role": "user", "content": state_prompt(history, turn)},
    ]
    call = _tool_call(name, call_id)
    return [
        {"messages": prefix + [call], "tools": [tool]},
        {
            "messages": prefix
            + [
                call,
                _tool_result(name, call_id, result),
                {"role": "assistant", "content": json.dumps({"guess": guess})},
            ],
            "tools": [tool],
        },
    ]


def _games_for_target(solver: WordleSolver, target: str) -> list[list[tuple[str, int]]]:
    starters: list[str | None] = [None, "audio", "seron", "lenta"]
    games: list[list[tuple[str, int]]] = []
    target_index = solver.answer_index[target]
    for starter in starters:
        candidates = solver.all_candidates
        history: list[tuple[str, int]] = []
        for turn in range(6):
            if turn == 0 and starter in solver.guess_index:
                guess = starter
            else:
                guess = solver.best_guess(candidates, 6 - turn)
            code = int(solver.matrix[solver.guess_index[guess], target_index])
            history.append((guess, code))
            if code == ALL_GREEN:
                break
            candidates = solver.apply_feedback(candidates, guess, code)
        games.append(history)
    return games


def _records_for_targets(solver: WordleSolver, targets: list[str]) -> list[dict[str, Any]]:
    pure: list[dict[str, Any]] = []
    agent: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    for target in targets:
        for game in _games_for_target(solver, target):
            history: list[tuple[str, int]] = []
            for turn_index, (guess, code) in enumerate(game, start=1):
                pure.append(_pure_record(history, turn_index, guess))
                candidates = solver.candidate_words(history)
                agent_records = _tool_records(
                    mode="agent",
                    history=history,
                    turn=turn_index,
                    guess=guess,
                    result={"count": len(candidates), "candidates": candidates},
                )
                # The call itself remains useful for large states, but a 616-word tool
                # result would push the supervised final answer beyond 512 tokens.
                agent.extend(agent_records if len(candidates) <= 80 else agent_records[:1])
                oracle.extend(
                    _tool_records(
                        mode="oracle",
                        history=history,
                        turn=turn_index,
                        guess=guess,
                        result={"guess": guess},
                    )
                )
                history = [*history, (guess, code)]

    rng = random.Random(20260814)
    rng.shuffle(agent)
    rng.shuffle(oracle)
    # Keep all pure records; select tool records to obtain an exact 60/30/10 mix.
    records = pure + agent[: len(pure) // 2] + oracle[: len(pure) // 6]
    rng.shuffle(records)
    return records


def generate_training_data() -> dict[str, int]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    solver = WordleSolver()
    counts: dict[str, int] = {}
    for split in ("train", "valid"):
        records = _records_for_targets(solver, solver.splits[split])
        path = TRAINING_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[split] = len(records)
    return counts
