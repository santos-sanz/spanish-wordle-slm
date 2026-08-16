from __future__ import annotations

import json
import random
from typing import Any

from .core import ALL_GREEN, history_text
from .data import ROOT, TRAINING_DIR
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
    "Modo PURE. Juegas Wordle en español. La palabra objetivo tiene cinco letras y dispones de seis "
    "intentos. 0=gris, 1=amarillo, 2=verde. Respeta letras repetidas. Responde únicamente "
    'con JSON válido: {"guess":"palabra"}.'
)
AGENT_SYSTEM = (
    "Modo AGENT. "
    + PURE_SYSTEM.removeprefix("Modo PURE. ")
    + " Puedes usar get_candidates una vez por turno para consultar soluciones compatibles."
)
ORACLE_SYSTEM = PURE_SYSTEM + " Usa best_guess para obtener la jugada del solver."
def repair_prompt(history: list[tuple[str, int]]) -> str:
    unavailable = ", ".join(word for word, _ in history) or "ninguna"
    return (
        "Respuesta inválida o repetida. Palabras no disponibles: "
        f"{unavailable}. Devuelve únicamente JSON con una palabra válida nueva: "
        '{"guess":"palabra"}.'
    )


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


def _agent_direct_record(history: list[tuple[str, int]], turn: int, guess: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": state_prompt(history, turn)},
            {"role": "assistant", "content": json.dumps({"guess": guess})},
        ]
    }


def _pure_repair_record(
    history: list[tuple[str, int]], turn: int, guess: str
) -> dict[str, Any]:
    # The invalid/repeated action is runtime context, not a target. Including
    # it as an assistant message makes masked SFT teach the adapter to repeat
    # the bad guess.  Keep the repair state in one user message and supervise
    # only the valid replacement that follows it.
    return {
        "messages": [
            {"role": "system", "content": PURE_SYSTEM},
            {
                "role": "user",
                "content": f"{state_prompt(history, turn)}\n{repair_prompt(history)}",
            },
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
            if turn == 0 and starter is not None and starter in solver.guess_index:
                guess = starter
            elif turn == 0:
                guess = solver.best_guess(candidates, 6 - turn)
            else:
                # A candidate-copy policy is deliberately used for the
                # competitive Pure curriculum.  It is deterministic, always
                # legal, and asks the adapter to learn the feedback filter
                # rather than approximate an entropy calculation.  The Agent
                # track uses the same policy after its tool response, making
                # the two tracks comparable while keeping Pure tool-free.
                compatible = solver.candidate_words(history)
                used = {word for word, _ in history}
                guess = next(word for word in compatible if word not in used)
            code = int(solver.matrix[solver.guess_index[guess], target_index])
            history.append((guess, code))
            if code == ALL_GREEN:
                break
            candidates = solver.apply_feedback(candidates, guess, code)
        games.append(history)
    return games


def _records_for_targets(solver: WordleSolver, targets: list[str]) -> list[dict[str, Any]]:
    pure: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    for target in targets:
        for game_index, game in enumerate(_games_for_target(solver, target)):
            history: list[tuple[str, int]] = []
            for turn_index, (guess, code) in enumerate(game, start=1):
                # Perturbed trajectories force audio/seron/lenta on turn one to
                # expose recovery states. Those actions are context, not policy
                # targets: supervising them would assign four different labels
                # to the identical empty-history prompt. Only the canonical
                # solver trajectory may supervise turn one.
                if game_index > 0 and turn_index == 1:
                    history = [*history, (guess, code)]
                    continue
                pure.append(_pure_record(history, turn_index, guess))
                if history:
                    repairs.append(_pure_repair_record(history, turn_index, guess))
                history = [*history, (guess, code)]

    # Agent is a different policy from Pure: it should exploit the information
    # it is actually allowed to see. Start without a tool call, then copy the
    # first compatible unused answer returned by get_candidates. This simple,
    # learnable policy solves every validation target within six turns, whereas
    # asking a 2.6B model to reconstruct the solver's entropy ranking from a
    # candidate list needlessly turns tool use into another reasoning task.
    agent: list[dict[str, Any]] = []
    for target in targets:
        target_index = solver.answer_index[target]
        history: list[tuple[str, int]] = []
        used: set[str] = set()
        for turn in range(1, 7):
            if turn == 1:
                guess = solver.best_guess(solver.all_candidates, 6)
                agent.append(_agent_direct_record(history, turn, guess))
            else:
                candidates = solver.candidate_words(history)
                guess = next(word for word in candidates if word not in used)
                agent.extend(
                    _tool_records(
                        mode="agent",
                        history=history,
                        turn=turn,
                        guess=guess,
                        result={"count": len(candidates), "candidates": candidates},
                    )
                )
            used.add(guess)
            code = int(solver.matrix[solver.guess_index[guess], target_index])
            history.append((guess, code))
            if code == ALL_GREEN:
                break

    rng = random.Random(20260814)
    # Competitive selection is Pure/Agent only; Oracle bypasses the model in
    # the harness. Duplicate the Agent curriculum once so the refinement mix
    # balances normal play, late-turn repair behavior, and candidate copying.
    records = pure + repairs + agent + agent
    rng.shuffle(records)
    return records


def generate_training_data() -> dict[str, int]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    pure_dir = ROOT / "data" / "pure-refinement"
    pure_dir.mkdir(parents=True, exist_ok=True)
    solver = WordleSolver()
    counts: dict[str, int] = {}
    for split in ("train", "valid"):
        records = _records_for_targets(solver, solver.splits[split])
        path = TRAINING_DIR / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        pure_records = [
            record
            for record in records
            if record.get("messages", [{}])[0].get("content", "").startswith("Modo PURE.")
        ]
        pure_path = pure_dir / f"{split}.jsonl"
        with pure_path.open("w", encoding="utf-8") as handle:
            for record in pure_records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts[split] = len(records)
        counts[f"pure_{split}"] = len(pure_records)
    return counts
