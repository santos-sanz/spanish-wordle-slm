import json
from pathlib import Path

from wordle_slm.benchmark_visualization import (
    _paired_decisive_interval,
    _paired_win_interval,
)
from wordle_slm.core import history_text
from wordle_slm.data import deterministic_split
from wordle_slm.dataset import _records_for_targets
from wordle_slm.solver import WordleSolver, entropy_for_counts
from wordle_slm.visualization import parse_training_log


def test_split_is_deterministic_and_disjoint() -> None:
    answers = [f"a{i:04d}" for i in range(100)]
    first = deterministic_split(answers)
    second = deterministic_split(answers)
    assert first == second
    assert len(first["train"]) == 70
    assert len(first["valid"]) == 10
    assert set(first["train"]).isdisjoint(first["test"])


def test_empty_history_prompt_matches_spanish_benchmark() -> None:
    assert history_text([]) == "Sin intentos previos."


def test_empty_history_has_one_canonical_supervised_action() -> None:
    solver = WordleSolver()
    records = _records_for_targets(solver, solver.splits["train"][:3])
    first_turn_answers = {
        record["messages"][-1]["content"]
        for record in records
        if "tools" not in record and record["messages"][1]["content"].startswith("Turno 1/6")
    }
    expected = solver.best_guess(solver.all_candidates, 6)
    assert first_turn_answers == {json.dumps({"guess": expected})}


def test_paired_win_interval_detects_clear_slm_advantage() -> None:
    slm = [{"target": str(index), "solved": True} for index in range(20)]
    rival = [{"target": str(index), "solved": False} for index in range(20)]
    observed, low, high = _paired_win_interval(slm, rival, samples=500)
    assert (observed, low, high) == (1.0, 1.0, 1.0)


def test_paired_decisive_interval_uses_turns_when_wins_tie() -> None:
    slm = [
        {"target": str(index), "solved": True, "scoredTurns": 3}
        for index in range(20)
    ]
    rival = [
        {"target": str(index), "solved": True, "scoredTurns": 5}
        for index in range(20)
    ]
    metric, observed, low, high = _paired_decisive_interval(slm, rival, samples=500)
    assert metric == "mean_scored_turns"
    assert (observed, low, high) == (2.0, 2.0, 2.0)


def test_entropy_prefers_even_partition() -> None:
    assert entropy_for_counts([2, 2]) > entropy_for_counts([3, 1])


def test_training_log_parser() -> None:
    log = """
Iter 50: Val loss 0.786, Val took 8.356s
Iter 50: Train loss 0.994, Learning Rate 2.000e-05, It/sec 1.398, Tokens/sec 13.701, Trained Tokens 486, Peak mem 3.094 GB
Iter 50: Saved adapter weights to adapter.safetensors.
"""
    parsed = parse_training_log(log)
    assert parsed.validation[0].loss == 0.786
    assert parsed.train[0].trained_tokens == 486
    assert parsed.checkpoints == [50]


def test_preference_data_excludes_hidden_test_and_has_strict_preferences() -> None:
    root = Path(__file__).parents[1] / "data" / "preferences"
    manifest = json.loads(root.joinpath("manifest.json").read_text(encoding="utf-8"))
    assert manifest["hidden_test_used"] is False
    assert manifest["train_validation_prompt_overlap"] == 0
    for split in ("train", "valid"):
        records = [
            json.loads(line)
            for line in root.joinpath(f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert all(
            tuple(record["chosen_solver_key"]) < tuple(record["rejected_solver_key"])
            for record in records
        )
