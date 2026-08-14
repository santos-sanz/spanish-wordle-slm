from wordle_slm.data import deterministic_split
from wordle_slm.solver import entropy_for_counts
from wordle_slm.visualization import parse_training_log


def test_split_is_deterministic_and_disjoint() -> None:
    answers = [f"a{i:04d}" for i in range(100)]
    first = deterministic_split(answers)
    second = deterministic_split(answers)
    assert first == second
    assert len(first["train"]) == 70
    assert len(first["valid"]) == 10
    assert set(first["train"]).isdisjoint(first["test"])


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
