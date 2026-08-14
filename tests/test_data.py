from wordle_slm.data import deterministic_split
from wordle_slm.solver import entropy_for_counts


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
