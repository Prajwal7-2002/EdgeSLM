import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import mcnemar_exact, wilson_interval


def test_mcnemar_identical_systems_is_not_significant():
    hits = [True, False, True, True]
    assert mcnemar_exact(hits, hits) == (0, 0, 1.0)


def test_mcnemar_counts_discordant_items_only():
    a = [True, True, True, False, True]
    b = [False, False, True, False, True]
    only_a, only_b, p = mcnemar_exact(a, b)
    assert (only_a, only_b) == (2, 0)
    assert p == 0.5  # two-sided exact binomial with n=2


def test_mcnemar_strong_difference_is_significant():
    a = [True] * 12 + [False] * 1
    b = [False] * 12 + [True] * 1
    assert mcnemar_exact(a, b)[2] < 0.01


def test_interval_contains_observed_accuracy():
    lower, upper = wilson_interval(29, 40)
    assert lower < 29 / 40 < upper


def test_interval_is_valid_for_perfect_score():
    lower, upper = wilson_interval(40, 40)
    assert 0 < lower < upper <= 1


def test_invalid_total_is_rejected():
    try:
        wilson_interval(0, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero total should be rejected")
