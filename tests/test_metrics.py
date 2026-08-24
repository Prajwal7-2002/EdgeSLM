import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.metrics import wilson_interval


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
