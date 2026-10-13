import math


def wilson_interval(correct, total, confidence=0.95):
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        raise ValueError("total must be greater than zero")
    if correct < 0 or correct > total:
        raise ValueError("correct must be between zero and total")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")

    z = 1.959963984540054
    if confidence != 0.95:
        raise ValueError("only 95% confidence is currently supported")

    proportion = correct / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def mcnemar_exact(a_correct, b_correct):
    """Exact two-sided McNemar test for two systems scored on the same items.

    Two accuracies on the same 80 documents are not independent samples, so
    comparing their Wilson intervals overstates the uncertainty. McNemar looks
    only at the documents where exactly one system is right. Returns
    (only_a_right, only_b_right, p_value).
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("both systems must be scored on the same items")
    only_a = sum(a and not b for a, b in zip(a_correct, b_correct))
    only_b = sum(b and not a for a, b in zip(a_correct, b_correct))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)
