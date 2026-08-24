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
