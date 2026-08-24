import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.policy import apply_policy_overrides


def test_restricted_overrides_model():
    assert apply_policy_overrides("The production password is here.", "PUBLIC") == "RESTRICTED"
    assert apply_policy_overrides("Company credentials are not for sharing.", "PUBLIC") == "RESTRICTED"


def test_explicit_public_overrides_model():
    assert apply_policy_overrides("This is publicly available on the website.", "CONFIDENTIAL") == "PUBLIC"


def test_confidential_contract_overrides_model():
    text = "The customer contract contains negotiated discounts."
    assert apply_policy_overrides(text, "PUBLIC") == "CONFIDENTIAL"


def test_ambiguous_text_delegates_to_model():
    assert apply_policy_overrides("Routine team process.", "INTERNAL") == "INTERNAL"
