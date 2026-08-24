import re


RESTRICTED_PATTERNS = (
    r"\b(password|passphrase|secret credential|master credential|recovery phrase)\b",
    r"\b(private key|encryption key|cryptographic material|signing key|encryption material)\b",
    r"\b(api key|access token|authentication token|bearer token|live token)s?\b",
    r"\b(credentials?|production signing key)s?\b",
    r"\b(bypass(?:ing)? an? (?:active )?access control)\b",
    r"\b(impersonate an internal service|authorize live transactions)\b",
)

PUBLIC_PATTERNS = (
    r"\b(publicly announced|publicly available|public website)\b",
    r"\b(approved for publication|approved publication|intended for external sharing)\b",
    r"\b(available to anyone|available to prospective customers)\b",
    r"\b(published|customer-facing|public brochure|public case study)\b",
)

CONFIDENTIAL_PATTERNS = (
    r"\b(customer contract|account-specific|negotiated (?:renewal )?discount)",
    r"\b(contract dispute|settlement terms|pricing concessions)",
    r"\b(employee salary|individual salary|compensation analysis)",
    r"\b(financial forecast|budget revision|projected expenses)",
    r"\b(private correspondence|private organizational chart)",
)


_RESTRICTED_RE = re.compile(
    "|".join(RESTRICTED_PATTERNS),
    re.IGNORECASE,
)
_PUBLIC_RE = re.compile(
    "|".join(PUBLIC_PATTERNS),
    re.IGNORECASE,
)
_CONFIDENTIAL_RE = re.compile(
    "|".join(CONFIDENTIAL_PATTERNS),
    re.IGNORECASE,
)


def apply_policy_overrides(text, model_prediction):
    """Apply only high-confidence safety/publication overrides."""
    if _RESTRICTED_RE.search(text):
        return "RESTRICTED"

    if _PUBLIC_RE.search(text):
        return "PUBLIC"

    if _CONFIDENTIAL_RE.search(text):
        return "CONFIDENTIAL"

    return model_prediction
