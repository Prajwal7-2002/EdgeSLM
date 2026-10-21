import math
import re
from collections import Counter


# High-confidence cryptographic keys, API tokens, and secret URI structures
STRUCTURAL_SECRET_PATTERNS = (
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b",
    r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b",
    r"\bsk_live_[0-9a-zA-Z]{24,}\b",
    r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b",
    r"\b(postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^:\s]+:[^@\s]+@[^\s/:]+",
    r"\b(?:password|passwd|pin|secret)\s*[:=]\s*[\"']?[^\s\"']{4,}",
    r"\b(?:password|pin)\s+(?:is|equals)\s+(?:the\s+)?[\"']?[^\s\"']{4,}",
)

RESTRICTED_PATTERNS = (
    r"\b(production password|master credential|recovery phrase|secret credential)\b",
    r"\b(private key|encryption key|cryptographic material|signing key|encryption material)\b",
    r"\b(api key|access token|authentication token|bearer token|live token)s?\b",
    r"\b((?:company|production|master|root|database|admin|login|user|api)\s+credentials?|production signing key)s?\b",
    r"\b(bypass(?:ing)? an? (?:active )?access control)\b",
    r"\b(impersonate an internal service|authorize live transactions)\b",
)

# Terms indicating IT guidance/policy rather than credential exposure
POLICY_GUIDANCE_PATTERNS = (
    r"\b(password (?:policy|manager|requirements?|reset|guidelines?)|strong password)\b",
    r"\b(acceptable use policy|security awareness|never share your password)\b",
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


_STRUCTURAL_SECRET_RE = re.compile("|".join(STRUCTURAL_SECRET_PATTERNS), re.IGNORECASE)
_RESTRICTED_RE = re.compile("|".join(RESTRICTED_PATTERNS), re.IGNORECASE)
_POLICY_GUIDANCE_RE = re.compile("|".join(POLICY_GUIDANCE_PATTERNS), re.IGNORECASE)
_PUBLIC_RE = re.compile("|".join(PUBLIC_PATTERNS), re.IGNORECASE)
_CONFIDENTIAL_RE = re.compile("|".join(CONFIDENTIAL_PATTERNS), re.IGNORECASE)


def shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy to distinguish random secret tokens from English prose."""
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def has_high_entropy_token(text: str) -> bool:
    """Detect unquoted high-entropy secrets in assignments, avoiding URLs, links, and email headers."""
    # Strip URLs and email headers to avoid false alarms on long query params and message IDs
    cleaned = re.sub(r'https?://[^\s)\]>"\'\}]+', ' ', text)
    cleaned = re.sub(r'(?i)message-id:[^\n]+', ' ', cleaned)
    cleaned = re.sub(r'mailto:[^\s>]+', ' ', cleaned)

    # 1. Config/env/header secret assignments: KEY=... or token: ...
    for m in re.finditer(
        r'(?i)\b\w*(?:secret|token|key|password|passwd|credential|auth)\w*\s*[:=]\s*["\']?([A-Za-z0-9+/=_-]{16,})',
        cleaned,
    ):
        val = m.group(1)
        if shannon_entropy(val) >= 3.8:
            return True

    # 2. Standalone raw base64/hex token blocks (>= 32 chars, mix of digits and letters)
    for token in re.findall(r'\b[A-Za-z0-9+/=_-]{32,}\b', cleaned):
        if any(c.isdigit() for c in token) and any(c.isalpha() for c in token):
            if shannon_entropy(token) >= 4.5:
                return True
    return False


def apply_policy_overrides(text, model_prediction):
    """Apply high-confidence deterministic rules before invoking the neural SLM."""
    # 1. Definite structural secret (key blocks, token formats, database URIs, entropy)
    if _STRUCTURAL_SECRET_RE.search(text) or has_high_entropy_token(text):
        return "RESTRICTED"

    # 2. General restricted terms, unless it's an educational/guidance policy document
    if _RESTRICTED_RE.search(text):
        if not _POLICY_GUIDANCE_RE.search(text):
            return "RESTRICTED"

    # 3. Official public disclosures & press releases (takes precedence over financial figures)
    if _PUBLIC_RE.search(text):
        return "PUBLIC"

    # 4. Confidential business contracts & salary reviews
    if _CONFIDENTIAL_RE.search(text):
        return "CONFIDENTIAL"

    return model_prediction
