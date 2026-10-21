# Shannon Entropy & Policy Overrides 🛡️

In security policy enforcement, high-entropy random strings (such as generated passwords, session cookies, and bearer tokens) often do not follow rigid vendor prefixes (e.g. `AKIA`).

EdgeSLM combines **Shannon Entropy analysis** with **structural context filtering** to catch arbitrary secrets while avoiding false positive flags on hashes or UUIDs.

---

## Mathematical Formulation: Shannon Entropy

The information entropy $H$ of a character string measures its unpredictability:

$$H(X) = -\sum_{i=1}^{K} P(x_i) \log_2 P(x_i)$$

Where:
- $K$ is the unique character set size.
- $P(x_i)$ is the empirical frequency of character $x_i$ in the token.

| String Type | Example | Calculated Entropy ($H$) | Verdict |
|---|---|---|---|
| **Natural English** | `"confidentiality"` | $H \approx 2.4 - 2.8$ | Normal prose |
| **Simple Words** | `"productionadmin"` | $H \approx 3.1$ | Low entropy |
| **High-Entropy Secret** | `"Copper!2291x9Q#"` | **$H \ge 3.8$** | **Secret Flagged** |

---

## Context-Filtering: Eliminating False Positives

Raw entropy scanners notoriously flag Git commit hashes, UUIDs, and base64 document thumbnails as security violations.

EdgeSLM enforces **Context Filtering**:
A high-entropy string ($H \ge 3.8$) only triggers a `RESTRICTED` override if it appears within an **assignment context**:
```regex
(?:password|secret|token|api[_-]?key|credential|auth)\s*[:=]\s*["']?([A-Za-z0-9!@#$%^&*()_+=\-]{12,})
```

This prevents random text strings, UUIDs, and software hashes from triggering false positives.

---

## Why Heuristic Regexes Were Purged (ADR-009)

Early iterations of EdgeSLM used heuristic regexes to flag words like `"confidential"`, `"NDA"`, or `"merger"`. 

This was completely purged:
- In real-world enterprise documents, public legal documents and textbooks frequently use the word *"confidential"* or discuss NDAs in an educational context.
- Hardcoded word heuristics broke neural transfer and inflated false positives.
- **Principle**: *The neural model handles language meaning; the policy layer only handles cryptographic secrets.*
