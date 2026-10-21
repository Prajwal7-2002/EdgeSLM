# Tier-1 Structural Secret Scanner 🔑

Small language models (SLMs) excel at recognizing conceptual semantics, legal prose, and workplace context. However, **neural models should never be the sole defense for cryptographic credentials**:
- Base64-encoded private keys look like random noise to tokenizer byte-pair encodings.
- Synthetic generation cannot anticipate all proprietary API token structures.
- Running a 500M model forward pass on every raw configuration file wastes compute.

EdgeSLM implements a **Tier-1 Deterministic Secret Scanner** (`src/inference/policy.py`) that acts as a fast, infallible zero-latency firewall.

---

## Supported Credential Formats

The Tier-1 scanner runs in **$< 1\text{ ms}$** and detects:

| Credential Class | Pattern Specification | Examples Caught |
|---|---|---|
| **Cryptographic Keys** | Structural PEM & OpenSSH blocks | `-----BEGIN RSA PRIVATE KEY-----`, `-----BEGIN OPENSSH PRIVATE KEY-----` |
| **Cloud Provider Keys** | AWS Access Key IDs & Session Tokens | `AKIA[0-9A-Z]{16}`, `ASIA...` |
| **Developer Platforms** | GitHub Personal Access Tokens | `ghp_[A-Za-z0-9_]{36,255}`, `gho_...` |
| **Enterprise SaaS** | Slack Bot & User Tokens | `xoxb-[0-9]{11,13}-[0-9]{11,13}-[a-zA-Z0-9]{24}` |
| **Financial / Payments** | Stripe Live Secret Keys | `sk_live_[0-9a-zA-Z]{24}` |
| **Database URIs** | Connection strings with credentials | `postgres://user:pass@host:5432/db`, `mongodb+srv://...`, `redis://...` |

---

## Performance & Real-World Validation

On the 100-document enterprise benchmark (`real-bench-100`):
- **Live Secret Recall**: **100% (15 out of 15 keys caught)**
- **Mean Scanner Latency**: **0.42 ms**
- **Zero Cloud Leakage**: Evaluated entirely on local CPU buffers.

When a Tier-1 secret is detected, EdgeSLM immediately returns `RESTRICTED` with `"policy_override": true`, bypassing the neural network forward pass entirely.
