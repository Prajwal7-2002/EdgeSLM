# EdgeSLM 🛡️

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Prajwal7-2002/EdgeSLM/blob/main/LICENSE)
[![CI](https://github.com/Prajwal7-2002/EdgeSLM/actions/workflows/tests.yml/badge.svg)](https://github.com/Prajwal7-2002/EdgeSLM/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-ee4c2c)
![Hardware](https://img.shields.io/badge/edge-Laptop%20CPU%20%7C%20RTX%203050-green.svg)

An **air-gapped document-sensitivity classifier and enterprise DLP (Data Loss Prevention) service** designed for local edge deployment on consumer hardware with zero data leakage.

---

## The Enterprise DLP Problem

In enterprise environments, scanning confidential internal documents, commercial contracts, and source code with cloud APIs (e.g., OpenAI or Anthropic) introduces severe compliance and data-leakage risks.

**EdgeSLM** runs entirely on-premise on standard CPU hardware or consumer GPUs:
- **Sensitivity Tiers**: Labels files `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, or `RESTRICTED`.
- **Zero Cloud Dependence**: The model, tokenizer, and policy scanner execute entirely in a local process or container.
- **Ultra-Compact Footprint**: 4-bit ONNX quantized build is **392 MB**, executes inference in $\approx 0.5\text{s}$ on a laptop CPU, and requires only $\approx 1\text{ GB}$ of RAM.

---

## Architecture Overview

```
Client Document (.pdf, .docx, .eml, source code)
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ EdgeSLM Ingestion Pipeline                                                       │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                    Tier-1 Structural Secret Scanner                        │  │
│  │  - Deterministic PEM, OpenSSH, AWS, GitHub, Slack, Stripe, DB URIs         │  │
│  │  - Context-filtered Shannon entropy ($H \ge 3.8$) (< 1 ms execution)      │  │
│  └──────────────────────────────────────┬─────────────────────────────────────┘  │
│                                         │                                        │
│                 Secret detected? ───────┼────────► RESTRICTED (Policy Override)  │
│                                         │                                        │
│                                         ▼ No secret                              │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                       Text Extraction & Chunking                           │  │
│  │  - Overlapping sliding windows (~435 tokens/chunk)                         │  │
│  └──────────────────────────────────────┬─────────────────────────────────────┘  │
│                                         │                                        │
│                                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                       Inference Systems Engine                             │  │
│  │  - Dynamic Tensor Chunk Batching (6.7x speedup on multi-page documents)    │  │
│  │  - Speculative Early-Exit (88.9% compute reduction on sensitive files)     │  │
│  │  - Qwen2.5-0.5B LoRA v3 / ONNX INT4 edge runtime                           │  │
│  └──────────────────────────────────────┬─────────────────────────────────────┘  │
│                                         │                                        │
│                                         ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐  │
│  │                    Document-Level Aggregation & Audit                      │  │
│  │  - Most-sensitive chunk escalation & unanimous chunk consensus             │  │
│  └────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Systems Engineering Highlights

| Capability | Industry Reality | EdgeSLM Solution |
|---|---|---|
| **Multi-Page Scanning** | Sequential chunk loops (1.5s per 8 chunks) | **Dynamic Tensor Batching**: 6.69x speedup ($1,497\text{ ms} \to 223\text{ ms}$) |
| **Compute Efficiency** | Scanning entire 20-page document even if page 1 leaks secrets | **Speculative Early-Exit**: 88.9% compute reduction |
| **Secret Detection** | Neural hallucination or missed cryptic keys | **Tier-1 Structural Scanner**: 100% recall on cryptographic keys in < 1 ms |
| **Generalization** | Flawed synthetic keyword heuristics | **v3 Fine-Tuning**: Outperforms TF-IDF on contrastive evaluation ($p = 0.043$) |
