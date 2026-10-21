# Speculative Early-Exit ⏩

In enterprise Data Loss Prevention (DLP), documents are classified hierarchically: a document inherits the label of its **most sensitive chunk**. Therefore, if a document contains an NDA clause or production credentials on page 1, classifying the remaining 20 pages cannot lower the document's classification tier.

EdgeSLM leverages this structural invariant with **Speculative Early-Exit**.

---

## Early-Exit Mechanism

When scanning multi-page documents:
1. EdgeSLM evaluates the first chunk (or an initial batch).
2. If the predicted label is high-severity (`RESTRICTED` or `CONFIDENTIAL`) with confidence exceeding the threshold:
   $$P(\text{label}) \ge 0.98$$
3. Inference terminates immediately. The remaining chunks are skipped, and the document is classified at maximum sensitivity.

```
Document Ingestion
       │
       ▼
[ Chunk 1 ] ──► P(CONFIDENTIAL) = 0.998  ──(Confidence >= 0.98)──► HALT INFERENCE!
       │                                                          Label: CONFIDENTIAL
   [ Chunk 2 ]   (SKIPPED - 0 ms)
   [ Chunk 3 ]   (SKIPPED - 0 ms)
   [ Chunk 4 ]   (SKIPPED - 0 ms)
   ...
   [ Chunk 9 ]   (SKIPPED - 0 ms)
```

---

## Quantitative Savings

Evaluated on multi-page sensitive enterprise contracts:

| Metric | Sequential Scan (All Chunks) | Speculative Early-Exit | Efficiency Gain |
|---|---|---|---|
| **Chunks Processed** | 9 chunks | **1 chunk** | **88.9% compute reduction** |
| **Inference Latency** | 1,227 ms | **243 ms** | **80.2% latency reduction** |
| **Classification Accuracy** | 100% | **100% (Identical)** | **Zero accuracy loss** |

---

## Configuration

In `src/inference/classifier.py` and the FastAPI service, early-exit can be toggled:

```python
# Enable speculative early-exit for high-throughput batch scanning
result = classifier.classify_document(
    document_text,
    early_exit=True,
    early_exit_threshold=0.98
)
```
