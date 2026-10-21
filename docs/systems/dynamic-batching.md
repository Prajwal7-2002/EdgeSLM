# Dynamic Tensor Chunk Batching 🚀

Real-world enterprise documents (such as board meeting minutes, legal vendor agreements, and engineering handbooks) often span multiple pages, resulting in anywhere from 2 to 25 token chunks.

---

## The Sequential Processing Bottleneck

In naive document classifiers, long documents are evaluated chunk-by-chunk in a sequential Python loop:

```python
# Naive sequential loop
chunk_scores = []
for chunk in document_chunks:
    scores = model.classify(chunk)  # Launches separate CUDA kernel each chunk!
    chunk_scores.append(scores)
```

This sequential pattern causes severe GPU underutilization:
1. **Kernel Launch Overhead**: Launching separate PyTorch forward passes sequentially pays host-to-device synchronization latency on every chunk.
2. **Memory Bandwidth Waste**: For each chunk, model parameters are loaded repeatedly from VRAM, keeping arithmetic intensity low.
3. **High Latency**: An 8-chunk document takes $\approx 1.5\text{ seconds}$ to process sequentially.

---

## The Dynamic Tensor Batching Solution

EdgeSLM implements **Dynamic Tensor Batching** (`src/inference/classifier.py`):
- All text chunks of a multi-page document are tokenized and dynamically padded to the batch's longest chunk.
- An attention mask ($B \times L$) ensures padded tokens do not influence attention projections.
- The entire document is evaluated in a **single batched forward pass** through the SLM's transformer layers.

```
Multi-Page Document (e.g. 8 Chunks)
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Chunk 1│ │ Chunk 2│ │ Chunk 3│ │ Chunk 4│ │ Chunk 5│ │ Chunk 6│ │ Chunk 7│ │ Chunk 8│
└────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
     │          │          │          │          │          │          │          │
     └──────────┴──────────┴──────────┴─────┬────┴──────────┴──────────┴──────────┘
                                            ▼
                           Dynamic Tensor Batch [8, Seq_Len]
                                            │
                                            ▼
                        Single Parallel Transformer Kernel (CUDA)
                                            │
                                            ▼
                           Class Probabilities [8, 4 Classes]
```

---

## Empirical Benchmark Results (RTX 3050 GPU)

Measured using `src/edge/benchmark_inference.py` across varying document lengths:

| Document Chunks | Sequential Loop Latency | Dynamic Batching Latency | Speedup Factor | Throughput (Chunks/sec) | Latency Saved |
|---|---|---|---|---|---|
| **1 Chunk** | 166.0 ms | 157.4 ms | 1.05x | 6.4 chunks/s | 5.2% |
| **2 Chunks** | 396.0 ms | 183.8 ms | **2.15x** | 10.9 chunks/s | 53.6% |
| **4 Chunks** | 684.3 ms | 245.3 ms | **2.79x** | 16.3 chunks/s | 64.2% |
| **8 Chunks** | 1,497.5 ms | **223.7 ms** | **6.69x** | **35.8 chunks/s** | **85.1%** |

Multi-chunk document latency plummets from **1.5 seconds down to 223 ms**—a **6.69x speedup** on consumer hardware with zero accuracy degradation.
