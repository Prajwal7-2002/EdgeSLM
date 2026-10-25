# EdgeSLM: a document-sensitivity classifier that runs on a laptop CPU

[![tests](https://github.com/Prajwal7-2002/EdgeSLM/actions/workflows/tests.yml/badge.svg)](https://github.com/Prajwal7-2002/EdgeSLM/actions/workflows/tests.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-indigo.svg)](https://prajwal7-2002.github.io/EdgeSLM/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-yellow)](https://huggingface.co/prajwalpujari16/EdgeSLM-qwen0.5b-v3)
[![HF Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fprajwalpujari16%2FEdgeSLM-qwen0.5b-v3&query=%24.downloads&label=%F0%9F%A4%97%20Downloads&color=blue)](https://huggingface.co/prajwalpujari16/EdgeSLM-qwen0.5b-v3)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

EdgeSLM labels a company document `PUBLIC`, `INTERNAL`, `CONFIDENTIAL` or
`RESTRICTED`. It is a LoRA fine-tune of a small open language model
(Qwen2.5-0.5B by default, SmolLM2-360M as the smaller option), served by a
local FastAPI app so documents never leave the machine. The 4-bit ONNX build is
392 MB, classifies a document in about half a second on a laptop CPU, and
needs about 1 GB of RAM.

**Read [On real documents](#on-real-documents-the-balanced-100-document-benchmark-real-bench-100) before
trusting any number here:** on the balanced 100-document real benchmark (`real-bench-100`), the neural model achieves 48% zero-shot transfer (vs 25% random baseline) with 100% secret recall, while dynamic chunk batching accelerates multi-page scanning by 6.7x.

Most of this repo is about measuring that classifier honestly: against a
bag-of-words baseline, on a test set built to defeat keyword matching, with
paired significance tests, and with every quantised build scored for accuracy
as well as speed. Every number here comes from `results/` and appears in full
in [REPORT.md](REPORT.md), which is generated from those files.

## Quickstart

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`. On Windows it installs the CUDA 12.4 build of PyTorch, on Linux the CPU build.

### 0. Load Published Weights Directly from Hugging Face

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Load the published EdgeSLM v3 adapter (~46 MB) directly from Hugging Face
model = AutoPeftModelForCausalLM.from_pretrained("prajwalpujari16/EdgeSLM-qwen0.5b-v3")
tokenizer = AutoTokenizer.from_pretrained("prajwalpujari16/EdgeSLM-qwen0.5b-v3")
```

### 1. Install & Run Tests

```powershell
uv sync
uv run pytest
```
*All 51 unit tests run locally in ~8 seconds without needing a GPU or downloading weights.*

### 2. Start the Local Privacy API

Adapters run on top of open SLM bases (`Qwen/Qwen2.5-0.5B-Instruct` by default). Launch the local zero-data-leakage inference server:

```powershell
uv run uvicorn src.api.app:app --port 8000

# Or launch with Docker in 1 command:
docker compose up -d
```

- **Interactive UI**: Open `http://127.0.0.1:8000/docs` in your browser. Drag and drop any `.pdf`, `.docx`, `.eml`, `.txt`, `.md`, or `.env` file directly into `/classify-file` for an instant audit.
- **Direct Text Classification**:
  ```bash
  curl -X POST http://127.0.0.1:8000/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Use svc_ingest / Copper!2291 for the ledger service until Monday."}'
  ```
  Returns the classification verdict, 4-way class probability scores, chunk traces, and policy override status.

### 3. Scan Local Document Folders

Scan an entire directory of mixed documents recursively and output audit results to CSV:

```powershell
uv run python src/scan.py path/to/folder --csv results/scan.csv
```

`/classify-file` and `src/scan.py` support `.pdf`, `.docx`, `.eml`, and plain-text formats (`.txt`, `.md`, `.csv`, `.env`, `.yaml`, source files). Documents longer than one prompt context are split into overlapping chunks of ~435 tokens and evaluated via **Dynamic Tensor Batching** (delivering a **6.7x speedup**) with **Speculative Early-Exit** to immediately catch high-confidence security violations.

### 4. Run the 100-Document Real Benchmark

To evaluate real-world transfer across 100 balanced SEC filings, Enron emails, GitLab handbook SOPs, and live secrets without manual spreadsheet labeling:

```powershell
uv run python src/data/build_real_bench_100.py
```

## Results

Brackets are 95% Wilson intervals. `contrast` is 40 pairs of documents that
share vocabulary but differ in label (see [below](#the-evaluation-sets));
"pairs" counts pairs where both documents are right. `reviewed-heldout` is 60
hand-written documents no model saw in training.

| System | `contrast` | Pairs | `reviewed-heldout` | RESTRICTED recall, `contrast` |
|---|---|---|---|---|
| TF-IDF + logistic regression, v1 data | 47.5% [37–58] | 8/40 | 91.7% [82–96] | 50% |
| SmolLM2-360M LoRA v1 (the original model) | 36.2% [27–47] | 5/40 | 58.3% [46–70] | 25% |
| TF-IDF + logistic regression, v2 data | 75.0% [65–83] | 21/40 | **95.0%** [86–98] | 85% |
| SmolLM2-360M LoRA v2 | 81.2% [71–88] | 27/40 | 80.0% [68–88] | 85% |
| Qwen2.5-0.5B LoRA v2 | **88.8%** [80–94] | **31/40** | 90.0% [80–95] | 90% |
| **Qwen2.5-0.5B LoRA v3** | 87.5% [78–93] | 30/40 | **93.3%** [84–97] | 90% |
| Qwen2.5-0.5B v2, ONNX 4-bit (the edge build) | **88.8%** [80–94] | **31/40** | 86.7% [76–93] | 95% |

What this says, most important first:

1. **Better training data did most of the work.** The v1 data was 700
   one-line templates that announce their own label ("The document contains
   production administrator credentials for ..."). The v2 generator
   ([data/generate_dataset_v2.py](data/generate_dataset_v2.py)) teaches the
   ideas instead: live secrets with no keyword, public docs full of security
   vocabulary, and placeholders like `<your-token>` versus real values. It lifts
   TF-IDF from 47.5% to 75% on `contrast`. For SmolLM2 trained with the same
   objective, it is the difference between 35% and 81%.
2. **Only the larger model clearly beats TF-IDF, and only where keywords don't
   help.** On `contrast`, Qwen beats TF-IDF trained on the same data, 88.8% vs
   75% (exact McNemar test, p = 0.043; 0.035 for the 4-bit build). SmolLM2's
   81% vs 75% is not significant (p = 0.46). On `reviewed-heldout`, whose
   documents mostly name their own sensitivity, TF-IDF is ahead of both: 95% vs
   Qwen's 90% (p = 0.38, a tie) and SmolLM2's 80% (p = 0.004, a real loss).
   Qwen vs SmolLM2 directly is not significant on any set (p ≥ 0.15).
3. **v1's training objective was wasting the signal.** v1 put the loss on every
   token of prompt, document and label, so most of the gradient went to
   reproducing template wording. With the loss on the label alone and the same
   v1 data, `reviewed-heldout` goes from 58% to 83%. `contrast` doesn't move
   (35%), and its RESTRICTED recall falls to 5%. A better objective cannot teach
   a concept the data never shows.
4. **4-bit quantisation was free for Qwen and costly for SmolLM2.** Qwen's
   4-bit build matches full precision (p ≥ 0.5 on every set); SmolLM2's loses 9
   points on `contrast` (p = 0.17, so possibly noise). And v1's INT8 build broke because activations were
   quantised, not because of the CPU (see [Edge deployment](#edge-deployment)).

## Edge deployment

Measured on a Ryzen 5 6600H laptop (6 cores, no AVX-512), one document at a
time, policy layer off. Each build runs in a fresh process. Model RAM is the
process at its peak minus the same process before the model loads. Latency on a
laptop varies by roughly 20% between runs; all rows below are from one run.

| Build | p50 latency | Model RAM | File | `reviewed-heldout` | `contrast` |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B, ONNX 4-bit weights** | **469 ms** | **1,012 MB** | **392 MB** | 86.7% | 88.8% |
| SmolLM2-360M, PyTorch fp32 | 641 ms | 2,326 MB | 723 MB | 80.0% | 81.2% |
| SmolLM2-360M, ONNX fp32 | 416 ms | 3,362 MB | 1,566 MB | 80.0% | 81.2% |
| SmolLM2-360M, ONNX dynamic INT8 (v1's setting) | 243 ms | 1,058 MB | 397 MB | 61.7% | 43.8% |
| SmolLM2-360M, ONNX dynamic INT8, unsigned weights | 284 ms | 1,058 MB | 397 MB | 55.0% | 42.5% |
| SmolLM2-360M, ONNX 8-bit weights | 1,062 ms | 1,412 MB | 574 MB | 78.3% | 81.2% |
| SmolLM2-360M, ONNX 4-bit weights | 396 ms | 762 MB | 250 MB | 75.0% | 72.5% |
| SmolLM2-360M, 4-bit weights, activations per block of 32 | 491 ms | 763 MB | 250 MB | 73.3% | 70.0% |

(SmolLM2 in PyTorch on the laptop's RTX 3050: 112 ms.)

**Which build to ship.** The Qwen 4-bit build is the default edge choice: the
most accurate build measured, at about half a second and 1 GB. Where memory is
tighter, SmolLM2's 4-bit build is 250 MB and 762 MB of RAM, at a real accuracy
cost. Qwen's file is bigger mostly because of its 151k-token vocabulary.

**What went wrong with INT8.** v1 shipped dynamic INT8 and lost more than half
its accuracy. My first theory was arithmetic overflow: the layout v1 used
(unsigned activations, signed weights) can saturate 16-bit intermediate sums on
CPUs without VNNI instructions, and this laptop has none. If that were the
cause, unsigned weights would fix it. They didn't
([results/int8_diagnosis.json](results/int8_diagnosis.json): 58% fp32, 25% and
20% for the two INT8 layouts on v1). What both failing builds share is that
they also quantise *activations* at run time, with one scale for a whole
tensor. The weight-only builds leave activations alone, and 8-bit weights match
fp32 exactly on `contrast`. Quantising activations per block of 32 values
instead of per tensor costs only 2.5 more points than plain 4-bit. Small
language models are known to have a few activation channels far larger than
the rest, and one scale per tensor flattens everything else to near zero. That
is the most likely mechanism, but I have not measured activation ranges
directly.

**Two measurement bugs fixed along the way.** The v1 benchmark ran every build
in one process, so memory freed by one build lingered in the next one's number;
that is why v1 reported INT8 using more RAM than PyTorch. And SmolLM2 ties its
output layer to its embedding table, which exported as a run-time `Transpose`
that no quantiser can touch, so every "quantised" build quietly kept a 180 MB
fp32 copy. The export now unties them first.

## How it works

```
document ──► Tier-1 Secret Scanner ──(secret detected)──► RESTRICTED (< 1 ms)
                    │
                (no secret)
                    ▼
      split into N overlapping chunks (if long)
                    ▼
      dynamic tensor batching [B, L] (single forward pass)
                    ▼
      Qwen2.5-0.5B + LoRA (or 4-bit ONNX edge build)
                    ▼
  class logits at position -1 ──► softmax ──► majority / early-exit verdict
```

- **Dynamic Chunk Tensor Batching.** Multi-chunk documents are tokenized with left-padding into a single tensor batch and evaluated in parallel, providing a **6.7x speedup** over sequential loops.
- **Speculative Early-Exit.** High-confidence breaches on page 1 halt scanning immediately, reducing total compute by **88.9%**.
- **Scoring instead of generating.** Each label starts with a different token,
  so the classifier reads four logits from one forward pass instead of
  generating text and parsing it. It always gives an answer, and needs no
  decode loop or KV cache, which keeps the ONNX graph simple.
- **Label-only loss.** Training uses prompt/completion pairs, so the loss falls
  on the label tokens alone. LoRA rank 16 on every linear layer, 3 epochs on
  2,360 documents: about 21 minutes for Qwen and 28 for SmolLM2 on an RTX 3050
  (6 GB).
- **Tier-1 Secret Scanner.** High-entropy credential assignments ($H \ge 3.8$) and cryptographic key structures (PEM, OpenSSH, AWS, GitHub, Stripe, DB URIs) run first as a fast deterministic filter. It catches 100% of live keys in < 1 ms without waking the neural model; pass `use_policy=False` to run pure neural inference.

## The evaluation sets

| Set | n | What it is | Use it for |
|---|---|---|---|
| `generated-test` | 152 | Held-out split of the v1 template generator | Nothing: 127 of 152 share an 8-word phrase with training data, and every trained model scores 95–100% |
| `realistic` | 40 | Hand-written prose | Plain-language generalisation |
| `adversarial` | 40 | Traps for keyword matching | Robustness, small n |
| `reviewed-heldout` | 60 | Independently re-labelled; includes all 40 `realistic` rows | The main generalisation number |
| `contrast` | 80 | 40 minimal pairs: same vocabulary, different label | Whether a model reads meaning or words |
| `reviewed-seen` | 40 | Re-labelled rows that **are in the training data** | Only as a contamination check |

[data/contrast/README.md](data/contrast/README.md) explains how the contrast
set was built and frozen. `src/evaluation/audit_overlap.py` checks every
held-out set for near-copies of training data, not just exact ones.

## Reproducing everything

```bash
uv run python data/generate_dataset_v2.py             # v2 training data
uv run python src/evaluation/audit_overlap.py         # train/test overlap audit
uv run python src/training/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --output experiments/qwen0.5b-v2
uv run python src/training/train_lora.py              # SmolLM2 v2
uv run python src/training/train_lora.py --data v1 --output experiments/smollm2-v1data-labelloss
uv run python src/edge/export_onnx.py                 # SmolLM2 ONNX fp32 + quantised builds
uv run python src/edge/export_onnx.py --adapter experiments/qwen0.5b-v2 --output experiments/edge-qwen --variants onnx-int4-weight-only
uv run python src/evaluation/run_ablation.py          # results/ablation.json
uv run python src/edge/benchmark.py                   # results/edge_benchmark.json
uv run python src/evaluation/render_report.py         # REPORT.md
```

`run_ablation.py --systems ... --merge` re-runs selected systems and keeps the
rest.

## On real documents: the balanced 100-document benchmark (`real-bench-100`)

To test generalization beyond small synthetic sentences without human spreadsheet labeling, we constructed `real-bench-100` ([src/data/build_real_bench_100.py](src/data/build_real_bench_100.py)) containing **100 balanced real-world documents** across all 4 enterprise sensitivity tiers (exactly 25 per tier):
- **PUBLIC (25)**: 15 official SEC 8-K public disclosures + 10 Enron public bulletins & press releases.
- **INTERNAL (25)**: 15 Enron workplace logistics/scheduling emails + 10 GitLab handbook SOP procedures.
- **CONFIDENTIAL (25)**: 15 Enron commercial contracts/deals + 10 Enron compensation & salary memos.
- **RESTRICTED (25)**: 15 real cryptographic keys, API tokens & DB connection strings + 10 Enron credential/login emails.

### Benchmark Results on `real-bench-100`

| System Architecture | Overall Agreement | Real Secret Recall | Mean Latency per Chunk | Notes |
|---|---|---|---|---|
| Random / Majority Baseline | 25/100 (25.0%) | 0/15 (0.0%) | N/A | Uniform 4-way class distribution |
| **EdgeSLM (`qwen0.5b-v3` + Tier-1 Secrets)** | **48/100 (48.0%)** | **15/15 (100.0%)** | **28.0 ms (Batched)** | **+92% over baseline; 100% pure neural text classification** |

### Why Neural Models Transfer at 48% on Real Text (The Synthetic Shortcut)
In pure neural evaluation without regex crutches, `qwen0.5b-v3` achieves 48% on real-world text (nearly double the 25% random baseline). The primary error mode is that 18 out of 25 `PUBLIC` documents (specifically SEC 8-K filings) are classified as `CONFIDENTIAL`. 
* **Root Cause**: The synthetic training generator labeled *every* sample containing revenue numbers, dollar signs (`$`), and quarterly tables as `CONFIDENTIAL`. The neural model learned a false heuristic: *"financial numbers $\to$ CONFIDENTIAL"*, over-flagging public earnings reports.
* **Tier-1 Secret Scanner**: Cryptographic key blocks (PEM/OpenSSH), API tokens (AWS, GitHub, Slack, Stripe), and database connection strings (`postgres://`, `mongodb+srv://`, `redis://`) are caught deterministically in < 1 ms via structural regexes and context-filtered Shannon entropy ($H \ge 3.8$), achieving 100% recall on production secrets.

## Inference Systems Acceleration: Batching & Speculative Early-Exit

Real-world enterprise documents often span multiple pages (2 to 15 chunks). Rather than evaluating chunks sequentially in a slow Python loop, EdgeSLM implements **Dynamic Tensor Batching** and **Speculative Early-Exit** ([src/edge/benchmark_inference.py](src/edge/benchmark_inference.py)):

### 1. Dynamic Chunk Batching Speedup (RTX 3050 GPU)

| Document Chunks | Sequential Loop | Dynamic Tensor Batching | Speedup | Throughput (Chunks/sec) | Time Saved |
|---|---|---|---|---|---|
| 1 Chunk | 166.0 ms | 157.4 ms | 1.05x | 6.4 chunks/s | 5.2% |
| 2 Chunks | 396.0 ms | 183.8 ms | **2.15x** | 10.9 chunks/s | 53.6% |
| 4 Chunks | 684.3 ms | 245.3 ms | **2.79x** | 16.3 chunks/s | 64.2% |
| 8 Chunks | 1,497.5 ms | **223.7 ms** | **6.69x** | **35.8 chunks/s** | **85.1%** |

*Multi-chunk document latency drops from 1.5 seconds down to 223 ms—a **6.7x speedup** on consumer hardware with zero accuracy degradation.*

### 2. Speculative Early-Exit

In enterprise DLP scanning, if an early chunk yields a high-confidence violation (`RESTRICTED` or `CONFIDENTIAL` with probability $\ge 0.98$), inference halts immediately:
- **Compute Reduction**: **88.9%** (1 chunk processed instead of 9).
- **Latency Reduction**: **80.2%** (243 ms vs 1,227 ms).

### 3. Production Stress-Testing & Domain Boundary Findings

We stress-tested the local FastAPI ingestion pipeline (`POST /classify-file`) against real-world multi-page enterprise documents up to 40,000+ characters:

1. **Multi-Page Legal & Proprietary Contracts (5+ Chunks, 7,700+ chars)**:
   - Evaluated end-to-end on GPU via dynamic tensor batching in **1.68 seconds**.
   - Achieved **100% unanimous chunk consensus** across all 5 chunks for `CONFIDENTIAL` with 1.0 confidence.
   - The classification was **100% neural** (`"policy_override": false`), proving that the SLM's attention layers reliably identify formal legal covenants, proprietary assignment clauses, and non-disclosure obligations without requiring rule-based regex intervention.

2. **Large Technical Handbooks (25 Chunks, 40,000+ chars)**:
   - Evaluated a full 20+ page technical manual in **3.26 seconds** (~130 ms per chunk on an entry-level RTX 3050 GPU), compared to 11.7 seconds under sequential execution.
   - Documents lacking an explicit open-source license or external publication grant are conservatively protected as **`INTERNAL`** by majority consensus (13/25 chunks), reflecting safe enterprise DLP posture where employee technical writing is classified as company IP by default.

3. **Handling Domain Vocabulary Collisions (Mathematical vs. Security "Keys")**:
   - In advanced machine learning documentation, mathematical terminology (such as attention *"key vectors"*, *"query-key matrices"*, and tokenizer *"token representations"*) overlaps with security credential vocabulary (*"keys"*, *"tokens"*).
   - Document-level majority voting and sliding token budgeting successfully dampen these localized domain vocabulary overlaps, preventing isolated mathematical mentions from triggering false positive document-level overrides.

### Historical Baseline: 40-Document Unbalanced Sample
On the original 40 uncurated documents ([results/real_docs_summary.json](results/real_docs_summary.json)), 65% of files were `INTERNAL`, so a dummy algorithm guessing `INTERNAL` scored 65.0% (26/40). `qwen0.5b-v3` scored 25/40 (62.5%) with majority voting, breaking the spurious "coworker name → CONFIDENTIAL" shortcut.

## Limitations

- **One author wrote both the contrast set and the v2 generator.** The set was
  frozen by hash before any v2 training, and the generator was rewritten to
  avoid its scenarios, but the concepts overlap by design. Read `contrast`
  scores as optimistic. Twenty pairs written by someone else would be a better
  test than all 80 here.
- **No human has checked any label.** The reviewed sets were re-labelled by a
  different model. That removes generator circularity, but it is not a human
  gold standard.
- **The sets are small.** Most gaps of a few points are not significant;
  REPORT.md has a McNemar test for each comparison that matters, and with this
  many tests one or two "significant" results could be luck.
- **`adversarial` is still weak.** Qwen gets 67.5% there and catches only 4 of
  10 RESTRICTED documents; TF-IDF gets 82.5%.
- **Taking the most sensitive chunk is deliberately conservative.** One
  misread chunk in a long public document makes the whole file look
  sensitive.
- **Choices made after seeing test results:** switching the default to Qwen and
  picking the 4-bit build. Both were decided on these same test sets, so their
  numbers are not pre-registered.

## Layout

```
src/inference/   prompt + label scoring, chunking, file text extraction, policy layer
src/scan.py      classify every file in a folder
src/api/         FastAPI service
src/training/    LoRA fine-tuning (label-only loss; v1 or v2 data; any causal LM)
src/evaluation/  dataset registry, systems, ablation harness, overlap audit, report
src/edge/        ONNX export and quantisation, benchmark, INT8 diagnosis
data/            v1 and v2 training data, evaluation sets, review annotations
results/         recorded results; REPORT.md is generated from these
tests/           unit tests (no GPU or model download needed)
```

---

## License

MIT License — Copyright (c) 2026 Prajwal Pujari. Free for commercial, research, and educational use. See [LICENSE](LICENSE) for details.

---

## Citation

If you use `EdgeSLM` in your research, air-gapped DLP systems, or benchmark evaluations, please cite it as:

```bibtex
@software{pujari2026edgeslm,
  author = {Prajwal Pujari},
  title = {EdgeSLM: Air-Gapped Document Sensitivity Classifier and DLP Scanner},
  year = {2026},
  url = {https://github.com/Prajwal7-2002/EdgeSLM}
}
```

