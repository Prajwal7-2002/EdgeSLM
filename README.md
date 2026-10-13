# EdgeSLM: a document-sensitivity classifier that runs on a laptop CPU

[![tests](https://github.com/Prajwal7-2002/EdgeSLM/actions/workflows/tests.yml/badge.svg)](https://github.com/Prajwal7-2002/EdgeSLM/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

EdgeSLM labels a company document `PUBLIC`, `INTERNAL`, `CONFIDENTIAL` or
`RESTRICTED`. It is a LoRA fine-tune of a small open language model
(Qwen2.5-0.5B by default, SmolLM2-360M as the smaller option), served by a
local FastAPI app so documents never leave the machine. The 4-bit ONNX build is
392 MB, classifies a document in about half a second on a laptop CPU, and
needs about 1 GB of RAM.

**Read [On real documents](#on-real-documents-it-does-not-work-yet) before
trusting any number here:** on 40 real emails and handbook pages it currently
does worse than always answering INTERNAL.

Most of this repo is about measuring that classifier honestly: against a
bag-of-words baseline, on a test set built to defeat keyword matching, with
paired significance tests, and with every quantised build scored for accuracy
as well as speed. Every number here comes from `results/` and appears in full
in [REPORT.md](REPORT.md), which is generated from those files.

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
| **Qwen2.5-0.5B LoRA v2** | **88.8%** [80–94] | **31/40** | 90.0% [80–95] | 90% |
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
document ──► regex policy layer ──(rule fires)──► label
                    │
               (no rule)
                    ▼
     prompt with the four label definitions
                    ▼
     Qwen2.5-0.5B + LoRA, one forward pass
                    ▼
 logits of each label's first token ──► softmax ──► label + four scores
```

- **Scoring instead of generating.** Each label starts with a different token,
  so the classifier reads four logits from one forward pass instead of
  generating text and parsing it. It always gives an answer, and needs no
  decode loop or KV cache, which keeps the ONNX graph simple.
- **Label-only loss.** Training uses prompt/completion pairs, so the loss falls
  on the label tokens alone. LoRA rank 16 on every linear layer, 3 epochs on
  2,360 documents: about 21 minutes for Qwen and 28 for SmolLM2 on an RTX 3050
  (6 GB).
- **The policy layer is a trade-off, not a free win.** Regexes for words like
  `password` and `bearer token` run first and override the model. With Qwen
  they catch one more RESTRICTED document on `contrast` (90% → 95% recall) but
  cost 5 points of accuracy (88.8% → 83.8%), because they also fire on public
  help articles about passwords. The API keeps them on, because missing a
  secret is the expensive mistake; send `"use_policy": false` to turn them off.

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

## Quickstart

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in
`uv.lock`. On Windows it installs the CUDA 12.4 build of PyTorch, on Linux the
CPU build.

```powershell
uv sync
uv run pytest
```

The tests need no GPU and no model download. Adapters are not in Git: train one
(`python src/training/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --output experiments/qwen0.5b-v2`)
or point `EDGESLM_ADAPTER` at an existing one, then start the API:

```powershell
uv run python -m uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Use svc_ingest / Copper!2291 for the ledger service until Monday."}'
```

The response gives the label, the probability of each of the four labels, and
whether the policy layer decided instead of the model.

### Real files and long documents

```bash
curl -X POST http://127.0.0.1:8000/classify-file -F "file=@board_minutes.docx"
uv run python src/scan.py path/to/folder --csv results/scan.csv
```

`/classify-file` and `src/scan.py` read `.pdf`, `.docx`, `.eml` and plain-text
formats (`.txt`, `.md`, `.csv`, `.env`, `.yaml`, source files and so on). A
scanned PDF with no text layer is rejected, not guessed at.

To test on real documents rather than ones written for this project:

```bash
uv run python src/data/download_real_docs.py --sec-user-agent "Your Name you@example.com"
# label each file in real_docs/labels.csv yourself, then:
uv run python src/scan.py real_docs --labels real_docs/labels.csv
```

This pulls a varied sample of Enron emails (sampled across mailboxes by
keyword so passwords, pay, deals, logistics and announcements all appear),
GitLab Handbook pages and SEC press releases into `real_docs/`, which is
gitignored because the Enron emails contain real people's details. The script
never assigns labels; the scanner reports agreement, RESTRICTED recall and
every disagreement against yours.

Documents longer than one prompt are split into overlapping chunks of about
435 tokens, each chunk is classified, and the document takes the label of its
most sensitive chunk, so a password on page 9 still makes the file RESTRICTED.
The response lists every chunk's label so a verdict can be traced to the chunk
behind it. Short documents are a single chunk, classified exactly as in the
evaluation; re-running the evaluation with chunking on changed no prediction.

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

## On real documents: it does not work yet

The numbers above come from short documents written for this project. On 40
real ones (20 Enron emails, 20 GitLab Handbook pages, fetched by
`src/data/download_real_docs.py`) every trained system loses to simply
answering INTERNAL every time
([results/real_docs_summary.json](results/real_docs_summary.json)):

| System | Agreement on 40 real documents |
|---|---|
| Always answer INTERNAL (no model) | **26/40** |
| TF-IDF, v2 data | 19/40 |
| TF-IDF, v1 data | 16/40 |
| Qwen2.5-0.5B v2 | 11/40 |
| SmolLM2-360M v2 | 11/40 |

The models over-flag. Qwen called 19 of 26 INTERNAL documents and 7 of 9
PUBLIC ones CONFIDENTIAL or RESTRICTED, usually with 97–99% confidence. Real
email is full of names, phone numbers and addresses, and the synthetic training
data mostly put those in CONFIDENTIAL examples, so the model appears to have
learned "people and numbers mean CONFIDENTIAL". Combining chunks by "most
sensitive wins" makes it worse (11/40; taking the first chunk would give 18/40),
but even single-chunk documents are only 4 of 14 right, so the model is the main
problem, not the chunking. It did catch the one real credential leak (an email
saying everyone's password equals their username).

Caveats: 40 documents, only 5 of them CONFIDENTIAL or RESTRICTED, and the labels
were assigned by Claude after the model's predictions had been seen, not by a
person. The direction of the result is clear anyway: **a model trained only on
synthetic sentences does not transfer to real documents.** The next step is
training data drawn from real documents, not more synthetic templates.

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
- **It fails on real documents** (see the section above). Every training and
  evaluation document behind the main results is one or two synthetic
  sentences.
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
