# EdgeSLM Content Security Classifier

Sorts company documents into `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, or
`RESTRICTED`. It's a LoRA fine-tune of SmolLM2-360M with a regex policy layer
bolted on top for credentials and keys, served from a local FastAPI app so the
documents never leave the machine.

I started this to fine-tune a small model. What I actually ended up with is an
evaluation project, because once I measured it properly the results weren't
what I expected. [REPORT.md](REPORT.md) has everything.

## What the numbers say

This is on `reviewed-heldout` — 60 documents I re-labelled independently, none
of which the adapter saw during training. Brackets are 95% Wilson intervals.

| System | Accuracy | RESTRICTED recall |
|---|---|---|
| TF-IDF + logistic regression | 91.7% [82–96] | 100% |
| LoRA + policy layer (what the API serves) | 65.0% [52–76] | 92.9% |
| LoRA on its own | 58.3% [46–70] | 71.4% |
| Just the regex layer | 31.7% [21–44] | 78.6% |
| SmolLM2-360M with label definitions | 23.3% [14–35] | 0% |
| SmolLM2-360M zero-shot | 0.0% [0–6] | 0% |

A few things fell out of this:

**TF-IDF beats my fine-tune by 27 points.** Same training data, trains in about
a second on a CPU. I didn't expect that and I'm not going to hide it — for this
task, as I've currently set it up, the language model isn't paying for itself.

**Fine-tuning was still necessary though.** The base model is genuinely useless
here. Give it a bare prompt and it starts writing a multiple-choice question
(`"\n\nA. Sensitive\nB"`), so nothing parses at all. Add the label definitions
and it answers, but it says `PUBLIC` 59 times out of 60. So the 23.3% is just
the class balance, not the model knowing anything.

**My old 94.7% number was measuring the wrong thing.** That's the score on the
generated test split, and it drops to 58.3% on real prose. The giveaway is that
TF-IDF gets a clean 100% on that same split — train and test come out of the
same templates and topic lists, so there's barely anything to learn.

**The regex layer is doing a lot of the work.** It runs before the model and
short-circuits it whenever a pattern hits, which on held-out prose is 35% of
documents — the model never sees them. It does help safety a lot (RESTRICTED
recall goes 71% → 93%), but it means the shipped accuracy isn't really the
model's accuracy, and I'd been reporting it as though it were.

**INT8 quantisation breaks it.** 1.8× faster on CPU and 4× smaller on disk, and
accuracy falls off a cliff — 58.3% down to 25%. I tried per-channel scaling and
I tried leaving the output projection in fp32; neither helped. The fp32 ONNX
export scores identically to PyTorch, so the export itself is fine. It's the
quantisation.

## The evaluation sets

| Set | n | Held out | What it is |
|---|---|---|---|
| `generated-test` | 152 | yes | Same templates as training. In-distribution. |
| `realistic` | 40 | yes | Prose I wrote by hand. |
| `adversarial` | 40 | yes | Traps — public docs full of credential words, internal docs that say "no passwords here". |
| `reviewed-heldout` | 60 | yes | Re-labelled independently. This is the number I trust most. |
| `reviewed-seen` | 40 | **no** | Also re-labelled, but it's in the training data. |

That last row is worth explaining. `prepare_review_queue.py` built the review
set partly from `boundary_training.jsonl`, and `train_lora.py` trains on that
file. So 40 of my 100 "independent" documents were things the model had already
memorised. I split them out instead of deleting them, so the contamination
shows up as a number rather than quietly inflating everything else.

### About those independent labels

I re-labelled all 100 documents from the definitions alone, without looking at
what the generator had assigned. Agreement came out at 98/100, and the two
disagreements are both genuinely arguable edge cases.

One caveat I want to be upfront about: **no human has checked these labels.**
The annotation pass came from a different model than the one being tested. That
kills the circularity of grading a model against labels from its own data
generator, but it isn't a human gold standard and I'm not going to call it one.

If you want to add a human pass, drop a file into `data/review/annotations/`
and re-run the merge — it'll report Cohen's kappa between annotators.
[data/review/METHODOLOGY.md](data/review/METHODOLOGY.md) has the details.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Adapters aren't in Git (too big). Put yours at
`experiments/smollm2-lora-policy`.

## Running it

```powershell
python -m uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

`GET /health` · `GET /version` · `POST /classify` · `GET /docs`

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "The deployment vault stores bearer tokens for production."}'
```

## Reproducing the results

```bash
python src/evaluation/merge_annotations.py   # build the reviewed eval set
python src/evaluation/run_ablation.py        # -> results/ablation.json
python src/edge/export_onnx.py               # ONNX + INT8 builds
python src/edge/benchmark.py                 # -> results/edge_benchmark.json
python src/evaluation/render_report.py       # -> REPORT.md
pytest
```

`run_ablation.py --merge` adds systems to an existing results file without
re-running everything, which saves a lot of time when you're only changing one
thing. `--device cpu` forces CPU.

`audit_dataset.py` checks the splits for duplicates and leakage.

REPORT.md is generated, not written — so the numbers in it can't drift from the
numbers the harness actually produced.

## Layout

```
src/inference/     classifier + the regex policy layer
src/api/           FastAPI service
src/training/      LoRA fine-tuning
src/evaluation/    dataset registry, systems, ablation harness, report generator
src/edge/          ONNX export, quantisation, latency/memory benchmark
data/review/       annotation queue, annotator passes, methodology
results/           recorded results (tracked — REPORT.md is built from these)
```

## Where this actually stands

It's a prototype. Things I know are wrong with it:

- The training data is template-generated, so its test split is close to free.

- The held-out sets are small (40–60). The confidence intervals in REPORT.md are
  wide and several of them overlap, so don't read too much into small gaps
  between systems.
- I wrote all the evaluation documents myself, and the "independent" labels came
  from a model, not a person.
- The documents mostly announce their own sensitivity — "published on the
  company website", "contains the master password". Real documents are hard
  precisely because they don't do that.
- INT8 is currently unusable. Only the fp32 ONNX build is trustworthy.

Given that TF-IDF wins, the next thing worth doing isn't a bigger adapter. It's
building an evaluation set where a bag-of-words model *can't* win, and seeing
whether this task needs a language model at all.
