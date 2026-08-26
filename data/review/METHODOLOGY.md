# Independent Review Set — Methodology and Provenance

## What this set is

`benchmark_labeled.jsonl` is the evaluation set produced by re-annotating the
100 documents in `benchmark_queue.jsonl` from the label definitions alone,
without reference to the label the dataset generator originally assigned and
without seeing any system's predictions.

Its purpose is to break one specific circularity: the training data and the
original test labels came from the same generator, so a model scoring well
against those labels may only be reproducing the generator's conventions.

## What this set is **not**

**No human has reviewed these labels.** The annotation pass in
`annotations/claude-opus-5.jsonl` was produced by Claude Opus 5 — a different
model family from the system under test (SmolLM2-360M), applying the published
label definitions to each document independently.

That makes it an *independent* annotation, not a *human* one. It removes
generator circularity. It does not establish a human gold standard, and it
carries whatever systematic biases a large language model brings to reading
policy definitions. Any claim built on this set should say so plainly.

A human pass is a drop-in: write `annotations/<your-name>.jsonl` with one
`{"id", "label", "notes"}` object per row and re-run
`src/evaluation/merge_annotations.py`. It will report raw agreement and Cohen's
kappa for every annotator pair, and the merged labels become a majority vote.

### Known contamination in the annotator's view

While assembling the queue, the first ten rows of the generator's original
labels were visible in a diagnostic listing before annotation began. Those ten
rows (`review-0001` … `review-0010`, all PUBLIC in both the generator's and the
annotator's labelling) are therefore not fully blind. They are left in the set
and flagged here rather than dropped, since removing them silently would be the
larger distortion.

## Composition and the training-overlap split

The queue was assembled by `src/evaluation/prepare_review_queue.py` from three
sources:

| Source | Rows | In LoRA training data? |
|---|---|---|
| `data/realistic_test.jsonl` | 40 | no |
| `data/raw/boundary_training.jsonl` | 40 | **yes** |
| `data/raw/documents.jsonl` | 20 | no |

`train_lora.py` trains on `data/splits/train.jsonl` **and**
`data/raw/boundary_training.jsonl`, so 40 of the 100 rows are documents the
adapter was fitted on. Evaluating the adapter against the full 100 would report
memorisation as accuracy.

`src/evaluation/datasets.py` therefore splits the set in two, and the ablation
reports them separately:

- **`reviewed-heldout`** (n=60) — never seen in training. The generalisation
  number.
- **`reviewed-seen`** (n=40) — training data. A contamination control, not a
  performance claim.

## Agreement with the generator's original labels

98 of 100 (98.0%). The two disagreements are definitional edge cases rather
than generator errors:

| Row | Generator | Annotator | Why |
|---|---|---|---|
| `review-0081` | PUBLIC | INTERNAL | "Company holiday calendar for 2026." carries no publication signal, so company-only is the default reading — though many organisations do publish one. Genuinely arguable. |
| `review-0100` | RESTRICTED | CONFIDENTIAL | "Security incident investigation containing highly sensitive forensic information." names no credential, key, token, password, or bypass instruction, and RESTRICTED is defined by exactly those. |

High agreement here is partly a property of the documents: these were written
with a target label in mind and mostly state their sensitivity explicitly
("published on the company website", "contains the master password"). It should
not be read as evidence that the label scheme is unambiguous on real corporate
documents, where the hard cases are hard precisely because no such phrase
appears.

## Rows flagged ambiguous by the annotator

`review-0016`, `review-0081`, and `review-0100` are marked AMBIGUOUS in the
notes field, with the competing reading recorded. A second annotator should
start there.
