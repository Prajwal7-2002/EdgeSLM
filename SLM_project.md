# EdgeSLM Stage 1: Content-Security Classification 

## Objective

Build and evaluate a local-first document classifier that assigns company text to one of four handling labels:

- `PUBLIC` — approved for external sharing.
- `INTERNAL` — routine company-only procedures or workflows.
- `CONFIDENTIAL` — private business, employee, customer, or contract information.
- `RESTRICTED` — credentials, keys, tokens, passwords, or access-bypass instructions.

The system must keep document contents on the local machine, provide an API suitable for local integration, and report evidence of both quality and safety. The work is not complete when a model scores well on generated examples: it must be measured on independently held-out, realistic, and adversarial documents.

## Target Architecture

```text
[Local client]
      |
      | POST /classify { text, use_policy }
      v
+------------------------------------------+
| FastAPI validation and service layer     |
+----------------------+-------------------+
                       |
          +------------+------------+
          |                         |
          v                         v 
+----------------------+   +--------------------------+
| Regex policy layer   |   | SmolLM2-360M + LoRA      |
| high-confidence      |   | guided, greedy label     |
| overrides            |   | generation               |
+----------+-----------+   +------------+-------------+
           |                            |
           +-------------+--------------+
                         v
            [label, confidence, provenance,
             device, latency]

Offline evaluation and edge delivery:

data generation/splits --> LoRA training --> evaluation harness --> REPORT.md
                                                |
                                                +--> merged FP32 / ONNX FP32 / INT8 variants
```

## Functional Requirements

1. **Local inference service.** Implement `GET /health`, `GET /version`, and `POST /classify` with FastAPI. Validate non-empty text and cap request size. The classification response must include the final label, model prediction, whether a policy rule overrode it, relative confidence, device, model version, and latency.
2. **Small-model fine-tuning.** Fine-tune `HuggingFaceTB/SmolLM2-360M-Instruct` using a LoRA adapter. Use a stable prompt containing the four label definitions, train only attention projection adapters (`q_proj` and `v_proj`), and save the adapter separately from the base model.
3. **Safety policy before model inference.** Add deterministic regex rules for high-confidence restricted content such as passwords, private keys, API keys, bearer tokens, credentials, and access-control bypass instructions. Include narrower explicit rules for public and confidential content. When a rule matches, short-circuit model inference and record the decision as a policy override.
4. **Evaluation that resists misleading results.** Evaluate each system on generated held-out data, hand-written realistic prose, adversarial keyword traps, and an independently re-annotated review set. Keep any re-annotated training examples visibly separate from genuinely held-out examples.
5. **Comparable baselines and ablations.** Measure at least a regex-only policy, TF-IDF + logistic regression trained on the same data, the base model with and without label guidance, LoRA with and without policy, and deployment backends where available.
6. **Edge deployment experiment.** Merge the LoRA adapter into the base model, export FP32 ONNX, and assess dynamic INT8 variants. Measure load time, artifact size, peak memory, and p50/p95 end-to-end latency; pair speed results with held-out accuracy so an unusable quantized build cannot be presented as a success.
7. **Reproducible reporting.** Store machine-readable run outputs under `results/` and generate `REPORT.md` from them. Report accuracy, macro F1, `RESTRICTED` recall, restricted documents released as `PUBLIC` or `INTERNAL`, non-answer rate, confusion matrices, and 95% Wilson confidence intervals.

## Evaluation and Safety Criteria

The main safety failure is under-classifying a `RESTRICTED` document as `PUBLIC` or `INTERNAL`. Treat it separately from ordinary accuracy errors. Policy rules may improve recall, but the report must disclose both their coverage and correctness whenever they fire; otherwise a policy-assisted score may be mistaken for model performance.

Generated train/test splits are useful only as in-distribution checks. They must not be the headline generalization claim when they share templates or topic lists. The headline result must be drawn from data the adapter did not see during training.

## Current Reference Results

The existing implementation is intentionally candid about its outcome on the 60-document reviewed held-out set:

| System | Accuracy | `RESTRICTED` recall |
|---|---:|---:|
| TF-IDF + logistic regression | 91.7% | 100.0% |
| LoRA + policy layer (served configuration) | 65.0% | 92.9% |
| LoRA without policy | 58.3% | 71.4% |
| Regex policy only | 31.7% | 78.6% |
| Guided base model | 23.3% | 0.0% |

These figures establish the learning objective: validate whether a language model adds real value over a cheap baseline, rather than assuming that fine-tuning is justified. In the current study, the TF-IDF baseline wins, and dynamic INT8 substantially reduces model correctness despite reducing size and latency.

## Suggested Implementation Sequence

### Prompt 1: Labels, Dataset, and Local Service

> Act as a machine-learning engineer building a local document-security classifier. Create a Python project that classifies text as `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, or `RESTRICTED`. Define concise label guidance, generate/split JSONL data without train/test leakage, and implement a FastAPI app with `GET /health`, `GET /version`, and `POST /classify`. Validate request text with Pydantic and return label provenance, confidence, device, model version, and latency. Do not transmit document contents to an external service.

### Prompt 2: LoRA Classifier and Deterministic Policy

> Implement a LoRA fine-tuning script for `HuggingFaceTB/SmolLM2-360M-Instruct` using a consistent classification prompt and adapters on `q_proj` and `v_proj`. Then implement inference that extracts exactly one of the four labels from greedy model output. Add a deterministic regex policy layer that short-circuits inference for credentials, passwords, private keys, API keys, bearer tokens, and access-bypass instructions, with carefully scoped public and confidential overrides. Return both the model prediction and final policy-adjusted label so policy contribution is auditable.

### Prompt 3: Honest Evaluation and Edge Readiness

> Build an evaluation harness that compares policy-only, TF-IDF + logistic regression, guided/unguided base model, LoRA without policy, and LoRA with policy on generated, realistic, adversarial, reviewed-heldout, and reviewed-seen datasets. Compute accuracy, macro F1, per-class support, restricted recall, restricted documents released as public/internal, non-answer rate, confusion matrices, and Wilson 95% intervals. Create an annotation merge step that can calculate Cohen's kappa. Export a merged LoRA model to FP32 ONNX and dynamic INT8 variants, benchmark latency/memory/artifact size, and generate a Markdown report solely from saved result JSON.

## Acceptance Checklist

- `pytest` covers policy precedence, metrics, label extraction, and annotation agreement.
- `POST /classify` can run locally with the adapter path supplied through `EDGESLM_ADAPTER`.
- Evaluation datasets are documented with sizes, origin, and held-out status.
- Training contamination, small sample sizes, and non-human labels are disclosed rather than hidden.
- Every edge benchmark is accompanied by its corresponding quality measurement.
- `REPORT.md` is generated from recorded results, not manually edited.

