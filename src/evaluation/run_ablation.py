"""Run every system against every evaluation set and record the results.

Writes results/ablation.json and prints a per-system summary.

    python src/evaluation/run_ablation.py
    python src/evaluation/run_ablation.py --device cpu --systems lora-policy
"""

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.datasets import DATASETS, load_dataset, load_jsonl
from evaluation.metrics import wilson_interval
from evaluation.systems import (
    LABELS,
    ModelSystem,
    OnnxSystem,
    PolicyOnlySystem,
    TfidfSystem,
)


DEFAULT_ADAPTER = Path("experiments/smollm2-lora-policy")
EDGE_ROOT = Path("experiments/edge")
RESULTS = Path("results/ablation.json")

# A RESTRICTED document predicted PUBLIC or INTERNAL is the failure this
# system exists to prevent, so it gets its own metric.
UNDER_CLASSIFIED = {"PUBLIC", "INTERNAL"}


def system_builders(adapter, device):
    train_rows = load_jsonl("data/splits/train.jsonl") + load_jsonl(
        "data/raw/boundary_training.jsonl"
    )
    return {
        "policy-only": (
            "Regex policy layer alone, no model",
            lambda: PolicyOnlySystem(),
        ),
        "tfidf-logreg": (
            "TF-IDF + logistic regression on the same training data",
            lambda: TfidfSystem(train_rows),
        ),
        "base-zero-shot": (
            "SmolLM2-360M-Instruct, no label definitions in the prompt",
            lambda: ModelSystem(adapter=None, guidance=False, device=device),
        ),
        "base-guided": (
            "SmolLM2-360M-Instruct with label definitions in the prompt",
            lambda: ModelSystem(adapter=None, guidance=True, device=device),
        ),
        "lora-no-policy": (
            "LoRA adapter, policy layer disabled",
            lambda: ModelSystem(adapter=adapter, guidance=True, device=device),
        ),
        "lora-policy": (
            "LoRA adapter + policy layer (the shipped configuration)",
            lambda: ModelSystem(
                adapter=adapter, guidance=True, use_policy=True, device=device
            ),
        ),
        # Quantisation is only meaningful if it preserves accuracy, so the ONNX
        # builds are scored by the same harness rather than trusted on latency
        # numbers alone. Available once src/edge/export_onnx.py has run.
        "onnx-fp32-no-policy": (
            "ONNX Runtime CPU fp32, merged adapter, policy layer disabled",
            lambda: OnnxSystem(EDGE_ROOT / "onnx-fp32"),
        ),
        "onnx-int8-no-policy": (
            "ONNX Runtime CPU dynamic INT8 (per-tensor scales), policy disabled",
            lambda: OnnxSystem(EDGE_ROOT / "onnx-int8"),
        ),
        "onnx-int8-perchannel-no-policy": (
            "ONNX Runtime CPU dynamic INT8 (per-channel scales), policy disabled",
            lambda: OnnxSystem(EDGE_ROOT / "onnx-int8-perchannel"),
        ),
        "onnx-int8-safehead-no-policy": (
            "ONNX Runtime CPU dynamic INT8 with the output projection left in "
            "fp32, policy disabled",
            lambda: OnnxSystem(EDGE_ROOT / "onnx-int8-safehead"),
        ),
    }


def onnx_export_missing(system_name):
    """ONNX systems need src/edge/export_onnx.py to have run first."""
    if not system_name.startswith("onnx-"):
        return False
    directory = EDGE_ROOT / system_name.replace("-no-policy", "")
    return not (directory.exists() and any(directory.glob("*.onnx")))


def score(rows, predictions):
    total = len(rows)
    actuals = [row["label"] for row in rows]
    correct = sum(a == p for a, p in zip(actuals, predictions))
    lower, upper = wilson_interval(correct, total)

    per_class = {}
    f1s = []
    for label in LABELS:
        tp = sum(a == label and p == label for a, p in zip(actuals, predictions))
        fp = sum(a != label and p == label for a, p in zip(actuals, predictions))
        fn = sum(a == label and p != label for a, p in zip(actuals, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": tp + fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        if tp + fn:
            f1s.append(f1)

    restricted_support = sum(a == "RESTRICTED" for a in actuals)
    dangerous = sum(
        a == "RESTRICTED" and p in UNDER_CLASSIFIED
        for a, p in zip(actuals, predictions)
    )

    confusion = defaultdict(Counter)
    for a, p in zip(actuals, predictions):
        confusion[a][p] += 1

    non_answers = sum(p in ("UNKNOWN", "ABSTAIN") for p in predictions)

    return {
        "n": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "wilson_95": [round(lower, 4), round(upper, 4)],
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_class": per_class,
        "restricted_recall": (
            per_class["RESTRICTED"]["recall"] if restricted_support else None
        ),
        "restricted_under_classified": dangerous,
        "restricted_support": restricted_support,
        "non_answer_rate": round(non_answers / total, 4),
        "confusion": {a: dict(c) for a, c in confusion.items()},
    }


def coverage_stats(rows, predictions):
    """Extra detail for rule-based systems: how often a rule fires, and whether
    it is right when it does."""
    fired = [
        (row["label"], prediction)
        for row, prediction in zip(rows, predictions)
        if prediction != "ABSTAIN"
    ]
    if not fired:
        return {"fire_rate": 0.0, "accuracy_when_fired": None, "fired_n": 0}
    right = sum(a == p for a, p in fired)
    return {
        "fire_rate": round(len(fired) / len(rows), 4),
        "accuracy_when_fired": round(right / len(fired), 4),
        "fired_n": len(fired),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--device", default=None)
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="update only the systems run here, keeping others already recorded",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    builders = system_builders(args.adapter, device)

    system_names = args.systems or list(builders)
    dataset_names = args.datasets or list(DATASETS)

    datasets = {name: load_dataset(name) for name in dataset_names}
    for name, rows in datasets.items():
        if not rows:
            raise SystemExit(
                f"dataset {name!r} is empty; run merge_annotations.py first"
            )

    results = {
        "meta": {
            "adapter": str(args.adapter),
            "device": device,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "platform": platform.platform(),
            "datasets": {
                name: {
                    "n": len(datasets[name]),
                    "held_out": DATASETS[name]["held_out"],
                    "note": DATASETS[name]["note"],
                }
                for name in dataset_names
            },
        },
        "systems": {},
    }

    for system_name in system_names:
        if onnx_export_missing(system_name):
            print(
                f"\n=== {system_name} === SKIPPED "
                "(no ONNX export; run src/edge/export_onnx.py)",
                flush=True,
            )
            continue

        description, build = builders[system_name]
        print(f"\n=== {system_name} ===", flush=True)
        started = time.perf_counter()
        system = build()
        load_seconds = time.perf_counter() - started

        entry = {
            "description": description,
            "load_seconds": round(load_seconds, 2),
            "datasets": {},
        }

        for dataset_name in dataset_names:
            rows = datasets[dataset_name]
            predictions = []
            latencies = []
            for row in rows:
                start = time.perf_counter()
                label, _confidence = system.predict(row["text"])
                latencies.append((time.perf_counter() - start) * 1000)
                predictions.append(label)

            summary = score(rows, predictions)
            summary["mean_latency_ms"] = round(sum(latencies) / len(latencies), 2)
            if system_name == "policy-only":
                summary["coverage"] = coverage_stats(rows, predictions)
            summary["predictions"] = predictions
            entry["datasets"][dataset_name] = summary

            print(
                f"  {dataset_name:18s} n={summary['n']:4d} "
                f"acc={summary['accuracy'] * 100:5.1f}%  "
                f"macroF1={summary['macro_f1']:.3f}  "
                f"restricted_recall={summary['restricted_recall']}",
                flush=True,
            )

        results["systems"][system_name] = entry
        system.close()

    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        merged = dict(previous.get("systems", {}))
        merged.update(results["systems"])
        results["systems"] = merged
        # Keep whichever device recorded the model systems visible, since a
        # merged file can mix a CUDA sweep with a CPU ONNX sweep.
        results["meta"]["merged_from"] = previous["meta"].get("device")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
