"""Run every system against every evaluation set and record the results.

Writes results/ablation.json and prints a per-system summary.

    python src/evaluation/run_ablation.py
    python src/evaluation/run_ablation.py --systems lora-v2 tfidf-v2data --datasets contrast --merge
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


ADAPTERS = {
    "v1": Path("experiments/smollm2-lora-policy"),
    "v1-labelloss": Path("experiments/smollm2-v1data-labelloss"),
    "v2": Path("experiments/smollm2-v2"),
    "qwen-v2": Path("experiments/qwen0.5b-v2"),
    "qwen-v3": Path("experiments/qwen0.5b-v3"),
}
EDGE_ROOT = Path("experiments/edge")
QWEN_INT4 = Path("experiments/edge-qwen/onnx-int4-weight-only")
RESULTS = Path("results/ablation.json")

# A RESTRICTED document predicted PUBLIC or INTERNAL is the failure this
# system exists to prevent, so it gets its own metric.
UNDER_CLASSIFIED = {"PUBLIC", "INTERNAL"}


ONNX_BUILDS = {
    "onnx-fp32": "v2 adapter merged, ONNX Runtime CPU fp32",
    "onnx-int8-u8s8": "ONNX Runtime CPU, dynamic INT8 with signed weights (v1's setting)",
    "onnx-int8-u8u8": "ONNX Runtime CPU, dynamic INT8 with unsigned weights",
    "onnx-int8-weight-only": "ONNX Runtime CPU, 8-bit weights, fp32 activations",
    "onnx-int4-weight-only": "ONNX Runtime CPU, 4-bit weights, fp32 activations",
    "onnx-int4-int8-compute": "ONNX Runtime CPU, 4-bit weights, activations quantised per block of 32",
}


def system_builders(device):
    """name -> (description, builder, path the builder needs or None)."""
    v1_train = load_jsonl("data/splits/train.jsonl") + load_jsonl(
        "data/raw/boundary_training.jsonl"
    )
    v2_train = v1_train + load_jsonl("data/v2/train.jsonl")

    def model(description, adapter_key=None, **kwargs):
        adapter = ADAPTERS[adapter_key] if adapter_key else None
        return (
            description,
            lambda: ModelSystem(adapter=adapter, device=device, **kwargs),
            adapter,
        )

    builders = {
        "policy-only": ("Regex policy layer alone, no model", PolicyOnlySystem, None),
        "tfidf-v1data": (
            "TF-IDF + logistic regression on the v1 training data",
            lambda: TfidfSystem(v1_train),
            None,
        ),
        "tfidf-v2data": (
            "TF-IDF + logistic regression on the v2 training data (same rows as lora-v2)",
            lambda: TfidfSystem(v2_train),
            None,
        ),
        "base-guided": model(
            "SmolLM2-360M-Instruct, no fine-tuning, label definitions in the prompt, label scoring",
        ),
        "lora-v1": model(
            "v1 adapter as shipped: full-sequence loss, v1 data, generated text parsed for a label",
            "v1",
            decode="generate",
        ),
        "lora-v1-scored": model(
            "Same v1 adapter, scored over the four labels instead of generating", "v1"
        ),
        "lora-v1data-labelloss": model(
            "Retrained on v1 data with the loss on the label only, label scoring",
            "v1-labelloss",
        ),
        "lora-v2": model("v2 adapter: label-only loss, v1 + v2 data, label scoring", "v2"),
        "lora-v2-policy": model(
            "SmolLM2 v2 adapter with the regex policy layer in front",
            "v2",
            use_policy=True,
        ),
        "qwen0.5b-v2": model(
            "Qwen2.5-0.5B-Instruct with the v2 recipe (same data and settings as lora-v2)",
            "qwen-v2",
        ),
        "qwen0.5b-v2-policy": model(
            "Qwen v2 adapter with the regex policy layer in front (the configuration the API serves)",
            "qwen-v2",
            use_policy=True,
        ),
        "qwen0.5b-v3": model(
            "Qwen2.5-0.5B-Instruct with the v3 recipe (balanced realistic emails + docs)",
            "qwen-v3",
        ),
        "qwen0.5b-v3-policy": model(
            "Qwen v3 adapter with the regex policy layer in front",
            "qwen-v3",
            use_policy=True,
        ),
    }
    # Quantisation is only meaningful if it preserves accuracy, so every ONNX
    # build of the v2 adapter is scored by the same harness rather than trusted
    # on latency numbers alone. Available once src/edge/export_onnx.py has run.
    for name, description in ONNX_BUILDS.items():
        builders[name] = (
            description,
            lambda name=name: OnnxSystem(EDGE_ROOT / name),
            EDGE_ROOT / name,
        )
    builders["qwen0.5b-v2-onnx-int4"] = (
        "Qwen2.5-0.5B v2 adapter merged, ONNX Runtime CPU, 4-bit weights",
        lambda: OnnxSystem(QWEN_INT4),
        QWEN_INT4,
    )
    return builders


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

    # On a contrast set a system that keys on shared vocabulary tends to get one
    # side of each pair right by luck, so a pair only counts when both are right.
    pair_accuracy = None
    if rows and all("pair" in row for row in rows):
        pairs = defaultdict(list)
        for row, p in zip(rows, predictions):
            pairs[row["pair"]].append(row["label"] == p)
        pair_accuracy = round(
            sum(all(hits) for hits in pairs.values()) / len(pairs), 4
        )

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
        "pair_accuracy": pair_accuracy,
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
    builders = system_builders(device)

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
            "adapters": {key: str(path) for key, path in ADAPTERS.items()},
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
        description, build, needs = builders[system_name]
        if needs is not None and not Path(needs).exists():
            print(f"\n=== {system_name} === SKIPPED ({needs} missing)", flush=True)
            continue

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
