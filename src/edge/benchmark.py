"""Latency, memory and artifact-size benchmark across inference backends.

This is the measurement behind the "edge" claim: what the model actually costs
to run on a CPU, and what quantisation buys.

    python src/edge/benchmark.py
    python src/edge/benchmark.py --backends torch-cpu-fp32 onnx-cpu-int8

Accuracy for the same backends is measured by src/evaluation/run_ablation.py,
which shares the system classes so the two reports describe the same builds.
"""

import argparse
import gc
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil
import torch

from evaluation.datasets import load_dataset
from evaluation.systems import ModelSystem, OnnxSystem


DEFAULT_ADAPTER = Path("experiments/smollm2-lora-policy")
EDGE_ROOT = Path("experiments/edge")
RESULTS = Path("results/edge_benchmark.json")


def directory_size_mb(path):
    path = Path(path)
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1024**2, 1)


def base_model_size_mb():
    """Size of the cached base checkpoint, which the PyTorch backends load."""
    cache = Path.home() / ".cache/huggingface/hub"
    matches = list(cache.glob("models--HuggingFaceTB--SmolLM2-360M-Instruct"))
    if not matches:
        return None
    weights = [
        f
        for f in matches[0].rglob("*")
        if f.is_file() and f.suffix in {".safetensors", ".bin"}
    ]
    if not weights:
        return None
    return round(sum(f.stat().st_size for f in weights) / 1024**2, 1)


def backend_specs(adapter):
    """Backends to measure.

    Each entry is (description, builder, artifact size, source directory). The
    directory is stored explicitly so availability checks never depend on
    reconstructing a path from the backend name.
    """
    onnx_fp32 = EDGE_ROOT / "onnx-fp32"
    onnx_int8 = EDGE_ROOT / "onnx-int8"
    onnx_int8_channel = EDGE_ROOT / "onnx-int8-perchannel"
    onnx_int8_safehead = EDGE_ROOT / "onnx-int8-safehead"

    specs = {
        "torch-cpu-fp32": (
            "PyTorch, base + LoRA adapter, CPU fp32",
            lambda: ModelSystem(adapter=adapter, device="cpu", dtype=torch.float32),
            base_model_size_mb(),
            None,
        ),
        "onnx-cpu-fp32": (
            "ONNX Runtime, merged adapter, CPU fp32",
            lambda: OnnxSystem(onnx_fp32),
            directory_size_mb(onnx_fp32),
            onnx_fp32,
        ),
        "onnx-cpu-int8": (
            "ONNX Runtime, merged adapter, CPU dynamic INT8 (per-tensor)",
            lambda: OnnxSystem(onnx_int8),
            directory_size_mb(onnx_int8),
            onnx_int8,
        ),
        "onnx-cpu-int8-perchannel": (
            "ONNX Runtime, merged adapter, CPU dynamic INT8 (per-channel)",
            lambda: OnnxSystem(onnx_int8_channel),
            directory_size_mb(onnx_int8_channel),
            onnx_int8_channel,
        ),
        "onnx-cpu-int8-safehead": (
            "ONNX Runtime, merged adapter, CPU dynamic INT8, fp32 output projection",
            lambda: OnnxSystem(onnx_int8_safehead),
            directory_size_mb(onnx_int8_safehead),
            onnx_int8_safehead,
        ),
    }
    if torch.cuda.is_available():
        specs["torch-cuda-fp16"] = (
            "PyTorch, base + LoRA adapter, CUDA fp16",
            lambda: ModelSystem(adapter=adapter, device="cuda", dtype=torch.float16),
            base_model_size_mb(),
            None,
        )
    return specs


def available(spec):
    """Skip ONNX backends when the export has not been produced yet."""
    directory = spec[3]
    if directory is None:
        return True
    return directory.exists() and any(directory.glob("*.onnx"))


def measure(system, documents, warmup, repeats):
    for text in documents[:warmup]:
        system.predict(text)

    latencies = []
    for _ in range(repeats):
        for text in documents:
            start = time.perf_counter()
            system.predict(text)
            latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", default="realistic")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()

    rows = load_dataset(args.dataset)[: args.limit]
    documents = [row["text"] for row in rows]
    if not documents:
        raise SystemExit(f"dataset {args.dataset!r} produced no documents")

    specs = backend_specs(args.adapter)
    names = args.backends or list(specs)

    process = psutil.Process()
    results = {
        "meta": {
            "dataset": args.dataset,
            "documents": len(documents),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "measurements_per_backend": len(documents) * args.repeats,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "torch": torch.__version__,
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "note": (
                "Latency is single-document, batch size 1, greedy decode capped "
                "at 5 new tokens, measured end to end including tokenisation. "
                "The policy layer is disabled so every measurement exercises the "
                "model rather than a regex short-circuit."
            ),
        },
        "backends": {},
    }

    for name in names:
        if name not in specs:
            raise SystemExit(f"unknown backend {name!r}; choose from {list(specs)}")
        if not available(specs[name]):
            print(f"\n=== {name} === SKIPPED (export missing; run export_onnx.py)")
            continue

        description, build, artifact_mb, _directory = specs[name]
        print(f"\n=== {name} ===", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rss_before = process.memory_info().rss

        started = time.perf_counter()
        system = build()
        load_seconds = time.perf_counter() - started

        latencies = measure(system, documents, args.warmup, args.repeats)
        rss_peak = process.memory_info().rss

        entry = {
            "description": description,
            "load_seconds": round(load_seconds, 2),
            "artifact_mb": artifact_mb,
            "latency_ms": {
                "mean": round(statistics.fmean(latencies), 2),
                "p50": round(percentile(latencies, 0.50), 2),
                "p95": round(percentile(latencies, 0.95), 2),
                "min": round(min(latencies), 2),
                "max": round(max(latencies), 2),
            },
            "throughput_docs_per_sec": round(
                1000 / statistics.fmean(latencies), 2
            ),
            "process_rss_mb": round(rss_peak / 1024**2, 1),
            "process_rss_delta_mb": round((rss_peak - rss_before) / 1024**2, 1),
        }
        if name.startswith("torch-cuda"):
            entry["cuda_peak_mb"] = round(
                torch.cuda.max_memory_allocated() / 1024**2, 1
            )

        results["backends"][name] = entry
        print(
            f"  load={entry['load_seconds']}s  "
            f"p50={entry['latency_ms']['p50']}ms  "
            f"p95={entry['latency_ms']['p95']}ms  "
            f"rss={entry['process_rss_mb']}MB  "
            f"artifact={artifact_mb}MB",
            flush=True,
        )

        system.close()
        del system
        gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
