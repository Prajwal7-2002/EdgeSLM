"""Latency, memory and artifact-size benchmark across inference backends.

This is the measurement behind the "edge" claim: what the model actually costs
to run on a CPU, and what quantisation buys.

    python src/edge/benchmark.py
    python src/edge/benchmark.py --backends torch-cpu-fp32 onnx-int4-weight-only

Each backend runs in its own subprocess. Measuring them one after another in a
single process (as v1 did) lets memory freed by one backend linger in the next
one's numbers, which made INT8 look hungrier than PyTorch fp32. Peak memory is
the OS-reported peak working set of that subprocess, and the baseline (Python,
torch and onnxruntime imported, no model) is recorded so the model's own share
is visible.

Accuracy for the same builds is measured by src/evaluation/run_ablation.py,
which uses the same system classes.
"""

import argparse
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil


DEFAULT_ADAPTER = Path("experiments/smollm2-v2")
EDGE_ROOT = Path("experiments/edge")
# Builds of other adapters, benchmarked alongside the SmolLM2 v2 ones.
EXTRA_ONNX = {
    "qwen0.5b-onnx-int4": (
        "Qwen2.5-0.5B v2 adapter, ONNX Runtime CPU, 4-bit weights",
        Path("experiments/edge-qwen/onnx-int4-weight-only"),
    ),
}
RESULTS = Path("results/edge_benchmark.json")

ONNX_BACKENDS = {
    "onnx-fp32": "ONNX Runtime CPU, fp32",
    "onnx-int8-u8s8": "ONNX Runtime CPU, dynamic INT8, signed weights (v1's setting)",
    "onnx-int8-u8u8": "ONNX Runtime CPU, dynamic INT8, unsigned weights",
    "onnx-int8-weight-only": "ONNX Runtime CPU, 8-bit weights, fp32 activations",
    "onnx-int4-weight-only": "ONNX Runtime CPU, 4-bit weights, fp32 activations",
    "onnx-int4-int8-compute": "ONNX Runtime CPU, 4-bit weights, activations quantised per block of 32",
}


def directory_size_mb(path):
    path = Path(path)
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1024**2, 1)


def base_checkpoint_mb(adapter):
    """Size of the cached base checkpoint plus adapter, which PyTorch loads."""
    config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    repo = "models--" + config["base_model_name_or_path"].replace("/", "--")
    cache = Path.home() / ".cache/huggingface/hub" / repo
    weights = [f for f in cache.rglob("*.safetensors") if f.is_file()] if cache.exists() else []
    if not weights:
        return None
    adapter_mb = sum(f.stat().st_size for f in Path(adapter).glob("*.safetensors")) / 1024**2
    return round(sum(f.stat().st_size for f in weights) / 1024**2 + adapter_mb, 1)


def backend_names():
    import torch

    names = ["torch-cpu-fp32"] + list(ONNX_BACKENDS) + list(EXTRA_ONNX)
    if torch.cuda.is_available():
        names.insert(0, "torch-cuda-fp16")
    return names


def build(name, adapter):
    import torch

    from evaluation.systems import ModelSystem, OnnxSystem

    if name == "torch-cuda-fp16":
        return ModelSystem(adapter=adapter, device="cuda", dtype=torch.float16)
    if name == "torch-cpu-fp32":
        return ModelSystem(adapter=adapter, device="cpu", dtype=torch.float32)
    if name in EXTRA_ONNX:
        return OnnxSystem(EXTRA_ONNX[name][1])
    return OnnxSystem(EDGE_ROOT / name)


def peak_rss_mb(process):
    info = process.memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux, KiB
    return round(peak / 1024**2, 1)


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * len(ordered))) - 1))
    return ordered[index]


def worker(name, adapter, documents, warmup, repeats):
    """Runs inside the subprocess: load one backend, time it, report JSON."""
    import onnxruntime  # noqa: F401  (part of the baseline for every backend)
    import torch

    process = psutil.Process()
    gc.collect()
    baseline_mb = round(process.memory_info().rss / 1024**2, 1)

    started = time.perf_counter()
    system = build(name, adapter)
    load_seconds = time.perf_counter() - started

    for text in documents[:warmup]:
        system.predict(text)
    latencies = []
    for _ in range(repeats):
        for text in documents:
            if name.startswith("torch-cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            system.predict(text)
            if name.startswith("torch-cuda"):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    peak_mb = peak_rss_mb(process)
    entry = {
        "load_seconds": round(load_seconds, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2),
            "p50": round(percentile(latencies, 0.50), 2),
            "p95": round(percentile(latencies, 0.95), 2),
            "min": round(min(latencies), 2),
            "max": round(max(latencies), 2),
        },
        "throughput_docs_per_sec": round(1000 / statistics.fmean(latencies), 2),
        "baseline_rss_mb": baseline_mb,
        "peak_rss_mb": peak_mb,
        "model_rss_mb": round(peak_mb - baseline_mb, 1),
    }
    if name.startswith("torch-cuda"):
        entry["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    print("RESULT " + json.dumps(entry), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--dataset", default="contrast")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--backends", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    from evaluation.datasets import load_dataset

    documents = [row["text"] for row in load_dataset(args.dataset)[: args.limit]]
    if not documents:
        raise SystemExit(f"dataset {args.dataset!r} produced no documents")

    if args.worker:
        worker(args.worker, args.adapter, documents, args.warmup, args.repeats)
        return

    import torch

    names = args.backends or backend_names()
    results = {
        "meta": {
            "adapter": str(args.adapter),
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
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "note": (
                "Single document, batch size 1, one forward pass scoring the four "
                "labels, timed end to end including tokenisation. The policy layer "
                "is disabled so every call exercises the model. Each backend runs in "
                "a fresh process."
            ),
        },
        "backends": {},
    }

    for name in names:
        if name in ONNX_BACKENDS or name in EXTRA_ONNX:
            directory = EXTRA_ONNX[name][1] if name in EXTRA_ONNX else EDGE_ROOT / name
            if not any(directory.glob("*.onnx")):
                print(f"=== {name} === SKIPPED (run src/edge/export_onnx.py)")
                continue
            description = EXTRA_ONNX[name][0] if name in EXTRA_ONNX else ONNX_BACKENDS[name]
            artifact_mb = directory_size_mb(directory)
        else:
            description = {
                "torch-cuda-fp16": "PyTorch, base + LoRA adapter, CUDA fp16",
                "torch-cpu-fp32": "PyTorch, base + LoRA adapter, CPU fp32",
            }[name]
            artifact_mb = base_checkpoint_mb(args.adapter)

        print(f"=== {name} ===", flush=True)
        completed = subprocess.run(
            [
                sys.executable, __file__, "--worker", name,
                "--adapter", str(args.adapter), "--dataset", args.dataset,
                "--limit", str(args.limit), "--warmup", str(args.warmup),
                "--repeats", str(args.repeats),
            ],
            capture_output=True, text=True,
        )
        lines = [l for l in completed.stdout.splitlines() if l.startswith("RESULT ")]
        if completed.returncode != 0 or not lines:
            print(completed.stderr[-2000:])
            raise SystemExit(f"backend {name} failed")

        entry = {"description": description, "artifact_mb": artifact_mb}
        entry.update(json.loads(lines[-1][len("RESULT "):]))
        results["backends"][name] = entry
        print(
            f"  p50={entry['latency_ms']['p50']}ms  p95={entry['latency_ms']['p95']}ms  "
            f"peak={entry['peak_rss_mb']}MB (model {entry['model_rss_mb']}MB)  "
            f"artifact={artifact_mb}MB",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
