"""Benchmark inference performance: Sequential vs. Batched chunk execution & Speculative Early-Exit.

Run:
    uv run python src/edge/benchmark_inference.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from inference.classifier import DEFAULT_ADAPTER, DocumentClassifier
from inference.prompt import build_prompt, label_token_ids, LABELS


def benchmark_batching(classifier, num_iterations=10):
    sample_chunks = [
        "Meeting agenda for Tuesday: review sprint goals, discuss backend deployment, and triage bugs.",
        "Vendor agreement: Pricing schedule details standard SLA terms of 99.9% uptime and net-30 terms.",
        "Internal engineering procedure: Deploying services to staging via GitHub Actions workflow.",
        "Draft financial statement: Preliminary Q2 revenue estimates indicate positive operating cash flow.",
        "HR memo: Annual open enrollment period for health benefits begins on November 1st.",
        "Customer renewal contract: Master services agreement addendum for account-specific pricing concessions.",
        "Incident postmortem: Root cause analysis of database connection pool exhaustion on production cluster.",
        "Public product roadmap: Features planned for the upcoming Q4 open source community release.",
    ]

    results = {}
    for count in [1, 2, 4, 8]:
        sub_chunks = sample_chunks[:count]

        # Warmup
        _ = classifier.batch_scores(sub_chunks)
        if classifier.device == "cuda":
            torch.cuda.synchronize()

        # Sequential
        seq_times = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            for c in sub_chunks:
                _ = classifier.scores(c)
            if classifier.device == "cuda":
                torch.cuda.synchronize()
            seq_times.append((time.perf_counter() - t0) * 1000)

        # Batched
        batch_times = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = classifier.batch_scores(sub_chunks)
            if classifier.device == "cuda":
                torch.cuda.synchronize()
            batch_times.append((time.perf_counter() - t0) * 1000)

        mean_seq = round(sum(seq_times) / len(seq_times), 2)
        mean_batch = round(sum(batch_times) / len(batch_times), 2)
        speedup = round(mean_seq / mean_batch, 2)
        pct_saved = round((mean_seq - mean_batch) / mean_seq * 100, 1)

        results[f"{count}_chunks"] = {
            "sequential_ms": mean_seq,
            "batched_ms": mean_batch,
            "speedup": f"{speedup}x",
            "time_saved_pct": f"{pct_saved}%",
            "sequential_throughput_chunks_per_sec": round(count / (mean_seq / 1000), 1),
            "batched_throughput_chunks_per_sec": round(count / (mean_batch / 1000), 1),
        }
        print(f"[{count} Chunks] Sequential: {mean_seq} ms | Batched: {mean_batch} ms | Speedup: {speedup}x ({pct_saved}% saved)")

    return results


def benchmark_early_exit(classifier):
    # Simulated multi-page sensitive document where breach appears on page 1
    doc_with_early_breach = (
        "CONFIDENTIAL EMPLOYEE SALARY MEMO\n"
        "Employee: John Doe. Base Salary: $185,000. Performance Bonus: $40,000.\n"
        "This compensation analysis is strictly confidential and restricted to executive leadership.\n"
        + ("\n" + "Routine corporate policy handbook filler text on workplace safety procedures. " * 300)
    )

    # 1. Full scan (processes all chunks)
    t0 = time.perf_counter()
    full_res = classifier.classify(doc_with_early_breach, use_policy=False, early_exit=False)
    full_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Speculative Early-Exit scan
    t0 = time.perf_counter()
    exit_res = classifier.classify(doc_with_early_breach, use_policy=False, early_exit=True, early_exit_threshold=0.90)
    exit_time_ms = round((time.perf_counter() - t0) * 1000, 2)

    chunks_evaluated_full = full_res["chunks"]
    chunks_evaluated_exit = exit_res["chunks"]
    compute_saved_pct = round((1 - chunks_evaluated_exit / chunks_evaluated_full) * 100, 1)
    latency_saved_pct = round((1 - exit_time_ms / full_time_ms) * 100, 1)

    print(f"\n[Early-Exit Benchmark] Full chunks: {chunks_evaluated_full} ({full_time_ms} ms) -> Early-exit chunks: {chunks_evaluated_exit} ({exit_time_ms} ms)")
    print(f"Compute reduction: {compute_saved_pct}% | Latency reduction: {latency_saved_pct}%")

    return {
        "full_scan": {
            "chunks_processed": chunks_evaluated_full,
            "latency_ms": full_time_ms,
            "label": full_res["label"],
            "confidence": full_res["confidence"],
        },
        "speculative_early_exit": {
            "chunks_processed": chunks_evaluated_exit,
            "latency_ms": exit_time_ms,
            "label": exit_res["label"],
            "confidence": exit_res["confidence"],
        },
        "compute_saved_pct": f"{compute_saved_pct}%",
        "latency_saved_pct": f"{latency_saved_pct}%",
    }


def main():
    print("=" * 60)
    print("EdgeSLM: Inference Systems Performance Benchmark")
    print("=" * 60)

    classifier = DocumentClassifier()
    print(f"Device: {classifier.device} | Adapter: {classifier.adapter_path.name}\n")

    print("--- 1. Dynamic Chunk Batching vs Sequential Loop ---")
    batch_results = benchmark_batching(classifier)

    print("\n--- 2. Speculative Early-Exit Performance ---")
    exit_results = benchmark_early_exit(classifier)

    output = {
        "device": classifier.device,
        "adapter": classifier.adapter_path.name,
        "batching_benchmark": batch_results,
        "speculative_early_exit_benchmark": exit_results,
    }

    out_file = Path("results/inference_speedup_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved official benchmark results to {out_file}")


if __name__ == "__main__":
    main()
