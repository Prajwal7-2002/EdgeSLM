"""Merge independent annotation passes into an evaluation set.

Each annotator writes ``data/review/annotations/<annotator>.jsonl`` with one
object per row: ``{"id": ..., "label": ..., "notes": ...}``. This script joins
those passes onto the review queue, reports agreement between annotators and
against the label the dataset generator originally assigned, and emits the
evaluation set used by the ablation harness.

Adding a human pass is a drop-in: write another file into the annotations
directory and re-run. Cohen's kappa is reported for every annotator pair.
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


LABELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")

QUEUE = Path("data/review/benchmark_queue.jsonl")
ANNOTATIONS_DIR = Path("data/review/annotations")
OUTPUT = Path("data/review/benchmark_labeled.jsonl")

# The queue was assembled from these files; their labels are the generator's
# originals and act as a comparison baseline, never as ground truth.
SOURCE_FILES = (
    Path("data/realistic_test.jsonl"),
    Path("data/raw/boundary_training.jsonl"),
    Path("data/raw/documents.jsonl"),
)


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_source_labels():
    labels = {}
    for path in SOURCE_FILES:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            labels.setdefault(row["text"].strip(), row["label"])
    return labels


def cohens_kappa(first, second):
    """Cohen's kappa for two equal-length label sequences."""
    if len(first) != len(second):
        raise ValueError("annotation passes must cover the same rows")
    total = len(first)
    if total == 0:
        raise ValueError("no overlapping rows to compare")

    observed = sum(a == b for a, b in zip(first, second)) / total

    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[label] / total) * (second_counts[label] / total)
        for label in LABELS
    )

    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    queue = load_jsonl(args.queue)
    by_id = {row["id"]: row for row in queue}

    passes = {}
    for path in sorted(args.annotations.glob("*.jsonl")):
        annotator = path.stem
        rows = load_jsonl(path)
        unknown = [r["id"] for r in rows if r["id"] not in by_id]
        if unknown:
            raise ValueError(f"{path}: unknown ids {unknown[:5]}")
        bad = [r["id"] for r in rows if r["label"] not in LABELS]
        if bad:
            raise ValueError(f"{path}: invalid labels on {bad[:5]}")
        passes[annotator] = {r["id"]: r for r in rows}

    if not passes:
        raise SystemExit(f"No annotation passes found in {args.annotations}")

    annotators = sorted(passes)
    covered = set.intersection(*(set(p) for p in passes.values()))
    missing = set(by_id) - covered
    if missing:
        print(f"WARNING: {len(missing)} rows lack a label from every annotator")

    source_labels = load_source_labels()

    print("=" * 68)
    print("INDEPENDENT ANNOTATION MERGE")
    print("=" * 68)
    print(f"Queue rows:  {len(queue)}")
    print(f"Annotators:  {', '.join(annotators)}")
    print(f"Fully covered rows: {len(covered)}")

    ordered = [row["id"] for row in queue if row["id"] in covered]

    print("\nLabel distribution per annotator")
    print("-" * 68)
    for annotator in annotators:
        counts = Counter(passes[annotator][i]["label"] for i in ordered)
        summary = "  ".join(f"{label}={counts[label]}" for label in LABELS)
        print(f"  {annotator}: {summary}")

    if len(annotators) > 1:
        print("\nInter-annotator agreement")
        print("-" * 68)
        for left, right in combinations(annotators, 2):
            a = [passes[left][i]["label"] for i in ordered]
            b = [passes[right][i]["label"] for i in ordered]
            raw = sum(x == y for x, y in zip(a, b)) / len(ordered)
            print(
                f"  {left} vs {right}: raw={raw * 100:.1f}%  "
                f"kappa={cohens_kappa(a, b):.3f}"
            )
    else:
        print(
            "\nOnly one annotation pass present; no inter-annotator agreement "
            "can be computed. Add a second pass to measure it."
        )

    print("\nAgreement with the generator's original label")
    print("-" * 68)
    disagreements = []
    for annotator in annotators:
        matched = 0
        agreed = 0
        for row_id in ordered:
            text = by_id[row_id]["text"].strip()
            if text not in source_labels:
                continue
            matched += 1
            original = source_labels[text]
            assigned = passes[annotator][row_id]["label"]
            if original == assigned:
                agreed += 1
            elif annotator == annotators[0]:
                disagreements.append((row_id, original, assigned, text))
        if matched:
            print(
                f"  {annotator}: {agreed}/{matched} "
                f"({agreed / matched * 100:.1f}%)"
            )

    if disagreements:
        print(f"\nDisagreements for {annotators[0]} ({len(disagreements)}):")
        for row_id, original, assigned, text in disagreements:
            print(f"  {row_id}: generator={original} -> annotator={assigned}")
            print(f"    {text}")

    # Majority vote across annotators; ties fall back to the strictest label,
    # which is the safe direction for a sensitivity classifier.
    strictness = {label: index for index, label in enumerate(LABELS)}
    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row_id in ordered:
            votes = Counter(passes[a][row_id]["label"] for a in annotators)
            top = max(votes.values())
            tied = [label for label, n in votes.items() if n == top]
            label = max(tied, key=lambda candidate: strictness[candidate])
            notes = {
                a: passes[a][row_id].get("notes")
                for a in annotators
                if passes[a][row_id].get("notes")
            }
            text = by_id[row_id]["text"].strip()
            handle.write(
                json.dumps(
                    {
                        "id": row_id,
                        "text": text,
                        "label": label,
                        "annotators": annotators,
                        "unanimous": len(tied) == 1 and top == len(annotators),
                        "source_label": source_labels.get(text),
                        "notes": notes or None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    counts = Counter(row["label"] for row in load_jsonl(args.output))
    print(f"\nWrote {written} labeled rows to {args.output}")
    for label in LABELS:
        print(f"  {label}: {counts[label]}")


if __name__ == "__main__":
    main()
