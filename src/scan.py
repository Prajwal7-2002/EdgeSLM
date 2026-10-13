"""Classify every supported file in a folder.

    python src/scan.py path/to/folder
    python src/scan.py path/to/folder --csv results/scan.csv --no-policy
    python src/scan.py real_docs --labels real_docs/labels.csv

Prints one line per file and optionally writes a CSV with the label, the
deciding chunk's confidence and the per-chunk labels, so a long document's
verdict can be traced to the chunk that caused it.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference.classifier import DEFAULT_ADAPTER, DocumentClassifier
from inference.extract import SUPPORTED_SUFFIXES, UnsupportedFile, extract_file
from inference.prompt import LABELS


def load_labels(path):
    """Your labels from labels.csv (written by src/data/download_real_docs.py)."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = {}
    for row in rows:
        label = row.get("your_label", "").strip().upper()
        if label:
            if label not in LABELS:
                raise SystemExit(f"{row['file']}: unknown label {label!r}; use one of {LABELS}")
            labels[row["file"]] = label
    return labels


def report_agreement(rows, labels):
    scored = [row for row in rows if row["file"] in labels]
    if not scored:
        print("\nNo labelled files matched; fill in your_label in labels.csv first.")
        return
    right = sum(labels[row["file"]] == row["label"] for row in scored)
    print(f"\nAgreement with your labels: {right}/{len(scored)} ({right / len(scored):.0%})")
    restricted = [row for row in scored if labels[row["file"]] == "RESTRICTED"]
    if restricted:
        caught = sum(row["label"] == "RESTRICTED" for row in restricted)
        print(f"RESTRICTED caught: {caught}/{len(restricted)}")
    confusion = Counter((labels[row["file"]], row["label"]) for row in scored)
    print("\nyours -> model")
    for (yours, model), n in sorted(confusion.items()):
        marker = "" if yours == model else "   <- disagreement"
        print(f"  {yours:12s} -> {model:12s} {n}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--no-policy", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None, help="labels.csv with a your_label column")
    args = parser.parse_args()

    # The labels file sits inside the folder it describes; it is not a document.
    skip = {(args.folder / "labels.csv").resolve()}
    if args.labels:
        skip.add(args.labels.resolve())
    files = sorted(
        path for path in args.folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and path.resolve() not in skip
    )
    if not files:
        raise SystemExit(f"no supported files under {args.folder}")

    classifier = DocumentClassifier(adapter_path=args.adapter)
    rows = []
    for path in files:
        relative = path.relative_to(args.folder).as_posix()
        try:
            text = extract_file(path)
        except UnsupportedFile as error:
            print(f"{'SKIPPED':12s} {relative}  ({error})")
            continue
        result = classifier.classify(text, use_policy=not args.no_policy)
        how = "policy rule" if result["policy_override"] else f"{result['chunks']} chunk(s)"
        print(f"{result['label']:12s} {relative}  ({how}, confidence {result['confidence']})")
        rows.append({
            "file": relative,
            "label": result["label"],
            "confidence": result["confidence"],
            "policy_override": result["policy_override"],
            "chunks": result["chunks"],
            "chunk_labels": " ".join(result["chunk_labels"]),
            "characters": len(text),
        })

    if args.labels:
        report_agreement(rows, load_labels(args.labels))

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
