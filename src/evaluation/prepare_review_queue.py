import json
from pathlib import Path


SOURCES = (
    Path("data/realistic_test.jsonl"),
    Path("data/raw/boundary_training.jsonl"),
    Path("data/raw/documents.jsonl"),
)
OUTPUT = Path("data/review/benchmark_queue.jsonl")


def load_rows(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    rows = []
    seen = set()

    for source in SOURCES:
        for row in load_rows(source):
            text = row["text"].strip()
            if text in seen:
                continue
            seen.add(text)
            rows.append(
                {
                    "id": f"review-{len(rows) + 1:04d}",
                    "text": text,
                    "reviewed_label": None,
                    "reviewer_id": None,
                    "review_notes": None,
                }
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created {len(rows)} review candidates: {OUTPUT}")
    print("The queue is blind; human review is required before evaluation.")


if __name__ == "__main__":
    main()
