import argparse
import json
from pathlib import Path


LABELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--reviewer", required=True)
    args = parser.parse_args()

    rows = []
    with args.path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    for index, row in enumerate(rows):
        if row.get("reviewed_label") in LABELS and row.get("reviewer_id"):
            continue

        print(f"\n[{index + 1}/{len(rows)}] {row['id']}")
        print(row["text"])
        print("Labels: " + ", ".join(LABELS))

        while True:
            label = input("reviewed_label: ").strip().upper()
            if label in LABELS:
                break
            print("Invalid label. Choose: " + ", ".join(LABELS))

        notes = input("review_notes (optional): ").strip()
        row["reviewed_label"] = label
        row["reviewer_id"] = args.reviewer
        row["review_notes"] = notes or None

        with args.path.open("w", encoding="utf-8") as handle:
            for saved_row in rows:
                handle.write(json.dumps(saved_row, ensure_ascii=False) + "\n")

    print(f"Annotation complete: {args.path}")


if __name__ == "__main__":
    main()
