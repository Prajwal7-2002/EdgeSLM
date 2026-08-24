import argparse
import json
from collections import Counter
from pathlib import Path


LABELS = {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"}


def validate(path):
    rows = []
    seen = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "text", "reviewed_label", "reviewer_id"}
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Line {line_number}: missing {sorted(missing)}")
            if not row["text"].strip():
                raise ValueError(f"Line {line_number}: empty text")
            if row["text"] in seen:
                raise ValueError(f"Line {line_number}: duplicate text")
            if row["reviewed_label"] not in LABELS:
                raise ValueError(
                    f"Line {line_number}: reviewed_label must be one of {sorted(LABELS)}"
                )
            if not row["reviewer_id"]:
                raise ValueError(f"Line {line_number}: reviewer_id is required")
            seen.add(row["text"])
            rows.append(row)

    if not rows:
        raise ValueError("Review set is empty")

    counts = Counter(row["reviewed_label"] for row in rows)
    print(f"Validated {len(rows)} independently reviewed examples")
    for label in sorted(LABELS):
        print(f"  {label}: {counts[label]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    validate(args.path)


if __name__ == "__main__":
    main()
