import json
from collections import Counter

FILES = [
    "data/splits/train.jsonl",
    "data/splits/validation.jsonl",
    "data/splits/test.jsonl",
]


for file_path in FILES:

    counts = Counter()
    total = 0

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            example = json.loads(line)

            label = example["label"]

            counts[label] += 1
            total += 1

    print()
    print("=" * 60)
    print(file_path)
    print("=" * 60)

    print(f"Total examples: {total}")

    for label in [
        "PUBLIC",
        "INTERNAL",
        "CONFIDENTIAL",
        "RESTRICTED",
    ]:

        count = counts[label]

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{label:<15} "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )