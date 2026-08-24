import json
from collections import Counter
from pathlib import Path


FILES = {
    "train": Path("data/splits/train.jsonl"),
    "validation": Path("data/splits/validation.jsonl"),
    "test": Path("data/splits/test.jsonl"),
}


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


datasets = {name: load(path) for name, path in FILES.items()}


print("=" * 70)
print("DATASET LEAKAGE / DUPLICATE AUDIT")
print("=" * 70)


# ---------------------------------------------------------
# 1. Exact duplicates within each split
# ---------------------------------------------------------

for name, rows in datasets.items():
    texts = [r["text"].strip() for r in rows]
    counts = Counter(texts)

    duplicates = {text: n for text, n in counts.items() if n > 1}

    print(f"\n{name.upper()}")
    print("-" * 70)
    print(f"Total examples:       {len(rows)}")
    print(f"Unique texts:          {len(counts)}")
    print(f"Duplicate texts:      {len(duplicates)}")
    print(f"Duplicate extra rows: {len(texts) - len(counts)}")

    if duplicates:
        print("\nExamples of duplicates:")
        for text, count in list(duplicates.items())[:5]:
            print(f"  [{count}x] {text}")


# ---------------------------------------------------------
# 2. Exact overlap BETWEEN splits
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CROSS-SPLIT EXACT OVERLAP")
print("=" * 70)

sets = {
    name: set(r["text"].strip() for r in rows)
    for name, rows in datasets.items()
}

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

for a, b in pairs:
    overlap = sets[a] & sets[b]

    print(f"\n{a} ∩ {b}")
    print(f"Exact overlapping texts: {len(overlap)}")

    if overlap:
        for text in list(overlap)[:10]:
            print(f"  {text}")


# ---------------------------------------------------------
# 3. Label consistency
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LABEL CONSISTENCY")
print("=" * 70)

all_rows = []

for split, rows in datasets.items():
    for row in rows:
        all_rows.append((row["text"].strip(), row["label"], split))

label_map = {}

for text, label, split in all_rows:
    label_map.setdefault(text, set()).add(label)

conflicts = {
    text: labels
    for text, labels in label_map.items()
    if len(labels) > 1
}

print(f"Texts appearing with multiple labels: {len(conflicts)}")

if conflicts:
    for text, labels in list(conflicts.items())[:10]:
        print(f"  Labels={labels}")
        print(f"  {text}")


# ---------------------------------------------------------
# 4. Template / sentence diversity
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEXT DIVERSITY")
print("=" * 70)

for name, rows in datasets.items():

    lengths = [len(r["text"].split()) for r in rows]

    print(f"\n{name.upper()}")
    print(f"Min words: {min(lengths)}")
    print(f"Max words: {max(lengths)}")
    print(f"Avg words: {sum(lengths) / len(lengths):.2f}")


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)