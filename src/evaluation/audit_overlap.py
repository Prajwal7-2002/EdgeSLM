"""Near-duplicate audit between the training data and every held-out set.

Exact-match checks miss paraphrases, and a paraphrase of a test document in the
training data inflates the test score just as surely. For every held-out
document this finds the most similar training document (TF-IDF cosine over
word 1-2 grams) and the longest run of words the two share, then writes the
summary to results/overlap_audit.json so REPORT.md can show it.

    python src/evaluation/audit_overlap.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.datasets import DATASETS, load_dataset, load_jsonl


TRAIN_FILES = (
    "data/splits/train.jsonl",
    "data/raw/boundary_training.jsonl",
    "data/v2/train.jsonl",
)
OUTPUT = Path("results/overlap_audit.json")

# A shared run this long is a copied phrase rather than common vocabulary.
LONG_RUN = 8


def words(text):
    return text.lower().split()


def longest_shared_run(a, b):
    a, b = words(a), words(b)
    best = 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer

    train = [row["text"] for path in TRAIN_FILES if Path(path).exists() for row in load_jsonl(path)]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train)
    train_matrix = vectorizer.transform(train)

    report = {"train_files": list(TRAIN_FILES), "long_run_words": LONG_RUN, "sets": {}}
    for name, spec in DATASETS.items():
        if not spec["held_out"]:
            continue
        rows = load_dataset(name)
        similarity = (vectorizer.transform([r["text"] for r in rows]) @ train_matrix.T).toarray()
        matches = []
        for row, sims in zip(rows, similarity):
            index = int(sims.argmax())
            matches.append({
                "text": row["text"],
                "nearest_train": train[index],
                "cosine": round(float(sims[index]), 3),
                "shared_run": longest_shared_run(row["text"], train[index]),
            })
        matches.sort(key=lambda m: m["cosine"], reverse=True)
        cosines = [m["cosine"] for m in matches]
        report["sets"][name] = {
            "n": len(rows),
            "max_cosine": max(cosines),
            "mean_cosine": round(sum(cosines) / len(cosines), 3),
            "over_0.8": sum(c > 0.8 for c in cosines),
            "long_runs": sum(m["shared_run"] >= LONG_RUN for m in matches),
            "closest": matches[:3],
        }
        summary = report["sets"][name]
        print(
            f"{name:18s} n={summary['n']:3d}  max cos={summary['max_cosine']:.3f}  "
            f"mean={summary['mean_cosine']:.3f}  >0.8: {summary['over_0.8']}  "
            f"runs>={LONG_RUN} words: {summary['long_runs']}"
        )
        for match in matches[:2]:
            print(f"    {match['cosine']:.3f}  test:  {match['text'][:90]}")
            print(f"           train: {match['nearest_train'][:90]}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
