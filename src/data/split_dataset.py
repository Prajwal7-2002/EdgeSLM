import json
import random
from pathlib import Path
from collections import Counter


SEED = 42

INPUT_PATH = Path(
    "data/raw/generated_documents.jsonl"
)

OUTPUT_DIR = Path(
    "data/splits"
)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_dataset():

    rows = []

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def validate_dataset(rows):

    texts = [
        row["text"].strip()
        for row in rows
    ]

    # Check global duplicates
    duplicate_count = (
        len(texts) - len(set(texts))
    )

    if duplicate_count > 0:

        raise ValueError(
            f"Dataset contains {duplicate_count} "
            "duplicate rows. Regenerate the dataset "
            "before splitting."
        )

    # Check labels
    valid_labels = {
        "PUBLIC",
        "INTERNAL",
        "CONFIDENTIAL",
        "RESTRICTED",
    }

    for row in rows:

        if row["label"] not in valid_labels:
            raise ValueError(
                f"Invalid label: {row['label']}"
            )


def split_dataset(rows):

    random.seed(SEED)

    by_label = {}

    for row in rows:

        label = row["label"]

        by_label.setdefault(
            label,
            []
        ).append(row)

    train = []
    validation = []
    test = []

    for label in sorted(by_label):

        examples = by_label[label].copy()

        random.shuffle(examples)

        n = len(examples)

        train_end = int(
            n * TRAIN_RATIO
        )

        val_end = train_end + int(
            n * VAL_RATIO
        )

        train.extend(
            examples[:train_end]
        )

        validation.extend(
            examples[train_end:val_end]
        )

        test.extend(
            examples[val_end:]
        )

    random.shuffle(train)
    random.shuffle(validation)
    random.shuffle(test)

    return train, validation, test


def check_split_leakage(
    train,
    validation,
    test,
):

    train_texts = {
        row["text"].strip()
        for row in train
    }

    val_texts = {
        row["text"].strip()
        for row in validation
    }

    test_texts = {
        row["text"].strip()
        for row in test
    }

    train_val = train_texts & val_texts
    train_test = train_texts & test_texts
    val_test = val_texts & test_texts

    if train_val:
        raise ValueError(
            f"Train/validation leakage: "
            f"{len(train_val)} texts"
        )

    if train_test:
        raise ValueError(
            f"Train/test leakage: "
            f"{len(train_test)} texts"
        )

    if val_test:
        raise ValueError(
            f"Validation/test leakage: "
            f"{len(val_test)} texts"
        )

    print("\nNO CROSS-SPLIT LEAKAGE DETECTED.")


def save_jsonl(rows, path):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_distribution(name, rows):

    counts = Counter(
        row["label"]
        for row in rows
    )

    print(
        f"\n{name}: {len(rows)} examples"
    )

    for label, count in sorted(
        counts.items()
    ):

        print(
            f"  {label:<12}: {count}"
        )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_dataset()

    print(
        "Loaded:",
        len(rows),
        "examples"
    )

    validate_dataset(rows)

    print(
        "Dataset contains",
        len(set(row["text"] for row in rows)),
        "unique texts."
    )

    train, validation, test = (
        split_dataset(rows)
    )

    check_split_leakage(
        train,
        validation,
        test,
    )

    save_jsonl(
        train,
        OUTPUT_DIR / "train.jsonl"
    )

    save_jsonl(
        validation,
        OUTPUT_DIR / "validation.jsonl"
    )

    save_jsonl(
        test,
        OUTPUT_DIR / "test.jsonl"
    )

    print_distribution(
        "TRAIN",
        train
    )

    print_distribution(
        "VALIDATION",
        validation
    )

    print_distribution(
        "TEST",
        test
    )

    print("\nFiles created:")

    print(
        OUTPUT_DIR / "train.jsonl"
    )

    print(
        OUTPUT_DIR / "validation.jsonl"
    )

    print(
        OUTPUT_DIR / "test.jsonl"
    )


if __name__ == "__main__":
    main()