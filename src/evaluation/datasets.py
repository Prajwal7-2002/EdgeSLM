"""Evaluation set registry.

Every set is declared with an explicit note about how it was built and whether
the LoRA adapter was trained on it, so the ablation report can never silently
present contaminated numbers as held out.
"""

import json
from pathlib import Path


TRAIN_FILES = (
    Path("data/splits/train.jsonl"),
    Path("data/raw/boundary_training.jsonl"),
)

# Rows of the reviewed benchmark that came from boundary_training.jsonl are in
# the LoRA training mix. They are split out rather than dropped so the
# contamination is measurable instead of hidden.
BOUNDARY_FILE = Path("data/raw/boundary_training.jsonl")


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def training_texts():
    texts = set()
    for path in TRAIN_FILES:
        if path.exists():
            texts.update(row["text"].strip() for row in load_jsonl(path))
    return texts


def _reviewed(seen_in_training):
    path = Path("data/review/benchmark_labeled.jsonl")
    if not path.exists():
        return []
    boundary = {row["text"].strip() for row in load_jsonl(BOUNDARY_FILE)}
    return [
        {"text": row["text"].strip(), "label": row["label"], "id": row["id"]}
        for row in load_jsonl(path)
        if (row["text"].strip() in boundary) == seen_in_training
    ]


def _plain(path):
    return [
        {"text": row["text"].strip(), "label": row["label"]}
        for row in load_jsonl(path)
    ]


DATASETS = {
    "generated-test": {
        "loader": lambda: _plain("data/splits/test.jsonl"),
        "held_out": True,
        "note": (
            "Held-out split of the template generator that also produced the "
            "training split. In-distribution: shares templates and topic lists "
            "with training data, so it measures template fit, not generalisation."
        ),
    },
    "realistic": {
        "loader": lambda: _plain("data/realistic_test.jsonl"),
        "held_out": True,
        "note": (
            "Hand-written prose, not template-generated. No overlap with "
            "training data."
        ),
    },
    "adversarial": {
        "loader": lambda: _plain("data/odd/adversarial_test.jsonl"),
        "held_out": True,
        "note": (
            "Hand-written cases chosen to defeat keyword matching: public docs "
            "containing credential vocabulary, internal docs that disclaim "
            "credentials, and similar traps. No overlap with training data."
        ),
    },
    "reviewed-heldout": {
        "loader": lambda: _reviewed(seen_in_training=False),
        "held_out": True,
        "note": (
            "Independently re-annotated rows that the adapter never saw in "
            "training. This is the headline generalisation number."
        ),
    },
    "reviewed-seen": {
        "loader": lambda: _reviewed(seen_in_training=True),
        "held_out": False,
        "note": (
            "Independently re-annotated rows drawn from boundary_training.jsonl, "
            "which IS in the LoRA training mix. Reported only as a "
            "contamination control: model scores here are memorisation, not "
            "generalisation."
        ),
    },
}


def load_dataset(name):
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}")
    return DATASETS[name]["loader"]()
