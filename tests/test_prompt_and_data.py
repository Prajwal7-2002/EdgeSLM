import os
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from evaluation.datasets import load_dataset
from evaluation.run_ablation import score
from inference.prompt import CLASSIFICATION_GUIDANCE, LABELS, build_prompt, label_token_ids


class FakeTokenizer:
    def __init__(self, first_ids):
        self.first_ids = first_ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.first_ids[text.strip()], 99]}


def test_prompt_ends_where_the_label_is_scored():
    prompt = build_prompt("Quarterly roadmap.")
    assert prompt.endswith("Classification:")
    assert CLASSIFICATION_GUIDANCE in prompt
    assert CLASSIFICATION_GUIDANCE not in build_prompt("x", guidance=False)


def test_label_token_ids_are_first_tokens():
    tokenizer = FakeTokenizer({"PUBLIC": 1, "INTERNAL": 2, "CONFIDENTIAL": 3, "RESTRICTED": 4})
    assert label_token_ids(tokenizer) == {"PUBLIC": 1, "INTERNAL": 2, "CONFIDENTIAL": 3, "RESTRICTED": 4}


def test_label_token_collision_is_rejected():
    tokenizer = FakeTokenizer({"PUBLIC": 1, "INTERNAL": 2, "CONFIDENTIAL": 3, "RESTRICTED": 3})
    with pytest.raises(ValueError):
        label_token_ids(tokenizer)


def test_contrast_set_is_balanced_minimal_pairs():
    rows = load_dataset("contrast")
    assert Counter(r["label"] for r in rows) == {label: 20 for label in LABELS}
    pairs = {}
    for row in rows:
        pairs.setdefault(row["pair"], []).append(row["label"])
    assert len(pairs) == 40
    assert all(len(labels) == 2 and labels[0] != labels[1] for labels in pairs.values())


def test_contrast_set_does_not_overlap_training_data():
    from evaluation.datasets import load_jsonl

    train = set()
    for path in ("data/splits/train.jsonl", "data/raw/boundary_training.jsonl", "data/v2/train.jsonl"):
        train.update(r["text"].strip() for r in load_jsonl(path))
    assert not train & {r["text"] for r in load_dataset("contrast")}


def test_pair_accuracy_needs_both_sides_right():
    rows = [
        {"text": "a", "label": "PUBLIC", "pair": "p1"},
        {"text": "b", "label": "RESTRICTED", "pair": "p1"},
        {"text": "c", "label": "INTERNAL", "pair": "p2"},
        {"text": "d", "label": "CONFIDENTIAL", "pair": "p2"},
    ]
    result = score(rows, ["PUBLIC", "RESTRICTED", "INTERNAL", "INTERNAL"])
    assert result["accuracy"] == 0.75
    assert result["pair_accuracy"] == 0.5


def test_pair_accuracy_absent_without_pairs():
    rows = [{"text": "a", "label": "PUBLIC"}]
    assert score(rows, ["PUBLIC"])["pair_accuracy"] is None
