import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.merge_annotations import cohens_kappa
from evaluation.run_ablation import score
from evaluation.systems import extract_label


def rows(*labels):
    return [{"text": f"doc {i}", "label": label} for i, label in enumerate(labels)]


def test_perfect_predictions_score_one():
    actual = rows("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")
    result = score(actual, ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"])
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["restricted_recall"] == 1.0
    assert result["restricted_under_classified"] == 0


def test_non_answers_count_as_incorrect():
    actual = rows("PUBLIC", "INTERNAL")
    result = score(actual, ["ABSTAIN", "UNKNOWN"])
    assert result["accuracy"] == 0.0
    assert result["non_answer_rate"] == 1.0


def test_under_classified_restricted_is_counted():
    actual = rows("RESTRICTED", "RESTRICTED", "RESTRICTED")
    # CONFIDENTIAL is wrong but not dangerous; PUBLIC and INTERNAL are.
    result = score(actual, ["CONFIDENTIAL", "PUBLIC", "INTERNAL"])
    assert result["restricted_recall"] == 0.0
    assert result["restricted_under_classified"] == 2


def test_restricted_recall_is_none_without_support():
    actual = rows("PUBLIC", "INTERNAL")
    result = score(actual, ["PUBLIC", "INTERNAL"])
    assert result["restricted_recall"] is None


def test_macro_f1_ignores_absent_classes():
    # Only two classes present, so macro F1 averages over two, not four.
    actual = rows("PUBLIC", "PUBLIC", "INTERNAL", "INTERNAL")
    result = score(actual, ["PUBLIC", "PUBLIC", "INTERNAL", "INTERNAL"])
    assert result["macro_f1"] == 1.0
    assert result["per_class"]["CONFIDENTIAL"]["support"] == 0


def test_confusion_matrix_records_every_pair():
    actual = rows("PUBLIC", "PUBLIC", "RESTRICTED")
    result = score(actual, ["PUBLIC", "INTERNAL", "RESTRICTED"])
    assert result["confusion"]["PUBLIC"] == {"PUBLIC": 1, "INTERNAL": 1}
    assert result["confusion"]["RESTRICTED"] == {"RESTRICTED": 1}


def test_wilson_interval_brackets_the_estimate():
    actual = rows(*(["PUBLIC"] * 10))
    result = score(actual, ["PUBLIC"] * 8 + ["INTERNAL"] * 2)
    lower, upper = result["wilson_95"]
    assert lower < result["accuracy"] < upper


def test_kappa_is_one_for_identical_passes():
    labels = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"]
    assert cohens_kappa(labels, labels) == 1.0


def test_kappa_is_near_zero_for_chance_agreement():
    first = ["PUBLIC", "INTERNAL"] * 10
    second = ["PUBLIC", "INTERNAL"] * 5 + ["INTERNAL", "PUBLIC"] * 5
    assert abs(cohens_kappa(first, second)) < 0.05


def test_kappa_rejects_mismatched_lengths():
    try:
        cohens_kappa(["PUBLIC"], ["PUBLIC", "INTERNAL"])
    except ValueError:
        return
    raise AssertionError("mismatched annotation lengths should be rejected")


def test_label_extraction_prefers_longest_match():
    # "CONFIDENTIAL" must not be shadowed by a shorter label substring.
    assert extract_label(" CONFIDENTIAL") == "CONFIDENTIAL"
    assert extract_label(" RESTRICTED\n") == "RESTRICTED"
    assert extract_label("  public ") == "PUBLIC"


def test_label_extraction_returns_unknown_for_junk():
    assert extract_label("the document is") == "UNKNOWN"
    assert extract_label("") == "UNKNOWN"
