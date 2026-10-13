import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.download_real_docs import write_labels
from scan import load_labels, report_agreement


def test_rerunning_the_download_keeps_labels_already_filled_in(tmp_path):
    first = tmp_path / "enron" / "a.eml"
    first.parent.mkdir()
    first.write_text("x")
    labels = write_labels(tmp_path, [{"file": first, "source": "enron", "selected_because": "k"}])

    rows = list(csv.DictReader(labels.open(encoding="utf-8")))
    rows[0]["your_label"] = "RESTRICTED"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    second = tmp_path / "enron" / "b.eml"
    second.write_text("y")
    write_labels(tmp_path, [
        {"file": first, "source": "enron", "selected_because": "k"},
        {"file": second, "source": "enron", "selected_because": "k"},
    ])
    assert load_labels(labels) == {"enron/a.eml": "RESTRICTED"}


def test_unknown_label_is_rejected(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("file,your_label\na.txt,SECRET\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_labels(path)


def test_agreement_report_counts_matches(capsys):
    rows = [
        {"file": "a", "label": "RESTRICTED"},
        {"file": "b", "label": "PUBLIC"},
        {"file": "c", "label": "INTERNAL"},
    ]
    report_agreement(rows, {"a": "RESTRICTED", "b": "INTERNAL"})
    out = capsys.readouterr().out
    assert "Agreement with your labels: 1/2" in out
    assert "RESTRICTED caught: 1/1" in out
