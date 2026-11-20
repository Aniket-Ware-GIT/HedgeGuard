import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_rule_based import evaluate_records  # noqa: E402


def record(pair_id, source, summary, label):
    return {"pair_id": pair_id, "source_sentence": source,
            "summary_sentence": summary, "label": label}


def test_evaluation_metrics_and_confusion_matrix_use_fixed_class_order():
    records = [
        record("p1", "The treatment may reduce symptoms.", "The treatment reduces symptoms.", "HARDENED"),
        record("p2", "The treatment reduces symptoms.", "The treatment may reduce symptoms.", "SOFTENED"),
        record("p3", "The system may restart.", "The system might restart.", "PRESERVED"),
        # Intentional mismatch: classifier says PRESERVED, truth says HARDENED.
        record("p4", "The system may restart.", "The system might restart.", "HARDENED"),
    ]
    result = evaluate_records(records)
    assert result["total"] == 4
    assert result["accuracy"] == pytest.approx(.75)
    assert result["macro_f1"] == pytest.approx(7 / 9)
    assert result["labels"] == ["PRESERVED", "HARDENED", "SOFTENED"]
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert result["per_class"]["HARDENED"]["support"] == 2
    assert [row["correct"] for row in result["predictions"]] == [True, True, True, False]


@pytest.mark.parametrize("records, message", [
    ([], "empty"),
    ([{"pair_id": "x", "label": "PRESERVED"}], "missing required columns"),
    ([record("x", "A source sentence.", "A summary sentence.", "EXCLUDE")], "unsupported ground-truth label"),
    ([record("x", "", "A summary sentence.", "PRESERVED")], "must be non-empty"),
])
def test_evaluation_rejects_invalid_inputs(records, message):
    with pytest.raises(ValueError, match=message):
        evaluate_records(records)


def test_per_class_metrics_include_all_target_labels():
    result = evaluate_records([
        record("p1", "A result may change.", "A result changes.", "HARDENED"),
    ])
    assert set(result["per_class"]) == {"PRESERVED", "HARDENED", "SOFTENED"}
    assert result["per_class"]["PRESERVED"]["support"] == 0
