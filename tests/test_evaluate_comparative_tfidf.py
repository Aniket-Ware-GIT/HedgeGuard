import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_comparative_tfidf import evaluate_records  # noqa: E402


def make_records():
    records = []
    label_text = {
        "HARDENED": ("The sensor may detect signal", "The sensor detects signal"),
        "PRESERVED": ("The sensor may detect signal", "The sensor might detect signal"),
        "SOFTENED": ("The sensor detects signal", "The sensor may detect signal"),
    }
    for label, (source_prefix, summary_prefix) in label_text.items():
        for group_number in range(3):
            group = f"{label}-{group_number}"
            for variant in range(2):
                claim = f"{group_number}-{variant}"
                records.append({
                    "pair_id": f"{group}-p{variant}", "source_group_id": group,
                    "source_sentence": f"{source_prefix} {claim}.",
                    "summary_sentence": f"{summary_prefix} {claim}.", "label": label,
                    "is_synthetic": "true", "review_status": "approved",
                    "include_in_training": "true", "original_label": label,
                    "reviewer_id": "reviewer-1",
                })
    return records


def test_three_models_share_identical_oof_examples_and_grouped_folds():
    records = make_records()
    result = evaluate_records(records, n_splits=3, random_state=42)
    assert result["total"] == len(records)
    assert len(result["predictions"]) == len(records)
    assert set(result["overall"]) == {"comparative_tfidf", "tfidf_logistic_regression", "rule_based"}
    assert {row["pair_id"] for row in result["predictions"]} == {row["pair_id"] for row in records}
    assert all(row[key] in {"PRESERVED", "HARDENED", "SOFTENED"}
               for row in result["predictions"] for key in
               ("comparative_tfidf", "tfidf_logistic_regression", "rule_based"))
    for fold in result["folds"]:
        assert set(fold["train_group_ids"]).isdisjoint(fold["validation_group_ids"])
        assert fold["group_overlap"] == []
    # A pair stores all three predictions together, proving identical OOF coverage.
    assert len({row["fold"] for row in result["predictions"]}) == 3


def test_evaluator_rejects_exclude_and_pending_examples():
    records = make_records()
    pending = dict(records[0], pair_id="pending", source_group_id="pending-group",
                   review_status="requires_manual_review", include_in_training="false")
    with pytest.raises(ValueError):
        evaluate_records([*records, pending])
