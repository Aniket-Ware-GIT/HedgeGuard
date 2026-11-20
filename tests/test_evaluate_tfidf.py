import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_tfidf import evaluate_records  # noqa: E402
from src.dataset.pipeline import DatasetError  # noqa: E402


def make_records():
    """Nine source families, two same-claim variants per group, balanced by label."""
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
                    "pair_id": f"{group}-p{variant}",
                    "source_group_id": group,
                    "source_sentence": f"{source_prefix} {claim}.",
                    "summary_sentence": f"{summary_prefix} {claim}.",
                    "label": label,
                    "is_synthetic": "true",
                    "review_status": "approved",
                    "include_in_training": "true",
                    "original_label": label,
                    "reviewer_id": "reviewer-1",
                })
    return records


def test_group_cv_uses_identical_out_of_fold_examples_for_both_models():
    result = evaluate_records(make_records(), n_splits=3, random_state=29)
    assert result["total"] == 18
    assert len(result["predictions"]) == 18
    assert len(result["folds"]) == 3
    assert set(result["overall"]) == {"tfidf_logistic_regression", "rule_based"}
    assert all(row["tfidf_prediction"] and row["rule_prediction"] for row in result["predictions"])
    # Each pair receives both predictions in the same record, so evaluation IDs match exactly.
    assert {row["pair_id"] for row in result["predictions"]} == {row["pair_id"] for row in make_records()}


def test_no_source_group_crosses_a_fold_boundary_and_each_group_validated_once():
    result = evaluate_records(make_records(), n_splits=3, random_state=29)
    validation_groups = []
    for fold in result["folds"]:
        train_groups = set(fold["train_group_ids"])
        valid_groups = set(fold["validation_group_ids"])
        assert train_groups.isdisjoint(valid_groups)
        assert fold["group_overlap"] == []
        validation_groups.extend(valid_groups)
    expected_groups = {row["source_group_id"] for row in make_records()}
    assert set(validation_groups) == expected_groups
    assert len(validation_groups) == len(expected_groups)


@pytest.mark.parametrize("bad_row", [
    {"pair_id": "exclude", "source_group_id": "g", "source_sentence": "A source sentence.",
     "summary_sentence": "A summary sentence.", "label": "EXCLUDE", "review_status": "excluded",
     "include_in_training": "false", "original_label": "EXCLUDE", "reviewer_id": "reviewer",
     "review_notes": "Not comparable.", "is_synthetic": "true"},
    {"pair_id": "pending", "source_group_id": "g", "source_sentence": "A source sentence.",
     "summary_sentence": "A summary sentence.", "label": "PRESERVED", "review_status": "requires_manual_review",
     "include_in_training": "false", "original_label": "PRESERVED", "reviewer_id": "", "review_notes": "",
     "is_synthetic": "true"},
])
def test_evaluation_rejects_noneligible_rows(bad_row):
    with pytest.raises((DatasetError, ValueError)):
        evaluate_records([bad_row], n_splits=2)


def test_evaluation_rejects_missing_groups_and_invalid_fold_count():
    records = make_records()
    records[0]["source_group_id"] = ""
    with pytest.raises(ValueError, match="source_group_id"):
        evaluate_records(records)
    with pytest.raises(ValueError, match="n_splits"):
        evaluate_records(make_records(), n_splits=10)
