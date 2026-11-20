import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.independent_evaluation import (  # noqa: E402
    REQUIRED_COLUMNS, read_independent_csv, validate_independent_records,
)


def make_reviewed_row(example_id="real-1", *, source="The report may show an increase.",
                      summary="The report might show an increase.", label="PRESERVED"):
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update({
        "example_id": example_id, "source_document_id": "doc-1",
        "source_sentence": source, "summary_sentence": summary, "label": label,
        "source_reference": "https://example.test/paper", "source_location": "Results, paragraph 2",
        "domain": "science", "collection_method": "manual", "collection_date": "2026-01-01",
        "collection_notes": "test fixture only", "review_status": "approved",
        "reviewer_1_id": "reviewer-a", "reviewer_1_label": label,
        "reviewer_1_at": "2026-01-02T10:00:00", "reviewer_1_notes": "First independent review.",
        "reviewer_2_id": "reviewer-b", "reviewer_2_label": label,
        "reviewer_2_at": "2026-01-03T10:00:00", "reviewer_2_notes": "Second independent review.",
    })
    return row


def test_header_only_template_reads_as_empty(tmp_path):
    template = tmp_path / "template.csv"
    template.write_text(",".join(REQUIRED_COLUMNS) + "\n", encoding="utf-8")
    assert read_independent_csv(template) == []


def test_valid_two_reviewer_agreement_is_eligible():
    result = validate_independent_records([make_reviewed_row()])
    assert len(result.eligible) == 1
    assert not result.excluded
    assert not result.rejected


def test_duplicate_pair_rejects_every_copy_including_reversed_order():
    first = make_reviewed_row("pair-a")
    second = make_reviewed_row("pair-b", source=first["summary_sentence"],
                               summary=first["source_sentence"])
    result = validate_independent_records([first, second])
    assert not result.eligible
    assert len(result.rejected) == 2
    assert result.reasons["duplicate_pair"] == 2


def test_duplicate_ids_reject_every_copy():
    first = make_reviewed_row("same-id")
    second = make_reviewed_row("same-id", source="A different source statement.",
                               summary="A different summary statement.")
    result = validate_independent_records([first, second])
    assert not result.eligible
    assert result.reasons["duplicate_example_id"] == 2


def test_invalid_label_is_rejected():
    row = make_reviewed_row(label="MAYBE")
    result = validate_independent_records([row])
    assert not result.eligible
    assert "invalid_label" in result.row_issues["real-1"]


def test_missing_provenance_is_rejected():
    row = make_reviewed_row()
    row["source_reference"] = ""
    result = validate_independent_records([row])
    assert not result.eligible
    assert "missing_source_reference" in result.row_issues["real-1"]


def test_incomplete_review_and_disagreement_never_enter_evaluation():
    pending = make_reviewed_row("pending")
    pending["review_status"] = "requires_manual_review"
    pending["reviewer_2_id"] = pending["reviewer_2_label"] = ""
    pending["reviewer_2_at"] = pending["reviewer_2_notes"] = ""
    conflict = make_reviewed_row("conflict")
    conflict["source_sentence"] = "The trial could reduce risk."
    conflict["summary_sentence"] = "The trial reduced risk."
    conflict["review_status"] = "disagreement"
    conflict["reviewer_2_label"] = "HARDENED"
    conflict["label"] = ""
    result = validate_independent_records([pending, conflict])
    assert not result.eligible
    assert len(result.rejected) == 2
    assert "incomplete_review" in result.row_issues["pending"]
    assert "unresolved_disagreement" in result.row_issues["conflict"]


def test_human_exclude_is_counted_separately_and_never_eligible():
    row = make_reviewed_row(label="EXCLUDE")
    row["review_status"] = "excluded"
    row["reviewer_1_label"] = row["reviewer_2_label"] = "EXCLUDE"
    row["exclude_reason"] = "The pair changes the subject."
    result = validate_independent_records([row])
    assert not result.eligible
    assert len(result.excluded) == 1
    assert result.reasons["human_excluded"] == 1


def test_training_data_collision_is_rejected():
    row = make_reviewed_row()
    training = [{"source_sentence": row["source_sentence"],
                 "summary_sentence": row["summary_sentence"]}]
    result = validate_independent_records([row], training_records=training)
    assert not result.eligible
    assert "training_data_collision" in result.row_issues["real-1"]


def test_disagreement_is_not_automatically_resolved():
    row = make_reviewed_row("conflict")
    row["review_status"] = "disagreement"
    row["reviewer_2_label"] = "HARDENED"
    row["label"] = ""
    result = validate_independent_records([row])
    assert not result.eligible
    assert "conflict" in result.row_issues
    assert "invalid_label" not in result.row_issues["conflict"]


def test_adjudication_must_be_explicit_and_independent():
    row = make_reviewed_row("adjudicated")
    row["review_status"] = "adjudicated"
    row["reviewer_2_label"] = "HARDENED"
    row["label"] = row["adjudicated_label"] = "PRESERVED"
    row["adjudicator_id"] = "independent-adjudicator"
    row["adjudicated_at"] = "2026-01-04T10:00:00"
    row["adjudication_notes"] = "Explicit third-party resolution."
    result = validate_independent_records([row])
    assert len(result.eligible) == 1
    row["adjudicator_id"] = "reviewer-a"
    result = validate_independent_records([row])
    assert not result.eligible
    assert "adjudicator_must_be_independent" in result.row_issues["adjudicated"]
