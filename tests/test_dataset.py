import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.pipeline import (  # noqa: E402
    DatasetError, class_counts, generate, group_split, prepare_review_rows, read_csv,
    apply_review_decision, training_rows, validate_review_rows, validate_rows, write_csv,
)
from scripts.build_training_dataset import build_training_dataset  # noqa: E402


def row(pair_id, source, summary, label, group="g1", **extra):
    return {"pair_id": pair_id, "source_group_id": group, "source_sentence": source,
            "summary_sentence": summary, "label": label, "domain": "software",
            "pair_origin": "synthetic_seed", "annotator_notes": "curated", **extra}


@pytest.fixture
def mandatory_pairs():
    return [
        row("HG1", "The system may restart during an update.",
            "The system will restart during an update.", "HARDENED"),
        row("HG2", "The system restarts during an update.",
            "The system may restart during an update.", "SOFTENED"),
        row("HG3", "The system may restart during an update.",
            "The system might restart during an update.", "PRESERVED"),
        row("HG4", "Employees may access the server.",
            "Employees are permitted to access the server.", "EXCLUDE", group="g2"),
    ]


def test_mandatory_pairs_and_all_three_classes(mandatory_pairs):
    generated = generate(mandatory_pairs)
    by_id = {r["pair_id"]: r for r in generated}
    assert by_id["HG1"]["label"] == "HARDENED"
    assert by_id["HG2"]["label"] == "SOFTENED"
    assert by_id["HG3"]["label"] == "PRESERVED"
    assert by_id["HG4"]["label"] == "EXCLUDE"
    assert class_counts(generated) == {"PRESERVED": 1, "HARDENED": 1, "SOFTENED": 1, "EXCLUDE": 1}


def test_no_unsafe_rewriting_and_permission_may_excluded(mandatory_pairs):
    generated = generate(mandatory_pairs)
    assert generated[0]["summary_sentence"] == "The system will restart during an update."
    assert generated[3]["label"] == "EXCLUDE"
    assert generated[3]["exclude_reason"] == "permission_modality"


def test_duplicates_removed_and_conflicting_duplicates_fail():
    pair = row("a", "System may restart during an update.", "System might restart during an update.", "PRESERVED")
    generated, duplicates = generate([pair, {**pair, "pair_id": "b"}], return_duplicate_count=True)
    assert len(generated) == 1 and duplicates == 1
    with pytest.raises(DatasetError, match="Conflicting labels"):
        generate([pair, {**pair, "label": "HARDENED"}])


def test_deterministic_generation_and_group_retention(mandatory_pairs):
    first = generate(mandatory_pairs)
    second = generate(mandatory_pairs)
    assert first == second
    assert {r["source_group_id"] for r in first[:3]} == {"g1"}


@pytest.mark.parametrize("bad", [
    row("a", "", "A summary sentence.", "PRESERVED"),
    row("a", "System restarts.", "System restarts.", "SOFTENED"),
    row("a", "System system restarts.", "System restarts during testing.", "PRESERVED"),
    row("a", "System restarts during testing.", "System may restart during testing.", "MAYBE"),
])
def test_invalid_rows_reported(bad):
    errors = validate_rows([bad])
    assert errors


def test_all_candidates_have_valid_text_and_are_review_marked(mandatory_pairs):
    generated = generate(mandatory_pairs)
    assert validate_rows(generated) == []
    assert all(r["source_sentence"].strip() and r["summary_sentence"].strip() for r in generated)
    assert all(r["review_status"] == "requires_manual_review" for r in generated)
    assert all(r["is_synthetic"] == "true" for r in generated)


def test_csv_round_trip(mandatory_pairs):
    target = Path(__file__).with_name(".roundtrip_test.csv")
    generated = generate(mandatory_pairs)
    try:
        write_csv(generated, target)
        loaded = read_csv(target)
        assert loaded == generated
        assert validate_rows(loaded) == []
    finally:
        target.unlink(missing_ok=True)


def test_group_split_keeps_groups_together_and_excludes_exclude(mandatory_pairs):
    # Duplicate group members are used to ensure a source family cannot cross splits.
    pairs = [*mandatory_pairs, row("HG5", "The system might restart during an update.",
                                    "The system may restart during an update.", "PRESERVED")]
    splits = group_split(generate(pairs), seed=7, ratios=(.34, .33, .33))
    groups = [{r["source_group_id"] for r in part} for part in splits.values()]
    assert all(not groups[i].intersection(groups[j]) for i in range(3) for j in range(i + 1, 3))
    assert all(r["label"] != "EXCLUDE" for part in splits.values() for r in part)


def test_review_queue_defaults_pending_and_preserves_annotations(mandatory_pairs):
    candidates = generate(mandatory_pairs)
    reviewed = prepare_review_rows(candidates)
    assert len(reviewed) == len(candidates)
    assert all(r["review_status"] == "requires_manual_review" for r in reviewed)
    assert all(r["include_in_training"] == "false" for r in reviewed)
    assert all(r["source_group_id"] == c["source_group_id"] for r, c in zip(reviewed, candidates))
    assert all(r["annotation_notes"] == c["annotation_notes"] for r, c in zip(reviewed, candidates))
    assert all(r["original_label"] == c["label"] for r, c in zip(reviewed, candidates))
    assert validate_review_rows(reviewed) == []


def test_approved_and_relabeled_rows_only_are_training_eligible(mandatory_pairs):
    pending = prepare_review_rows(generate(mandatory_pairs))
    approved = apply_review_decision(pending[0], "approved", "reviewer-1")
    relabeled = apply_review_decision(pending[1], "relabeled", "reviewer-1", label="HARDENED",
                                      notes="Manual comparison supports stronger certainty.")
    excluded = apply_review_decision(pending[3], "excluded", "reviewer-1", notes="Permission sense.")
    combined = [approved, relabeled, excluded, pending[2]]
    assert validate_review_rows(combined) == []
    eligible = training_rows(combined)
    assert [r["pair_id"] for r in eligible] == [approved["pair_id"], relabeled["pair_id"]]
    assert all(r["label"] != "EXCLUDE" for r in eligible)


def test_unreviewed_or_excluded_rows_cannot_be_enabled_for_training(mandatory_pairs):
    pending = prepare_review_rows(generate(mandatory_pairs))
    pending[0]["include_in_training"] = "true"
    assert any("unreviewed row cannot" in e for e in validate_review_rows(pending))
    with pytest.raises(DatasetError, match="not safe for training"):
        training_rows(pending)

    pending = prepare_review_rows(generate(mandatory_pairs))
    pending[3]["review_status"] = "excluded"
    pending[3]["reviewer_id"] = "reviewer-1"
    pending[3]["review_notes"] = "Permission sense."
    pending[3]["include_in_training"] = "true"
    assert any("EXCLUDE examples can never" in e for e in validate_review_rows(pending))


def test_exclude_cannot_be_approved_or_relabeled():
    candidate = prepare_review_rows(generate([row("x", "A test result is inconclusive.",
                                                    "A test result is inconclusive.", "EXCLUDE")]))[0]
    with pytest.raises(DatasetError, match="cannot be approved"):
        apply_review_decision(candidate, "approved", "reviewer-1")
    with pytest.raises(DatasetError, match="relabeled rows require"):
        apply_review_decision(candidate, "relabeled", "reviewer-1", label="EXCLUDE")


def test_review_status_values_and_reviewer_attribution_are_enforced(mandatory_pairs):
    pending = prepare_review_rows(generate(mandatory_pairs))
    pending[0]["review_status"] = "done"
    assert any("invalid review_status" in e for e in validate_review_rows(pending))
    with pytest.raises(DatasetError, match="reviewer_id is required"):
        apply_review_decision(prepare_review_rows(generate(mandatory_pairs))[0], "approved", "")


def test_experimental_v3_cannot_be_used_directly_and_review_copy_preserves_groups():
    source = Path(__file__).resolve().parents[1] / "data" / "processed" / "experimental_candidates_v3.csv"
    candidates = read_csv(source)
    with pytest.raises(DatasetError, match="not safe for training"):
        training_rows(candidates)
    reviewed = prepare_review_rows(candidates)
    assert len(reviewed) == len(candidates)
    assert all(r["source_group_id"] == c["source_group_id"] for r, c in zip(reviewed, candidates))
    assert all(r["annotator_notes"] == c["annotator_notes"] for r, c in zip(reviewed, candidates))
    assert all(r["include_in_training"] == "false" for r in reviewed)


def test_training_dataset_builder_only_returns_approved_or_relabeled(mandatory_pairs):
    pending = prepare_review_rows(generate(mandatory_pairs))
    approved = apply_review_decision(pending[0], "approved", "student-1")
    excluded = apply_review_decision(pending[3], "excluded", "student-1", notes="Permission usage.")
    training = build_training_dataset([approved, excluded, pending[1]])
    assert [row["pair_id"] for row in training] == [approved["pair_id"]]
    assert all(row["review_status"] in {"approved", "relabeled"} for row in training)
    assert all(row["include_in_training"] == "true" and row["label"] != "EXCLUDE" for row in training)


@pytest.mark.parametrize("decision", ["excluded", "pending"])
def test_training_dataset_builder_rejects_excluded_or_pending_enabled_rows(mandatory_pairs, decision):
    pending = prepare_review_rows(generate(mandatory_pairs))
    if decision == "excluded":
        bad = apply_review_decision(pending[3], "excluded", "student-1", notes="Not comparable.")
    else:
        bad = pending[0]
    bad["include_in_training"] = "true"
    with pytest.raises(DatasetError):
        build_training_dataset([bad])
