"""Schema and validation tests use in-memory fixture records, not corpus data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import (  # noqa: E402
    REAL_WORLD_FIELDS, STAGED_ANNOTATION_VERSION, RealWorldDatasetError, assert_group_disjoint, compare_calendar_dates,
    compare_negation, compare_number_mentions, license_is_reusable, normalized_pair_key,
    stable_example_id, validate_real_world_rows,
)


def make_row(example_id="RW-1", *, source="The intervention may reduce symptoms.",
             summary="The intervention might reduce symptoms.", label="PRESERVED"):
    row = {field: "" for field in REAL_WORLD_FIELDS}
    row.update({
        "example_id": example_id, "source_document_id": "PMC-fixture-1",
        "source_document_title": "Test fixture document", "source_url": "https://example.test/article",
        "publication_date": "2024-01-01", "source_sentence": source, "summary_sentence": summary,
        "domain": "fixture", "source_section": "Results", "source_location": "p. 3, sentence 2",
        "summary_location": "Abstract, sentence 1", "collection_method": "test fixture",
        "collection_date": "2026-01-01", "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_group_id": "PMC-fixture-1", "label": label,
        "annotation_status": "complete", "annotation_method": "llm_assisted",
        "model_name": "qwen3:4b-instruct", "model_version": "fixture-digest",
        "proposition_comparable": "true", "source_certainty": "POSSIBLE",
        "summary_certainty": "POSSIBLE", "certainty_direction": "PRESERVED",
        "source_evidence": '["may"]', "summary_evidence": '["might"]',
        "explanation": "Fixture annotation.", "subject_match": "true", "negation_match": "true",
        "numbers_match": "true", "dates_match": "true", "requires_review": "false",
    })
    return row


def make_v2_row(**kwargs):
    row=make_row(**kwargs)
    row.update(annotation_version=STAGED_ANNOTATION_VERSION, comparability_status="comparable",
               source_proposition=row["source_sentence"], summary_proposition=row["summary_sentence"],
               source_evidence='{"quote":"may","evidence_type":"explicit_uncertainty","interpretation":"Possibility."}',
               summary_evidence='{"quote":"might","evidence_type":"explicit_uncertainty","interpretation":"Possibility."}')
    return row


def test_valid_licensed_machine_annotation_is_eligible_but_not_human_review():
    result = validate_real_world_rows([make_row()])
    assert len(result.eligible) == 1
    assert result.eligible[0]["annotation_method"] == "llm_assisted"
    assert not result.excluded and not result.review_required and not result.rejected


def test_unassessed_row_is_review_required_with_a_reason():
    row=make_row();row.update(label="",annotation_status="not_annotated",requires_review="true")
    result=validate_real_world_rows([row])
    assert len(result.review_required)==1
    assert "not_yet_annotated_or_incomplete" in result.row_reasons["RW-1"]


def test_missing_provenance_is_rejected():
    row = make_row()
    row["source_url"] = ""
    result = validate_real_world_rows([row])
    assert not result.eligible
    assert "missing_source_url" in result.row_reasons["RW-1"]


def test_non_allowlisted_license_is_rejected():
    row = make_row()
    row["license"] = "All rights reserved"
    row["license_url"] = "https://publisher.example/rights"
    result = validate_real_world_rows([row])
    assert not result.eligible
    assert "license_not_in_reuse_allowlist" in result.row_reasons["RW-1"]


def test_duplicate_and_reversed_sentence_pairs_are_rejected():
    first = make_row("first")
    reversed_row = make_row("reversed", source=first["summary_sentence"], summary=first["source_sentence"])
    result = validate_real_world_rows([first, reversed_row])
    assert not result.eligible
    assert len(result.rejected) == 2
    assert result.reason_counts["duplicate_or_reversed_pair"] == 2
    assert normalized_pair_key(first) == normalized_pair_key(reversed_row)


def test_duplicate_ids_are_rejected():
    first = make_row("same-id")
    second = make_row("same-id", source="A distinct source may matter.", summary="A distinct source might matter.")
    result = validate_real_world_rows([first, second])
    assert len(result.rejected) == 2
    assert all("duplicate_example_id" in result.row_reasons[row["example_id"]] for row in (first, second))


def test_invalid_label_and_incomplete_annotation_are_not_eligible():
    invalid = make_row(label="MAYBE")
    incomplete = make_row("incomplete", label="")
    incomplete["annotation_status"] = "requires_review"
    incomplete["requires_review"] = "true"
    result = validate_real_world_rows([invalid, incomplete])
    assert not result.eligible
    assert "invalid_label" in result.row_reasons["RW-1"]
    assert "llm_requires_review" in result.row_reasons["incomplete"]


def test_training_collision_is_rejected():
    row = make_row()
    training = [{"source_sentence": row["source_sentence"], "summary_sentence": row["summary_sentence"]}]
    result = validate_real_world_rows([row], training_records=training)
    assert "synthetic_training_collision" in result.row_reasons["RW-1"]


@pytest.mark.parametrize(("field", "source", "summary", "reason"), [
    ("negation", "The intervention may reduce symptoms.", "The intervention may not reduce symptoms.",
     "detected_negation_difference"),
    ("numbers", "The intervention may reduce symptoms by 2 units.",
     "The intervention may reduce symptoms by 3 units.", "detected_number_difference"),
    ("dates", "The intervention may begin on March 1, 2024.", "The intervention may begin on March 1, 2025.",
     "detected_date_difference"),
])
def test_unsupported_content_differences_require_human_review(field, source, summary, reason):
    row = make_row(source=source, summary=summary)
    row[field + "_match"] = "uncertain"
    result = validate_real_world_rows([row])
    assert not result.eligible
    assert "RW-1" in result.row_reasons
    assert reason in result.row_reasons["RW-1"]


def test_llm_exclude_is_never_training_eligible():
    row = make_row(label="EXCLUDE")
    row["proposition_comparable"] = "false"
    row["annotation_status"] = "complete"
    result = validate_real_world_rows([row])
    assert not result.eligible
    assert len(result.excluded) == 1


def test_exclude_with_quality_flags_stays_excluded():
    row = make_row(label="EXCLUDE")
    row.update(proposition_comparable="false", annotation_status="requires_review", requires_review="true",
               subject_match="false", certainty_direction="EXCLUDE")
    result = validate_real_world_rows([row])
    assert len(result.excluded) == 1
    assert not result.eligible


def test_inconsistent_llm_strength_and_label_requires_review_not_rejection():
    row = make_row()
    row["summary_certainty"] = "ASSERTED"
    row["certainty_direction"] = "PRESERVED"
    result = validate_real_world_rows([row])
    assert len(result.review_required) == 1
    assert not result.rejected and not result.eligible


def test_pilot_preserved_but_hardened_direction_contradiction_requires_review():
    row=make_v2_row()
    row.update(label="PRESERVED",certainty_direction="HARDENED",source_certainty="UNCLEAR",
               summary_certainty="UNCLEAR")
    result=validate_real_world_rows([row])
    assert not result.eligible
    assert "label_direction_mismatch" in result.row_reasons["RW-1"]


def test_equal_certainty_softened_pilot_pattern_requires_review():
    row=make_v2_row(label="SOFTENED")
    row.update(source_certainty="PROBABLE",summary_certainty="PROBABLE",certainty_direction="SOFTENED")
    result=validate_real_world_rows([row])
    assert not result.eligible
    assert "certainty_label_direction_inconsistent" in result.row_reasons["RW-1"]


def test_invalid_certainty_direction_remains_review_required():
    row=make_v2_row()
    row["certainty_direction"]="UNCERTAIN"
    result=validate_real_world_rows([row])
    assert not result.eligible
    assert "invalid_certainty_direction" in result.row_reasons["RW-1"]


def test_annotation_pipeline_stage_b_version_uses_structured_evidence_branch(monkeypatch):
    from src.dataset.real_world_annotation import assess_certainty_shift

    source="The intervention may reduce symptoms."
    summary="The intervention will reduce symptoms."
    payload={
        "source_certainty":"POSSIBLE", "summary_certainty":"ASSERTED",
        "certainty_direction":"HARDENED", "label":"HARDENED",
        "source_evidence":{"quote":"may reduce symptoms", "evidence_type":"explicit_uncertainty",
                            "interpretation":"May expresses possibility."},
        "summary_evidence":{"quote":"will reduce symptoms", "evidence_type":"certainty",
                            "interpretation":"Will states the outcome directly."},
        "explanation":"The summary expresses greater certainty.",
    }
    monkeypatch.setattr("src.dataset.real_world_annotation._structured_request", lambda *args, **kwargs: payload)
    annotation=assess_certainty_shift(source, summary, comparability_status="comparable")
    assert annotation["annotation_version"] == "hedgeguard-real-world-v2"

    row=make_v2_row(source=source, summary=summary, label="HARDENED")
    row.update(annotation)
    row.update(annotation_method="llm_assisted", model_name="qwen3:4b-instruct", model_version="fixture-digest",
               prompt_version="hedgeguard-real-world-v2", annotation_stage="certainty",
               comparability_status="comparable", proposition_comparable="true",
               subject_match="true", negation_match="true", numbers_match="true", dates_match="true",
               source_proposition=source, summary_proposition=summary,
               # Preserve review status so this regression test cannot make a
               # row eligible merely by removing the false schema errors.
               annotation_status="requires_review", requires_review="true", review_reasons="")
    result=validate_real_world_rows([row])
    reasons=result.row_reasons["RW-1"]
    assert "invalid_source_evidence_schema" not in reasons
    assert "invalid_summary_evidence_schema" not in reasons
    assert "nonexact_or_missing_source_evidence" not in reasons
    assert "nonexact_or_missing_summary_evidence" not in reasons
    assert len(result.review_required)==1 and not result.eligible


@pytest.mark.parametrize("field,value",[("source_evidence",'{}'),
    ("summary_evidence",'{"quote":"not in sentence","evidence_type":"explicit_uncertainty","interpretation":"wrong"}')])
def test_staged_annotation_requires_exact_nonempty_evidence(field,value):
    row=make_v2_row();row[field]=value
    result=validate_real_world_rows([row])
    assert not result.eligible
    assert any(reason.startswith("nonexact_or_missing_") for reason in result.row_reasons["RW-1"])


def test_staged_annotation_requires_an_independent_comparability_assessment():
    row=make_v2_row();row["comparability_status"]=""
    result=validate_real_world_rows([row])
    assert not result.eligible
    assert "invalid_comparability_status" in result.row_reasons["RW-1"]


def test_staged_exclude_must_agree_with_noncomparability_assessment():
    row=make_v2_row(label="EXCLUDE")
    row.update(comparability_status="non_comparable",proposition_comparable="true",
               certainty_direction="EXCLUDE",annotation_status="requires_review",requires_review="true")
    result=validate_real_world_rows([row])
    assert not result.eligible and not result.excluded
    assert len(result.review_required)==1
    assert "comparability_status_field_inconsistent" in result.row_reasons["RW-1"]


def test_legacy_list_evidence_and_staged_annotation_alias_remain_supported():
    legacy = make_row()
    assert len(validate_real_world_rows([legacy]).eligible) == 1
    staged = make_v2_row()
    staged["annotation_version"] = "staged-v2"
    result = validate_real_world_rows([staged])
    assert len(result.eligible) == 1
    assert "invalid_source_evidence_schema" not in result.row_reasons.get("RW-1", ())


def test_uncertain_annotation_stays_review_required():
    row = make_row(label="")
    row["annotation_status"] = "requires_review"
    row["requires_review"] = "true"
    row["proposition_comparable"] = "false"
    row["certainty_direction"] = "UNCERTAIN"
    result = validate_real_world_rows([row])
    assert not result.eligible
    assert len(result.review_required) == 1


def test_stable_id_and_license_allowlist():
    assert stable_example_id("doc", "source", "summary") == stable_example_id("doc", "source", "summary")
    assert license_is_reusable("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/")
    assert license_is_reusable("CC0", "https://creativecommons.org/publicdomain/zero/1.0/")
    assert not license_is_reusable("arXiv distribution license", "")


def test_number_and_calendar_comparisons_reduce_format_false_positives_conservatively():
    assert compare_number_mentions("Allocations were 10%, 20%, 30%, 40%, and 50%.",
                                   "Allocations were 10%-50%.") == "match"
    assert compare_number_mentions("Used the 2021 equation.", "Used the 2012 equation.") == "mismatch"
    assert compare_calendar_dates("Used the 2021 equation.", "Used the 2012 equation.") == "match"
    assert compare_calendar_dates("Measured on March 1, 2024.", "Measured on March 1, 2025.") == "mismatch"


def test_numeric_comparison_ignores_doi_and_reference_numbers_and_abstains_on_multiple_metrics():
    assert compare_number_mentions(
        "These cases were not included in the main study.",
        "The online version contains supplementary material at https://doi.org/10.1186/s12903-026-09560-4.",
    ) == "match"
    assert compare_number_mentions(
        "AUROC was 0.661 and AUPRC was 0.126.",
        "AUROC was 0.806 and AUPRC was 0.258.",
    ) == "mismatch"
    assert compare_number_mentions(
        "AUROC was 0.661 (95% CI, 0.605-0.715).",
        "AUROC was 0.661 (95% CI, 0.606-0.716).",
    ) == "mismatch"
    assert compare_number_mentions("A 12% reduction was observed.", "A 13% reduction was observed.") == "mismatch"
    assert compare_number_mentions("The cohort included 20 participants.",
                                   "The cohort included 30 participants.") == "mismatch"


@pytest.mark.parametrize(("source", "summary", "expected"), [
    ("The estimate was 1,170 participants.", "The estimate was 1170 participants.", "match"),
    ("The claim is supported [2].", "The claim is supported.", "match"),
    ("See Figure 2 and Table S5 for details.", "See Figure 3 and Table S6 for details.", "match"),
    ("The 2021 equation was used.", "The 2012 equation was used.", "mismatch"),
    ("The dose was 2 mg daily.", "The dose was 3 mg daily.", "mismatch"),
    ("For bite collapse, kappa was 0.000.", "For bite collapse, kappa was 0.231 and drifting kappa was 0.116.", "mismatch"),
    ("Adults aged >=18 years were eligible.", "Adults with diabetes were enrolled.", "uncertain"),
    ("The primary threshold was X/Y <0.70, corresponding to X being more than 30% lower than Y.",
     "The primary threshold was X/Y <0.70.", "match"),
    ("AUROC was 0.661 and AUPRC was 0.126 for the clinical core.",
     "External AUROC was 0.806 and AUPRC was 0.258, versus 0.661 and 0.126 for the clinical core.",
     "uncertain"),
])
def test_numeric_observed_pilot_regressions(source, summary, expected):
    assert compare_number_mentions(source, summary) == expected


def test_numeric_regression_sentences_from_reported_pilot_candidates():
    # The same bite-collapse kappa statistic conflicts despite other values.
    source = "Similarly, for bite collapse, high percent agreement (90.0%) was accompanied by a kappa value of 0.000, again attributable to the prevalence paradox."
    summary = "Complexity-related variables showed variable reliability, with substantial agreement for flaring (κ = 0.715) but lower kappa values for bite collapse (κ = 0.231) and drifting (κ = 0.116), influenced by low prevalence."
    assert compare_number_mentions(source, summary) == "mismatch"
    # Citation-only [2] is not a claim quantity.
    source = "International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated [2]."
    summary = source.replace(" [2]", "")
    assert compare_number_mentions(source, summary) == "match"


def test_numeric_regressions_from_all_five_named_pilot_examples():
    assert compare_number_mentions(
        "Similarly, for bite collapse, high percent agreement (90.0%) was accompanied by a kappa value of 0.000, again attributable to the prevalence paradox.",
        "Complexity-related variables showed variable reliability, with substantial agreement for flaring (κ = 0.715) but lower kappa values for bite collapse (κ = 0.231) and drifting (κ = 0.116), influenced by low prevalence.",
    ) == "mismatch"
    assert compare_number_mentions(
        "International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated [2].",
        "International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated.",
    ) == "match"
    assert compare_number_mentions(
        "This retrospective two-center study included adults aged ≥18 years with a recorded clinical diagnosis of diabetes and paired serum creatinine and cystatin C measurements.",
        "We conducted a retrospective two-center study of adults with diabetes.",
    ) == "uncertain"
    assert compare_number_mentions(
        "The primary phenotype (D2) was defined as eGFRcys/eGFRcr <0.70, corresponding to eGFRcys being more than 30% lower than eGFRcr.",
        "Large negative eGFR discordance (D2) was defined as eGFRcys/eGFRcr <0.70.",
    ) == "match"
    assert compare_number_mentions(
        "The clinical core achieved an AUROC of 0.661 (95% CI, 0.605-0.715) and an AUPRC of 0.126 (95% CI, 0.096-0.176).",
        "In independent external validation, AUROC was 0.806 (95% CI, 0.769-0.840) and AUPRC was 0.258 (95% CI, 0.193-0.337), compared with 0.661 and 0.126, respectively, for the clinical core.",
    ) == "uncertain"


def test_staged_number_and_negation_checks_require_exact_comparable_stage_a():
    row = make_v2_row(source="The intervention may reduce symptoms by 2 units.",
                      summary="The intervention may not reduce symptoms by 3 units.")
    row.update(comparability_status="uncertain", proposition_comparable="uncertain",
               source_proposition="", summary_proposition="")
    result = validate_real_world_rows([row])
    reasons = result.row_reasons["RW-1"]
    assert "detected_number_difference" not in reasons
    assert "detected_negation_difference" not in reasons
    assert "number_comparison_uncertain" not in reasons


def test_negation_marker_difference_with_unaligned_predicates_abstains():
    assert compare_negation(
        "The trial was not registered, while treatment may reduce pain.",
        "Treatment may reduce pain.",
    ) == "uncertain"


def test_negation_comparison_ignores_negative_adjective_and_not_only_idiom():
    assert compare_negation("Large negative discordance is observed.", "Large discordance is observed.") == "match"
    assert compare_negation("It is not only possible but likely.", "It is possible and likely.") == "match"
    assert compare_negation("The treatment may help.", "The treatment may not help.") == "mismatch"
    assert compare_negation("The drug does not reduce pain, but improves sleep.",
                            "The drug reduces pain but does not improve sleep.") == "uncertain"


@pytest.mark.parametrize(("source", "summary", "expected"), [
    ("The benefit was not apparent.", "The benefit was apparent.", "mismatch"),
    ("A benefit cannot be inferred.", "A benefit can be inferred.", "mismatch"),
    ("The intervention is not yet sufficient.", "The intervention is sufficient.", "mismatch"),
    ("The result holds with or without refitting.", "The result holds with refitting.", "uncertain"),
    ("The result holds without refitting.", "The result holds with refitting.", "uncertain"),
    ("The trial was not registered; treatment may reduce pain.", "Treatment may reduce pain.", "uncertain"),
    ("There was an absence of evidence for benefit.", "There was evidence for benefit.", "uncertain"),
    ("Large negative discordance was observed.", "Large discordance was observed.", "match"),
])
def test_negation_scope_regressions(source, summary, expected):
    assert compare_negation(source, summary) == expected


def test_source_document_groups_must_not_cross_partitions():
    other_document = make_row("other", source="other source")
    other_document["source_document_id"] = other_document["source_group_id"] = "PMC-fixture-2"
    assert_group_disjoint({"train": [make_row()], "test": [other_document]})
    with pytest.raises(RealWorldDatasetError, match="document leakage"):
        assert_group_disjoint({"train": [make_row()], "test": [make_row("other", source="different sentence")]})
