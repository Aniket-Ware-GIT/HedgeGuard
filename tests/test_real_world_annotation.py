"""Mocked tests for local Qwen annotation and audit/split safeguards."""
from __future__ import annotations
import json, sys
from pathlib import Path
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world_annotation import (
 OllamaAnnotationError, STAGED_PROMPT_VERSION, annotate_pair, assess_certainty_shift,
 assess_proposition_comparability, stage_b_candidates, verify_ollama,
)

def payload(label="PRESERVED",comparable=True,review=False):
 return {"proposition_comparable":comparable,"label":label,"requires_review":review,"source_certainty":"POSSIBLE","summary_certainty":"POSSIBLE","certainty_direction":label if label else "UNCERTAIN","source_evidence":["may"],"summary_evidence":["may"],"explanation":"The modal wording is retained.","subject_match":"true","negation_match":"true","numbers_match":"true","dates_match":"true"}

def response(obj): return {"message":{"content":json.dumps(obj)}}

@pytest.mark.parametrize("label",["PRESERVED","HARDENED","SOFTENED"])
def test_valid_class_annotation(label):
 with patch("src.dataset.real_world_annotation._request",return_value=response(payload(label))):
  row=annotate_pair("The drug may help.","The drug may help.",model_version="digest")
 assert row["label"]==label and row["annotation_method"]=="llm_assisted" and row["annotation_status"]=="complete"

def test_noncomparable_pair_is_exclude_and_requires_review():
 with patch("src.dataset.real_world_annotation._request",return_value=response(payload("EXCLUDE",False))):
  row=annotate_pair("Source","Different proposition",model_version="digest")
 assert row["label"]=="EXCLUDE" and row["requires_review"]=="true" and row["annotation_status"]=="requires_review"

def test_uncertain_pair_has_no_forced_label():
 obj=payload("PRESERVED",review=True)
 with patch("src.dataset.real_world_annotation._request",return_value=response(obj)):
  row=annotate_pair("Source","Summary",model_version="digest")
 assert row["label"]=="" and row["requires_review"]=="true"

def test_invalid_label_and_malformed_json_rejected():
 bad=payload("OTHER")
 with patch("src.dataset.real_world_annotation._request",return_value=response(bad)),pytest.raises(OllamaAnnotationError): annotate_pair("S","T")
 with patch("src.dataset.real_world_annotation._request",return_value={"message":{"content":"{"}}),pytest.raises(OllamaAnnotationError): annotate_pair("S","T")

def test_missing_required_schema_rejected():
 with patch("src.dataset.real_world_annotation._request",return_value=response({"label":"PRESERVED"})),pytest.raises(OllamaAnnotationError): annotate_pair("S","T")

def test_empty_input_rejected_before_http():
 with patch("src.dataset.real_world_annotation._request") as req,pytest.raises(ValueError): annotate_pair(" ","Summary")
 req.assert_not_called()

def test_model_preflight_requires_exact_installed_tag():
 with patch("src.dataset.real_world_annotation._request",return_value={"models":[{"name":"qwen3:4b-instruct","digest":"sha"}]}):
  assert verify_ollama()=="sha"
 with patch("src.dataset.real_world_annotation._request",return_value={"models":[]} ),pytest.raises(OllamaAnnotationError,match="ollama pull"): verify_ollama()

def test_timeout_and_connection_errors_are_useful():
 with patch("src.dataset.real_world_annotation.urllib.request.urlopen",side_effect=TimeoutError()),pytest.raises(OllamaAnnotationError,match="timed out"): annotate_pair("S","T",timeout=1)
 with patch("src.dataset.real_world_annotation.urllib.request.urlopen",side_effect=OSError("connection refused")),pytest.raises(OllamaAnnotationError): annotate_pair("S","T")


def test_staged_prompt_and_annotation_version_share_one_contract():
 from src.dataset.real_world import STAGED_ANNOTATION_VERSION
 assert STAGED_PROMPT_VERSION == STAGED_ANNOTATION_VERSION == "hedgeguard-real-world-v2"


def test_annotation_failures_abstain_and_are_retryable_on_explicit_resume():
 from scripts.annotate_real_pairs import _certainty_error, _is_retryable_error
 row = _certainty_error("example", "malformed JSON", "digest")
 assert row["label"] == ""
 assert row["annotation_status"] == "requires_review"
 assert row["requires_review"] == "true"
 assert _is_retryable_error(row)
 assert not _is_retryable_error({"annotation_status": "complete", "annotation_raw_json": "{}"})


def stage_a(status="comparable", *, subject="true", negation="true", numbers="true", dates="true"):
 return {"comparability_status":status,"source_proposition":"The treatment may reduce symptoms.",
         "summary_proposition":"The treatment might reduce symptoms.","subject_match":subject,
         "negation_match":negation,"numbers_match":numbers,"dates_match":dates,
         "reason":"Both sentences state the same treatment effect."}


def stage_b(label="PRESERVED", source="POSSIBLE", summary="POSSIBLE", direction="PRESERVED",
            source_quote="may", summary_quote="might"):
 return {"source_certainty":source,"summary_certainty":summary,"certainty_direction":direction,
         "label":label,
         "source_evidence":{"quote":source_quote,"evidence_type":"explicit_uncertainty","interpretation":"May marks possibility."},
         "summary_evidence":{"quote":summary_quote,"evidence_type":"explicit_uncertainty","interpretation":"Might marks possibility."},
         "explanation":"Both statements express similar possibility."}


def test_stage_a_comparability_does_not_predict_certainty_label():
 with patch("src.dataset.real_world_annotation._structured_request",return_value=stage_a()):
  row=assess_proposition_comparability("The treatment may reduce symptoms.","The treatment might reduce symptoms.")
 assert row["comparability_status"]=="comparable" and row["proposition_comparable"]=="true"
 assert "label" not in row and row["annotation_status"]=="complete"


@pytest.mark.parametrize("status",["non_comparable","uncertain"])
def test_stage_a_noncomparable_or_uncertain_never_completes(status):
 with patch("src.dataset.real_world_annotation._structured_request",return_value=stage_a(status)):
  row=assess_proposition_comparability("The treatment may reduce symptoms.","The treatment might reduce symptoms.")
 assert row["annotation_status"]=="requires_review" and row["requires_review"]=="true"


def test_stage_a_flags_conflict_to_uncertain_and_requires_exact_proposition_quotes():
 with patch("src.dataset.real_world_annotation._structured_request",return_value=stage_a(subject="false")):
  row=assess_proposition_comparability("The treatment may reduce symptoms.","The treatment might reduce symptoms.")
 assert row["comparability_status"]=="uncertain"
 invalid=stage_a();invalid["source_proposition"]="possible effect"
 with patch("src.dataset.real_world_annotation._structured_request",return_value=invalid):
  row=assess_proposition_comparability("The treatment may reduce symptoms.","The treatment might reduce symptoms.")
 assert row["comparability_status"]=="uncertain"


def test_stage_b_is_refused_unless_stage_a_says_comparable():
 with patch("src.dataset.real_world_annotation._structured_request") as request,pytest.raises(ValueError,match="only when Stage A"):
  assess_certainty_shift("S","T",comparability_status="uncertain")
 request.assert_not_called()


def test_stage_b_candidate_selector_excludes_noncomparable_uncertain_and_completed():
 candidates=[{"example_id":"a"},{"example_id":"b"},{"example_id":"c"},{"example_id":"d"}]
 assessments=[{"example_id":"a","comparability_status":"comparable"},
              {"example_id":"b","comparability_status":"non_comparable"},
              {"example_id":"c","comparability_status":"uncertain"},
              {"example_id":"d","comparability_status":"comparable"}]
 assert stage_b_candidates(candidates,assessments,{"d"})==[{"example_id":"a"}]


def test_stage_b_accepts_consistent_exact_nonempty_evidence():
 with patch("src.dataset.real_world_annotation._structured_request",return_value=stage_b()):
  row=assess_certainty_shift("The treatment may reduce symptoms.","The treatment might reduce symptoms.",comparability_status="comparable")
 assert row["annotation_status"]=="complete" and row["requires_review"]=="false"


def test_stage_b_contradictory_preserved_hardened_and_equal_certainty_softened_require_review():
 contradictory=stage_b(label="PRESERVED",direction="HARDENED")
 with patch("src.dataset.real_world_annotation._structured_request",return_value=contradictory):
  row=assess_certainty_shift("The treatment may reduce symptoms.","The treatment might reduce symptoms.",comparability_status="comparable")
 assert row["annotation_status"]=="requires_review"
 assert "certainty_label_direction_inconsistent" in row["review_reasons"]
 softened=stage_b(label="SOFTENED",direction="SOFTENED",source="PROBABLE",summary="PROBABLE")
 with patch("src.dataset.real_world_annotation._structured_request",return_value=softened):
  row=assess_certainty_shift("The treatment may reduce symptoms.","The treatment might reduce symptoms.",comparability_status="comparable")
 assert row["annotation_status"]=="requires_review"


def test_stage_b_requires_verbatim_evidence_and_interpretation():
 bad=stage_b(source_quote="possibly")
 bad["summary_evidence"]={"quote":"","evidence_type":"explicit_uncertainty","interpretation":""}
 with patch("src.dataset.real_world_annotation._structured_request",return_value=bad):
  row=assess_certainty_shift("The treatment may reduce symptoms.","The treatment might reduce symptoms.",comparability_status="comparable")
 assert row["annotation_status"]=="requires_review"
 assert "invalid_source_evidence" in row["review_reasons"]
 assert "invalid_summary_evidence" in row["review_reasons"]


def test_stage_b_no_explicit_cue_can_use_quoted_asserted_predicate():
 data=stage_b(label="PRESERVED",direction="PRESERVED",source="ASSERTED",summary="ASSERTED",
              source_quote="reduces",summary_quote="reduces")
 data["source_evidence"]["evidence_type"]="asserted_predicate"
 data["summary_evidence"]["evidence_type"]="asserted_predicate"
 with patch("src.dataset.real_world_annotation._structured_request",return_value=data):
  row=assess_certainty_shift("The treatment reduces symptoms.","The treatment reduces symptoms.",comparability_status="comparable")
 assert row["annotation_status"]=="complete"
