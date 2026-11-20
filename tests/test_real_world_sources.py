"""Source and pair extraction tests use synthetic in-memory fixtures only."""
from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world_sources import (  # noqa: E402
    Document, SourceCollectionError, extract_document_candidates, parse_pmc_oai_record,
    search_pmc, split_sentences, _usable_candidate_sentence,
)


class FixtureScorer:
    model_name = "fixture-embeddings"

    def score_pairs(self, pairs):
        return [0.9 if "treatment" in left.lower() and "treatment" in right.lower() else 0.05
                for left, right in pairs]

PMC_FIXTURE = b'''<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
 <GetRecord><record><metadata><article xmlns="http://jats.nlm.nih.gov" xmlns:xlink="http://www.w3.org/1999/xlink">
  <front><article-meta><article-id pub-id-type="pmc">PMC9990001</article-id>
   <article-id pub-id-type="doi">10.0000/fixture</article-id>
   <title-group><article-title>Fixture title for parser tests</article-title></title-group>
   <pub-date><year>2024</year><month>3</month><day>5</day></pub-date>
   <abstract><p>The treatment may lower blood pressure in adults with hypertension.</p></abstract>
   <permissions><license xlink:href="https://creativecommons.org/licenses/by/4.0/">
    <license-p>Creative Commons Attribution 4.0 International</license-p>
   </license></permissions>
  </article-meta></front>
  <body><sec><title>Results</title><p>Our randomized study suggests the treatment may lower blood pressure in adults with hypertension during follow-up.</p></sec></body>
 </article></metadata></record></GetRecord>
</OAI-PMH>'''


def test_pmc_parser_preserves_metadata_license_abstract_and_body():
    document = parse_pmc_oai_record(PMC_FIXTURE, domain="biomedical", collection_query="fixture only")
    assert document.metadata["source_document_id"] == "PMC9990001"
    assert document.metadata["source_document_title"] == "Fixture title for parser tests"
    assert document.metadata["doi"] == "10.0000/fixture"
    assert document.metadata["publication_date"] == "2024-03-05"
    assert "creativecommons.org/licenses/by/4.0/" in document.metadata["license_url"]
    assert document.abstract_paragraphs
    assert document.body_paragraphs[0][0] == "Results"


def test_parser_rejects_missing_article():
    with pytest.raises(SourceCollectionError, match="did not contain an article"):
        parse_pmc_oai_record(b"<root />")


def test_candidate_extraction_only_uses_same_document_and_keeps_locations():
    document = parse_pmc_oai_record(PMC_FIXTURE, domain="biomedical", collection_query="fixture only")
    candidates = extract_document_candidates(document, scorer=FixtureScorer(), minimum_semantic_score=0.2)
    assert candidates
    row = candidates[0]
    assert row["source_document_id"] == row["source_group_id"] == "PMC9990001"
    assert row["source_sentence"] in document.body_paragraphs[0][1]
    assert row["summary_sentence"] in document.abstract_paragraphs[0]
    assert row["source_location"].startswith("Results;")
    assert row["summary_location"].startswith("Abstract;")
    assert row["annotation_status"] == "not_annotated"
    assert row["requires_review"] == "true"
    assert row["alignment_status"] == "requires_proposition_assessment"
    assert row["semantic_score"] == "0.900000"
    assert row["alignment_model"] == "fixture-embeddings"
    assert row["machine_label_eligible"] == "false"


def test_unrelated_body_abstract_text_does_not_make_a_candidate():
    document = Document(
        {"source_document_id": "local-fixture", "source_document_title": "Fixture", "source_url": "",
         "publication_date": "", "domain": "fixture", "doi": "", "license": "", "license_url": ""},
        ("The treatment may lower blood pressure in adults with hypertension.",),
        (("Methods", "We prepared a standard laboratory instrument for sample storage."),), b"fixture",
    )
    assert not extract_document_candidates(document, scorer=FixtureScorer(), minimum_semantic_score=0.2)


def test_pmc_search_calls_are_serial_and_license_filtered():
    search_response = json.dumps({"esearchresult": {"idlist": ["9990001"]}}).encode()
    with patch("src.dataset.real_world_sources._http_get", side_effect=[search_response, PMC_FIXTURE]) as get:
        documents = search_pmc("test query", 1, domain="fixture", minimum_interval=0)
    assert len(documents) == 1
    assert get.call_count == 2
    assert urllib.parse.parse_qs(urllib.parse.urlparse(get.call_args_list[0].args[0]).query)["db"] == ["pmc"]


def test_pmc_search_does_not_return_non_allowlisted_content():
    non_reusable = PMC_FIXTURE.replace(b"creativecommons.org/licenses/by/4.0/",
                                       b"publisher.example/all-rights-reserved")
    search_response = json.dumps({"esearchresult": {"idlist": ["9990001"]}}).encode()
    noted = []
    with patch("src.dataset.real_world_sources._http_get", side_effect=[search_response, non_reusable]):
        documents = search_pmc("test query", 1, domain="fixture", minimum_interval=0,
                               on_candidate=noted.append)
    assert not documents
    assert noted[0]["rejection_reason"] == "license_not_in_reuse_allowlist"


def test_sentence_splitter_does_not_split_inside_non_boundary_space():
    parts = split_sentences("One sentence. Next sentence with uncertainty.")
    assert parts == ["One sentence.", "Next sentence with uncertainty."]


def test_candidate_extraction_filters_reference_and_caption_fragments_before_scoring():
    document = Document(
        {"source_document_id": "filter-fixture", "source_document_title": "Fixture",
         "source_url": "https://example.test", "publication_date": "2024-01-01",
         "domain": "fixture", "doi": "", "license": "CC BY 4.0",
         "license_url": "https://creativecommons.org/licenses/by/4.0/",
         "collection_method": "fixture", "collection_date": "2026-01-01"},
        ("The intervention may reduce symptoms in adults with chronic conditions. "
         "Supplementary material available at https://doi.org/10.1234/5678.",),
        (("Results", "The intervention may reduce symptoms in adults with chronic conditions. "
          "Figure 2: The intervention may reduce symptoms in adults with chronic conditions."),),
        b"fixture",
    )
    class InterventionScorer:
        model_name = "fixture-embeddings"

        def score_pairs(self, pairs):
            return [0.9 if "intervention" in left.lower() and "intervention" in right.lower() else 0.05
                    for left, right in pairs]

    candidates = extract_document_candidates(document, scorer=InterventionScorer(), minimum_semantic_score=0.2)
    assert candidates
    assert all(not row["source_sentence"].casefold().startswith("figure 2") for row in candidates)
    assert all("doi.org/" not in row["summary_sentence"].casefold() for row in candidates)


def test_formula_only_fragments_are_filtered_but_prose_definitions_are_retained():
    assert not _usable_candidate_sentence("result = alpha * beta / gamma + delta - epsilon < threshold + zeta")
    assert not _usable_candidate_sentence(
        "The full text of the article is available from https://example.org/paper."
    )
    assert _usable_candidate_sentence(
        "The primary phenotype was defined as the eGFR cystatin ratio below 0.70."
    )
