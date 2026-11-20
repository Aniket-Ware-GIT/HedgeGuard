"""Create an audit addendum without changing the frozen random audit sample."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
from src.dataset.real_world import AUDIT_FIELDS, HUMAN_AUDIT_FIELDS, write_real_world_csv


def build_audit_row(candidate: dict[str, str], annotation: dict[str, str], *, audit_id: str) -> dict[str, str]:
    return {
        "audit_id": audit_id, "example_id": candidate.get("example_id", ""),
        "source_document_id": candidate.get("source_document_id", ""),
        "source_document_title": candidate.get("source_document_title", ""),
        "source_url": candidate.get("source_url", ""), "source_location": candidate.get("source_location", ""),
        "summary_location": candidate.get("summary_location", ""),
        "source_sentence": candidate.get("source_sentence", ""),
        "summary_sentence": candidate.get("summary_sentence", ""),
        "llm_label": annotation.get("label", ""),
        "llm_source_certainty": annotation.get("source_certainty", ""),
        "llm_summary_certainty": annotation.get("summary_certainty", ""),
        "llm_explanation": annotation.get("explanation", ""),
        "llm_source_evidence": annotation.get("source_evidence", ""),
        "llm_summary_evidence": annotation.get("summary_evidence", ""),
        "human_label": "", "auditor_id": "", "audited_at": "", "audit_notes": "", "disagreement_reason": "",
    }


def choose_hardened_case(candidates: Iterable[dict[str, str]], annotations: Iterable[dict[str, str]],
                         reviewable_ids: set[str]) -> tuple[dict[str, str], dict[str, str]]:
    candidates_by_id = {row.get("example_id", ""): row for row in candidates}
    for annotation in annotations:
        example_id = annotation.get("example_id", "")
        if annotation.get("label") == "HARDENED" and example_id in reviewable_ids and example_id in candidates_by_id:
            return candidates_by_id[example_id], annotation
    raise ValueError("No reviewable HARDENED annotation is available for the audit addendum.")


def write_hardened_audit_case(candidate: dict[str, str], annotation: dict[str, str], *,
                              audit_path: str | Path, results_path: str | Path) -> None:
    audit_file, results_file = Path(audit_path), Path(results_path)
    if audit_file.exists() or results_file.exists():
        raise FileExistsError("Refusing to overwrite existing HARDENED audit addendum files.")
    row = build_audit_row(candidate, annotation, audit_id="AUDIT-HARDENED-001")
    write_real_world_csv(audit_file, [row], fields=AUDIT_FIELDS)
    write_real_world_csv(results_file, [], fields=HUMAN_AUDIT_FIELDS)
