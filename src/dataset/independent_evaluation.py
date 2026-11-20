"""Validation and eligibility rules for the independent evaluation corpus."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LABELS = ("PRESERVED", "HARDENED", "SOFTENED", "EXCLUDE")
MODEL_LABELS = LABELS[:3]
REVIEW_STATUSES = ("requires_manual_review", "approved", "excluded", "disagreement", "adjudicated")
REQUIRED_COLUMNS = (
    "example_id", "source_document_id", "source_sentence", "summary_sentence", "label",
    "source_reference", "source_location", "domain", "collection_method", "collection_date",
    "collection_notes", "review_status", "reviewer_1_id", "reviewer_1_label",
    "reviewer_1_at", "reviewer_1_notes", "reviewer_2_id", "reviewer_2_label",
    "reviewer_2_at", "reviewer_2_notes", "adjudicator_id", "adjudicated_label",
    "adjudicated_at", "adjudication_notes", "exclude_reason",
)
PROVENANCE_FIELDS = ("source_document_id", "source_reference", "source_location", "domain")


class IndependentDatasetError(ValueError):
    """Raised when the independent evaluation CSV itself cannot be read safely."""


@dataclass(frozen=True)
class ValidationResult:
    eligible: list[dict[str, str]]
    excluded: list[dict[str, str]]
    rejected: list[dict[str, str]]
    reasons: dict[str, int]
    row_issues: dict[str, tuple[str, ...]]


def read_independent_csv(path: str | Path) -> list[dict[str, str]]:
    """Read the independent CSV and require its complete, versioned header."""
    file = Path(path)
    try:
        with file.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise IndependentDatasetError(f"{file} has no CSV header.")
            missing = [field for field in REQUIRED_COLUMNS if field not in reader.fieldnames]
            if missing:
                raise IndependentDatasetError("Missing required CSV columns: " + ", ".join(missing))
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise IndependentDatasetError(f"Could not read {file}: {exc}") from exc
    return rows


def _norm_sentence(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    """Canonicalize pair order too, so copied content reversed in the CSV is caught."""
    return tuple(sorted((_norm_sentence(row.get("source_sentence", "")),
                         _norm_sentence(row.get("summary_sentence", "")))))  # type: ignore[return-value]


def _filled(row: dict[str, str], field: str) -> str:
    return str(row.get(field, "") or "").strip()


def _reviewer_issues(row: dict[str, str]) -> list[str]:
    issues: list[str] = []
    for index in (1, 2):
        names = (f"reviewer_{index}_id", f"reviewer_{index}_label", f"reviewer_{index}_at",
                 f"reviewer_{index}_notes")
        values = [_filled(row, name) for name in names]
        if any(values) and not all(values):
            issues.append(f"incomplete_reviewer_{index}")
        if values[1] and values[1].upper() not in LABELS:
            issues.append(f"invalid_reviewer_{index}_label")
    id1, id2 = _filled(row, "reviewer_1_id"), _filled(row, "reviewer_2_id")
    if id1 and id2 and id1.casefold() == id2.casefold():
        issues.append("reviewers_not_independent")
    return issues


def validate_independent_records(
    records: Iterable[dict[str, str]], *, training_records: Iterable[dict[str, str]] = (),
) -> ValidationResult:
    """Validate rows, keep explicit EXCLUDE separate, and fail closed on all other issues.

    Rejected rows remain available in ``row_issues`` for an auditable summary. No
    disagreement is resolved automatically; only a recorded adjudication can qualify.
    """
    rows = [dict(row) for row in records]
    training = list(training_records)
    issues_by_index: dict[int, list[str]] = defaultdict(list)
    pair_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    id_indices: dict[str, list[int]] = defaultdict(list)
    training_pairs = {_pair_key(row) for row in training if _pair_key(row) != ("", "")}

    for index, row in enumerate(rows):
        example_id = _filled(row, "example_id")
        if not example_id:
            issues_by_index[index].append("missing_example_id")
        else:
            id_indices[example_id.casefold()].append(index)
        for field in ("source_sentence", "summary_sentence", *PROVENANCE_FIELDS):
            if not _filled(row, field):
                issues_by_index[index].append(f"missing_{field}")
        source, summary = _filled(row, "source_sentence"), _filled(row, "summary_sentence")
        if source and summary:
            pair = _pair_key(row)
            pair_indices[pair].append(index)
            if pair in training_pairs:
                issues_by_index[index].append("training_data_collision")

        label = _filled(row, "label").upper()
        if label not in LABELS and not (not label and _filled(row, "review_status").lower() == "disagreement"):
            issues_by_index[index].append("invalid_label" if label else "missing_label")
        status = _filled(row, "review_status").lower()
        if status not in REVIEW_STATUSES:
            issues_by_index[index].append("invalid_review_status")
        issues_by_index[index].extend(_reviewer_issues(row))
        review1, review2 = (_filled(row, "reviewer_1_label").upper(),
                            _filled(row, "reviewer_2_label").upper())
        both_reviewed = bool(_filled(row, "reviewer_1_id") and _filled(row, "reviewer_2_id")
                             and review1 in LABELS and review2 in LABELS
                             and _filled(row, "reviewer_1_at") and _filled(row, "reviewer_2_at")
                             and _filled(row, "reviewer_1_notes") and _filled(row, "reviewer_2_notes"))

        if status == "requires_manual_review":
            issues_by_index[index].append("incomplete_review")
        elif status == "approved":
            if not both_reviewed:
                issues_by_index[index].append("incomplete_review")
            if not (review1 == review2 == label and label in MODEL_LABELS):
                issues_by_index[index].append("approval_requires_matching_model_labels")
        elif status == "excluded":
            if not both_reviewed:
                issues_by_index[index].append("incomplete_review")
            if not (review1 == review2 == "EXCLUDE" and label == "EXCLUDE"):
                issues_by_index[index].append("exclusion_requires_matching_exclude_labels")
            if not _filled(row, "exclude_reason"):
                issues_by_index[index].append("missing_exclude_reason")
        elif status == "disagreement":
            if not both_reviewed:
                issues_by_index[index].append("incomplete_review")
            if review1 == review2:
                issues_by_index[index].append("disagreement_requires_different_labels")
            if label:
                issues_by_index[index].append("disagreement_must_not_have_resolved_label")
        elif status == "adjudicated":
            if not both_reviewed:
                issues_by_index[index].append("incomplete_review")
            adjudicator_fields = ("adjudicator_id", "adjudicated_label", "adjudicated_at", "adjudication_notes")
            if not all(_filled(row, field) for field in adjudicator_fields):
                issues_by_index[index].append("incomplete_adjudication")
            adjudicated = _filled(row, "adjudicated_label").upper()
            if adjudicated not in LABELS or label != adjudicated:
                issues_by_index[index].append("invalid_adjudicated_label")
            if _filled(row, "adjudicator_id").casefold() in {
                _filled(row, "reviewer_1_id").casefold(), _filled(row, "reviewer_2_id").casefold()
            }:
                issues_by_index[index].append("adjudicator_must_be_independent")

    for indexes in id_indices.values():
        if len(indexes) > 1:
            for index in indexes:
                issues_by_index[index].append("duplicate_example_id")
    for indexes in pair_indices.values():
        if len(indexes) > 1:
            for index in indexes:
                issues_by_index[index].append("duplicate_pair")

    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    row_issues: dict[str, tuple[str, ...]] = {}
    reasons: Counter[str] = Counter()
    id_counts = Counter(_filled(row, "example_id").casefold() for row in rows if _filled(row, "example_id"))
    for index, row in enumerate(rows):
        row_id = _filled(row, "example_id") or f"row_{index + 1}"
        issues = tuple(dict.fromkeys(issues_by_index[index]))
        if issues:
            rejected.append(row)
            display_id = f"{row_id}#row_{index + 1}" if id_counts[row_id.casefold()] > 1 else row_id
            row_issues[display_id] = issues
            reasons.update(issues)
        elif _filled(row, "label").upper() == "EXCLUDE":
            excluded.append(row)
            reasons["human_excluded"] += 1
        elif _filled(row, "review_status").lower() in {"approved", "adjudicated"}:
            eligible.append(row)
        elif _filled(row, "review_status").lower() == "disagreement":
            rejected.append(row)
            row_issues[row_id] = ("unresolved_disagreement",)
            reasons["unresolved_disagreement"] += 1
        else:
            rejected.append(row)
            row_issues[row_id] = ("incomplete_review",)
            reasons["incomplete_review"] += 1
    return ValidationResult(eligible, excluded, rejected, dict(sorted(reasons.items())), row_issues)
