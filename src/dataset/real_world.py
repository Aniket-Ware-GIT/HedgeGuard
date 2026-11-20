"""Schemas, provenance checks, candidate eligibility, and split safeguards for real data.

Machine annotations in this module remain explicitly distinct from human review
and from the existing synthetic training eligibility gate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LABELS = ("PRESERVED", "HARDENED", "SOFTENED", "EXCLUDE")
MODEL_LABELS = LABELS[:3]
CERTAINTY_LEVELS = ("POSSIBLE", "PROBABLE", "ASSERTED", "UNCLEAR")
# Canonical schema identifier emitted by the staged annotation pipeline.
STAGED_ANNOTATION_VERSION = "hedgeguard-real-world-v2"
# `staged-v2` was accepted by the validator before the canonical identifier
# was shared with the annotation pipeline. Keep it as a structured-v2 alias.
STRUCTURED_ANNOTATION_VERSIONS = frozenset({STAGED_ANNOTATION_VERSION, "staged-v2"})
ANNOTATION_STATUSES = ("not_annotated", "complete", "requires_review", "error")
PIPELINE_STATUSES = ("candidate", "eligible", "excluded", "review_required", "rejected")
QUALITY_FLAGS = ("subject_match", "negation_match", "numbers_match", "dates_match")

REAL_WORLD_FIELDS = (
    "example_id", "source_document_id", "source_document_title", "source_url", "doi",
    "publication_date", "source_sentence", "summary_sentence", "domain", "source_section",
    "source_location", "summary_location", "collection_method", "collection_date",
    "license", "license_url", "source_version", "source_content_hash", "source_group_id", "candidate_score",
    "label", "proposition_comparable", "source_certainty", "summary_certainty",
    "certainty_direction", "source_evidence", "summary_evidence", "explanation",
    "subject_match", "negation_match", "numbers_match", "dates_match", "requires_review",
    "annotation_method", "model_name", "model_version", "prompt_version", "annotation_status",
    "pipeline_status", "machine_label_eligible", "rejection_reasons", "annotation_raw_json",
    "split", "split_seed",
)

# Versioned schema for the revised retrieval/staged-annotation pipeline. Keep the
# legacy schema intact so existing pilot CSVs remain readable and unchanged.
ALIGNMENT_FIELDS = ("lexical_score", "semantic_score", "alignment_model", "alignment_status")
REAL_WORLD_V2_FIELDS = REAL_WORLD_FIELDS + ALIGNMENT_FIELDS + (
    "annotation_version", "annotation_stage", "proposition_reason",
    "comparability_status", "source_proposition", "summary_proposition", "review_reasons",
)

DOCUMENT_FIELDS = (
    "source_document_id", "source_document_title", "source_url", "doi", "publication_date",
    "domain", "repository", "repository_record_id", "license", "license_url", "collection_method",
    "collection_query", "collection_date", "source_version", "source_content_hash", "abstract_text",
    "local_source_path",
)

AUDIT_FIELDS = (
    "audit_id", "example_id", "source_document_id", "source_document_title", "source_url",
    "source_location", "summary_location", "source_sentence", "summary_sentence", "llm_label",
    "llm_source_certainty", "llm_summary_certainty", "llm_explanation", "llm_source_evidence",
    "llm_summary_evidence", "human_label", "auditor_id", "audited_at", "audit_notes",
    "disagreement_reason",
)
HUMAN_AUDIT_FIELDS = ("audit_id", "example_id", "human_label", "auditor_id", "audited_at", "audit_notes", "disagreement_reason")

PROVENANCE_REQUIRED = (
    "source_document_id", "source_document_title", "source_url", "publication_date", "domain",
    "source_location", "summary_location", "collection_method", "collection_date", "license", "license_url",
)
SENTENCE_FIELDS = ("source_sentence", "summary_sentence")
LICENSE_ALLOWLIST = ("creativecommons.org/publicdomain/zero/1.0", "creativecommons.org/licenses/by/")
_SPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)
_NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_RANGE_RE = re.compile(rf"(?<!\w)(?P<low>{_NUM})\s*(?P<u1>%|percent|mg|ml|kg|cm|mm|years?|months?|days?)?\s*(?:-|–|—|\bto\b)\s*(?P<high>{_NUM})\s*(?P<u2>%|percent|mg|ml|kg|cm|mm|years?|months?|days?)?", re.I)
_NUMBER_RE = re.compile(rf"(?<!\w){_NUM}\s*(?:%|percent|mg|ml|kg|cm|mm|years?|months?|days?)?", re.I)
_DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?)\b", re.I,
)
_NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|nor|without|cannot|can't|doesn't|don't|didn't|isn't|aren't|wasn't|weren't|hasn't|haven't|won't|wouldn't|shouldn't|absence|absent|insufficient)\b", re.I)
_SPACE_NORMALIZER = re.compile(r"\s+")
_NUMERIC_METADATA_RE = re.compile(
    r"https?://\S+|\bdoi\s*:\s*\S+|\b10\.\d{4,9}/\S+|"
    r"\b(?:fig(?:ure)?|table|equation|eq\.?|section|supplementary\s+(?:table|figure|methods)|ref(?:erence)?)\s*\.?\s*\w+|"
    r"\[\s*\d+(?:\s*[-,;]\s*\d+)*\s*\]", re.I,
)


class RealWorldDatasetError(ValueError):
    """Raised for an unusable real-world dataset file or unsafe operation."""


@dataclass(frozen=True)
class RealWorldValidation:
    eligible: list[dict[str, str]]
    excluded: list[dict[str, str]]
    review_required: list[dict[str, str]]
    rejected: list[dict[str, str]]
    reason_counts: dict[str, int]
    row_reasons: dict[str, tuple[str, ...]]


def _value(row: dict[str, str], field: str) -> str:
    return str(row.get(field, "") or "").strip()


def _exact_quote(sentence: str, quote: str) -> bool:
    return bool(quote.strip()) and _SPACE_NORMALIZER.sub(" ", quote.strip()).casefold() in _SPACE_NORMALIZER.sub(
        " ", sentence.strip()
    ).casefold()


def _decimal(value: str) -> str:
    from decimal import Decimal
    return str(Decimal(value.replace(",", "")).normalize())


def _numeric_mentions(text: str) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    # References and URLs are bibliographic metadata, not claim quantities.
    text = _NUMERIC_METADATA_RE.sub(" ", text)
    ranges: list[tuple[str, str, str]] = []
    consumed: list[tuple[int, int]] = []
    for match in _RANGE_RE.finditer(text):
        unit = (match.group("u2") or match.group("u1") or "").casefold().replace("percent", "%")
        ranges.append((_decimal(match.group("low")), _decimal(match.group("high")), unit))
        consumed.append(match.span())
    remaining = list(text)
    for start, end in consumed:
        remaining[start:end] = " " * (end - start)
    points: list[tuple[str, str]] = []
    for match in _NUMBER_RE.finditer("".join(remaining)):
        token = match.group(0)
        number = re.match(_NUM, token)
        if not number:
            continue
        unit = token[number.end():].strip().casefold().replace("percent", "%")
        points.append((_decimal(number.group(0)), unit))
    return points, ranges


_METRIC_RE = re.compile(
    r"\b(AUROC|AUPRC|Brier(?:\s+score)?|kappa(?:\s+value)?|confidence\s+interval|CI|"
    r"percent\s+agreement|participants?|patients?|cases?|subjects?|years?|months?|days?)\b|κ",
    re.I,
)
_ENTITY_CUE_RE = re.compile(
    r"\b(?:for|among|in)\s+([a-z][a-z-]*(?:\s+(?!high\b|low\b|percent\b|value\b|was\b|were\b|is\b|are\b|with\b|and\b|but\b|the\b)[a-z][a-z-]*){0,2})",
    re.I,
)


def _numeric_quantity_records(text: str) -> list[tuple[str, str, str]]:
    """Return (metric/context key, normalized value, unit) for safe alignments.

    Keys are deliberately coarse: they only support warnings when both sides
    contain an unmistakable shared metric/context. They are not semantic parsing.
    """
    cleaned = list(_NUMERIC_METADATA_RE.sub(lambda m: " " * len(m.group(0)), text))
    cleaned_text = "".join(cleaned)
    mentions: list[tuple[int, int, str, str]] = []
    consumed: list[tuple[int, int]] = []
    for match in _RANGE_RE.finditer(cleaned_text):
        unit = (match.group("u2") or match.group("u1") or "").casefold().replace("percent", "%")
        value = f"{_decimal(match.group('low'))}:{_decimal(match.group('high'))}"
        mentions.append((*match.span(), value, unit))
        consumed.append(match.span())
    remaining = list(cleaned_text)
    for start, end in consumed:
        remaining[start:end] = " " * (end - start)
    for match in _NUMBER_RE.finditer("".join(remaining)):
        token = match.group(0)
        number = re.match(_NUM, token)
        if number:
            unit = token[number.end():].strip().casefold().replace("percent", "%")
            mentions.append((*match.span(), _decimal(number.group(0)), unit))
    records = []
    for start, end, value, unit in sorted(mentions):
        prefix = cleaned_text[max(0, start - 110):start]
        metric_matches = list(_METRIC_RE.finditer(prefix))
        metric = metric_matches[-1].group(0).casefold() if metric_matches else ""
        suffix = cleaned_text[end:end + 32].lstrip()
        trailing_metric = _METRIC_RE.match(suffix)
        if trailing_metric and not metric:
            metric = trailing_metric.group(0).casefold()
        if metric in {"κ", "kappa value"}:
            metric = "kappa"
        elif metric == "confidence interval":
            metric = "ci"
        elif metric == "brier score":
            metric = "brier"
        elif metric in {"ci", "confidence interval"}:
            ci_position = metric_matches[-1].start() if metric_matches else 0
            prior_metrics = [m.group(0).casefold() for m in _METRIC_RE.finditer(prefix[:ci_position])
                             if m.group(0).casefold() not in {"ci", "confidence interval"}]
            metric = (prior_metrics[-1] + "_ci") if prior_metrics else "ci"
        if re.search(r"\b(?:compared\s+(?:with|to)|versus|vs\.?)\b", prefix, re.I):
            # Lists of comparator values (often followed by respectively) need
            # role-to-metric alignment; do not inherit the preceding CI label.
            metric = ""
        entity_matches = list(_ENTITY_CUE_RE.finditer(prefix))
        entity = " ".join(entity_matches[-1].group(1).casefold().split()) if entity_matches else ""
        # Nearby metric-specific terms distinguish 90% agreement from a 95% CI.
        key = "|".join(part for part in (entity, metric) if part)
        records.append((key, value, unit))
    return records


_RATIO_RE = re.compile(r"\b([\w]+)\s*/\s*([\w]+)\s*<\s*(\d+(?:\.\d+)?)\b", re.I)
_LOWER_PERCENT_RE = re.compile(
    r"\b([\w]+)(?:\s+being)?\s+more\s+than\s+(\d+(?:\.\d+)?)\s*%\s+lower\s+than\s+([\w]+)\b", re.I,
)


def _remove_safe_ratio_restatement(text: str, other: str) -> str:
    """Remove a redundant verbal ratio restatement only when algebra matches."""
    other_ratios = {(m.group(1).casefold(), m.group(2).casefold(), m.group(3))
                    for m in _RATIO_RE.finditer(other)}
    if not other_ratios:
        return text
    from decimal import Decimal
    for verbal in _LOWER_PERCENT_RE.finditer(text):
        numerator, percent, denominator = verbal.group(1).casefold(), verbal.group(2), verbal.group(3).casefold()
        for ratio_num, ratio_den, threshold in _RATIO_RE.findall(text):
            if (ratio_num.casefold(), ratio_den.casefold(), threshold) not in other_ratios:
                continue
            if numerator == ratio_num.casefold() and denominator == ratio_den.casefold():
                if Decimal("1") - Decimal(threshold) == Decimal(percent) / Decimal("100"):
                    return text[:verbal.start()] + " " * (verbal.end() - verbal.start()) + text[verbal.end():]
    return text


def compare_number_mentions(source: str, summary: str) -> str:
    """Conservatively compare quantities; unmatched multi-quantity mappings abstain.

    Bare years remain numeric values here (calendar dates are checked separately).
    A clear one-to-one quantity difference or a conflicting shared metric/context
    is a mismatch. Added, omitted, or ambiguously aligned quantities return
    ``uncertain`` rather than being compared as unrelated raw numbers.
    """
    source = _remove_safe_ratio_restatement(source, summary)
    summary = _remove_safe_ratio_restatement(summary, source)
    source_points, source_ranges = _numeric_mentions(source)
    summary_points, summary_ranges = _numeric_mentions(summary)
    if Counter(source_points) == Counter(summary_points) and Counter(source_ranges) == Counter(summary_ranges):
        return "match"
    # A complete evenly spaced list and an endpoint range can be equivalent in
    # reporting (e.g. 10%, 20%, ... 50% versus 10%-50%).
    def list_matches_range(points, ranges, other_points, other_ranges):
        if len(ranges) != 1 or other_ranges or len(other_points) < 2:
            return False
        low, high, unit = ranges[0]
        ordered = sorted((float(value), current_unit) for value, current_unit in other_points)
        if any(current_unit != unit for _, current_unit in ordered):
            return False
        values = [value for value, _ in ordered]
        step = values[1] - values[0]
        if step <= 0 or not math.isclose(values[0], float(low)) or not math.isclose(values[-1], float(high)):
            return False
        return all(math.isclose(values[i] - values[i - 1], step) for i in range(2, len(values)))
    if list_matches_range(source_points, source_ranges, summary_points, summary_ranges):
        return "match"
    if list_matches_range(summary_points, summary_ranges, source_points, source_ranges):
        return "match"
    # A clearly shared metric/context with different values is a genuine warning,
    # even when a sentence contains multiple statistics (e.g. kappa by variable).
    source_records = _numeric_quantity_records(source)
    summary_records = _numeric_quantity_records(summary)
    source_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    summary_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, value, unit in source_records:
        if key:
            source_by_key[(key, unit)].append(value)
    for key, value, unit in summary_records:
        if key:
            summary_by_key[(key, unit)].append(value)
    source_all_values = {value for _key, value, _unit in source_records}
    summary_all_values = {value for _key, value, _unit in summary_records}
    for key in source_by_key.keys() & summary_by_key.keys():
        if Counter(source_by_key[key]) != Counter(summary_by_key[key]):
            # In a comparison sentence, the same statistic may be repeated for
            # a second cohort (e.g. external 0.806 versus clinical-core 0.661).
            # If either side's values occur elsewhere on the opposite side, the
            # checker cannot map them safely and must abstain.
            source_values, summary_values = set(source_by_key[key]), set(summary_by_key[key])
            preserved_elsewhere = bool(source_values & summary_all_values) or bool(summary_values & source_all_values)
            if not preserved_elsewhere:
                return "mismatch"
    source_count = len(source_points) + 2 * len(source_ranges)
    summary_count = len(summary_points) + 2 * len(summary_ranges)
    if source_count == 1 and summary_count == 1:
        return "mismatch"
    # Same numeric values with different units/formatting could be equivalent,
    # but the current checker has no safe unit conversion or semantic mapping.
    # Likewise, omitted/additional values need proposition-level alignment.
    return "uncertain"


def _normalized_date(value: str) -> str:
    from datetime import datetime
    candidate = re.sub(r"\s+", " ", value.strip())
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y", "%B %d", "%b %d"):
        try:
            parsed = datetime.strptime(candidate, fmt)
            return parsed.strftime("%Y-%m-%d") if "%Y" in fmt else parsed.strftime("%m-%d")
        except ValueError:
            continue
    return candidate.casefold()


def compare_calendar_dates(source: str, summary: str) -> str:
    """Compare explicit calendar dates; do not misclassify bare years as dates."""
    left = [_normalized_date(match.group(0)) for match in _DATE_RE.finditer(source)]
    right = [_normalized_date(match.group(0)) for match in _DATE_RE.finditer(summary)]
    return "match" if Counter(left) == Counter(right) else "mismatch"


def _negation_markers(text: str) -> list[str]:
    markers = []
    lower = text.casefold()
    for match in _NEGATION_RE.finditer(text):
        token = match.group(0).casefold()
        tail = lower[match.end():].lstrip()
        if token == "not" and (tail.startswith("only ") or tail.startswith("just ")):
            continue
        # The idiom “with or without X” explicitly includes both alternatives;
        # it is not a single-polarity negation cue.
        prefix = lower[max(0, match.start() - 24):match.start()]
        if token == "without" and re.search(r"\bwith\s+or\s+$", prefix):
            continue
        markers.append(token)
    return markers


def compare_negation(source: str, summary: str) -> str:
    """A narrow polarity check; complex/multiple negations stay uncertain."""
    source_with_or_without = bool(re.search(r"\bwith\s+or\s+without\b", source, re.I))
    summary_with_or_without = bool(re.search(r"\bwith\s+or\s+without\b", summary, re.I))
    if source_with_or_without != summary_with_or_without:
        return "uncertain"
    left, right = _negation_markers(source), _negation_markers(summary)
    if any(len(markers) > 1 for markers in (left, right)):
        return "uncertain"
    if Counter(left) != Counter(right):
        # A marker difference alone is not proof that the same predicate flipped.
        # Require strong lexical overlap after removing negation words; otherwise
        # the marker may belong to a different clause and the safe result is review.
        def predicate_terms(text: str) -> set[str]:
            without_negation = _NEGATION_RE.sub(" ", text.casefold())
            return {term for term in re.findall(r"[a-z][a-z'-]{2,}", without_negation)
                    if term not in {"only", "just", "never"}}
        source_terms, summary_terms = predicate_terms(source), predicate_terms(summary)
        overlap = source_terms & summary_terms
        union = source_terms | summary_terms
        # “Without” often marks methods/procedures rather than proposition
        # polarity. A changed method phrase should be reviewed, not asserted to
        # reverse the claim. “Insufficient” and “absence” also need scope review.
        if any(token in {"without", "insufficient", "absence", "absent"} for token in (*left, *right)):
            return "uncertain"
        if not union or len(union - overlap) > 1 or len(overlap) < 2:
            return "uncertain"
        return "mismatch"
    if not left:
        return "match"
    def scope(text: str) -> tuple[str, ...]:
        normalized = text.casefold()
        signatures = []
        for match in _NEGATION_RE.finditer(text):
            token = match.group(0).casefold()
            tail = normalized[match.end():].lstrip()
            if token == "not" and (tail.startswith("only ") or tail.startswith("just ")):
                continue
            start = max(normalized.rfind(mark, 0, match.start()) for mark in (".", ";", ","))
            end_candidates = [pos for mark in (".", ";", ",") if (pos := normalized.find(mark, match.end())) >= 0]
            end = min(end_candidates) if end_candidates else len(text)
            clause = normalized[start + 1:end]
            before = re.findall(r"[a-z0-9'-]+", clause[:match.start() - start - 1])[-3:]
            after = re.findall(r"[a-z0-9'-]+", clause[match.end() - start - 1:])[:4]
            signatures.append(" ".join((*before, token, *after)))
        return tuple(signatures)
    return "match" if scope(source) == scope(summary) else "uncertain"


def normalize_sentence(text: str) -> str:
    return _SPACE_RE.sub(" ", _TOKEN_RE.sub(" ", text.casefold())).strip()


def normalized_pair_key(row: dict[str, str]) -> tuple[str, str]:
    """Canonical key intentionally treats a reversed pair as a duplicate."""
    return tuple(sorted((normalize_sentence(_value(row, SENTENCE_FIELDS[0])),
                         normalize_sentence(_value(row, SENTENCE_FIELDS[1])))))


def stable_example_id(source_document_id: str, source_sentence: str, summary_sentence: str) -> str:
    """Generate a deterministic ID from document identity and exact pair text."""
    raw = "\0".join((source_document_id.strip(), source_sentence.strip(), summary_sentence.strip()))
    return "RW-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16].upper()


def license_is_reusable(license_value: str, license_url: str,
                        allowlist: Iterable[str] = LICENSE_ALLOWLIST) -> bool:
    """Fail closed unless recorded license text/URL matches the explicit allowlist."""
    text = f"{license_value} {license_url}".casefold()
    return any(marker.casefold() in text for marker in allowlist)


def read_real_world_csv(path: str | Path, *, fields: tuple[str, ...] = REAL_WORLD_FIELDS) -> list[dict[str, str]]:
    file = Path(path)
    try:
        with file.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise RealWorldDatasetError(f"{file} has no CSV header.")
            missing = [field for field in fields if field not in reader.fieldnames]
            if missing:
                raise RealWorldDatasetError("Missing required CSV columns: " + ", ".join(missing))
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RealWorldDatasetError(f"Could not read {file}: {exc}") from exc


def write_real_world_csv(path: str | Path, rows: Iterable[dict[str, object]], *,
                         fields: tuple[str, ...] = REAL_WORLD_FIELDS) -> None:
    """Write a versioned CSV schema without silently dropping source metadata."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
    except (OSError, csv.Error) as exc:
        raise RealWorldDatasetError(f"Could not write {file}: {exc}") from exc


def _annotation_issues(row: dict[str, str], label: str, status: str) -> list[str]:
    issues: list[str] = []
    if status == "not_annotated":
        return issues
    if status not in ANNOTATION_STATUSES:
        return ["invalid_annotation_status"]
    if status == "error":
        return ["annotation_error"]
    if label != "EXCLUDE" and (status == "requires_review" or _value(row, "requires_review").lower() in {"true", "1", "yes"}):
        issues.append("llm_requires_review")
        issues.extend(reason for reason in _value(row, "review_reasons").split(";") if reason)
    comparable = _value(row, "proposition_comparable").lower()
    if comparable not in {"true", "false", "uncertain"}:
        issues.append("invalid_proposition_comparable")
    if label and label not in LABELS:
        issues.append("invalid_label")
    if _value(row, "annotation_version") in STRUCTURED_ANNOTATION_VERSIONS:
        proposition_status = _value(row, "comparability_status").lower()
        if proposition_status not in {"comparable", "non_comparable", "uncertain"}:
            issues.append("invalid_comparability_status")
        expected_comparable = {"comparable": "true", "non_comparable": "false", "uncertain": "uncertain"}.get(proposition_status)
        if expected_comparable and comparable != expected_comparable:
            issues.append("comparability_status_field_inconsistent")
        if proposition_status == "comparable":
            for field, sentence_field in (("source_proposition", "source_sentence"),
                                          ("summary_proposition", "summary_sentence")):
                if not _exact_quote(_value(row, sentence_field), _value(row, field)):
                    issues.append(f"nonexact_or_missing_{field}")
            if label == "EXCLUDE":
                issues.append("exclude_label_marked_comparable")
        elif status == "complete" or label in MODEL_LABELS:
            issues.append("certainty_requires_comparable_proposition")
    if status == "complete" and (not label or (label != "EXCLUDE" and comparable != "true")
                                  or (label == "EXCLUDE" and comparable == "true")):
        issues.append("incomplete_or_noncomparable_annotation")
    if status in {"complete", "requires_review"} and label in MODEL_LABELS:
        source_level = _value(row, "source_certainty").upper()
        summary_level = _value(row, "summary_certainty").upper()
        level_order = {"POSSIBLE": 1, "PROBABLE": 2, "ASSERTED": 3}
        for field in ("source_certainty", "summary_certainty"):
            if _value(row, field).upper() not in CERTAINTY_LEVELS:
                issues.append(f"invalid_{field}")
        direction = _value(row, "certainty_direction").upper()
        if direction not in MODEL_LABELS:
            issues.append("invalid_certainty_direction")
        elif label != direction:
            issues.append("label_direction_mismatch")
        if source_level in level_order and summary_level in level_order:
            expected = ("PRESERVED" if level_order[source_level] == level_order[summary_level]
                        else "HARDENED" if level_order[summary_level] > level_order[source_level]
                        else "SOFTENED")
            if label != expected or direction != expected:
                issues.append("certainty_label_direction_inconsistent")
        elif _value(row, "annotation_version") in STRUCTURED_ANNOTATION_VERSIONS:
            issues.append("unclear_certainty_requires_review")
        if not _value(row, "explanation"):
            issues.append("missing_explanation")
        for field in ("source_evidence", "summary_evidence"):
            try:
                values = json.loads(_value(row, field))
            except json.JSONDecodeError:
                issues.append(f"invalid_{field}_json")
            else:
                if _value(row, "annotation_version") in STRUCTURED_ANNOTATION_VERSIONS:
                    sentence_field = "source_sentence" if field == "source_evidence" else "summary_sentence"
                    quote = str(values.get("quote", "")) if isinstance(values, dict) else ""
                    exact_quote = _SPACE_RE.sub(" ", quote.strip()).casefold() in _SPACE_RE.sub(
                        " ", _value(row, sentence_field)
                    ).casefold()
                    evidence_type = str(values.get("evidence_type", "")).strip() if isinstance(values, dict) else ""
                    interpretation = str(values.get("interpretation", "")).strip() if isinstance(values, dict) else ""
                    if not quote.strip() or not exact_quote:
                        issues.append(f"nonexact_or_missing_{field}")
                    if not evidence_type or not interpretation:
                        issues.append(f"incomplete_{field}")
                    elif evidence_type not in {"explicit_uncertainty", "probability", "certainty", "asserted_predicate", "contextual"}:
                        issues.append(f"invalid_{field}_type")
                elif not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                    issues.append(f"invalid_{field}_schema")
        for field in QUALITY_FLAGS:
            flag = _value(row, field).lower()
            if flag not in {"true", "false", "uncertain"}:
                issues.append(f"invalid_{field}")
            elif flag != "true":
                issues.append(f"quality_{field}_{flag}")
        if _value(row, "annotation_method") != "llm_assisted":
            issues.append("annotation_method_must_be_llm_assisted")
        if not _value(row, "model_name") or not _value(row, "model_version"):
            issues.append("missing_model_provenance")
    if label == "EXCLUDE" and status == "complete" and comparable == "true":
        issues.append("exclude_label_marked_comparable")
    return issues


def validate_real_world_rows(
    rows: Iterable[dict[str, str]], *,
    training_records: Iterable[dict[str, str]] = (),
    allowlist: Iterable[str] = LICENSE_ALLOWLIST,
) -> RealWorldValidation:
    """Validate provenance, licensing, duplicates, annotation state, and QC flags.

    This machine-annotation gate is not human review and is intentionally separate
    from ``src.dataset.pipeline.training_rows``.
    """
    materialized = [dict(row) for row in rows]
    training_pairs = {
        normalized_pair_key(row) for row in training_records
        if _value(row, "source_sentence") and _value(row, "summary_sentence")
    }
    issues: dict[int, list[str]] = defaultdict(list)
    ids: dict[str, list[int]] = defaultdict(list)
    pairs: dict[tuple[str, str], list[int]] = defaultdict(list)

    for index, row in enumerate(materialized):
        example_id = _value(row, "example_id")
        if not example_id:
            issues[index].append("missing_example_id")
        else:
            ids[example_id.casefold()].append(index)
        for field in PROVENANCE_REQUIRED:
            if not _value(row, field):
                issues[index].append(f"missing_{field}")
        for field in SENTENCE_FIELDS:
            if not _value(row, field):
                issues[index].append(f"missing_{field}")
        if _value(row, "source_document_id") and _value(row, "source_document_id") != _value(row, "source_group_id"):
            issues[index].append("source_group_id_must_match_document")
        license_url = _value(row, "license_url")
        license_text = _value(row, "license")
        if license_text and license_url and not license_is_reusable(license_text, license_url, allowlist):
            issues[index].append("license_not_in_reuse_allowlist")
        if _value(row, "source_sentence") and _value(row, "summary_sentence"):
            pair = normalized_pair_key(row)
            pairs[pair].append(index)
            if pair in training_pairs:
                issues[index].append("synthetic_training_collision")
        label = _value(row, "label").upper()
        status = _value(row, "annotation_status").lower() or "not_annotated"
        if label and label not in LABELS:
            issues[index].append("invalid_label")
        issues[index].extend(_annotation_issues(row, label, status))

        # Only run claim-level checks when Stage A supplied a positive, exact
        # proposition comparison. Similarity or a failed check cannot establish
        # equivalence; uncertain/non-comparable pairs remain reviewable on that
        # basis and are not subjected to sentence-wide number/negation matching.
        source, summary = _value(row, "source_sentence"), _value(row, "summary_sentence")
        stage_a_comparable = (
            _value(row, "annotation_version") in STRUCTURED_ANNOTATION_VERSIONS
            and _value(row, "comparability_status").lower() == "comparable"
            and _value(row, "proposition_comparable").lower() == "true"
            and _exact_quote(source, _value(row, "source_proposition"))
            and _exact_quote(summary, _value(row, "summary_proposition"))
        )
        if source and summary and (not _value(row, "annotation_version") or stage_a_comparable):
            number_status = compare_number_mentions(source, summary)
            date_status = compare_calendar_dates(source, summary)
            negation_status = compare_negation(source, summary)
            if number_status == "mismatch":
                issues[index].append("detected_number_difference")
            elif number_status == "uncertain":
                issues[index].append("number_comparison_uncertain")
            if date_status == "mismatch":
                issues[index].append("detected_date_difference")
            if negation_status == "mismatch":
                issues[index].append("detected_negation_difference")
            elif negation_status == "uncertain":
                issues[index].append("negation_comparison_uncertain")

    for indices in ids.values():
        if len(indices) > 1:
            for index in indices:
                issues[index].append("duplicate_example_id")
    for indices in pairs.values():
        if len(indices) > 1:
            for index in indices:
                issues[index].append("duplicate_or_reversed_pair")

    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    review_required: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    reason_counts: Counter[str] = Counter()
    row_reasons: dict[str, tuple[str, ...]] = {}
    for index, row in enumerate(materialized):
        row_id = _value(row, "example_id") or f"row_{index + 1}"
        label = _value(row, "label").upper()
        row_issues = tuple(dict.fromkeys(issues[index]))
        if row_issues:
            # LLM uncertainty and mismatches stay reviewable; broken provenance,
            # IDs, duplicates, disallowed reuse, and collisions are rejected.
            review_tags = ("llm_requires_review", "quality_subject_match_", "quality_negation_match_",
                           "quality_numbers_match_", "quality_dates_match_",
                           "detected_number_difference", "detected_date_difference",
                           "detected_negation_difference", "incomplete_or_noncomparable_annotation",
                           "certainty_label_direction_inconsistent", "label_direction_mismatch",
                           "unclear_certainty_requires_review", "nonexact_or_missing_", "incomplete_",
                           "invalid_source_evidence", "invalid_summary_evidence", "missing_explanation",
                           "certainty_requires_comparable_proposition", "number_comparison_uncertain",
                           "negation_comparison_uncertain", "annotation_error", "invalid_",
                           "comparability_status_field_inconsistent", "exclude_label_marked_comparable")
            exclusion_conflicts = ("comparability_status_field_inconsistent", "exclude_label_marked_comparable")
            if label == "EXCLUDE" and any(reason.startswith(exclusion_conflicts) for reason in row_issues):
                review_required.append(row)
            elif label == "EXCLUDE" and all(reason.startswith(review_tags) for reason in row_issues):
                excluded.append(row)
                reason_counts["llm_exclude"] += 1
                row_reasons[row_id] = row_issues
            elif all(reason.startswith(review_tags) for reason in row_issues):
                review_required.append(row)
            else:
                rejected.append(row)
            row_reasons[row_id] = row_issues
            reason_counts.update(row_issues)
            continue
        label = _value(row, "label").upper()
        status = _value(row, "annotation_status").lower() or "not_annotated"
        if label == "EXCLUDE":
            excluded.append(row)
            reason_counts["llm_exclude"] += 1
            row_reasons[row_id] = ("llm_exclude",)
        elif status == "not_annotated" or not label:
            review_required.append(row)
            reason_counts["not_yet_annotated_or_incomplete"] += 1
            row_reasons[row_id] = ("not_yet_annotated_or_incomplete",)
        elif status == "requires_review" or _value(row, "requires_review").lower() in {"true", "1", "yes"}:
            review_required.append(row)
            reason_counts["llm_requires_review"] += 1
            row_reasons[row_id] = ("llm_requires_review",)
        elif status == "complete" and label in MODEL_LABELS:
            eligible.append(row)
        else:
            review_required.append(row)
            reason_counts["not_yet_annotated_or_incomplete"] += 1
            row_reasons[row_id] = ("not_yet_annotated_or_incomplete",)
    return RealWorldValidation(
        eligible, excluded, review_required, rejected,
        dict(sorted(reason_counts.items())), row_reasons,
    )


def assert_group_disjoint(partitions: dict[str, Iterable[dict[str, str]]]) -> None:
    """Raise if any source document occurs in more than one partition."""
    groups = {
        name: {(_value(row, "source_document_id") or _value(row, "source_group_id")) for row in rows}
        for name, rows in partitions.items()
    }
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            overlap = groups[left] & groups[right]
            overlap.discard("")
            if overlap:
                raise RealWorldDatasetError(
                    f"Source-document leakage between {left} and {right}: {sorted(overlap)}"
                )
