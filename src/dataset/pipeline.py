"""Curated pair ingestion and conservative validation for HedgeGuard datasets.

This module deliberately does not invent sentence rewrites. The seed file already
contains curated source-summary pairs and labels; model-quality transformations
must be reviewed and added as explicit pairs by an annotator.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path
from typing import Iterable

LABELS = ("PRESERVED", "HARDENED", "SOFTENED", "EXCLUDE")
MODEL_LABELS = LABELS[:3]
FIELDS = (
    "pair_id", "source_group_id", "source_sentence", "summary_sentence", "label", "domain",
    "generation_method", "transformation_type", "is_synthetic", "annotation_notes",
    "source_stance", "summary_stance", "source_cue", "summary_cue", "pair_origin", "split",
    "include_in_training", "exclude_reason", "annotator_notes", "review_status",
)
REVIEW_FIELDS = FIELDS + ("original_label", "reviewer_id", "review_notes")
REVIEW_STATUSES = ("approved", "relabeled", "excluded", "requires_manual_review")
REQUIRED_SEED = ("source_sentence", "summary_sentence", "label")
PERMISSION_RE = re.compile(r"\b(?:may|could)\s+(?:access|enter|use|submit|view|request|apply|leave)\b", re.I)
PERMITTED_RE = re.compile(r"\b(?:is|are)\s+permitted\s+to\b", re.I)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
REPEATED_WORD_RE = re.compile(r"\b([a-z]+)\s+\1\b", re.I)


class DatasetError(ValueError):
    """Raised for malformed or unusable dataset inputs."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _id(*parts: str) -> str:
    return hashlib.sha1("\0".join(parts).encode("utf-8")).hexdigest()[:12]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV and report malformed/missing inputs clearly."""
    file = Path(path)
    if not file.is_file():
        raise DatasetError(f"CSV not found: {file}. Provide the curated paired seed file first.")
    try:
        with file.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not reader.fieldnames:
                raise DatasetError(f"CSV has no header: {file}")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise DatasetError(f"CSV contains duplicate column names: {file}")
            return [dict(row) for row in reader]
    except (OSError, csv.Error, UnicodeError) as exc:
        raise DatasetError(f"Could not read CSV {file}: {exc}") from exc


def validate_rows(rows: Iterable[dict[str, str]], *, allow_exclude: bool = True) -> list[str]:
    """Return structural/label errors. These checks cannot establish semantic equivalence."""
    errors: list[str] = []
    allowed = LABELS if allow_exclude else MODEL_LABELS
    for i, row in enumerate(rows, 1):
        for field in ("pair_id", "source_group_id", "source_sentence", "summary_sentence", "label"):
            if not str(row.get(field, "")).strip():
                errors.append(f"row {i}: missing {field}")
        source = str(row.get("source_sentence", "")).strip()
        summary = str(row.get("summary_sentence", "")).strip()
        label = str(row.get("label", "")).strip().upper()
        if label and label not in allowed:
            errors.append(f"row {i}: unsupported label {label!r}; expected {', '.join(allowed)}")
        for field, sentence in (("source_sentence", source), ("summary_sentence", summary)):
            if sentence and (len(WORD_RE.findall(sentence)) < 2 or REPEATED_WORD_RE.search(sentence)):
                errors.append(f"row {i}: obviously malformed {field}")
        if source and summary and _norm(source) == _norm(summary) and label in {"HARDENED", "SOFTENED"}:
            errors.append(f"row {i}: unchanged sentence cannot be labeled {label}")
        if str(row.get("is_synthetic", "")).strip().lower() not in {"true", "false", "1", "0"}:
            errors.append(f"row {i}: is_synthetic must be true/false")
    return errors


def _canonicalize(seed: dict[str, str], index: int) -> dict[str, str]:
    """Map the existing annotated seed schema into the canonical review schema."""
    source = str(seed.get("source_sentence", "")).strip()
    summary = str(seed.get("summary_sentence", "")).strip()
    label = str(seed.get("label", "")).strip().upper()
    group = str(seed.get("source_group_id", "")).strip() or _id(source)
    pair_id = str(seed.get("pair_id", "")).strip() or _id(group, source, summary, label)
    origin = str(seed.get("pair_origin", "curated_seed")).strip() or "curated_seed"
    notes = str(seed.get("annotator_notes", "")).strip()
    exclude_reason = str(seed.get("exclude_reason", "")).strip()
    # Explicitly keep permission modality out of the epistemic labels. Preserve
    # the source annotation in raw data; only the derived candidate is relabeled.
    if PERMISSION_RE.search(source) and (PERMITTED_RE.search(summary) or PERMISSION_RE.search(summary)):
        label = "EXCLUDE"
        exclude_reason = exclude_reason or "permission_modality"
        notes = "; ".join(x for x in (notes, "Permission use of may/could is not epistemic uncertainty.") if x)
    synthetic_val = str(seed.get("is_synthetic", "")).strip().lower()
    if synthetic_val not in {"true", "false", "1", "0"}:
        synthetic_val = "true" if "synthetic" in origin.lower() else "false"
    synthetic = "true" if synthetic_val in {"true", "1"} else "false"
    return {
        "pair_id": pair_id,
        "source_group_id": group,
        "source_sentence": source,
        "summary_sentence": summary,
        "label": label,
        "domain": str(seed.get("domain", "unspecified")).strip() or "unspecified",
        "generation_method": "curated_pair",
        "transformation_type": str(seed.get("transformation_type", "")).strip() or "annotated_pair",
        "is_synthetic": synthetic,
        "annotation_notes": notes or "Curated pair; manual review required before training.",
        "source_stance": str(seed.get("source_stance", "")).strip(),
        "summary_stance": str(seed.get("summary_stance", "")).strip(),
        "source_cue": str(seed.get("source_cue", "")).strip(),
        "summary_cue": str(seed.get("summary_cue", "")).strip(),
        "pair_origin": origin,
        "split": str(seed.get("split", "")).strip(),
        "include_in_training": str(seed.get("include_in_training", "")).strip(),
        "exclude_reason": exclude_reason,
        "annotator_notes": notes,
        "review_status": "requires_manual_review",
    }


def generate(seed_rows: Iterable[dict[str, str]], *, return_duplicate_count: bool = False):
    """Convert explicitly paired, labeled seed rows; do not auto-rewrite sentences.

    Duplicate source-summary pairs are omitted. Identical inputs always produce
    identical output. By default returns rows for backwards-compatible callers;
    set ``return_duplicate_count`` to receive ``(rows, duplicate_count)``.
    """
    out: list[dict[str, str]] = []
    seen: dict[tuple[str, str], str] = {}
    duplicates = 0
    for index, seed in enumerate(seed_rows, 1):
        row = _canonicalize(seed, index)
        key = (_norm(row["source_sentence"]), _norm(row["summary_sentence"]))
        if key in seen:
            duplicates += 1
            if seen[key] != row["label"]:
                raise DatasetError(f"Conflicting labels for duplicate pair at input row {index}.")
            continue
        seen[key] = row["label"]
        out.append(row)
    return (out, duplicates) if return_duplicate_count else out


def write_csv(rows: Iterable[dict[str, str]], path: str | Path,
              fields: tuple[str, ...] = FIELDS) -> None:
    """Write canonical candidate rows in stable column order."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_review_rows(candidate_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Create a separate review queue, resetting every candidate to pending/excluded from training."""
    rows: list[dict[str, str]] = []
    candidates = list(candidate_rows)
    errors = validate_rows(candidates)
    if errors:
        raise DatasetError("Cannot prepare review queue:\n- " + "\n- ".join(errors[:20]))
    for candidate in candidates:
        row = dict(candidate)
        row["original_label"] = str(candidate["label"]).strip().upper()
        row["review_status"] = "requires_manual_review"
        row["include_in_training"] = "false"
        row["reviewer_id"] = ""
        row["review_notes"] = ""
        rows.append(row)
    return rows


def apply_review_decision(candidate: dict[str, str], status: str, reviewer_id: str,
                          *, label: str | None = None, notes: str = "") -> dict[str, str]:
    """Apply an explicit manual decision and derive training eligibility safely.

    ``relabeled`` means the reviewer changed the original label and approves the
    replacement. ``excluded`` sets the annotation label to EXCLUDE.
    """
    status = status.strip().lower()
    if status not in REVIEW_STATUSES:
        raise DatasetError(f"Unsupported review_status {status!r}")
    row = dict(candidate)
    original = str(row.get("original_label", row.get("label", ""))).strip().upper()
    chosen = str(label if label is not None else row.get("label", "")).strip().upper()
    if status in {"approved", "relabeled", "excluded"} and not reviewer_id.strip():
        raise DatasetError(f"reviewer_id is required for {status} decisions")
    if status == "approved" and chosen != original:
        raise DatasetError("approved rows must retain original_label; use relabeled for a label change")
    if status == "relabeled" and (chosen not in MODEL_LABELS or chosen == original):
        raise DatasetError("relabeled rows require a changed PRESERVED/HARDENED/SOFTENED label")
    if status == "approved" and chosen not in MODEL_LABELS:
        raise DatasetError("EXCLUDE cannot be approved for training; mark it excluded")
    if status == "excluded":
        chosen = "EXCLUDE"
        if not (notes.strip() or str(row.get("exclude_reason", "")).strip()):
            raise DatasetError("excluded rows require review_notes or exclude_reason")
    row.update({
        "label": chosen,
        "original_label": original,
        "review_status": status,
        "include_in_training": "true" if status in {"approved", "relabeled"} else "false",
        "reviewer_id": reviewer_id.strip() if status != "requires_manual_review" else "",
        "review_notes": notes.strip(),
    })
    errors = validate_review_rows([row])
    if errors:
        raise DatasetError("Invalid review decision: " + "; ".join(errors))
    return row


def validate_review_rows(rows: Iterable[dict[str, str]]) -> list[str]:
    """Validate review state, label transitions, reviewer attribution, and training gate."""
    materialized = list(rows)
    errors = validate_rows(materialized)
    for i, row in enumerate(materialized, 1):
        status = str(row.get("review_status", "")).strip().lower()
        label = str(row.get("label", "")).strip().upper()
        original = str(row.get("original_label", "")).strip().upper()
        include = str(row.get("include_in_training", "")).strip().lower()
        reviewer = str(row.get("reviewer_id", "")).strip()
        review_notes = str(row.get("review_notes", "")).strip()
        exclude_reason = str(row.get("exclude_reason", "")).strip()
        if status not in REVIEW_STATUSES:
            errors.append(f"row {i}: invalid review_status {status!r}")
        if original not in LABELS:
            errors.append(f"row {i}: invalid or missing original_label {original!r}")
        if include not in {"true", "false"}:
            errors.append(f"row {i}: include_in_training must be true or false")
        if status == "requires_manual_review" and include != "false":
            errors.append(f"row {i}: unreviewed row cannot be included in training")
        if status == "excluded":
            if include != "false":
                errors.append(f"row {i}: excluded row cannot be included in training")
            if label != "EXCLUDE":
                errors.append(f"row {i}: excluded status requires label EXCLUDE")
            if not reviewer:
                errors.append(f"row {i}: reviewer_id required for excluded row")
            if not (review_notes or exclude_reason):
                errors.append(f"row {i}: excluded row requires a reason")
        if status == "approved":
            if label not in MODEL_LABELS or label != original:
                errors.append(f"row {i}: approved row must retain an original model label")
            if include != "true":
                errors.append(f"row {i}: approved row must explicitly be enabled for training")
            if not reviewer:
                errors.append(f"row {i}: reviewer_id required for approved row")
        if status == "relabeled":
            if label not in MODEL_LABELS or label == original:
                errors.append(f"row {i}: relabeled row must change to a model label")
            if include != "true":
                errors.append(f"row {i}: relabeled row must explicitly be enabled for training")
            if not reviewer:
                errors.append(f"row {i}: reviewer_id required for relabeled row")
        if label == "EXCLUDE" and include == "true":
            errors.append(f"row {i}: EXCLUDE examples can never enter model training")
        if include == "true" and status not in {"approved", "relabeled"}:
            errors.append(f"row {i}: only approved or relabeled rows may enter training")
    return errors


def training_rows(reviewed_rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Return only manually approved rows; fail closed on any inconsistent review metadata."""
    rows = list(reviewed_rows)
    errors = validate_review_rows(rows)
    if errors:
        raise DatasetError("Reviewed dataset is not safe for training:\n- " + "\n- ".join(errors[:20]))
    return [row for row in rows
            if str(row.get("include_in_training", "")).strip().lower() == "true"
            and str(row.get("review_status", "")).strip().lower() in {"approved", "relabeled"}
            and str(row.get("label", "")).strip().upper() in MODEL_LABELS]


def class_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    """Return counts for every annotation label, including zero counts."""
    counts = {label: 0 for label in LABELS}
    for row in rows:
        label = str(row.get("label", "")).upper()
        if label in counts:
            counts[label] += 1
    return counts


def group_split(rows: list[dict[str, str]], seed: int = 42,
                ratios: tuple[float, float, float] = (.8, .1, .1)) -> dict[str, list[dict[str, str]]]:
    """Reproducibly split whole source groups; intentionally not called by the builder."""
    import random
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise DatasetError("ratios must be three positive values summing to 1")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("label", "")).upper() == "EXCLUDE":
            continue
        groups.setdefault(row["source_group_id"], []).append(row)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    train_end = round(n * ratios[0])
    val_end = train_end + round(n * ratios[1])
    buckets = (keys[:train_end], keys[train_end:val_end], keys[val_end:])
    return {name: [row for key in bucket for row in groups[key]]
            for name, bucket in zip(("train", "validation", "test"), buckets)}
