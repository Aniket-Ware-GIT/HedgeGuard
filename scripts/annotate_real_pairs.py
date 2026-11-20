"""Run Stage A proposition review or Stage B certainty review locally with Ollama."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import (REAL_WORLD_V2_FIELDS, STAGED_ANNOTATION_VERSION,
                                    read_real_world_csv, write_real_world_csv)
from src.dataset.real_world_annotation import (
    MODEL_NAME, STAGED_PROMPT_VERSION, OllamaAnnotationError,
    assess_certainty_shift, assess_proposition_comparability, stage_b_candidates, verify_ollama,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real_world"
PROPOSITION_FIELDS = (
    "example_id", "comparability_status", "proposition_comparable", "source_proposition",
    "summary_proposition", "subject_match", "negation_match", "numbers_match", "dates_match",
    "reason", "annotation_method", "model_name", "model_version", "prompt_version",
    "annotation_version", "annotation_stage", "annotation_status", "requires_review",
    "annotation_raw_json",
)
CERTAINTY_FIELDS = (
    "example_id", "label", "source_certainty", "summary_certainty", "certainty_direction",
    "source_evidence", "summary_evidence", "explanation", "requires_review", "review_reasons",
    "annotation_method", "model_name", "model_version", "prompt_version", "annotation_version",
    "annotation_stage", "annotation_status", "annotation_raw_json",
)


def _args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("comparability", "certainty"), required=True)
    p.add_argument("--candidates", type=Path, default=DATA / "candidate_pairs_v2.csv")
    p.add_argument("--propositions", type=Path, default=DATA / "proposition_assessments_v2.csv")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None, help="Maximum total rows in this stage output.")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _proposition_error(example_id: str, message: str, digest: str) -> dict[str, str]:
    row = {field: "" for field in PROPOSITION_FIELDS}
    row.update(example_id=example_id, comparability_status="uncertain", proposition_comparable="uncertain",
               subject_match="uncertain", negation_match="uncertain", numbers_match="uncertain",
               dates_match="uncertain", reason="annotation_error: " + message,
               annotation_method="llm_assisted", model_name=MODEL_NAME, model_version=digest,
               prompt_version=STAGED_PROMPT_VERSION, annotation_version=STAGED_ANNOTATION_VERSION,
               annotation_stage="proposition_comparability", annotation_status="requires_review",
               requires_review="true", annotation_raw_json=json.dumps({"error": message}))
    return row


def _certainty_error(example_id: str, message: str, digest: str) -> dict[str, str]:
    row = {field: "" for field in CERTAINTY_FIELDS}
    row.update(example_id=example_id, requires_review="true", review_reasons="annotation_error",
               annotation_method="llm_assisted", model_name=MODEL_NAME, model_version=digest,
               prompt_version=STAGED_PROMPT_VERSION, annotation_version=STAGED_ANNOTATION_VERSION,
               annotation_stage="certainty", annotation_status="requires_review",
               annotation_raw_json=json.dumps({"error": message}))
    return row


def _is_retryable_error(row: dict[str, str]) -> bool:
    """Identify recorded transport/parse failures for explicit --resume retry."""
    if row.get("annotation_status", "").casefold() == "error":
        return True
    try:
        payload = json.loads(row.get("annotation_raw_json", "{}"))
    except (TypeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and bool(payload.get("error"))


def main() -> int:
    args = _args()
    candidates_path = _path(args.candidates)
    output = _path(args.output) if args.output else DATA / (
        "proposition_assessments_v2.csv" if args.stage == "comparability" else "certainty_assessments_v2.csv"
    )
    fields = PROPOSITION_FIELDS if args.stage == "comparability" else CERTAINTY_FIELDS
    if output.exists() and not args.resume:
        print(f"Refusing to overwrite {output}; use --resume.", file=sys.stderr)
        return 2
    try:
        digest = verify_ollama()
    except OllamaAnnotationError as exc:
        print(f"Ollama preflight failed: {exc}", file=sys.stderr)
        return 2
    candidates = read_real_world_csv(candidates_path, fields=REAL_WORLD_V2_FIELDS)
    existing_rows = read_real_world_csv(output, fields=fields) if output.exists() else []
    existing = {row["example_id"]: row for row in existing_rows}
    retryable_ids = {example_id for example_id, row in existing.items() if _is_retryable_error(row)}
    if args.stage == "comparability":
        pending = [row for row in candidates
                   if row["example_id"] not in existing or row["example_id"] in retryable_ids]
    else:
        propositions = read_real_world_csv(_path(args.propositions), fields=PROPOSITION_FIELDS)
        prop_by_id = {row["example_id"]: row for row in propositions}
        pending = stage_b_candidates(candidates, propositions, set(existing) - retryable_ids)
    if args.limit is not None:
        successful_existing = len(existing) - len(retryable_ids)
        pending = pending[:max(0, args.limit - successful_existing)]
    errors = 0
    for index, row in enumerate(pending, 1):
        try:
            if args.stage == "comparability":
                result = assess_proposition_comparability(row["source_sentence"], row["summary_sentence"])
            else:
                result = assess_certainty_shift(row["source_sentence"], row["summary_sentence"],
                                                comparability_status="comparable")
        except (OllamaAnnotationError, ValueError) as exc:
            errors += 1
            result = (_proposition_error if args.stage == "comparability" else _certainty_error)(
                row["example_id"], str(exc), digest
            )
        result.update(example_id=row["example_id"], annotation_method="llm_assisted",
                      model_name=MODEL_NAME, model_version=digest, prompt_version=STAGED_PROMPT_VERSION,
                      annotation_version=STAGED_ANNOTATION_VERSION,
                      annotation_stage="proposition_comparability" if args.stage == "comparability" else "certainty")
        existing[row["example_id"]] = {field: str(result.get(field, "")) for field in fields}
        write_real_world_csv(output, existing.values(), fields=fields)
        print(f"{args.stage} {index}/{len(pending)}: {row['example_id']} [{result.get('comparability_status', result.get('label', 'REVIEW'))}]")
    print(f"Stage={args.stage}; rows={len(existing)}; errors={errors}; output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
