"""Write the supplied AI-assisted recommendations into a review copy, pending human confirmation."""
import argparse
import csv
import shutil
from pathlib import Path

RECOMMENDATIONS = {
    "HG001": ("APPROVE", "PRESERVED", '"may" and "might" express similar possibility in this context.'),
    "HG002": ("APPROVE", "PRESERVED", '"likely to" and "will probably" communicate approximately the same degree of likelihood.'),
    "HG003": ("APPROVE", "PRESERVED", "Both sentences express the same definite factual assertion with different wording."),
    "HG004": ("APPROVE", "PRESERVED", '"could" and "may" both express possibility in this context.'),
    "HG005": ("NEEDS CLOSER HUMAN REVIEW", "PRESERVED", 'Both statements are qualified, but "suggest" and "indicate that ... probably" have potentially different evidential strength. Do not automatically approve this example.'),
    "HG006": ("APPROVE", "PRESERVED", '"failed" and "stopped working" communicate the same definite event.'),
    "HG007": ("APPROVE", "PRESERVED", '"appears to" and "seems to" express approximately the same inferential uncertainty.'),
    "HG008": ("APPROVE", "PRESERVED", '"might" and "could" both communicate possibility.'),
    "HG009": ("APPROVE", "HARDENED", 'The source expresses possibility through "may", while the summary states the result directly.'),
    "HG010": ("APPROVE", "HARDENED", "A probable causal explanation becomes a definite causal assertion."),
    "HG011": ("APPROVE", "HARDENED", '"could indicate" becomes a direct assertion through "indicate".'),
    "HG012": ("APPROVE", "HARDENED", '"appears to be damaged" becomes "is damaged".'),
    "HG013": ("APPROVE", "HARDENED", '"might improve" becomes a direct assertion.'),
    "HG014": ("APPROVE", "HARDENED", '"possibly related" becomes a definite statement of relation.'),
    "HG015": ("APPROVE", "HARDENED", "A probable medical effect becomes a definite assertion."),
    "HG016": ("APPROVE", "HARDENED", '"may indicate" becomes a direct assertion.'),
    "HG017": ("APPROVE", "SOFTENED", 'A definite statement becomes an inference through "appears to".'),
    "HG018": ("APPROVE", "SOFTENED", 'A definite past event becomes a possible event through "may have".'),
    "HG019": ("APPROVE", "SOFTENED", '"confirmed" becomes "suggests", weakening the assertion.'),
    "HG020": ("APPROVE", "SOFTENED", 'A direct numerical assertion becomes qualified through "probably".'),
    "HG021": ("APPROVE", "SOFTENED", "A definite detection becomes a possible detection."),
    "HG022": ("APPROVE", "SOFTENED", "A definite availability claim becomes an inference."),
    "HG023": ("APPROVE", "SOFTENED", '"found" becomes "seems to have found".'),
    "HG024": ("APPROVE", "SOFTENED", "A definite landing becomes a possible landing."),
    "HG025": ("EXCLUDE", "EXCLUDE", "The main entity changes from valve to pump."),
    "HG026": ("EXCLUDE", "EXCLUDE", "Negation changes between the original and summary."),
    "HG027": ("EXCLUDE", "EXCLUDE", "Quantifier scope changes from some users to all users."),
    "HG028": ("EXCLUDE", "EXCLUDE", "The date changes from June to July."),
    "HG029": ("EXCLUDE", "EXCLUDE", "The summary introduces an unsupported numerical claim."),
    "HG030": ("EXCLUDE", "EXCLUDE", "The summary introduces a specific cause that the source does not establish."),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/reviewed_candidates.csv")
    parser.add_argument("--output", default="data/processed/reviewed_candidates_ai_prereview.csv")
    args = parser.parse_args()
    source_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists():
        raise SystemExit(f"Output already exists: {output_path}; choose another --output")
    with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not reader.fieldnames:
            raise SystemExit("Review CSV has no header")
        fields = list(reader.fieldnames)
        rows = list(reader)
    ids = [row.get("pair_id", "") for row in rows]
    if len(rows) != 30 or len(ids) != len(set(ids)) or set(ids) != set(RECOMMENDATIONS):
        raise SystemExit("Expected exactly one row for each ID HG001 through HG030; file not changed")
    required = {"include_in_training", "review_status", "reviewer_id", "review_notes", "label", "original_label"}
    if not required.issubset(fields):
        raise SystemExit("Review CSV is missing required review/label columns; file not changed")
    for row in rows:
        pair_id = row["pair_id"]
        recommendation, recommended_label, rationale = RECOMMENDATIONS[pair_id]
        if row["label"].strip().upper() != row["original_label"].strip().upper():
            raise SystemExit(f"{pair_id} label differs from original_label; file not changed")
        row["include_in_training"] = "false"
        row["review_status"] = "requires_manual_review"
        row["reviewer_id"] = "AI_ASSISTED_PRE_REVIEW"
        row["review_notes"] = (
            f"AI-assisted recommendation: {recommendation}; recommended label: {recommended_label}. "
            f"Rationale: {rationale} Human confirmation is pending; do not use for training yet."
        )
    temp = output_path.with_name(output_path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        with temp.open("rb") as source, output_path.open("xb") as target:
            shutil.copyfileobj(source, target)
    finally:
        if temp.exists():
            temp.unlink()
    print(f"Wrote AI-assisted review notes for {len(rows)} rows to {output_path}; no labels were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
