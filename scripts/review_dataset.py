"""Create or validate a separate manual-review copy of experimental candidates."""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.pipeline import (  # noqa: E402
    DatasetError, REVIEW_FIELDS, REVIEW_STATUSES, prepare_review_rows, read_csv,
    training_rows, validate_review_rows, write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="create a pending review copy")
    mode.add_argument("--validate", action="store_true", help="validate an edited review copy")
    parser.add_argument("--input", default="data/processed/experimental_candidates_v3.csv",
                        help="candidate CSV for --prepare")
    parser.add_argument("--reviewed", default="data/processed/reviewed_candidates.csv",
                        help="review CSV for --validate")
    parser.add_argument("--output", default="data/processed/reviewed_candidates.csv",
                        help="new review CSV path for --prepare")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing the selected review output")
    args = parser.parse_args()
    try:
        if args.prepare:
            output = Path(args.output)
            if output.exists() and not args.overwrite:
                raise DatasetError(f"Review output already exists: {output}; choose another path or pass --overwrite.")
            rows = prepare_review_rows(read_csv(args.input))
            errors = validate_review_rows(rows)
            if errors:
                raise DatasetError("Prepared review file is invalid: " + "; ".join(errors[:20]))
            write_csv(rows, output, fields=REVIEW_FIELDS)
            print(f"Created {len(rows)} review records at {output}")
            print("Every record is requires_manual_review with include_in_training=false.")
            print("Edit review_status, label when relabeling, reviewer_id, review_notes, and inclusion as documented.")
            return 0

        rows = read_csv(args.reviewed)
        errors = validate_review_rows(rows)
        if errors:
            raise DatasetError("Review validation failed:\n- " + "\n- ".join(errors[:30]))
        eligible = training_rows(rows)
        counts = Counter(str(row["review_status"]).lower() for row in rows)
        print(f"Review records: {len(rows)}")
        for status in REVIEW_STATUSES:
            print(f"{status}: {counts[status]}")
        print(f"Validated training-eligible reviewed records: {len(eligible)}")
        print("No model training or dataset splitting was performed.")
        return 0
    except DatasetError as exc:
        print(f"Review workflow failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
