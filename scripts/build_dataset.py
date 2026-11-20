"""Build an experimental candidate file from explicitly curated sentence pairs."""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.pipeline import (  # noqa: E402
    DatasetError, LABELS, REQUIRED_SEED, class_counts, generate, read_csv,
    validate_rows, write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", default="data/raw/seed_examples.csv")
    parser.add_argument("--output", default="data/processed/experimental_candidates.csv")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing the selected output file")
    args = parser.parse_args()
    output = Path(args.output)
    try:
        seeds = read_csv(args.seed)
        if not seeds:
            raise DatasetError("Seed CSV contains no examples.")
        missing = [field for field in REQUIRED_SEED if field not in seeds[0]]
        if missing:
            raise DatasetError("Paired seed CSV must include columns: " + ", ".join(REQUIRED_SEED))
        if output.exists() and not args.overwrite:
            raise DatasetError(f"Output already exists: {output}. Choose another --output or pass --overwrite.")
        rows, duplicate_count = generate(seeds, return_duplicate_count=True)
        errors = validate_rows(rows)
        if errors:
            raise DatasetError("Candidate validation failed:\n- " + "\n- ".join(errors[:20]))
        write_csv(rows, output)
        # Re-open the generated CSV to verify its serialization and row shape.
        round_trip = read_csv(output)
        round_trip_errors = validate_rows(round_trip)
        if round_trip_errors:
            raise DatasetError("Generated CSV failed round-trip validation: " + "; ".join(round_trip_errors[:10]))
        counts = class_counts(rows)
        print(f"Total candidates: {len(rows)}")
        for label in LABELS:
            print(f"{label}: {counts[label]}")
        print(f"Duplicate pairs omitted: {duplicate_count}")
        print(f"Candidates requiring manual review: {len(rows)}")
        missing_classes = [label for label in LABELS[:3] if counts[label] == 0]
        if missing_classes:
            print("Class shortfall: seed does not provide " + ", ".join(missing_classes))
        print(f"Experimental candidate CSV: {output}")
        print("No dataset split was performed. EXCLUDE rows remain annotation-only and must not enter model training.")
        return 0
    except DatasetError as exc:
        print(f"Dataset build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
