"""Validate and summarize the independent evaluation CSV without scoring models."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.independent_evaluation import read_independent_csv, validate_independent_records  # noqa: E402
from src.dataset.pipeline import read_csv as read_training_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/evaluation/independent_evaluation_template.csv")
    parser.add_argument("--training", default="data/processed/training_dataset.csv")
    args = parser.parse_args()
    try:
        rows = read_independent_csv(args.data)
        training = read_training_csv(args.training)
        result = validate_independent_records(rows, training_records=training)
    except (OSError, ValueError) as exc:
        print(f"Independent dataset validation failed: {exc}", file=sys.stderr)
        return 2
    print(f"Records: {len(rows)}; eligible: {len(result.eligible)}; "
          f"human-excluded: {len(result.excluded)}; rejected/incomplete: {len(result.rejected)}")
    if result.reasons:
        print("Counts by reason (a row can contribute to multiple reasons):")
        for reason, count in result.reasons.items():
            print(f"  {reason}: {count}")
    else:
        print("No excluded or rejected records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
