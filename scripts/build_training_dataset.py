"""Build a training-ready CSV from manually reviewed records; does not split or train."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.pipeline import (  # noqa: E402
    DatasetError, MODEL_LABELS, class_counts, read_csv, training_rows, write_csv,
)

REQUIRED_COLUMNS = ("pair_id", "source_group_id", "source_sentence", "summary_sentence", "label")


def build_training_dataset(reviewed_records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Return only validated, manually approved or relabeled model examples."""
    records = list(reviewed_records)
    if not records:
        raise DatasetError("Reviewed dataset is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in records[0]]
    if missing:
        raise DatasetError("Reviewed dataset is missing required columns: " + ", ".join(missing))
    return training_rows(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/processed/reviewed_candidates.csv")
    parser.add_argument("--output", default="data/processed/training_dataset.csv")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output file")
    args = parser.parse_args()
    output = Path(args.output)
    try:
        reviewed = read_csv(args.input)
        if output.exists() and not args.overwrite:
            raise DatasetError(f"Output already exists: {output}; choose another path or pass --overwrite.")
        eligible = build_training_dataset(reviewed)
        # Retain the complete reviewed-record schema and annotations for traceability.
        fields = tuple(reviewed[0].keys())
        write_csv(eligible, output, fields=fields)
        counts = class_counts(eligible)
        print(f"Total eligible records: {len(eligible)}")
        print("Class distribution:")
        for label in MODEL_LABELS:
            print(f"  {label}: {counts[label]}")
        print(f"Saved training dataset: {output}")
        print("No train/test split or model training was performed.")
        return 0
    except DatasetError as exc:
        print(f"Training dataset build failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
