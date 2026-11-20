"""Exploratory evaluation of the rule baseline against reviewed CSV labels.

This evaluates the supplied rows as-is. It does not fit a model or split data;
results on the tiny synthetic set are not evidence of real-world generalization.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)

from src.dataset.pipeline import read_csv  # noqa: E402
from src.models.rule_based import LABELS, predict  # noqa: E402

REQUIRED_COLUMNS = ("pair_id", "source_sentence", "summary_sentence", "label")


def evaluate_records(records: Iterable[dict[str, str]]) -> dict:
    """Predict every input pair and return fixed-label metrics and inspection rows."""
    records = list(records)
    if not records:
        raise ValueError("Evaluation dataset is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in records[0]]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    truth: list[str] = []
    predicted: list[str] = []
    details: list[dict[str, str | int | bool]] = []
    for index, row in enumerate(records, 1):
        label = str(row.get("label", "")).strip().upper()
        if label not in LABELS:
            raise ValueError(f"Row {index}: unsupported ground-truth label {label!r}.")
        source = str(row.get("source_sentence", ""))
        summary = str(row.get("summary_sentence", ""))
        if not source.strip() or not summary.strip():
            raise ValueError(f"Row {index}: source_sentence and summary_sentence must be non-empty.")

        result = predict(source, summary)
        truth.append(label)
        predicted.append(result.label)
        details.append({
            "pair_id": str(row.get("pair_id", f"row-{index}")),
            "ground_truth": label,
            "prediction": result.label,
            "correct": result.label == label,
            "source_strength": result.source_strength,
            "summary_strength": result.summary_strength,
            "source_sentence": source,
            "summary_sentence": summary,
            "explanation": result.explanation,
        })

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(LABELS), zero_division=0,
    )
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(LABELS)
    }
    return {
        "total": len(records),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(sum(f1) / len(LABELS)),
        "per_class": per_class,
        "labels": list(LABELS),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(LABELS)).tolist(),
        "predictions": details,
    }


def print_report(result: dict) -> None:
    """Print aggregate metrics, confusion matrix, and every pair's result."""
    print("Exploratory rule-based evaluation (no fitting or data split)")
    print(f"Examples: {result['total']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Macro F1: {result['macro_f1']:.3f}")
    print("\nPer-class metrics:")
    print("Class       Precision  Recall  F1      Support")
    for label, scores in result["per_class"].items():
        print(f"{label:<11} {scores['precision']:<10.3f} {scores['recall']:<7.3f} "
              f"{scores['f1']:<7.3f} {scores['support']}")

    labels = result["labels"]
    print("\nConfusion matrix (rows=ground truth, columns=prediction):")
    print("actual\\pred  " + "  ".join(f"{label:>10}" for label in labels))
    for label, values in zip(labels, result["confusion_matrix"]):
        print(f"{label:<12} " + "  ".join(f"{value:>10}" for value in values))

    print("\nIndividual predictions:")
    for row in result["predictions"]:
        state = "CORRECT" if row["correct"] else "INCORRECT"
        print(f"\n[{state}] {row['pair_id']}: true={row['ground_truth']}, "
              f"predicted={row['prediction']}, strengths="
              f"{row['source_strength']}/{row['summary_strength']}")
        print(f"  Source:  {row['source_sentence']}")
        print(f"  Summary: {row['summary_sentence']}")
        print(f"  Rule:    {row['explanation']}")

    print("\nCaution: this is an in-sample exploratory check on a very small synthetic dataset; "
          "it is not evidence of generalization to real-world summaries.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/training_dataset.csv")
    args = parser.parse_args()
    try:
        result = evaluate_records(read_csv(args.data))
        print_report(result)
        return 0
    except (OSError, ValueError) as exc:
        print(f"Rule-based evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
