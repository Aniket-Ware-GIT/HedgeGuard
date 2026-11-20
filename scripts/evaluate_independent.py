"""Evaluate frozen HedgeGuard model configurations on an independent reviewed dataset.

The two learned estimators are fitted only on the existing approved training CSV.
Independent examples are passed only to ``predict`` and never to ``fit`` or tuning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.independent_evaluation import (  # noqa: E402
    MODEL_LABELS, read_independent_csv, validate_independent_records,
)
from src.dataset.pipeline import read_csv as read_training_csv, training_rows  # noqa: E402
from src.models.comparative_tfidf import build_comparative_tfidf_pipeline  # noqa: E402
from src.models.rule_based import predict as rule_predict  # noqa: E402
from src.models.tfidf_model import build_tfidf_logistic_pipeline, pair_frame  # noqa: E402

MODEL_KEYS = ("rule_based", "tfidf_logistic_regression", "comparative_tfidf")


def _metrics(truth: list[str], predictions: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(MODEL_LABELS), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(MODEL_LABELS)
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(MODEL_LABELS)).tolist(),
    }


def evaluate_records(
    training_records: Iterable[dict[str, str]], evaluation_records: Iterable[dict[str, str]],
) -> dict:
    """Fit on the reviewed training CSV only, then score eligible independent rows."""
    train = training_rows(training_records)
    evaluation = list(evaluation_records)
    validation = validate_independent_records(evaluation, training_records=train)
    if not validation.eligible:
        return {
            "labels": list(MODEL_LABELS), "training_count": len(train), "evaluation_count": 0,
            "input_count": len(evaluation), "human_excluded_count": len(validation.excluded),
            "rejected_count": len(validation.rejected), "reasons": validation.reasons,
            "row_issues": validation.row_issues, "metrics": {}, "predictions": [],
        }

    y_train = [str(row["label"]).strip().upper() for row in train]
    y_eval = [str(row["label"]).strip().upper() for row in validation.eligible]
    train_frame = pair_frame(train)
    eval_frame = pair_frame(validation.eligible)

    # These are the only fit calls. Their input is exclusively training_dataset.csv.
    tfidf = build_tfidf_logistic_pipeline(random_state=42)
    comparative = build_comparative_tfidf_pipeline(random_state=42)
    tfidf.fit(train_frame, y_train)
    comparative.fit(train_frame, y_train)
    predictions = {
        "rule_based": [rule_predict(row["source_sentence"], row["summary_sentence"]).label
                       for row in validation.eligible],
        "tfidf_logistic_regression": [str(label) for label in tfidf.predict(eval_frame)],
        "comparative_tfidf": [str(label) for label in comparative.predict(eval_frame)],
    }
    return {
        "labels": list(MODEL_LABELS),
        "training_count": len(train),
        "evaluation_count": len(validation.eligible),
        "input_count": len(evaluation),
        "human_excluded_count": len(validation.excluded),
        "rejected_count": len(validation.rejected),
        "reasons": validation.reasons,
        "row_issues": validation.row_issues,
        "metrics": {key: _metrics(y_eval, predictions[key]) for key in MODEL_KEYS},
        "predictions": [
            {"example_id": row["example_id"], "actual": y_eval[i],
             **{key: predictions[key][i] for key in MODEL_KEYS}}
            for i, row in enumerate(validation.eligible)
        ],
    }


def print_report(result: dict) -> None:
    print("Independent evaluation (one-time fit on training data; no evaluation fitting/tuning)")
    print(f"Training rows: {result['training_count']}; eligible evaluation rows: {result['evaluation_count']}; "
          f"human EXCLUDE: {result['human_excluded_count']}; rejected/incomplete: {result['rejected_count']}")
    if result["reasons"]:
        print("Exclusion/rejection reason counts (a row may have multiple validation issues):")
        for reason, count in result["reasons"].items():
            print(f"  {reason}: {count}")
    if not result["evaluation_count"]:
        print("No models were scored because there are no eligible reviewed records.")
        return
    print("\nIndividual predictions (eligible reviewed examples only):")
    print("Example ID                         Actual       Rule        TF-IDF      Comparative")
    for row in result["predictions"]:
        print(f"{row['example_id']:<34} {row['actual']:<12} {row['rule_based']:<11} "
              f"{row['tfidf_logistic_regression']:<11} {row['comparative_tfidf']}")
    for key in MODEL_KEYS:
        metrics = result["metrics"][key]
        print(f"\n{key}: accuracy={metrics['accuracy']:.3f}, macro F1={metrics['macro_f1']:.3f}")
        print("Class       Precision  Recall  F1      Support")
        for label in MODEL_LABELS:
            row = metrics["per_class"][label]
            print(f"{label:<11} {row['precision']:<10.3f} {row['recall']:<7.3f} "
                  f"{row['f1']:<7.3f} {row['support']}")
        print("Confusion matrix (rows=actual; columns=predicted; order=PRESERVED,HARDENED,SOFTENED):")
        for label, values in zip(MODEL_LABELS, metrics["confusion_matrix"]):
            print(f"{label:<11} " + "  ".join(str(value) for value in values))
    print("\nCross-validation scores on synthetic data are internal development estimates. These results "
          "are a separate independent evaluation and must not be merged with or presented as CV scores.")
    print("A small independent set gives unstable estimates; the models were fitted on only 24 synthetic examples.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/evaluation/independent_evaluation_template.csv")
    parser.add_argument("--training", default="data/processed/training_dataset.csv")
    args = parser.parse_args()
    try:
        evaluation = read_independent_csv(args.data)
        training = read_training_csv(args.training)
        result = evaluate_records(training, evaluation)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Independent evaluation failed: {exc}", file=sys.stderr)
        return 2
    print_report(result)
    return 0 if result["evaluation_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
