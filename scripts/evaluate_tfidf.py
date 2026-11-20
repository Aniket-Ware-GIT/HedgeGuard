"""Grouped cross-validation for TF-IDF/Logistic Regression and the rule baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.pipeline import read_csv, training_rows  # noqa: E402
from src.models.rule_based import LABELS as RULE_LABELS, predict as rule_predict  # noqa: E402
from src.models.tfidf_model import build_tfidf_logistic_pipeline, pair_frame  # noqa: E402

LABELS = tuple(RULE_LABELS)
REQUIRED_COLUMNS = ("pair_id", "source_group_id", "source_sentence", "summary_sentence", "label")


def _metrics(truth: list[str], predictions: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(LABELS), zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(np.mean(f1)),
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]),
                   "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=list(LABELS)).tolist(),
    }


def evaluate_records(records: Iterable[dict[str, str]], *, n_splits: int = 3,
                     random_state: int = 42) -> dict:
    """Evaluate both baselines on identical grouped out-of-fold examples.

    TF-IDF remains a Pipeline step and is therefore fitted only on each fold's
    training rows. The reviewed-dataset eligibility gate is applied before CV.
    """
    records = list(records)
    if not records:
        raise ValueError("Evaluation dataset is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in records[0]]
    if missing:
        raise ValueError("Training dataset is missing required columns: " + ", ".join(missing))
    try:
        eligible = training_rows(records)
    except ValueError as exc:
        raise ValueError(f"Training eligibility validation failed: {exc}") from exc
    if len(eligible) != len(records):
        raise ValueError("Input must contain only eligible approved or relabeled records; EXCLUDE rows are not accepted.")

    y = np.asarray([str(row["label"]).strip().upper() for row in eligible], dtype=object)
    groups = np.asarray([str(row["source_group_id"]).strip() for row in eligible], dtype=object)
    if any(not group for group in groups):
        raise ValueError("Every record must have a non-empty source_group_id.")
    if n_splits < 2 or n_splits > len(set(groups)):
        raise ValueError("n_splits must be at least 2 and no greater than the number of source groups.")
    X = pair_frame(eligible)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    truth_oof = np.empty(len(eligible), dtype=object)
    tfidf_oof = np.empty(len(eligible), dtype=object)
    rule_oof = np.empty(len(eligible), dtype=object)
    seen = np.zeros(len(eligible), dtype=int)
    folds: list[dict] = []

    for fold_number, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), 1):
        train_groups = set(groups[train_index])
        validation_groups = set(groups[validation_index])
        overlap = train_groups.intersection(validation_groups)
        if overlap:
            raise RuntimeError(f"Fold {fold_number} has source_group_id leakage: {sorted(overlap)}")

        # New, unfitted pipeline each fold: both vocabularies are fit on train only.
        estimator = build_tfidf_logistic_pipeline(random_state=random_state)
        estimator.fit(X.iloc[train_index], y[train_index])
        fold_tfidf = estimator.predict(X.iloc[validation_index])
        fold_rule = [
            rule_predict(eligible[i]["source_sentence"], eligible[i]["summary_sentence"]).label
            for i in validation_index
        ]

        truth_oof[validation_index] = y[validation_index]
        tfidf_oof[validation_index] = fold_tfidf
        rule_oof[validation_index] = fold_rule
        seen[validation_index] += 1

        fold_truth = list(y[validation_index])
        folds.append({
            "fold": fold_number,
            "train_size": int(len(train_index)),
            "validation_size": int(len(validation_index)),
            "train_group_count": len(train_groups),
            "validation_group_count": len(validation_groups),
            "train_group_ids": sorted(train_groups),
            "validation_group_ids": sorted(validation_groups),
            "group_overlap": sorted(overlap),
            "tfidf_logistic_regression": _metrics(fold_truth, list(fold_tfidf)),
            "rule_based": _metrics(fold_truth, list(fold_rule)),
        })

    if not np.all(seen == 1):
        raise RuntimeError("Cross-validation must predict every example exactly once.")
    if any(value is None for value in tfidf_oof) or any(value is None for value in rule_oof):
        raise RuntimeError("Cross-validation left an example without an out-of-fold prediction.")

    predictions = []
    for i, row in enumerate(eligible):
        predictions.append({
            "pair_id": row["pair_id"],
            "source_group_id": row["source_group_id"],
            "ground_truth": str(truth_oof[i]),
            "tfidf_prediction": str(tfidf_oof[i]),
            "rule_prediction": str(rule_oof[i]),
            "tfidf_correct": bool(tfidf_oof[i] == truth_oof[i]),
            "rule_correct": bool(rule_oof[i] == truth_oof[i]),
        })

    return {
        "labels": list(LABELS),
        "n_splits": n_splits,
        "random_state": random_state,
        "total": len(eligible),
        "folds": folds,
        "overall": {
            "tfidf_logistic_regression": _metrics(list(truth_oof), list(tfidf_oof)),
            "rule_based": _metrics(list(truth_oof), list(rule_oof)),
        },
        "predictions": predictions,
    }


def _print_metrics(name: str, metrics: dict, labels: list[str]) -> None:
    print(f"\n{name}")
    print(f"Accuracy: {metrics['accuracy']:.3f}; Macro F1: {metrics['macro_f1']:.3f}")
    print("Class       Precision  Recall  F1      Support")
    for label in labels:
        item = metrics["per_class"][label]
        print(f"{label:<11} {item['precision']:<10.3f} {item['recall']:<7.3f} "
              f"{item['f1']:<7.3f} {item['support']}")
    print("Confusion matrix (rows=ground truth, columns=prediction):")
    print("actual\\pred  " + "  ".join(f"{label:>10}" for label in labels))
    for label, values in zip(labels, metrics["confusion_matrix"]):
        print(f"{label:<12} " + "  ".join(f"{value:>10}" for value in values))


def print_report(result: dict) -> None:
    print("Grouped out-of-fold baseline comparison")
    print(f"Examples: {result['total']}; folds: {result['n_splits']}; seed: {result['random_state']}")
    print("Each row is validated exactly once; TF-IDF is fit inside each fold's training pipeline.")
    for fold in result["folds"]:
        print(f"\nFold {fold['fold']}: train={fold['train_size']} / validation={fold['validation_size']}; "
              f"groups train={fold['train_group_count']} / validation={fold['validation_group_count']}; "
              f"overlap={fold['group_overlap']}")
        for key, title in (("tfidf_logistic_regression", "TF-IDF + Logistic Regression"),
                           ("rule_based", "Rule baseline")):
            metrics = fold[key]
            print(f"  {title}: accuracy={metrics['accuracy']:.3f}, macro_F1={metrics['macro_f1']:.3f}")
    _print_metrics("TF-IDF + Logistic Regression (overall OOF)",
                   result["overall"]["tfidf_logistic_regression"], result["labels"])
    _print_metrics("Rule baseline (same overall OOF examples)",
                   result["overall"]["rule_based"], result["labels"])
    print("\nIndividual out-of-fold predictions:")
    for row in result["predictions"]:
        print(f"{row['pair_id']} [{row['source_group_id']}] true={row['ground_truth']} | "
              f"TF-IDF={row['tfidf_prediction']} ({'correct' if row['tfidf_correct'] else 'wrong'}) | "
              f"Rule={row['rule_prediction']} ({'correct' if row['rule_correct'] else 'wrong'})")
    print("\nCaution: 24 synthetic examples are insufficient for reliable generalization claims. "
          "These cross-validation results are exploratory and highly sensitive to individual examples.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/training_dataset.csv")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = evaluate_records(read_csv(args.data), n_splits=args.folds, random_state=args.seed)
        print_report(result)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TF-IDF evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
