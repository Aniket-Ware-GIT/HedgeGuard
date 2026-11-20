"""Compare comparative TF-IDF, existing directional TF-IDF, and rules on shared OOF folds."""
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
from src.models.comparative_tfidf import build_comparative_tfidf_pipeline  # noqa: E402
from src.models.rule_based import LABELS, predict as rule_predict  # noqa: E402
from src.models.tfidf_model import build_tfidf_logistic_pipeline, pair_frame  # noqa: E402

REQUIRED_COLUMNS = ("pair_id", "source_group_id", "source_sentence", "summary_sentence", "label")
MODEL_KEYS = ("comparative_tfidf", "tfidf_logistic_regression", "rule_based")


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
    """Evaluate all three systems in one identical fold loop."""
    records = list(records)
    if not records:
        raise ValueError("Evaluation dataset is empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in records[0]]
    if missing:
        raise ValueError("Training dataset is missing required columns: " + ", ".join(missing))
    eligible = training_rows(records)
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
    oof = {key: np.empty(len(eligible), dtype=object) for key in MODEL_KEYS}
    seen = np.zeros(len(eligible), dtype=int)
    folds: list[dict] = []

    for fold_number, (train_index, validation_index) in enumerate(splitter.split(X, y, groups), 1):
        train_groups = set(groups[train_index])
        validation_groups = set(groups[validation_index])
        overlap = train_groups & validation_groups
        if overlap:
            raise RuntimeError(f"Fold {fold_number} has source_group_id leakage: {sorted(overlap)}")

        # Each learned transformer is fitted only on this fold's training rows.
        old_model = build_tfidf_logistic_pipeline(random_state=random_state)
        comparative_model = build_comparative_tfidf_pipeline(random_state=random_state)
        old_model.fit(X.iloc[train_index], y[train_index])
        comparative_model.fit(X.iloc[train_index], y[train_index])
        fold_predictions = {
            "tfidf_logistic_regression": old_model.predict(X.iloc[validation_index]),
            "comparative_tfidf": comparative_model.predict(X.iloc[validation_index]),
            "rule_based": [
                rule_predict(eligible[i]["source_sentence"], eligible[i]["summary_sentence"]).label
                for i in validation_index
            ],
        }
        for key, values in fold_predictions.items():
            oof[key][validation_index] = values
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
            **{key: _metrics(fold_truth, list(fold_predictions[key])) for key in MODEL_KEYS},
        })

    if not np.all(seen == 1):
        raise RuntimeError("Cross-validation must predict every example exactly once.")
    predictions = [
        {"pair_id": row["pair_id"], "source_group_id": row["source_group_id"],
         "fold": next(fold["fold"] for fold in folds if row["source_group_id"] in fold["validation_group_ids"]),
         "ground_truth": str(y[i]), **{key: str(oof[key][i]) for key in MODEL_KEYS}}
        for i, row in enumerate(eligible)
    ]
    return {
        "labels": list(LABELS), "total": len(eligible), "n_splits": n_splits,
        "random_state": random_state, "folds": folds,
        "overall": {key: _metrics(list(y), list(oof[key])) for key in MODEL_KEYS},
        "predictions": predictions,
    }


def _print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}: accuracy={metrics['accuracy']:.3f}, macro F1={metrics['macro_f1']:.3f}")
    print("Class       Precision  Recall  F1      Support")
    for label in LABELS:
        row = metrics["per_class"][label]
        print(f"{label:<11} {row['precision']:<10.3f} {row['recall']:<7.3f} "
              f"{row['f1']:<7.3f} {row['support']}")
    print("Confusion matrix (rows=actual; columns=predicted; order=PRESERVED,HARDENED,SOFTENED):")
    for label, values in zip(LABELS, metrics["confusion_matrix"]):
        print(f"{label:<11} " + "  ".join(str(value) for value in values))


def print_report(result: dict) -> None:
    print(f"Three-model grouped OOF comparison: {result['total']} examples, "
          f"{result['n_splits']} folds, seed {result['random_state']}")
    for fold in result["folds"]:
        print(f"Fold {fold['fold']}: train={fold['train_size']}, validation={fold['validation_size']}, "
              f"group overlap={fold['group_overlap']}")
        for key in MODEL_KEYS:
            print(f"  {key}: accuracy={fold[key]['accuracy']:.3f}, macro F1={fold[key]['macro_f1']:.3f}")
    for key, title in (("comparative_tfidf", "Comparative TF-IDF + Logistic Regression"),
                       ("tfidf_logistic_regression", "Existing TF-IDF + Logistic Regression"),
                       ("rule_based", "Rule baseline")):
        _print_metrics(title, result["overall"][key])
    print("\nCaution: 24 synthetic examples are insufficient for reliable generalization claims; "
          "cross-validation scores are exploratory and highly sensitive to individual examples.")


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
        print(f"Comparative TF-IDF evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
