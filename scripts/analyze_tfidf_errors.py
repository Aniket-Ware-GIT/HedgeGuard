"""Inspect TF-IDF out-of-fold errors and fold-local Logistic Regression features."""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_tfidf import evaluate_records  # noqa: E402
from src.dataset.pipeline import read_csv, training_rows  # noqa: E402
from src.models.tfidf_model import build_tfidf_logistic_pipeline, pair_frame  # noqa: E402


def _feature_contributions(estimator, pair_features, actual: str, predicted: str,
                           top_k: int) -> dict[str, list[dict[str, float | str]]]:
    """Find active features whose class-weight differences favor either label."""
    feature_transformer = estimator.named_steps["features"]
    classifier = estimator.named_steps["classifier"]
    feature_names = feature_transformer.get_feature_names_out()
    classes = list(classifier.classes_)
    actual_index = classes.index(actual)
    predicted_index = classes.index(predicted)

    vector = feature_transformer.transform(pair_features).toarray().ravel()
    coefficient_delta = classifier.coef_[predicted_index] - classifier.coef_[actual_index]
    intercept_delta = float(classifier.intercept_[predicted_index] - classifier.intercept_[actual_index])
    contributions = vector * coefficient_delta
    active_indices = np.flatnonzero(vector)

    def describe(indices: list[int], *, reverse: bool) -> list[dict[str, float | str]]:
        ordered = sorted(indices, key=lambda index: contributions[index], reverse=reverse)[:top_k]
        return [
            {
                "feature": str(feature_names[index]),
                "tfidf_weight": float(vector[index]),
                "coefficient_difference": float(coefficient_delta[index]),
                "logit_contribution": float(contributions[index]),
            }
            for index in ordered
        ]

    favored_predicted = [int(i) for i in active_indices if contributions[i] > 0]
    favored_actual = [int(i) for i in active_indices if contributions[i] < 0]
    return {
        "intercept_difference": intercept_delta,
        "active_feature_logit_difference": float(contributions.sum()),
        "total_logit_difference": float(intercept_delta + contributions.sum()),
        "favoring_prediction": describe(favored_predicted, reverse=True),
        "favoring_actual": describe(favored_actual, reverse=False),
    }


def analyze_records(records: Iterable[dict[str, str]], *, n_splits: int = 3,
                    random_state: int = 42, top_k: int = 3) -> dict:
    """Recreate the existing OOF folds, attach fold IDs, and explain errors.

    Predictions are checked against ``evaluate_records`` to ensure this analysis
    reports the same out-of-fold model decisions as the existing evaluator.
    Coefficients are inspected only within the fold model that predicted each row;
    fold-specific vocabularies are not combined.
    """
    records = list(records)
    if not records:
        raise ValueError("Error analysis dataset is empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    eligible = training_rows(records)
    if len(eligible) != len(records):
        raise ValueError("Error analysis accepts only eligible approved or relabeled training rows.")

    reference = evaluate_records(eligible, n_splits=n_splits, random_state=random_state)
    reference_by_id = {row["pair_id"]: row for row in reference["predictions"]}
    labels = np.asarray([str(row["label"]).strip().upper() for row in eligible], dtype=object)
    groups = np.asarray([str(row["source_group_id"]).strip() for row in eligible], dtype=object)
    features = pair_frame(eligible)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    predictions: list[dict] = []
    errors: list[dict] = []
    fold_summaries: list[dict] = []
    seen = np.zeros(len(eligible), dtype=int)
    for fold_number, (train_indices, validation_indices) in enumerate(
            splitter.split(features, labels, groups), 1):
        train_group_ids = set(groups[train_indices])
        validation_group_ids = set(groups[validation_indices])
        overlap = train_group_ids & validation_group_ids
        if overlap:
            raise RuntimeError(f"Fold {fold_number} has source_group_id overlap: {sorted(overlap)}")

        estimator = build_tfidf_logistic_pipeline(random_state=random_state)
        estimator.fit(features.iloc[train_indices], labels[train_indices])
        fold_predictions = estimator.predict(features.iloc[validation_indices])
        seen[validation_indices] += 1
        fold_error_count = 0

        for local_index, row_index in enumerate(validation_indices):
            row = eligible[int(row_index)]
            actual = str(labels[row_index])
            predicted = str(fold_predictions[local_index])
            reference_row = reference_by_id[row["pair_id"]]
            if predicted != reference_row["tfidf_prediction"]:
                raise RuntimeError(
                    f"Reproduced OOF prediction differs for {row['pair_id']}; analysis stopped."
                )
            detail = {
                "pair_id": row["pair_id"],
                "source_group_id": row["source_group_id"],
                "fold": fold_number,
                "source_sentence": row["source_sentence"],
                "summary_sentence": row["summary_sentence"],
                "actual_label": actual,
                "predicted_label": predicted,
                "correct": actual == predicted,
            }
            predictions.append(detail)
            if actual != predicted:
                fold_error_count += 1
                detail["feature_contributions"] = _feature_contributions(
                    estimator,
                    features.iloc[[int(row_index)]],
                    actual,
                    predicted,
                    top_k,
                )
                errors.append(detail)

        fold_summaries.append({
            "fold": fold_number,
            "validation_size": int(len(validation_indices)),
            "train_group_count": len(train_group_ids),
            "validation_group_count": len(validation_group_ids),
            "group_overlap": sorted(overlap),
            "error_count": fold_error_count,
        })

    if not np.all(seen == 1):
        raise RuntimeError("Every record must appear in exactly one validation fold.")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for error in errors:
        grouped[(error["actual_label"], error["predicted_label"])].append(error)
    error_groups = [
        {"actual_label": actual, "predicted_label": predicted, "count": len(grouped[(actual, predicted)]),
         "errors": grouped[(actual, predicted)]}
        for actual in sorted({pair[0] for pair in grouped})
        for predicted in sorted({pair[1] for pair in grouped if pair[0] == actual})
    ]
    return {
        "total": len(eligible),
        "error_count": len(errors),
        "n_splits": n_splits,
        "random_state": random_state,
        "folds": fold_summaries,
        "predictions": predictions,
        "errors": errors,
        "error_groups": error_groups,
    }


def _print_feature_list(title: str, items: list[dict]) -> None:
    print(f"    {title}:")
    if not items:
        print("      (no active lexical features in this direction)")
    for item in items:
        print(f"      {item['feature']}: contribution={item['logit_contribution']:+.4f}, "
              f"coefficient_delta={item['coefficient_difference']:+.4f}, "
              f"tfidf={item['tfidf_weight']:.4f}")


def print_report(result: dict) -> None:
    print(f"TF-IDF error analysis: {result['error_count']} errors among {result['total']} OOF examples")
    print(f"Grouped folds: {result['n_splits']}; random seed: {result['random_state']}")
    print("Fold checks:")
    for fold in result["folds"]:
        print(f"  Fold {fold['fold']}: {fold['error_count']} errors, "
              f"train groups={fold['train_group_count']}, validation groups={fold['validation_group_count']}, "
              f"overlap={fold['group_overlap']}")

    print("\nErrors grouped by actual and predicted class:")
    if not result["error_groups"]:
        print("  No errors.")
    for group in result["error_groups"]:
        print(f"\n  Actual {group['actual_label']} -> predicted {group['predicted_label']} "
              f"({group['count']} examples)")
        for error in group["errors"]:
            print(f"\n  {error['pair_id']} | fold {error['fold']} | group {error['source_group_id']}")
            print(f"    Source:  {error['source_sentence']}")
            print(f"    Summary: {error['summary_sentence']}")
            print(f"    Actual: {error['actual_label']}; predicted: {error['predicted_label']}")
            print("    Active coefficient contributions (predicted class minus actual class):")
            features = error["feature_contributions"]
            print(f"      intercept={features['intercept_difference']:+.4f}; "
                  f"active-feature sum={features['active_feature_logit_difference']:+.4f}; "
                  f"total logit difference={features['total_logit_difference']:+.4f}")
            _print_feature_list("Features pushing toward the prediction", features["favoring_prediction"])
            _print_feature_list("Features favoring the actual class", features["favoring_actual"])

    print("\nInterpretation limits:")
    print("  Features are directional (source_tfidf__... versus summary_tfidf__...), but the model "
          "does not explicitly calculate a semantic change between them.")
    print("  Coefficients are fold-specific and based on very few synthetic examples. Local contributions "
          "show associations in this fitted fold, not causal explanations or stable linguistic rules.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/training_dataset.csv")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        result = analyze_records(read_csv(args.data), n_splits=args.folds,
                                 random_state=args.seed, top_k=args.top_k)
        print_report(result)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"TF-IDF error analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
