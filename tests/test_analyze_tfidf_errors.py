import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_tfidf_errors import analyze_records  # noqa: E402
from src.dataset.pipeline import read_csv  # noqa: E402


def test_analysis_lists_oof_errors_with_fold_text_and_directional_coefficients():
    dataset = Path(__file__).resolve().parents[1] / "data" / "processed" / "training_dataset.csv"
    result = analyze_records(read_csv(dataset), n_splits=3, random_state=42, top_k=2)

    assert result["total"] == 24
    assert result["error_count"] == len(result["errors"])
    assert {row["fold"] for row in result["predictions"]} == {1, 2, 3}
    assert all(not fold["group_overlap"] for fold in result["folds"])
    assert len(result["predictions"]) == 24
    assert all(error["actual_label"] != error["predicted_label"] for error in result["errors"])
    for error in result["errors"]:
        assert error["source_sentence"] and error["summary_sentence"]
        assert error["actual_label"] and error["predicted_label"]
        contributions = error["feature_contributions"]
        assert contributions["total_logit_difference"] > 0
        for direction in ("favoring_prediction", "favoring_actual"):
            assert len(contributions[direction]) <= 2
            assert all(feature["feature"].startswith(("source_tfidf__", "summary_tfidf__"))
                       for feature in contributions[direction])

    grouped_errors = [error for group in result["error_groups"] for error in group["errors"]]
    assert {error["pair_id"] for error in grouped_errors} == {error["pair_id"] for error in result["errors"]}
    assert all(group["count"] == len(group["errors"]) for group in result["error_groups"])


def test_analysis_requires_eligible_training_records():
    pending = {
        "pair_id": "pending", "source_group_id": "g", "source_sentence": "A source sentence.",
        "summary_sentence": "A summary sentence.", "label": "PRESERVED", "is_synthetic": "true",
        "review_status": "requires_manual_review", "include_in_training": "false",
        "original_label": "PRESERVED", "reviewer_id": "", "review_notes": "",
    }
    try:
        analyze_records([pending], n_splits=2)
        assert False, "pending rows should be rejected"
    except ValueError as exc:
        assert "training" in str(exc).lower()
