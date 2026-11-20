"""Leakage tests use synthetic in-memory fixtures only; they are not corpus records."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_independent as evaluator  # noqa: E402
from src.dataset.independent_evaluation import REQUIRED_COLUMNS  # noqa: E402
from src.dataset.pipeline import read_csv  # noqa: E402


def make_eligible_row():
    row = {column: "" for column in REQUIRED_COLUMNS}
    row.update({
        "example_id": "fixture-1", "source_document_id": "fixture-doc",
        "source_sentence": "A study may detect change.",
        "summary_sentence": "A study might detect change.", "label": "PRESERVED",
        "source_reference": "fixture reference", "source_location": "section 1", "domain": "science",
        "review_status": "approved", "reviewer_1_id": "r1", "reviewer_1_label": "PRESERVED",
        "reviewer_1_at": "2026-01-01", "reviewer_1_notes": "fixture review one",
        "reviewer_2_id": "r2", "reviewer_2_label": "PRESERVED",
        "reviewer_2_at": "2026-01-02", "reviewer_2_notes": "fixture review two",
    })
    return row


class SpyEstimator:
    def __init__(self):
        self.fit_frame = None
        self.fit_labels = None
        self.prediction_frame = None

    def fit(self, frame, labels):
        self.fit_frame = frame.copy()
        self.fit_labels = list(labels)
        return self

    def predict(self, frame):
        self.prediction_frame = frame.copy()
        return ["PRESERVED"] * len(frame)


def test_ml_fits_receive_training_rows_and_eval_rows_only_reach_predict(monkeypatch):
    training = read_csv(Path(__file__).resolve().parents[1] / "data/processed/training_dataset.csv")
    spies = []

    def factory(*, random_state=42):
        spy = SpyEstimator()
        spies.append(spy)
        return spy

    monkeypatch.setattr(evaluator, "build_tfidf_logistic_pipeline", factory)
    monkeypatch.setattr(evaluator, "build_comparative_tfidf_pipeline", factory)
    evaluation = make_eligible_row()
    result = evaluator.evaluate_records(training, [evaluation])
    assert result["training_count"] == 24
    assert result["evaluation_count"] == 1
    assert len(spies) == 2
    for spy in spies:
        assert len(spy.fit_frame) == 24
        assert "A study may detect change." not in spy.fit_frame["source_sentence"].tolist()
        assert spy.prediction_frame["source_sentence"].tolist() == [evaluation["source_sentence"]]
