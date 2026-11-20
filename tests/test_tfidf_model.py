import sys
from pathlib import Path

import pytest
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.tfidf_model import build_tfidf_logistic_pipeline, pair_frame  # noqa: E402


def test_pipeline_keeps_vectorizers_inside_estimator_and_directional_branches():
    model = build_tfidf_logistic_pipeline(random_state=17)
    assert isinstance(model, Pipeline)
    features = model.named_steps["features"]
    assert isinstance(features, ColumnTransformer)
    assert [name for name, _, _ in features.transformers] == ["source_tfidf", "summary_tfidf"]
    assert all(isinstance(transformer, TfidfVectorizer) for _, transformer, _ in features.transformers)
    assert [column for _, _, column in features.transformers] == ["source_sentence", "summary_sentence"]
    assert all(not hasattr(transformer, "vocabulary_") for _, transformer, _ in features.transformers)


def test_pair_frame_keeps_source_and_summary_in_distinct_columns():
    frame = pair_frame([{"source_sentence": "The system may restart.",
                         "summary_sentence": "The system will restart."}])
    assert list(frame.columns) == ["source_sentence", "summary_sentence"]
    assert frame.iloc[0].to_dict() == {
        "source_sentence": "The system may restart.",
        "summary_sentence": "The system will restart.",
    }


@pytest.mark.parametrize("records", [
    [],
    [{"source_sentence": "A source sentence."}],
    [{"source_sentence": " ", "summary_sentence": "A summary sentence."}],
])
def test_pair_frame_rejects_empty_or_incomplete_input(records):
    with pytest.raises(ValueError):
        pair_frame(records)
