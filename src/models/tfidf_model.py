"""Directional TF-IDF + Logistic Regression baseline for sentence pairs."""
from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PAIR_COLUMNS = ("source_sentence", "summary_sentence")


def pair_frame(records: Iterable[dict[str, str]]) -> pd.DataFrame:
    """Build the two-column directional input expected by the estimator."""
    rows = list(records)
    if not rows:
        raise ValueError("Cannot build pair features from an empty dataset.")
    missing = [column for column in PAIR_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError("Sentence pairs are missing columns: " + ", ".join(missing))
    frame = pd.DataFrame({column: [str(row[column]) for row in rows] for column in PAIR_COLUMNS})
    if frame.isna().any().any() or (frame.apply(lambda column: column.str.strip().eq("")).any().any()):
        raise ValueError("Source and summary sentences must be non-empty strings.")
    return frame


def build_tfidf_logistic_pipeline(random_state: int = 42) -> Pipeline:
    """Create an unfitted estimator with both TF-IDF vocabularies inside CV folds.

    Source and summary sentences get independent word unigram/bigram spaces.
    Their feature blocks are concatenated directionally, so (for example) a
    source-side ``may`` feature is distinct from a summary-side ``may`` feature.
    """
    directional_tfidf = ColumnTransformer(
        transformers=[
            ("source_tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True), "source_sentence"),
            ("summary_tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True), "summary_sentence"),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("features", directional_tfidf),
        ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])
