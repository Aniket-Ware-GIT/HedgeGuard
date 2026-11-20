"""Logistic Regression using explicitly comparative source-summary features."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.comparative_pair_features import ComparativePairFeatures


def build_comparative_tfidf_pipeline(random_state: int = 42) -> Pipeline:
    """Create an unfitted fold-local feature pipeline with the same LR settings as baseline."""
    return Pipeline([
        ("features", ComparativePairFeatures(ngram_range=(1, 2), lowercase=True)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=random_state)),
    ])
