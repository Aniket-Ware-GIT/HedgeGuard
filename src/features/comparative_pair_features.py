"""Fold-fitted directional text and cue features for source-summary pairs."""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

PAIR_COLUMNS = ("source_sentence", "summary_sentence")
CUE_FEATURE_NAMES = (
    "source_cue_count", "summary_cue_count", "cue_count_delta",
    "source_strength", "summary_strength", "strength_delta",
)

# A deliberately small, auditable cue inventory. Strengths match the rule
# baseline's ordinal scale: 1 possible, 2 probable/inferred, 3 asserted/certain.
_CUES = (
    ("cannot rule out", 1), ("can't rule out", 1), ("can not rule out", 1),
    ("not certain", 1), ("not sure", 1), ("inconclusive", 1), ("uncertain", 1),
    ("evidence suggests", 2), ("appears to", 2), ("seems to", 2),
    ("probably", 2), ("unlikely", 2), ("likely", 2), ("suggests", 2), ("indicates", 2),
    ("definitely", 3), ("certainly", 3), ("is certain to", 3), ("are certain to", 3),
    ("confirmed", 3), ("will", 3), ("may", 1), ("might", 1), ("could", 1), ("possibly", 1),
)
_CUE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term, _ in sorted(_CUES, key=lambda item: -len(item[0]))) + r")\b",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"\b(?:may|could)\s+(?:(?:not|also|still)\s+)?"
    r"(?:access|enter|use|submit|view|request|apply|leave|participate|be permitted|be allowed)\b",
    re.IGNORECASE,
)
_ABILITY = re.compile(
    r"\bcould\s+(?:swim|walk|run|speak|read|write|play|drive|lift|remember|see|hear|afford)\b",
    re.IGNORECASE,
)
_NEGATED_REPORTING = re.compile(
    r"\b(?:does not|do not|did not|doesn't|don't|didn't|has not|have not)\s+(?:strongly\s+)?suggests?\b",
    re.IGNORECASE,
)
_PROBABILITY_WITH_WILL = re.compile(r"\bwill\s+(?:probably|possibly|likely)\b", re.IGNORECASE)
_CUE_STRENGTH = dict(_CUES)


def _as_pair_frame(X: Iterable[dict[str, str]] | pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the required directional pair columns."""
    if isinstance(X, pd.DataFrame):
        missing = [column for column in PAIR_COLUMNS if column not in X.columns]
        if missing:
            raise ValueError("Sentence pairs are missing columns: " + ", ".join(missing))
        frame = X.loc[:, list(PAIR_COLUMNS)].copy()
    else:
        rows = list(X)
        if not rows:
            raise ValueError("Cannot extract pair features from an empty dataset.")
        missing = [column for column in PAIR_COLUMNS if column not in rows[0]]
        if missing:
            raise ValueError("Sentence pairs are missing columns: " + ", ".join(missing))
        frame = pd.DataFrame({column: [row[column] for row in rows] for column in PAIR_COLUMNS})
    for column in PAIR_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"{column} contains a missing value.")
        frame[column] = frame[column].map(str)
        if frame[column].str.strip().eq("").any():
            raise ValueError(f"{column} must contain non-empty sentences.")
    return frame.reset_index(drop=True)


def _cue_statistics(sentence: str) -> tuple[int, int]:
    """Return cue occurrence count and first-cue ordinal strength (default asserted)."""
    text = " ".join(sentence.split())
    ignored_spans = [match.span() for match in (*_PERMISSION.finditer(text), *_ABILITY.finditer(text))]
    matches = []
    for match in _CUE_PATTERN.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in ignored_spans):
            continue
        term = match.group().lower()
        if _NEGATED_REPORTING.search(text[max(0, match.start() - 16):match.end()]) and term.startswith("suggest"):
            continue
        if term == "will" and _PROBABILITY_WITH_WILL.match(text[match.start():]):
            continue
        matches.append((match.start(), term))
    if not matches:
        return 0, 3
    matches.sort(key=lambda item: item[0])
    return len(matches), _CUE_STRENGTH[matches[0][1]]


class ComparativePairFeatures(BaseEstimator, TransformerMixin):
    """Build shared-vocabulary source, summary, signed-difference and cue features.

    ``fit`` learns the TF-IDF vocabulary/IDF only from its supplied training
    records. ``transform`` reuses that fitted space, and computes summary minus
    source so reversing a pair negates the difference block.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), lowercase: bool = True):
        self.ngram_range = ngram_range
        self.lowercase = lowercase

    def fit(self, X, y=None):
        frame = _as_pair_frame(X)
        training_texts = pd.concat(
            [frame["source_sentence"], frame["summary_sentence"]], ignore_index=True,
        )
        self.vectorizer_ = TfidfVectorizer(ngram_range=self.ngram_range, lowercase=self.lowercase)
        self.vectorizer_.fit(training_texts)
        self.n_text_features_ = len(self.vectorizer_.get_feature_names_out())
        self.n_features_in_ = len(PAIR_COLUMNS)
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=("vectorizer_", "n_text_features_"))
        frame = _as_pair_frame(X)
        source = self.vectorizer_.transform(frame["source_sentence"])
        summary = self.vectorizer_.transform(frame["summary_sentence"])
        delta = summary - source
        numeric = np.asarray([
            (source_count, summary_count, summary_count - source_count,
             source_strength, summary_strength, summary_strength - source_strength)
            for source_count, source_strength, summary_count, summary_strength in (
                (*_cue_statistics(row.source_sentence), *_cue_statistics(row.summary_sentence))
                for row in frame.itertuples(index=False)
            )
        ], dtype=float)
        return hstack([source, summary, delta, csr_matrix(numeric)], format="csr")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, attributes=("vectorizer_", "n_text_features_"))
        vocabulary = self.vectorizer_.get_feature_names_out()
        return np.asarray([
            *(f"source__{term}" for term in vocabulary),
            *(f"summary__{term}" for term in vocabulary),
            *(f"summary_minus_source__{term}" for term in vocabulary),
            *CUE_FEATURE_NAMES,
        ], dtype=object)
