import sys
from pathlib import Path

import numpy as np
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.comparative_pair_features import ComparativePairFeatures, CUE_FEATURE_NAMES  # noqa: E402


def test_shared_vocabulary_signed_difference_reverses_with_pair():
    training = [{"source_sentence": "The device may restart.",
                 "summary_sentence": "The device will restart."}]
    extractor = ComparativePairFeatures().fit(training)
    forward = extractor.transform(training)
    reverse = extractor.transform([{"source_sentence": training[0]["summary_sentence"],
                                    "summary_sentence": training[0]["source_sentence"]}])

    assert issparse(forward) and issparse(reverse)
    vocab_size = extractor.n_text_features_
    assert forward.shape[1] == reverse.shape[1] == (3 * vocab_size + len(CUE_FEATURE_NAMES))
    forward_array = forward.toarray()[0]
    reverse_array = reverse.toarray()[0]
    np.testing.assert_allclose(forward_array[:vocab_size], reverse_array[vocab_size:2 * vocab_size])
    np.testing.assert_allclose(forward_array[vocab_size:2 * vocab_size], reverse_array[:vocab_size])
    np.testing.assert_allclose(forward_array[2 * vocab_size:3 * vocab_size],
                               -reverse_array[2 * vocab_size:3 * vocab_size])
    np.testing.assert_allclose(forward_array[-6:-4], reverse_array[-6:-4][::-1])
    np.testing.assert_allclose(forward_array[-4], -reverse_array[-4])
    np.testing.assert_allclose(forward_array[-3:-1], reverse_array[-3:-1][::-1])
    np.testing.assert_allclose(forward_array[-1], -reverse_array[-1])


def test_tfidf_vocabulary_is_shared_between_both_pair_sides():
    extractor = ComparativePairFeatures().fit([{
        "source_sentence": "sourceonly may alter result",
        "summary_sentence": "summaryonly will alter result",
    }])
    source = extractor.vectorizer_.transform(["sourceonly may alter result"])
    summary = extractor.vectorizer_.transform(["summaryonly will alter result"])
    assert source.shape == summary.shape == (1, extractor.n_text_features_)
    assert set(extractor.vectorizer_.vocabulary_) >= {"sourceonly", "summaryonly"}


def test_fit_does_not_learn_terms_from_a_held_out_pair():
    extractor = ComparativePairFeatures().fit([{
        "source_sentence": "knownsource may improve throughput",
        "summary_sentence": "knownsummary improves throughput",
    }])
    assert "validationonly" not in extractor.vectorizer_.vocabulary_
    held_out = extractor.transform([{
        "source_sentence": "validationonly may improve throughput",
        "summary_sentence": "heldoutonly improves throughput",
    }])
    assert held_out.shape[1] == 3 * extractor.n_text_features_ + len(CUE_FEATURE_NAMES)
    assert "validationonly" not in extractor.vectorizer_.vocabulary_


def test_feature_names_mark_direction_and_signed_delta():
    extractor = ComparativePairFeatures().fit([{
        "source_sentence": "The system may restart",
        "summary_sentence": "The system might restart",
    }])
    names = list(extractor.get_feature_names_out())
    assert "source__may" in names
    assert "summary__may" in names
    assert "summary_minus_source__may" in names
    assert names[-len(CUE_FEATURE_NAMES):] == list(CUE_FEATURE_NAMES)
