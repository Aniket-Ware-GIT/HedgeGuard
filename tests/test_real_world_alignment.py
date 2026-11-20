"""Tests for embedding retrieval with fake vectors; no Ollama/model required."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world_alignment import (  # noqa: E402
    OllamaEmbeddingScorer, SemanticAlignmentError, cosine_similarity,
)


def test_cosine_similarity_validates_and_scores_vectors():
    assert cosine_similarity([1, 0], [0.9, 0.1]) == pytest.approx(0.9938837)
    with pytest.raises(ValueError, match="equal"):
        cosine_similarity([1], [1, 2])
    with pytest.raises(ValueError, match="non-zero"):
        cosine_similarity([0], [1])


def test_pair_scoring_deduplicates_embedding_inputs_and_preserves_order():
    scorer = OllamaEmbeddingScorer("fixture-embed")
    with patch.object(scorer, "embed_texts", return_value=[[1, 0], [0.9, 0.1], [0, 1]]) as embed:
        scores = scorer.score_pairs([("source A", "summary A"), ("source A", "summary B")])
    assert embed.call_args.args[0] == ["source A", "summary A", "summary B"]
    assert scores[0] > scores[1]


def test_only_installed_exact_tag_is_accepted():
    scorer = OllamaEmbeddingScorer("embedding-test:1b")
    with patch.object(scorer, "_request", return_value={"models": [{"name": "embedding-test:1b"}]}):
        scorer.verify_installed()
    with patch.object(scorer, "_request", return_value={"models": [{"name": "qwen3:4b-instruct"}]}):
        with pytest.raises(SemanticAlignmentError, match="No model was downloaded"):
            scorer.verify_installed()


def test_embedding_response_shape_is_validated():
    scorer = OllamaEmbeddingScorer("fixture-embed")
    with patch.object(scorer, "_request", return_value={"embeddings": [[1, 0], [0, 1]]}):
        assert scorer.embed_texts(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
    with patch.object(scorer, "_request", return_value={"embeddings": [[1, 0]]}):
        with pytest.raises(SemanticAlignmentError, match="embedding count"):
            scorer.embed_texts(["a", "b"])


def test_embedding_requests_are_batched_for_large_documents():
    scorer=OllamaEmbeddingScorer("fixture-embed",batch_size=2)
    with patch.object(scorer,"_request",side_effect=[{"embeddings":[[1,0],[0,1]]},{"embeddings":[[1,1]]}]) as request:
        vectors=scorer.embed_texts(["a","b","c"])
    assert len(vectors)==3 and request.call_count==2
