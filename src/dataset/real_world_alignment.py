"""Local semantic retrieval scoring; cosine scores are never proposition judgments."""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Protocol, Sequence

OLLAMA_URL = "http://localhost:11434"


class SemanticAlignmentError(RuntimeError):
    """Raised when the requested local embedding service/model is unavailable."""


class PairScorer(Protocol):
    model_name: str

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must have equal, non-zero dimensions.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding vectors must be non-zero.")
    return dot / (left_norm * right_norm)


class OllamaEmbeddingScorer:
    """Use an already-installed Ollama embedding model; never pulls a model."""

    def __init__(self, model_name: str, *, base_url: str = OLLAMA_URL,
                 timeout: float = 60.0, batch_size: int = 64) -> None:
        if not model_name.strip():
            raise ValueError("An explicitly installed embedding model name is required.")
        if batch_size < 1:
            raise ValueError("Embedding batch_size must be positive.")
        self.model_name = model_name.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size

    def _request(self, endpoint: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + endpoint, data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise SemanticAlignmentError(f"Ollama embedding request failed: {exc}") from exc
        if not isinstance(value, dict):
            raise SemanticAlignmentError("Ollama embedding endpoint returned a non-object response.")
        return value

    def verify_installed(self) -> None:
        result = self._request("/api/tags")
        models = result.get("models", [])
        if not any(m.get("name") == self.model_name or m.get("model") == self.model_name for m in models):
            raise SemanticAlignmentError(
                f"Embedding model {self.model_name!r} is not installed in Ollama. "
                "No model was downloaded; install one only after explicit approval."
            )

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        normalized: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            result = self._request("/api/embed", {"model": self.model_name, "input": batch})
            vectors = result.get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != len(batch):
                raise SemanticAlignmentError("Ollama returned an unexpected embedding count.")
            try:
                normalized.extend([[float(value) for value in vector] for vector in vectors])
            except (TypeError, ValueError) as exc:
                raise SemanticAlignmentError("Ollama returned malformed embedding vectors.") from exc
        dimensions = {len(vector) for vector in normalized}
        if len(dimensions) != 1 or not dimensions or not next(iter(dimensions)):
            raise SemanticAlignmentError("Ollama returned inconsistent embedding dimensions.")
        if any(not math.isfinite(value) for vector in normalized for value in vector):
            raise SemanticAlignmentError("Ollama returned non-finite embedding values.")
        return normalized

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        texts = list(dict.fromkeys(text for pair in pairs for text in pair))
        vectors = self.embed_texts(texts)
        by_text = dict(zip(texts, vectors))
        # Scores only rank retrieval candidates. They do not establish equivalence.
        return [cosine_similarity(by_text[left], by_text[right]) for left, right in pairs]
