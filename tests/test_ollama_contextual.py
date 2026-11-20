"""Ollama adapter tests use mocked HTTP; no server or model download is needed."""
from __future__ import annotations

import json
import socket
import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.ollama_contextual import (  # noqa: E402
    MODEL_NAME,
    OllamaContextualClassifier,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)


class FakeResponse:
    def __init__(self, result):
        content = json.dumps(result)
        self.body = json.dumps({"message": {"content": content}}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def response_data(label="PRESERVED", *, comparable=True):
    return {
        "proposition_comparable": comparable,
        "label": label,
        "explanation": "Both sentences express a similar degree of possibility.",
        "source_evidence": ["may reduce"],
        "summary_evidence": ["might reduce"],
    }


def classify_with(result):
    classifier = OllamaContextualClassifier(timeout=7)
    with patch("src.models.ollama_contextual.urllib.request.urlopen", return_value=FakeResponse(result)) as call:
        prediction = classifier.predict("The treatment may reduce pain.", "The treatment might reduce pain.")
    request = call.call_args.args[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:11434/api/chat"
    assert payload["model"] == MODEL_NAME
    assert payload["format"]["type"] == "object"
    assert call.call_args.kwargs["timeout"] == 7.0
    return prediction


@pytest.mark.parametrize("label", ["PRESERVED", "HARDENED", "SOFTENED"])
def test_valid_class_predictions(label):
    prediction = classify_with(response_data(label))
    assert prediction.label == label
    assert prediction.proposition_comparable is True
    assert prediction.requires_review is False
    assert prediction.source_evidence == ("may reduce",)
    assert prediction.summary_evidence == ("might reduce",)


def test_non_comparable_pair_withholds_label_and_requires_review():
    result = response_data(None, comparable=False)
    result["explanation"] = "The sentences describe different actors."
    prediction = classify_with(result)
    assert prediction.label is None
    assert prediction.proposition_comparable is False
    assert prediction.requires_review is True


def test_invalid_model_label_is_rejected():
    with pytest.raises(OllamaResponseError, match="label must be"):
        classify_with(response_data("EXCLUDE"))


def test_malformed_model_json_is_rejected():
    classifier = OllamaContextualClassifier()
    api_response = json.dumps({"message": {"content": "not-json"}}).encode()
    with patch("src.models.ollama_contextual.urllib.request.urlopen") as call:
        call.return_value.__enter__.return_value.read.return_value = api_response
        with pytest.raises(OllamaResponseError, match="malformed JSON content"):
            classifier.predict("Source statement.", "Summary statement.")


def test_ollama_connection_failure_has_actionable_error():
    classifier = OllamaContextualClassifier()
    with patch("src.models.ollama_contextual.urllib.request.urlopen",
               side_effect=urllib.error.URLError(ConnectionRefusedError("refused"))):
        with pytest.raises(OllamaUnavailableError, match="Start Ollama"):
            classifier.predict("Source statement.", "Summary statement.")


@pytest.mark.parametrize("failure", [TimeoutError("late"), socket.timeout("late")])
def test_ollama_timeout_has_useful_message(failure):
    classifier = OllamaContextualClassifier(timeout=2.5)
    with patch("src.models.ollama_contextual.urllib.request.urlopen", side_effect=failure):
        with pytest.raises(OllamaTimeoutError, match="2.5 seconds"):
            classifier.predict("Source statement.", "Summary statement.")


@pytest.mark.parametrize("source,summary", [("", "summary"), ("source", "  "), (None, "summary")])
def test_empty_or_non_string_sentence_is_rejected(source, summary):
    classifier = OllamaContextualClassifier()
    with pytest.raises(ValueError, match="non-empty string"):
        classifier.predict(source, summary)
