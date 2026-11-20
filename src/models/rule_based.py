"""Small, explainable rules for comparing epistemic certainty in sentence pairs.

This baseline assumes the source and summary express the same proposition. It
uses a three-point ordinal scale, not a calibrated probability or a semantic
understanding of the sentences.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

PRESERVED = "PRESERVED"
HARDENED = "HARDENED"
SOFTENED = "SOFTENED"
LABELS = (PRESERVED, HARDENED, SOFTENED)

# Lower values mean more uncertainty. Keep phrases explicit and inspectable.
STRENGTH_NAMES = {1: "possible", 2: "probable_or_inferred", 3: "asserted_or_certain"}

_LOW_CUES = re.compile(
    r"\b(?:may|might|could|possibly|possibly|cannot rule out|can't rule out|"
    r"can not rule out|not certain(?: to| that)?|not sure(?: whether| if)?|"
    r"uncertain|inconclusive)\b",
    re.IGNORECASE,
)
_MID_CUES = re.compile(
    r"\b(?:probably|likely|unlikely|appears? to|seems? to|suggests?(?: that)?|"
    r"indicates?(?: that)?|evidence suggests?(?: that)?)\b",
    re.IGNORECASE,
)
_HIGH_CUES = re.compile(
    r"\b(?:definitely|certainly|confirmed|is certain to|are certain to|will)\b",
    re.IGNORECASE,
)
_PERMISSION = re.compile(
    r"\b(?:may|could)\s+(?:(?:not|also|still)\s+)?"
    r"(?:access|enter|use|submit|view|request|apply|leave|participate|"
    r"be permitted|be allowed)\b",
    re.IGNORECASE,
)
_ABILITY = re.compile(
    r"\bcould\s+(?:swim|walk|run|speak|read|write|play|drive|lift|"
    r"remember|see|hear|afford)\b",
    re.IGNORECASE,
)
_NEGATED_REPORTING = re.compile(
    r"\b(?:does not|do not|did not|doesn't|don't|didn't|has not|have not)\s+"
    r"(?:strongly\s+)?suggests?\b",
    re.IGNORECASE,
)
_PROBABILITY_WITH_WILL = re.compile(r"\bwill\s+(?:probably|possibly|likely)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RulePrediction:
    """Stable output structure for one source-summary comparison."""

    label: str
    source_strength: int
    summary_strength: int
    explanation: str

    def to_dict(self) -> dict[str, str | int]:
        """Return the prediction in a serialization-friendly form."""
        return asdict(self)


def _sentence_strength(sentence: str) -> tuple[int, str]:
    """Assign an ordinal certainty strength and the cue/default explanation."""
    text = " ".join(sentence.split())
    if not text:
        raise ValueError("Sentence must not be empty.")

    # Permission and familiar past-ability readings are not epistemic cues.
    permission_spans = [m.span() for m in _PERMISSION.finditer(text)]
    ability_spans = [m.span() for m in _ABILITY.finditer(text)]

    hits: list[tuple[int, int, str, int]] = []
    for strength, pattern in ((1, _LOW_CUES), (2, _MID_CUES), (3, _HIGH_CUES)):
        for match in pattern.finditer(text):
            span = match.span()
            if any(a <= span[0] and span[1] <= b for a, b in permission_spans + ability_spans):
                continue
            if strength == 2 and _NEGATED_REPORTING.search(text[max(0, span[0] - 16):span[1]]):
                continue
            # In "will probably happen", probability determines the strength;
            # the auxiliary "will" does not turn it into a definite prediction.
            if strength == 3 and match.group(0).lower() == "will" and _PROBABILITY_WITH_WILL.match(text[match.start():]):
                continue
            hits.append((span[0], span[1], match.group(0), strength))

    if not hits:
        return 3, "no explicit epistemic cue (treated as an assertion)"

    # Use the first cue in sentence order. This is easy to explain and avoids
    # treating later cues from another clause as if they modified the first claim.
    _, _, cue, strength = min(hits, key=lambda hit: (hit[0], -(hit[1] - hit[0])))
    return strength, f"cue '{cue}' means {STRENGTH_NAMES[strength]} (strength {strength})"


def predict(source_sentence: str, summary_sentence: str) -> RulePrediction:
    """Compare two matched sentences and return a 3-class rule-based prediction.

    Strength scale: 1 = possible, 2 = probable/inferred, 3 = asserted/certain.
    If no supported epistemic cue is found, the sentence defaults to strength 3.
    """
    if not isinstance(source_sentence, str) or not source_sentence.strip():
        raise ValueError("source_sentence must be a non-empty string.")
    if not isinstance(summary_sentence, str) or not summary_sentence.strip():
        raise ValueError("summary_sentence must be a non-empty string.")

    source_strength, source_reason = _sentence_strength(source_sentence)
    summary_strength, summary_reason = _sentence_strength(summary_sentence)
    if source_strength == summary_strength:
        label = PRESERVED
        direction = "the assigned certainty strengths are equal"
    elif summary_strength > source_strength:
        label = HARDENED
        direction = "the summary has a higher assigned certainty strength"
    else:
        label = SOFTENED
        direction = "the summary has a lower assigned certainty strength"

    explanation = (
        f"Source: {source_reason}; summary: {summary_reason}; {direction}. "
        "This is a cue-based heuristic and does not verify proposition equivalence."
    )
    return RulePrediction(label, source_strength, summary_strength, explanation)


class RuleBasedClassifier:
    """Tiny object wrapper for callers that prefer a classifier interface."""

    def predict(self, source_sentence: str, summary_sentence: str) -> RulePrediction:
        return predict(source_sentence, summary_sentence)
