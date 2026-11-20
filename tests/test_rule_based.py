import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.rule_based import (  # noqa: E402
    HARDENED, PRESERVED, SOFTENED, RuleBasedClassifier, predict,
)


def test_hardened_from_possibility_to_unhedged_assertion():
    result = predict("The treatment may reduce symptoms.", "The treatment reduces symptoms.")
    assert (result.label, result.source_strength, result.summary_strength) == (HARDENED, 1, 3)
    assert "higher assigned certainty" in result.explanation


def test_softened_from_assertion_to_possibility():
    result = predict("The treatment reduces symptoms.", "The treatment may reduce symptoms.")
    assert (result.label, result.source_strength, result.summary_strength) == (SOFTENED, 3, 1)


def test_preserved_between_possibility_cues():
    result = predict("The system may restart.", "The system might restart.")
    assert (result.label, result.source_strength, result.summary_strength) == (PRESERVED, 1, 1)


def test_no_explicit_cues_default_to_asserted_strength_and_preserved():
    result = predict("The sensor detected a temperature rise.", "The sensor recorded a temperature rise.")
    assert (result.label, result.source_strength, result.summary_strength) == (PRESERVED, 3, 3)
    assert "no explicit epistemic cue" in result.explanation


def test_unhedged_source_compared_with_probable_summary_is_softened():
    result = predict("The device fails under high temperatures.", "The device will probably fail under high temperatures.")
    assert (result.label, result.source_strength, result.summary_strength) == (SOFTENED, 3, 2)


@pytest.mark.parametrize("pair", [
    ("Employees may access the server.", "Employees are permitted to access the server."),
    ("Employees could use the equipment.", "Employees could use the equipment."),
])
def test_permission_modals_are_not_epistemic_uncertainty(pair):
    result = predict(*pair)
    assert result.label == PRESERVED
    assert result.source_strength == result.summary_strength == 3


def test_negated_reporting_verb_is_not_mistaken_for_a_hedge_cue():
    result = predict("The report does not suggest a change.", "The report does not suggest a change.")
    assert (result.label, result.source_strength, result.summary_strength) == (PRESERVED, 3, 3)


def test_epistemic_modal_under_negation_is_still_detected():
    result = predict("The treatment may not reduce symptoms.", "The treatment will not reduce symptoms.")
    assert (result.label, result.source_strength, result.summary_strength) == (HARDENED, 1, 3)


def test_prediction_structure_is_consistent_and_serializable():
    result = RuleBasedClassifier().predict("The results are likely to improve.", "The results will probably improve.")
    assert set(result.to_dict()) == {"label", "source_strength", "summary_strength", "explanation"}
    assert result.label == PRESERVED
    assert isinstance(result.explanation, str) and result.explanation


@pytest.mark.parametrize("source,summary", [("", "A summary sentence."), ("A source sentence.", " ")])
def test_empty_inputs_raise_clear_error(source, summary):
    with pytest.raises(ValueError):
        predict(source, summary)
