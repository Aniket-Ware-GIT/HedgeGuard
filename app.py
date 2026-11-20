"""Streamlit demonstration comparing HedgeGuard's four prediction components."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.dataset.pipeline import read_csv, training_rows
from src.models.comparative_tfidf import build_comparative_tfidf_pipeline
from src.models.ollama_contextual import (
    OllamaContextualClassifier,
    OllamaContextualError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from src.models.rule_based import RuleBasedClassifier
from src.models.tfidf_model import build_tfidf_logistic_pipeline, pair_frame

PROJECT_ROOT = Path(__file__).resolve().parent
TRAINING_DATA = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"


@st.cache_resource(show_spinner="Preparing the TF-IDF baselines from reviewed training data…")
def load_tfidf_models():
    """Fit each ML baseline once, using only rows allowed by the training gate."""
    records = training_rows(read_csv(TRAINING_DATA))
    if not records:
        raise ValueError("The reviewed training dataset has no eligible examples.")
    labels = [str(row["label"]).strip().upper() for row in records]
    features = pair_frame(records)
    original = build_tfidf_logistic_pipeline(random_state=42)
    comparative = build_comparative_tfidf_pipeline(random_state=42)
    original.fit(features, labels)
    comparative.fit(features, labels)
    return original, comparative, len(records)


def _baseline_results(source: str, summary: str) -> tuple[list[dict[str, str]], str | None]:
    results = []
    try:
        rule = RuleBasedClassifier().predict(source, summary)
        results.append({"name": "Rule-based", "label": rule.label, "explanation": rule.explanation})
    except Exception as exc:  # keep the other components usable if one baseline errors
        results.append({"name": "Rule-based", "label": "Unavailable", "explanation": str(exc)})

    try:
        original, comparative, training_count = load_tfidf_models()
        frame = pd.DataFrame([{"source_sentence": source, "summary_sentence": summary}])
        original_label = str(original.predict(frame)[0])
        comparative_label = str(comparative.predict(frame)[0])
        results.extend([
            {
                "name": "Original TF-IDF",
                "label": original_label,
                "explanation": (
                    f"Uses separately learned source-side and summary-side word features. "
                    f"Fitted on {training_count} eligible training examples; it does not provide a "
                    "faithful natural-language rationale for this individual prediction."
                ),
            },
            {
                "name": "Comparative TF-IDF",
                "label": comparative_label,
                "explanation": (
                    f"Uses source and summary TF-IDF, signed summary-minus-source features, and cue "
                    f"differences. Fitted on {training_count} eligible training examples; it does not "
                    "provide a faithful natural-language rationale for this individual prediction."
                ),
            },
        ])
        training_note = f"The two TF-IDF baselines were fitted once on {training_count} eligible reviewed examples."
        return results, training_note
    except Exception as exc:
        results.extend([
            {"name": "Original TF-IDF", "label": "Unavailable", "explanation": str(exc)},
            {"name": "Comparative TF-IDF", "label": "Unavailable", "explanation": str(exc)},
        ])
        return results, None


st.set_page_config(page_title="HedgeGuard", page_icon="🛡️", layout="wide")
st.title("HedgeGuard – Uncertainty Preservation and Detection in Text Summarization")
st.caption("Comparing linguistic uncertainty between source text and generated summaries.")
st.caption("The four components make independent predictions; their outputs are not combined into an ensemble.")

source_text = st.text_area(
    "Source sentence",
    placeholder="For example: The treatment may reduce symptoms.",
    height=120,
)
summary_text = st.text_area(
    "Summary sentence",
    placeholder="For example: The treatment reduces symptoms.",
    height=120,
)
analyze = st.button("Analyze", type="primary", use_container_width=True)

if analyze:
    if not source_text.strip() or not summary_text.strip():
        st.warning("Enter both a source sentence and a summary sentence.")
    else:
        baseline_results, training_note = _baseline_results(source_text, summary_text)
        llm_prediction = None
        llm_error = None
        try:
            with st.spinner("Asking the local Qwen3 model for a contextual comparison…"):
                llm_prediction = OllamaContextualClassifier().predict(source_text, summary_text)
        except OllamaTimeoutError as exc:
            llm_error = f"The local Ollama request timed out. Check that Ollama is running, then retry. Details: {exc}"
        except OllamaUnavailableError as exc:
            llm_error = f"The local Ollama model is unavailable. Start Ollama and verify qwen3:4b-instruct is installed. Details: {exc}"
        except OllamaContextualError as exc:
            llm_error = f"Ollama returned an unusable response. The three baseline predictions are still available. Details: {exc}"
        except Exception as exc:
            llm_error = f"The LLM component failed. The three baseline predictions are still available. Details: {exc}"

        if training_note:
            st.caption(training_note)
        st.subheader("Model results")
        columns = st.columns(4)
        for column, result in zip(columns[:3], baseline_results):
            with column:
                st.markdown(f"#### {result['name']}")
                st.metric("Prediction", result["label"])
                st.write(result["explanation"])

        with columns[3]:
            st.markdown("#### Qwen3 contextual LLM")
            st.caption("Pretrained model prompted for this pair; it is not fine-tuned.")
            if llm_error:
                st.error(llm_error)
            elif llm_prediction is not None and llm_prediction.requires_review:
                st.warning("Not comparable — human review required")
                st.write(llm_prediction.explanation)
                st.write("**Source evidence:** " + (", ".join(llm_prediction.source_evidence) or "None returned"))
                st.write("**Summary evidence:** " + (", ".join(llm_prediction.summary_evidence) or "None returned"))
                st.caption("This is a review flag, not one of the three prediction classes or an EXCLUDE prediction.")
            elif llm_prediction is not None:
                st.metric("Prediction", llm_prediction.label)
                st.write(llm_prediction.explanation)
                st.write("**Source evidence:** " + (", ".join(llm_prediction.source_evidence) or "None returned"))
                st.write("**Summary evidence:** " + (", ".join(llm_prediction.summary_evidence) or "None returned"))

        st.caption(
            "These are demonstration predictions. The TF-IDF models are fitted on the 24-example "
            "synthetic training dataset; this interface does not report independent evaluation results."
        )
