# HedgeGuard local LLM demonstration

## What the LLM component is

The contextual component uses the pretrained `qwen3:4b-instruct` model through Ollama's local chat API. HedgeGuard sends a prompt for each submitted source-summary pair and requests a JSON-schema-constrained response. This is **prompting-based inference**. The model is not fine-tuned, trained, or updated by HedgeGuard.

The LLM predicts only `PRESERVED`, `HARDENED`, or `SOFTENED`. If the sentences do not express a comparable proposition, it returns no class label and marks the pair for human review. `EXCLUDE` remains a dataset-review label and is never a model prediction. The returned evidence consists of short text snippets, not a guaranteed faithful chain-of-thought or proof of correctness.

## Install and launch

The project virtual environment should be used from the repository root. Install the declared dependencies, including Streamlit:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen3:4b-instruct
ollama list
```

Ollama must be running locally and expose `http://localhost:11434`. The app uses a 60-second request timeout and a 4K context setting. Start the demonstration with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Enter one source sentence and one summary sentence, then choose **Analyze**. The LLM error is shown separately; the rule-based and TF-IDF cards remain available if the LLM request fails.

## Model behavior in the app

- **Rule-based:** applies the existing cue-strength heuristic and returns its explanation.
- **Original TF-IDF:** compares learned source-side and summary-side word features.
- **Comparative TF-IDF:** uses source/summary features, signed summary-minus-source TF-IDF differences, and uncertainty-cue features.
- **Qwen3 contextual LLM:** reasons over the pair in context and returns a concise explanation plus evidence snippets.

The Streamlit app loads `data/processed/training_dataset.csv`, passes rows through the existing `training_rows()` eligibility gate, and fits both TF-IDF pipelines once through Streamlit's resource cache. It does not read the independent evaluation template, and the independent examples are never used for fitting. The TF-IDF models do not expose a faithful per-example natural-language explanation; the UI explains their feature representation rather than inventing a rationale for a prediction.

## Limits

The two learned baselines are fitted on only 24 synthetic examples. LLM predictions are prompt-sensitive and may misread modality, negation, scope, or proposition alignment. A non-comparable result should be reviewed by a person. The UI is a demonstration, not an evaluation benchmark, and its predictions should not be interpreted as evidence of generalization to real-world summaries.
