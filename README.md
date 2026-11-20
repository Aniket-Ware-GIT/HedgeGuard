# HedgeGuard

HedgeGuard studies whether a generated summary preserves the level of uncertainty in a source sentence. It compares a source sentence with a matched summary sentence and assigns one of three labels:

- **PRESERVED** — approximately the same certainty level.
- **HARDENED** — the summary states the proposition with greater certainty.
- **SOFTENED** — the summary states it with less certainty.

The project is an undergraduate research prototype, not a production factuality system. It does not establish that the two sentences express the same proposition by itself. `EXCLUDE` is an annotation/review disposition for non-comparable pairs, never a model prediction class.

## Current project state

- 24 manually approved synthetic training examples (8 per class), kept separate from real-world candidates.
- A rule-based baseline, an original TF-IDF + Logistic Regression baseline, and a comparative TF-IDF + Logistic Regression baseline.
- A local Ollama Qwen3 prompting adapter and a four-component Streamlit demonstration. Qwen3 is pretrained and prompted; it is **not fine-tuned**.
- A separately licensed PMC/arXiv collection, pair retrieval, staged LLM-assisted annotation, validation, and human-audit workflow. Automated annotations are not human-reviewed ground truth.
- An initially empty independent evaluation template, separate from grouped cross-validation.

## Setup

PowerShell, from this directory:

```powershell
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe -m pytest -q
```

The declared dependencies are pandas, NumPy, scikit-learn, pytest, and Streamlit. The existing environment uses Python 3.12.14. No model weights are downloaded by Python setup.

## Dataset safeguards

`data/processed/training_dataset.csv` is the 24-row synthetic training dataset. Review and training eligibility are checked by `src/dataset/pipeline.py`; `EXCLUDE` and unreviewed rows cannot pass the gate. Do not edit reviewed labels to improve model scores. `data/evaluation/independent_evaluation_template.csv` is separate and currently contains only its schema header. The independent evaluator fits TF-IDF models on synthetic training rows only and never fits transformations on evaluation rows.

Real-world source material and annotations live under `data/real_world/`. The source manifest and raw PMC XML preserve source IDs, titles, URLs, publication dates, license text/URLs, collection information, and content hashes. The collection workflow fails closed when reuse rights are missing or outside its allowlist. Check each source license before redistribution. Semantic cosine similarity is used only to retrieve/rank candidates; Stage A independently assesses proposition comparability, and Stage B runs only on pairs marked comparable. Qwen annotations remain preliminary and are never represented as human reviews or independent accuracy results.

See [the real-world dataset workflow](docs/real_world_dataset_pipeline.md), [the general annotation guide](docs/annotation_guide.md), and [the independent evaluation protocol](docs/independent_evaluation_protocol.md).

## Baseline evaluation

Run the rule-based exploratory evaluation:

```powershell
\.venv\Scripts\python.exe scripts\evaluate_rule_based.py
```

Run the original and comparative TF-IDF grouped cross-validation evaluations:

```powershell
\.venv\Scripts\python.exe scripts\evaluate_tfidf.py
\.venv\Scripts\python.exe scripts\evaluate_comparative_tfidf.py
```

Both ML evaluations use 3-fold `StratifiedGroupKFold` (seed 42). Source-group disjointness is checked per fold, and all learned TF-IDF transformations are fitted inside each training fold. The comparative model has separate directional source and summary vectors, their signed `summary - source` difference, and transparent cue features. The rule-based result is 100% on this tiny, hand-constructed synthetic set; this is not evidence of generalization. In the current exploratory grouped OOF run, original TF-IDF accuracy is 25% (macro F1 0.249) and comparative TF-IDF accuracy is 87.5% (macro F1 0.875). With only 24 synthetic examples, all scores are unstable and must not be generalized to real summaries.

## Streamlit demonstration

Install Ollama separately and ensure the local server is running with `qwen3:4b-instruct` already installed. Then start:

```powershell
\.venv\Scripts\python.exe -m streamlit run app.py
```

The four components run independently; their predictions are not combined into an ensemble. TF-IDF models are fitted once from the eligible synthetic training data and cached by the app. The Qwen component uses Ollama’s local chat API; if it is unavailable, the other baseline predictions remain usable. The app does not use the independent evaluation template. See [the LLM demo notes](docs/llm_demo.md).

## Real-world pilot commands

The five-document PMC pilot is already collected locally. These commands do not collect new documents or call an LLM unless an annotation command is explicitly run:

```powershell
# Validate the existing saved v2 candidates/annotations into separate final outputs.
\.venv\Scripts\python.exe scripts\validate_real_dataset.py --output-suffix _final

# Inspect the versioned results.
Get-Content data\real_world\validation_report_final.json
```

The validator refuses to overwrite its chosen output set unless `--resume` is explicitly provided. The default `_final` output set is separate from historical v2 files. `eligible_pairs_final.csv` means machine-label-eligible under automated gates only; it is not human-approved training data. In the current saved pilot, 0 records pass eligibility. A zero is retained when evidence is insufficient; no balance target or acceptance threshold is used to manufacture positives. Do not split or train on real-world data without a separate, explicitly reviewed protocol.

## Limitations

The synthetic training set is tiny and deliberately balanced. The initial rule-based score reflects the hand-built cue patterns. The real-world pilot contains only five documents from a small biomedical sample; abstract/body sentence retrieval can return mismatched propositions, and automated certainty judgments can be wrong. Number and negation checks are conservative alarms, not semantic proof. Human review is required before any real-world record is used as training or evaluation ground truth. See [the project completion report](docs/PROJECT_COMPLETION_REPORT.md) for the verified status and remaining concerns.
