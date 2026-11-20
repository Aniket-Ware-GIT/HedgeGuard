# Architecture and implementation plan

## Current repository inspection

The selected workspace had NLP course PDFs, slide decks, and an image, but no HedgeGuard files. Thus there was no existing schema or annotation policy to migrate. The new project starts in a separate `HedgeGuard` subdirectory.

## Planned stages

1. Inspect existing assets and establish the architecture (complete: no project assets were present).
2. Define canonical data schema, label validation, controlled seed-based generation, deduplication, and group-safe splitting (implemented in this phase).
3. Rule-based baseline and exploratory evaluation (implemented in `src/models/rule_based.py` and `scripts/evaluate_rule_based.py`); TF-IDF + Logistic Regression baseline with grouped CV (implemented in `src/models/tfidf_model.py` and `scripts/evaluate_tfidf.py`).
4. Comparative TF-IDF feature experiment with Logistic Regression (implemented in `src/features/comparative_pair_features.py`, `src/models/comparative_tfidf.py`, and `scripts/evaluate_comparative_tfidf.py`).
5. Configurable DistilBERT fine-tuning and checkpoint handling.
6. Semantic sentence matching and inference.
7. Evaluation and export.
8. Streamlit dashboard.
9. Optional LLM provider.
10. Automated integration checks and full reproducible documentation.

## Data flow

Raw, human-curated source-summary pairs are validated and mapped to the canonical schema. The builder does not auto-rewrite sentences: grammar and claim preservation cannot be guaranteed by generic cue replacement. Exact normalized duplicates are dropped and counted. Rows retain their `source_group_id`, and all candidates are flagged for review. No split is created until examples are reviewed. `EXCLUDE` remains in the review output and must be removed from model-ready data.

## Label policy

- `PRESERVED`: epistemic uncertainty remains approximately equivalent.
- `HARDENED`: the summary increases certainty.
- `SOFTENED`: the summary decreases certainty.
- `EXCLUDE`: comparison is invalid or not epistemic (for example, changed claim or permission use of “may”). This status is never encoded as a model class.

The generator does not infer these difficult cases automatically. It marks generated rows synthetic, uses a narrow list of cue transformations, and requires human review.

## Rule-based baseline

The rule baseline maps explicit possibility cues to ordinal strength 1, probability/inference cues to strength 2, and explicit certainty or uncued assertions to strength 3. Equal strengths map to `PRESERVED`; a lower summary strength maps to `SOFTENED`; a higher summary strength maps to `HARDENED`. Basic exceptions cover common permission/ability readings and selected negation patterns. This transparent rule set does not establish claim equivalence or understand every contextual use; it serves as a simple reference baseline.

The evaluator reports metrics and individual examples against the current 24-row synthetic training dataset. It is an in-sample exploratory check only. It does not train a model, create a split, or estimate performance on real-world summaries.

`scripts/analyze_tfidf_errors.py` reproduces the grouped OOF folds to attach fold IDs and inspect local TF-IDF coefficient contributions for each error. It checks that predictions match the main evaluator. Contributions are fold-specific and should be treated as descriptive associations, not stable explanations.

## TF-IDF + Logistic Regression baseline

The model has two separate TF-IDF branches: one vectorizes `source_sentence`, the other vectorizes `summary_sentence`. Their features are concatenated with source/summary roles still distinct, then passed to Logistic Regression. Both vectorizers are steps inside the estimator pipeline, so cross-validation refits vocabularies and IDF weights using only each fold's training records. `StratifiedGroupKFold` keeps every `source_group_id` wholly in either training or validation for a fold. The evaluator also computes rule-based predictions on the exact same validation examples. This comparison remains exploratory because the dataset has only 24 synthetic examples.

## Comparative TF-IDF feature experiment

`ComparativePairFeatures` fits one shared vocabulary on the training fold's source and summary sentences. Each pair is represented by source TF-IDF, summary TF-IDF, and the signed sparse difference `summary - source`. It appends six transparent features: source/summary cue counts, their difference, source/summary ordinal cue strengths, and their difference. These features are passed to Logistic Regression with the same hyperparameters as the original model. The evaluator fits both TF-IDF models within one `StratifiedGroupKFold` loop and scores them and the rule baseline on the same validation rows. The data remains tiny and synthetic, so results are unstable and cannot establish generalization.
