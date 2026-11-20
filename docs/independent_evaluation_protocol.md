# Independent real-world evaluation protocol

## Purpose and data boundary

`data/evaluation/independent_evaluation_template.csv` is a separate schema for naturally occurring source-summary sentence pairs. It is intentionally header-only until examples are collected and reviewed by humans. Do not copy examples from the synthetic candidate, reviewed, or training CSVs. Keep `data/processed/training_dataset.csv` as the sole training source for the current baselines.

An independent evaluation measures transfer to examples outside the synthetic development corpus. It is distinct from the existing 3-fold grouped cross-validation: cross-validation estimates performance across held-out groups within the small synthetic dataset; independent evaluation scores a separately collected corpus after models have been fit on the training dataset. Do not combine the two result sets or describe CV results as independent evaluation.

## Annotation guidelines

Annotate the *change in epistemic certainty* about the same proposition. Read both sentences in context when context is needed to resolve the claim, modality, or referent. Judge the expressed certainty, not whether the claim is true.

- **PRESERVED**: the summary expresses approximately the same degree of uncertainty as the source. Wording can differ (for example, “may” and “might”) when their force is comparable in context.
- **HARDENED**: the summary presents the proposition as more certain or definite. For example, a source possibility stated as an unqualified fact is hardened.
- **SOFTENED**: the summary makes the proposition less certain than the source, such as turning an assertion into a possibility or weakening a strong conclusion to a tentative one.
- **EXCLUDE**: the pair does not express the same underlying proposition, or a material mismatch makes the uncertainty comparison invalid. Examples include changed actors, dates, polarity/negation, causal claims, quantities, unsupported claims, or a summary that omits/replaces the relevant claim. EXCLUDE is a review outcome and never a model prediction.

### Ambiguous cases

1. Check that the source and summary concern the same actor, event, polarity, time, and claim. If not, use EXCLUDE rather than guessing a shift.
2. Interpret modal words in context. “May” meaning permission, “could” meaning ability, or “will” describing intention is not automatically epistemic uncertainty. If the pair is otherwise comparable, label the uncertainty actually expressed; if context is insufficient, leave it pending for review.
3. Treat evidential language such as “suggests” or “appears to” as uncertainty only when it qualifies the proposition. Do not count a word mechanically when it has another sense.
4. Consider negation and scope. “May not occur” differs from “will not occur”; if the summary changes polarity or scope in a way that changes the proposition, use EXCLUDE.
5. For mixed clauses or multiple cues, compare the certainty of the shared central proposition. If reviewers cannot identify a stable shared proposition or agree on the net direction, record disagreement; do not infer a majority label.
6. Use PRESERVED only when the difference is small enough that a reasonable reader would not infer a material increase or decrease in certainty. When that judgment is unclear, use disagreement/pending review rather than forcing a label.

## Independent reviews and disagreements

Two reviewers annotate independently in `reviewer_1_*` and `reviewer_2_*` fields. Each review records a distinct reviewer ID, one of the four labels, a review timestamp, and a short rationale. Do not show one reviewer’s label to the other before both initial reviews are recorded.

- Set `review_status=approved` only when both reviewers select the same model label (PRESERVED/HARDENED/SOFTENED), and set `label` to that shared label.
- Set `review_status=excluded` only when both reviewers select EXCLUDE, set `label=EXCLUDE`, and document `exclude_reason`.
- If reviewers differ, use `review_status=disagreement` and leave `label` blank. Do not automatically resolve a disagreement by majority, averaging, or choosing the first reviewer.
- Resolution is optional and must be explicit. A qualified third person may fill the adjudication fields and set `review_status=adjudicated`; `label` must match `adjudicated_label`. An adjudicated EXCLUDE remains excluded from evaluation.
- `requires_manual_review`, partial reviewer records, disagreement rows, and incomplete adjudications are not evaluation eligible.

## Provenance and collection fields

For every record, preserve `source_document_id`, a durable `source_reference` (such as DOI or URL), `source_location` (section, page, paragraph, timestamp, or other locator), and `domain`. Also record `collection_method`, `collection_date`, and `collection_notes` when available. Do not invent missing provenance; a row with missing required provenance is rejected until corrected. Follow the source's access and quotation terms when collecting sentence pairs.

`source_document_id` lets reviewers identify multiple pairs from one document. Do not count repeated pairs as independent evidence. The validator rejects normalized duplicate sentence pairs within the evaluation file and exact pair collisions (including reversed pair order) with the approved training dataset. This catches literal reuse, not paraphrase or semantic copying; provenance and human inspection are still required.

## Validation and eligibility

Run the validator from the repository root:

```powershell
python scripts\validate_independent_evaluation.py
```

It reports eligible, human-excluded, and rejected/incomplete row counts, plus counts by validation reason. Rows with structural errors, invalid labels, missing provenance, duplicate IDs/pairs, training collisions, incomplete reviews, or unresolved disagreement are rejected. Rows with a valid EXCLUDE decision are separately counted as human-excluded. Only valid `approved` or explicitly `adjudicated` rows with a final model label can be scored. The evaluator repeats these checks before prediction.

## Evaluation and leakage controls

After the dataset has been populated and reviewed, run:

```powershell
python scripts\evaluate_independent.py
```

The rule baseline predicts directly. The original and comparative TF-IDF pipelines are each fitted once using only rows returned by the existing reviewed training-data gate from `data/processed/training_dataset.csv`; their fixed existing settings are not tuned. The independent examples are passed only to `.predict`. Both vectorizers and comparative transformations therefore learn their vocabulary/statistics from training rows only. No evaluation labels or text enter `.fit`, model selection, or tuning.

The report includes eligible and excluded/rejected counts with reasons, individual predictions, accuracy, per-class precision/recall/F1, macro F1, and confusion matrices. A tiny or imbalanced evaluation set makes these estimates highly unstable; report class support and treat findings as preliminary. The models themselves were developed from only 24 synthetic examples, so even a larger independent set may expose substantial domain shift.
