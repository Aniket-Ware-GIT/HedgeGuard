# HedgeGuard project completion report

**Verification date:** 14 November 2026  
**Environment:** Windows, project `.venv`, Python 3.12.14

## Executive summary

The project now has more conservative real-world validation, improved retrieval filters, versioned validation outputs, explicit numeric/negation and annotation regressions, and updated reproduction documentation. The saved five-document pilot was revalidated without downloading sources or calling Ollama. It remains **0 eligible, 4 excluded, 161 review-required, and 0 rejected**. No checks were relaxed to increase acceptance.

The 24 manually approved synthetic training examples, historical pilot and audit files, the three existing baselines, the Qwen adapter, and independent evaluation template were preserved. The Streamlit app received one additive caption clarifying that its four predictions are independent and are not ensembled. No model implementation, training, split, human annotation, or real-world accuracy evaluation was performed.

## Final architecture

1. PMC/arXiv collection retains source IDs, source location, publication information, reuse license, collection details, and source hashes. Retrieval uses same-document body/abstract sentences.
2. Retrieval filters obvious citation/DOI/URL notices, figure/table captions, formula-only fragments, and short fragments; embedding scores rank candidates only.
3. Stage A separately decides proposition comparability and requires exact proposition quotes. Stage B is selected only for Stage A comparable records.
4. Validation checks provenance, reuse, duplicates/collisions, stage/version contract, evidence quotes, certainty consistency, and conservative number/date/negation warnings.
5. The synthetic dataset passes through its existing human-review training gate. Real-world machine annotations remain separate and cannot pass as human-reviewed truth.
6. Rule-based, original TF-IDF, comparative TF-IDF, and pretrained prompt-based Qwen3 components remain independent in the demo; they are not an ensemble.

## Implemented and tested

### Numeric validation

In `src/dataset/real_world.py`, common DOI/URL, bracketed citation, table/figure/equation, and reference identifiers are removed before numeric extraction. Decimal formatting such as `1,170` / `1170` normalizes safely. A small contextual quantity mapper associates clear metric and entity anchors, so the bite-collapse kappa conflict (`0.000` vs `0.231`) remains a direct mismatch even among several statistics. A narrowly checked ratio-to-percent restatement is accepted only when numerator, denominator, direction, and exact arithmetic agree (`X/Y < 0.70` and `X more than 30% lower than Y`). Added, omitted, cohort-crossing, and otherwise ambiguously aligned details return `uncertain` rather than a fabricated direct conflict. The dropped `≥18 years` restriction is therefore detectable as an unmatched numeric detail (`uncertain`), never treated as a match.

### Negation validation

Negation markers now include selected semantic absence/insufficiency forms, but `negative` remains an adjective, not a polarity marker. `with or without` is not treated as a simple negation; `without` changes and uncertain scopes abstain. A polarity mismatch requires near-identical predicate terms after removing negation; unrelated clause differences and complex cases remain uncertain. Tests cover `not apparent`, `cannot be inferred`, `not yet sufficient`, `without refitting`, `with or without`, separate clauses, and “negative discordance.”

### Stage A/B and annotation reliability

Sentence-level numeric/date/negation checks run on staged rows only if Stage A marks them comparable and both proposition spans are exact input quotes. These checks do not establish equivalence. Stage B remains blocked unless Stage A says `comparable`; exact, nonempty Stage B evidence and certainty/label/direction consistency remain mandatory. The staged prompt version now refers to the shared annotation-version constant. Annotation errors are written as review-required with no forced class; explicit `--resume` retries rows whose saved raw payload contains an error while skipping completed IDs. Historical annotations were not regenerated.

### Candidate retrieval

`src/dataset/real_world_sources.py` now filters obvious DOI/reference notices, URLs in access notices, standalone figure/table captions, formula-only expressions, and short fragments before semantic ranking. Candidate ranking remains deterministic by score and source position, within the configured top-k. Embedding similarity remains retrieval-only. The historical 165 candidate rows were not regenerated, so retrieval changes are unit-tested but not measured against a new extraction output.

### Versioned validation outputs

`scripts/validate_real_dataset.py` accepts an output suffix and refuses to overwrite files with that suffix without explicit `--resume`. The run below used the distinct `_verified` suffix; earlier `_final` and historical v2 files were left in place.

## Five-document pilot validation

Saved inputs: the 165-row `candidate_pairs_v2.csv`, `proposition_assessments_v2.csv`, and `certainty_assessments_v2.csv`. The validator ran locally on these CSVs without Ollama. New outputs:

- `data/real_world/eligible_pairs_verified.csv`
- `data/real_world/excluded_pairs_verified.csv`
- `data/real_world/review_required_verified.csv`
- `data/real_world/validation_rejected_verified.csv`
- `data/real_world/validation_report_verified.json`

| Status | Previous v2 | Verified revalidation |
|---|---:|---:|
| Candidates | 165 | 165 |
| Eligible | 0 | 0 |
| Excluded | 4 | 4 |
| Review-required | 161 | 161 |
| Rejected | 0 | 0 |
| Provenance complete | 165/165 | 165/165 |
| Direct numeric-difference flags | 78 | 0 |
| Uncertain numeric comparisons | 0 recorded | 1 |
| Direct negation-difference flags | 12 | 0 |
| Uncertain negation comparisons | 0 recorded | 0 |

The previous numeric and negation checks compared full sentences without a proposition gate. The current run compares only 4 Stage A `comparable` pairs with exact quotes; 4 are Stage A non-comparable and 157 uncertain. Thus, the lower direct-flag counts do **not** establish that all former flags were false; uncertain/non-comparable rows remain review-required and are not given sentence-wide contradiction flags. One comparable pair with additional/omitted statistics has an uncertain numeric mapping and remains under review.

Pilot regression examples were checked separately by the comparator tests: the bite-collapse kappa values warn as a mismatch; citation `[2]` does not count as a claim number; the omitted age threshold returns uncertain; the ratio/percentage example matches under the narrow arithmetic rule; the AUROC/AUPRC cohort comparison abstains as uncertain. At pipeline level the first, third, and fifth examples are not directly checked because Stage A marked them uncertain/non-comparable; they remain review-required. The citation example remains review-required for separate Stage B evidence/direction issues. No record was newly eligible.

Saved machine labels: 3 `PRESERVED`, 0 `HARDENED`, 1 `SOFTENED`, and 4 `EXCLUDE`. These are Qwen-assisted, not human-reviewed. Remaining Stage B validation reasons include 2 certainty/label consistency issues, 2 invalid certainty directions, 2 invalid source-evidence issues, and 2 non-exact/missing source-evidence issues. The structured source/summary schema mismatch errors are absent; exact quote checks remain active. Reason categories may overlap by record.

## Model evaluation results

The 24-row synthetic set remains unchanged (8 examples per target class). The original 3-fold grouped cross-validation methodology (seed 42) was rerun. TF-IDF fitting remains inside each training fold, and each fold had zero source-group overlap.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Rule-based | 1.000 | 1.000 |
| Original TF-IDF + Logistic Regression | 0.250 | 0.249 |
| Comparative TF-IDF + Logistic Regression | 0.875 | 0.875 |

Per-class precision / recall / F1:

| Model | PRESERVED | HARDENED | SOFTENED |
|---|---|---|---|
| Rule-based | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Original TF-IDF | 0.125 / 0.125 / 0.125 | 0.400 / 0.250 / 0.308 | 0.273 / 0.375 / 0.316 |
| Comparative TF-IDF | 0.857 / 0.750 / 0.800 | 1.000 / 1.000 / 1.000 | 0.778 / 0.875 / 0.824 |

Confusion matrices use rows=actual and columns=predicted in class order PRESERVED, HARDENED, SOFTENED:

- Rule-based: `[[8,0,0],[0,8,0],[0,0,8]]`
- Original TF-IDF: `[[1,2,5],[3,2,3],[4,1,3]]`
- Comparative TF-IDF: `[[6,0,2],[0,8,0],[1,0,7]]`

These exploratory results are from only 24 synthetic examples and do not support generalization claims. The independent evaluation template is header-only; there is no independent test score. No model was trained in this completion task beyond the existing evaluation scripts’ fold-local fits.

## Quality assurance

- Focused dataset, annotation, and extraction tests: **81 passed**.
- Full project suite: **168 passed**.
- `python -m compileall -q src scripts app.py`: completed successfully.
- `scripts/validate_independent_evaluation.py`: 0 records, 0 eligible, 0 human-excluded, 0 rejected/incomplete.
- Rule-based, original TF-IDF, and comparative TF-IDF evaluation scripts completed with the results above.
- Streamlit smoke attempts: `streamlit.testing.v1.AppTest` stalled after runner start; a headless local server emitted event-loop startup logs but did not return its health endpoint within 20 seconds. No UI exception was captured. The app module compiles, but live UI startup is **not verified in this execution environment**.

Commands run for the final verification included:

```powershell
\.venv\Scripts\python.exe -m pytest tests/test_real_world_pipeline.py tests/test_real_world_sources.py tests/test_real_world_annotation.py -q
\.venv\Scripts\python.exe scripts\validate_real_dataset.py --output-suffix _verified
\.venv\Scripts\python.exe -m pytest -q
\.venv\Scripts\python.exe -m compileall -q src scripts app.py
\.venv\Scripts\python.exe scripts\validate_independent_evaluation.py
\.venv\Scripts\python.exe scripts\evaluate_rule_based.py
\.venv\Scripts\python.exe scripts\evaluate_tfidf.py
\.venv\Scripts\python.exe scripts\evaluate_comparative_tfidf.py
```

The attempted UI checks were `.venv\Scripts\python.exe -u -c "from streamlit.testing.v1 import AppTest; AppTest.from_file('app.py', default_timeout=20).run()"` and a headless `python -m streamlit run app.py --server.headless true --server.port 8765` process checked at `http://localhost:8765/_stcore/health`. Both were stopped after the bounded wait; neither changed project data.

## Reproduction commands

From the project root in PowerShell:

```powershell
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe -m pytest -q
\.venv\Scripts\python.exe -m compileall -q src scripts app.py
\.venv\Scripts\python.exe scripts\validate_real_dataset.py --output-suffix _verified
\.venv\Scripts\python.exe scripts\validate_independent_evaluation.py
\.venv\Scripts\python.exe scripts\evaluate_rule_based.py
\.venv\Scripts\python.exe scripts\evaluate_tfidf.py
\.venv\Scripts\python.exe scripts\evaluate_comparative_tfidf.py
\.venv\Scripts\python.exe -m streamlit run app.py
```

The real-world collection/extraction/annotation commands and their API/license/model safeguards are documented in [real_world_dataset_pipeline.md](real_world_dataset_pipeline.md). Do not rerun collection or annotation to reproduce this validator-only pilot check. Ollama must be separately running with `qwen3:4b-instruct` for annotation/demo LLM requests; no LLM was called here. The demo’s two TF-IDF pipelines fit once on the eligible synthetic training data and are cached. The independent evaluation template is not used by the demo.

## Files changed and preserved

Implementation/tests/docs changed:

- `src/dataset/real_world.py`
- `src/dataset/real_world_sources.py`
- `src/dataset/real_world_annotation.py`
- `app.py` (additive independent-prediction caption only)
- `scripts/validate_real_dataset.py`
- `scripts/annotate_real_pairs.py`
- `tests/test_real_world_pipeline.py`
- `tests/test_real_world_sources.py`
- `tests/test_real_world_annotation.py`
- `README.md`
- `docs/real_world_dataset_pipeline.md`
- `docs/PROJECT_COMPLETION_REPORT.md`

The five `_verified` outputs above are the authoritative revalidation result. An earlier intermediate `_completion` output set and the prior `_final` set remain preserved as separate files. Protected-file checking before the app caption change found that 21 of 22 recorded baseline checksums matched. `excluded_pairs_v2.csv` has current SHA-256 `5609bc3191877a4b5af5e1c1d2a3655a537f78a1c29aaaf35cf7600a9be4fbe7`, which differs from the previously recorded checksum string; its modification time predates this run, and no command in this run wrote to a v2 path. This one checksum discrepancy appears to be in the saved baseline value, but cannot be proven from the absent Git history. The app caption change is intentional; all other checked protected files matched, including the synthetic dataset, independent template, source/annotation inputs, audit files, and model modules. The repository has no `.git` directory, so Git status/diff was unavailable.

## Readiness and remaining work

- **Implemented and tested:** dataset safeguards, versioned no-overwrite validation outputs, conservative number/negation checks, retrieval filters, annotation error abstention/retry support, grouped synthetic evaluations.
- **Implemented but not fully verified live:** Streamlit UI startup; retry the smoke test on the normal desktop runtime before the demonstration.
- **Requires human review:** all real-world review-required pairs; no manual audit has been completed. Qwen labels are not ground truth.
- **Requires independent evaluation:** collecting and independently human-reviewing a sufficiently sized real-world evaluation set; the current template is empty.
- **Research readiness:** suitable as a documented undergraduate prototype and demonstration after confirming the UI starts locally. Not ready for real-world accuracy/generalization claims or training on real-world pilot annotations.

## Known limitations

Number keys and negation overlap are transparent heuristics, not semantic parsing. Contextual numeric labels may miss paraphrases, denominator shifts, repeated cohort mappings, and unit conversions; those cases should remain uncertain. Negation scope across clauses and technical terms is difficult to align without deeper parsing. Retrieval filters remove obvious noise but do not prove sentence-pair relevance. Human adjudication and independent real-world evaluation remain necessary.
