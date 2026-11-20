# HedgeGuard real-world dataset pipeline

This pipeline collects naturally occurring source/abstract sentence pairs into a separate corpus. It does not change the reviewed synthetic dataset, the independent evaluation template, existing classifiers, or Streamlit app. Qwen labels are preliminary `llm_assisted` annotations, never human-reviewed ground truth and never evidence of model accuracy.

## Provenance and reuse

PMC is accessed only through its documented OAI-PMH API for licensed full text and NCBI E-utilities for search. The collector checks the article's own license and fails closed unless it recognizes CC0 or CC BY. It retains document IDs, title, URL, DOI when available, publication date, repository, collection query/date/method, license text/URL, source version, content hash, and local checkpoint path. Requests are sequential, paced at one second by default, and HTTP 429 responses receive bounded exponential backoff. Do not delete source attribution when redistributing derived data, and check the license terms for each retained article.

arXiv retrieval requires an explicit reusable right in OAI metadata before fetching TeX. It is paced at least 3.1 seconds between requests. Candidate extraction uses same-document abstract/body pairs and an installed Ollama embedding model to rank semantic similarity. Cosine similarity is retrieval-only: it cannot establish proposition equivalence or eligibility. Stage A separately assesses comparability. The pilot's existing lexical candidates remain preserved as v1 outputs.

The default collector limit is five documents. Larger collection requires both `--confirm-scale` and a human-reviewed `data/real_world/pilot_review.json` containing `{ "pilot_reviewed": true }`. A report is not implicitly reviewed by running the scripts. The 100-pair goal is a target only; there is no class-balancing or fabrication step.

## Pipeline stages

```powershell
# 1. Before any retrieval, verify local Ollama and the exact model tag.
.\.venv\Scripts\python.exe -c "from src.dataset.real_world_annotation import verify_ollama; print(verify_ollama())"

# 2. Retrieve up to five licensed documents for a pilot.
.\.venv\Scripts\python.exe scripts\collect_real_documents.py --source pmc --query 'your PMC query' --domain biomedical --limit 5

# 3. Extract and semantically rank same-document candidates. The named model
# must already appear in `ollama list`; this command never downloads it.
.\.venv\Scripts\python.exe scripts\extract_real_pairs.py --embedding-model 'your-installed-embedding-model-tag'

# 4. Stage A: proposition comparability only.
.\.venv\Scripts\python.exe scripts\annotate_real_pairs.py --stage comparability --limit 20

# 5. Stage B: only candidates Stage A marked comparable.
.\.venv\Scripts\python.exe scripts\annotate_real_pairs.py --stage certainty --limit 20

# 6. Validate saved v2 annotations into a separate output set.
.\.venv\Scripts\python.exe scripts\validate_real_dataset.py --output-suffix _final
.\.venv\Scripts\python.exe scripts\sample_real_world_audit.py --eligible data/real_world/eligible_pairs_final.csv --review data/real_world/review_required_final.csv --excluded data/real_world/excluded_pairs_final.csv --seed 42 --size 20 --output data/real_world/audit_sample_final.csv --results data/real_world/human_audit_results_final.csv
```

Ollama must be running locally and `qwen3:4b-instruct` installed before annotating. Stage A and B outputs are separate in `proposition_assessments_v2.csv` and `certainty_assessments_v2.csv`; revised candidate outputs use `candidate_pairs_v2.csv`. These versioned files leave the pilot files untouched. The embedding model must already be installed before extraction; `nomic-embed-text` was previously smoke-tested against Ollama's local embedding endpoint. Collection, extraction, annotation, and validation are separate commands, so validation alone does not contact Ollama or create an audit sample. No Python package or model is installed by the scripts.

Annotation schema and staged prompt use the shared `STAGED_ANNOTATION_VERSION` contract. Invalid JSON and transport errors are recorded as review-required rows without a forced label. On explicit `--resume`, rows whose raw annotation payload records an error can be retried; completed IDs are skipped to avoid duplicate annotations.

## Labels and quality gates

- `PRESERVED`, `HARDENED`, and `SOFTENED` describe only certainty direction when both sentences express the same proposition.
- `EXCLUDE` is not a prediction class. It means Qwen judged the propositions non-comparable; these records remain outside model datasets and are surfaced for human audit.
- Unclear judgments have a null label and `requires_review=true`.

The validator checks required provenance, allowlisted reuse, unique IDs, direct and reversed duplicate pairs, collision against the existing synthetic training rows, allowed labels, independent Stage A comparability, exact proposition quotes, certainty/label/direction agreement, exact evidence quotes and types, and subject/negation/number/date checks. Claim-level number, date, and negation checks run for staged records only after Stage A marks the pair comparable and both proposition spans are exact quotes. A similarity score or absence of deterministic flags never establishes equivalence. Obvious DOI, URL, citation, and numbered table/figure/equation metadata are removed before numeric comparison. Quantities are compared using a conservative metric/context key where one is available (for example, kappa by named variable), normalized decimal values, and units. A directly aligned, conflicting metric/context (for example, kappa for bite collapse) can flag a mismatch even when other metrics occur in the same sentence. Citation/DOI/table/figure metadata is removed, 1,170 and 1170 normalize identically, and narrowly validated ratio/percentage restatements are recognized only when the same numerator, denominator, direction, and exact threshold relation are present. If a quantity is added, omitted, repeated across comparison groups, or cannot be aligned, the result is uncertain for review rather than an automatic conflict. Calendar dates are checked separately, and complex or weakly aligned negation remains uncertain. Provenance, duplicate, license, and training-collision failures are rejected with reasons. Only complete, comparable, non-EXCLUDE rows with passing checks enter the machine-label-eligible output; that output is not training-ready or gold data.

The checked-in v2 pilot files are historical inputs. Revalidation after a validator change must use a new suffix, for example `python scripts/validate_real_dataset.py --output-suffix _final`. This writes `eligible_pairs_final.csv`, `excluded_pairs_final.csv`, `review_required_final.csv`, `validation_rejected_final.csv`, and `validation_report_final.json`. Existing outputs for the chosen suffix are protected unless `--resume` is explicitly passed. This is deterministic validation of saved candidates and assessments; it does not call Ollama, rewrite annotations, or collect documents.

## Human audit and disagreement

The sample script sorts eligible/reviewable candidates by stable ID, then samples with `random.Random(seed)` (default seed 42). The original pilot `audit_sample.csv` and `human_audit_results.csv` are frozen. Version 2 uses distinct `audit_sample_v2.csv` and `human_audit_results_v2.csv`. `scripts/create_hardened_audit_case.py` adds a separate `audit_hardened_case.csv` worksheet and `human_audit_hardened_results.csv`; it does not alter the random sample or its results. Human outcomes remain separate from Qwen annotations and disagreements are recorded, not automatically resolved.

## Pilot run (14 November 2026)

- 5 PMC documents passed the license check and were checkpointed.
- 149 same-document candidate pairs were extracted.
- 28 pairs were annotated by local `qwen3:4b-instruct`. The pilot batch exceeded 20 while checkpoint/resume behavior was being corrected; future `--limit` runs now cap the total annotation count, not just each invocation.
- Among those annotations, 7 were `EXCLUDE`, 1 was review-required with a null label, and 20 received target labels: 5 `PRESERVED`, 1 `HARDENED`, and 14 `SOFTENED`. The validator found 1 eligible record in the full 149-row candidate corpus, 7 excluded records, 141 review-required records (including 121 not-yet-annotated candidates), and 0 structurally rejected rows. Eligible class counts: 0 `PRESERVED`, 0 `HARDENED`, 1 `SOFTENED`.
- Provenance completeness: 149/149 candidates. The seeded audit draw produced 20 annotated, eligible or reviewable pairs. Audit outcomes have not yet been entered.
- Validation detected unsupported number/date/negation changes and label-strength inconsistencies. This confirms the need for human review; it does not establish Qwen accuracy.

No scaling beyond the five-document pilot has been run. Review the audit and validation report before creating `pilot_review.json` and collecting more documents.

## Grouped splitting

`scripts/split_real_dataset.py` makes deterministic 80/10/10 train/validation/test partitions using two `GroupShuffleSplit` stages and checks that `source_group_id` values are disjoint. It accepts only `eligible_pairs.csv`, not Qwen's raw annotations or the audit file, and is gated on a reviewed pilot report. With a small number of documents, resulting fractions and class coverage can be poor; the script requires at least four source groups and reports the realized sizes. These splits are exploratory and must not be described as independent evaluation. Keep the independent evaluation dataset separate.

## Limitations

Abstract/body sentence similarity is only a retrieval heuristic; it can pair statements with different scope or context. Qwen certainty judgments can be wrong, and automated flags are not calibrated. Five documents from one domain are not representative. A 100-pair target is not guaranteed, and the current pilot has only one machine-label-eligible pair. Any study of real-world generalization requires an adequately sized, independently human-reviewed set collected under the relevant licenses.
