# HedgeGuard v2 Real-World Pilot: Read-Only Error Analysis

## Scope and method

This report examines the existing five-document v2 pilot outputs only. It reads the candidate, Stage A, Stage B, validation-report, and eligible/excluded/review/rejected CSV files. It does not call Ollama, create annotations, alter data, run tests, train models, or split data. Candidate pairs below are identified by `example_id`; sentence text and scores are copied from the v2 CSVs. The Stage B evidence objects are shown as recorded.

The Qwen labels are machine-assisted annotations, not independent ground truth. There are no human adjudications in these files, so text-based observations below identify plausible failures or reasons to abstain; they do not establish the correct label for a pair.

## Pilot-level results

| Measure | Count |
|---|---:|
| Candidate pairs | 165 |
| Stage A comparable | 4 |
| Stage A non-comparable | 4 |
| Stage A uncertain | 157 |
| Stage A request errors | 8 (all recorded as uncertain after 90-second timeouts) |
| Stage B assessments | 4 (all 4 Stage A comparable IDs; no other pair received Stage B) |
| Stage B labels | PRESERVED 3; HARDENED 0; SOFTENED 1 |
| Stage A non-comparable pairs mapped to `EXCLUDE` by validation | 4 |
| Validation: excluded | 4 |
| Validation: review-required | 161 |
| Validation: rejected | 0 |
| Validation: eligible for training | **0** |
| Complete provenance | 165/165 |

Stage A uncertain rows comprise 84 where the proposition quote was empty or not an exact quote, 8 request timeouts, 4 with a `subject_match=uncertain` conflict, and 61 other uncertain responses. These categories describe validation outcomes, not 157 proven model mistakes: abstaining is appropriate when comparability cannot be established.

Validation reason counts overlap at the row level: `detected_number_difference` 78; `detected_negation_difference` 12; `certainty_label_direction_inconsistent` 2; `invalid_certainty_direction` 2; Stage B `invalid_source_evidence` 2; `invalid_source_evidence_schema` 4; `invalid_summary_evidence_schema` 4; `llm_requires_review` 161. No `detected_date_difference` occurred. The 4 non-comparable pairs were assigned `EXCLUDE`; no rejected candidates were emitted by the v2 validator.

## 1. All Stage A comparable pairs and Stage B results



### RW-47DD26BF6C9B5C5F (`RW-47DD26BF6C9B5C5F`)

Document: `PMC13754737`; location: Discussion; paragraph 86; sentence 3 → Abstract; paragraph 3; sentence 3.

**Source:** At the illustrative 20% testing allocation, the model identified 61.2% of D2 cases in development and 55.4% in external validation, with 3.06-fold and 2.77-fold enrichment over random testing, respectively.

**Summary:** At the 20% testing allocation, risk-guided testing identified 61.2% of D2 cases in development and 55.4% externally, with 3.06-fold and 2.77-fold enrichment over random testing.

**Alignment:** lexical `0.562500`; semantic `0.901806` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `comparable` (flags subject/negation/numbers/dates: `true` / `true` / `true` / `true`). Both sentences describe the same testing allocation (20%), the same identification rates (61.2% in development, 55.4% externally), and the same enrichment factors (3.06-fold and 2.77-fold over random testing). The source sentence uses 'model' and the summary sentence uses 'risk-guided testing', but this is a minor difference in terminology without altering the underlying proposition. All key claims, subject, scope, polarity, and quantities are preserved.

**Stage A quoted propositions:** source 'At the illustrative 20% testing allocation, the model identified 61.2% of D2 cases in development and 55.4% in external validation, with 3.06-fold and 2.77-fold enrichment over random testing, respectively.'; summary 'At the 20% testing allocation, risk-guided testing identified 61.2% of D2 cases in development and 55.4% externally, with 3.06-fold and 2.77-fold enrichment over random testing.'.

**Stage B:** label `SOFTENED`; certainty `ASSERTED` → `ASSERTED`; direction `UNCERTAIN`; annotation status `requires_review`; review reasons `certainty_label_direction_inconsistent.

**Stage B source evidence:** quote: 'identified 61.2% of D2 cases in development and 55.4% in external validation'; type: asserted_predicate; interpretation: The phrase 'identified 61.2% of D2 cases in development and 55.4% in external validation' expresses a factual claim with specific numbers and clear predicates, indicating an asserted level of certainty.

**Stage B summary evidence:** quote: 'risk-guided testing identified 61.2% of D2 cases in development and 55.4% externally'; type: asserted_predicate; interpretation: The phrase 'risk-guided testing identified 61.2% of D2 cases in development and 55.4% externally' contains specific numerical values and a clear predicate, indicating an asserted level of certainty.

**Stage B explanation:** Both the source and summary sentences are highly specific and contain precise numerical values (e.g., 61.2%, 55.4%, 3.06-fold, 2.77-fold) and consistent terminology (e.g., 'testing allocation', 'D2 cases', 'enrichment over random testing'). The summary sentence is a rephrasing of the source sentence with minimal changes (e.g., 'illustrative' removed, 'in development' and 'externally' replaced with 'externally' and 'externally'), preserving all key factual content. However, there is no explicit uncertainty cue in either sentence, and both appear to assert the facts with full confidence. Since the task requires comparing certainty strength and both sentences are presented as factual assertions without any qualifiers or hedging, the certainty strength is equal and asserted. Nevertheless, the instruction specifies that if either strength is UNCLEAR or no valid exact evidence quote can be provided, the result must be UNCERTAIN/null. While both sentences are clearly asserted in content, the absence of an explicit uncertainty cue does not automatically imply a lower certainty. However, the instruction requires a certainty strength label (POSSIBLE, PROBABLE, ASSERTED) and a direction (e.g., source > summary or source < summary). Since both sentences are identical in content and certainty, and no evidence shows a difference in strength, the direction is neutral. Given that the task is to compare certainty strength and both are asserted, the result should be ASSERTED. But the instruction says: 'If either strength is UNCLEAR or you cannot provide a valid exact evidence quote for each sentence, use UNCERTAIN/null.' There is no uncertainty in either sentence, and valid evidence quotes can be provided. Therefore, the correct response should not be UNCERTAIN. However, the instruction says: 'Compare only certainty strength.' Since both sentences are equally certain and asserted, the direction is neutral. But the output must have a direction. The only valid direction is when one is stronger than the other. Since both are identical, no direction can be assigned. Therefore, the certainty_direction must be UNCERTAIN because no valid directional comparison exists. Thus, the final label and certainty_direction must both equal the direction implied by the two strengths. Since no directional difference exists, and both are asserted, the direction is neutral, which is not one of the options. Hence, the only valid response under the rules is UNCERTAIN.

**Validation:** `review_required`; reasons: `llm_requires_review;certainty_label_direction_inconsistent;invalid_certainty_direction;invalid_source_evidence_schema;invalid_summary_evidence_schema`.

### RW-B7A737B70D2ACA8B (`RW-B7A737B70D2ACA8B`)

Document: `PMC13754737`; location: Discussion; paragraph 86; sentence 5 → Abstract; paragraph 3; sentence 4.

**Source:** At the same 20% allocation externally, XGBoost identified 62 D2 cases versus 42 with the clinical core, 30 with eGFRcr-only logistic regression, and 43 with an eGFRcr + BUN + ALB logistic model; the paired recall difference versus the latter was +17.0 percentage points (95% CI, 8.6–27.2).

**Summary:** At this allocation in external validation, XGBoost identified 62 D2 cases versus 43 with the three-variable model (recall difference, 17.0 percentage points; 95% CI, 8.6-27.2).

**Alignment:** lexical `0.400000`; semantic `0.860537` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `comparable` (flags subject/negation/numbers/dates: `true` / `true` / `true` / `true`). Both sentences report the same external allocation, the same number of D2 cases identified by XGBoost (62), and the same recall difference (17.0 percentage points) with a 95% CI of 8.6–27.2. The summary sentence simplifies the comparison by combining the three-variable model into a single reference (43 cases), which aligns with the original statement about the eGFRcr + BUN + ALB model (43 cases). The claim about recall difference and confidence interval is preserved in both, and the scope (external validation) and subject (XGBoost vs. models) remain consistent.

**Stage A quoted propositions:** source 'At the same 20% allocation externally, XGBoost identified 62 D2 cases versus 42 with the clinical core, 30 with eGFRcr-only logistic regression, and 43 with an eGFRcr + BUN + ALB logistic model; the paired recall difference versus the latter was +17.0 percentage points (95% CI, 8.6–27.2).'; summary 'At this allocation in external validation, XGBoost identified 62 D2 cases versus 43 with the three-variable model (recall difference, 17.0 percentage points; 95% CI, 8.6-27.2).'.

**Stage B:** label `PRESERVED`; certainty `ASSERTED` → `ASSERTED`; direction `PRESERVED`; annotation status `requires_review`; review reasons `invalid_source_evidence.

**Stage B source evidence:** quote: 'XGBoost identified 62 D2 cases versus 43 with an eGFRcr + BUN + ALB logistic model; the paired recall difference versus the latter was +17.0 percentage points (95% CI, 8.6–27.2)'; type: asserted_predicate; interpretation: The predicate 'XGBoost identified 62 D2 cases versus 43 with an eGFRcr + BUN + ALB logistic model; the paired recall difference versus the latter was +17.0 percentage points (95% CI, 8.6–27.2)' explicitly asserts the comparison and the recall difference with a confidence interval.

**Stage B summary evidence:** quote: 'XGBoost identified 62 D2 cases versus 43 with the three-variable model (recall difference, 17.0 percentage points; 95% CI, 8.6-27.2)'; type: asserted_predicate; interpretation: The predicate 'XGBoost identified 62 D2 cases versus 43 with the three-variable model (recall difference, 17.0 percentage points; 95% CI, 8.6-27.2)' directly asserts the same numerical comparison and confidence interval as in the source sentence, with the same precision and structure.

**Stage B explanation:** Both sentences report the same key numerical comparison: XGBoost identifying 62 D2 cases versus 43 with the eGFRcr + BUN + ALB logistic model, with a recall difference of +17.0 percentage points (95% CI, 8.6–27.2). The source sentence provides the full detail of the comparison (including the 30 with eGFRcr-only logistic regression), while the summary sentence simplifies it to the comparison with the three-variable model. The core claim—recall difference of 17.0 percentage points—is identical in both, and the confidence interval is explicitly stated in both. The certainty of the numerical claim is thus preserved. The evidence in both sentences is explicit and directly supports the proposition with no ambiguity or uncertainty cues.

**Validation:** `review_required`; reasons: `llm_requires_review;invalid_source_evidence;invalid_source_evidence_schema;invalid_summary_evidence_schema;detected_number_difference`.

### RW-74599612416467FF (`RW-74599612416467FF`)

Document: `PMC13756842`; location: Background; paragraph 1; sentence 4 → Abstract; paragraph 1; sentence 4.

**Source:** International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated [2].

**Summary:** International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated.

**Alignment:** lexical `1.000000`; semantic `0.994460` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `comparable` (flags subject/negation/numbers/dates: `true` / `true` / `true` / `true`). The source sentence and summary sentence are identical in content and structure, with the citation [2] omitted in the summary. The proposition expressed—diagnostic delays of more than a decade being common and patients presenting with advanced organ involvement before treatment—is fully preserved in both. There is no change in claim, subject, scope, polarity, quantity, or time.

**Stage A quoted propositions:** source 'International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated'; summary 'International registry data suggest that diagnostic delays of more than a decade are common, with many patients presenting with advanced organ involvement before treatment is initiated'.

**Stage B:** label `PRESERVED`; certainty `ASSERTED` → `ASSERTED`; direction `UNCERTAIN`; annotation status `requires_review`; review reasons `certainty_label_direction_inconsistent.

**Stage B source evidence:** quote: 'suggest that diagnostic delays of more than a decade are common'; type: asserted_predicate; interpretation: The predicate 'suggest that diagnostic delays of more than a decade are common' expresses a claim based on data, and the presence of a citation [2] indicates that the claim is supported by a source. This constitutes an asserted claim with evidentiary backing.

**Stage B summary evidence:** quote: 'suggest that diagnostic delays of more than a decade are common'; type: asserted_predicate; interpretation: The summary sentence is identical to the source sentence, so it inherits the same certainty. The predicate 'suggest that diagnostic delays of more than a in a decade are common' is presented as a claim based on registry data, with no uncertainty cues. Thus, the certainty is asserted.

**Stage B explanation:** Both the source_sentence and summary_sentence are identical in content. There is no explicit uncertainty cue in either sentence, and the content is presented as a suggestion based on registry data. However, the original sentence includes a citation [2], which implies a source of evidence. Since the summary sentence omits the citation, it lacks the evidentiary support present in the source. Without a clear indication of uncertainty in either sentence, and given that both sentences are identical, the certainty strength is not clearly distinguishable. Therefore, the overall certainty is UNCERTAIN due to the absence of explicit uncertainty cues and the lack of a valid distinction in certainty strength between the two. No verbatim evidence quote can be provided that expresses uncertainty in either sentence, as no such expression exists.

**Validation:** `review_required`; reasons: `llm_requires_review;certainty_label_direction_inconsistent;invalid_certainty_direction;invalid_source_evidence_schema;invalid_summary_evidence_schema;detected_number_difference`.

### RW-F50FBAE8EA173169 (`RW-F50FBAE8EA173169`)

Document: `PMC13757175`; location: Methods; paragraph 176; sentence 1 → Abstract; paragraph 2; sentence 2.

**Source:** For each patient, a standardized case file was created, including sociodemographic variables (age, gender, and smoking status), detailed medical and dental history, full-mouth periodontal charting, orthopantomogram (OPG), and intraoral photographs.

**Summary:** Comprehensive clinical and radiographic records were obtained for each patient.

**Alignment:** lexical `0.068966`; semantic `0.739855` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `comparable` (flags subject/negation/numbers/dates: `true` / `true` / `true` / `true`). Both sentences convey the collection of detailed clinical and radiographic data for each patient, with the source specifying the components (sociodemographic variables, medical/dental history, periodontal charting, OPG, intraoral photos) and the summary generalizing them as comprehensive clinical and radiographic records. The scope and subject (each patient) are consistent, and the core proposition of obtaining detailed records is preserved.

**Stage A quoted propositions:** source 'For each patient, a standardized case file was created, including sociodemographic variables (age, gender, and smoking status), detailed medical and dental history, full-mouth periodontal charting, orthopantomogram (OPG), and intraoral photographs.'; summary 'Comprehensive clinical and radiographic records were obtained for each patient.'.

**Stage B:** label `PRESERVED`; certainty `ASSERTED` → `ASSERTED`; direction `PRESERVED`; annotation status `requires_review`; review reasons `invalid_source_evidence.

**Stage B source evidence:** quote: 'a standardized case file was created, including sociodemographic variables (age, gender, and smoking status), detailed medical and dental history, full-mouth periodonting charting, orthopantomogram (OPG), and intraoral photographs'; type: asserted_predicate; interpretation: a standardized case file was created, including sociodemographic variables (age, gender, and smoking status), detailed medical and dental history, full-mouth periodontal charting, orthopantomogram (OPG), and intraoral photographs

**Stage B summary evidence:** quote: 'Comprehensive clinical and radiographic records were obtained for each patient'; type: asserted_predicate; interpretation: Comprehensive clinical and radiographic records were obtained for each patient

**Stage B explanation:** The source sentence specifies that a standardized case file was created with detailed medical and dental history, full-mouth periodontal charting, OPG, and intraoral photographs, which collectively constitute comprehensive clinical and radiographic records. The summary sentence generalizes this by stating that comprehensive clinical and radiographic records were obtained, which is fully supported by the detailed components in the source sentence. There is no reduction or weakening of certainty; the summary is a more general but accurate reflection of the source, preserving the asserted strength of the claim.

**Validation:** `review_required`; reasons: `llm_requires_review;invalid_source_evidence;invalid_source_evidence_schema;invalid_summary_evidence_schema`.


## 2. All Stage A non-comparable pairs

Stage B was not run for these pairs. In the descriptions below, `EXCLUDE` is the review disposition, not a model prediction class. The sentence-level inspection makes the non-comparability decisions plausible, but does not substitute for human review.



### RW-F94DADEC55ACD798 (`RW-F94DADEC55ACD798`)

Document: `PMC13754737`; location: Results; paragraph 51; sentence 2 → Abstract; paragraph 3; sentence 2.

**Source:** The clinical core achieved an AUROC of 0.661 (95% CI, 0.605-0.715) and an AUPRC of 0.126 (95% CI, 0.096-0.176).

**Summary:** In independent external validation, AUROC was 0.806 (95% CI, 0.769-0.840) and AUPRC was 0.258 (95% CI, 0.193-0.337), compared with 0.661 and 0.126, respectively, for the clinical core.

**Alignment:** lexical `0.400000`; semantic `0.887141` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `non_comparable` (flags subject/negation/numbers/dates: `false` / `false` / `false` / `false`). The source sentence reports AUROC and AUPRC values for the clinical core, while the summary sentence compares these values to external validation results, introducing a new comparison point (external validation) not present in the source. The claim, subject, and scope differ materially.

**Stage A quoted propositions:** source 'The clinical core achieved an AUROC of 0.661 (95% CI, 0.605-0.715) and an AUPRC of 0.126 (95% CI, 0.096-0.176).'; summary 'In independent external validation, AUROC was 0.806 (95% CI, 0.769-0.840) and AUPRC was 0.258 (95% CI, 0.193-0.337), compared with 0.661 and 0.126, respectively, for the clinical core.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `excluded`; reasons: `detected_number_difference`.

### RW-494700B95B6DE2FD (`RW-494700B95B6DE2FD`)

Document: `PMC13756842`; location: Background; paragraph 1; sentence 1 → Abstract; paragraph 1; sentence 1.

**Source:** Fabry disease (FD) is a rare X-linked lysosomal storage disorder (LSD) characterised by progressive involvement of mainly the kidneys, heart and cerebrovascular system.

**Summary:** Fabry disease (FD) is a rare X-linked lysosomal storage disorder caused by deficient activity of the enzyme α-galactosidase A, resulting in glycosphingolipid accumulation and progressive involvement of multiple organ systems, particularly the kidneys, heart and cerebrovascular system.

**Alignment:** lexical `0.464286`; semantic `0.939704` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `non_comparable` (flags subject/negation/numbers/dates: `true` / `false` / `false` / `false`). The source sentence states that Fabry disease is characterized by progressive involvement of mainly the kidneys, heart and cerebrovascular system. The summary sentence adds that it is caused by deficient activity of the enzyme α-galactosidase A, resulting in glycosphingolipid accumulation and progressive involvement of multiple organ systems, particularly the kidneys, heart and cerebrovascular system. The addition of a causal mechanism and the expansion of organ systems from 'mainly' to 'multiple' introduces a material change in scope and content, altering the underlying proposition.

**Stage A quoted propositions:** source 'Fabry disease (FD) is a rare X-linked lysosomal storage disorder (LSD) characterised by progressive involvement of mainly the kidneys, heart and cerebrovascular system.'; summary 'Fabry disease (FD) is a rare X-linked lysosomal storage disorder caused by deficient activity of the enzyme α-galactosidase A, resulting in glycosphingolipid accumulation and progressive involvement of multiple organ systems, particularly the kidneys,  heart and cerebrovascular system.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `excluded`; reasons: `llm_exclude`.

### RW-E09835C50E0E7105 (`RW-E09835C50E0E7105`)

Document: `PMC13756906`; location: Discussion; paragraph 182; sentence 6 → Abstract; paragraph 3; sentence 5.

**Source:** Considerable heterogeneity was also observed in outcome definitions and reported performance metrics.

**Summary:** However, substantial heterogeneity was observed in AI methodologies, target outcomes, validation strategies, and reported performance metrics.

**Alignment:** lexical `0.312500`; semantic `0.793202` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `non_comparable` (flags subject/negation/numbers/dates: `false` / `false` / `false` / `false`). The source sentence discusses heterogeneity in outcome definitions and reported performance metrics, while the summary sentence expands this to include heterogeneity in AI methodologies, target outcomes, validation strategies, and reported performance metrics. The scope and claim are materially different due to the addition of new elements (AI methodologies, validation strategies, target outcomes), indicating a change in the underlying proposition.

**Stage A quoted propositions:** source 'Considerable heterogeneity was also observed in outcome definitions and reported performance metrics.'; summary 'However, substantial heterogeneity was observed in AI methodologies, target outcomes, validation strategies, and reported performance metrics.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `excluded`; reasons: `llm_exclude`.

### RW-E9D992C93293A9FA (`RW-E9D992C93293A9FA`)

Document: `PMC13757175`; location: Methods; paragraph 11; sentence 2 → Abstract; paragraph 5; sentence 1.

**Source:** These cases were not included in the main study and were selected from the clinic archive.

**Summary:** The online version contains supplementary material available at https://doi.org/10.1186/s12903-026-09560-4.

**Alignment:** lexical `0.000000`; semantic `0.560996` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `non_comparable` (flags subject/negation/numbers/dates: `false` / `false` / `false` / `false`). The source sentence discusses the exclusion and selection of cases from a clinic archive, while the summary sentence refers to supplementary material in an online version. There is no shared proposition about cases, study inclusion, or archival selection. The claims, subjects, and scope are entirely different.

**Stage A quoted propositions:** source 'These cases were not included in the main study and were selected from the clinic archive.'; summary 'The online version contains supplementary material available at https://doi.org/10.1186/s12903-026-09560-4.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `excluded`; reasons: `detected_number_difference;detected_negation_difference`.


## 3. All eight Stage A annotation errors

Each of these rows records `annotation_error: Ollama request timed out after 90s.` The pipeline marked each uncertain and did not run Stage B. The explanation field therefore contains the timeout rather than a proposition analysis; absence of an annotation is not a classification error about certainty. The additional validator flags shown for some rows are independent deterministic checks and remain review flags.



### RW-84D7714903A8324D (`RW-84D7714903A8324D`)

Document: `PMC13754737`; location: Discussion; paragraph 84; sentence 1 → Abstract; paragraph 1; sentence 1.

**Source:** Large negative discordance between creatinine- and cystatin C-based eGFR is increasingly recognized as clinically relevant.

**Summary:** Large negative discordance between cystatin C- and creatinine-based estimated glomerular filtration rate (eGFR) may reveal clinically relevant renal-function differences that are not apparent from creatinine alone.

**Alignment:** lexical `0.347826`; semantic `0.879661` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_negation_difference`.

### RW-1949BA73B6A12B8E (`RW-1949BA73B6A12B8E`)

Document: `PMC13755501`; location: Conclusion; paragraph 97; sentence 1 → Abstract; paragraph 1; sentence 2.

**Source:** This study systematically identified, appraised, and synthesized evidence on the use of the OEP for fall prevention in adults after stroke.

**Summary:** This evidence summary aimed to identify, appraise, and synthesize the best available evidence on the OEP for fall prevention in adults after stroke.

**Alignment:** lexical `0.350000`; semantic `0.945812` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-640B2E85791258E4 (`RW-640B2E85791258E4`)

Document: `PMC13755501`; location: Methods; paragraph 4; sentence 1 → Abstract; paragraph 2; sentence 1.

**Source:** This study was conducted as a systematic, implementation‐oriented, evidence summary based on the 6S evidence model and the Joanna Briggs Institute (JBI) evidence appraisal approach.

**Summary:** A systematic, implementation‐oriented evidence summary guided by the 6S evidence model and the Joanna Briggs Institute evidence appraisal methods.

**Alignment:** lexical `0.588235`; semantic `0.947273` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-DB6D467069132BC8 (`RW-DB6D467069132BC8`)

Document: `PMC13756842`; location: Background; paragraph 4; sentence 3 → Abstract; paragraph 1; sentence 9.

**Source:** Advances in molecular diagnostics, biomarkers and cardiac imaging have contributed to more individualised approaches to treatment initiation, monitoring and risk stratification [2, 7–9].

**Summary:** Advances in biomarkers, molecular diagnostics and cardiac imaging have further refined approaches to disease assessment and treatment decision-making.

**Alignment:** lexical `0.421053`; semantic `0.908302` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference`.

### RW-5A74DC137DF2A226 (`RW-5A74DC137DF2A226`)

Document: `PMC13756842`; location: Conclusion; paragraph 87; sentence 2 → Abstract; paragraph 1; sentence 11.

**Source:** An adaptive framework should link resource-stratified diagnostic and monitoring pathways with transparent treatment prioritisation, multidisciplinary reassessment and prospective local evidence generation.

**Summary:** It proposes an adaptive, resource-stratified implementation framework that integrates evolving international recommendations with transparent treatment prioritisation, longitudinal monitoring, local real-world evidence generation and periodic reassessment.

**Alignment:** lexical `0.440000`; semantic `0.869882` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-14315C93BE8FCCCC (`RW-14315C93BE8FCCCC`)

Document: `PMC13756842`; location: Conclusion; paragraph 87; sentence 3 → Abstract; paragraph 1; sentence 12.

**Source:** In particular, decisions concerning high-cost disease-specific therapy should consider expected benefit, uncertainty, treatment burden, equity, affordability and opportunity cost.

**Summary:** The framework recognises that treatment effects differ by phenotype, sex, organ involvement and disease stage, and that decisions concerning high-cost disease-specific therapies must consider expected benefit, uncertainty, equity, affordability and opportunity cost.

**Alignment:** lexical `0.464286`; semantic `0.860257` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-C8B9B93755394EF9 (`RW-C8B9B93755394EF9`)

Document: `PMC13756906`; location: Introduction; paragraph 6; sentence 2 → Abstract; paragraph 1; sentence 3.

**Source:** Therefore, this systematic review aimed to evaluate the current evidence regarding the use of AI-based approaches for precision dosing of narrow therapeutic index drugs, particularly their predictive performance and applications across different therapeutic areas.

**Summary:** This systematic review evaluated the predictive performance and applications of AI-based approaches for precision dosing in narrow therapeutic index drugs.

**Alignment:** lexical `0.520000`; semantic `0.979416` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-5251D3215676EE9C (`RW-5251D3215676EE9C`)

Document: `PMC13757175`; location: Discussion; paragraph 404; sentence 2 → Abstract; paragraph 4; sentence 1.

**Source:** However, differentiation between Stage III and Stage IV remains a clinically challenging area that may benefit from structured calibration, decision-support tools, and targeted postgraduate education.

**Summary:** Risk-based components of the 2018 Classification demonstrate high reproducibility, whereas staging, particularly differentiation between Stage III and Stage IV, remains challenging.

**Alignment:** lexical `0.230769`; semantic `0.755137` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). annotation_error: Ollama request timed out after 90s.

**Stage A quoted propositions:** source ''; summary ''.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference`.


## 4. Ten representative uncertain pairs

These examples cover invalid proposition quotes, subject/scope uncertainty, uncertain propositions with otherwise coherent explanations, and number/negation flags. Stage B was correctly skipped for every row because Stage A did not return `comparable`. Stage B labels, certainty levels, and certainty evidence are therefore unavailable; the Stage A proposition excerpts below are not certainty evidence.



### RW-FD0B2442458DCE12 (`RW-FD0B2442458DCE12`)

Document: `PMC13754737`; location: Methods; paragraph 5; sentence 1 → Abstract; paragraph 2; sentence 1.

**Source:** This retrospective two-center study included adults aged ≥18 years with a recorded clinical diagnosis of diabetes and paired serum creatinine and cystatin C measurements.

**Summary:** We conducted a retrospective two-center study of adults with diabetes.

**Alignment:** lexical `0.294118`; semantic `0.828736` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). Stage A proposition evidence was empty or not an exact input quote.

**Stage A quoted propositions:** source 'A retrospective two-center study included adults aged ≥18 years with a recorded clinical diagnosis of diabetes and paired serum creatinine and cystatin C measurements.'; summary 'We conducted a retrospective two-center study of adults with diabetes.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference`.

### RW-9DE5F5825A40298E (`RW-9DE5F5825A40298E`)

Document: `PMC13754737`; location: Methods; paragraph 7; sentence 3 → Abstract; paragraph 2; sentence 2.

**Source:** The primary phenotype (D2) was defined as eGFRcys/eGFRcr <0.70, corresponding to eGFRcys being more than 30% lower than eGFRcr.

**Summary:** Large negative eGFR discordance (D2) was defined as eGFRcys/eGFRcr <0.70.

**Alignment:** lexical `0.230769`; semantic `0.822521` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `true` / `true` / `true` / `true`). Stage A proposition evidence was empty or not an exact input quote.

**Stage A quoted propositions:** source 'D2 is defined as eGFRcys/eGFRcr < 0.70'; summary 'D2 is defined as eGFRcys/eGFRcr < 0-70'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference`.

### RW-4C21A011CB878E13 (`RW-4C21A011CB878E13`)

Document: `PMC13754737`; location: Methods; paragraph 13; sentence 1 → Abstract; paragraph 2; sentence 3.

**Source:** The two leading algorithms were subjected to SHAP-guided feature reduction (21).

**Summary:** Seven machine-learning algorithms were evaluated in a development cohort of 1,170 participants (214 D2-positive), followed by SHAP-guided feature reduction.

**Alignment:** lexical `0.266667`; semantic `0.791415` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). Stage A proposition evidence was empty or not an exact input quote.

**Stage A quoted propositions:** source 'The two leading algorithms were subjected to SHAP-guided feature reduction.'; summary 'Seven machine-learning algorithms were evaluated in a development cohort of 1,170 participants (214 D2-positive), followed by SHAP-guided feature reduction.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference`.

### RW-8D9E904E1FDF2296 (`RW-8D9E904E1FDF2296`)

Document: `PMC13754737`; location: Methods; paragraph 15; sentence 2 → Abstract; paragraph 2; sentence 5.

**Source:** A clinical core comparator was prespecified as logistic regression using age, sex, and eGFRcr.

**Summary:** Performance was compared with a prespecified clinical core (age, sex, and eGFRcr); additional analyses evaluated logistic regression using eGFRcr alone or with blood urea nitrogen and albumin.

**Alignment:** lexical `0.450000`; semantic `0.862674` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `false` / `false` / `false`). Stage A proposition evidence was empty or not an exact input quote.

**Stage A quoted propositions:** source 'A clinical core comparator was prespecified as logistic regression using age, sex, and eGFRcr.'; summary 'Performance was compared with a prespecified clinical core (age, sex, and eG-than eGFRcr); additional analyses evaluated logistic regression using eGFRcr alone or with blood urea nitrogen and albumin.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-F3D6BC1F758D5CFC (`RW-F3D6BC1F758D5CFC`)

Document: `PMC13754737`; location: Discussion; paragraph 86; sentence 2 → Abstract; paragraph 1; sentence 2.

**Source:** When universal cystatin C measurement is impractical, ranking based on routine laboratory data may help identify patients with a higher likelihood of D2 for additional testing.

**Summary:** When universal cystatin C testing is impractical, a method to prioritize patients for additional testing may improve testing efficiency.

**Alignment:** lexical `0.333333`; semantic `0.863866` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `false` / `false` / `false`). The source sentence specifies a method (ranking based on routine laboratory data) that helps identify patients with a higher likelihood of D2 for additional testing, while the summary sentence refers to a general method to prioritize patients for additional testing without specifying the outcome (D2) or the basis of prioritization. The claim about 'impracticality' is shared, but the scope and specific content of the proposition differ, particularly regarding the outcome (D2) and the nature of the method. The subject and scope are similar but not identical, and the proposition is not fully preserved due to missing details.

**Stage A quoted propositions:** source 'When universal cystatin C measurement is impractical, ranking based on routine laboratory data may help identify patients with a higher likelihood of D2 for additional testing.'; summary 'When universal cystatin C testing is impractical, a method to prioritize patients for additional testing may improve testing efficiency.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-8342E30625124E8C (`RW-8342E30625124E8C`)

Document: `PMC13754737`; location: Introduction; paragraph 3; sentence 1 → Abstract; paragraph 1; sentence 2.

**Source:** When cystatin C testing capacity is constrained, a practical question is whom to test first.

**Summary:** When universal cystatin C testing is impractical, a method to prioritize patients for additional testing may improve testing efficiency.

**Alignment:** lexical `0.166667`; semantic `0.846891` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `false` / `false` / `false`). The source sentence discusses prioritization when testing capacity is constrained, while the summary sentence addresses prioritization when universal testing is impractical. The claims differ in scope and context: one focuses on capacity constraints, the other on impracticality of universal testing. The subject (testing priority) is similar, but the conditions and implications differ materially.

**Stage A quoted propositions:** source 'When cystatin C testing capacity is constrained, a practical question is whom to test first.'; summary 'When universal cystatin C testing is impractical, a method to prioritize patients for additional testing may improve testing efficiency.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-2793052F2E6340E8 (`RW-2793052F2E6340E8`)

Document: `PMC13754737`; location: Discussion; paragraph 83; sentence 1 → Abstract; paragraph 2; sentence 1.

**Source:** In this two-center study of adults with diabetes, a nine-predictor model based on routinely available demographic and laboratory data showed better discrimination and precision-recall performance for large negative eGFR discordance than an age-sex-eGFRcr clinical core in both the development and independent external validation cohorts.

**Summary:** We conducted a retrospective two-center study of adults with diabetes.

**Alignment:** lexical `0.121212`; semantic `0.774756` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `uncertain` / `uncertain` / `uncertain` / `uncertain`). The source sentence describes a specific study with a nine-predictor model and compares performance metrics between two models, while the summary sentence only states that a retrospective two-center study was conducted. The summary lacks detail on the model, performance, or comparison, and does not express the same underlying proposition in terms of claim, scope, or content.

**Stage A quoted propositions:** source 'In this two-center study of adults with diabetes, a nine-predictor model based on routinely available demographic and laboratory data showed better discrimination and precision-recall performance for large negative eGFR discordance than an age-sex-eGFRcr clinical core in both the development and independent external validation cohorts.'; summary 'We conducted a retrospective two-center study of adults with diabetes.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review`.

### RW-2AF3F26B21E57464 (`RW-2AF3F26B21E57464`)

Document: `PMC13754737`; location: Results; paragraph 60; sentence 4 → Abstract; paragraph 1; sentence 1.

**Source:** The eGFR difference is eGFRcys − eGFRcr. eGFRcr/eGFRcys, creatinine-/cystatin C-based estimated glomerular filtration rates.

**Summary:** Large negative discordance between cystatin C- and creatinine-based estimated glomerular filtration rate (eGFR) may reveal clinically relevant renal-function differences that are not apparent from creatinine alone.

**Alignment:** lexical `0.200000`; semantic `0.889411` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `false` / `false` / `false` / `false`). The source sentence provides a mathematical definition of eGFR difference and ratio, while the summary sentence discusses a clinical interpretation of discordance between eGFR methods. The claim, subject, and scope differ significantly: one is technical and definitional, the other is interpretive and clinical. The proposition is not the same.

**Stage A quoted propositions:** source 'The eGFR difference is eGFRcys − eGFRcr. eGFRcr/eGFRcys, creatinine-/cystatin C-based estimated glomerular filtration rates.'; summary 'Large negative discordance between cystatin C- and creatinine-based estimated glomerular filtration rate (eGFR) may reveal clinically relevant renal-function differences that are not apparent from creatinine alone.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_negation_difference`.

### RW-EBD3357F71DAB070 (`RW-EBD3357F71DAB070`)

Document: `PMC13754737`; location: Introduction; paragraph 2; sentence 4 → Abstract; paragraph 2; sentence 2.

**Source:** These findings provide a clinical rationale for identifying patients likely to have large negative eGFR discordance.

**Summary:** Large negative eGFR discordance (D2) was defined as eGFRcys/eGFRcr <0.70.

**Alignment:** lexical `0.285714`; semantic `0.822121` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `false` / `false` / `false` / `false`). The source sentence discusses a clinical rationale for identifying patients with large negative eGFR discordance, while the summary sentence defines what large negative eGFR discordance means. The source does not state or imply a definition, only a rationale. The abstract sentence introduces a definition, which is not present in the source. This introduces a material change in content: one is about a rationale, the other about a definition. The proposition is not the same.

**Stage A quoted propositions:** source 'These findings provide a clinical rationale for identifying patients likely to have large negative eGFR discordance.'; summary 'Large negative eGFR discordance (D2) was defined as eGFRcys/eGFRcr <0.70.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference`.

### RW-96977FAA423DE7D8 (`RW-96977FAA423DE7D8`)

Document: `PMC13754737`; location: Methods; paragraph 15; sentence 3 → Abstract; paragraph 2; sentence 5.

**Source:** Additional post-review comparator analyses used logistic regression with eGFRcr alone or eGFRcr, blood urea nitrogen, and albumin; these models were developed using development data and then applied without refitting to the external cohort (Supplementary Methods 4).

**Summary:** Performance was compared with a prespecified clinical core (age, sex, and eGFRcr); additional analyses evaluated logistic regression using eGFRcr alone or with blood urea nitrogen and albumin.

**Alignment:** lexical `0.333333`; semantic `0.849802` (nomic-embed-text:latest). These are retrieval/ranking scores only.

**Stage A:** `uncertain` (flags subject/negation/numbers/dates: `false` / `false` / `false` / `false`). Stage A proposition evidence was empty or not an exact input quote.

**Stage A quoted propositions:** source 'Additional post-review comparator analyses used logistic regression with eGFRcr alone or eGFRcr, blood urea nitrogen, and albumin; these models were developed using development data and then applied without refitting to the external cohort.'; summary 'Performance was compared with a prespecified clinical core (age, sex, and eGFRcr); additional analyses evaluated logistic regression using eGFRcr alone or with blood ure  nitrogen and albumin.'.

**Stage B:** not run (correctly gated because Stage A did not mark the pair comparable).

**Certainty levels and Stage B evidence:** not assessed.

**Validation:** `review_required`; reasons: `llm_requires_review;detected_number_difference;detected_negation_difference`.


## 5. Number and negation validation examples

The cases referenced here are already detailed above, avoiding a second rendering of their sentence text.

### Numbers

- `RW-F94DADEC55ACD798` has different AUROC/AUPRC values for the clinical core and external validation. The Stage A explanation also identifies a scope change; the numeric check is consistent with a real quantitative difference, though the pair should be excluded because the propositions differ, not because a generic numeric threshold proves that.
- `RW-E9D992C93293A9FA` is an unrelated pair: a sentence about cases not included in a study is paired with a sentence about an online supplement. The summary DOI contains digits, so the number check fires on metadata embedded in text. This is a clear candidate retrieval failure; numeric disagreement is not useful evidence about a shared claim.
- `RW-9DE5F5825A40298E` compares `eGFRcys/eGFRcr <0.70` with the same threshold in the abstract, while the source also spells out the equivalent “more than 30% lower” interpretation. The validator flags a number difference; this looks like a possible representation/omission false positive, but the malformed Stage A quote means the pair still lacks a sound comparability assessment.
- `RW-FD0B2442458DCE12` includes `aged ≥18 years` in the source but omits the age restriction in the summary. That number/scope loss is plausibly material and warrants review; it should not be auto-accepted as equivalent.
- `RW-B7A737B70D2ACA8B` carries 42 and 30 additional comparator counts in the source that are omitted from the summary; the selected XGBoost-versus-three-variable comparison and recall interval are retained. The broad number check appropriately prompts inspection, but whether the omissions change the target proposition needs human adjudication.
- `RW-74599612416467FF` is otherwise textually identical apart from citation `[2]`; the citation number appears to trigger the count mismatch. This is likely metadata rather than a changed numerical claim.

### Negation

- `RW-E9D992C93293A9FA` has “not included” in the source but no corresponding proposition in the summary. The negation flag reflects token presence across an unrelated pair, not polarity reversal of a matched claim.
- `RW-84D7714903A8324D` has the phrase “negative discordance” in the source and “not apparent from creatinine alone” in the summary. A token-level check may conflate the technical adjective “negative” with sentence negation. The Stage A call timed out, so this pair remains unresolved.
- `RW-96977FAA423DE7D8` says models were applied “without refitting” in the source; the summary discusses comparator analyses but omits that detail and appears to pair a different abstract sentence. Here negation could mark a material method detail, but the Stage A invalid quote and unrelated scope make this insufficient to conclude a true polarity change.
- `RW-7A903C45654DDCE3` contrasts source text that says available evidence is “not yet sufficient” for routine implementation with summary text saying AI approaches “may improve” individualized dosing. The negation has substantive scope, yet the two sentences also make different claims. Review is appropriate; a token mismatch alone cannot resolve the proposition.

No date-mismatch flags occurred. The examples show that the deterministic checks are useful alarms but can respond to citations, DOIs, technical vocabulary, and differences between propositions. They are not evidence that a sentence pair is comparable or non-comparable by themselves.

## 6. Failure diagnosis

### Candidate retrieval failure

The extractor ranks same-document body/abstract sentence pairs by embedding cosine and emits up to three candidates for each abstract sentence. High semantic scores do not ensure that the sentences state the same proposition: all four Stage A non-comparable rows have semantic scores from about 0.56 to 0.94. `RW-E9D992C93293A9FA` is especially diagnostic: lexical score 0.000 and semantic score 0.560996, but the source is about excluded archive cases and the abstract sentence is a DOI/supplement notice. This is a retrieval/ranking failure, not an embedding-equivalence decision—the system correctly keeps Stage A as the equivalence gate. Multiple uncertain candidates similarly share topic terms while differing in claim, scope, or method.

### Stage A annotation and evidence

The 8 timeouts are operational annotation failures and correctly abstained. A further 84 Stage A outputs were demoted to uncertain because proposition evidence was not an exact, nonempty quote; examples include `RW-FD0B2442458DCE12` (the source quote begins “A retrospective…” but the source says “This retrospective…”) and `RW-9DE5F5825A40298E` (the quoted threshold is corrupted as `< 0-70`). This is a failure to satisfy the evidence contract, even when the underlying pair may be related. Four other rows had `subject_match=uncertain`; that conflict was conservatively converted to uncertain.

Many remaining Stage A reasons explicitly describe omitted details or different scopes. That is not automatically a wrong answer: for a certainty-shift task, abstaining on non-equivalent or underspecified pairs is the correct safe behavior. No human labels exist to determine whether each abstention was necessary.

### Stage B certainty, evidence, and consistency

All four Stage B assessments are review-required. Two are internally inconsistent:

- `RW-47DD26BF6C9B5C5F`: source and summary certainty are both `ASSERTED`, but the label is `SOFTENED` and direction is `UNCERTAIN`. The explanation itself says the strengths are equal and no directional change is shown. This is a certainty/label failure and the correct pipeline action is abstention.
- `RW-74599612416467FF`: both strengths are `ASSERTED` and the label is `PRESERVED`, but direction is `UNCERTAIN`; the explanation alternates between preserved certainty and no distinguishable direction. This is contradictory output and must remain review-required.

The other two have `ASSERTED`/`ASSERTED` and `PRESERVED`, but Stage B evidence is not fully supported by the sentence text. In `RW-B7A737B70D2ACA8B`, the source evidence quote is not an exact substring because it omits the leading “At the same 20% allocation externally,”. In `RW-F50FBAE8EA173169`, the source quote contains “periodonting charting,” while the sentence says “periodontal charting.” These are evidence failures, not proof that the labels are wrong. With no explicit uncertainty cue, the model’s use of `asserted_predicate` may be reasonable, but it is not human-verified.

### Validation integration issue

The Stage B rows store `annotation_version=hedgeguard-real-world-v2`, while the validator’s staged-evidence branch checks for the literal value `staged-v2`. As a result, the validator treats the Stage B evidence JSON objects as legacy list-shaped evidence and adds `invalid_source_evidence_schema` and `invalid_summary_evidence_schema` to all four rows. The raw quote validator in Stage B still caught the two actual nonexact source quotes described above. This version-tag mismatch is a distinct implementation/configuration issue; fixing it would remove schema-misclassification flags, but would not resolve the two certainty contradictions, the two invalid source quotes, or any other review flags.

### Deterministic numeric and negation checks

The report records 78 numeric-difference flags and 12 negation-difference flags, with no date flags. Some are useful alarms for omitted or changed quantities (for example, age restrictions or performance measures). Others are likely over-broad: citation numbers, DOI digits, equivalent ratio/percentage formulations, and additional comparator statistics omitted from an otherwise related comparison. Negation flags also do not model scope: “negative” in “negative discordance” is not the same as a negating operator, and `not` in a sentence cannot be compared meaningfully with no `not` in an unrelated sentence.

The pipeline sends these cases to review rather than auto-accepting them. The evidence does not show that the rules caused any false training acceptance; instead, they add conservative review flags, sometimes redundantly after Stage A already marks a pair non-comparable or uncertain.

## 7. Smallest recommended changes (not implemented)

1. **Align the annotation-version contract.** Use one shared version identifier or a schema/version enum so Stage B dict evidence is validated by the staged-v2 branch. Add a focused test using an actual annotation-version value emitted by the annotation script.
2. **Keep timeouts as abstentions, but make long runs observable and resumable.** Record per-call duration and timeout status; do not convert timeout rows into labels. The existing ID-based resume behavior should remain.
3. **Tighten retrieval before Stage A without using a cosine threshold as equivalence.** Filter obvious non-proposition material such as figure captions, citation-only/URL notices, and fragments; diversify top-ranked candidates so near-duplicate abstract/body sentences do not crowd out other useful pairs. Keep Stage A as the independent proposition decision.
4. **Retain exact-quote validation and improve prompt examples for copyable spans.** Ask the model to copy short verbatim spans, avoid paraphrased propositions where the schema requires an exact quote, and preserve abstention on malformed quotes. Do not relax exact matching.
5. **Make numeric comparison proposition-aware but conservative.** Distinguish citation/DOI/year metadata from claim quantities; recognize equivalent representations such as ratio cutoffs and equivalent percentages; keep conflicting or unaligned quantities in review. Do not let a deterministic match establish eligibility.
6. **Compare negation in context.** Distinguish negative-valued domain terms from negation and compare polarity only within a Stage-A-aligned proposition. If scope cannot be established, keep review-required.
7. **Continue requiring human audit before treating any v2 label as trusted.** The v2 report has 0 eligible rows and no human-reviewed labels; this pilot cannot support model accuracy or generalization claims.

These recommendations are not performance claims. The observed results identify operational, retrieval, evidence-contract, consistency, and validation-integration issues; they do not establish that the v2 pipeline improved or that any Qwen class label is correct.

## Files inspected

- `data/real_world/candidate_pairs_v2.csv`
- `data/real_world/proposition_assessments_v2.csv`
- `data/real_world/certainty_assessments_v2.csv`
- `data/real_world/validation_report_v2.json`
- `data/real_world/eligible_pairs_v2.csv`
- `data/real_world/excluded_pairs_v2.csv`
- `data/real_world/review_required_v2.csv`
- `data/real_world/validation_rejected_v2.csv`
- `src/dataset/real_world.py` and `scripts/annotate_real_pairs.py` (read only, to verify the evidence-version contract)
