# Annotation guide

Annotate matched source-summary claims, not overall factual correctness.

1. Verify both sentences discuss the same claim, actor, polarity, conditions, and relevant values. If not, use `EXCLUDE`.
2. Determine whether the source expresses epistemic uncertainty (uncertainty about truth/outcome). Permission uses of “may” are not epistemic uncertainty.
3. Compare uncertainty strength: same → `PRESERVED`; summary more definite → `HARDENED`; summary more tentative → `SOFTENED`.
4. Use `EXCLUDE` for negation/subject/condition/number changes, contradiction, non-matching claims, or otherwise invalid comparisons. Add a concise reason in `annotation_notes`.
5. Do not label a sentence as uncertain merely because it is negative, subjective, or emotionally toned.

Add transformations as explicit source-summary rows, with a shared `source_group_id` for variations of one underlying claim. Do not assume hedge terms are interchangeable in every context. Review every synthetic example before training. Keep independently sourced, manually reviewed evaluation examples separate from transformed training families.
