# Manual review workflow

`data/processed/experimental_candidates_v3.csv` is the immutable candidate source for this review round. The workflow writes a separate editable file and never changes the experimental source.

## Create the review copy

From the HedgeGuard project directory in PowerShell:

```powershell
python scripts\review_dataset.py --prepare `
  --input data\processed\experimental_candidates_v3.csv `
  --output data\processed\reviewed_candidates.csv
```

Every copied record starts with `review_status=requires_manual_review` and `include_in_training=false`. Existing `source_group_id`, source and summary text, labels, exclusion reasons, and annotation fields are retained. `original_label` records the candidate label so relabeling can be audited.

## Record decisions

Edit only `reviewed_candidates.csv`. Supported statuses:

| `review_status` | Reviewer action | `include_in_training` |
|---|---|---|
| `requires_manual_review` | No completed decision yet | `false` |
| `approved` | Accept the existing `original_label` | `true` only for PRESERVED/HARDENED/SOFTENED |
| `relabeled` | Change `label` from `original_label` to a reviewed model label | `true` |
| `excluded` | Set `label=EXCLUDE` and document why | `false` |

For completed decisions, enter `reviewer_id`. Add `review_notes` for rationale; exclusions require `review_notes` or an `exclude_reason`. Never approve an `EXCLUDE` label. A row still pending review must remain excluded from training.

The Python helper `apply_review_decision(row, status, reviewer_id, label=..., notes=...)` sets the status and training flag consistently. Validation rejects invalid statuses, missing attribution, unreviewed inclusion, class labels marked excluded, and any `EXCLUDE` row enabled for training.

## Validate the edited review file

```powershell
python scripts\review_dataset.py --validate `
  --reviewed data\processed\reviewed_candidates.csv
```

The command reports review counts and the number of eligible reviewed rows. It does not train a model or create data splits. Future training code must obtain examples through `training_rows()` in `src/dataset/pipeline.py`; that function fails closed on invalid review metadata and returns only explicitly enabled approved or relabeled model labels.
