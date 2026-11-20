"""Validate versioned candidates with separate proposition and certainty assessments."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import REAL_WORLD_V2_FIELDS, read_real_world_csv, validate_real_world_rows, write_real_world_csv

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"real_world"
PROPOSITION_FIELDS=("example_id","comparability_status","proposition_comparable","source_proposition","summary_proposition","subject_match","negation_match","numbers_match","dates_match","reason","annotation_method","model_name","model_version","prompt_version","annotation_version","annotation_stage","annotation_status","requires_review","annotation_raw_json")
CERTAINTY_FIELDS=("example_id","label","source_certainty","summary_certainty","certainty_direction","source_evidence","summary_evidence","explanation","requires_review","review_reasons","annotation_method","model_name","model_version","prompt_version","annotation_version","annotation_stage","annotation_status","annotation_raw_json")

def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--candidates",type=Path,default=DATA/"candidate_pairs_v2.csv"); p.add_argument("--propositions",type=Path,default=DATA/"proposition_assessments_v2.csv"); p.add_argument("--certainty",type=Path,default=DATA/"certainty_assessments_v2.csv"); p.add_argument("--output-dir",type=Path,default=DATA); p.add_argument("--output-suffix",default="_final",help="Suffix for derived output filenames (default: _final); never overwrites the historical v2 outputs."); p.add_argument("--training",type=Path,default=ROOT/"data"/"processed"/"training_dataset.csv"); p.add_argument("--resume",action="store_true",help="Explicitly allow replacing outputs with this exact suffix."); a=p.parse_args()
 resolve=lambda x:x if x.is_absolute() else ROOT/x
 if not a.output_suffix or any(char in a.output_suffix for char in "/\\"):
  p.error("--output-suffix must be a non-empty filename suffix without path separators")
 outdir=resolve(a.output_dir); suffix=a.output_suffix; paths={"eligible":outdir/f"eligible_pairs{suffix}.csv","excluded":outdir/f"excluded_pairs{suffix}.csv","review":outdir/f"review_required{suffix}.csv","rejected":outdir/f"validation_rejected{suffix}.csv","report":outdir/f"validation_report{suffix}.json"}
 if not a.resume and any(path.exists() for path in paths.values()): print(f"Refusing to overwrite existing derived outputs for suffix {suffix!r}; choose another suffix or pass --resume.",file=sys.stderr); return 2
 candidates=read_real_world_csv(resolve(a.candidates),fields=REAL_WORLD_V2_FIELDS)
 propositions=read_real_world_csv(resolve(a.propositions),fields=PROPOSITION_FIELDS); props={r["example_id"]:r for r in propositions}
 certainties=read_real_world_csv(resolve(a.certainty),fields=CERTAINTY_FIELDS); certainty={r["example_id"]:r for r in certainties}
 merged=[]
 for candidate in candidates:
  row={field:candidate.get(field,"") for field in REAL_WORLD_V2_FIELDS}; prop=props.get(candidate["example_id"])
  if prop:
   row.update(prop); row["proposition_reason"]=prop.get("reason","")
   status=prop.get("comparability_status","")
   if status=="non_comparable": row.update(label="EXCLUDE",certainty_direction="EXCLUDE",annotation_status="requires_review",requires_review="true")
   elif status!="comparable": row.update(label="",annotation_status="requires_review",requires_review="true",proposition_comparable="uncertain")
   else:
    assessed=certainty.get(candidate["example_id"])
    if assessed: row.update(assessed)
    else: row.update(label="",annotation_status="not_annotated",requires_review="true")
  else: row.update(annotation_status="not_annotated",requires_review="true")
  merged.append(row)
 training_path=resolve(a.training)
 if training_path.exists():
  with training_path.open("r",encoding="utf-8-sig",newline="") as f: training=list(csv.DictReader(f))
 else: training=[]
 result=validate_real_world_rows(merged,training_records=training); outdir.mkdir(parents=True,exist_ok=True)
 for key,rows in (("eligible",result.eligible),("excluded",result.excluded),("review",result.review_required),("rejected",result.rejected)):
  name={"eligible":"eligible","excluded":"excluded","review":"review_required","rejected":"validation_rejected"}[key]
  enriched=[{**row,"pipeline_status":{"eligible":"eligible","excluded":"excluded","review":"review_required","rejected":"rejected"}[key],"rejection_reasons":";".join(result.row_reasons.get(row.get("example_id",""),()))} for row in rows]
  write_real_world_csv(paths[key],enriched,fields=REAL_WORLD_V2_FIELDS)
 classes={label:sum(r.get("label")==label for r in result.eligible) for label in ("PRESERVED","HARDENED","SOFTENED")}
 annotation_labels={label:sum(r.get("label","").upper()==label for r in merged) for label in ("PRESERVED","HARDENED","SOFTENED","EXCLUDE")}
 provenance_fields=("source_document_id","source_document_title","source_url","publication_date","domain","source_location","summary_location","collection_method","collection_date","license","license_url")
 flag_reasons=("detected_number_difference","number_comparison_uncertain","detected_date_difference","detected_negation_difference","negation_comparison_uncertain")
 report={"candidate_count":len(merged),"machine_label_eligible_count":len(result.eligible),"exclude_count":len(result.excluded),"review_required_count":len(result.review_required),"rejected_count":len(result.rejected),"annotation_label_counts":annotation_labels,"eligible_class_distribution":classes,"proposition_status_counts":{key:sum(r.get("comparability_status")==key for r in propositions) for key in ("comparable","non_comparable","uncertain")},"certainty_annotation_count":len(certainty),"provenance_complete_count":sum(not any(not r.get(field,"").strip() for field in provenance_fields) for r in merged),"numeric_negation_flag_counts":{reason:result.reason_counts.get(reason,0) for reason in flag_reasons},"reason_counts":result.reason_counts,"annotation_is_human_review":False,"input_files":{"candidates":str(resolve(a.candidates).relative_to(ROOT)),"propositions":str(resolve(a.propositions).relative_to(ROOT)),"certainty":str(resolve(a.certainty).relative_to(ROOT))},"output_suffix":suffix}
 paths["report"].write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
