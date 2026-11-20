"""Select a reproducible human-audit sample, stored separately from Qwen annotations."""
from __future__ import annotations
import argparse, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import AUDIT_FIELDS, HUMAN_AUDIT_FIELDS, read_real_world_csv, write_real_world_csv
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"real_world"

def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--eligible",type=Path,default=DATA/"eligible_pairs.csv"); p.add_argument("--review",type=Path,default=DATA/"review_required.csv"); p.add_argument("--excluded",type=Path,default=DATA/"excluded_pairs.csv"); p.add_argument("--output",type=Path,default=DATA/"audit_sample.csv"); p.add_argument("--results",type=Path,default=DATA/"human_audit_results.csv"); p.add_argument("--seed",type=int,default=42); p.add_argument("--size",type=int,default=20); a=p.parse_args()
 resolve=lambda x:x if x.is_absolute() else ROOT/x; out=resolve(a.output)
 results_path=resolve(a.results)
 if out.exists() or results_path.exists(): print("Refusing to overwrite existing audit outputs.",file=sys.stderr); return 2
 pool={r["example_id"]:r for file in (a.eligible,a.review,a.excluded) for r in read_real_world_csv(resolve(file))
       if r.get("annotation_status") in {"complete", "requires_review"}}
 ordered=[pool[k] for k in sorted(pool)]; count=min(a.size,len(ordered)); selected=random.Random(a.seed).sample(ordered,count)
 rows=[]
 for i,r in enumerate(selected,1):
  rows.append({"audit_id":f"AUDIT-{i:03d}","example_id":r.get("example_id",""),"source_document_id":r.get("source_document_id",""),"source_document_title":r.get("source_document_title",""),"source_url":r.get("source_url",""),"source_location":r.get("source_location",""),"summary_location":r.get("summary_location",""),"source_sentence":r.get("source_sentence",""),"summary_sentence":r.get("summary_sentence",""),"llm_label":r.get("label",""),"llm_source_certainty":r.get("source_certainty",""),"llm_summary_certainty":r.get("summary_certainty",""),"llm_explanation":r.get("explanation",""),"llm_source_evidence":r.get("source_evidence",""),"llm_summary_evidence":r.get("summary_evidence",""),"human_label":"","auditor_id":"","audited_at":"","audit_notes":"","disagreement_reason":""})
 write_real_world_csv(out,rows,fields=AUDIT_FIELDS); write_real_world_csv(results_path,[],fields=HUMAN_AUDIT_FIELDS); print(f"Audit sample: {len(rows)} pairs (requested {a.size}), seed={a.seed}; human results are stored separately in {results_path}. {out}"); return 0
if __name__=="__main__": raise SystemExit(main())
