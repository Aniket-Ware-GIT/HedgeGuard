"""Create a separate, human-only audit addendum for one existing HARDENED case."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import read_real_world_csv
from src.dataset.real_world_audit import choose_hardened_case, write_hardened_audit_case

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"real_world"

def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--candidates",type=Path,default=DATA/"candidate_pairs.csv"); p.add_argument("--annotations",type=Path,default=DATA/"annotations.csv"); p.add_argument("--review",type=Path,default=DATA/"review_required.csv"); p.add_argument("--audit-output",type=Path,default=DATA/"audit_hardened_case.csv"); p.add_argument("--results-output",type=Path,default=DATA/"human_audit_hardened_results.csv"); a=p.parse_args()
 resolve=lambda x:x if x.is_absolute() else ROOT/x
 candidates=read_real_world_csv(resolve(a.candidates)); annotations=read_real_world_csv(resolve(a.annotations),fields=("example_id","label","source_certainty","summary_certainty","source_evidence","summary_evidence","explanation")); review=read_real_world_csv(resolve(a.review));
 try:
  candidate,annotation=choose_hardened_case(candidates,annotations,{r["example_id"] for r in review})
  write_hardened_audit_case(candidate,annotation,audit_path=resolve(a.audit_output),results_path=resolve(a.results_output))
 except (ValueError,FileExistsError) as exc:
  print(f"Could not create HARDENED audit addendum: {exc}",file=sys.stderr); return 2
 print(f"Created separate HARDENED audit case for {candidate['example_id']}; human results file: {resolve(a.results_output)}")
 return 0
if __name__=="__main__": raise SystemExit(main())
