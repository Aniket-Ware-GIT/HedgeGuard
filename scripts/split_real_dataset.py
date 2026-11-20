"""Create deterministic group-disjoint train/validation/test files from eligible rows."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import GroupShuffleSplit
from src.dataset.real_world import RealWorldDatasetError, assert_group_disjoint, read_real_world_csv, write_real_world_csv
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/"data"/"real_world"

def main()->int:
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,default=DATA/"eligible_pairs.csv"); p.add_argument("--output-dir",type=Path,default=DATA/"splits"); p.add_argument("--seed",type=int,default=42); p.add_argument("--confirm-pilot-reviewed",action="store_true"); a=p.parse_args(); resolve=lambda x:x if x.is_absolute() else ROOT/x; out=resolve(a.output_dir)
 report=DATA/"pilot_review.json"
 if not a.confirm_pilot_reviewed or not report.exists(): print("Splitting is gated until the pilot report has been reviewed and --confirm-pilot-reviewed is supplied.",file=sys.stderr); return 2
 rows=read_real_world_csv(resolve(a.input)); groups=[r.get("source_group_id") or r.get("source_document_id") for r in rows]
 unique=set(groups)
 if len(unique)<4: print(f"Need at least 4 distinct source documents for group-disjoint 80/10/10 partitions; found {len(unique)}.",file=sys.stderr); return 2
 if out.exists() and any(out.iterdir()): print(f"Refusing to overwrite nonempty split directory {out}.",file=sys.stderr); return 2
 # First hold out 20% of groups for test, then 12.5% of the remainder for validation.
 first=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=a.seed); trainval_idx,test_idx=next(first.split(rows,groups=groups))
 trainval=[rows[i] for i in trainval_idx]; tv_groups=[groups[i] for i in trainval_idx]
 second=GroupShuffleSplit(n_splits=1,test_size=.125,random_state=a.seed+1); train_idx,val_idx=next(second.split(trainval,groups=tv_groups))
 parts={"train":[trainval[i] for i in train_idx],"validation":[trainval[i] for i in val_idx],"test":[rows[i] for i in test_idx]}
 try: assert_group_disjoint(parts)
 except RealWorldDatasetError as exc: print(str(exc),file=sys.stderr); return 2
 out.mkdir(parents=True,exist_ok=True)
 for name,part in parts.items():
  stamped=[{**row,"split":name,"split_seed":str(a.seed)} for row in part]
  write_real_world_csv(out/f"{name}.csv",stamped)
 print("Group-disjoint split sizes: "+", ".join(f"{k}={len(v)}" for k,v in parts.items())+f"; seed={a.seed}; groups={len(unique)}")
 return 0
if __name__=="__main__": raise SystemExit(main())
