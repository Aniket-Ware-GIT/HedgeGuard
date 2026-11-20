"""Tests for additive manual-audit files, isolated from the frozen random sample."""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from src.dataset.real_world_audit import choose_hardened_case, write_hardened_audit_case

def test_hardened_case_selects_only_a_reviewable_existing_candidate():
 candidates=[{"example_id":"p","source_sentence":"preserved"},{"example_id":"h","source_sentence":"hardening"}]
 annotations=[{"example_id":"h","label":"HARDENED","source_evidence":"quote"}]
 candidate,annotation=choose_hardened_case(candidates,annotations,{"h"})
 assert candidate["example_id"]==annotation["example_id"]=="h"
 with pytest.raises(ValueError,match="No reviewable"):
  choose_hardened_case(candidates,annotations,set())

def test_hardened_audit_outputs_are_separate_and_refuse_overwrite(tmp_path):
 candidate={"example_id":"h","source_sentence":"source","summary_sentence":"summary"}
 annotation={"example_id":"h","label":"HARDENED","source_evidence":"source evidence"}
 audit=tmp_path/"audit_hardened_case.csv"; results=tmp_path/"human_audit_hardened_results.csv"
 frozen=tmp_path/"audit_sample.csv";frozen.write_bytes(b"frozen original audit sample\n");before=hashlib.sha256(frozen.read_bytes()).hexdigest()
 write_hardened_audit_case(candidate,annotation,audit_path=audit,results_path=results)
 assert hashlib.sha256(frozen.read_bytes()).hexdigest()==before
 assert "AUDIT-HARDENED-001" in audit.read_text(encoding="utf-8")
 assert "source evidence" in audit.read_text(encoding="utf-8")
 assert results.read_text(encoding="utf-8").count("\n")==1
 with pytest.raises(FileExistsError):
  write_hardened_audit_case(candidate,annotation,audit_path=audit,results_path=results)
