"""Collect a bounded set of reusable PMC or arXiv documents with provenance."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import DOCUMENT_FIELDS, RealWorldDatasetError, read_real_world_csv, write_real_world_csv  # noqa: E402
from src.dataset.real_world_sources import SourceCollectionError, search_arxiv, search_pmc  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "real_world"
DEFAULT_OUTPUT = DATA_ROOT / "documents.csv"
REJECTION_FIELDS = ("repository_record_id", "source_document_title", "source_url", "license", "license_url", "rejection_reason")


def _safe_filename(document_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", document_id).strip("._")
    if not safe:
        raise ValueError("Source returned an invalid document identifier.")
    return safe


def _write_rejections(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REJECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("pmc", "arxiv"), required=True)
    parser.add_argument("--query", required=True, help="Repository query; saved into the document manifest.")
    parser.add_argument("--domain", required=True, help="Domain/category label for this collection query.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum reusable documents (default: pilot-sized 5).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--raw-dir", type=Path, default=DATA_ROOT / "raw")
    parser.add_argument("--resume", action="store_true", help="Resume an existing real-world manifest by document ID.")
    parser.add_argument("--confirm-scale", action="store_true",
                        help="Required above five documents; use only after reviewing the pilot report.")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit > 5 and not args.confirm_scale:
        parser.error("More than 5 documents requires --confirm-scale after the pilot results have been reviewed.")
    if args.limit > 5:
        pilot_review = DATA_ROOT / "pilot_review.json"
        try:
            import json
            reviewed = json.loads(pilot_review.read_text(encoding="utf-8")).get("pilot_reviewed") is True
        except (OSError, ValueError):
            reviewed = False
        if not reviewed:
            parser.error("Scaling is locked until data/real_world/pilot_review.json contains {\"pilot_reviewed\": true} after human review.")

    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else PROJECT_ROOT / args.raw_dir
    if output.exists() and not args.resume:
        print(f"Refusing to overwrite existing manifest {output}; rerun with --resume.", file=sys.stderr)
        return 2
    existing = read_real_world_csv(output, fields=DOCUMENT_FIELDS) if output.exists() else []
    existing_by_id = {row["source_document_id"]: row for row in existing}
    skip_ids = set(existing_by_id)
    rejections: list[dict[str, str]] = []

    def save_checkpoint(document) -> None:
        document_id = document.metadata["source_document_id"]
        suffix = ".xml" if args.source == "pmc" else ".tar"
        raw_dir.mkdir(parents=True, exist_ok=True)
        content_path = raw_dir / (_safe_filename(document_id) + suffix)
        if content_path.exists():
            raise RealWorldDatasetError(f"Refusing to overwrite existing source file {content_path}.")
        content_path.write_bytes(document.content)
        document.metadata["local_source_path"] = str(content_path.relative_to(PROJECT_ROOT))
        existing_by_id[document_id] = {field: document.metadata.get(field, "") for field in DOCUMENT_FIELDS}
        write_real_world_csv(output, existing_by_id.values(), fields=DOCUMENT_FIELDS)

    def note_candidate(metadata: dict[str, str]) -> None:
        if metadata.get("rejection_reason"):
            rejections.append({field: str(metadata.get(field, "")) for field in REJECTION_FIELDS})

    try:
        collector = search_pmc if args.source == "pmc" else search_arxiv
        documents = collector(args.query, args.limit, domain=args.domain, skip_ids=skip_ids,
                              on_candidate=note_candidate, on_document=save_checkpoint)
    except (SourceCollectionError, OSError, ValueError, RealWorldDatasetError) as exc:
        _write_rejections(DATA_ROOT / "collection_rejections.csv", rejections)
        print(f"Collection stopped: {exc}", file=sys.stderr)
        print(f"Any completed document checkpoints remain in {output} and {raw_dir}.", file=sys.stderr)
        return 2
    _write_rejections(DATA_ROOT / "collection_rejections.csv", rejections)
    total = len(existing_by_id)
    print(f"New documents retrieved: {len(documents)}; reusable documents in manifest: {total}")
    print(f"License-filtered candidates recorded: {len(rejections)}")
    print(f"Manifest: {output}")
    print(f"Full-text checkpoints: {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
