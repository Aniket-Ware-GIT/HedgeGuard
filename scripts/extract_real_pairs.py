"""Extract same-document body/abstract pair candidates from licensed source checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.real_world import (  # noqa: E402
    LICENSE_ALLOWLIST, REAL_WORLD_V2_FIELDS, RealWorldDatasetError, read_real_world_csv,
    license_is_reusable, validate_real_world_rows, write_real_world_csv,
)
from src.dataset.real_world_sources import (  # noqa: E402
    Document, SourceCollectionError, _extract_arxiv_tex, extract_candidates, parse_pmc_oai_record,
)
from src.dataset.real_world_alignment import OllamaEmbeddingScorer, SemanticAlignmentError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "real_world"


def _source_bytes(row: dict[str, str]) -> tuple[bytes, Path]:
    raw_path = Path(row.get("local_source_path", ""))
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    resolved = raw_path.resolve()
    allowed_root = (DATA_ROOT / "raw").resolve()
    if allowed_root not in resolved.parents:
        raise RealWorldDatasetError(f"Source checkpoint path is outside the real-world raw directory: {raw_path}")
    content = resolved.read_bytes()
    expected = row.get("source_content_hash", "").strip()
    actual = hashlib.sha256(content).hexdigest()
    if expected and expected != actual:
        raise RealWorldDatasetError(f"Content hash mismatch for source document {row.get('source_document_id')}.")
    return content, resolved


def _document_from_manifest(row: dict[str, str]) -> Document:
    content, _path = _source_bytes(row)
    repository = row.get("repository", "")
    if repository.startswith("PubMed Central"):
        parsed = parse_pmc_oai_record(content, fallback_pmcid=row.get("source_document_id", ""),
                                      domain=row.get("domain", ""),
                                      collection_query=row.get("collection_query", ""))
        metadata = {**parsed.metadata, **row}
        return Document(metadata, parsed.abstract_paragraphs, parsed.body_paragraphs, content)
    if repository == "arXiv":
        body = _extract_arxiv_tex(content)
        abstract = row.get("abstract_text", "").strip()
        if not abstract:
            raise SourceCollectionError(f"arXiv document {row.get('source_document_id')} has no abstract text.")
        return Document(dict(row), (abstract,), (("Body", body),), content)
    raise SourceCollectionError(f"Unsupported repository {repository!r} for {row.get('source_document_id')}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DATA_ROOT / "documents.csv")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "candidate_pairs_v2.csv")
    parser.add_argument("--rejected", type=Path, default=DATA_ROOT / "rejected_candidates_v2.csv")
    parser.add_argument("--embedding-model", required=True,
                        help="Exact Ollama embedding model tag that is already installed; no model is pulled.")
    parser.add_argument("--minimum-semantic-score", type=float, default=0.25,
                        help="Retrieval-only cutoff; never treated as proposition equivalence or eligibility.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    scorer = OllamaEmbeddingScorer(args.embedding_model)
    try:
        scorer.verify_installed()
    except SemanticAlignmentError as exc:
        print(f"Semantic retrieval unavailable: {exc}", file=sys.stderr)
        return 2
    documents_path = args.documents if args.documents.is_absolute() else PROJECT_ROOT / args.documents
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    rejected_path = args.rejected if args.rejected.is_absolute() else PROJECT_ROOT / args.rejected
    if output.exists() and not args.resume:
        print(f"Refusing to overwrite {output}; use --resume to merge deterministically.", file=sys.stderr)
        return 2
    documents = read_real_world_csv(documents_path, fields=(
        "source_document_id", "source_document_title", "source_url", "publication_date", "domain",
        "repository", "license", "license_url", "collection_method", "collection_date",
        "source_version", "source_content_hash", "abstract_text", "local_source_path",
    ))
    existing = read_real_world_csv(output, fields=REAL_WORLD_V2_FIELDS) if output.exists() else []
    by_id = {row["example_id"]: row for row in existing}
    rejected = read_real_world_csv(rejected_path, fields=REAL_WORLD_V2_FIELDS) if rejected_path.exists() else []
    rejected_by_id = {row["example_id"]: row for row in rejected if row.get("example_id")}
    processed = 0
    try:
        for row in documents:
            if row.get("license") and row.get("license_url") and not license_is_reusable(
                row["license"], row["license_url"], LICENSE_ALLOWLIST
            ):
                raise RealWorldDatasetError(f"Non-reusable license found in document manifest: {row['source_document_id']}")
            document = _document_from_manifest(row)
            candidates = extract_candidates([document], scorer=scorer, top_k=args.top_k,
                                            minimum_semantic_score=args.minimum_semantic_score)
            for candidate in candidates:
                if candidate["example_id"] not in by_id:
                    by_id[candidate["example_id"]] = candidate
            processed += 1
    except (OSError, SourceCollectionError, RealWorldDatasetError, SemanticAlignmentError, ValueError) as exc:
        print(f"Extraction stopped after {processed} document(s): {exc}", file=sys.stderr)
        return 2

    rows = list(by_id.values())
    validation = validate_real_world_rows(rows)
    for row in validation.rejected:
        rejected_by_id[row.get("example_id", "")] = {
            **row,
            "rejection_reasons": ";".join(validation.row_reasons.get(row.get("example_id", ""), ())),
            "pipeline_status": "rejected",
        }
    write_real_world_csv(output, rows, fields=REAL_WORLD_V2_FIELDS)
    write_real_world_csv(rejected_path, rejected_by_id.values(), fields=REAL_WORLD_V2_FIELDS)
    print(f"Documents processed: {processed}; new candidates: {len(rows) - len(existing)}; total candidates: {len(rows)}")
    print(f"Candidate ranking uses same-document embedding cosine only for retrieval; Stage A must independently assess proposition comparability (top-k={args.top_k}, model={args.embedding_model}).")
    print(f"Preserved rejected candidate rows: {len(rejected_by_id)}; output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
