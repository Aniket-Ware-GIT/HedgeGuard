"""Rate-limited PMC/arXiv collection and same-document sentence pairing."""
from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

from src.dataset.real_world import (
    LICENSE_ALLOWLIST, REAL_WORLD_FIELDS, REAL_WORLD_V2_FIELDS, DOCUMENT_FIELDS, license_is_reusable,
    stable_example_id, write_real_world_csv,
)
from src.dataset.real_world_alignment import PairScorer

PMC_OAI_URL = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
NCBI_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ARXIV_QUERY_URL = "https://export.arxiv.org/api/query"
ARXIV_OAI_URL = "https://export.arxiv.org/oai2"
ARXIV_EPRINT_URL = "https://export.arxiv.org/e-print/"
USER_AGENT = "HedgeGuardResearch/1.0 (open-access corpus preparation)"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?:[\"'’”)]*)\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[a-z][a-z0-9-]{1,}", re.I)
_STOPWORDS = frozenset("a an and are as at be been by can could did do does for from had has have he her his i if in into is it its may might more most not of on or our should that the their them then there these they this those to was were will with would we you your".split())
_JATS_NS = "{http://www.w3.org/1999/xhtml}"


class SourceCollectionError(RuntimeError):
    """Raised for source API, parsing, or licensing failures."""


@dataclass(frozen=True)
class Document:
    metadata: dict[str, str]
    abstract_paragraphs: tuple[str, ...]
    body_paragraphs: tuple[tuple[str, str], ...]
    content: bytes


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return re.sub(r"\s+", " ", "".join(element.itertext())).strip()


def _http_get(url: str, *, timeout: float = 45.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                retry_after = exc.headers.get("Retry-After", "")
                delay = min(60.0, max(float(retry_after) if retry_after.isdigit() else 0.0, 5.0 * (2 ** attempt)))
                time.sleep(delay)
                continue
            raise SourceCollectionError(f"Request failed for {url}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise SourceCollectionError(f"Request failed for {url}: {exc}") from exc
    raise SourceCollectionError(f"Request rate-limited repeatedly by source: {url}")


class RateLimiter:
    """Serial pacing helper. The caller must not run requests concurrently."""

    def __init__(self, minimum_interval: float) -> None:
        if minimum_interval < 0:
            raise ValueError("minimum_interval must be non-negative")
        self.minimum_interval = minimum_interval
        self._last_request = 0.0

    def wait(self) -> None:
        remaining = self.minimum_interval - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request = time.monotonic()


def _date_from_jats(article_meta: ET.Element) -> str:
    for pub_date in article_meta.iter():
        if _local_name(pub_date.tag) != "pub-date":
            continue
        pieces = { _local_name(child.tag): _text(child) for child in pub_date }
        year = pieces.get("year", "")
        if year:
            month = pieces.get("month", "01").zfill(2)
            day = pieces.get("day", "01").zfill(2)
            try:
                return date(int(year), int(month), int(day)).isoformat()
            except ValueError:
                return year
    return ""


def parse_pmc_oai_record(xml_content: bytes, *, fallback_pmcid: str = "",
                         domain: str = "biomedical", collection_query: str = "") -> Document:
    """Parse a PMC OAI-PMH full-text response while retaining license metadata."""
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise SourceCollectionError(f"PMC returned invalid XML: {exc}") from exc
    article = next((node for node in root.iter() if _local_name(node.tag) == "article"), None)
    if article is None:
        raise SourceCollectionError("PMC record did not contain an article element.")
    article_meta = next((node for node in article.iter() if _local_name(node.tag) == "article-meta"), None)
    if article_meta is None:
        raise SourceCollectionError("PMC article did not contain article-meta.")
    title = next((_text(node) for node in article_meta.iter() if _local_name(node.tag) == "article-title"), "")
    id_values = {
        node.attrib.get("pub-id-type", "").lower(): _text(node)
        for node in article_meta.iter() if _local_name(node.tag) == "article-id"
    }
    pmcid = id_values.get("pmc", "") or fallback_pmcid
    if pmcid and not pmcid.upper().startswith("PMC"):
        pmcid = "PMC" + pmcid
    license_node = next((node for node in article.iter() if _local_name(node.tag) == "license"), None)
    license_text = _text(license_node)
    license_url = ""
    if license_node is not None:
        for node in license_node.iter():
            for key, value in node.attrib.items():
                if _local_name(key) == "href" and ("creativecommons.org" in value.lower()):
                    license_url = value
                    break
            if license_url:
                break
    if not license_url and "creativecommons.org/licenses/" in license_text.lower():
        match = re.search(r"https?://creativecommons\.org/(?:licenses|publicdomain)/[^\s)]+", license_text, re.I)
        license_url = match.group(0) if match else ""

    abstracts = []
    for abstract in (node for node in article_meta.iter() if _local_name(node.tag) == "abstract"):
        for paragraph in (child for child in abstract.iter() if _local_name(child.tag) == "p"):
            if _text(paragraph):
                abstracts.append(_text(paragraph))
    if not abstracts:
        abstracts = [_text(node) for node in article_meta.iter() if _local_name(node.tag) == "abstract" and _text(node)]

    body = next((node for node in article.iter() if _local_name(node.tag) == "body"), None)
    body_paragraphs: list[tuple[str, str]] = []
    if body is not None:
        for section_index, section in enumerate((node for node in body.iter() if _local_name(node.tag) == "sec"), 1):
            heading = next((_text(child) for child in section if _local_name(child.tag) == "title"), f"Section {section_index}")
            for paragraph in (node for node in section.iter() if _local_name(node.tag) == "p"):
                value = _text(paragraph)
                if value:
                    body_paragraphs.append((heading, value))
        if not body_paragraphs:
            body_paragraphs = [("Body", _text(node)) for node in body.iter() if _local_name(node.tag) == "p" and _text(node)]

    metadata = {
        "source_document_id": pmcid,
        "source_document_title": title,
        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else "",
        "doi": id_values.get("doi", ""),
        "publication_date": _date_from_jats(article_meta),
        "domain": domain,
        "repository": "PubMed Central Open Access",
        "repository_record_id": pmcid,
        "license": license_text,
        "license_url": license_url,
        "collection_method": "PMC OAI-PMH JATS XML; abstract-to-body candidate retrieval",
        "collection_query": collection_query,
        "collection_date": datetime.now(timezone.utc).date().isoformat(),
        "source_version": id_values.get("pmc", "") or pmcid,
        "source_content_hash": hashlib.sha256(xml_content).hexdigest(),
        "abstract_text": "\n\n".join(abstracts),
    }
    return Document(metadata, tuple(abstracts), tuple(body_paragraphs), xml_content)


def search_pmc(query: str, limit: int, *, domain: str, minimum_interval: float = 1.0,
               license_allowlist: Iterable[str] = LICENSE_ALLOWLIST,
               on_candidate=None, on_document=None, skip_ids: Iterable[str] = ()) -> list[Document]:
    """Search PMC with E-utilities and retrieve only explicitly reusable JATS records.

    Candidate IDs come from the PMC open-access filter. License terms are still
    checked in the full-text record before content is returned to the caller.
    Requests are sequential and slower than PMC's maximum rate.
    """
    if not query.strip() or limit < 1:
        raise ValueError("query must be non-empty and limit must be positive")
    limiter = RateLimiter(minimum_interval)
    term = f"({query}) AND open access[filter]"
    params = urllib.parse.urlencode({"db": "pmc", "term": term, "retmode": "json", "retmax": str(max(limit * 8, 50))})
    limiter.wait()
    try:
        result = json.loads(_http_get(f"{NCBI_ESEARCH_URL}?{params}"))
        ids = result["esearchresult"]["idlist"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SourceCollectionError("Could not parse PMC E-utilities search response.") from exc
    documents: list[Document] = []
    skipped = set(skip_ids)
    for numeric_id in ids:
        if len(documents) >= limit:
            break
        pmcid = f"PMC{numeric_id}"
        if pmcid in skipped or numeric_id in skipped:
            continue
        identifier = f"oai:pubmedcentral.nih.gov:{numeric_id}"
        oai_params = urllib.parse.urlencode({"verb": "GetRecord", "identifier": identifier, "metadataPrefix": "pmc"})
        limiter.wait()
        document = parse_pmc_oai_record(_http_get(f"{PMC_OAI_URL}?{oai_params}"),
                                        fallback_pmcid=f"PMC{numeric_id}", domain=domain,
                                        collection_query=query)
        reusable = license_is_reusable(document.metadata["license"], document.metadata["license_url"], license_allowlist)
        if on_candidate:
            candidate_metadata = dict(document.metadata)
            if not reusable:
                candidate_metadata["rejection_reason"] = "license_not_in_reuse_allowlist"
            on_candidate(candidate_metadata)
        if reusable:
            documents.append(document)
            if on_document:
                on_document(document)
    return documents


def search_arxiv(query: str, limit: int, *, domain: str, minimum_interval: float = 3.1,
                 license_allowlist: Iterable[str] = LICENSE_ALLOWLIST,
                 on_candidate=None, on_document=None, skip_ids: Iterable[str] = ()) -> list[Document]:
    """Search arXiv metadata and retrieve only records with an explicit reusable OAI right.

    The API's descriptive metadata rights do not substitute for the e-print's
    license. This client therefore checks the per-record OAI ``dc:rights`` before
    requesting the source package. The supported source format is TeX tar only.
    """
    if not query.strip() or limit < 1:
        raise ValueError("query must be non-empty and limit must be positive")
    limiter = RateLimiter(minimum_interval)
    params = urllib.parse.urlencode({"search_query": query, "start": "0", "max_results": str(max(limit * 8, 50))})
    limiter.wait()
    try:
        atom = ET.fromstring(_http_get(f"{ARXIV_QUERY_URL}?{params}"))
    except ET.ParseError as exc:
        raise SourceCollectionError(f"arXiv returned invalid Atom XML: {exc}") from exc
    atom_entries = [node for node in atom if _local_name(node.tag) == "entry"]
    documents: list[Document] = []
    skipped = set(skip_ids)
    for entry in atom_entries:
        if len(documents) >= limit:
            break
        abs_url = next((_text(node) for node in entry if _local_name(node.tag) == "id"), "")
        match = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v(\d+))?", abs_url)
        if not match:
            continue
        paper_id, version = match.group(1), match.group(2) or "1"
        if paper_id in skipped:
            continue
        title = next((_text(node) for node in entry if _local_name(node.tag) == "title"), "")
        abstract = next((_text(node) for node in entry if _local_name(node.tag) == "summary"), "")
        published = next((_text(node) for node in entry if _local_name(node.tag) == "published"), "")
        doi = next((_text(node) for node in entry.iter() if _local_name(node.tag) == "doi"), "")

        oai_id = urllib.parse.quote(f"oai:arXiv.org:{paper_id}", safe=":")
        oai_params = urllib.parse.urlencode({"verb": "GetRecord", "identifier": oai_id, "metadataPrefix": "oai_dc"})
        limiter.wait()
        try:
            rights_xml = ET.fromstring(_http_get(f"{ARXIV_OAI_URL}?{oai_params}"))
        except ET.ParseError as exc:
            raise SourceCollectionError(f"arXiv OAI returned invalid XML for {paper_id}: {exc}") from exc
        rights = [_text(node) for node in rights_xml.iter() if _local_name(node.tag) == "rights"]
        right_text = " ".join(rights)
        if not license_is_reusable(right_text, right_text, license_allowlist):
            if on_candidate:
                on_candidate({"source_document_id": paper_id, "source_document_title": title,
                              "source_url": abs_url, "license": right_text,
                              "license_url": right_text, "rejection_reason": "license_not_in_reuse_allowlist"})
            continue

        limiter.wait()
        package = _http_get(f"{ARXIV_EPRINT_URL}{paper_id}v{version}")
        tex = _extract_arxiv_tex(package)
        if on_candidate:
            on_candidate({"source_document_id": paper_id, "source_document_title": title,
                          "source_url": abs_url, "license": right_text, "license_url": right_text})
        metadata = {
            "source_document_id": paper_id,
            "source_document_title": title,
            "source_url": abs_url,
            "doi": doi,
            "publication_date": published[:10],
            "domain": domain,
            "repository": "arXiv",
            "repository_record_id": paper_id,
            "license": right_text,
            "license_url": right_text,
            "collection_method": "arXiv API metadata + OAI-PMH rights + licensed TeX e-print",
            "collection_query": query,
            "collection_date": datetime.now(timezone.utc).date().isoformat(),
            "source_version": f"{paper_id}v{version}",
            "source_content_hash": hashlib.sha256(package).hexdigest(),
            "abstract_text": abstract,
        }
        document = Document(metadata, (abstract,), (("Body", tex),), package)
        documents.append(document)
        if on_document:
            on_document(document)
    return documents


def _strip_tex(tex: str) -> str:
    text = re.sub(r"(?m)^\s*%.*$", " ", tex)
    text = re.sub(r"\\(?:cite\w*|ref|label|url|href|footnote|footnotemark)\s*(?:\[[^]]*\])?\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\begin\{(?:equation\*?|align\*?|math|figure\*?|table\*?)\}.*?\\end\{(?:equation\*?|align\*?|math|figure\*?|table\*?)\}", " ", text, flags=re.S)
    text = re.sub(r"\$\$?.*?\$\$?", " ", text, flags=re.S)
    text = re.sub(r"\\(?:emph|textit|textbf|mathrm|mathbf|text)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^]]*\])?\s*(?:\{[^{}]*\})?", " ", text)
    text = text.replace("~", " ").replace("\\\\", " ")
    text = re.sub(r"[{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_arxiv_tex(package: bytes) -> str:
    try:
        with tarfile.open(fileobj=io.BytesIO(package), mode="r:*") as archive:
            sources = [member for member in archive.getmembers()
                       if member.isfile() and member.name.lower().endswith((".tex", ".ltx"))]
            for member in sorted(sources, key=lambda item: ("\\begin{document}" not in (archive.extractfile(item).read().decode("utf-8", errors="ignore") if archive.extractfile(item) else ""), item.name)):
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                tex = stream.read().decode("utf-8", errors="replace")
                body_start = tex.find("\\begin{document}")
                body = tex[body_start + len("\\begin{document}"):] if body_start >= 0 else tex
                body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", " ", body, flags=re.S)
                body = re.sub(r"\\end\{document\}.*$", " ", body, flags=re.S)
                plain = _strip_tex(body)
                if len(plain.split()) >= 50:
                    return plain
    except (tarfile.TarError, OSError) as exc:
        raise SourceCollectionError(f"Could not parse arXiv TeX archive: {exc}") from exc
    raise SourceCollectionError("arXiv source package had no parseable TeX document.")


def split_sentences(paragraph: str) -> list[str]:
    """Conservative sentence segmentation; short fragments are filtered later."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip()]


def _content_terms(sentence: str) -> set[str]:
    return {token.casefold() for token in _WORD_RE.findall(sentence)
            if token.casefold() not in _STOPWORDS and len(token) > 2}


_REFERENCE_ONLY_RE = re.compile(
    r"^(?:https?://|doi\s*:|10\.\d{4,9}/|(?:references?|bibliography|supplementary\s+material)\b)", re.I,
)
_CAPTION_RE = re.compile(r"^(?:fig(?:ure)?|table|scheme|supplementary\s+(?:fig(?:ure)?|table))\s*[\dA-Z.-]+\s*[:.)-]", re.I)
_URL_NOTICE_RE = re.compile(
    r"^(?:the\s+)?(?:online\s+version|full\s+text|article|supplementary\s+(?:material|file|data)|dataset|additional\s+resources?)\b.*\b(?:available|access(?:ible)?|found)\b.*https?://",
    re.I,
)
_FRAGMENT_RE = re.compile(r"^(?:[\W\d_]+|[A-Z][\w-]{0,12}\s*[:.)-])$")
_FORMULA_ONLY_RE = re.compile(
    r"^\s*[A-Za-z0-9_().]+(?:\s*(?:<=|>=|≤|≥|=|<|>|/|\*|\+|-)\s*[A-Za-z0-9_().%+-]+){2,}\s*[.;]?\s*$"
)


def _usable_candidate_sentence(sentence: str) -> bool:
    """Drop obvious bibliographic/caption fragments before semantic ranking.

    This is a retrieval-quality filter only. It neither labels nor accepts pairs.
    """
    text = sentence.strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if (len(words) < 8 or _REFERENCE_ONLY_RE.match(text) or _CAPTION_RE.match(text)
            or _URL_NOTICE_RE.match(text) or _FORMULA_ONLY_RE.fullmatch(text)):
        return False
    if "doi.org/" in text.casefold() or "supplementary material available" in text.casefold():
        return False
    if _FRAGMENT_RE.fullmatch(text):
        return False
    return True


def _candidate_similarity(left: str, right: str) -> tuple[float, int]:
    left_terms, right_terms = _content_terms(left), _content_terms(right)
    overlap = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    return (overlap / union if union else 0.0), overlap


def extract_document_candidates(document: Document, *, scorer: PairScorer,
                                top_k: int = 3,
                                minimum_semantic_score: float = 0.25) -> list[dict[str, str]]:
    """Rank same-document sentence pairs semantically for Stage A review.

    Cosine similarity is only a retrieval/ranking feature. Every emitted row is
    explicitly pending proposition assessment and is never marked eligible.
    """
    metadata = document.metadata
    abstracts = [(para_index, sentence_index, sentence)
                 for para_index, paragraph in enumerate(document.abstract_paragraphs, 1)
                 for sentence_index, sentence in enumerate(split_sentences(paragraph), 1)
                 if _usable_candidate_sentence(sentence)]
    body = [(section, para_index, sentence_index, sentence)
            for para_index, (section, paragraph) in enumerate(document.body_paragraphs, 1)
            for sentence_index, sentence in enumerate(split_sentences(paragraph), 1)
            if _usable_candidate_sentence(sentence)]
    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    all_pairs = [(source_sentence, summary_sentence)
                 for _section, _body_para, _body_sentence_index, source_sentence in body
                 for _abs_para, _abs_sentence_index, summary_sentence in abstracts]
    all_scores = scorer.score_pairs(all_pairs)
    scores_by_pair = dict(zip(all_pairs, all_scores))
    for abs_para, abs_sentence_index, summary_sentence in abstracts:
        ranked = []
        for section, body_para, body_sentence_index, source_sentence in body:
            semantic_score = scores_by_pair[(source_sentence, summary_sentence)]
            lexical_score, _shared = _candidate_similarity(source_sentence, summary_sentence)
            if semantic_score >= minimum_semantic_score:
                ranked.append((semantic_score, lexical_score, section, body_para,
                               body_sentence_index, source_sentence))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[3], item[4]))
        for semantic_score, lexical_score, section, body_para, body_sentence_index, source_sentence in ranked[:top_k]:
            norm_pair = tuple(sorted((source_sentence.casefold(), summary_sentence.casefold())))
            if norm_pair in seen:
                continue
            seen.add(norm_pair)
            doc_id = metadata["source_document_id"]
            example_id = stable_example_id(doc_id, source_sentence, summary_sentence)
            row = {field: "" for field in REAL_WORLD_V2_FIELDS}
            row.update({
                "example_id": example_id,
                "source_document_id": doc_id,
                "source_group_id": doc_id,
                "source_document_title": metadata.get("source_document_title", ""),
                "source_url": metadata.get("source_url", ""),
                "doi": metadata.get("doi", ""),
                "publication_date": metadata.get("publication_date", ""),
                "source_sentence": source_sentence,
                "summary_sentence": summary_sentence,
                "domain": metadata.get("domain", ""),
                "source_section": section,
                "source_location": f"{section}; paragraph {body_para}; sentence {body_sentence_index}",
                "summary_location": f"Abstract; paragraph {abs_para}; sentence {abs_sentence_index}",
                "collection_method": metadata.get("collection_method", ""),
                "collection_date": metadata.get("collection_date", ""),
                "license": metadata.get("license", ""),
                "license_url": metadata.get("license_url", ""),
                "source_version": metadata.get("source_version", ""),
                "source_content_hash": metadata.get("source_content_hash", ""),
                "candidate_score": f"{semantic_score:.6f}",
                "lexical_score": f"{lexical_score:.6f}",
                "semantic_score": f"{semantic_score:.6f}",
                "alignment_model": scorer.model_name,
                "alignment_status": "requires_proposition_assessment",
                "annotation_version": "retrieval-v2",
                "annotation_status": "not_annotated",
                "pipeline_status": "candidate",
                "machine_label_eligible": "false",
                "requires_review": "true",
            })
            results.append(row)
    return results


def extract_candidates(documents: Iterable[Document], *, scorer: PairScorer, **options) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        for row in extract_document_candidates(document, scorer=scorer, **options):
            pair = (row["source_document_id"],) + tuple(sorted((row["source_sentence"].casefold(),
                                                                  row["summary_sentence"].casefold())))
            if pair not in seen:
                seen.add(pair)
                candidates.append(row)
    return candidates


def save_documents(path: str | Path, documents: Iterable[Document]) -> None:
    rows = []
    for document in documents:
        row = {field: document.metadata.get(field, "") for field in DOCUMENT_FIELDS}
        rows.append(row)
    write_real_world_csv(path, rows, fields=DOCUMENT_FIELDS)
