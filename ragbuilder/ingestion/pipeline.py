"""Ingestion pipeline: PDFs on disk -> pages and chunks in PostgreSQL.

Incremental by design. Each document's SHA-256 is stored, and a re-run skips any
file whose bytes are unchanged *and* whose chunks for the requested strategy are
already present. Adding one new statute to a 24-document corpus costs seconds
rather than re-parsing everything.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ragbuilder import db
from ragbuilder.chunking.base import Chunk
from ragbuilder.chunking.strategies import get_chunker
from ragbuilder.config import Config, get_config
from ragbuilder.ingestion.corpora import Document, download_corpus, get_corpus
from ragbuilder.ingestion.parser import parse_pdf

LOG = logging.getLogger("ragbuilder.ingest")


@dataclass
class IngestionStats:
    seen: int = 0
    new: int = 0
    changed: int = 0
    skipped: int = 0
    chunks: int = 0
    pages: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "seen": self.seen,
            "new": self.new,
            "changed": self.changed,
            "skipped": self.skipped,
            "chunks": self.chunks,
            "pages": self.pages,
            "failures": len(self.failures),
            "duration_seconds": round(self.duration_seconds, 2),
        }


def _document_metadata(path: Path, document: Document | None) -> tuple[str, str, str, str, str]:
    """(doc_id, title, url, language, category) for a file, corpus entry or not."""
    if document is not None:
        return document.doc_id, document.title, document.url, document.language, document.category
    return path.stem.lower(), path.stem, "", "de", "uploaded"


def ingest_documents(
    paths: Sequence[Path],
    strategies: Sequence[str] | None = None,
    config: Config | None = None,
    corpus_documents: dict[str, Document] | None = None,
    force: bool = False,
    embed_fn: Callable | None = None,
    corpus_name: str = "custom",
) -> IngestionStats:
    """Parse and chunk ``paths``, writing pages and chunks to PostgreSQL."""
    config = config or get_config()
    strategies = list(strategies or [config.chunking.strategy])
    corpus_documents = corpus_documents or {}
    stats = IngestionStats()
    started = time.perf_counter()

    run_id = db.start_run(corpus_name, ",".join(strategies))
    error: str | None = None

    try:
        for path in paths:
            stats.seen += 1
            document = corpus_documents.get(path.name) or corpus_documents.get(path.stem.lower())
            doc_id, title, url, language, category = _document_metadata(path, document)

            try:
                parsed = parse_pdf(path, doc_id=doc_id)
            except Exception as exc:  # noqa: BLE001 - a corrupt PDF is not fatal
                LOG.error("failed to parse %s: %s", path.name, exc)
                stats.failures.append((path.name, str(exc)))
                continue

            existing_hash = db.get_document_hash(doc_id)
            unchanged = existing_hash == parsed.content_hash

            if unchanged and not force and _has_all_chunks(doc_id, strategies):
                LOG.info("%-14s unchanged, skipping", doc_id)
                stats.skipped += 1
                continue

            if existing_hash is None:
                stats.new += 1
            elif not unchanged:
                stats.changed += 1
                LOG.info("%-14s content changed, re-ingesting", doc_id)

            db.upsert_document(
                {
                    "doc_id": doc_id,
                    "title": title or parsed.title_from_pdf or doc_id,
                    "source_url": url,
                    "filename": parsed.filename,
                    "language": language,
                    "category": category,
                    "content_hash": parsed.content_hash,
                    "page_count": parsed.page_count,
                    "char_count": parsed.char_count,
                    "table_count": parsed.table_count,
                    "metadata": {"pdf_title": parsed.title_from_pdf},
                }
            )

            pages = [
                {
                    "page_number": page.page_number,
                    "text": page.text,
                    "section": page.section,
                    "has_tables": page.has_tables,
                }
                for page in parsed.pages
            ]
            stats.pages += db.replace_pages(doc_id, pages)

            for strategy in strategies:
                chunker = get_chunker(strategy, config, embed_fn=embed_fn)
                chunks: list[Chunk] = chunker(pages, doc_id)
                # The chunkers do not know which strategy invoked them; stamping
                # it here is what makes chunk ids unique across strategies.
                for chunk in chunks:
                    chunk.strategy = strategy
                written = db.replace_chunks(doc_id, strategy, [c.as_row() for c in chunks])
                stats.chunks += written
                LOG.info("%-14s %-9s -> %4d chunks", doc_id, strategy, written)

    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        raise
    finally:
        stats.duration_seconds = time.perf_counter() - started
        db.finish_run(run_id, stats.as_dict(), error)

    return stats


def _has_all_chunks(doc_id: str, strategies: Sequence[str]) -> bool:
    with db.cursor() as cur:
        cur.execute(
            "SELECT strategy, count(*) FROM chunks WHERE doc_id = %s GROUP BY strategy",
            (doc_id,),
        )
        present = {row[0]: row[1] for row in cur.fetchall()}
    return all(present.get(strategy, 0) > 0 for strategy in strategies)


def ingest_corpus(
    corpus_name: str | None = None,
    strategies: Sequence[str] | None = None,
    config: Config | None = None,
    download: bool = True,
    force: bool = False,
    embed_fn: Callable | None = None,
    limit: int | None = None,
) -> IngestionStats:
    """Download (if needed) and ingest a named corpus."""
    config = config or get_config()
    corpus_name = corpus_name or config.corpus.name
    destination = config.pdf_path

    if download:
        download_corpus(corpus_name, destination)

    documents = {d.filename: d for d in get_corpus(corpus_name)}
    documents.update({d.doc_id: d for d in get_corpus(corpus_name)})

    paths = sorted(p for p in destination.glob("*.pdf") if p.stat().st_size > 0)
    if limit:
        paths = paths[:limit]

    if not paths:
        raise RuntimeError(f"no PDFs found in {destination}; run the download step first")

    LOG.info("ingesting %d PDF(s) from %s", len(paths), destination)
    return ingest_documents(
        paths,
        strategies=strategies,
        config=config,
        corpus_documents=documents,
        force=force,
        embed_fn=embed_fn,
        corpus_name=corpus_name,
    )
