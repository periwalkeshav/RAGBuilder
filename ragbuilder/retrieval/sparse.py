"""BM25 sparse retrieval over the chunk table.

Dense retrieval is bad at exactly one thing that matters enormously in a legal
corpus: **rare literal tokens**. Ask for "§ 622 Absatz 2" and an embedding model
will happily return a semantically similar paragraph about notice periods that
is not the one you asked for. BM25 matches the literal string and ranks it
first. That complementary failure mode is the whole argument for hybrid search.

The index is built in memory from PostgreSQL and cached per strategy. For a
corpus this size that is milliseconds; past a few hundred thousand chunks you
would move to PostgreSQL full-text search or Elasticsearch.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from ragbuilder import db
from ragbuilder.retrieval.fusion import RetrievedChunk

LOG = logging.getLogger("ragbuilder.sparse")

# Keep § and digits: they carry most of the signal in statutory text.
TOKEN_PATTERN = re.compile(r"[§]|[0-9]+[a-z]?|[\wäöüßÄÖÜ]+", re.UNICODE)

GERMAN_STOPWORDS = {
    "der",
    "die",
    "das",
    "des",
    "dem",
    "den",
    "ein",
    "eine",
    "einer",
    "eines",
    "einem",
    "einen",
    "und",
    "oder",
    "aber",
    "auch",
    "ist",
    "sind",
    "war",
    "waren",
    "wird",
    "werden",
    "wurde",
    "nicht",
    "kein",
    "keine",
    "im",
    "in",
    "an",
    "auf",
    "für",
    "von",
    "zu",
    "zum",
    "zur",
    "mit",
    "bei",
    "nach",
    "aus",
    "über",
    "unter",
    "durch",
    "gegen",
    "ohne",
    "um",
    "als",
    "wenn",
    "dass",
    "so",
    "wie",
    "nur",
    "noch",
    "schon",
    "sich",
    "es",
    "er",
    "sie",
    "man",
    "dieser",
    "diese",
}

ENGLISH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "with",
    "by",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "not",
    "no",
    "can",
    "will",
    "would",
    "should",
    "may",
    "if",
    "than",
    "then",
    "so",
}

STOPWORDS = GERMAN_STOPWORDS | ENGLISH_STOPWORDS


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in TOKEN_PATTERN.findall(text)]
    # Single characters are dropped, except § which is meaningful on its own.
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t == "§")]


@dataclass
class BM25Index:
    strategy: str
    chunk_ids: list[str]
    documents: list[dict]
    bm25: BM25Okapi
    built_at: float

    def search(self, query: str, limit: int = 20) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]
        results: list[RetrievedChunk] = []
        for position in ranked:
            if scores[position] <= 0:
                continue
            row = self.documents[position]
            results.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    score=float(scores[position]),
                    text=row["text"],
                    doc_id=row["doc_id"],
                    title=row.get("title", ""),
                    section=row.get("section"),
                    page_start=row.get("page_start"),
                    page_end=row.get("page_end"),
                    retriever="sparse",
                )
            )
        return results


_INDEXES: dict[str, BM25Index] = {}
_LOCK = threading.Lock()


def build_index(strategy: str, rows: Sequence[dict] | None = None) -> BM25Index:
    started = time.perf_counter()
    documents = list(rows if rows is not None else db.iter_chunks(strategy))
    if not documents:
        raise RuntimeError(f"no chunks found for strategy {strategy!r} - ingest first")

    corpus = [tokenize(row["text"]) for row in documents]
    index = BM25Index(
        strategy=strategy,
        chunk_ids=[row["chunk_id"] for row in documents],
        documents=documents,
        bm25=BM25Okapi(corpus),
        built_at=time.time(),
    )
    LOG.info(
        "built BM25 index for %r: %d chunks, %d unique tokens, %.2fs",
        strategy,
        len(documents),
        len({t for doc in corpus for t in doc}),
        time.perf_counter() - started,
    )
    return index


def get_index(strategy: str, refresh: bool = False) -> BM25Index:
    with _LOCK:
        if refresh or strategy not in _INDEXES:
            _INDEXES[strategy] = build_index(strategy)
        return _INDEXES[strategy]


def invalidate(strategy: str | None = None) -> None:
    """Drop the cached index - call after ingesting new documents."""
    with _LOCK:
        if strategy:
            _INDEXES.pop(strategy, None)
        else:
            _INDEXES.clear()


def search(query: str, strategy: str, limit: int = 20) -> list[RetrievedChunk]:
    return get_index(strategy).search(query, limit=limit)
