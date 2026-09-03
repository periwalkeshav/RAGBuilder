"""Embedding pipeline: chunks in PostgreSQL -> vectors in Qdrant.

Batched with progress reporting, and resumable: only chunks with
``embedded = FALSE`` are processed, so an interrupted run picks up where it left
off instead of re-encoding everything.

The batch is 250 rather than 1,000 because the checkpoint is per batch. On CPU a
1,000-chunk batch of e5-large is roughly 20 minutes of complete silence, and an
interruption anywhere in it throws all of that work away.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from ragbuilder import db
from ragbuilder.config import Config, get_config
from ragbuilder.embeddings.encoder import get_encoder
from ragbuilder.embeddings.store import get_store

LOG = logging.getLogger("ragbuilder.embed")


@dataclass
class EmbeddingStats:
    strategy: str = ""
    chunks: int = 0
    batches: int = 0
    duration_seconds: float = 0.0
    dimensions: int = 0

    @property
    def chunks_per_second(self) -> float:
        return self.chunks / self.duration_seconds if self.duration_seconds > 0 else 0.0


def build_payload(chunk: dict, strategy: str) -> dict:
    """What travels with the vector into Qdrant.

    Deliberately small: enough to filter and to render a citation without a
    round trip, but the authoritative text stays in PostgreSQL. Duplicating full
    chunk bodies into the vector store is how the two copies drift apart.
    """
    return {
        "doc_id": chunk["doc_id"],
        "strategy": strategy,
        "chunk_index": chunk["chunk_index"],
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "section": chunk.get("section"),
        "title": chunk.get("title"),
        "language": chunk.get("language"),
        "category": chunk.get("category"),
        "preview": (chunk["text"][:300] + "…") if len(chunk["text"]) > 300 else chunk["text"],
    }


def embed_strategy(
    strategy: str | None = None,
    config: Config | None = None,
    batch_size: int = 250,
    force: bool = False,
    recreate_collection: bool = False,
    prune: bool = True,
) -> EmbeddingStats:
    """Encode every chunk of ``strategy`` and upsert it into Qdrant."""
    config = config or get_config()
    strategy = strategy or config.chunking.strategy
    encoder = get_encoder(config)
    store = get_store(config)

    store.ensure_collection(dimensions=encoder.dimensions, recreate=recreate_collection)
    if force:
        db.reset_embedded(strategy)

    # Prune before embedding, against the full current chunk set - not the
    # unembedded subset, which would delete everything already stored.
    if prune and not recreate_collection:
        stats_prune = store.prune(strategy, [c["chunk_id"] for c in db.iter_chunks(strategy)])
        if stats_prune:
            LOG.info("removed %d orphaned vector(s) before embedding", stats_prune)

    chunks = db.iter_chunks(strategy, only_unembedded=not force and not recreate_collection)
    stats = EmbeddingStats(strategy=strategy, dimensions=encoder.dimensions)

    if not chunks:
        LOG.info("nothing to embed for strategy %r", strategy)
        return stats

    LOG.info(
        "embedding %d chunk(s) for strategy %r with %s (%d dimensions)",
        len(chunks),
        strategy,
        config.embedding.model,
        encoder.dimensions,
    )
    started = time.perf_counter()

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        batch_started = time.perf_counter()

        vectors = encoder.encode_passages([c["text"] for c in batch], show_progress=False)
        chunk_ids = [c["chunk_id"] for c in batch]
        payloads = [build_payload(c, strategy) for c in batch]

        store.upsert(chunk_ids, vectors, payloads)
        db.mark_embedded(chunk_ids)

        stats.chunks += len(batch)
        stats.batches += 1
        elapsed = time.perf_counter() - batch_started
        LOG.info(
            "  batch %d: %d chunk(s) in %.1fs (%.1f chunks/s) - %d/%d done",
            stats.batches,
            len(batch),
            elapsed,
            len(batch) / elapsed if elapsed else 0,
            stats.chunks,
            len(chunks),
        )

    stats.duration_seconds = time.perf_counter() - started
    LOG.info(
        "embedded %d chunk(s) in %.1fs (%.1f chunks/s); collection now holds %d point(s)",
        stats.chunks,
        stats.duration_seconds,
        stats.chunks_per_second,
        store.count(strategy),
    )
    return stats


def embed_all(
    strategies: list[str] | None = None,
    config: Config | None = None,
    force: bool = False,
) -> list[EmbeddingStats]:
    config = config or get_config()
    strategies = strategies or list(config.evaluation.strategies)
    results = []
    for index, strategy in enumerate(strategies):
        results.append(
            embed_strategy(
                strategy,
                config=config,
                force=force,
                # Only the first call may recreate; the rest share the collection.
                recreate_collection=False,
            )
        )
        if index == 0:
            LOG.info("---")
    return results
