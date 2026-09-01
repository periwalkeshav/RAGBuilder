"""Qdrant vector store.

Why Qdrant and not Chroma or Pinecone - the question every interviewer asks:

* **Self-hosted and open source.** The whole point of the local-LLM stack is that
  no document leaves the machine. A managed vector database undoes that, which
  for a German client under GDPR is not a detail.
* **Real payload filtering.** Filters are applied *inside* the HNSW traversal,
  not as a post-filter over the top-k. Restricting a search to one statute
  therefore stays fast instead of silently returning fewer results than asked
  for.
* **Named collections with explicit dimensions.** Changing the embedding model
  is a new collection, not a corrupted one.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Sequence

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from ragbuilder.config import Config, get_config

LOG = logging.getLogger("ragbuilder.store")

DISTANCES = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclidean": qmodels.Distance.EUCLID,
}


def _point_id(chunk_id: str) -> str:
    """Qdrant point ids must be a UUID or an unsigned int.

    Chunk ids are human-readable strings, so they are hashed into a stable UUID5
    and kept verbatim in the payload. Same chunk id in, same point id out - which
    is what makes re-embedding an upsert instead of a duplicate.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragbuilder://{chunk_id}"))


class VectorStore:
    def __init__(self, config: Config | None = None, client: QdrantClient | None = None) -> None:
        self.config = config or get_config()
        self.collection = self.config.vector_store.collection
        self.client = client or QdrantClient(url=self.config.vector_store.url, timeout=60)

    # ----------------------------------------------------------- collection
    def ensure_collection(self, dimensions: int | None = None, recreate: bool = False) -> None:
        size = dimensions or self.config.embedding.dimensions
        exists = self.client.collection_exists(self.collection)

        if exists and not recreate:
            info = self.client.get_collection(self.collection)
            current = info.config.params.vectors.size
            if current != size:
                raise RuntimeError(
                    f"collection {self.collection!r} has {current} dimensions but the configured "
                    f"model produces {size}. Re-create it: `make reset-vectors`."
                )
            return

        if exists:
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qmodels.VectorParams(
                size=size,
                distance=DISTANCES[self.config.vector_store.distance],
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=self.config.vector_store.hnsw_m,
                ef_construct=self.config.vector_store.hnsw_ef_construct,
            ),
        )
        # Indexed payload fields keep filtered search inside the graph traversal.
        for field, schema in (
            ("strategy", qmodels.PayloadSchemaType.KEYWORD),
            ("doc_id", qmodels.PayloadSchemaType.KEYWORD),
            ("category", qmodels.PayloadSchemaType.KEYWORD),
            ("language", qmodels.PayloadSchemaType.KEYWORD),
        ):
            self.client.create_payload_index(self.collection, field_name=field, field_schema=schema)

        LOG.info("created collection %s (%d dimensions)", self.collection, size)

    def drop_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
            LOG.info("dropped collection %s", self.collection)

    # --------------------------------------------------------------- upsert
    def upsert(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]],
        batch_size: int = 256,
    ) -> int:
        if len(chunk_ids) != len(vectors) or len(chunk_ids) != len(payloads):
            raise ValueError("chunk_ids, vectors and payloads must be the same length")

        written = 0
        for start in range(0, len(chunk_ids), batch_size):
            end = start + batch_size
            points = [
                qmodels.PointStruct(
                    id=_point_id(chunk_id),
                    vector=vector.tolist(),
                    payload={**payload, "chunk_id": chunk_id},
                )
                for chunk_id, vector, payload in zip(
                    chunk_ids[start:end], vectors[start:end], payloads[start:end]
                )
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            written += len(points)
        return written

    # --------------------------------------------------------------- search
    def search(
        self,
        vector: np.ndarray,
        limit: int = 20,
        strategy: str | None = None,
        doc_ids: Sequence[str] | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions: list[qmodels.FieldCondition] = []
        if strategy:
            conditions.append(
                qmodels.FieldCondition(key="strategy", match=qmodels.MatchValue(value=strategy))
            )
        if doc_ids:
            conditions.append(qmodels.FieldCondition(key="doc_id", match=qmodels.MatchAny(any=list(doc_ids))))
        if category:
            conditions.append(
                qmodels.FieldCondition(key="category", match=qmodels.MatchValue(value=category))
            )

        results = self.client.search(
            collection_name=self.collection,
            query_vector=vector.tolist(),
            limit=limit,
            query_filter=qmodels.Filter(must=conditions) if conditions else None,
            with_payload=True,
        )
        return [
            {
                "chunk_id": point.payload.get("chunk_id"),
                "score": float(point.score),
                "payload": point.payload,
            }
            for point in results
        ]

    # ---------------------------------------------------------------- prune
    def prune(self, strategy: str, valid_chunk_ids: Sequence[str], batch_size: int = 1024) -> int:
        """Delete points for ``strategy`` whose chunk is no longer in PostgreSQL.

        Chunk ids are content-addressed, so re-chunking a document produces new
        ids and leaves the old points behind. Those orphans still match a
        filtered search, still look like valid hits, and resolve to no row in
        PostgreSQL - retrieval quietly degrades to the 300-character payload
        preview. The vector store is derived state, so anything not currently in
        the system of record has to go.
        """
        if not self.client.collection_exists(self.collection):
            return 0

        valid = set(valid_chunk_ids)
        stale: list[str] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="strategy", match=qmodels.MatchValue(value=strategy))]
                ),
                limit=batch_size,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            stale.extend(point.id for point in points if point.payload.get("chunk_id") not in valid)
            if offset is None:
                break

        if stale:
            self.client.delete(
                collection_name=self.collection,
                points_selector=qmodels.PointIdsList(points=stale),
                wait=True,
            )
            LOG.warning("pruned %d stale point(s) from strategy %r", len(stale), strategy)
        return len(stale)

    # ---------------------------------------------------------------- stats
    def count(self, strategy: str | None = None) -> int:
        if not self.client.collection_exists(self.collection):
            return 0
        query_filter = None
        if strategy:
            query_filter = qmodels.Filter(
                must=[qmodels.FieldCondition(key="strategy", match=qmodels.MatchValue(value=strategy))]
            )
        return int(self.client.count(self.collection, count_filter=query_filter, exact=True).count)

    def info(self) -> dict[str, Any]:
        if not self.client.collection_exists(self.collection):
            return {"exists": False, "collection": self.collection}
        details = self.client.get_collection(self.collection)
        return {
            "exists": True,
            "collection": self.collection,
            "points": details.points_count,
            "dimensions": details.config.params.vectors.size,
            "distance": str(details.config.params.vectors.distance),
            "status": str(details.status),
        }

    def healthy(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:  # noqa: BLE001 - health checks must not raise
            return False


_STORE: VectorStore | None = None


def get_store(config: Config | None = None) -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore(config)
    return _STORE
