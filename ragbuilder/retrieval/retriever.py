"""The retriever: dense, sparse, hybrid, plus query expansion.

``Retriever.retrieve()`` is the single entry point used by the RAG chain, the
API and the evaluation harness, so all three measure the same thing.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Sequence

from ragbuilder import db
from ragbuilder.config import Config, get_config
from ragbuilder.embeddings.encoder import get_encoder
from ragbuilder.embeddings.store import get_store
from ragbuilder.retrieval import sparse
from ragbuilder.retrieval.fusion import RetrievedChunk, merge_results

LOG = logging.getLogger("ragbuilder.retriever")


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    mode: str = "hybrid"
    strategy: str = "sentence"
    expanded_queries: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    # BM25 saturation constant. score/(score+k) maps an unbounded BM25 score onto
    # 0-1 and is ~0.5 at k, which on this corpus is roughly the boundary between
    # "shares a rare term with the query" and "shares only common ones".
    BM25_SATURATION = 12.0

    @property
    def confidence(self) -> float:
        """A 0-1 estimate of whether the context can support an answer.

        Deliberately simple and inspectable rather than a learned scorer:

        * the top retriever score, which says "is anything actually close",
        * in hybrid mode, the share of results both retrievers agree on, which
          says "do two independent methods point at the same evidence".

        It is used only as a refusal threshold, so being roughly right and
        explainable beats being precisely wrong.
        """
        if not self.chunks:
            return 0.0

        dense_scores = [c.dense_score for c in self.chunks if c.dense_score is not None]
        if dense_scores:
            # Cosine over normalised e5 vectors sits in roughly 0.70-0.95 for
            # good matches, so rescale that band onto 0-1 rather than using it
            # raw - an unscaled 0.82 would look like high confidence.
            strength = min(1.0, max(0.0, (max(dense_scores) - 0.70) / 0.25))
        else:
            # Sparse-only: BM25 is unbounded, so saturate rather than rescale.
            # Without this the whole mode reports confidence 0 and refuses every
            # question, which is what the retrieval evaluation grid exposed.
            sparse_scores = [c.sparse_score for c in self.chunks if c.sparse_score is not None]
            top = max(sparse_scores) if sparse_scores else 0.0
            strength = top / (top + self.BM25_SATURATION) if top > 0 else 0.0

        if self.mode == "hybrid":
            agreed = sum(1 for c in self.chunks if c.dense_rank and c.sparse_rank)
            agreement = agreed / len(self.chunks)
            return round(0.7 * strength + 0.3 * agreement, 4)
        return round(strength, 4)


class Retriever:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.encoder = get_encoder(self.config)
        self.store = get_store(self.config)

    # ----------------------------------------------------------------- dense
    def dense_search(
        self, query: str, strategy: str, limit: int, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        vector = self.encoder.encode_query(query)
        hits = self.store.search(vector, limit=limit, strategy=strategy, doc_ids=doc_ids)
        if not hits:
            return []

        texts = db.get_chunks_by_id([h["chunk_id"] for h in hits])
        results: list[RetrievedChunk] = []
        for hit in hits:
            row = texts.get(hit["chunk_id"], {})
            payload = hit["payload"]
            results.append(
                RetrievedChunk(
                    chunk_id=hit["chunk_id"],
                    score=hit["score"],
                    text=row.get("text", payload.get("preview", "")),
                    doc_id=row.get("doc_id") or payload.get("doc_id", ""),
                    # PostgreSQL wins over the Qdrant payload: it is the system
                    # of record, so a re-ingest that corrects metadata takes
                    # effect without re-embedding 1,400 chunks.
                    title=row.get("title") or payload.get("title", ""),
                    section=row.get("section") or payload.get("section"),
                    page_start=row.get("page_start") or payload.get("page_start"),
                    page_end=row.get("page_end") or payload.get("page_end"),
                    source_url=row.get("source_url", ""),
                    retriever="dense",
                )
            )
        return results

    # ---------------------------------------------------------------- sparse
    def sparse_search(self, query: str, strategy: str, limit: int) -> list[RetrievedChunk]:
        try:
            return sparse.search(query, strategy, limit=limit)
        except RuntimeError as exc:
            LOG.warning("sparse retrieval unavailable: %s", exc)
            return []

    # ------------------------------------------------------- query expansion
    def expand_query(self, query: str, variants: int = 3) -> list[str]:
        """Produce paraphrases so retrieval is not hostage to one phrasing.

        Rule-based rather than LLM-generated on purpose: expansion sits on the
        latency critical path of every single question, and a 2-second LLM call
        to rephrase a question is a poor trade against the recall it buys. The
        variants that matter for this corpus are cheap and deterministic -
        normalising ``§``/``Paragraf``/``Artikel``, stripping the interrogative
        frame so the query looks more like the statutory text it should match,
        and dropping stopwords to give BM25 a keyword-shaped query.
        """
        expansions: list[str] = []
        normalized = query.strip()

        section_forms = {
            "paragraf": "§",
            "paragraph": "§",
            "artikel": "art.",
            "absatz": "abs.",
            "satz": "s.",
        }
        lowered = normalized.lower()
        replaced = normalized
        for word, symbol in section_forms.items():
            if word in lowered:
                replaced = re.sub(word, symbol, replaced, flags=re.IGNORECASE)
        if replaced != normalized:
            expansions.append(replaced)

        # Strip the question frame: "Wie lange ist die Kündigungsfrist?" ->
        # "die Kündigungsfrist", which is closer to how the statute phrases it.
        stripped = re.sub(
            r"^(?:wie|was|wann|wo|wer|warum|welche[rsn]?|wieviel|" r"what|when|where|who|why|which|how)\s+"
            # Repeat rather than match once: "Wie lange ist die Kündigungsfrist"
            # needs both "lange" and "ist" removed to leave the noun phrase.
            r"(?:(?:lange|viel|viele|oft|ist|sind|war|kann|darf|muss|soll|"
            r"does|do|is|are|can|must|long|many|much)\s+)*",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip(" ?")
        if stripped and stripped.lower() != normalized.lower().strip(" ?"):
            expansions.append(stripped)

        keywords = " ".join(sparse.tokenize(normalized))
        if keywords and keywords.lower() != normalized.lower():
            expansions.append(keywords)

        deduped: list[str] = []
        seen = {normalized.lower()}
        for expansion in expansions:
            if expansion.lower() not in seen and expansion.strip():
                deduped.append(expansion)
                seen.add(expansion.lower())
        return deduped[:variants]

    # -------------------------------------------------------------- retrieve
    def retrieve(
        self,
        query: str,
        strategy: str | None = None,
        mode: str | None = None,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        expand: bool | None = None,
    ) -> RetrievalResult:
        config = self.config
        strategy = strategy or config.chunking.strategy
        mode = mode or config.retrieval.strategy
        top_k = top_k or config.retrieval.top_k
        expand = config.retrieval.query_expansion if expand is None else expand
        candidate_k = config.retrieval.candidate_k

        started = time.perf_counter()
        queries = [query]
        if expand:
            queries.extend(self.expand_query(query, config.retrieval.expansion_variants))

        dense_hits: list[RetrievedChunk] = []
        sparse_hits: list[RetrievedChunk] = []

        if mode in ("dense", "hybrid"):
            seen: set[str] = set()
            for variant in queries:
                for hit in self.dense_search(variant, strategy, candidate_k, doc_ids):
                    if hit.chunk_id not in seen:
                        seen.add(hit.chunk_id)
                        dense_hits.append(hit)
            # Merging several expansions loses the per-query ordering, so
            # re-sort by similarity before the ranks feed into fusion.
            dense_hits.sort(key=lambda c: -c.score)
            dense_hits = dense_hits[:candidate_k]

        if mode in ("sparse", "hybrid"):
            seen = set()
            for variant in queries:
                for hit in self.sparse_search(variant, strategy, candidate_k):
                    if hit.chunk_id not in seen:
                        seen.add(hit.chunk_id)
                        sparse_hits.append(hit)
            sparse_hits.sort(key=lambda c: -c.score)
            sparse_hits = sparse_hits[:candidate_k]

        if mode == "dense":
            chunks = dense_hits[:top_k]
            for rank, chunk in enumerate(chunks, start=1):
                chunk.dense_rank, chunk.dense_score = rank, chunk.score
        elif mode == "sparse":
            chunks = sparse_hits[:top_k]
            for rank, chunk in enumerate(chunks, start=1):
                chunk.sparse_rank, chunk.sparse_score = rank, chunk.score
        else:
            chunks = merge_results(dense_hits, sparse_hits, k=config.retrieval.rrf_k, top_k=top_k)

        result = RetrievalResult(
            query=query,
            chunks=chunks,
            mode=mode,
            strategy=strategy,
            expanded_queries=queries[1:],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        LOG.info(
            "retrieved %d chunk(s) for %r [%s/%s] in %.0f ms (confidence %.2f)",
            len(chunks),
            query[:60],
            mode,
            strategy,
            result.latency_ms,
            result.confidence,
        )
        return result


_RETRIEVER: Retriever | None = None


def get_retriever(config: Config | None = None) -> Retriever:
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = Retriever(config)
    return _RETRIEVER
