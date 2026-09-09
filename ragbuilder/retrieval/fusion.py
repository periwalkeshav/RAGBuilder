"""Reciprocal Rank Fusion.

RRF combines ranked lists by *position*, never by score::

    score(d) = Σ over lists  1 / (k + rank_list(d))        rank is 1-based

Why that beats simply merging scores, which is the question this project exists
to answer well:

* **The scores are not comparable.** A cosine similarity of 0.83 and a BM25
  score of 14.2 live on different scales with different distributions. Any
  attempt to merge them needs normalisation, and every normalisation scheme
  (min-max, z-score) is sensitive to the score spread of the particular query -
  so the weighting silently changes from question to question.
* **Rank is the robust statistic.** RRF only asks "did this retriever put the
  document near the top", which is exactly the signal worth keeping.
* **Agreement is rewarded.** A chunk ranked 3rd by both retrievers outranks one
  ranked 1st by a single retriever and missing from the other - which is the
  behaviour you want, because the two retrievers fail in uncorrelated ways.

``k`` (60 by default, from Cormack et al. 2009) damps the influence of the very
top ranks so that one retriever's confident mistake cannot dominate the fusion.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class RetrievedChunk:
    """A chunk plus how it was found, carried through fusion into the prompt."""

    chunk_id: str
    score: float = 0.0
    text: str = ""
    doc_id: str = ""
    title: str = ""
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_url: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    retriever: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end or self.page_end is None
            else f"pp. {self.page_start}-{self.page_end}"
        )
        parts = [self.title or self.doc_id]
        if self.section:
            parts.append(self.section)
        parts.append(pages)
        return ", ".join(parts)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one ranking, best first.

    ``weights`` lets one retriever count for more than another; the default
    weights every list equally, which is the standard formulation.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")

    scores: dict[str, float] = defaultdict(float)
    for ranked, weight in zip(ranked_lists, weights):
        for position, item_id in enumerate(ranked, start=1):
            scores[item_id] += weight / (k + position)

    # Ties break on id so the output is deterministic - important for tests and
    # for reproducible evaluation runs.
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


def ranks_of(ranked: Iterable[str]) -> dict[str, int]:
    return {item_id: position for position, item_id in enumerate(ranked, start=1)}


def merge_results(
    dense: Sequence[RetrievedChunk],
    sparse: Sequence[RetrievedChunk],
    k: int = 60,
    top_k: int = 5,
    weights: Sequence[float] | None = None,
) -> list[RetrievedChunk]:
    """Fuse dense and sparse hits into one ranked list of ``RetrievedChunk``."""
    dense_ranks = ranks_of(c.chunk_id for c in dense)
    sparse_ranks = ranks_of(c.chunk_id for c in sparse)

    by_id: dict[str, RetrievedChunk] = {}
    for chunk in list(sparse) + list(dense):  # dense wins on conflict
        by_id.setdefault(chunk.chunk_id, chunk)
        if chunk.chunk_id in by_id:
            existing = by_id[chunk.chunk_id]
            if chunk.text and not existing.text:
                existing.text = chunk.text

    fused = reciprocal_rank_fusion(
        [[c.chunk_id for c in dense], [c.chunk_id for c in sparse]], k=k, weights=weights
    )

    dense_scores = {c.chunk_id: c.score for c in dense}
    sparse_scores = {c.chunk_id: c.score for c in sparse}

    output: list[RetrievedChunk] = []
    for chunk_id, score in fused[:top_k]:
        chunk = by_id[chunk_id]
        chunk.score = score
        chunk.dense_rank = dense_ranks.get(chunk_id)
        chunk.sparse_rank = sparse_ranks.get(chunk_id)
        chunk.dense_score = dense_scores.get(chunk_id)
        chunk.sparse_score = sparse_scores.get(chunk_id)
        chunk.retriever = (
            "hybrid"
            if chunk.dense_rank and chunk.sparse_rank
            else ("dense" if chunk.dense_rank else "sparse")
        )
        output.append(chunk)
    return output
