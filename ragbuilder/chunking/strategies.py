"""The three chunking strategies, and the registry that selects between them.

    fixed     512 tokens, 50-token overlap, ignores structure
    sentence  packs whole sentences up to a token budget, 1-sentence overlap
    semantic  starts a new chunk where consecutive sentences stop being similar

The interesting comparison is *fixed vs sentence* on a structured corpus. Fixed
chunking will happily cut "§ 3 Absatz 2 Satz 1" in half; the retrieved fragment
then looks relevant to the embedding model but is useless to the reader, and it
is the single most common reason a first RAG prototype gives confident nonsense.
``make evaluate`` scores all three so the claim is measured, not asserted.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

from ragbuilder.chunking.base import (
    Chunk,
    Piece,
    build_chunk,
    count_tokens,
    decode_tokens,
    encode_tokens,
    pages_to_sentences,
)

LOG = logging.getLogger("ragbuilder.chunking")

EmbedFn = Callable[[list[str]], np.ndarray]


# ------------------------------------------------------------------- fixed
def fixed_chunker(
    pages: Sequence[dict],
    doc_id: str,
    chunk_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[Chunk]:
    """Sliding token window over the whole document.

    Page attribution is exact rather than estimated: each page is tokenised
    separately and every token carries its page number, so a chunk that spans a
    page break reports the real range.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    tokens: list = []
    token_pages: list[int] = []
    token_sections: list[str | None] = []

    for page in pages:
        encoded = encode_tokens(page["text"])
        tokens.extend(encoded)
        token_pages.extend([int(page["page_number"])] * len(encoded))
        token_sections.extend([page.get("section")] * len(encoded))

    if not tokens:
        return []

    stride = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    index = 0

    for start in range(0, len(tokens), stride):
        window = tokens[start : start + chunk_tokens]
        if not window:
            break
        text = decode_tokens(window).strip()
        if not text:
            continue
        pages_in_window = token_pages[start : start + chunk_tokens]
        sections = [s for s in token_sections[start : start + chunk_tokens] if s]
        chunks.append(
            Chunk(
                text=text,
                chunk_index=index,
                page_start=min(pages_in_window),
                page_end=max(pages_in_window),
                section=sections[0] if sections else None,
                token_count=len(window),
                doc_id=doc_id,
            )
        )
        index += 1
        if start + chunk_tokens >= len(tokens):
            break

    return chunks


# ---------------------------------------------------------------- sentence
def sentence_chunker(
    pages: Sequence[dict],
    doc_id: str,
    max_tokens: int = 512,
    overlap_sentences: int = 1,
    language: str = "german",
) -> list[Chunk]:
    """Greedily pack whole sentences until the token budget is reached.

    No sentence is ever split, so a retrieved chunk always starts and ends at a
    grammatical boundary. ``overlap_sentences`` carries the tail of one chunk
    into the head of the next, which preserves the antecedent of a pronoun or a
    cross-reference that would otherwise be orphaned.
    """
    pieces = pages_to_sentences(pages, language=language)
    if not pieces:
        return []

    chunks: list[Chunk] = []
    current: list[Piece] = []
    current_tokens = 0
    index = 0

    for piece in pieces:
        piece_tokens = count_tokens(piece.text)

        # A single sentence longer than the budget still has to go somewhere.
        if piece_tokens > max_tokens:
            if current:
                chunks.append(build_chunk(current, index, doc_id))
                index += 1
                current, current_tokens = [], 0
            chunks.append(build_chunk([piece], index, doc_id))
            index += 1
            continue

        if current_tokens + piece_tokens > max_tokens and current:
            chunks.append(build_chunk(current, index, doc_id))
            index += 1
            tail = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = list(tail)
            current_tokens = sum(count_tokens(p.text) for p in current)

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append(build_chunk(current, index, doc_id))

    return chunks


# ---------------------------------------------------------------- semantic
def semantic_chunker(
    pages: Sequence[dict],
    doc_id: str,
    embed_fn: EmbedFn,
    max_tokens: int = 512,
    breakpoint_percentile: int = 85,
    min_sentences: int = 3,
    language: str = "german",
) -> list[Chunk]:
    """Split where the topic changes rather than where the token counter runs out.

    Consecutive sentences are embedded and the cosine *distance* between each
    adjacent pair is measured. A distance above the Nth percentile is treated as
    a topic boundary. The token budget still applies as a hard ceiling, because
    a section with no internal topic shift would otherwise produce one enormous
    chunk that the embedding model truncates at 512 tokens anyway.
    """
    pieces = pages_to_sentences(pages, language=language)
    if not pieces:
        return []
    if len(pieces) <= min_sentences:
        return [build_chunk(pieces, 0, doc_id)]

    vectors = embed_fn([p.text for p in pieces])
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms

    similarities = np.sum(unit[:-1] * unit[1:], axis=1)
    distances = 1.0 - similarities

    threshold = float(np.percentile(distances, breakpoint_percentile))
    LOG.debug(
        "%s: %d sentences, distance p%d = %.4f (min %.4f, max %.4f)",
        doc_id,
        len(pieces),
        breakpoint_percentile,
        threshold,
        distances.min(),
        distances.max(),
    )

    chunks: list[Chunk] = []
    current: list[Piece] = [pieces[0]]
    current_tokens = count_tokens(pieces[0].text)
    index = 0

    for position, piece in enumerate(pieces[1:]):
        piece_tokens = count_tokens(piece.text)
        topic_shift = distances[position] >= threshold and len(current) >= min_sentences
        over_budget = current_tokens + piece_tokens > max_tokens

        if (topic_shift or over_budget) and current:
            chunks.append(build_chunk(current, index, doc_id))
            index += 1
            current, current_tokens = [], 0

        current.append(piece)
        current_tokens += piece_tokens

    if current:
        chunks.append(build_chunk(current, index, doc_id))

    return chunks


# ---------------------------------------------------------------- registry
def get_chunker(strategy: str, config, embed_fn: EmbedFn | None = None):
    """Return a ``(pages, doc_id) -> list[Chunk]`` callable for ``strategy``."""
    chunking = config.chunking

    if strategy == "fixed":
        return lambda pages, doc_id: fixed_chunker(
            pages,
            doc_id,
            chunk_tokens=chunking.fixed.chunk_tokens,
            overlap_tokens=chunking.fixed.overlap_tokens,
        )

    if strategy == "sentence":
        return lambda pages, doc_id: sentence_chunker(
            pages,
            doc_id,
            max_tokens=chunking.sentence.max_tokens,
            overlap_sentences=chunking.sentence.overlap_sentences,
            language=chunking.sentence.language,
        )

    if strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embedding function")
        return lambda pages, doc_id: semantic_chunker(
            pages,
            doc_id,
            embed_fn,
            max_tokens=chunking.semantic.max_tokens,
            breakpoint_percentile=chunking.semantic.breakpoint_percentile,
            min_sentences=chunking.semantic.min_sentences,
            language=chunking.sentence.language,
        )

    raise ValueError(f"unknown chunking strategy {strategy!r}; expected fixed, sentence or semantic")


STRATEGIES = ("fixed", "sentence", "semantic")
