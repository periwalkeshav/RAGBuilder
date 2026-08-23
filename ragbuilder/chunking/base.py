"""Shared chunking primitives: tokens, sentences, and the ``Chunk`` record.

Every chunker produces the same ``Chunk`` shape, so the retrieval, embedding and
evaluation layers never need to know which strategy produced a row. That is what
makes the three-way comparison in ``make evaluate`` a fair test.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

LOG = logging.getLogger("ragbuilder.chunking")


@dataclass
class Chunk:
    """One retrievable unit of text plus the provenance needed to cite it."""

    text: str
    chunk_index: int
    page_start: int
    page_end: int
    section: str | None = None
    token_count: int = 0
    doc_id: str = ""
    strategy: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Stable id: same document + strategy + position + text = same id.

        Hashing the text (not just the index) means an edit upstream produces a
        new id, so a stale vector can never masquerade as a current one.

        The strategy is part of the id because all three chunk sets share one
        table and one Qdrant collection. On a short statute, sentence and
        semantic chunking legitimately produce an *identical* first chunk - and
        without the strategy in the id that is a primary key collision that
        aborts the whole ingestion run.
        """
        digest = hashlib.sha1(
            f"{self.doc_id}|{self.strategy}|{self.chunk_index}|{self.text}".encode("utf-8")
        ).hexdigest()[:16]
        prefix = f"{self.doc_id}_{self.strategy}" if self.strategy else self.doc_id
        return f"{prefix}_{self.chunk_index:05d}_{digest}"

    def as_row(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "token_count": self.token_count,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section": self.section,
        }


# ----------------------------------------------------------------- tokenizing
@lru_cache(maxsize=1)
def _encoder():
    """cl100k_base as a stable token yardstick.

    It is not the tokenizer e5 uses, but chunk sizes only need to be *consistent*
    and roughly calibrated - and cl100k is the number everyone quotes, which
    makes "512 tokens" mean the same thing here as in every other RAG write-up.
    """
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail ingestion
        LOG.warning("tiktoken unavailable (%s); falling back to whitespace token counting", exc)
        return None


def count_tokens(text: str) -> int:
    encoder = _encoder()
    if encoder is None:
        return max(1, len(text.split()))
    return len(encoder.encode(text, disallowed_special=()))


def encode_tokens(text: str) -> list[int] | list[str]:
    encoder = _encoder()
    if encoder is None:
        return text.split()
    return encoder.encode(text, disallowed_special=())


def decode_tokens(tokens: Sequence) -> str:
    encoder = _encoder()
    if encoder is None:
        return " ".join(str(t) for t in tokens)
    return encoder.decode(list(tokens))


# ------------------------------------------------------------------ sentences
_ABBREVIATIONS = (
    # German legal text is full of these, and a naive "split on ." shreds it.
    "Abs",
    "Nr",
    "lit",
    "vgl",
    "bzw",
    "ggf",
    "insb",
    "sog",
    "z.B",
    "u.a",
    "d.h",
    "Art",
    "S",
    "Ziff",
    "Buchst",
    "Satz",
    "Halbs",
    "evtl",
    "inkl",
    "ca",
    "Nrn",
    "BGBl",
    "i.V.m",
    "i.S.d",
    "i.d.R",
    "u.U",
    "z.T",
)

_SENTENCE_FALLBACK = re.compile(r"(?<=[.!?:;])\s+(?=[A-ZÄÖÜ§(])")


@lru_cache(maxsize=4)
def _nltk_tokenizer(language: str):
    """Return ``sent_tokenize`` if punkt is usable for ``language``, else None.

    ``nltk.sent_tokenize`` is used rather than loading the model by resource
    path because the path moved between NLTK versions: 3.8 ships the pickled
    ``punkt`` model, 3.9 replaced it with the pickle-free ``punkt_tab``. Calling
    the public function and testing it on a real sentence works on both, and
    proves the data is present rather than assuming it.
    """
    try:
        import nltk

        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.sent_tokenize("Ein Satz. Noch einer.", language=language)
                return nltk.sent_tokenize
            except LookupError:
                nltk.download(resource, quiet=True)
        nltk.sent_tokenize("Ein Satz. Noch einer.", language=language)
        return nltk.sent_tokenize
    except Exception as exc:  # noqa: BLE001
        LOG.warning("NLTK punkt for %r unavailable (%s); using the regex splitter", language, exc)
        return None


def split_sentences(text: str, language: str = "german") -> list[str]:
    """Split into sentences, preferring NLTK's punkt model for the language."""
    if not text.strip():
        return []

    tokenizer = _nltk_tokenizer(language)
    if tokenizer is not None:
        try:
            sentences = tokenizer(text, language=language)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("punkt failed, falling back: %s", exc)
            sentences = _SENTENCE_FALLBACK.split(text)
    else:
        sentences = _SENTENCE_FALLBACK.split(text)

    merged: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # Re-join a split that landed straight after a known abbreviation.
        if merged and any(merged[-1].rstrip().endswith(f"{abbr}.") for abbr in _ABBREVIATIONS):
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    return merged


# --------------------------------------------------------------- page mapping
@dataclass
class Piece:
    """A sentence (or token run) together with the page it came from."""

    text: str
    page: int
    section: str | None


def pages_to_sentences(pages: Sequence[dict], language: str = "german") -> list[Piece]:
    """Flatten parsed pages into sentences that remember page *and* section.

    Section tracking happens per sentence, not per page. A single page of a
    statute routinely spans several paragraphs, so attributing every chunk on it
    to whichever heading appeared first produces citations that point at the
    wrong paragraph - the one failure mode that makes a citation worse than
    useless, because it looks authoritative and is wrong.
    """
    from ragbuilder.ingestion.parser import detect_section

    pieces: list[Piece] = []
    current: str | None = None

    for page in pages:
        # A page with no heading of its own continues the previous section.
        current = page.get("section") or current
        for sentence in split_sentences(page["text"], language=language):
            heading = detect_section(sentence.split("\n")[0])
            if heading:
                current = heading
            pieces.append(Piece(text=sentence, page=int(page["page_number"]), section=current))
    return pieces


def build_chunk(
    pieces: Sequence[Piece],
    index: int,
    doc_id: str,
    joiner: str = " ",
) -> Chunk:
    text = joiner.join(p.text for p in pieces).strip()
    return Chunk(
        text=text,
        chunk_index=index,
        page_start=min(p.page for p in pieces),
        page_end=max(p.page for p in pieces),
        section=next((p.section for p in pieces if p.section), None),
        token_count=count_tokens(text),
        doc_id=doc_id,
    )


Chunker = Callable[[Sequence[dict], str], list[Chunk]]
