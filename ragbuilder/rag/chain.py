"""The RAG chain: retrieve -> ground -> generate -> attribute.

Two behaviours here are what separate this from a demo:

* **It refuses.** If retrieval confidence is below ``rag.min_confidence`` the
  chain never calls the LLM. There is no prompt wording that reliably stops a
  7B model inventing an answer when the context is empty, so the guard belongs
  before generation, not inside it.
* **It reports what it used.** Every answer comes back with the sources, their
  pages, which retriever found them, and which of them the answer actually
  cited. An uncited source is visible, and so is a claim with no citation.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Iterator, Sequence

from ragbuilder import db
from ragbuilder.config import Config, get_config
from ragbuilder.llm.client import LLMError, get_client
from ragbuilder.llm.prompts import (
    REFUSAL_DE,
    REFUSAL_EN,
    build_qa_prompt,
    build_system_prompt,
    detect_language,
    extract_citations,
    is_refusal,
)
from ragbuilder.retrieval.fusion import RetrievedChunk
from ragbuilder.retrieval.retriever import RetrievalResult, get_retriever

LOG = logging.getLogger("ragbuilder.chain")


@dataclass
class Turn:
    question: str
    answer: str


@dataclass
class Source:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str | None
    page_start: int | None
    page_end: int | None
    citation: str
    excerpt: str
    score: float
    retriever: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    cited: bool = False

    @classmethod
    def from_chunk(cls, index: int, chunk: RetrievedChunk, excerpt_chars: int = 600) -> "Source":
        excerpt = chunk.text or ""
        if len(excerpt) > excerpt_chars:
            excerpt = excerpt[:excerpt_chars].rsplit(" ", 1)[0] + " […]"
        return cls(
            index=index,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            title=chunk.title,
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            citation=chunk.citation,
            excerpt=excerpt,
            score=round(chunk.score, 6),
            retriever=chunk.retriever,
            dense_rank=chunk.dense_rank,
            sparse_rank=chunk.sparse_rank,
        )


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    strategy: str = ""
    retrieval_mode: str = ""
    model: str = ""
    language: str = ""
    expanded_queries: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def cited_source_count(self) -> int:
        return sum(1 for s in self.sources if s.cited)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["cited_source_count"] = self.cited_source_count
        return payload


class ConversationMemory:
    """The last N turns, used only to resolve references like "und danach?".

    History is deliberately *not* a retrieval source: letting a previous answer
    act as evidence is how a small early mistake becomes a confidently repeated
    one three turns later.
    """

    def __init__(self, max_turns: int = 3) -> None:
        self.max_turns = max_turns
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def add(self, question: str, answer: str) -> None:
        self._turns.append(Turn(question=question, answer=answer))

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def as_dicts(self) -> list[dict]:
        return [asdict(t) for t in self._turns]


class RagChain:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.retriever = get_retriever(self.config)
        self.llm = get_client(self.config)
        self._memories: dict[str, ConversationMemory] = {}

    def memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(self.config.rag.memory_turns)
        return self._memories[session_id]

    def clear_memory(self, session_id: str) -> None:
        self._memories.pop(session_id, None)

    # ------------------------------------------------------------- internals
    def _prepare(
        self,
        question: str,
        strategy: str | None,
        mode: str | None,
        top_k: int | None,
        doc_ids: Sequence[str] | None,
    ) -> tuple[RetrievalResult, list[Source], str]:
        retrieval = self.retriever.retrieve(
            question, strategy=strategy, mode=mode, top_k=top_k, doc_ids=doc_ids
        )
        sources = [Source.from_chunk(i, c) for i, c in enumerate(retrieval.chunks, start=1)]
        language = detect_language(question)
        return retrieval, sources, language

    def _refusal_answer(
        self, question: str, retrieval: RetrievalResult, sources: list[Source], language: str, started: float
    ) -> RagAnswer:
        LOG.info(
            "refusing: confidence %.2f below threshold %.2f",
            retrieval.confidence,
            self.config.rag.min_confidence,
        )
        return RagAnswer(
            question=question,
            answer=REFUSAL_DE if language == "German" else REFUSAL_EN,
            sources=sources,
            confidence=retrieval.confidence,
            refused=True,
            strategy=retrieval.strategy,
            retrieval_mode=retrieval.mode,
            model="",
            language=language,
            expanded_queries=retrieval.expanded_queries,
            retrieval_ms=round(retrieval.latency_ms, 1),
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    # ------------------------------------------------------------------ ask
    def ask(
        self,
        question: str,
        session_id: str = "default",
        strategy: str | None = None,
        mode: str | None = None,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        use_memory: bool = True,
        log: bool = True,
    ) -> RagAnswer:
        started = time.perf_counter()
        retrieval, sources, language = self._prepare(question, strategy, mode, top_k, doc_ids)

        if not retrieval.chunks or retrieval.confidence < self.config.rag.min_confidence:
            answer = self._refusal_answer(question, retrieval, sources, language, started)
            if log:
                self._log(answer)
            return answer

        memory = self.memory(session_id)
        system = build_system_prompt(self.config.rag.language, question)
        prompt = build_qa_prompt(
            question,
            retrieval.chunks,
            history=memory.turns if use_memory else [],
            max_context_chars=self.config.rag.max_context_chars,
        )

        generation_started = time.perf_counter()
        try:
            completion = self.llm.generate(prompt, system=system)
        except LLMError as exc:
            LOG.error("generation failed: %s", exc)
            raise
        generation_ms = (time.perf_counter() - generation_started) * 1000

        cited = set(extract_citations(completion.text, len(sources)))
        for source in sources:
            source.cited = source.index in cited

        answer = RagAnswer(
            question=question,
            answer=completion.text,
            sources=sources,
            confidence=retrieval.confidence,
            refused=is_refusal(completion.text),
            strategy=retrieval.strategy,
            retrieval_mode=retrieval.mode,
            model=completion.model,
            language=language,
            expanded_queries=retrieval.expanded_queries,
            retrieval_ms=round(retrieval.latency_ms, 1),
            generation_ms=round(generation_ms, 1),
            total_ms=round((time.perf_counter() - started) * 1000, 1),
        )

        if use_memory and not answer.refused:
            memory.add(question, completion.text)
        if log:
            self._log(answer, session_id)
        return answer

    # --------------------------------------------------------------- stream
    def stream(
        self,
        question: str,
        session_id: str = "default",
        strategy: str | None = None,
        mode: str | None = None,
        top_k: int | None = None,
    ) -> Iterator[dict]:
        """Yield ``sources`` first, then ``token`` events, then ``done``.

        Sources go out before the first token so the UI can render citations
        while the model is still writing.
        """
        started = time.perf_counter()
        retrieval, sources, language = self._prepare(question, strategy, mode, top_k, None)

        yield {
            "type": "sources",
            "sources": [asdict(s) for s in sources],
            "confidence": retrieval.confidence,
            "mode": retrieval.mode,
            "strategy": retrieval.strategy,
        }

        if not retrieval.chunks or retrieval.confidence < self.config.rag.min_confidence:
            refusal = REFUSAL_DE if language == "German" else REFUSAL_EN
            yield {"type": "token", "token": refusal}
            yield {
                "type": "done",
                "refused": True,
                "confidence": retrieval.confidence,
                "total_ms": round((time.perf_counter() - started) * 1000, 1),
            }
            return

        memory = self.memory(session_id)
        system = build_system_prompt(self.config.rag.language, question)
        prompt = build_qa_prompt(
            question,
            retrieval.chunks,
            history=memory.turns,
            max_context_chars=self.config.rag.max_context_chars,
        )

        collected: list[str] = []
        for token in self.llm.stream(prompt, system=system):
            collected.append(token)
            yield {"type": "token", "token": token}

        text = "".join(collected).strip()
        cited = set(extract_citations(text, len(sources)))
        if not is_refusal(text):
            memory.add(question, text)

        total_ms = (time.perf_counter() - started) * 1000
        yield {
            "type": "done",
            "refused": is_refusal(text),
            "confidence": retrieval.confidence,
            "cited": sorted(cited),
            "model": self.llm.resolve_model(),
            "total_ms": round(total_ms, 1),
        }
        self._log(
            RagAnswer(
                question=question,
                answer=text,
                sources=sources,
                confidence=retrieval.confidence,
                refused=is_refusal(text),
                strategy=retrieval.strategy,
                retrieval_mode=retrieval.mode,
                model=self.llm.resolve_model(),
                language=language,
                retrieval_ms=round(retrieval.latency_ms, 1),
                total_ms=round(total_ms, 1),
            ),
            session_id,
        )

    # ------------------------------------------------------------- logging
    @staticmethod
    def _log(answer: RagAnswer, session_id: str = "default") -> None:
        try:
            db.log_query(
                {
                    "session_id": session_id,
                    "question": answer.question,
                    "answer": answer.answer,
                    "strategy": answer.strategy,
                    "retrieval_mode": answer.retrieval_mode,
                    "confidence": answer.confidence,
                    "refused": answer.refused,
                    "chunk_ids": [s.chunk_id for s in answer.sources],
                    "latency_ms": int(answer.total_ms),
                    "llm_model": answer.model,
                }
            )
        except Exception as exc:  # noqa: BLE001 - logging must never break answering
            LOG.warning("could not write query log: %s", exc)


_CHAIN: RagChain | None = None


def get_chain(config: Config | None = None) -> RagChain:
    global _CHAIN
    if _CHAIN is None:
        _CHAIN = RagChain(config)
    return _CHAIN
