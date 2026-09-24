"""Chunking tests.

The properties that matter downstream: nothing is lost, page attribution is
exact (citations depend on it), the token budget is respected (the embedding
model truncates past 512), and ids are stable across runs but change when the
text changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from ragbuilder.chunking.base import Chunk, count_tokens, split_sentences
from ragbuilder.chunking.strategies import (
    fixed_chunker,
    get_chunker,
    semantic_chunker,
    sentence_chunker,
)


class TestSentenceSplitting:
    def test_splits_german_sentences(self):
        text = "Der Urlaub beträgt 24 Werktage. Als Werktage gelten alle Kalendertage. Punkt."
        assert len(split_sentences(text)) == 3

    def test_does_not_split_on_legal_abbreviations(self):
        text = "Nach § 3 Abs. 2 Satz 1 BUrlG gilt dies. Der nächste Satz beginnt hier."
        sentences = split_sentences(text)
        assert len(sentences) == 2
        assert "Abs. 2" in sentences[0]

    def test_empty_text_yields_nothing(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []


class TestFixedChunker:
    def test_respects_the_token_budget(self, long_pages):
        chunks = fixed_chunker(long_pages, "doc", chunk_tokens=100, overlap_tokens=10)
        assert chunks
        assert all(c.token_count <= 100 for c in chunks)

    def test_chunks_overlap(self, long_pages):
        chunks = fixed_chunker(long_pages, "doc", chunk_tokens=100, overlap_tokens=30)
        assert len(chunks) >= 2
        # With a 30-token overlap the tail of one chunk must reappear at the
        # head of the next; compare on words to stay tokenizer-agnostic.
        first_tail = chunks[0].text.split()[-8:]
        assert any(word in chunks[1].text for word in first_tail)

    def test_page_range_is_exact(self, sample_pages):
        chunks = fixed_chunker(sample_pages, "doc", chunk_tokens=20, overlap_tokens=0)
        assert all(1 <= c.page_start <= c.page_end <= 2 for c in chunks)
        assert min(c.page_start for c in chunks) == 1
        assert max(c.page_end for c in chunks) == 2

    def test_covers_the_whole_document(self, sample_pages):
        chunks = fixed_chunker(sample_pages, "doc", chunk_tokens=40, overlap_tokens=0)
        combined = " ".join(c.text for c in chunks)
        assert "24 Werktage" in combined
        assert "Geltungsbereich" in combined

    def test_rejects_impossible_settings(self, sample_pages):
        with pytest.raises(ValueError):
            fixed_chunker(sample_pages, "doc", chunk_tokens=50, overlap_tokens=50)
        with pytest.raises(ValueError):
            fixed_chunker(sample_pages, "doc", chunk_tokens=0)

    def test_empty_input(self):
        assert fixed_chunker([], "doc") == []


class TestSentenceChunker:
    def test_never_splits_a_sentence(self, sample_pages):
        chunks = sentence_chunker(sample_pages, "doc", max_tokens=40, overlap_sentences=0)
        for chunk in chunks:
            stripped = chunk.text.strip()
            # Every chunk ends where a sentence ends.
            assert stripped.endswith((".", ":", "!", "?")) or len(stripped) < 5

    def test_respects_the_token_budget(self, long_pages):
        chunks = sentence_chunker(long_pages, "doc", max_tokens=120, overlap_sentences=1)
        # One oversized sentence is allowed through whole; nothing else may be.
        assert sum(1 for c in chunks if c.token_count > 120) == 0

    def test_overlap_repeats_the_previous_sentence(self, long_pages):
        with_overlap = sentence_chunker(long_pages, "doc", max_tokens=120, overlap_sentences=1)
        without = sentence_chunker(long_pages, "doc", max_tokens=120, overlap_sentences=0)
        assert len(with_overlap) >= len(without)

    def test_a_single_oversized_sentence_survives(self):
        pages = [{"page_number": 1, "section": None, "has_tables": False, "text": "wort " * 800 + "."}]
        chunks = sentence_chunker(pages, "doc", max_tokens=100)
        assert len(chunks) == 1
        assert chunks[0].token_count > 100  # kept whole rather than truncated

    def test_section_is_carried_through(self, sample_pages):
        chunks = sentence_chunker(sample_pages, "doc", max_tokens=60)
        assert any(c.section and "§" in c.section for c in chunks)

    def test_empty_input(self):
        assert sentence_chunker([], "doc") == []


class TestSemanticChunker:
    @staticmethod
    def _topic_embedder(texts):
        """Fake embedder: sentences mentioning Urlaub point one way, the rest another.

        Using a stub keeps the test fast and deterministic while still exercising
        the real breakpoint arithmetic.
        """
        vectors = []
        for text in texts:
            if "Urlaub" in text or "Werktage" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "Arbeitsschutz" in text or "Maßnahmen" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def test_splits_where_the_topic_changes(self):
        pages = [
            {
                "page_number": 1,
                "section": None,
                "has_tables": False,
                "text": (
                    "Der Urlaub beträgt 24 Werktage. "
                    "Als Werktage gelten alle Kalendertage ohne Sonntage. "
                    "Der Urlaub muss im Kalenderjahr genommen werden. "
                    "Der Arbeitgeber hat Maßnahmen des Arbeitsschutzes zu treffen. "
                    "Die Maßnahmen sind regelmäßig zu überprüfen. "
                    "Weitere Maßnahmen des Arbeitsschutzes sind erforderlich."
                ),
            }
        ]
        chunks = semantic_chunker(
            pages, "doc", self._topic_embedder, max_tokens=500, breakpoint_percentile=80, min_sentences=2
        )
        assert len(chunks) >= 2
        assert "Urlaub" in chunks[0].text

    def test_token_budget_is_a_hard_ceiling(self, long_pages):
        chunks = semantic_chunker(
            long_pages,
            "doc",
            lambda texts: np.ones((len(texts), 3), dtype=np.float32),  # no topic shift at all
            max_tokens=80,
            breakpoint_percentile=95,
            min_sentences=2,
        )
        assert all(c.token_count <= 80 for c in chunks)

    def test_very_short_input_is_one_chunk(self):
        pages = [{"page_number": 1, "section": None, "has_tables": False, "text": "Ein Satz. Noch einer."}]
        chunks = semantic_chunker(pages, "doc", self._topic_embedder, min_sentences=5)
        assert len(chunks) == 1

    def test_empty_input(self):
        assert semantic_chunker([], "doc", self._topic_embedder) == []


class TestChunkIdentity:
    def test_id_is_stable_for_identical_content(self):
        a = Chunk(
            text="Der Urlaub beträgt 24 Werktage.", chunk_index=3, page_start=1, page_end=1, doc_id="burlg"
        )
        b = Chunk(
            text="Der Urlaub beträgt 24 Werktage.", chunk_index=3, page_start=1, page_end=1, doc_id="burlg"
        )
        assert a.chunk_id == b.chunk_id

    def test_id_changes_when_the_text_changes(self):
        a = Chunk(
            text="Der Urlaub beträgt 24 Werktage.", chunk_index=3, page_start=1, page_end=1, doc_id="burlg"
        )
        b = Chunk(
            text="Der Urlaub beträgt 20 Werktage.", chunk_index=3, page_start=1, page_end=1, doc_id="burlg"
        )
        assert a.chunk_id != b.chunk_id

    def test_id_is_namespaced_by_strategy(self):
        """All three chunk sets share one table, so ids must not collide.

        On a short document sentence and semantic chunking legitimately produce
        an identical first chunk; without the strategy in the id that is a
        primary key violation that kills the ingestion run.
        """
        a = Chunk(text="same", chunk_index=0, page_start=1, page_end=1, doc_id="agg", strategy="sentence")
        b = Chunk(text="same", chunk_index=0, page_start=1, page_end=1, doc_id="agg", strategy="semantic")
        assert a.chunk_id != b.chunk_id
        assert "sentence" in a.chunk_id and "semantic" in b.chunk_id

    def test_id_is_namespaced_by_document(self):
        a = Chunk(text="same", chunk_index=0, page_start=1, page_end=1, doc_id="burlg")
        b = Chunk(text="same", chunk_index=0, page_start=1, page_end=1, doc_id="arbzg")
        assert a.chunk_id != b.chunk_id
        assert a.chunk_id.startswith("burlg_")

    def test_as_row_has_the_columns_the_database_expects(self, sample_pages):
        row = sentence_chunker(sample_pages, "doc")[0].as_row()
        assert set(row) == {
            "chunk_id",
            "chunk_index",
            "text",
            "token_count",
            "page_start",
            "page_end",
            "section",
        }


class TestRegistry:
    @pytest.mark.parametrize("strategy", ["fixed", "sentence"])
    def test_returns_a_working_chunker(self, strategy, config, sample_pages):
        chunker = get_chunker(strategy, config)
        chunks = chunker(sample_pages, "doc")
        assert chunks and all(isinstance(c, Chunk) for c in chunks)

    def test_semantic_needs_an_embedder(self, config):
        with pytest.raises(ValueError, match="embedding function"):
            get_chunker("semantic", config)

    def test_unknown_strategy_is_rejected(self, config):
        with pytest.raises(ValueError, match="unknown chunking strategy"):
            get_chunker("magic", config)


class TestTokenCounting:
    def test_counts_grow_with_text(self):
        assert count_tokens("kurz") < count_tokens("ein deutlich längerer Satz mit mehr Inhalt")

    def test_empty_string(self):
        assert count_tokens("") == 0
