"""Configuration loading: nested sections, env overrides, type coercion.

The nested-section test is not decoration. When ``config.chunking.sentence``
silently stays a plain dict, nothing raises until a chunker reaches for
``.max_tokens`` halfway through ingesting a corpus.
"""

from __future__ import annotations

import textwrap
from dataclasses import is_dataclass

import pytest

from ragbuilder.config import (
    ChunkingConfig,
    Config,
    EmbeddingConfig,
    SentenceChunkingConfig,
    load_config,
)


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        textwrap.dedent(
            """
            corpus:
              name: german_labour_law
              pdf_dir: data/pdfs
            embedding:
              model: intfloat/multilingual-e5-small
              batch_size: 16
              normalize: true
            chunking:
              strategy: semantic
              fixed:
                chunk_tokens: 256
                overlap_tokens: 25
              sentence:
                max_tokens: 400
                overlap_sentences: 2
                language: german
              semantic:
                max_tokens: 300
                breakpoint_percentile: 90
                min_sentences: 4
            retrieval:
              strategy: dense
              top_k: 8
              rrf_k: 30
              query_expansion: false
            llm:
              model: llama3:8b
              temperature: 0.5
            evaluation:
              strategies: [fixed, sentence]
            """
        ),
        encoding="utf-8",
    )
    return path


class TestNestedSections:
    def test_every_section_is_a_dataclass_not_a_dict(self, config_file):
        config = load_config(config_file)
        for section in (
            config.corpus,
            config.embedding,
            config.chunking,
            config.retrieval,
            config.llm,
            config.rag,
            config.evaluation,
            config.api,
            config.vector_store,
        ):
            assert is_dataclass(section), f"{type(section)} is not a dataclass"

    def test_chunking_sub_sections_are_dataclasses(self, config_file):
        chunking = load_config(config_file).chunking
        assert isinstance(chunking.fixed.chunk_tokens, int)
        assert isinstance(chunking.sentence, SentenceChunkingConfig)
        assert chunking.sentence.max_tokens == 400
        assert chunking.semantic.breakpoint_percentile == 90

    def test_values_come_from_the_file(self, config_file):
        config = load_config(config_file)
        assert config.chunking.strategy == "semantic"
        assert config.retrieval.top_k == 8
        assert config.retrieval.rrf_k == 30
        assert config.llm.model == "llama3:8b"
        assert config.llm.temperature == pytest.approx(0.5)

    def test_unspecified_values_keep_their_defaults(self, config_file):
        config = load_config(config_file)
        assert config.retrieval.candidate_k == 20
        assert config.rag.memory_turns == 3


class TestEnvironmentOverrides:
    def test_env_beats_the_file(self, config_file, monkeypatch):
        monkeypatch.setenv("RETRIEVAL_TOP_K", "3")
        monkeypatch.setenv("LLM_MODEL", "mistral:latest")
        config = load_config(config_file)
        assert config.retrieval.top_k == 3
        assert config.llm.model == "mistral:latest"

    def test_nested_sections_map_to_prefixed_env_names(self, config_file, monkeypatch):
        monkeypatch.setenv("CHUNKING_SENTENCE_MAX_TOKENS", "128")
        assert load_config(config_file).chunking.sentence.max_tokens == 128

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("0", False),
            ("no", False),
            ("", False),
        ],
    )
    def test_booleans_are_coerced(self, config_file, monkeypatch, raw, expected):
        monkeypatch.setenv("RETRIEVAL_QUERY_EXPANSION", raw)
        assert load_config(config_file).retrieval.query_expansion is expected

    def test_integers_are_coerced(self, config_file, monkeypatch):
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
        assert load_config(config_file).embedding.batch_size == 64

    def test_lists_are_coerced_from_a_comma_string(self, config_file, monkeypatch):
        monkeypatch.setenv("EVALUATION_STRATEGIES", "fixed,semantic")
        assert load_config(config_file).evaluation.strategies == ["fixed", "semantic"]


class TestEmbeddingDimensions:
    @pytest.mark.parametrize(
        "model,dimensions",
        [
            ("intfloat/multilingual-e5-small", 384),
            ("intfloat/multilingual-e5-base", 768),
            ("intfloat/multilingual-e5-large", 1024),
        ],
    )
    def test_dimensions_follow_the_model(self, tmp_path, model, dimensions):
        """A wrong dimension count creates a collection every search then fails against."""
        path = tmp_path / "c.yaml"
        path.write_text(f"embedding:\n  model: {model}\n  dimensions: 999\n", encoding="utf-8")
        assert load_config(path).embedding.dimensions == dimensions

    def test_an_unknown_model_keeps_the_declared_dimensions(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("embedding:\n  model: some/custom-model\n  dimensions: 512\n", encoding="utf-8")
        assert load_config(path).embedding.dimensions == 512


class TestDefaults:
    def test_a_missing_file_falls_back_to_defaults(self, tmp_path):
        config = load_config(tmp_path / "does-not-exist.yaml")
        assert isinstance(config, Config)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.chunking, ChunkingConfig)

    def test_e5_prefixes_are_set(self):
        embedding = Config().embedding
        assert embedding.query_prefix == "query: "
        assert embedding.passage_prefix == "passage: "

    def test_postgres_dsn_is_well_formed(self):
        dsn = Config().postgres.dsn
        for part in ("host=", "port=", "dbname=", "user=", "password="):
            assert part in dsn
