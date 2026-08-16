"""Configuration loading: ``config.yaml`` overlaid with environment variables.

The precedence is env > yaml > dataclass default. Nested keys map to env vars by
joining with an underscore and upper-casing, so ``llm.model`` is ``LLM_MODEL``
and ``embedding.batch_size`` is ``EMBEDDING_BATCH_SIZE``. That is what lets the
same image run against a different model or vector store without a rebuild.
"""

# NOTE: deliberately no `from __future__ import annotations` here.
# `_build` inspects `dataclasses.fields(...).type` to decide whether a field is
# itself a config section and to coerce environment strings to the right type.
# Under PEP 563 those types arrive as *strings*, `is_dataclass("ChunkingConfig")`
# is False, and every nested section silently stays a raw dict.

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env", override=False)

T = TypeVar("T")


# ------------------------------------------------------------------ sections
@dataclass
class CorpusConfig:
    name: str = "german_labour_law"
    pdf_dir: str = "data/pdfs"


@dataclass
class EmbeddingConfig:
    model: str = "intfloat/multilingual-e5-large"
    dimensions: int = 1024
    batch_size: int = 32
    max_seq_length: int = 512
    normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "


@dataclass
class FixedChunkingConfig:
    chunk_tokens: int = 512
    overlap_tokens: int = 50


@dataclass
class SentenceChunkingConfig:
    max_tokens: int = 512
    overlap_sentences: int = 1
    language: str = "german"


@dataclass
class SemanticChunkingConfig:
    max_tokens: int = 512
    breakpoint_percentile: int = 85
    min_sentences: int = 3


@dataclass
class ChunkingConfig:
    strategy: str = "sentence"
    fixed: FixedChunkingConfig = field(default_factory=FixedChunkingConfig)
    sentence: SentenceChunkingConfig = field(default_factory=SentenceChunkingConfig)
    semantic: SemanticChunkingConfig = field(default_factory=SemanticChunkingConfig)


@dataclass
class VectorStoreConfig:
    collection: str = "ragbuilder_chunks"
    distance: str = "cosine"
    hnsw_m: int = 16
    hnsw_ef_construct: int = 128
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))


@dataclass
class RetrievalConfig:
    strategy: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    rrf_k: int = 60
    query_expansion: bool = True
    expansion_variants: int = 3
    min_score: float = 0.0


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "mistral:latest"
    fallback_model: str = "qwen2.5-coder:7b"
    temperature: float = 0.1
    top_p: float = 0.9
    num_ctx: int = 8192
    max_tokens: int = 1024
    timeout_seconds: int = 180
    stream: bool = True
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))


@dataclass
class RagConfig:
    memory_turns: int = 3
    min_confidence: float = 0.30
    max_context_chars: int = 8000
    cite_sources: bool = True
    language: str = "auto"


@dataclass
class EvaluationConfig:
    judge_model: str = "mistral:latest"
    testset: str = "ragbuilder/evaluation/testset.json"
    mlflow_experiment: str = "ragbuilder"
    strategies: list[str] = field(default_factory=lambda: ["fixed", "sentence", "semantic"])
    mlflow_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))


@dataclass
class ApiConfig:
    rate_limit: str = "30/minute"
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    url: str = field(default_factory=lambda: os.getenv("API_URL", "http://localhost:8000"))


@dataclass
class PostgresConfig:
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5433")))
    database: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "ragbuilder"))
    user: str = field(default_factory=lambda: os.getenv("POSTGRES_USER", "ragbuilder"))
    password: str = field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "ragbuilder"))

    @property
    def dsn(self) -> str:
        return (
            f"host={self.host} port={self.port} dbname={self.database} "
            f"user={self.user} password={self.password}"
        )


@dataclass
class Config:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RagConfig = field(default_factory=RagConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    postgres: PostgresConfig = field(default_factory=PostgresConfig)

    @property
    def pdf_path(self) -> Path:
        path = Path(self.corpus.pdf_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path


# ------------------------------------------------------------------ loading
def _coerce(value: Any, target_type: type) -> Any:
    """Turn an environment string into the type the dataclass field declares."""
    if target_type is bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list or getattr(target_type, "__origin__", None) is list:
        if isinstance(value, list):
            return value
        return [part.strip() for part in str(value).split(",") if part.strip()]
    return value


def _build(cls: type[T], data: dict[str, Any], prefix: str = "") -> T:
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        key = f.name
        env_name = f"{prefix}{key}".upper()
        raw = data.get(key)

        if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
            kwargs[key] = _build(f.type, raw or {}, prefix=f"{prefix}{key}_")
            continue

        if env_name in os.environ:
            kwargs[key] = _coerce(os.environ[env_name], f.type)
        elif raw is not None:
            kwargs[key] = raw
    return cls(**kwargs)  # type: ignore[return-value]


def load_config(path: str | Path | None = None) -> Config:
    """Load the configuration, or the defaults if the file is missing."""
    config_path = Path(path or os.getenv("RAGBUILDER_CONFIG", DEFAULT_CONFIG_PATH))
    data: dict[str, Any] = {}
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    config = Config(
        corpus=_build(CorpusConfig, data.get("corpus", {}), "corpus_"),
        embedding=_build(EmbeddingConfig, data.get("embedding", {}), "embedding_"),
        chunking=_build(ChunkingConfig, data.get("chunking", {}), "chunking_"),
        vector_store=_build(VectorStoreConfig, data.get("vector_store", {}), "vector_store_"),
        retrieval=_build(RetrievalConfig, data.get("retrieval", {}), "retrieval_"),
        llm=_build(LLMConfig, data.get("llm", {}), "llm_"),
        rag=_build(RagConfig, data.get("rag", {}), "rag_"),
        evaluation=_build(EvaluationConfig, data.get("evaluation", {}), "evaluation_"),
        api=_build(ApiConfig, data.get("api", {}), "api_"),
        postgres=PostgresConfig(),
    )

    # A wrong dimension count is a silent retrieval killer: Qdrant would accept
    # the collection and every search would then fail or return nonsense.
    known_dims = {
        "intfloat/multilingual-e5-small": 384,
        "intfloat/multilingual-e5-base": 768,
        "intfloat/multilingual-e5-large": 1024,
    }
    expected = known_dims.get(config.embedding.model)
    if expected and config.embedding.dimensions != expected:
        config.embedding.dimensions = expected

    return config


_CACHED: Config | None = None


def get_config(reload: bool = False) -> Config:
    global _CACHED
    if _CACHED is None or reload:
        _CACHED = load_config()
    return _CACHED
