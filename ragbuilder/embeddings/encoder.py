"""Sentence-transformers wrapper for the multilingual-e5 family.

The one thing to get right here is the **asymmetric prefix**. e5 is trained with
``query: `` in front of questions and ``passage: `` in front of documents. Skip
them and everything still runs - it just retrieves measurably worse, because the
query and the passage land in slightly different regions of the space. It is a
silent failure, which is exactly why it belongs in a wrapper rather than at
every call site.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Iterable, Sequence

import numpy as np

from ragbuilder.config import Config, get_config

LOG = logging.getLogger("ragbuilder.encoder")


class Encoder:
    """Lazily-loaded embedding model with e5 prefixes applied automatically."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self._model = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ model
    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    name = self.config.embedding.model
                    started = time.perf_counter()
                    LOG.info("loading embedding model %s (first run downloads it)", name)
                    self._model = SentenceTransformer(name)
                    self._model.max_seq_length = self.config.embedding.max_seq_length
                    LOG.info(
                        "loaded %s in %.1fs (%d dimensions)",
                        name,
                        time.perf_counter() - started,
                        self._model.get_sentence_embedding_dimension(),
                    )
        return self._model

    @property
    def dimensions(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    # -------------------------------------------------------------- encoding
    def _encode(self, texts: Sequence[str], prefix: str, show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.config.embedding.dimensions), dtype=np.float32)
        prefixed = [f"{prefix}{text}" for text in texts]
        vectors = self.model.encode(
            prefixed,
            batch_size=self.config.embedding.batch_size,
            normalize_embeddings=self.config.embedding.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_passages(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Embed documents for storage."""
        return self._encode(texts, self.config.embedding.passage_prefix, show_progress)

    def encode_queries(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Embed questions for search."""
        return self._encode(texts, self.config.embedding.query_prefix, show_progress)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode_queries([text])[0]

    def encode_plain(self, texts: Sequence[str]) -> np.ndarray:
        """No prefix - used by semantic chunking, which compares sentences to
        each other rather than a query to a passage."""
        return self._encode(texts, "")

    # ---------------------------------------------------------------- batches
    def iter_batches(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> Iterable[tuple[int, list[str]]]:
        size = batch_size or 1000
        for start in range(0, len(texts), size):
            yield start, list(texts[start : start + size])


_ENCODER: Encoder | None = None


def get_encoder(config: Config | None = None) -> Encoder:
    """Process-wide singleton - loading the model twice would double the RAM."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = Encoder(config)
    return _ENCODER


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)
