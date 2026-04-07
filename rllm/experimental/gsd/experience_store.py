"""Embedding-based experience buffer for GSD.

Stores past (question, hint, outcome) tuples and retrieves the most
similar entries for a new question via cosine similarity over sentence
embeddings.

The embedding model (default ``all-MiniLM-L6-v2``, ~80 MB) is lazy-loaded
on first use and runs on CPU via ``sentence-transformers``.  Embedding
computation is offloaded to a thread via ``asyncio.to_thread`` so it
doesn't block the training loop.

Usage::

    store = EmbeddingExperienceStore(max_size=500)

    # After a successful distillation step:
    await store.add(question, {"hint": hint_text, "summary": "...", "improvement": 0.5})

    # Before hint generation:
    experiences = await store.query(question, top_k=3)
    hint_messages = build_hint_prompt(question, experiences)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperienceEntry:
    """A single experience in the buffer."""

    text: str  # Problem statement
    embedding: np.ndarray  # Pre-computed embedding vector
    metadata: dict[str, Any]  # hint, summary, improvement, etc.
    created_at: int = 0  # Training step (for staleness tracking)


class EmbeddingExperienceStore:
    """Top-K embedding similarity retrieval buffer for GSD.

    Thread-safe via ``asyncio.Lock``.  All public methods are async.

    Args:
        model_name: Sentence-transformers model name.
        device: Device for the embedding model (``"cpu"`` recommended).
        max_size: Maximum number of entries. FIFO eviction when full.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        max_size: int = 500,
    ) -> None:
        self._model_name = model_name
        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self.max_size = max_size
        self._entries: list[ExperienceEntry] = []
        self._lock = asyncio.Lock()
        self._encoder = None  # lazy-loaded
        self._step = 0

    def _get_encoder(self):
        """Lazy-load the sentence-transformers model on first use."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model '{self._model_name}' on {self._device}")
            self._encoder = SentenceTransformer(self._model_name, device=self._device)
        return self._encoder

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text string (blocking, called via to_thread)."""
        encoder = self._get_encoder()
        return encoder.encode(text, normalize_embeddings=True)

    @property
    def size(self) -> int:
        return len(self._entries)

    def set_step(self, step: int) -> None:
        """Update the current training step (for staleness tracking)."""
        self._step = step

    async def add(
        self,
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        """Add an experience entry with auto-computed embedding.

        Args:
            text: Problem statement (used as the embedding key).
            metadata: Arbitrary metadata dict.  Recommended keys:
                ``"hint"``, ``"summary"``, ``"improvement"``.
        """
        embedding = await asyncio.to_thread(self._encode, text)
        async with self._lock:
            if len(self._entries) >= self.max_size:
                self._entries.pop(0)  # FIFO eviction
            self._entries.append(
                ExperienceEntry(
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                    created_at=self._step,
                )
            )

    async def query(self, text: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve the top-K most similar experiences.

        Returns a list of dicts with keys ``"text"``, plus all metadata keys
        (``"hint"``, ``"summary"``, etc.).  Returns fewer than ``top_k`` if
        the buffer has fewer entries.  Returns ``[]`` if the buffer is empty.
        """
        async with self._lock:
            if not self._entries:
                return []
            entries_snapshot = list(self._entries)

        query_emb = await asyncio.to_thread(self._encode, text)

        # Batch cosine similarity (embeddings are L2-normalized)
        all_embs = np.stack([e.embedding for e in entries_snapshot])
        similarities = all_embs @ query_emb
        k = min(top_k, len(entries_snapshot))
        top_indices = np.argpartition(-similarities, k)[:k]
        # Sort the top-K by similarity (descending)
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results = []
        for idx in top_indices:
            entry = entries_snapshot[idx]
            result = {"text": entry.text, **entry.metadata}
            results.append(result)
        return results

    async def clear(self) -> None:
        """Remove all entries."""
        async with self._lock:
            self._entries.clear()
