"""Phase 2 — Embed bookmark documents via Ollama with local disk cache."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import ollama

from src.models import Bookmark

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = True,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.cache_dir: Optional[Path] = Path(cache_dir) if cache_dir else None
        self._client = ollama.Client(host=base_url)

        if self.cache_dir and cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check_connection(self) -> None:
        """Raise RuntimeError if Ollama is unreachable or the model is missing."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at the configured URL. "
                f"Make sure Ollama is running. Details: {exc}"
            ) from exc

        # Strip tag suffix for comparison (e.g. "nomic-embed-text:latest")
        base_names = [m.split(":")[0] for m in available]
        model_base = self.model.split(":")[0]
        if model_base not in base_names:
            raise RuntimeError(
                f"Embedding model '{self.model}' is not available in Ollama. "
                f"Run: ollama pull {self.model}"
            )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}::{text}".encode()).hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        if self.cache_dir and self.cache_enabled:
            return self.cache_dir / f"{key}.json"
        return None

    def _load_cache(self, key: str) -> Optional[list[float]]:
        path = self._cache_path(key)
        if path and path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def _save_cache(self, key: str, embedding: list[float]) -> None:
        path = self._cache_path(key)
        if path:
            with open(path, "w") as f:
                json.dump(embedding, f)

    # ------------------------------------------------------------------
    # Single embedding
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        response = self._client.embeddings(model=self.model, prompt=text)
        embedding = response["embedding"]
        self._save_cache(key, embedding)
        return embedding

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, using cache where available."""
        results: list[Optional[list[float]]] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._load_cache(self._cache_key(text))
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if uncached_indices:
            logger.debug("Embedding %d uncached texts…", len(uncached_indices))
            for start in range(0, len(uncached_indices), self.batch_size):
                chunk_indices = uncached_indices[start : start + self.batch_size]
                for idx in chunk_indices:
                    emb = self.embed_text(texts[idx])  # caches internally
                    results[idx] = emb

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Embed bookmarks
    # ------------------------------------------------------------------

    def embed_bookmarks(self, bookmarks: list[Bookmark]) -> list[Bookmark]:
        """Attach embeddings to all bookmarks in-place and return them."""
        from tqdm import tqdm

        texts = [b.document for b in bookmarks]
        logger.info("Embedding %d bookmarks (model=%s)…", len(bookmarks), self.model)

        embeddings = []
        for start in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Embedding",
            unit="batch",
        ):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self.embed_batch(batch))

        for bm, emb in zip(bookmarks, embeddings):
            bm.embedding = emb

        logger.info("Embedding complete.")
        return bookmarks

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def embed_label(self, label: str) -> list[float]:
        """Embed a single short label (for root bucket comparison)."""
        return self.embed_text(label)

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom == 0:
            return 0.0
        return float(np.dot(va, vb) / denom)
