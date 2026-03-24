"""Phase 3 — Assign bookmarks to preset Level-1 root buckets via cosine similarity."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.embeddings.embedder import Embedder
from src.models import Bookmark

logger = logging.getLogger(__name__)


class RootAssigner:
    def __init__(
        self,
        root_buckets: list[str],
        embedder: Embedder,
        threshold: float = 0.75,
    ) -> None:
        self.root_buckets = root_buckets
        self.embedder = embedder
        self.threshold = threshold
        self._root_embeddings: dict[str, list[float]] = {}

    def _ensure_root_embeddings(self) -> None:
        if self._root_embeddings:
            return
        logger.info("Embedding %d root bucket labels…", len(self.root_buckets))
        for name in self.root_buckets:
            # Use a rich description so embeddings capture intent better
            doc = f"Category: {name} | Topic area: {name} bookmarks and resources"
            self._root_embeddings[name] = self.embedder.embed_label(doc)

    def assign(self, bookmarks: list[Bookmark]) -> tuple[list[Bookmark], list[Bookmark]]:
        """
        Attempt to assign each bookmark to a root bucket.

        Returns:
            assigned   — bookmarks with category_path[0] set
            unsorted   — bookmarks whose max similarity < threshold
        """
        self._ensure_root_embeddings()

        assigned: list[Bookmark] = []
        unsorted: list[Bookmark] = []

        root_matrix = np.array(
            [self._root_embeddings[name] for name in self.root_buckets], dtype=np.float32
        )

        for bm in bookmarks:
            if bm.embedding is None:
                unsorted.append(bm)
                continue

            vec = np.array(bm.embedding, dtype=np.float32)
            norms = np.linalg.norm(root_matrix, axis=1) * np.linalg.norm(vec)
            # Guard against zero-norm vectors
            with np.errstate(invalid="ignore"):
                similarities = np.where(norms > 0, root_matrix @ vec / norms, 0.0)

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.threshold:
                bm.category_path = [self.root_buckets[best_idx]]
                assigned.append(bm)
            else:
                unsorted.append(bm)

        logger.info(
            "Root assignment: %d assigned, %d unsorted (threshold=%.2f)",
            len(assigned), len(unsorted), self.threshold,
        )
        return assigned, unsorted

    def root_embeddings(self) -> dict[str, list[float]]:
        self._ensure_root_embeddings()
        return dict(self._root_embeddings)
