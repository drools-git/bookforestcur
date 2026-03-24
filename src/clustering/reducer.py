"""Dimensionality reduction via UMAP before clustering."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class UMAPReducer:
    def __init__(
        self,
        n_components: int = 10,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "cosine",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self._reducer = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embeddings to n_components dimensions."""
        import umap

        n = len(embeddings)
        if n < 2:
            logger.warning("UMAP: only %d sample(s), skipping reduction.", n)
            return embeddings

        # n_neighbors must be < number of samples
        effective_neighbors = min(self.n_neighbors, n - 1)
        effective_components = min(self.n_components, n - 1)

        logger.info(
            "UMAP: reducing %d×%d → %d dims (n_neighbors=%d)…",
            n, embeddings.shape[1], effective_components, effective_neighbors,
        )

        self._reducer = umap.UMAP(
            n_components=effective_components,
            n_neighbors=effective_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state,
            low_memory=True,
            verbose=False,
        )
        reduced = self._reducer.fit_transform(embeddings)
        logger.info("UMAP complete → shape %s", reduced.shape)
        return reduced

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform new embeddings using a previously fitted reducer."""
        if self._reducer is None:
            raise RuntimeError("UMAPReducer has not been fitted yet.")
        return self._reducer.transform(embeddings)
