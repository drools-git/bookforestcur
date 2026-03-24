"""Phase 4 — HDBSCAN clustering on UMAP-reduced embeddings."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from src.models import ClusterResult

logger = logging.getLogger(__name__)


def _centroid(embeddings: np.ndarray, indices: list[int]) -> list[float]:
    return embeddings[indices].mean(axis=0).tolist()


def _top_k_representatives(
    embeddings: np.ndarray,
    indices: list[int],
    ids: list[str],
    centroid: list[float],
    k: int = 5,
) -> list[str]:
    """Return IDs of the k items closest to the cluster centroid."""
    c = np.array(centroid, dtype=np.float32)
    dists = np.linalg.norm(embeddings[indices] - c, axis=1)
    sorted_idx = np.argsort(dists)[:k]
    return [ids[indices[i]] for i in sorted_idx]


def run_hdbscan(
    embeddings_reduced: np.ndarray,
    embeddings_full: np.ndarray,
    ids: list[str],
    min_cluster_size: int = 10,
    min_samples: int = 5,
) -> list[ClusterResult]:
    """
    Run HDBSCAN on reduced embeddings, compute centroids on full embeddings.

    Returns list of ClusterResult (cluster_id=-1 → noise/MISC).
    """
    import hdbscan

    n = len(ids)
    logger.info("HDBSCAN: clustering %d points (min_cluster_size=%d)…", n, min_cluster_size)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings_reduced)

    unique_labels = sorted(set(labels.tolist()))
    logger.info("HDBSCAN: found %d clusters + noise", sum(1 for l in unique_labels if l >= 0))

    results: list[ClusterResult] = []
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        centroid = _centroid(embeddings_full, indices)
        reps = _top_k_representatives(embeddings_full, indices, ids, centroid, k=5)
        results.append(
            ClusterResult(
                cluster_id=int(label),
                bookmark_ids=[ids[i] for i in indices],
                centroid=centroid,
                representative_ids=reps,
            )
        )

    return results
