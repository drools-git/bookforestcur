"""K-Means fallback clusterer (used when HDBSCAN produces a single giant cluster)."""
from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import KMeans

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
    c = np.array(centroid, dtype=np.float32)
    dists = np.linalg.norm(embeddings[indices] - c, axis=1)
    sorted_idx = np.argsort(dists)[:k]
    return [ids[indices[i]] for i in sorted_idx]


def run_kmeans(
    embeddings_full: np.ndarray,
    ids: list[str],
    k_factor: int = 50,
    min_cluster_size: int = 10,
) -> list[ClusterResult]:
    """
    Run K-Means with k = max(2, n // k_factor).
    All items are assigned to a cluster (no noise).
    Small clusters (< min_cluster_size) are merged into cluster 0 (MISC-equivalent).
    """
    n = len(ids)
    k = max(2, n // k_factor)
    logger.info("K-Means: clustering %d points into k=%d clusters…", n, k)

    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(embeddings_full)

    results: list[ClusterResult] = []
    unique_labels = sorted(set(labels.tolist()))
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        if len(indices) < min_cluster_size:
            # Tag as noise so it flows to MISC
            effective_label = -1
        else:
            effective_label = label
        centroid = _centroid(embeddings_full, indices)
        reps = _top_k_representatives(embeddings_full, indices, ids, centroid, k=5)
        results.append(
            ClusterResult(
                cluster_id=effective_label,
                bookmark_ids=[ids[i] for i in indices],
                centroid=centroid,
                representative_ids=reps,
            )
        )

    return results
