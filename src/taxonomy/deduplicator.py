"""Phase 9 — Exact URL deduplication and semantic near-duplicate detection."""
from __future__ import annotations

import logging

import numpy as np

from src.models import Bookmark

logger = logging.getLogger(__name__)


def remove_exact_duplicates(bookmarks: list[Bookmark]) -> tuple[list[Bookmark], int]:
    """
    Remove bookmarks that share the same URL hash (id).
    Returns (deduplicated list, count removed).
    """
    seen: dict[str, Bookmark] = {}
    duplicates = 0
    for bm in bookmarks:
        if bm.id in seen:
            duplicates += 1
        else:
            seen[bm.id] = bm
    logger.info("Exact dedup: removed %d duplicates", duplicates)
    return list(seen.values()), duplicates


def find_semantic_duplicates(
    bookmarks: list[Bookmark],
    threshold: float = 0.95,
) -> tuple[list[Bookmark], int]:
    """
    Detect and mark bookmarks whose embeddings are too similar (> threshold).

    Strategy: brute-force pairwise cosine on the full matrix.
    For 10K bookmarks this is a 10K×10K matrix — we chunk it to stay memory-safe.
    The bookmark with the longer title is kept as canonical.

    Returns (list with duplicates marked, count marked).
    """
    embedded = [b for b in bookmarks if b.embedding is not None]
    if not embedded:
        return bookmarks, 0

    logger.info(
        "Semantic dedup: scanning %d bookmarks (threshold=%.2f)…",
        len(embedded), threshold,
    )

    matrix = np.array([b.embedding for b in embedded], dtype=np.float32)
    # L2-normalize for fast cosine via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    matrix /= norms

    duplicate_ids: set[str] = set()
    canonical_of: dict[str, str] = {}  # id → canonical id

    chunk_size = 500
    n = len(embedded)

    for i in range(0, n, chunk_size):
        chunk = matrix[i : i + chunk_size]
        # Dot product with all subsequent rows (upper triangle only)
        sims = chunk @ matrix[i:].T  # shape (chunk, n-i)
        for ci, row in enumerate(sims):
            gi = i + ci  # global index
            if embedded[gi].id in duplicate_ids:
                continue
            # Check only j > gi to avoid double-marking
            for j_local, sim in enumerate(row[ci + 1 :], start=ci + 1):
                gj = i + j_local
                if embedded[gj].id in duplicate_ids:
                    continue
                if float(sim) > threshold:
                    # Keep the one with the longer (richer) title
                    if len(embedded[gi].title) >= len(embedded[gj].title):
                        duplicate_ids.add(embedded[gj].id)
                        canonical_of[embedded[gj].id] = embedded[gi].id
                    else:
                        duplicate_ids.add(embedded[gi].id)
                        canonical_of[embedded[gi].id] = embedded[gj].id
                        break

    for bm in bookmarks:
        if bm.id in duplicate_ids:
            bm.is_duplicate = True
            bm.duplicate_of = canonical_of.get(bm.id)

    logger.info("Semantic dedup: marked %d near-duplicates", len(duplicate_ids))
    return bookmarks, len(duplicate_ids)


def get_active_bookmarks(bookmarks: list[Bookmark]) -> list[Bookmark]:
    """Return only non-duplicate bookmarks."""
    return [b for b in bookmarks if not b.is_duplicate]
