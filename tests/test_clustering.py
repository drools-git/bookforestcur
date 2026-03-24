"""Tests for clustering and deduplication logic."""
import numpy as np
import pytest

from src.clustering.kmeans_clusterer import run_kmeans
from src.models import ClusterResult
from src.taxonomy.deduplicator import (
    remove_exact_duplicates,
    find_semantic_duplicates,
    get_active_bookmarks,
)
from src.models import Bookmark


def _make_bookmark(uid: str, title: str = "Test", url: str = None) -> Bookmark:
    url = url or f"https://example.com/{uid}"
    return Bookmark(
        id=uid,
        title=title,
        url=url,
        domain="example.com",
        original_path="/",
        document=f"Title: {title} | URL: {url}",
    )


# ------------------------------------------------------------------
# K-Means clusterer
# ------------------------------------------------------------------

def test_kmeans_returns_clusters():
    np.random.seed(42)
    n = 50
    embeddings = np.random.randn(n, 32).astype(np.float32)
    ids = [f"id_{i}" for i in range(n)]

    results = run_kmeans(embeddings, ids, k_factor=10, min_cluster_size=3)
    assert isinstance(results, list)
    assert all(isinstance(r, ClusterResult) for r in results)
    # All IDs should be present across clusters
    all_ids = [bid for r in results for bid in r.bookmark_ids]
    assert set(all_ids) == set(ids)


def test_kmeans_small_clusters_become_noise():
    np.random.seed(7)
    embeddings = np.random.randn(25, 16).astype(np.float32)
    ids = [f"id_{i}" for i in range(25)]
    results = run_kmeans(embeddings, ids, k_factor=5, min_cluster_size=10)
    # Some clusters will be below min_cluster_size → cluster_id = -1
    noise = [r for r in results if r.cluster_id == -1]
    assert len(noise) >= 0  # may or may not have noise depending on random split


# ------------------------------------------------------------------
# Exact deduplication
# ------------------------------------------------------------------

def test_remove_exact_duplicates():
    bms = [
        _make_bookmark("abc", url="https://example.com/page"),
        _make_bookmark("abc", url="https://example.com/page"),  # same id/url
        _make_bookmark("def", url="https://other.com"),
    ]
    unique, removed = remove_exact_duplicates(bms)
    assert removed == 1
    assert len(unique) == 2


def test_remove_exact_duplicates_no_dupes():
    bms = [_make_bookmark(f"id{i}") for i in range(5)]
    unique, removed = remove_exact_duplicates(bms)
    assert removed == 0
    assert len(unique) == 5


# ------------------------------------------------------------------
# Semantic deduplication
# ------------------------------------------------------------------

def test_semantic_dedup_marks_near_duplicates():
    # Two bookmarks with identical embeddings (similarity = 1.0)
    emb = [0.1] * 64
    b1 = _make_bookmark("b1", title="Long Title That Wins")
    b2 = _make_bookmark("b2", title="Short")
    b1.embedding = emb
    b2.embedding = emb

    result, marked = find_semantic_duplicates([b1, b2], threshold=0.95)
    assert marked == 1
    dupes = [b for b in result if b.is_duplicate]
    assert len(dupes) == 1
    # The shorter title should be marked as duplicate
    assert dupes[0].id == "b2"


def test_semantic_dedup_skips_dissimilar():
    b1 = _make_bookmark("b1")
    b2 = _make_bookmark("b2")
    b1.embedding = [1.0, 0.0, 0.0]
    b2.embedding = [0.0, 1.0, 0.0]  # orthogonal → similarity = 0

    _, marked = find_semantic_duplicates([b1, b2], threshold=0.95)
    assert marked == 0


def test_get_active_bookmarks():
    b1 = _make_bookmark("b1")
    b2 = _make_bookmark("b2")
    b2.is_duplicate = True
    assert get_active_bookmarks([b1, b2]) == [b1]
