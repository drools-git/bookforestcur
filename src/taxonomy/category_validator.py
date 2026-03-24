"""Phase 7 — Validate, merge, and cap categories at all levels."""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np

from src.models import Bookmark, Category

logger = logging.getLogger(__name__)

_BANNED_NAMES = frozenset(
    ["misc", "stuff", "links", "resources", "other", "general", "various", "unknown"]
)


def _category_id(name: str, level: int) -> str:
    return hashlib.sha256(f"L{level}::{name.lower()}".encode()).hexdigest()[:12]


def _cosine(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


class CategoryValidator:
    def __init__(
        self,
        min_cluster_size: int = 10,
        min_confidence: float = 0.75,
        merge_threshold: float = 0.85,
        max_categories: int = 50,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_confidence = min_confidence
        self.merge_threshold = merge_threshold
        self.max_categories = max_categories
        self._total_created: int = 0

    @property
    def cap_reached(self) -> bool:
        return self._total_created >= self.max_categories

    def is_valid_name(self, name: str) -> bool:
        return name.lower().strip() not in _BANNED_NAMES

    def can_create(
        self,
        cluster_size: int,
        confidence: float,
        name: str,
    ) -> bool:
        """Return True if a new category may be created."""
        if self.cap_reached:
            logger.debug("Category cap (%d) reached — skipping '%s'", self.max_categories, name)
            return False
        if cluster_size < self.min_cluster_size:
            logger.debug(
                "Cluster too small (%d < %d) — skipping '%s'",
                cluster_size, self.min_cluster_size, name,
            )
            return False
        if confidence < self.min_confidence:
            logger.debug(
                "Confidence too low (%.2f < %.2f) — skipping '%s'",
                confidence, self.min_confidence, name,
            )
            return False
        return True

    def register(self) -> None:
        """Call when a new category is officially added to the taxonomy."""
        self._total_created += 1

    def find_merge_target(
        self,
        candidate_centroid: list[float],
        candidate_name: str,
        existing: list[Category],
    ) -> Optional[Category]:
        """
        If a sufficiently similar category already exists, return it (merge target).
        Compares both centroid cosine similarity and name overlap.
        """
        candidate_name_lower = candidate_name.lower()
        for cat in existing:
            if not cat.centroid:
                continue
            sim = _cosine(candidate_centroid, cat.centroid)
            if sim >= self.merge_threshold:
                logger.debug(
                    "Merge: '%s' → '%s' (centroid sim=%.3f)",
                    candidate_name, cat.name, sim,
                )
                return cat
            # Exact name match at same level is always a merge
            if cat.name.lower() == candidate_name_lower:
                logger.debug("Merge: identical name '%s'", candidate_name)
                return cat
        return None

    def merge_into(
        self,
        target: Category,
        source_bookmarks: list[Bookmark],
        source_centroid: list[float],
    ) -> Category:
        """Merge source cluster into target category in-place."""
        # Weighted centroid update
        total = target.member_count + len(source_bookmarks)
        if total > 0 and target.centroid:
            tc = np.array(target.centroid, dtype=np.float32)
            sc = np.array(source_centroid, dtype=np.float32)
            new_centroid = (
                tc * target.member_count + sc * len(source_bookmarks)
            ) / total
            target.centroid = new_centroid.tolist()
        target.member_count = total
        return target

    def make_category(
        self,
        name: str,
        level: int,
        centroid: list[float],
        member_count: int,
        confidence: float,
        parent_id: Optional[str] = None,
        is_preset: bool = False,
    ) -> Category:
        self.register()
        return Category(
            id=_category_id(name, level),
            name=name,
            level=level,
            parent_id=parent_id,
            centroid=centroid,
            member_count=member_count,
            confidence=confidence,
            is_preset=is_preset,
        )
