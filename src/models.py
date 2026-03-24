from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class Bookmark(BaseModel):
    id: str                          # sha256(url)[:16]
    title: str
    url: str
    domain: str
    original_path: str               # folder path from source HTML
    document: str                    # enriched text sent to embedder
    embedding: Optional[list[float]] = None
    category_path: list[str] = Field(default_factory=list)  # [L1, L2, L3, L4]
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None  # id of canonical bookmark
    scraped: bool = False
    scraped_text: Optional[str] = None

    def label(self) -> str:
        """Short display label for the GUI node."""
        return f"{self.title} — {self.domain}"

    def leaf_category(self) -> Optional[str]:
        return self.category_path[-1] if self.category_path else None


class Category(BaseModel):
    id: str
    name: str
    level: int                       # 1 = ROOT, 2 = DOMAIN, 3 = SUBDOMAIN, 4 = TOPIC
    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    centroid: list[float] = Field(default_factory=list)
    member_count: int = 0
    confidence: float = 1.0
    is_preset: bool = False          # true for the 7 hard-coded root buckets


class ClusterResult(BaseModel):
    """Intermediate result from one clustering pass."""
    cluster_id: int                  # -1 = noise/MISC
    bookmark_ids: list[str]
    centroid: list[float]
    representative_ids: list[str]    # top-5 nearest to centroid


class LabelResult(BaseModel):
    category_name: str
    confidence: float


class TaxonomyTree(BaseModel):
    """Complete resolved hierarchy ready for output."""
    categories: dict[str, Category] = Field(default_factory=dict)
    bookmarks: list[Bookmark] = Field(default_factory=list)

    def roots(self) -> list[Category]:
        return [c for c in self.categories.values() if c.level == 1]

    def children_of(self, category_id: str) -> list[Category]:
        cat = self.categories.get(category_id)
        if not cat:
            return []
        return [self.categories[cid] for cid in cat.children_ids if cid in self.categories]

    def bookmarks_in(self, category_id: str) -> list[Bookmark]:
        cat = self.categories.get(category_id)
        if not cat:
            return []
        return [b for b in self.bookmarks if b.leaf_category() == cat.name]
