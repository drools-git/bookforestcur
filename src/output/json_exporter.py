"""Phase 10 — Export TaxonomyTree as a structured JSON file."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.models import Bookmark, Category, TaxonomyTree

logger = logging.getLogger(__name__)


def _category_to_dict(cat: Category, tree: TaxonomyTree, include_embeddings: bool = False) -> dict:
    d: dict[str, Any] = {
        "id": cat.id,
        "name": cat.name,
        "level": cat.level,
        "member_count": cat.member_count,
        "confidence": round(cat.confidence, 4),
        "is_preset": cat.is_preset,
        "children": [
            _category_to_dict(tree.categories[cid], tree, include_embeddings)
            for cid in cat.children_ids
            if cid in tree.categories
        ],
    }
    if include_embeddings and cat.centroid:
        d["centroid"] = cat.centroid
    return d


def _bookmark_to_dict(bm: Bookmark, include_embeddings: bool = False) -> dict:
    d: dict[str, Any] = {
        "id": bm.id,
        "title": bm.title,
        "url": bm.url,
        "domain": bm.domain,
        "original_path": bm.original_path,
        "category_path": bm.category_path,
        "is_duplicate": bm.is_duplicate,
        "duplicate_of": bm.duplicate_of,
        "scraped": bm.scraped,
    }
    if include_embeddings and bm.embedding:
        d["embedding"] = bm.embedding
    return d


class JSONExporter:
    def __init__(self, tree: TaxonomyTree) -> None:
        self.tree = tree

    def export(
        self,
        output_path: str | Path,
        include_embeddings: bool = False,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        roots = sorted(self.tree.roots(), key=lambda c: c.name)
        taxonomy = [_category_to_dict(r, self.tree, include_embeddings) for r in roots]

        active_bms = [b for b in self.tree.bookmarks if not b.is_duplicate]
        duplicate_bms = [b for b in self.tree.bookmarks if b.is_duplicate]

        payload = {
            "meta": {
                "total_bookmarks": len(self.tree.bookmarks),
                "active_bookmarks": len(active_bms),
                "duplicate_bookmarks": len(duplicate_bms),
                "total_categories": len(self.tree.categories),
            },
            "taxonomy": taxonomy,
            "bookmarks": [_bookmark_to_dict(b, include_embeddings) for b in active_bms],
            "duplicates": [_bookmark_to_dict(b) for b in duplicate_bms],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info("JSON exported → %s", output_path)
        return output_path

    def export_graph(self, output_path: str | Path) -> Path:
        """
        Export a compact graph JSON for the Sigma.js GUI.
        Nodes = categories + bookmarks, edges = parent-child + bookmark-to-leaf.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        nodes = []
        edges = []

        # Category nodes
        for cat in self.tree.categories.values():
            nodes.append({
                "key": cat.id,
                "label": cat.name,
                "type": "category",
                "level": cat.level,
                "member_count": cat.member_count,
                "is_preset": cat.is_preset,
                "confidence": round(cat.confidence, 4),
            })
            # Parent edge
            if cat.parent_id and cat.parent_id in self.tree.categories:
                edges.append({"source": cat.parent_id, "target": cat.id, "type": "hierarchy"})

        # Bookmark nodes (active only)
        for bm in self.tree.bookmarks:
            if bm.is_duplicate:
                continue
            nodes.append({
                "key": bm.id,
                "label": bm.label(),
                "type": "bookmark",
                "level": 5,
                "url": bm.url,
                "domain": bm.domain,
                "title": bm.title,
                "category_path": bm.category_path,
            })
            # Edge to leaf category
            leaf = bm.leaf_category()
            if leaf:
                # Find category by name
                for cat in self.tree.categories.values():
                    if cat.name == leaf:
                        edges.append({"source": cat.id, "target": bm.id, "type": "contains"})
                        break

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False)

        logger.info("Graph JSON exported → %s (%d nodes, %d edges)", output_path, len(nodes), len(edges))
        return output_path
