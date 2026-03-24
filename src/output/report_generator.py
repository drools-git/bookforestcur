"""Phase 10 — Generate a human-readable text report."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from src.models import TaxonomyTree

logger = logging.getLogger(__name__)


def _tree_lines(tree: TaxonomyTree) -> list[str]:
    lines = []

    def walk(cat_id: str, indent: int) -> None:
        cat = tree.categories.get(cat_id)
        if not cat:
            return
        prefix = "  " * indent
        lines.append(f"{prefix}{'└─' if indent else ''}[L{cat.level}] {cat.name}  ({cat.member_count} items, conf={cat.confidence:.2f})")
        for child_id in cat.children_ids:
            walk(child_id, indent + 1)

    roots = sorted(tree.roots(), key=lambda c: c.name)
    for root in roots:
        walk(root.id, 0)
        lines.append("")

    return lines


class ReportGenerator:
    def __init__(self, tree: TaxonomyTree) -> None:
        self.tree = tree

    def generate(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total = len(self.tree.bookmarks)
        active = sum(1 for b in self.tree.bookmarks if not b.is_duplicate)
        duplicates = total - active
        cats = len(self.tree.categories)

        by_level: dict[int, int] = {}
        for cat in self.tree.categories.values():
            by_level[cat.level] = by_level.get(cat.level, 0) + 1

        # Distribution: how many bookmarks per root
        root_dist: dict[str, int] = {}
        for bm in self.tree.bookmarks:
            if bm.is_duplicate:
                continue
            root = bm.category_path[0] if bm.category_path else "UNKNOWN"
            root_dist[root] = root_dist.get(root, 0) + 1

        lines = [
            "=" * 60,
            "  BookForest2 — Processing Report",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "SUMMARY",
            "-------",
            f"  Total bookmarks parsed  : {total}",
            f"  Active (unique)         : {active}",
            f"  Duplicates removed      : {duplicates}",
            f"  Total categories        : {cats}",
            "",
            "CATEGORIES BY LEVEL",
            "-------------------",
        ]
        for level in sorted(by_level):
            label = {1: "Root", 2: "Domain", 3: "Subdomain", 4: "Topic"}.get(level, f"L{level}")
            lines.append(f"  L{level} {label:<12}: {by_level[level]}")

        lines += [
            "",
            "DISTRIBUTION BY ROOT BUCKET",
            "---------------------------",
        ]
        for root, count in sorted(root_dist.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, count // max(1, active // 40))
            lines.append(f"  {root:<12} {count:>5}  {bar}")

        lines += [
            "",
            "FULL CATEGORY TREE",
            "------------------",
        ]
        lines.extend(_tree_lines(self.tree))

        content = "\n".join(lines) + "\n"
        output_path.write_text(content, encoding="utf-8")
        logger.info("Report exported → %s", output_path)
        return output_path
