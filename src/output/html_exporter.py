"""Phase 10 — Export TaxonomyTree as a clean Netscape Bookmark HTML file."""
from __future__ import annotations

import logging
from pathlib import Path

from src.models import Bookmark, Category, TaxonomyTree

logger = logging.getLogger(__name__)

_HEADER = """\
<!DOCTYPE NETSCAPE-Bookmark-file-1>
<!-- This is an automatically generated file.
     It will be read and overwritten.
     DO NOT EDIT! -->
<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
<TITLE>Bookmarks</TITLE>
<H1>Bookmarks</H1>
<DL><p>
"""

_FOOTER = "</DL><p>\n"


class HTMLExporter:
    def __init__(self, tree: TaxonomyTree) -> None:
        self.tree = tree
        # Build reverse map: category_name → list of bookmarks (leaf only)
        self._bm_by_leaf: dict[str, list[Bookmark]] = {}
        for bm in tree.bookmarks:
            if bm.is_duplicate:
                continue
            leaf = bm.leaf_category()
            if leaf:
                self._bm_by_leaf.setdefault(leaf, []).append(bm)

    def _render_bookmarks(self, bookmarks: list[Bookmark], indent: int) -> str:
        pad = "    " * indent
        lines = []
        for bm in bookmarks:
            title = bm.title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            url = bm.url.replace("&", "&amp;")
            lines.append(f'{pad}<DT><A HREF="{url}">{title}</A>')
        return "\n".join(lines) + "\n" if lines else ""

    def _render_category(self, cat: Category, tree: TaxonomyTree, indent: int) -> str:
        pad = "    " * indent
        lines = [f'{pad}<DT><H3>{cat.name}</H3>']
        lines.append(f"{pad}<DL><p>")

        # Render child categories
        for child_id in cat.children_ids:
            child = tree.categories.get(child_id)
            if child:
                lines.append(self._render_category(child, tree, indent + 1))

        # Render bookmarks at this leaf
        leaf_bms = self._bm_by_leaf.get(cat.name, [])
        if leaf_bms:
            lines.append(self._render_bookmarks(leaf_bms, indent + 1))

        lines.append(f"{pad}</DL><p>")
        return "\n".join(lines) + "\n"

    def export(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parts = [_HEADER]
        roots = sorted(self.tree.roots(), key=lambda c: c.name)
        for root in roots:
            parts.append(self._render_category(root, self.tree, indent=1))
        parts.append(_FOOTER)

        content = "".join(parts)
        output_path.write_text(content, encoding="utf-8")
        logger.info("HTML exported → %s", output_path)
        return output_path
