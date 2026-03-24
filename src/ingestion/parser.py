"""Phase 1 — Parse a Chrome/Brave bookmarks HTML file into flat Bookmark objects."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

from src.models import Bookmark

logger = logging.getLogger(__name__)


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _build_document(title: str, url: str, path: str) -> str:
    return f"Title: {title} | URL: {url} | Path: {path}"


def parse_bookmarks_html(filepath: str | Path) -> list[Bookmark]:
    """
    Parse a Netscape bookmark HTML file.

    Returns a flat list of Bookmark objects with deduplication by URL hash.
    The original folder structure is preserved in `original_path`.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Bookmark file not found: {filepath}")

    logger.info("Parsing bookmarks from %s", filepath)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    seen_ids: set[str] = set()
    bookmarks: list[Bookmark] = []

    def walk(node: Tag, path_parts: list[str]) -> None:
        for child in node.children:
            if not isinstance(child, Tag):
                continue

            if child.name == "dt":
                _process_dt(child, path_parts, seen_ids, bookmarks)

            elif child.name == "dl":
                walk(child, path_parts)

    def _process_dt(dt: Tag, path_parts: list[str], seen: set[str], acc: list[Bookmark]) -> None:
        first = next((c for c in dt.children if isinstance(c, Tag)), None)
        if first is None:
            return

        if first.name == "h3":
            # Folder entry — recurse into the sibling <dl> if present
            folder_name = first.get_text(strip=True)
            new_path = path_parts + [folder_name]
            sibling_dl = dt.find_next_sibling("dl")
            if sibling_dl:
                walk(sibling_dl, new_path)  # type: ignore[arg-type]

        elif first.name == "a":
            url = first.get("href", "").strip()
            if not url or not url.startswith(("http://", "https://")):
                return

            uid = _hash_url(url)
            if uid in seen:
                return
            seen.add(uid)

            title = first.get_text(strip=True) or url
            original_path = " > ".join(path_parts) if path_parts else "/"
            domain = _extract_domain(url)
            document = _build_document(title, url, original_path)

            acc.append(
                Bookmark(
                    id=uid,
                    title=title,
                    url=url,
                    domain=domain,
                    original_path=original_path,
                    document=document,
                )
            )

    root_dl = soup.find("dl")
    if root_dl:
        walk(root_dl, [])  # type: ignore[arg-type]

    logger.info("Parsed %d unique bookmarks", len(bookmarks))
    return bookmarks


def _extract_domain(url: str) -> str:
    """Extract bare domain from a URL (no scheme, no www, no path)."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        host = parsed.netloc or ""
        host = host.removeprefix("www.")
        return host.lower()
    except Exception:
        return ""
