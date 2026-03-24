"""Enrich Bookmark objects with extracted keywords and an improved document string."""
from __future__ import annotations

import re
import logging
from typing import Optional

from src.models import Bookmark

logger = logging.getLogger(__name__)

# Common stop-words to strip from keyword extraction
_STOP = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might must can could of in on at to for with by from and or "
    "but not so yet both either neither nor as if though although because since "
    "while when where how what who which that this these those it its i you we "
    "he she they me him her us them my your his our their".split()
)


def _split_tokens(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-_]", " ", text)
    return [t.strip("-_") for t in text.split() if t.strip("-_")]


def _url_keywords(url: str) -> list[str]:
    """Extract path segments and query-value words from a URL."""
    from urllib.parse import urlparse, unquote
    try:
        parsed = urlparse(url)
        path = unquote(parsed.path)
        parts = re.split(r"[/\-_\.\?=&%+]", path)
        return [p for p in parts if len(p) > 2 and not p.isdigit()]
    except Exception:
        return []


def _extract_keywords(title: str, url: str, max_kw: int = 10) -> list[str]:
    title_tokens = [t for t in _split_tokens(title) if t not in _STOP and len(t) > 2]
    url_tokens = [t for t in _url_keywords(url) if t not in _STOP]
    # Title tokens first, URL tokens fill the rest
    seen: set[str] = set()
    kw: list[str] = []
    for t in title_tokens + url_tokens:
        if t not in seen:
            seen.add(t)
            kw.append(t)
        if len(kw) >= max_kw:
            break
    return kw


def enrich_bookmark(bookmark: Bookmark) -> Bookmark:
    """
    Rebuild the `document` field with extracted keywords for richer embeddings.
    Mutates and returns the same object.
    """
    keywords = _extract_keywords(bookmark.title, bookmark.url)
    kw_str = " ".join(keywords)
    bookmark.document = (
        f"Title: {bookmark.title} | "
        f"Domain: {bookmark.domain} | "
        f"Path: {bookmark.original_path} | "
        f"Keywords: {kw_str}"
    )
    return bookmark


def enrich_all(bookmarks: list[Bookmark]) -> list[Bookmark]:
    logger.info("Enriching %d bookmarks", len(bookmarks))
    return [enrich_bookmark(b) for b in bookmarks]
