"""Optional page scraper — appends visible body text to Bookmark.document."""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.models import Bookmark

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    )
}


def _fetch_text(url: str, timeout: int, max_chars: int) -> Optional[str]:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        return text[:max_chars]
    except Exception as exc:
        logger.debug("Scrape failed for %s: %s", url, exc)
        return None


def scrape_bookmark(
    bookmark: Bookmark,
    timeout: int = 10,
    max_chars: int = 2000,
    delay: float = 0.2,
) -> Bookmark:
    """
    Fetch page text and append to the bookmark's document string.
    Mutates and returns the same object.
    """
    text = _fetch_text(bookmark.url, timeout, max_chars)
    if text:
        bookmark.scraped_text = text
        bookmark.scraped = True
        bookmark.document = f"{bookmark.document} | Content: {text}"
    time.sleep(delay)
    return bookmark


def scrape_all(
    bookmarks: list[Bookmark],
    timeout: int = 10,
    max_chars: int = 2000,
    delay: float = 0.2,
) -> list[Bookmark]:
    from tqdm import tqdm

    logger.info("Scraping %d bookmarks (this may take a while)…", len(bookmarks))
    results = []
    for b in tqdm(bookmarks, desc="Scraping"):
        results.append(scrape_bookmark(b, timeout=timeout, max_chars=max_chars, delay=delay))
    scraped = sum(1 for b in results if b.scraped)
    logger.info("Scraped %d / %d pages successfully", scraped, len(results))
    return results
