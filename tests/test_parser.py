"""Tests for the HTML bookmark parser."""
import textwrap
import tempfile
from pathlib import Path

import pytest

from src.ingestion.parser import parse_bookmarks_html, _extract_domain


SAMPLE_HTML = textwrap.dedent("""\
    <!DOCTYPE NETSCAPE-Bookmark-file-1>
    <META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">
    <TITLE>Bookmarks</TITLE>
    <H1>Bookmarks</H1>
    <DL><p>
        <DT><H3>AI</H3>
        <DL><p>
            <DT><A HREF="https://openai.com">OpenAI</A>
            <DT><A HREF="https://huggingface.co">HuggingFace</A>
            <DT><H3>Tools</H3>
            <DL><p>
                <DT><A HREF="https://ollama.com">Ollama</A>
            </DL><p>
        </DL><p>
        <DT><H3>DEV</H3>
        <DL><p>
            <DT><A HREF="https://github.com">GitHub</A>
            <DT><A HREF="https://openai.com">OpenAI (dup)</A>
        </DL><p>
        <DT><A HREF="ftp://invalid.com">FTP link (should be skipped)</A>
    </DL><p>
""")


@pytest.fixture
def sample_file(tmp_path):
    f = tmp_path / "bookmarks.html"
    f.write_text(SAMPLE_HTML, encoding="utf-8")
    return str(f)


def test_parse_returns_bookmarks(sample_file):
    bms = parse_bookmarks_html(sample_file)
    urls = [b.url for b in bms]
    assert "https://openai.com" in urls
    assert "https://huggingface.co" in urls
    assert "https://ollama.com" in urls
    assert "https://github.com" in urls


def test_parse_deduplicates_by_url(sample_file):
    bms = parse_bookmarks_html(sample_file)
    urls = [b.url for b in bms]
    assert urls.count("https://openai.com") == 1


def test_parse_skips_non_http(sample_file):
    bms = parse_bookmarks_html(sample_file)
    assert not any("ftp://" in b.url for b in bms)


def test_parse_original_path(sample_file):
    bms = parse_bookmarks_html(sample_file)
    ollama_bm = next(b for b in bms if b.url == "https://ollama.com")
    assert "AI" in ollama_bm.original_path
    assert "Tools" in ollama_bm.original_path


def test_parse_file_not_found():
    with pytest.raises(FileNotFoundError):
        parse_bookmarks_html("/nonexistent/file.html")


def test_extract_domain():
    assert _extract_domain("https://www.github.com/user/repo") == "github.com"
    assert _extract_domain("https://huggingface.co/models") == "huggingface.co"
    assert _extract_domain("") == ""
