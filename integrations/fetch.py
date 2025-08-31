"""
fetch.py - HTML content fetching and extraction for research/RAG.

- Uses httpx for fast async fetch with timeouts
- Parses with BeautifulSoup (lxml parser) to extract readable text
- Strips scripts/styles and collapses whitespace
"""

from __future__ import annotations

import re
from typing import Optional

try:
    import httpx
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # optional deps
    httpx = None  # type: ignore
    BeautifulSoup = None  # type: ignore


_whitespace_re = re.compile(r"\s+")


async def fetch_text(
    url: str, timeout: float = 8.0, max_chars: int = 8000
) -> Optional[str]:
    if not httpx or not BeautifulSoup:
        return None
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            r = await client.get(url)
            if r.status_code >= 400:
                return None
            html = r.text
    except Exception:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
        # remove scripts and styles
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(" ")
        text = _whitespace_re.sub(" ", text).strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text
    except Exception:
        return None
