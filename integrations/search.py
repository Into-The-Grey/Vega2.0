"""
search.py - Web and image search integrations for Vega2.0

- Uses duckduckgo_search for privacy-friendly web and image search
- Provides graceful fallback when package or network is unavailable
"""

from __future__ import annotations

from typing import List, Dict, Optional

try:
    from duckduckgo_search import DDGS  # type: ignore
except Exception:  # pragma: no cover - optional dep
    DDGS = None  # type: ignore


def web_search(
    query: str, max_results: int = 5, safesearch: str = "moderate"
) -> List[Dict]:
    """Return a list of web search results with title, href, and snippet."""
    if not DDGS:
        return []
    out: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, safesearch=safesearch, max_results=max_results):
                out.append(
                    {
                        "title": r.get("title"),
                        "href": r.get("href"),
                        "snippet": r.get("body"),
                        "source": r.get("source"),
                    }
                )
    except Exception:
        return []
    return out


ess_safe_values = {"off", "moderate", "strict"}


def image_search(
    query: str, max_results: int = 5, safesearch: str = "moderate"
) -> List[Dict]:
    """Return a list of image results with title, image URL, and thumbnail."""
    if not DDGS:
        return []
    if safesearch not in ess_safe_values:
        safesearch = "moderate"
    out: List[Dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.images(query, safesearch=safesearch, max_results=max_results):
                out.append(
                    {
                        "title": r.get("title"),
                        "image": r.get("image"),
                        "thumbnail": r.get("thumbnail"),
                        "url": r.get("url"),
                        "width": r.get("width"),
                        "height": r.get("height"),
                    }
                )
    except Exception:
        return []
    return out
