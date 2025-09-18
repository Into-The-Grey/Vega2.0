"""
loader_txt.py - Parse .txt files into (prompt, response) pairs

Strategy:
- Treat each line as a standalone prompt, response is blank string initially
- If you have Q/A structured data, customize here
"""

from __future__ import annotations

from typing import Iterable, Tuple


def load_txt(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            yield text, ""
