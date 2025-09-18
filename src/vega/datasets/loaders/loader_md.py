"""
loader_md.py - Parse .md files into (prompt, response) pairs

Strategy:
- Use headings as prompts and following paragraph as response stub
- Very naive; adapt to your markdown convention
"""

from __future__ import annotations

from typing import Iterable, Tuple


def load_md(path: str) -> Iterable[Tuple[str, str]]:
    prompt = None
    response_lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                # flush previous
                if prompt is not None:
                    yield prompt, " ".join(l.strip() for l in response_lines).strip()
                prompt = line.lstrip("#").strip()
                response_lines = []
            else:
                response_lines.append(line)
    if prompt is not None:
        yield prompt, " ".join(l.strip() for l in response_lines).strip()
