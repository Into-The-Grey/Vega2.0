"""
loader_json.py - Parse .json files into (prompt, response) pairs

Input assumptions:
- File contains either a list of {"prompt": str, "response": str}
  or a dict with key "data" as above.
"""

from __future__ import annotations

import json
from typing import Iterable, Tuple


def load_json_pairs(path: str) -> Iterable[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        return []
    for item in data:
        if not isinstance(item, dict):
            continue
        p = str(item.get("prompt", "")).strip()
        r = str(item.get("response", "")).strip()
        if p:
            yield p, r
