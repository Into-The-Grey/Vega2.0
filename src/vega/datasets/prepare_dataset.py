"""
prepare_dataset.py - Build a JSONL dataset from a directory of text-like files.

- Supports .txt, .md, .json files via dedicated loaders
- Normalizes into JSONL lines: {"prompt": "...", "response": "..."}
- Output path: datasets/output.jsonl
- Usage: imported by CLI (vega dataset build ./path)
"""

from __future__ import annotations

import json
import os
from typing import Iterable, Tuple

from ..datasets.loaders.loader_txt import load_txt
from ..datasets.loaders.loader_md import load_md
from ..datasets.loaders.loader_json import load_json_pairs

SUPPORTED_EXTS = {".txt", ".md", ".json"}
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output.jsonl")


def _iter_pairs_from_dir(root: str) -> Iterable[Tuple[str, str]]:
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            ext = os.path.splitext(fname.lower())[1]
            if ext not in SUPPORTED_EXTS:
                continue
            if ext == ".txt":
                for p, r in load_txt(path):
                    yield p, r
            elif ext == ".md":
                for p, r in load_md(path):
                    yield p, r
            elif ext == ".json":
                for p, r in load_json_pairs(path):
                    yield p, r


def build_dataset(input_dir: str) -> str:
    """Build output.jsonl from files in input_dir. Returns output path."""
    count = 0
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for prompt, response in _iter_pairs_from_dir(input_dir):
            f.write(
                json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False)
                + "\n"
            )
            count += 1
    return OUTPUT_PATH


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "."
    out = build_dataset(path)
    print(f"Wrote {out}")
