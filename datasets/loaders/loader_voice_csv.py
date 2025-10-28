"""
Loader for voice lines CSV files.

Usage:
    from datasets.loaders.loader_voice_csv import iter_voice_lines
    for row in iter_voice_lines("datasets/voice_lines/voice_lines.csv"):
        print(row["id"], row["text"])  # etc.

Schema:
    - id (str)
    - text (str)
    - audio_path (str, optional)
    - speaker (str, optional)
    - emotion (str, optional)
    - language (str, optional)
    - dataset_split (str, optional)
    - metadata (str, optional JSON)

This module deliberately avoids coupling with the main dataset build pipeline to
keep voice data flows independent. Downstream training scripts can import and
use these iterators directly.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Dict, Optional


def iter_voice_lines(csv_path: str | Path) -> Iterator[Dict[str, str]]:
    """Yield rows from a CSV of voice lines as dicts.

    Args:
        csv_path: Path to a CSV file with a header row.

    Yields:
        Dictionary per row with string keys/values.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Ensure expected keys exist even if empty
            row.setdefault("id", "")
            row.setdefault("text", "")
            row.setdefault("audio_path", "")
            row.setdefault("speaker", "")
            row.setdefault("emotion", "")
            row.setdefault("language", "")
            row.setdefault("dataset_split", "")
            row.setdefault("metadata", "")
            yield row


def count_voice_lines(csv_path: str | Path) -> int:
    """Return number of non-header rows in the CSV."""
    return sum(1 for _ in iter_voice_lines(csv_path))
