"""
learning/learn.py - Lightweight learning and self-improvement utilities

Features:
- curate_dataset(min_rating, reviewed_only, out_path): export high-quality rows from DB to JSONL
- evaluate_model(eval_file, system_prompt): simple heuristic scoring against expected responses
- optimize_system_prompt(candidates_file, eval_file, out_dir): pick best system prompt by average score

This is intentionally minimal and local-only, designed to work with the existing Vega2.0 stack.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Dict, Optional

from ..db import get_for_training
from ..llm import query_llm
from ..config import get_config
import asyncio


def curate_dataset(
    min_rating: int = 4,
    reviewed_only: bool = False,
    out_path: str = "datasets/curated.jsonl",
) -> str:
    """Export high-quality conversation pairs from SQLite to JSONL for training."""
    rows = get_for_training(min_rating=min_rating, reviewed_only=reviewed_only)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            rec = {"prompt": r["prompt"], "response": r["response"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return out_path


async def _score_pair(
    prompt: str, expected: str, system_prompt: Optional[str]
) -> float:
    p = prompt
    if system_prompt:
        p = f"System: {system_prompt}\n\nUser: {prompt}"
    cfg = get_config()
    try:
        got = await asyncio.wait_for(
            query_llm(p, stream=False), timeout=cfg.llm_timeout_sec
        )
        got = str(got)
    except asyncio.TimeoutError:
        return 0.0
    except Exception:
        return 0.0
    # Tiny heuristic scorer: exactness and containment
    if got.strip() == expected.strip():
        return 1.0
    if expected.strip() and expected.strip().lower() in got.strip().lower():
        return 0.7
    return 0.3 if got else 0.0


async def evaluate_model(eval_file: str, system_prompt: Optional[str] = None) -> float:
    """Return average score on a small eval JSONL of {prompt,response}."""
    total = 0.0
    count = 0
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += await _score_pair(rec["prompt"], rec["response"], system_prompt)
            count += 1
    return total / count if count else 0.0


async def optimize_system_prompt(
    candidates_file: str,
    eval_file: str,
    out_dir: str = "prompts",
) -> Dict[str, float]:
    """Try multiple system prompt candidates; write best to prompts/system_prompt.txt.

    Returns dict of {candidate_text: score}
    """
    os.makedirs(out_dir, exist_ok=True)
    scores: Dict[str, float] = {}
    with open(candidates_file, "r", encoding="utf-8") as f:
        for line in f:
            cand = line.strip()
            if not cand:
                continue
            score = await evaluate_model(eval_file, system_prompt=cand)
            scores[cand] = score
    # Pick best
    best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else None
    if best is not None:
        with open(
            os.path.join(out_dir, "system_prompt.txt"), "w", encoding="utf-8"
        ) as g:
            g.write(best)
    return scores
