"""
security.py - Lightweight PII masking utilities

Simple regex-based masking for emails, phone numbers, and secrets-like tokens.
"""

from __future__ import annotations

import re
from typing import Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?\d[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b")
SECRET_RE = re.compile(r"\b([A-Za-z0-9]{24,})\b")


def mask_pii(text: str) -> str:
    if not text:
        return text
    t = EMAIL_RE.sub("[email]", text)
    t = PHONE_RE.sub("[phone]", t)
    t = SECRET_RE.sub("[secret]", t)
    return t
