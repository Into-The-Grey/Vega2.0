#!/usr/bin/env python3
"""Compatibility stub for tests.federated.validation.test_validation_suite."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_impl = import_module("tests.federated.validation.test_validation_suite")

for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name


if __name__ == "__main__":
    import asyncio
    import sys

    success = asyncio.run(_impl.run_comprehensive_tests())
    sys.exit(0 if success else 1)
