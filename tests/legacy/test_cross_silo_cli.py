#!/usr/bin/env python3
"""Legacy entry point for cross-silo CLI validation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


if __name__ == "__main__":
    from tests.federated.validation.validate_cross_silo_cli import main

    raise SystemExit(main())
