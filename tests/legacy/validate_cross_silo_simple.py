"""Compatibility stub for tests.federated.validation.validate_cross_silo_simple."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_impl = import_module("tests.federated.validation.validate_cross_silo_simple")

for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name


if __name__ == "__main__":
    _impl.demo()
