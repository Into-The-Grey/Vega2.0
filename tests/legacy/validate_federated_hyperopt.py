#!/usr/bin/env python3
"""Compatibility stub for tests.federated.validation.validate_federated_hyperopt."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

_impl = import_module("tests.federated.validation.validate_federated_hyperopt")

for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name


if __name__ == "__main__":
    import asyncio
    import random
    import sys

    import numpy as np

    np.random.seed(42)
    random.seed(42)

    asyncio.run(_impl.validate_federated_hyperparameter_optimization())
    sys.exit(0)
