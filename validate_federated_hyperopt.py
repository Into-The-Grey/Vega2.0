#!/usr/bin/env python3
"""Compatibility stub for tests.federated.validation.validate_federated_hyperopt."""

from __future__ import annotations

from importlib import import_module

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
