#!/usr/bin/env python3
"""Compatibility stub for tests.federated.integration.test_orchestrator_standalone."""

from __future__ import annotations

from importlib import import_module

_impl = import_module("tests.federated.integration.test_orchestrator_standalone")

for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name


if __name__ == "__main__":
    import asyncio
    import sys

    import numpy as np
    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    success = asyncio.run(_impl.run_all_tests())
    sys.exit(0 if success else 1)
