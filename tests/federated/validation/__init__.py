"""Federated validation test suites and standalone validators."""

from importlib import import_module
from typing import List

_SUBMODULES: List[str] = [
    "test_validation_suite",
    "validate_cross_silo_simple",
    "validate_cross_silo_hierarchical",
    "validate_federated_hyperopt",
]

for _module in _SUBMODULES:
    globals()[_module] = import_module(f"{__name__}.{_module}")

del _module
