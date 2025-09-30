"""Compatibility stub for tests.federated.validation.validate_cross_silo_simple."""

from __future__ import annotations

from importlib import import_module

_impl = import_module("tests.federated.validation.validate_cross_silo_simple")

for _name in dir(_impl):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_impl, _name)

del _name


if __name__ == "__main__":
    _impl.demo()
