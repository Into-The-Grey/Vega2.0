"""Compatibility package exposing the Vega modules from ``src.vega``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_src_path = Path(__file__).resolve().parent.parent / "src"
if _src_path.is_dir() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

_src_pkg = importlib.import_module("src.vega")
__all__ = getattr(_src_pkg, "__all__", [])

# Mirror selected submodules so ``import vega.<module>`` works out of the box.
_SUBMODULES = [
    "core",
    "federated",
    "datasets",
    "integrations",
    "intelligence",
    "learning",
    "personality",
    "training",
    "user",
    "voice",
]
for name in _SUBMODULES:
    try:
        module = importlib.import_module(f"src.vega.{name}")
        sys.modules[f"vega.{name}"] = module
    except ImportError:  # pragma: no cover - optional modules
        continue


def __getattr__(name: str):  # pragma: no cover - simple delegation
    return getattr(_src_pkg, name)


def __dir__():  # pragma: no cover
    return sorted(set(globals().keys()) | set(dir(_src_pkg)))
