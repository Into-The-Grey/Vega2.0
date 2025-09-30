"""Thin compatibility wrapper for the orchestrator integration test."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


def _ensure_tests_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    tests_dir = repo_root / "tests"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))


async def main() -> bool:
    _ensure_tests_on_path()
    from tests.federated.integration.test_orchestrator_direct import run_all_tests

    return await run_all_tests()


if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        exit_code = 0 if result else 1
    except KeyboardInterrupt:  # pragma: no cover - manual interrupt convenience
        exit_code = 130

    raise SystemExit(exit_code)
