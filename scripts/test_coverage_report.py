"""
test_coverage_report.py - Test Coverage Analysis Tool

Analyzes test coverage gaps in the Vega2.0 project.
Run with: python -m scripts.test_coverage_report

Outputs:
- Modules with no corresponding test file
- Coverage percentage by module
- Recommended test priorities
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class CoverageReport:
    """Test coverage analysis report"""

    total_modules: int
    tested_modules: int
    untested_modules: List[str]
    coverage_by_package: Dict[str, Tuple[int, int]]  # (tested, total)
    priority_modules: List[str]  # High-priority untested modules


def analyze_coverage(project_root: Path) -> CoverageReport:
    """Analyze test coverage for the project"""

    src_path = project_root / "src" / "vega"
    tests_path = project_root / "tests"

    # Find all Python modules (excluding __init__.py and test files)
    all_modules: Set[str] = set()
    module_to_path: Dict[str, Path] = {}

    for py_file in src_path.rglob("*.py"):
        if py_file.name.startswith("__"):
            continue
        if py_file.name.startswith("test_"):
            continue

        # Get relative module path
        rel_path = py_file.relative_to(src_path)
        module_name = str(rel_path).replace("/", ".").replace(".py", "")
        all_modules.add(module_name)
        module_to_path[module_name] = py_file

    # Find all test files
    tested_modules: Set[str] = set()

    for test_file in tests_path.rglob("test_*.py"):
        # Extract module name from test file
        test_name = test_file.stem  # e.g., "test_app"
        module_name = test_name.replace("test_", "")

        # Try to match to actual modules
        for mod in all_modules:
            if mod.endswith(module_name) or mod.split(".")[-1] == module_name:
                tested_modules.add(mod)

    # Calculate untested modules
    untested = sorted(all_modules - tested_modules)

    # Coverage by package
    coverage_by_pkg: Dict[str, Tuple[int, int]] = {}
    for mod in all_modules:
        pkg = mod.split(".")[0] if "." in mod else "core"
        tested, total = coverage_by_pkg.get(pkg, (0, 0))
        total += 1
        if mod in tested_modules:
            tested += 1
        coverage_by_pkg[pkg] = (tested, total)

    # Identify priority modules (core functionality, large files)
    priority_modules = []
    priority_keywords = ["app", "cli", "llm", "db", "config", "api", "security"]

    for mod in untested:
        mod_name = mod.split(".")[-1]
        if any(kw in mod_name for kw in priority_keywords):
            priority_modules.append(mod)
        elif module_to_path.get(mod):
            # Large files are also high priority
            try:
                size = module_to_path[mod].stat().st_size
                if size > 10000:  # > 10KB
                    priority_modules.append(mod)
            except:
                pass

    return CoverageReport(
        total_modules=len(all_modules),
        tested_modules=len(tested_modules),
        untested_modules=untested[:50],  # Limit output
        coverage_by_package=coverage_by_pkg,
        priority_modules=priority_modules[:20],
    )


def print_report(report: CoverageReport) -> None:
    """Print a formatted coverage report"""

    coverage_pct = (report.tested_modules / report.total_modules * 100) if report.total_modules > 0 else 0

    print("\n" + "=" * 60)
    print("VEGA2.0 TEST COVERAGE REPORT")
    print("=" * 60)

    print(f"\n📊 Overall Coverage: {coverage_pct:.1f}%")
    print(f"   Total modules: {report.total_modules}")
    print(f"   Tested modules: {report.tested_modules}")
    print(f"   Untested modules: {len(report.untested_modules)}")

    print("\n📦 Coverage by Package:")
    for pkg, (tested, total) in sorted(report.coverage_by_package.items()):
        pct = (tested / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"   {pkg:20} [{bar}] {pct:5.1f}% ({tested}/{total})")

    if report.priority_modules:
        print("\n🎯 High-Priority Untested Modules:")
        for mod in report.priority_modules:
            print(f"   • {mod}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    report = analyze_coverage(project_root)
    print_report(report)
