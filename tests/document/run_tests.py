"""
Test runner for document intelligence comprehensive test suite
"""

import pytest
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class DocumentIntelligenceTestRunner:
    """Test runner for document intelligence module tests"""

    def __init__(self):
        self.test_directory = Path(__file__).parent
        self.results = {}

    def run_all_tests(
        self, verbose: bool = True, coverage: bool = True
    ) -> Dict[str, Any]:
        """Run all document intelligence tests"""
        print("🚀 Starting Document Intelligence Test Suite")
        print("=" * 50)

        # Pytest arguments
        args = [
            str(self.test_directory),
            "-v" if verbose else "",
            "--tb=short",
            "--asyncio-mode=auto",  # Handle async tests
            "-x",  # Stop on first failure
        ]

        if coverage:
            args.extend(
                [
                    "--cov=src.vega.document",
                    "--cov-report=html:coverage_html",
                    "--cov-report=term-missing",
                ]
            )

        # Filter out empty args
        args = [arg for arg in args if arg]

        # Run tests
        exit_code = pytest.main(args)

        self.results = {
            "exit_code": exit_code,
            "success": exit_code == 0,
            "coverage_enabled": coverage,
        }

        if exit_code == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code: {exit_code}")

        return self.results

    def run_specific_module(self, module: str) -> Dict[str, Any]:
        """Run tests for a specific module"""
        valid_modules = [
            "base",
            "understanding",
            "classification",
            "workflow",
            "legal",
            "technical",
        ]

        if module not in valid_modules:
            raise ValueError(f"Module must be one of: {valid_modules}")

        test_file = self.test_directory / f"test_{module}.py"

        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")

        print(f"🧪 Running tests for {module} module")
        print("=" * 40)

        args = [str(test_file), "-v", "--tb=short", "--asyncio-mode=auto"]

        exit_code = pytest.main(args)

        return {"module": module, "exit_code": exit_code, "success": exit_code == 0}

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run only performance tests"""
        print("⚡ Running Performance Tests")
        print("=" * 30)

        args = [
            str(self.test_directory),
            "-v",
            "-k",
            "performance or load or concurrent",
            "--tb=short",
            "--asyncio-mode=auto",
        ]

        exit_code = pytest.main(args)

        return {
            "test_type": "performance",
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run only integration tests"""
        print("🔗 Running Integration Tests")
        print("=" * 30)

        args = [
            str(self.test_directory),
            "-v",
            "-k",
            "integration or health_check",
            "--tb=short",
            "--asyncio-mode=auto",
        ]

        exit_code = pytest.main(args)

        return {
            "test_type": "integration",
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def run_error_tests(self) -> Dict[str, Any]:
        """Run only error handling tests"""
        print("💥 Running Error Handling Tests")
        print("=" * 35)

        args = [
            str(self.test_directory),
            "-v",
            "-k",
            "error or failure or invalid or timeout",
            "--tb=short",
            "--asyncio-mode=auto",
        ]

        exit_code = pytest.main(args)

        return {
            "test_type": "error_handling",
            "exit_code": exit_code,
            "success": exit_code == 0,
        }

    def generate_test_report(self, output_file: str = "test_report.html") -> str:
        """Generate HTML test report"""
        print("📊 Generating Test Report")
        print("=" * 25)

        report_path = self.test_directory.parent / output_file

        args = [
            str(self.test_directory),
            f"--html={report_path}",
            "--self-contained-html",
            "--asyncio-mode=auto",
        ]

        pytest.main(args)

        print(f"📋 Test report generated: {report_path}")
        return str(report_path)


def main():
    """Main CLI interface for test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Document Intelligence Test Runner")
    parser.add_argument(
        "--module",
        choices=[
            "base",
            "understanding",
            "classification",
            "workflow",
            "legal",
            "technical",
        ],
        help="Run tests for specific module",
    )
    parser.add_argument(
        "--type",
        choices=["all", "performance", "integration", "error"],
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Enable coverage reporting",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate HTML test report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True, help="Verbose output"
    )

    args = parser.parse_args()

    runner = DocumentIntelligenceTestRunner()

    try:
        if args.module:
            # Run specific module tests
            result = runner.run_specific_module(args.module)
        elif args.type == "performance":
            result = runner.run_performance_tests()
        elif args.type == "integration":
            result = runner.run_integration_tests()
        elif args.type == "error":
            result = runner.run_error_tests()
        else:
            # Run all tests
            result = runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)

        if args.report:
            runner.generate_test_report()

        # Exit with appropriate code
        sys.exit(0 if result["success"] else 1)

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
