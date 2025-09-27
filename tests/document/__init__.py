"""
Document Intelligence Test Suite

This package contains comprehensive tests for the Vega 2.0 document intelligence modules.

Test Structure:
- test_base.py: Base functionality and infrastructure tests
- test_understanding.py: Document understanding and analysis tests
- test_classification.py: Document classification tests
- test_workflow.py: Workflow processing tests (when implemented)
- test_legal.py: Legal document intelligence tests (when implemented)
- test_technical.py: Technical documentation tests (when implemented)

Test Categories:
- Unit tests: Individual component testing
- Integration tests: Cross-component interaction testing
- Performance tests: Load and performance benchmarking
- Error handling tests: Edge cases and failure scenarios

Usage:
    # Run all tests
    python -m pytest tests/document/

    # Run specific module tests
    python -m pytest tests/document/test_understanding.py

    # Run with coverage
    python -m pytest tests/document/ --cov=src.vega.document

    # Run performance tests only
    python -m pytest tests/document/ -k performance

    # Use the test runner
    python tests/document/run_tests.py --type=all --coverage

Test Fixtures:
- Sample documents for various document types
- Mock processors for testing base functionality
- Performance testing utilities
- Async testing helpers
- Error scenario generators

Requirements:
- pytest>=7.0.0
- pytest-asyncio>=0.21.0
- pytest-cov>=4.0.0
- pytest-html>=3.1.0 (for HTML reports)
"""

__version__ = "1.0.0"
__author__ = "Vega 2.0 Team"

# Test configuration
TEST_CONFIG = {
    "timeout": 30.0,
    "performance_requests": 50,
    "concurrency_level": 10,
    "memory_threshold_mb": 100,
    "coverage_threshold": 80,
}

# Available test modules
TEST_MODULES = [
    "base",
    "understanding",
    "classification",
    "workflow",
    "legal",
    "technical",
]

# Test markers used across the suite
TEST_MARKERS = {
    "unit": "Unit tests for individual components",
    "integration": "Integration tests for component interaction",
    "performance": "Performance and load tests",
    "slow": "Tests that take longer to run",
    "asyncio": "Async tests requiring event loop",
    "error": "Error handling and edge case tests",
}
