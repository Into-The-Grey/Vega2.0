"""
Pytest configuration for Vega2.0 test suite

This conftest.py handles test setup, fixtures, and environment configuration
to ensure tests run smoothly without heavy import issues.
"""

import os
import sys
from pathlib import Path

# Set environment variable to prevent heavy ML library loading during tests
os.environ["VEGA_TEST_MODE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Ensure src is in path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def mock_llm():
    """Mock LLM client for testing without actual model calls"""
    mock = MagicMock()
    mock.generate.return_value = "Test response"
    return mock


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "API_KEY": "test-api-key",
        "MODEL_NAME": "test-model",
        "HOST": "127.0.0.1",
        "PORT": 8000,
        "RETENTION_DAYS": 30,
    }


@pytest.fixture
def test_api_key():
    """Standard test API key"""
    return "test-api-key"


# Pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line(
        "markers", "federated: mark test as federated learning test"
    )
    config.addinivalue_line("markers", "security: mark test as security related test")
    config.addinivalue_line("markers", "asyncio: mark test as async test")


# Collection hooks to handle import errors gracefully
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle import issues"""
    for item in items:
        # Add markers based on test path
        if "federated" in str(item.fspath):
            item.add_marker(pytest.mark.federated)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "test_slow" in item.name:
            item.add_marker(pytest.mark.slow)
