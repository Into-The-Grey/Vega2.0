"""
Pytest configuration for document intelligence tests
"""

import pytest
import asyncio
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Configure logging with proper file handler management
    # Remove any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Use a context-managed file handler approach via logging config
    file_handler = logging.FileHandler("test.log", mode="a")
    file_handler.setLevel(logging.INFO)

    # Store file handler on config for cleanup
    if not hasattr(config, "_log_file_handler"):
        config._log_file_handler = file_handler

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Disable noisy loggers during testing
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

    # Add custom markers
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "asyncio: mark test as async")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items"""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)

        # Add slow marker to performance tests
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Add integration marker to health check tests
        if "health_check" in item.name.lower():
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def clean_up_processors():
    """Clean up any processors after each test"""
    yield
    # Cleanup code can be added here if needed
    # For example, clearing caches, closing connections, etc.


@pytest.fixture(scope="session")
def test_data_directory():
    """Provide path to test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def temp_directory():
    """Provide a temporary directory for test files"""
    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir

    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


# Performance test configuration
@pytest.fixture
def performance_timeout():
    """Timeout for performance tests"""
    return 30.0  # 30 seconds


@pytest.fixture
def load_test_config():
    """Configuration for load tests"""
    return {"num_requests": 50, "concurrency": 10, "timeout": 5.0}


# Mock configuration
@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "enable_all_features": True,
        "timeout_seconds": 30.0,
        "min_confidence": 0.5,  # Lower for testing
        "use_transformers": False,  # Disable for faster tests
        "max_content_length": 10000,
    }


# Session-scoped model caching to avoid reloading heavy models
_cached_models = {}


@pytest.fixture(scope="session")
def cached_legal_model():
    """Cache legal-bert model for entire test session to improve performance"""
    if "legal_bert" not in _cached_models:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            model_name = "nlpaueb/legal-bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)

            _cached_models["legal_bert"] = {
                "tokenizer": tokenizer,
                "model": model,
                "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            }
        except Exception as e:
            _cached_models["legal_bert"] = None

    return _cached_models.get("legal_bert")


@pytest.fixture(scope="session")
def disable_heavy_ml_models(monkeypatch_session):
    """Disable heavy ML model loading for faster tests"""
    # This would require pytest-monkeypatch-session plugin
    # For now, we'll rely on configuration flags
    pass


# Error testing fixtures
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing"""
    return {
        "empty_input": "",
        "whitespace_only": "   \n\t   ",
        "too_long": "x" * 100000,  # 100K characters
        "invalid_characters": "\x00\x01\x02",
        "malformed_json": '{"incomplete": ',
        "non_string_input": {"invalid": "type"},
    }


# Test data fixtures
@pytest.fixture
def sample_test_documents():
    """Sample documents for testing"""
    return {
        "short": "This is a short test document.",
        "medium": "This is a medium-length test document that contains multiple sentences and provides more content for analysis. It includes various topics and should be sufficient for most testing scenarios.",
        "long": """
        This is a long test document that contains extensive content for thorough testing.
        It includes multiple paragraphs, various sentence structures, and different types of content
        that can be used to test the robustness of document processing algorithms.
        
        The document covers several topics including technology, business, and general information.
        It is designed to trigger various classification and analysis pathways in the system.
        
        Additionally, this document contains enough text to test performance characteristics
        and memory usage patterns of the document intelligence system.
        """,
        "multilingual": """
        English: This document contains multiple languages for testing.
        Spanish: Este documento contiene múltiples idiomas para pruebas.
        French: Ce document contient plusieurs langues pour les tests.
        """,
        "special_chars": "Test with émojis 🚀 and spëcial çharaçters: àáâãäåæçèéêëìíîï",
    }


# Async testing helpers
@pytest.fixture
async def async_test_helper():
    """Helper for async testing operations"""

    class AsyncTestHelper:
        @staticmethod
        async def wait_for(condition, timeout=5.0):
            """Wait for a condition with timeout"""
            import time

            start = time.time()
            while time.time() - start < timeout:
                if (
                    await condition()
                    if asyncio.iscoroutinefunction(condition)
                    else condition()
                ):
                    return True
                await asyncio.sleep(0.1)
            return False

        @staticmethod
        async def run_with_timeout(coro, timeout=5.0):
            """Run coroutine with timeout"""
            return await asyncio.wait_for(coro, timeout=timeout)

    return AsyncTestHelper()


# Monitoring and metrics fixtures
@pytest.fixture
def metrics_collector():
    """Provide a metrics collector for testing"""
    from src.vega.document.base import MetricsCollector

    return MetricsCollector()


@pytest.fixture
def memory_monitor():
    """Memory usage monitoring for tests"""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    yield {
        "initial": initial_memory,
        "get_current": lambda: process.memory_info().rss,
        "get_increase": lambda: process.memory_info().rss - initial_memory,
    }


# Test result validation
@pytest.fixture
def result_validator():
    """Validator for test results"""

    class ResultValidator:
        @staticmethod
        def validate_processing_result(result):
            """Validate ProcessingResult structure"""
            assert hasattr(result, "success")
            assert hasattr(result, "data")
            assert hasattr(result, "error")
            assert hasattr(result, "context")
            assert hasattr(result, "processing_time")

            if result.success:
                assert result.data is not None
                assert result.error is None
            else:
                assert result.error is not None
                assert isinstance(result.error, str)
                assert len(result.error) > 0

        @staticmethod
        def validate_health_check(health):
            """Validate health check result structure"""
            assert "status" in health
            assert health["status"] in [
                "healthy",
                "degraded",
                "unhealthy",
                "not_initialized",
            ]
            assert "last_check" in health
            assert "initialized" in health

    return ResultValidator()


# Test environment setup
def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    print("🧪 Starting Document Intelligence Test Session")
    print(f"📁 Project root: {project_root}")
    print("=" * 50)


def pytest_sessionfinish(session, exitstatus):
    """Called after whole test run finished"""
    print("\n" + "=" * 50)
    if exitstatus == 0:
        print("✅ Test session completed successfully")
    else:
        print(f"❌ Test session failed with exit status: {exitstatus}")
    print("🧪 Document Intelligence Test Session Complete")

    # Clean up file handler to prevent resource leak
    if hasattr(session.config, "_log_file_handler"):
        try:
            session.config._log_file_handler.close()
            logging.getLogger().removeHandler(session.config._log_file_handler)
        except Exception:
            pass


# NOTE: pytest_plugins must be defined at the root conftest.py, not here
# If you need custom plugins, move them to /home/ncacord/Vega2.0/conftest.py

# Import test fixture data from the fixtures module so these are available
# as session fixtures for all document tests. Pytest only auto-registers
# fixtures declared in conftest.py or test modules; tests/document/fixtures.py
# contains useful sample data and helpers but isn't a conftest, so import
# the values here and expose lightweight wrappers.
try:
    from tests.document import fixtures as doc_fixtures
except Exception:
    doc_fixtures = None


@pytest.fixture
def sample_documents():
    """Provide project-wide sample documents for document tests."""
    if doc_fixtures is None:
        return {}
    return getattr(doc_fixtures, "SAMPLE_DOCUMENTS", {})


@pytest.fixture
def sample_technical_documents():
    """Alias wrapper for technical document test data."""
    if doc_fixtures is None:
        return {}
    # fixtures module exposes a module-level dict named sample_technical_documents
    return getattr(doc_fixtures, "sample_technical_documents", {})


@pytest.fixture
def sample_workflow_documents():
    """Alias wrapper for workflow document test data."""
    if doc_fixtures is None:
        return {}
    return getattr(doc_fixtures, "sample_workflow_documents", {})


@pytest.fixture
def sample_legal_documents():
    """Alias wrapper for legal document test data."""
    if doc_fixtures is None:
        return {}
    # Check if it's a pytest fixture function first (it is in fixtures.py)
    # If we got the function object, we need to return a dict directly instead
    fixture_func_or_data = getattr(doc_fixtures, "sample_legal_documents", None)

    # Return dict mapping directly - don't try to call the fixture function
    # since we're outside the pytest fixture resolution context
    sd = getattr(doc_fixtures, "SAMPLE_DOCUMENTS", {})
    return {
        "nda": sd.get("contract", ""),
        "service_agreement": sd.get("contract", ""),
        "privacy_policy": sd.get("workflow_doc", ""),
        "liability_clause": sd.get("contract", ""),
        "complex_contract": sd.get("contract", ""),
        "entity_heavy_contract": sd.get("research_paper", ""),
    }


@pytest.fixture
def create_test_context():
    """Return a helper that builds ProcessingContext objects for tests.

    Tests expect to be able to call create_test_context(session_id=..., ...).
    Provide a flexible wrapper that accepts the common args and delegates to
    the TestFixtures helper if available.
    """

    def _create(
        content: str | None = None,
        document_type: str = "test",
        processing_mode: str = "analysis",
        session_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        metadata: dict | None = None,
    ):
        # Prefer the helper in the fixtures module when available
        if doc_fixtures is not None and hasattr(doc_fixtures, "TestFixtures"):
            return doc_fixtures.TestFixtures.create_processing_context(
                context_id=context_id or "test_context",
                user_id=user_id or "test_user",
                session_id=session_id or "test_session",
                metadata=metadata or {},
            )

        # Fallback: construct a minimal ProcessingContext by importing class
        try:
            from src.vega.document.base import ProcessingContext

            return ProcessingContext(
                context_id=context_id or "test_context",
                user_id=user_id or "test_user",
                session_id=session_id or "test_session",
                metadata=metadata or {},
            )
        except Exception:
            # Last resort: return a simple namespace-like object
            class _Ctx:
                def __init__(self):
                    self.context_id = context_id or "test_context"
                    self.user_id = user_id or "test_user"
                    self.session_id = session_id or "test_session"
                    self.metadata = metadata or {}

            return _Ctx()

    return _create
