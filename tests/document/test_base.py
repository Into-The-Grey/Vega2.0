"""
Comprehensive tests for document intelligence base functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass
from typing import Dict, Any, List

from src.vega.document.base import (
    BaseDocumentProcessor,
    ProcessingContext,
    ProcessingResult,
    ConfigurableComponent,
    MetricsCollector,
    DocumentIntelligenceError,
    ProcessingError,
    ValidationError,
    handle_import_error,
)

from .fixtures import (
    MockDocumentProcessor,
    TestFixtures,
    sample_documents,
    mock_responses,
    processing_context,
    mock_processor,
    failing_processor,
    slow_processor,
    initialized_processor,
    metrics_collector,
    PerformanceTestHelper,
    AsyncTestUtils,
)


class TestConfigurableComponent:
    """Test the ConfigurableComponent base class"""

    @dataclass
    class TestConfig(ConfigurableComponent):
        test_value: str = "default"
        numeric_value: int = 42
        flag_value: bool = True

        def validate_config(self) -> List[str]:
            errors = []
            if self.numeric_value < 0:
                errors.append("numeric_value must be non-negative")
            if not self.test_value:
                errors.append("test_value cannot be empty")
            return errors

    def test_default_configuration(self):
        """Test default configuration values"""
        config = self.TestConfig()
        assert config.test_value == "default"
        assert config.numeric_value == 42
        assert config.flag_value is True

    def test_custom_configuration(self):
        """Test custom configuration values"""
        config = self.TestConfig(
            test_value="custom", numeric_value=100, flag_value=False
        )
        assert config.test_value == "custom"
        assert config.numeric_value == 100
        assert config.flag_value is False

    def test_configuration_validation_success(self):
        """Test successful configuration validation"""
        config = self.TestConfig(test_value="valid", numeric_value=10)
        errors = config.validate_config()
        assert len(errors) == 0

    def test_configuration_validation_failure(self):
        """Test configuration validation with errors"""
        config = self.TestConfig(test_value="", numeric_value=-5)
        errors = config.validate_config()
        assert len(errors) == 2
        assert "test_value cannot be empty" in errors
        assert "numeric_value must be non-negative" in errors


class TestProcessingContext:
    """Test ProcessingContext functionality"""

    def test_context_creation(self):
        """Test basic context creation"""
        context = ProcessingContext(
            context_id="test_123", user_id="user_456", session_id="session_789"
        )

        assert context.context_id == "test_123"
        assert context.user_id == "user_456"
        assert context.session_id == "session_789"
        assert context.metadata == {}
        assert context.created_at is not None

    def test_context_with_metadata(self):
        """Test context with metadata"""
        metadata = {"source": "api", "priority": "high"}
        context = ProcessingContext(
            context_id="test", user_id="user", session_id="session", metadata=metadata
        )

        assert context.metadata == metadata

    def test_context_string_representation(self):
        """Test context string representation"""
        context = ProcessingContext("test", "user", "session")
        context_str = str(context)
        assert "test" in context_str
        assert "user" in context_str
        assert "session" in context_str


class TestProcessingResult:
    """Test ProcessingResult functionality"""

    def test_successful_result(self):
        """Test successful processing result"""
        context = TestFixtures.create_processing_context()
        data = {"processed": True, "count": 5}

        result = ProcessingResult.success(data, context)

        assert result.success is True
        assert result.data == data
        assert result.context == context
        assert result.error is None
        assert result.processing_time > 0

    def test_error_result(self):
        """Test error processing result"""
        context = TestFixtures.create_processing_context()
        error_msg = "Processing failed"

        result = ProcessingResult.error(error_msg, context)

        assert result.success is False
        assert result.data is None
        assert result.context == context
        assert result.error == error_msg
        assert result.processing_time > 0

    def test_result_string_representation(self):
        """Test result string representation"""
        context = TestFixtures.create_processing_context()
        result = ProcessingResult.success({"test": True}, context)
        result_str = str(result)

        assert "success=True" in result_str
        assert "test_context" in result_str


class TestMetricsCollector:
    """Test MetricsCollector functionality"""

    def test_timer_functionality(self, metrics_collector):
        """Test timer start/end functionality"""
        metrics_collector.start_timer("test_operation")

        # Verify timer started
        assert "test_operation" in metrics_collector.timers

        # End timer
        metrics_collector.end_timer("test_operation")

        # Check metrics
        metrics = metrics_collector.get_metrics()
        assert "test_operation" in metrics
        assert "duration" in metrics["test_operation"]
        assert metrics["test_operation"]["duration"] > 0

    def test_counter_functionality(self, metrics_collector):
        """Test counter increment functionality"""
        metrics_collector.increment_counter("test_counter")
        metrics_collector.increment_counter("test_counter", 5)

        metrics = metrics_collector.get_metrics()
        assert metrics["test_counter"]["count"] == 6

    def test_gauge_functionality(self, metrics_collector):
        """Test gauge set functionality"""
        metrics_collector.set_gauge("memory_usage", 1024)
        metrics_collector.set_gauge("cpu_usage", 45.5)

        metrics = metrics_collector.get_metrics()
        assert metrics["memory_usage"]["value"] == 1024
        assert metrics["cpu_usage"]["value"] == 45.5

    def test_timer_context_manager(self, metrics_collector):
        """Test timer context manager"""
        with metrics_collector.timer("context_operation"):
            # Simulate some work
            import time

            time.sleep(0.01)

        metrics = metrics_collector.get_metrics()
        assert "context_operation" in metrics
        assert metrics["context_operation"]["duration"] > 0

    def test_metrics_reset(self, metrics_collector):
        """Test metrics reset functionality"""
        metrics_collector.increment_counter("test")
        metrics_collector.set_gauge("test_gauge", 100)

        # Verify metrics exist
        assert len(metrics_collector.get_metrics()) > 0

        # Reset and verify empty
        metrics_collector.reset()
        assert len(metrics_collector.get_metrics()) == 0


class TestBaseDocumentProcessor:
    """Test BaseDocumentProcessor functionality"""

    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """Test processor initialization"""
        processor = MockDocumentProcessor()
        assert not processor.is_initialized

        await processor.initialize()
        assert processor.is_initialized

    @pytest.mark.asyncio
    async def test_processor_double_initialization(self, initialized_processor):
        """Test that double initialization is handled gracefully"""
        # Should not raise error
        await initialized_processor.initialize()
        assert initialized_processor.is_initialized

    @pytest.mark.asyncio
    async def test_successful_processing(self, processing_context):
        """Test successful document processing"""
        processor = MockDocumentProcessor()
        await processor.initialize()

        input_data = "test document content"
        result = await processor.process(input_data, processing_context)

        assert isinstance(result, ProcessingResult)
        assert result.success is True
        assert result.data["processed"] is True
        assert result.data["input_data"] == input_data
        assert result.context == processing_context

    @pytest.mark.asyncio
    async def test_processing_without_initialization(self, processing_context):
        """Test processing without initialization should auto-initialize"""
        processor = MockDocumentProcessor()

        input_data = "test content"
        result = await processor.process(input_data, processing_context)

        assert processor.is_initialized
        assert result.success is True

    @pytest.mark.asyncio
    async def test_processing_failure(self, processing_context):
        """Test processing failure handling"""
        processor = MockDocumentProcessor(should_fail=True)
        await processor.initialize()

        result = await processor.process("test", processing_context)

        assert result.success is False
        assert "Mock processing error" in result.error
        assert result.data is None

    @pytest.mark.asyncio
    async def test_processing_timeout(self):
        """Test processing timeout handling"""
        processor = MockDocumentProcessor(processing_time=2.0)
        await processor.initialize()

        context = TestFixtures.create_processing_context()

        with pytest.raises(asyncio.TimeoutError):
            await AsyncTestUtils.run_with_timeout(
                processor.process("test", context), timeout=1.0
            )

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, initialized_processor):
        """Test health check for healthy processor"""
        health = await initialized_processor.health_check()

        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert "last_check" in health

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check for uninitialized processor"""
        processor = MockDocumentProcessor()
        health = await processor.health_check()

        assert health["status"] == "not_initialized"
        assert health["initialized"] is False

    @pytest.mark.asyncio
    async def test_processing_with_metrics(self, processing_context):
        """Test that processing generates metrics"""
        processor = MockDocumentProcessor()
        await processor.initialize()

        result = await processor.process("test", processing_context)

        # Check that processor has metrics
        assert hasattr(processor, "metrics")
        metrics = processor.metrics.get_metrics()
        assert len(metrics) > 0


class TestErrorHandling:
    """Test error handling functionality"""

    def test_document_intelligence_error(self):
        """Test base DocumentIntelligenceError"""
        error = DocumentIntelligenceError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_processing_error(self):
        """Test ProcessingError"""
        error = ProcessingError("Processing failed")
        assert str(error) == "Processing failed"
        assert isinstance(error, DocumentIntelligenceError)

    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert isinstance(error, DocumentIntelligenceError)

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors are properly propagated and handled"""
        processor = MockDocumentProcessor(should_fail=True)
        context = TestFixtures.create_processing_context()

        result = await processor.process("test", context)

        # Error should be caught and returned in result
        assert result.success is False
        assert result.error is not None
        assert isinstance(result.error, str)


class TestImportErrorHandling:
    """Test optional dependency import handling"""

    def test_successful_import(self):
        """Test successful import handling"""
        result = handle_import_error("json", optional=False)  # json is always available
        assert result is True

    def test_missing_optional_import(self):
        """Test handling of missing optional import"""
        result = handle_import_error("nonexistent_module_xyz", optional=True)
        assert result is False

    def test_missing_required_import(self):
        """Test handling of missing required import"""
        with pytest.raises(ImportError):
            handle_import_error("nonexistent_module_xyz", optional=False)


class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_processing_performance(self):
        """Test processing performance measurement"""
        processor = MockDocumentProcessor(processing_time=0.1)
        await processor.initialize()

        input_data = "test content"
        context = TestFixtures.create_processing_context()

        perf_result = await PerformanceTestHelper.measure_processing_time(
            processor, input_data, context
        )

        assert perf_result["processing_time"] >= 0.1
        assert perf_result["result"].success is True

    @pytest.mark.asyncio
    async def test_load_testing(self):
        """Test processor under load"""
        processor = MockDocumentProcessor(processing_time=0.05)
        await processor.initialize()

        load_result = await PerformanceTestHelper.run_load_test(
            processor, "test content", num_requests=20, concurrency=5
        )

        assert load_result["total_requests"] == 20
        assert load_result["successful_requests"] == 20
        assert load_result["error_count"] == 0
        assert load_result["avg_processing_time"] >= 0.05

    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        processor = MockDocumentProcessor(processing_time=0.1)
        await processor.initialize()

        # Run multiple processing tasks concurrently
        tasks = []
        for i in range(10):
            context = TestFixtures.create_processing_context(f"context_{i}")
            task = processor.process(f"content_{i}", context)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 10

        # Verify each got unique input
        for i, result in enumerate(results):
            assert result.data["input_data"] == f"content_{i}"
            assert result.data["context_id"] == f"context_{i}"


class TestAsyncUtilities:
    """Test async testing utilities"""

    @pytest.mark.asyncio
    async def test_wait_for_condition_success(self):
        """Test waiting for condition that becomes true"""
        counter = {"value": 0}

        def condition():
            counter["value"] += 1
            return counter["value"] >= 3

        result = await AsyncTestUtils.wait_for_condition(
            condition, timeout=1.0, poll_interval=0.1
        )
        assert result is True
        assert counter["value"] >= 3

    @pytest.mark.asyncio
    async def test_wait_for_condition_timeout(self):
        """Test waiting for condition that times out"""

        def condition():
            return False

        result = await AsyncTestUtils.wait_for_condition(
            condition, timeout=0.2, poll_interval=0.1
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_run_with_timeout_success(self):
        """Test running coroutine that completes within timeout"""

        async def quick_operation():
            await asyncio.sleep(0.1)
            return "completed"

        result = await AsyncTestUtils.run_with_timeout(quick_operation(), timeout=1.0)
        assert result == "completed"

    @pytest.mark.asyncio
    async def test_run_with_timeout_failure(self):
        """Test running coroutine that exceeds timeout"""

        async def slow_operation():
            await asyncio.sleep(1.0)
            return "completed"

        with pytest.raises(TimeoutError):
            await AsyncTestUtils.run_with_timeout(slow_operation(), timeout=0.2)
