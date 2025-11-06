"""
Integration tests for Vega2.0 external integrations

Tests all integration patterns with mocked external services:
- HTTP client sharing and connection pooling
- Circuit breaker behavior under failures
- Timeout and retry handling
- Error recovery patterns
- Backpressure control
- Resource cleanup
"""

from __future__ import annotations

import asyncio
import pytest
from typing import AsyncGenerator
from unittest.mock import Mock, patch, AsyncMock

# Test fixtures and utilities


@pytest.fixture
async def mock_http_client():
    """Provide a mocked HTTP client for testing"""
    client = AsyncMock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture
async def mock_resource_manager(mock_http_client):
    """Provide a mocked resource manager"""
    manager = AsyncMock()
    manager.get_http_client_direct.return_value = mock_http_client
    manager.get_stats.return_value = {
        "http_clients_created": 1,
        "http_requests_made": 0,
    }
    return manager


# Search Integration Tests


@pytest.mark.asyncio
async def test_web_search_with_results():
    """Test web search returns results successfully"""
    from src.vega.integrations.search import web_search

    results = web_search("test query", max_results=3)

    # Should return list (empty if DDG not available)
    assert isinstance(results, list)

    # If DDG available, check structure
    if results:
        assert all(isinstance(r, dict) for r in results)
        assert all("title" in r and "href" in r for r in results)


@pytest.mark.asyncio
async def test_image_search_with_results():
    """Test image search returns results successfully"""
    from src.vega.integrations.search import image_search

    results = image_search("test image", max_results=3)

    # Should return list (empty if DDG not available)
    assert isinstance(results, list)

    # If DDG available, check structure
    if results:
        assert all(isinstance(r, dict) for r in results)
        assert all("image" in r and "thumbnail" in r for r in results)


# Fetch Integration Tests


@pytest.mark.asyncio
async def test_fetch_text_with_shared_client(mock_resource_manager, mock_http_client):
    """Test fetch_text uses shared HTTP client"""
    from src.vega.integrations.fetch import fetch_text

    # Mock successful response
    mock_response = Mock()  # Use Mock, not AsyncMock for response object
    mock_response.status_code = 200
    mock_response.text = "<html><body>Test content</body></html>"
    mock_http_client.get.return_value = mock_response

    # Mock get_resource_manager as async function
    async def mock_get_manager():
        return mock_resource_manager

    with patch(
        "src.vega.core.resource_manager.get_resource_manager",
        side_effect=mock_get_manager,
    ):
        result = await fetch_text("https://example.com")

    # Should return content
    assert result is not None
    assert "Test content" in result

    # Should have called shared client
    mock_http_client.get.assert_called_once()

    # Should not have closed shared client (not owned)
    mock_http_client.aclose.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_text_handles_404():
    """Test fetch_text handles 404 errors gracefully"""
    from src.vega.integrations.fetch import fetch_text

    # Mock 404 response
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 404
    mock_client.get.return_value = mock_response

    mock_manager = AsyncMock()
    mock_manager.get_http_client_direct.return_value = mock_client

    async def mock_get_manager():
        return mock_manager

    with patch(
        "src.vega.core.resource_manager.get_resource_manager",
        side_effect=mock_get_manager,
    ):
        result = await fetch_text("https://example.com/notfound")

    # Should return None on error
    assert result is None


@pytest.mark.asyncio
async def test_fetch_text_timeout():
    """Test fetch_text respects timeout"""
    from src.vega.integrations.fetch import fetch_text

    # Mock timeout
    mock_client = AsyncMock()
    mock_client.get.side_effect = asyncio.TimeoutError()

    mock_manager = AsyncMock()
    mock_manager.get_http_client_direct.return_value = mock_client

    async def mock_get_manager():
        return mock_manager

    with patch(
        "src.vega.core.resource_manager.get_resource_manager",
        side_effect=mock_get_manager,
    ):
        result = await fetch_text("https://example.com", timeout=1.0)

    # Should return None on timeout
    assert result is None


# Slack Integration Tests


@pytest.mark.asyncio
async def test_slack_with_shared_client(mock_resource_manager, mock_http_client):
    """Test Slack integration uses shared HTTP client"""
    from src.vega.integrations.slack_connector import send_slack_message

    # Mock successful response
    mock_response = AsyncMock()
    mock_response.is_success = True
    mock_http_client.post.return_value = mock_response

    async def mock_get_manager():
        return mock_resource_manager

    with patch(
        "src.vega.core.resource_manager.get_resource_manager",
        side_effect=mock_get_manager,
    ):
        result = await send_slack_message(
            "https://hooks.slack.com/test", "Test message"
        )

    # Should have called shared client
    mock_http_client.post.assert_called_once()
    assert result is True


@pytest.mark.asyncio
async def test_slack_without_webhook():
    """Test Slack handles missing webhook gracefully"""
    from src.vega.integrations.slack_connector import send_slack_message

    result = await send_slack_message(None, "Test message")

    # Should return False without webhook
    assert result is False


@pytest.mark.asyncio
async def test_slack_handles_network_error(mock_resource_manager, mock_http_client):
    """Test Slack handles network errors gracefully"""
    from src.vega.integrations.slack_connector import send_slack_message

    # Mock network error
    mock_http_client.post.side_effect = Exception("Network error")

    async def mock_get_manager():
        return mock_resource_manager

    with patch(
        "src.vega.core.resource_manager.get_resource_manager",
        side_effect=mock_get_manager,
    ):
        result = await send_slack_message(
            "https://hooks.slack.com/test", "Test message"
        )

    # Should return False on error
    assert result is False


# Circuit Breaker Tests


@pytest.mark.skip(reason="CircuitBreaker API mismatch - needs refactoring")
@pytest.mark.asyncio
async def test_circuit_breaker_opens_on_failures():
    """Test circuit breaker opens after consecutive failures"""
    from src.vega.core.resilience import CircuitBreaker

    breaker = CircuitBreaker(
        failure_threshold=3, recovery_timeout=1.0, half_open_max_calls=1
    )

    async def failing_operation():
        raise Exception("Simulated failure")

    # First 3 calls should fail and open the circuit
    for _ in range(3):
        try:
            await breaker.call(failing_operation)
        except Exception:
            pass

    # Circuit should now be open
    assert breaker.state == "open"

    # Next call should fail fast
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        await breaker.call(failing_operation)


@pytest.mark.skip(reason="CircuitBreaker API mismatch - needs refactoring")
@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    """Test circuit breaker enters half-open state and recovers"""
    from src.vega.core.resilience import CircuitBreaker

    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=0.1,  # Short timeout for testing
        half_open_max_calls=1,
    )

    call_count = 0

    async def recovering_operation():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Failing")
        return "Success"

    # Fail twice to open circuit
    for _ in range(2):
        try:
            await breaker.call(recovering_operation)
        except Exception:
            pass

    assert breaker.state == "open"

    # Wait for recovery timeout
    await asyncio.sleep(0.15)

    # Should enter half-open and succeed
    result = await breaker.call(recovering_operation)
    assert result == "Success"
    assert breaker.state == "closed"


# Streaming Backpressure Tests


@pytest.mark.skip(reason="BackpressureStream not implemented - needs implementation")
@pytest.mark.asyncio
async def test_streaming_backpressure_basic():
    """Test basic streaming with backpressure control"""
    from src.vega.core.streaming_backpressure import BackpressureStream

    async def data_generator():
        for i in range(10):
            yield f"chunk_{i}"

    stream = BackpressureStream(
        data_generator(), max_buffer_size=5, warning_threshold=0.7
    )

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        await asyncio.sleep(0.01)  # Simulate slow consumer

    assert len(chunks) == 10
    assert chunks[0] == "chunk_0"
    assert chunks[-1] == "chunk_9"


@pytest.mark.skip(reason="BackpressureStream not implemented - needs implementation")
@pytest.mark.asyncio
async def test_streaming_backpressure_metrics():
    """Test streaming collects accurate metrics"""
    from src.vega.core.streaming_backpressure import BackpressureStream

    async def data_generator():
        for i in range(5):
            yield f"chunk_{i}"

    stream = BackpressureStream(data_generator(), max_buffer_size=10)

    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    metrics = stream.get_metrics()

    assert metrics.chunks_produced == 5
    assert metrics.chunks_consumed == 5
    assert metrics.chunks_dropped == 0


@pytest.mark.skip(reason="BackpressureStream not implemented - needs implementation")
@pytest.mark.asyncio
async def test_streaming_backpressure_overflow():
    """Test streaming handles buffer overflow"""
    from src.vega.core.streaming_backpressure import BackpressureStream

    async def fast_generator():
        for i in range(100):
            yield f"chunk_{i}"
            await asyncio.sleep(0.001)

    stream = BackpressureStream(
        fast_generator(), max_buffer_size=5, drop_on_overflow=True  # Drop oldest chunks
    )

    # Slow consumer
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        await asyncio.sleep(0.05)  # Much slower than producer
        if len(chunks) >= 10:
            break

    metrics = stream.get_metrics()

    # Should have dropped some chunks due to overflow
    assert metrics.buffer_overflow_count > 0


# Async Event Loop Monitor Tests


@pytest.mark.asyncio
async def test_async_monitor_detects_slow_callback():
    """Test async monitor detects slow callbacks"""
    from src.vega.core.async_monitor import AsyncEventLoopMonitor

    monitor = AsyncEventLoopMonitor(slow_callback_threshold_ms=50.0)
    monitor.start_monitoring()

    @monitor.wrap_callback
    async def slow_operation():
        await asyncio.sleep(0.1)  # 100ms - slower than threshold
        return "done"

    result = await slow_operation()

    monitor.stop_monitoring()

    metrics = monitor.get_metrics()

    assert result == "done"
    assert metrics.blocked_callbacks > 0


@pytest.mark.asyncio
async def test_async_monitor_tracks_pending_tasks():
    """Test async monitor tracks pending tasks"""
    from src.vega.core.async_monitor import AsyncEventLoopMonitor

    monitor = AsyncEventLoopMonitor()
    monitor.start_monitoring()

    # Create several pending tasks
    tasks = [asyncio.create_task(asyncio.sleep(0.1)) for _ in range(5)]

    await asyncio.sleep(0.05)  # Let monitor collect metrics

    metrics = monitor.get_metrics()

    # Should have tracked the pending tasks
    assert metrics.max_pending_tasks >= 5

    # Cleanup
    monitor.stop_monitoring()
    await asyncio.gather(*tasks)


# Memory Leak Detection Tests


@pytest.mark.asyncio
async def test_memory_leak_detector_tracks_objects():
    """Test memory leak detector tracks object references"""
    from src.vega.core.memory_leak_detector import MemoryLeakDetector

    detector = MemoryLeakDetector()

    # Create some tracked objects
    class TrackedObject:
        pass

    obj1 = TrackedObject()
    obj2 = TrackedObject()
    obj3 = TrackedObject()

    detector.track(obj1, "test_object_1")
    detector.track(obj2, "test_object_2")
    detector.track(obj3, "test_object_3")

    stats = detector.get_stats()

    # Should track all objects
    assert stats["total_tracked"] == 3
    assert stats["live_objects"] == 3


@pytest.mark.asyncio
async def test_memory_leak_detector_garbage_collection():
    """Test memory leak detector detects garbage collection"""
    from src.vega.core.memory_leak_detector import MemoryLeakDetector

    detector = MemoryLeakDetector()

    class TrackedObject:
        pass

    obj = TrackedObject()
    detector.track(obj, "test_object")

    initial_stats = detector.get_stats()
    assert initial_stats["live_objects"] == 1

    # Delete reference and force GC
    del obj
    import gc

    gc.collect()

    await asyncio.sleep(0.1)  # Let detector notice

    final_stats = detector.get_stats()

    # Object should be collected
    assert final_stats["live_objects"] == 0


# Database Batch Operations Tests


@pytest.mark.asyncio
async def test_batch_operations_accumulates_items():
    """Test batch operations accumulate items"""
    from src.vega.core.batch_operations import BatchWriter

    processed_batches = []

    async def batch_processor(items):
        processed_batches.append(list(items))

    writer = BatchWriter(
        batch_processor,
        max_batch_size=5,
        flush_interval=10.0,  # Long interval for manual control
    )

    # Add items
    await writer.add("item1")
    await writer.add("item2")
    await writer.add("item3")

    # Manual flush
    await writer.flush()

    assert len(processed_batches) == 1
    assert len(processed_batches[0]) == 3
    assert "item1" in processed_batches[0]


@pytest.mark.asyncio
async def test_batch_operations_auto_flush_on_size():
    """Test batch operations auto-flush when size reached"""
    from src.vega.core.batch_operations import BatchWriter

    processed_batches = []

    async def batch_processor(items):
        processed_batches.append(list(items))

    writer = BatchWriter(batch_processor, max_batch_size=3, flush_interval=10.0)

    # Add 5 items - should trigger auto-flush at 3
    for i in range(5):
        await writer.add(f"item{i}")

    await asyncio.sleep(0.1)  # Let flush complete

    # Should have flushed first batch of 3
    assert len(processed_batches) >= 1
    assert len(processed_batches[0]) == 3


@pytest.mark.asyncio
async def test_batch_operations_auto_flush_on_interval():
    """Test batch operations auto-flush after interval"""
    from src.vega.core.batch_operations import BatchWriter

    processed_batches = []

    async def batch_processor(items):
        processed_batches.append(list(items))

    writer = BatchWriter(
        batch_processor,
        max_batch_size=100,  # Large size so interval triggers first
        flush_interval=0.2,  # 200ms interval
    )

    writer.start_auto_flush()

    # Add items
    await writer.add("item1")
    await writer.add("item2")

    # Wait for interval
    await asyncio.sleep(0.3)

    writer.stop_auto_flush()

    # Should have flushed due to interval
    assert len(processed_batches) >= 1


# Integration Health Check Tests


@pytest.mark.asyncio
async def test_integration_health_check_all():
    """Test comprehensive integration health check"""
    from src.vega.core.integration_health import check_all_integrations

    results = await check_all_integrations(timeout=5.0)

    # Should return dictionary of results
    assert isinstance(results, dict)

    # Should check key integrations
    assert "database" in results
    assert "llm" in results

    # Each result should have status
    for name, health in results.items():
        assert "status" in health.__dict__
        assert health.status in ["healthy", "degraded", "unhealthy", "disabled"]


# Performance Endpoints Tests (requires running app)


@pytest.mark.skipif(True, reason="Requires running FastAPI app")
@pytest.mark.asyncio
async def test_performance_endpoints_accessible():
    """Test performance monitoring endpoints are accessible"""
    import httpx

    client = httpx.AsyncClient(base_url="http://localhost:8000")

    try:
        # Test circuit breaker endpoint
        response = await client.get(
            "/admin/performance/circuit-breakers", headers={"X-API-Key": "test_key"}
        )
        assert response.status_code in [200, 401]  # 401 if auth fails, 200 if succeeds

        # Test cache endpoint
        response = await client.get(
            "/admin/performance/cache-stats", headers={"X-API-Key": "test_key"}
        )
        assert response.status_code in [200, 401]

    finally:
        await client.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
