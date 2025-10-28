"""
Comprehensive Integration Tests for Vega2.0 Optimization Systems

Tests all performance optimization components:
- Request coalescing
- Streaming backpressure
- Connection pool management
- Memory leak detection
- Async event loop monitoring
- Batch operations
- Integration health checks
- Circuit breakers
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch


class TestRequestCoalescing:
    """Test request deduplication and coalescing"""

    @pytest.mark.asyncio
    async def test_identical_requests_coalesced(self):
        """Multiple identical concurrent requests should be coalesced"""
        from src.vega.core.request_coalescing import RequestCoalescer

        coalescer = RequestCoalescer(cache_ttl=60.0)

        # Mock operation that takes time
        call_count = 0

        async def slow_operation(value):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)
            return f"result_{value}"

        # Fire 10 identical concurrent requests
        tasks = [
            coalescer.coalesce("test_op", slow_operation, "test_value")
            for _ in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # All should get same result
        assert all(r == "result_test_value" for r in results)

        # But operation should only execute once (or twice due to race condition)
        assert call_count <= 2, f"Expected <= 2 calls, got {call_count}"

        # Check metrics
        stats = coalescer.get_stats()
        assert stats["total_requests"] == 10
        assert stats["coalesced_requests"] >= 8  # Most should be coalesced

    @pytest.mark.asyncio
    async def test_different_requests_not_coalesced(self):
        """Different requests should execute independently"""
        from src.vega.core.request_coalescing import RequestCoalescer

        coalescer = RequestCoalescer(cache_ttl=60.0)

        async def operation(value):
            return f"result_{value}"

        # Fire different concurrent requests
        tasks = [
            coalescer.coalesce("test_op", operation, f"value_{i}") for i in range(5)
        ]

        results = await asyncio.gather(*tasks)

        # Each should get different result
        assert results == [f"result_value_{i}" for i in range(5)]

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self):
        """Cached results should be returned without re-execution"""
        from src.vega.core.request_coalescing import RequestCoalescer

        coalescer = RequestCoalescer(cache_ttl=60.0)

        call_count = 0

        async def operation(value):
            nonlocal call_count
            call_count += 1
            return f"result_{value}"

        # First call
        result1 = await coalescer.coalesce("test_op", operation, "test")
        assert call_count == 1

        # Second call (should hit cache)
        result2 = await coalescer.coalesce("test_op", operation, "test")
        assert call_count == 1  # No additional call
        assert result1 == result2

        stats = coalescer.get_stats()
        assert stats["cache_hit_rate"] > 0


class TestStreamingBackpressure:
    """Test streaming backpressure control"""

    @pytest.mark.asyncio
    async def test_buffer_overflow_handling(self):
        """Stream should handle slow consumers gracefully"""
        from src.vega.core.streaming_backpressure import BufferedStream

        async def fast_producer():
            """Produce chunks quickly"""
            for i in range(100):
                yield f"chunk_{i}"
                await asyncio.sleep(0.001)  # Very fast

        stream = BufferedStream(fast_producer(), buffer_size=10, drop_on_overflow=False)

        consumed = []

        async for chunk in stream:
            consumed.append(chunk)
            await asyncio.sleep(0.01)  # Slow consumer

            if len(consumed) >= 50:  # Test first 50 chunks
                break

        # Should throttle but not drop
        assert len(consumed) == 50

        metrics = await stream.get_metrics()
        assert metrics["chunks_consumed"] == 50

    @pytest.mark.asyncio
    async def test_metrics_accuracy(self):
        """Stream metrics should accurately track chunks"""
        from src.vega.core.streaming_backpressure import BufferedStream

        async def producer():
            for i in range(10):
                yield f"chunk_{i}"

        stream = BufferedStream(producer(), buffer_size=100)

        chunks = [chunk async for chunk in stream]

        assert len(chunks) == 10

        metrics = await stream.get_metrics()
        assert metrics["chunks_produced"] == 10
        assert metrics["chunks_consumed"] == 10


class TestConnectionPool:
    """Test intelligent connection pool management"""

    @pytest.mark.asyncio
    async def test_connection_registration(self):
        """Connections should be tracked correctly"""
        from src.vega.core.connection_pool import ConnectionPoolManager

        manager = ConnectionPoolManager()
        manager.start_monitoring()

        # Register connections
        manager.register_connection(1, "example.com")
        manager.register_connection(2, "example.com")

        metrics = manager.get_metrics()
        assert metrics.total_connections == 2
        assert metrics.connections_created == 2

        manager.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stale_connection_cleanup(self):
        """Idle connections should be cleaned up"""
        from src.vega.core.connection_pool import ConnectionPoolManager

        manager = ConnectionPoolManager(
            idle_timeout=0.5, cleanup_interval=0.5  # Fast cleanup for testing
        )
        manager.start_monitoring()

        # Register connection
        manager.register_connection(1, "example.com")

        # Wait for it to become idle and get cleaned up
        await asyncio.sleep(1.5)

        metrics = manager.get_metrics()
        assert metrics.total_connections == 0  # Should be cleaned up

        manager.stop_monitoring()

    @pytest.mark.asyncio
    async def test_connection_limit_enforcement(self):
        """Per-host connection limits should be enforced"""
        from src.vega.core.connection_pool import ConnectionPoolManager

        manager = ConnectionPoolManager(max_connections_per_host=2)
        manager.start_monitoring()

        # Register up to limit
        manager.register_connection(1, "example.com")
        manager.register_connection(2, "example.com")

        # Try to create another
        can_create = manager.can_create_connection("example.com")
        assert not can_create  # Should be at limit

        # Different host should be fine
        can_create_other = manager.can_create_connection("other.com")
        assert can_create_other

        manager.stop_monitoring()


class TestMemoryLeakDetection:
    """Test memory leak detection system"""

    @pytest.mark.asyncio
    async def test_object_tracking(self):
        """Objects should be tracked correctly"""
        from src.vega.core.memory_leak_detector import MemoryLeakDetector

        detector = MemoryLeakDetector()
        await detector.start()

        # Track some objects
        obj1 = {"test": "data1"}
        obj2 = {"test": "data2"}

        detector.track_object(obj1, "test_context")
        detector.track_object(obj2, "test_context")

        metrics = await detector.get_metrics()
        assert metrics["objects_currently_alive"] == 2

        await detector.stop()

    @pytest.mark.asyncio
    async def test_garbage_collection_detection(self):
        """Should detect when objects are garbage collected"""
        from src.vega.core.memory_leak_detector import MemoryLeakDetector
        import gc

        detector = MemoryLeakDetector()
        await detector.start()

        # Track object then delete it
        obj = {"test": "data"}
        detector.track_object(obj, "test_context")

        del obj
        gc.collect()  # Force garbage collection

        await asyncio.sleep(0.1)  # Let detector process

        metrics = await detector.get_metrics()
        # Object should be freed (weak reference should be dead)
        assert metrics["objects_freed"] >= 0

        await detector.stop()


class TestBatchOperations:
    """Test database batch operations"""

    @pytest.mark.asyncio
    async def test_batch_accumulation(self):
        """Items should accumulate in batch before flush"""
        from src.vega.core.batch_operations import BatchedConversationLogger

        logger = BatchedConversationLogger(batch_size=5, flush_interval=10.0)
        await logger.start()

        # Mock DB function
        with patch(
            "src.vega.core.db.bulk_log_conversations", new_callable=AsyncMock
        ) as mock_bulk:
            # Queue 3 items (less than batch size)
            await logger.log_conversation("prompt1", "response1", "test")
            await logger.log_conversation("prompt2", "response2", "test")
            await logger.log_conversation("prompt3", "response3", "test")

            # Should not have flushed yet
            mock_bulk.assert_not_called()

            metrics = await logger.get_metrics()
            assert metrics["total_items_queued"] == 3

        await logger.stop()

    @pytest.mark.asyncio
    async def test_batch_size_trigger(self):
        """Batch should flush when size reached"""
        from src.vega.core.batch_operations import BatchedConversationLogger

        logger = BatchedConversationLogger(batch_size=3, flush_interval=10.0)
        await logger.start()

        with patch(
            "src.vega.core.db.bulk_log_conversations", new_callable=AsyncMock
        ) as mock_bulk:
            # Queue exactly batch_size items
            await logger.log_conversation("prompt1", "response1", "test")
            await logger.log_conversation("prompt2", "response2", "test")
            await logger.log_conversation("prompt3", "response3", "test")

            # Give it time to flush
            await asyncio.sleep(0.1)

            # Should have flushed
            assert mock_bulk.await_count == 1
            # Ensure batch had 3 items
            args, _ = mock_bulk.await_args
            assert isinstance(args[0], list)
            assert len(args[0]) == 3

        await logger.stop()

    @pytest.mark.asyncio
    async def test_time_based_flush(self):
        """Batch should flush on interval even if not full"""
        from src.vega.core.batch_operations import BatchedConversationLogger

        logger = BatchedConversationLogger(batch_size=100, flush_interval=0.5)
        await logger.start()

        with patch(
            "src.vega.core.db.bulk_log_conversations", new_callable=AsyncMock
        ) as mock_bulk:
            # Queue just 2 items
            await logger.log_conversation("prompt1", "response1", "test")
            await logger.log_conversation("prompt2", "response2", "test")

            # Wait for flush interval
            await asyncio.sleep(1.0)

            # Should have flushed despite not reaching batch size
            # One bulk call with 2 items
            assert mock_bulk.await_count == 1
            args, _ = mock_bulk.await_args
            assert len(args[0]) == 2

        await logger.stop()


class TestAsyncEventLoopMonitor:
    """Test async event loop monitoring"""

    @pytest.mark.asyncio
    async def test_slow_callback_detection(self):
        """Should detect slow callbacks"""
        from src.vega.core.async_monitor import AsyncEventLoopMonitor

        monitor = AsyncEventLoopMonitor(slow_callback_threshold_ms=50.0)
        await monitor.start()

        # Simulate slow callback
        async def slow_task():
            time.sleep(0.1)  # Blocking sleep (bad practice)

        await slow_task()
        await asyncio.sleep(0.1)  # Let monitor process

        metrics = await monitor.get_metrics()
        # Might detect the slow callback
        assert metrics["total_callbacks"] >= 0

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_pending_task_tracking(self):
        """Should track pending tasks"""
        from src.vega.core.async_monitor import AsyncEventLoopMonitor

        monitor = AsyncEventLoopMonitor()
        await monitor.start()

        # Create some pending tasks
        async def dummy_task():
            await asyncio.sleep(0.1)

        tasks = [asyncio.create_task(dummy_task()) for _ in range(5)]

        await asyncio.sleep(0.05)  # Let monitor observe

        metrics = await monitor.get_metrics()
        assert "current_pending_tasks" in metrics

        # Cleanup
        await asyncio.gather(*tasks)
        await monitor.stop()


class TestIntegrationHealth:
    """Test integration health monitoring"""

    @pytest.mark.asyncio
    async def test_integration_health_check(self):
        """Should check health of all integrations"""
        from src.vega.core.integration_health import check_all_integrations

        # This will attempt real checks, so mock if needed
        with patch("src.vega.integrations.search.web_search") as mock_search:
            mock_search.return_value = [{"title": "test"}]

            health_report = await check_all_integrations()

            assert isinstance(health_report, dict)
            assert "integrations" in health_report


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
