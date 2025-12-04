#!/usr/bin/env python3
"""
Tests for Vega2.0 Hardened Core System
======================================

These tests ensure that the hardened core system:
1. Never crashes under any circumstances
2. Handles all edge cases gracefully
3. Learns from failures properly
4. Reports issues correctly
5. Provides useful suggestions
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Set test mode
os.environ["VEGA_TEST_MODE"] = "1"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vega.core.hardened_core import (
    # Enums
    Severity,
    OperationStatus,
    # Data classes
    OperationResult,
    IssueRecord,
    ImprovementSuggestion,
    # Decorators
    hardened,
    hardened_sync,
    # Safe execution
    safe_execute,
    safe_execute_sync,
    # Validation
    validate_input,
    sanitize_string,
    # Knowledge base
    KnowledgeBase,
    get_knowledge_base,
    # Reporting
    IssueReporter,
    get_issue_reporter,
    # Constants
    DEFAULT_TIMEOUT,
    QUICK_TIMEOUT,
    LONG_TIMEOUT,
)


# =============================================================================
# OperationResult Tests
# =============================================================================


class TestOperationResult:
    """Tests for OperationResult data class"""

    def test_ok_result(self):
        """Test creating a successful result"""
        result = OperationResult.ok("test_value", duration_ms=100.0)

        assert result.success is True
        assert result.value == "test_value"
        assert result.error is None
        assert result.duration_ms == 100.0
        assert result.status == OperationStatus.SUCCESS

    def test_fail_result(self):
        """Test creating a failed result"""
        result = OperationResult.fail("Something went wrong", "TestError", suggestions=["Try again later"])

        assert result.success is False
        assert result.value is None
        assert result.error == "Something went wrong"
        assert result.error_type == "TestError"
        assert result.suggestions == ["Try again later"]

    def test_partial_result(self):
        """Test creating a partial/degraded result"""
        result = OperationResult.partial("fallback_value", "Original operation failed, using fallback")

        assert result.success is True
        assert result.value == "fallback_value"
        assert result.degraded is True
        assert result.status == OperationStatus.PARTIAL


# =============================================================================
# @hardened Decorator Tests
# =============================================================================


class TestHardenedDecorator:
    """Tests for the @hardened async decorator"""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test normal successful execution"""

        @hardened(timeout=5.0)
        async def successful_func():
            return "success"

        result = await successful_func()

        assert result.success is True
        assert result.value == "success"
        assert result.retries == 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test that timeouts are handled gracefully"""

        @hardened(timeout=0.1, retries=0, default="timeout_default")
        async def slow_func():
            await asyncio.sleep(5)
            return "should_not_reach"

        result = await slow_func()

        assert result.success is True  # Because we have a default
        assert result.value == "timeout_default"
        assert result.degraded is True

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that retries work correctly"""
        attempt_count = 0

        @hardened(timeout=5.0, retries=3)
        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Not yet")
            return "finally_worked"

        result = await flaky_func()

        assert result.success is True
        assert result.value == "finally_worked"
        assert result.retries == 2  # Succeeded on 3rd attempt

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test that fallback function is called on failure"""

        async def fallback_func():
            return "fallback_value"

        @hardened(timeout=5.0, retries=0, fallback=fallback_func)
        async def failing_func():
            raise RuntimeError("Always fails")

        result = await failing_func()

        assert result.success is True
        assert result.value == "fallback_value"
        assert result.degraded is True

    @pytest.mark.asyncio
    async def test_default_on_failure(self):
        """Test that default value is returned on failure"""

        @hardened(timeout=5.0, retries=0, default="default_value")
        async def failing_func():
            raise RuntimeError("Always fails")

        result = await failing_func()

        assert result.success is True
        assert result.value == "default_value"
        assert result.degraded is True

    @pytest.mark.asyncio
    async def test_complete_failure(self):
        """Test behavior when all options are exhausted"""

        @hardened(timeout=5.0, retries=1)
        async def always_fails():
            raise RuntimeError("Permanent failure")

        result = await always_fails()

        assert result.success is False
        assert result.error_type == "RuntimeError"
        assert result.error is not None and "Permanent failure" in result.error

    @pytest.mark.asyncio
    async def test_cancellation_handling(self):
        """Test that cancellation is handled gracefully"""

        @hardened(timeout=10.0)
        async def cancellable_func():
            await asyncio.sleep(10)
            return "should_not_reach"

        task = asyncio.create_task(cancellable_func())
        await asyncio.sleep(0.01)
        task.cancel()

        result = await task

        assert result.success is False
        assert result.status == OperationStatus.CANCELLED


# =============================================================================
# @hardened_sync Decorator Tests
# =============================================================================


class TestHardenedSyncDecorator:
    """Tests for the @hardened_sync decorator"""

    def test_successful_execution(self):
        """Test normal successful execution"""

        @hardened_sync()
        def successful_func():
            return "success"

        result = successful_func()

        assert result.success is True
        assert result.value == "success"

    def test_retry_on_failure(self):
        """Test that retries work correctly"""
        attempt_count = 0

        @hardened_sync(retries=3)
        def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Not yet")
            return "worked"

        result = flaky_func()

        assert result.success is True
        assert result.value == "worked"

    def test_default_on_failure(self):
        """Test default value on failure"""

        @hardened_sync(retries=0, default="default")
        def failing_func():
            raise RuntimeError("Fails")

        result = failing_func()

        assert result.success is True
        assert result.value == "default"
        assert result.degraded is True


# =============================================================================
# Safe Execution Tests
# =============================================================================


class TestSafeExecution:
    """Tests for safe_execute functions"""

    @pytest.mark.asyncio
    async def test_safe_execute_success(self):
        """Test safe_execute with successful coroutine"""

        async def good_coro():
            return "result"

        result = await safe_execute(good_coro(), "default", "test_op")

        assert result == "result"

    @pytest.mark.asyncio
    async def test_safe_execute_timeout(self):
        """Test safe_execute with timeout"""

        async def slow_coro():
            await asyncio.sleep(10)
            return "too_late"

        result = await safe_execute(slow_coro(), "default", "test_op", timeout=0.1)

        assert result == "default"

    @pytest.mark.asyncio
    async def test_safe_execute_exception(self):
        """Test safe_execute with exception"""

        async def bad_coro():
            raise ValueError("Error!")

        result = await safe_execute(bad_coro(), "default", "test_op")

        assert result == "default"

    def test_safe_execute_sync_success(self):
        """Test safe_execute_sync with success"""

        def good_func():
            return "result"

        result = safe_execute_sync(good_func, default="default", operation_name="test")

        assert result == "result"

    def test_safe_execute_sync_exception(self):
        """Test safe_execute_sync with exception"""

        def bad_func():
            raise ValueError("Error!")

        result = safe_execute_sync(bad_func, default="default", operation_name="test")

        assert result == "default"


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation functions"""

    def test_validate_correct_type(self):
        """Test validation with correct type"""
        result = validate_input("hello", str, "test_input")

        assert result.success is True
        assert result.value == "hello"

    def test_validate_type_conversion(self):
        """Test automatic type conversion"""
        result = validate_input(123, str, "test_input")

        assert result.success is True
        assert result.value == "123"

    def test_validate_none_allowed(self):
        """Test None handling when allowed"""
        result = validate_input(None, str, "test_input", allow_none=True)

        assert result.success is True
        assert result.value is None

    def test_validate_none_not_allowed(self):
        """Test None handling when not allowed"""
        result = validate_input(None, str, "test_input", allow_none=False)

        assert result.success is False
        assert result.error is not None and "cannot be None" in result.error

    def test_validate_max_length(self):
        """Test string max length enforcement"""
        result = validate_input("hello world", str, "test_input", max_length=5)

        assert result.success is True
        assert result.value == "hello"  # Truncated

    def test_validate_numeric_range(self):
        """Test numeric range validation"""
        result = validate_input(5, int, "test_input", min_value=1, max_value=10)
        assert result.success is True

        result = validate_input(0, int, "test_input", min_value=1, max_value=10)
        assert result.success is False
        assert result.error is not None and "must be >=" in result.error

    def test_sanitize_string(self):
        """Test string sanitization"""
        result = sanitize_string("  hello\x00world  ")

        assert result == "helloworld"
        assert "\x00" not in result

    def test_sanitize_string_truncation(self):
        """Test string truncation"""
        long_string = "a" * 1000
        result = sanitize_string(long_string, max_length=100)

        assert len(result) == 100


# =============================================================================
# Knowledge Base Tests
# =============================================================================


class TestKnowledgeBase:
    """Tests for the KnowledgeBase class"""

    def test_knowledge_base_creation(self):
        """Test creating a knowledge base"""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(Path(tmpdir) / "test_kb.json")

            assert kb is not None
            assert len(kb.error_resolutions) == 0

    def test_record_error(self):
        """Test recording an error"""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(Path(tmpdir) / "test_kb.json")

            error = ValueError("Test error")
            signature = kb.record_error("test_op", error, {"key": "value"})

            assert signature is not None
            assert len(signature) == 12  # MD5 hash prefix
            assert kb.error_frequency[signature] == 1

    def test_error_signature_consistency(self):
        """Test that same errors get same signature"""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(Path(tmpdir) / "test_kb.json")

            error1 = ValueError("Connection refused")
            error2 = ValueError("Connection refused")

            sig1 = kb.get_error_signature(type(error1).__name__, str(error1))
            sig2 = kb.get_error_signature(type(error2).__name__, str(error2))

            assert sig1 == sig2

    def test_record_resolution(self):
        """Test recording a resolution strategy"""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(Path(tmpdir) / "test_kb.json")

            error = ValueError("Test error")
            signature = kb.record_error("test_op", error, {})

            kb.record_resolution(signature, "retry", {}, success=True)

            assert kb.error_resolutions[signature]["total_attempts"] == 1
            assert kb.error_resolutions[signature]["strategies"]["retry"]["successes"] == 1

    def test_get_best_resolution(self):
        """Test getting the best resolution strategy"""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(Path(tmpdir) / "test_kb.json")

            error = ValueError("Test error")
            signature = kb.record_error("test_op", error, {})

            # Record multiple resolutions
            kb.record_resolution(signature, "retry", {}, success=False)
            kb.record_resolution(signature, "retry", {}, success=False)
            kb.record_resolution(signature, "restart", {}, success=True)
            kb.record_resolution(signature, "restart", {}, success=True)

            best = kb.get_best_resolution(signature)

            assert best == "restart"  # 100% success rate vs 0%

    def test_persistence(self):
        """Test that knowledge base persists to disk"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_kb.json"

            # Create and populate
            kb1 = KnowledgeBase(path)
            error = ValueError("Persistent error")
            signature = kb1.record_error("test_op", error, {})
            kb1.record_resolution(signature, "restart", {}, success=True)

            # Reload
            kb2 = KnowledgeBase(path)

            assert signature in kb2.error_resolutions
            assert kb2.error_resolutions[signature]["strategies"]["restart"]["successes"] == 1


# =============================================================================
# Issue Reporter Tests
# =============================================================================


class TestIssueReporter:
    """Tests for the IssueReporter class"""

    def test_queue_report(self):
        """Test queueing an issue for reporting"""
        reporter = IssueReporter()

        issue = IssueRecord(
            id="test123",
            timestamp=datetime.utcnow(),
            operation="test_op",
            error_type="TestError",
            error_message="Test message",
            stack_trace="",
            context={},
            severity=Severity.WARNING,
        )

        reporter.queue_report(issue)

        assert len(reporter._pending_reports) == 1

    def test_critical_issue_triggers_report(self):
        """Test that critical issues trigger immediate report"""
        reporter = IssueReporter()

        issue = IssueRecord(
            id="crit123",
            timestamp=datetime.utcnow(),
            operation="test_op",
            error_type="CriticalError",
            error_message="Critical failure",
            stack_trace="",
            context={},
            severity=Severity.CRITICAL,
        )

        with patch.object(reporter, "_send_report") as mock_send:
            reporter.queue_report(issue)
            mock_send.assert_called_once()

    def test_get_summary(self):
        """Test getting issue summary"""
        reporter = IssueReporter()

        summary = reporter.get_summary()

        assert "pending_reports" in summary
        assert "unresolved_issues" in summary
        assert "improvement_suggestions" in summary


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs"""

    @pytest.mark.asyncio
    async def test_hardened_with_none_return(self):
        """Test hardened function returning None"""

        @hardened()
        async def returns_none():
            return None

        result = await returns_none()

        assert result.success is True
        assert result.value is None

    @pytest.mark.asyncio
    async def test_hardened_with_empty_string(self):
        """Test hardened function returning empty string"""

        @hardened()
        async def returns_empty():
            return ""

        result = await returns_empty()

        assert result.success is True
        assert result.value == ""

    @pytest.mark.asyncio
    async def test_hardened_with_complex_return(self):
        """Test hardened function returning complex object"""

        @hardened()
        async def returns_complex():
            return {"list": [1, 2, 3], "nested": {"key": "value"}}

        result = await returns_complex()

        assert result.success is True
        assert result.value["list"] == [1, 2, 3]

    def test_validate_input_with_special_characters(self):
        """Test validation with special characters"""
        result = validate_input("hello\n\t\r world", str, "test")

        assert result.success is True

    def test_sanitize_with_unicode(self):
        """Test sanitization with unicode"""
        result = sanitize_string("Hello 世界 🌍")

        assert result == "Hello 世界 🌍"

    def test_sanitize_with_very_long_string(self):
        """Test sanitization with very long string"""
        long_string = "x" * 1000000  # 1 million chars
        result = sanitize_string(long_string, max_length=10000)

        assert len(result) == 10000

    @pytest.mark.asyncio
    async def test_nested_hardened_calls(self):
        """Test nested hardened function calls"""

        @hardened()
        async def inner():
            return "inner"

        @hardened()
        async def outer():
            inner_result = await inner()
            return f"outer-{inner_result.value}"

        result = await outer()

        assert result.success is True
        assert result.value == "outer-inner"


# =============================================================================
# Stress Tests
# =============================================================================


class TestStress:
    """Stress tests for the hardened core"""

    @pytest.mark.asyncio
    async def test_many_concurrent_operations(self):
        """Test many concurrent hardened operations"""

        @hardened(timeout=5.0)
        async def quick_op(n: int):
            await asyncio.sleep(0.01)
            return n * 2

        # Run 100 concurrent operations
        tasks = [quick_op(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 100
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """Test mix of successful and failing operations"""

        @hardened(timeout=5.0, default=-1)
        async def mixed_op(n: int):
            if n % 3 == 0:
                raise ValueError(f"Failed on {n}")
            return n

        results = await asyncio.gather(*[mixed_op(i) for i in range(30)])

        successes = [r for r in results if not r.degraded]
        degraded = [r for r in results if r.degraded]

        # 10 out of 30 should fail (0, 3, 6, 9, 12, 15, 18, 21, 24, 27)
        assert len(degraded) == 10
        assert all(r.value == -1 for r in degraded)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components"""

    @pytest.mark.asyncio
    async def test_full_failure_cycle(self):
        """Test complete failure-recovery-learning cycle"""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(Path(tmpdir) / "test_kb.json")

            # Create a failing function that eventually succeeds
            call_count = 0

            @hardened(timeout=5.0, retries=2, operation_name="test_cycle")
            async def eventually_succeeds():
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Network unavailable")
                return "connected"

            # First call - will retry and succeed
            result1 = await eventually_succeeds()
            assert result1.success is True

            # Record that retry worked
            kb.record_resolution("network_sig", "retry", {}, True)

            # Verify knowledge was gained
            best = kb.get_best_resolution("network_sig")
            assert best == "retry"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
