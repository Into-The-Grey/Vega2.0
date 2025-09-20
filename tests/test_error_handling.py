#!/usr/bin/env python3
"""
Error Handling System Test Script
===============================

Test script to validate the error handling and logging system.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.error_handler import (
    get_error_handler,
    ErrorCode,
    ErrorSeverity,
    ErrorContext,
    log_info,
    log_error,
    log_warning,
    error_context,
)
from tests.exceptions import *


async def test_basic_error_handling():
    """Test basic error creation and handling"""
    print("🧪 Testing basic error handling...")

    handler = get_error_handler()

    # Test error creation
    error = handler.create_error(
        code=ErrorCode.INVALID_INPUT,
        message="Test validation error",
        user_message="Please check your input",
        severity=ErrorSeverity.LOW,
    )

    print(f"✅ Created error: {error.error_id}")
    print(f"   Code: {error.code.value}")
    print(f"   Category: {error.category.value}")
    print(f"   Severity: {error.severity.value}")
    print(f"   User Message: {error.user_message}")

    # Test error handling
    handled_error = handler.handle_error(error)
    print(f"✅ Handled error: {handled_error.error_id}")


def test_custom_exceptions():
    """Test custom exception hierarchy"""
    print("\n🧪 Testing custom exceptions...")

    try:
        raise InvalidInputError("test_field", "invalid_value")
    except VegaException as e:
        print(f"✅ Caught InvalidInputError: {e.code.value}")
        print(f"   Message: {e.message}")
        print(f"   Context: {e.context}")

    try:
        raise LLMRateLimitError("openai", retry_after=30)
    except VegaException as e:
        print(f"✅ Caught LLMRateLimitError: {e.code.value}")
        print(f"   Retry after: {e.retry_after}s")

    try:
        raise ProcessCrashError("test_process", exit_code=1)
    except VegaException as e:
        print(f"✅ Caught ProcessCrashError: {e.code.value}")
        print(f"   Context: {e.context}")


async def test_error_context():
    """Test error context manager"""
    print("\n🧪 Testing error context manager...")

    try:
        async with error_context(
            component="test_component",
            operation="test_operation",
            user_id="test_user",
            session_id="test_session",
        ) as ctx:
            print(f"✅ Created context: {ctx.component}/{ctx.operation}")
            # Simulate an error
            raise ValueError("Test context error")
    except Exception as e:
        print(f"✅ Caught context error: {type(e).__name__}")


def test_structured_logging():
    """Test structured logging"""
    print("\n🧪 Testing structured logging...")

    log_info("Test info message", component="test", action="testing")
    log_warning("Test warning message", level="medium", issue="none")
    log_error("Test error message", error_type="test", severity="low")

    print("✅ Logged structured messages (check log files)")


def test_error_to_http():
    """Test error to HTTP exception conversion"""
    print("\n🧪 Testing HTTP exception conversion...")

    handler = get_error_handler()

    # Test different error categories
    errors = [
        handler.create_error(ErrorCode.INVALID_API_KEY, "Invalid key"),
        handler.create_error(ErrorCode.INVALID_INPUT, "Bad input"),
        handler.create_error(ErrorCode.LLM_PROVIDER_ERROR, "LLM failed"),
        handler.create_error(ErrorCode.DATABASE_CONNECTION, "DB down"),
        handler.create_error(ErrorCode.INTERNAL_ERROR, "Internal error"),
    ]

    for error in errors:
        http_exc = error.to_http_exception()
        print(f"✅ {error.code.value} -> HTTP {http_exc.status_code}")
        print(f"   Detail: {http_exc.detail}")


def test_error_stats():
    """Test error statistics tracking"""
    print("\n🧪 Testing error statistics...")

    handler = get_error_handler()

    # Generate some test errors
    for i in range(5):
        error = handler.create_error(
            ErrorCode.INVALID_INPUT, f"Test error {i}", severity=ErrorSeverity.LOW
        )
        handler.handle_error(error)

    # Generate different error types
    error = handler.create_error(
        ErrorCode.LLM_TIMEOUT, "LLM timeout test", severity=ErrorSeverity.MEDIUM
    )
    handler.handle_error(error)

    # Get statistics
    stats = handler.get_error_stats()
    print(f"✅ Error stats:")
    print(f"   Total errors: {stats['total_errors']}")
    print(f"   Breakdown: {stats['error_breakdown']}")


async def main():
    """Run all tests"""
    print("🚀 Starting Vega2.0 Error Handling System Tests\n")

    try:
        await test_basic_error_handling()
        test_custom_exceptions()
        await test_error_context()
        test_structured_logging()
        test_error_to_http()
        test_error_stats()

        print("\n✅ All error handling tests completed successfully!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
