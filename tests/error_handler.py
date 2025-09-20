#!/usr/bin/env python3
"""
error_handler.py - Error handling utilities for tests

Provides error handling classes and utilities used by test modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from contextlib import contextmanager
import traceback
import logging
import time


class ErrorCode(Enum):
    """Error codes for test scenarios"""

    UNKNOWN_ERROR = "UNKNOWN_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling"""

    error_code: ErrorCode
    severity: ErrorSeverity
    message: str
    details: Dict[str, Any]
    timestamp: float
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None


class ErrorHandler:
    """Mock error handler for testing"""

    def __init__(self):
        self.errors: List[ErrorContext] = []
        self.callbacks: Dict[ErrorCode, List[Callable]] = {}
        self.logger = logging.getLogger("test_error_handler")

    def handle_error(self, context: ErrorContext) -> None:
        """Handle an error with context"""
        self.errors.append(context)

        # Log the error
        self._log_error(context)

        # Execute callbacks
        self._execute_callbacks(context)

    def _log_error(self, context: ErrorContext) -> None:
        """Log error based on severity"""
        message = f"[{context.error_code.value}] {context.message}"

        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message, extra={"context": context.details})
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(message, extra={"context": context.details})
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message, extra={"context": context.details})
        else:
            self.logger.info(message, extra={"context": context.details})

    def _execute_callbacks(self, context: ErrorContext) -> None:
        """Execute registered callbacks for error code"""
        callbacks = self.callbacks.get(context.error_code, [])
        for callback in callbacks:
            try:
                callback(context)
            except Exception as e:
                self.logger.error(f"Error in callback: {e}")

    def register_callback(self, error_code: ErrorCode, callback: Callable) -> None:
        """Register callback for specific error code"""
        if error_code not in self.callbacks:
            self.callbacks[error_code] = []
        self.callbacks[error_code].append(callback)

    def get_errors(self, error_code: Optional[ErrorCode] = None) -> List[ErrorContext]:
        """Get all errors or errors of specific type"""
        if error_code is None:
            return self.errors.copy()
        return [error for error in self.errors if error.error_code == error_code]

    def clear_errors(self) -> None:
        """Clear all recorded errors"""
        self.errors.clear()

    def get_error_count(self, severity: Optional[ErrorSeverity] = None) -> int:
        """Get count of errors by severity"""
        if severity is None:
            return len(self.errors)
        return len([error for error in self.errors if error.severity == severity])


# Global error handler instance
_error_handler = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def log_error(
    error_code: ErrorCode,
    message: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Log an error using global handler"""
    context = ErrorContext(
        error_code=error_code,
        severity=severity,
        message=message,
        details=details or {},
        timestamp=time.time(),
        stack_trace=traceback.format_exc(),
        correlation_id=correlation_id,
    )

    handler = get_error_handler()
    handler.handle_error(context)


def log_info(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log an info message"""
    log_error(ErrorCode.UNKNOWN_ERROR, message, ErrorSeverity.LOW, details)


def log_warning(message: str, details: Optional[Dict[str, Any]] = None) -> None:
    """Log a warning message"""
    log_error(ErrorCode.UNKNOWN_ERROR, message, ErrorSeverity.MEDIUM, details)


@contextmanager
def error_context(
    error_code: ErrorCode,
    message: str,
    severity: ErrorSeverity = ErrorSeverity.HIGH,
    reraise: bool = True,
):
    """Context manager for error handling"""
    try:
        yield
    except Exception as e:
        log_error(
            error_code,
            f"{message}: {str(e)}",
            severity,
            {"exception_type": type(e).__name__},
        )
        if reraise:
            raise


def reset_error_handler() -> None:
    """Reset global error handler"""
    global _error_handler
    if _error_handler is not None:
        _error_handler.clear_errors()


# Mock error simulation utilities
class MockErrorSimulator:
    """Simulate various error conditions for testing"""

    def __init__(self):
        self.error_rates: Dict[ErrorCode, float] = {}
        self.call_counts: Dict[str, int] = {}

    def set_error_rate(self, error_code: ErrorCode, rate: float) -> None:
        """Set error rate for specific error code (0.0 to 1.0)"""
        self.error_rates[error_code] = max(0.0, min(1.0, rate))

    def should_fail(self, operation: str, error_code: ErrorCode) -> bool:
        """Check if operation should fail based on error rate"""
        import random

        self.call_counts[operation] = self.call_counts.get(operation, 0) + 1
        rate = self.error_rates.get(error_code, 0.0)

        return random.random() < rate

    def simulate_network_error(self) -> None:
        """Simulate network error"""
        if self.should_fail("network", ErrorCode.NETWORK_ERROR):
            raise ConnectionError("Simulated network error")

    def simulate_timeout_error(self) -> None:
        """Simulate timeout error"""
        if self.should_fail("timeout", ErrorCode.TIMEOUT_ERROR):
            raise TimeoutError("Simulated timeout error")

    def simulate_database_error(self) -> None:
        """Simulate database error"""
        if self.should_fail("database", ErrorCode.DATABASE_ERROR):
            raise RuntimeError("Simulated database error")

    def get_call_count(self, operation: str) -> int:
        """Get call count for operation"""
        return self.call_counts.get(operation, 0)

    def reset_counters(self) -> None:
        """Reset call counters"""
        self.call_counts.clear()


# Global error simulator
_error_simulator = None


def get_error_simulator() -> MockErrorSimulator:
    """Get global error simulator instance"""
    global _error_simulator
    if _error_simulator is None:
        _error_simulator = MockErrorSimulator()
    return _error_simulator


# Exception classes for testing
class TestError(Exception):
    """Base test error class"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR):
        super().__init__(message)
        self.error_code = error_code


class ValidationTestError(TestError):
    """Validation error for testing"""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.VALIDATION_ERROR)


class AuthenticationTestError(TestError):
    """Authentication error for testing"""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.AUTHENTICATION_ERROR)


class NetworkTestError(TestError):
    """Network error for testing"""

    def __init__(self, message: str):
        super().__init__(message, ErrorCode.NETWORK_ERROR)
