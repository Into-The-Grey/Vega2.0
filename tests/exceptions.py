#!/usr/bin/env python3
"""
exceptions.py - Custom exception classes for tests

Provides custom exception classes and utilities used by test modules.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from enum import Enum


class TestErrorCode(Enum):
    """Error codes for test exceptions"""

    UNKNOWN = "UNKNOWN"
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    TIMEOUT = "TIMEOUT"
    CONFIGURATION = "CONFIGURATION"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    RATE_LIMIT = "RATE_LIMIT"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Base exception classes
class VegaTestException(Exception):
    """Base exception for Vega test errors"""

    def __init__(
        self,
        message: str,
        error_code: TestErrorCode = TestErrorCode.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_code": self.error_code.value,
            "message": self.message,
            "details": self.details,
            "exception_type": type(self).__name__,
        }


class VegaTestValidationError(VegaTestException):
    """Validation error for test scenarios"""

    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)

        super().__init__(message, TestErrorCode.VALIDATION, details)


class VegaTestAuthenticationError(VegaTestException):
    """Authentication error for test scenarios"""

    def __init__(self, message: str, user_id: Optional[str] = None):
        details = {}
        if user_id:
            details["user_id"] = user_id

        super().__init__(message, TestErrorCode.AUTHENTICATION, details)


class VegaTestAuthorizationError(VegaTestException):
    """Authorization error for test scenarios"""

    def __init__(self, message: str, required_permission: Optional[str] = None):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(message, TestErrorCode.AUTHORIZATION, details)


class VegaTestNetworkError(VegaTestException):
    """Network error for test scenarios"""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
    ):
        details = {}
        if endpoint:
            details["endpoint"] = endpoint
        if status_code:
            details["status_code"] = status_code

        super().__init__(message, TestErrorCode.NETWORK, details)


class VegaTestDatabaseError(VegaTestException):
    """Database error for test scenarios"""

    def __init__(
        self, message: str, query: Optional[str] = None, table: Optional[str] = None
    ):
        details = {}
        if query:
            details["query"] = query
        if table:
            details["table"] = table

        super().__init__(message, TestErrorCode.DATABASE, details)


class VegaTestTimeoutError(VegaTestException):
    """Timeout error for test scenarios"""

    def __init__(
        self,
        message: str,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ):
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation

        super().__init__(message, TestErrorCode.TIMEOUT, details)


class VegaTestConfigurationError(VegaTestException):
    """Configuration error for test scenarios"""

    def __init__(
        self, message: str, config_key: Optional[str] = None, config_value: Any = None
    ):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)

        super().__init__(message, TestErrorCode.CONFIGURATION, details)


class VegaTestExternalServiceError(VegaTestException):
    """External service error for test scenarios"""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        service_url: Optional[str] = None,
    ):
        details = {}
        if service_name:
            details["service_name"] = service_name
        if service_url:
            details["service_url"] = service_url

        super().__init__(message, TestErrorCode.EXTERNAL_SERVICE, details)


class VegaTestResourceNotFoundError(VegaTestException):
    """Resource not found error for test scenarios"""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(message, TestErrorCode.RESOURCE_NOT_FOUND, details)


class VegaTestResourceConflictError(VegaTestException):
    """Resource conflict error for test scenarios"""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        conflicting_id: Optional[str] = None,
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if conflicting_id:
            details["conflicting_id"] = conflicting_id

        super().__init__(message, TestErrorCode.RESOURCE_CONFLICT, details)


class VegaTestRateLimitError(VegaTestException):
    """Rate limit error for test scenarios"""

    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ):
        details = {}
        if limit:
            details["limit"] = limit
        if window_seconds:
            details["window_seconds"] = window_seconds

        super().__init__(message, TestErrorCode.RATE_LIMIT, details)


class VegaTestInternalError(VegaTestException):
    """Internal error for test scenarios"""

    def __init__(self, message: str, component: Optional[str] = None):
        details = {}
        if component:
            details["component"] = component

        super().__init__(message, TestErrorCode.INTERNAL_ERROR, details)


# Exception handler utilities
class TestExceptionHandler:
    """Exception handler for test scenarios"""

    def __init__(self):
        self.handled_exceptions: List[VegaTestException] = []
        self.exception_callbacks: Dict[TestErrorCode, List[callable]] = {}

    def handle_exception(self, exception: VegaTestException) -> None:
        """Handle and record exception"""
        self.handled_exceptions.append(exception)

        # Execute callbacks for this error code
        callbacks = self.exception_callbacks.get(exception.error_code, [])
        for callback in callbacks:
            try:
                callback(exception)
            except Exception:
                # Ignore callback errors in tests
                pass

    def register_callback(self, error_code: TestErrorCode, callback: callable) -> None:
        """Register callback for specific error code"""
        if error_code not in self.exception_callbacks:
            self.exception_callbacks[error_code] = []
        self.exception_callbacks[error_code].append(callback)

    def get_exceptions(
        self, error_code: Optional[TestErrorCode] = None
    ) -> List[VegaTestException]:
        """Get handled exceptions by error code"""
        if error_code is None:
            return self.handled_exceptions.copy()

        return [exc for exc in self.handled_exceptions if exc.error_code == error_code]

    def clear_exceptions(self) -> None:
        """Clear handled exceptions"""
        self.handled_exceptions.clear()

    def get_exception_count(self, error_code: Optional[TestErrorCode] = None) -> int:
        """Get count of exceptions by error code"""
        if error_code is None:
            return len(self.handled_exceptions)

        return len(
            [exc for exc in self.handled_exceptions if exc.error_code == error_code]
        )


# Mock exception simulator
class TestExceptionSimulator:
    """Simulate exceptions for testing error handling"""

    def __init__(self):
        self.exception_rates: Dict[TestErrorCode, float] = {}
        self.call_counts: Dict[str, int] = {}

    def set_exception_rate(self, error_code: TestErrorCode, rate: float) -> None:
        """Set exception rate for specific error code (0.0 to 1.0)"""
        self.exception_rates[error_code] = max(0.0, min(1.0, rate))

    def should_raise_exception(self, operation: str, error_code: TestErrorCode) -> bool:
        """Check if operation should raise exception based on rate"""
        import random

        self.call_counts[operation] = self.call_counts.get(operation, 0) + 1
        rate = self.exception_rates.get(error_code, 0.0)

        return random.random() < rate

    def maybe_raise_validation_error(self, operation: str = "validation") -> None:
        """Maybe raise validation error"""
        if self.should_raise_exception(operation, TestErrorCode.VALIDATION):
            raise VegaTestValidationError(f"Simulated validation error in {operation}")

    def maybe_raise_network_error(self, operation: str = "network") -> None:
        """Maybe raise network error"""
        if self.should_raise_exception(operation, TestErrorCode.NETWORK):
            raise VegaTestNetworkError(f"Simulated network error in {operation}")

    def maybe_raise_timeout_error(self, operation: str = "timeout") -> None:
        """Maybe raise timeout error"""
        if self.should_raise_exception(operation, TestErrorCode.TIMEOUT):
            raise VegaTestTimeoutError(f"Simulated timeout error in {operation}")

    def maybe_raise_database_error(self, operation: str = "database") -> None:
        """Maybe raise database error"""
        if self.should_raise_exception(operation, TestErrorCode.DATABASE):
            raise VegaTestDatabaseError(f"Simulated database error in {operation}")

    def get_call_count(self, operation: str) -> int:
        """Get call count for operation"""
        return self.call_counts.get(operation, 0)

    def reset_counters(self) -> None:
        """Reset call counters"""
        self.call_counts.clear()


# Global instances
_exception_handler = None
_exception_simulator = None


def get_exception_handler() -> TestExceptionHandler:
    """Get global exception handler instance"""
    global _exception_handler
    if _exception_handler is None:
        _exception_handler = TestExceptionHandler()
    return _exception_handler


def get_exception_simulator() -> TestExceptionSimulator:
    """Get global exception simulator instance"""
    global _exception_simulator
    if _exception_simulator is None:
        _exception_simulator = TestExceptionSimulator()
    return _exception_simulator


def reset_exception_handler() -> None:
    """Reset global exception handler"""
    global _exception_handler
    if _exception_handler is not None:
        _exception_handler.clear_exceptions()


def reset_exception_simulator() -> None:
    """Reset global exception simulator"""
    global _exception_simulator
    if _exception_simulator is not None:
        _exception_simulator.reset_counters()


# Context manager for exception testing
from contextlib import contextmanager


@contextmanager
def expect_exception(exception_type: type, message_pattern: Optional[str] = None):
    """Context manager to expect specific exception"""
    try:
        yield
        raise AssertionError(
            f"Expected {exception_type.__name__} but no exception was raised"
        )
    except exception_type as e:
        if message_pattern and message_pattern not in str(e):
            raise AssertionError(
                f"Exception message '{str(e)}' does not contain '{message_pattern}'"
            )
    except Exception as e:
        raise AssertionError(
            f"Expected {exception_type.__name__} but got {type(e).__name__}: {str(e)}"
        )


# Convenience functions for creating common exceptions
def validation_error(
    message: str, field: Optional[str] = None
) -> VegaTestValidationError:
    """Create validation error"""
    return VegaTestValidationError(message, field)


def authentication_error(
    message: str, user_id: Optional[str] = None
) -> VegaTestAuthenticationError:
    """Create authentication error"""
    return VegaTestAuthenticationError(message, user_id)


def authorization_error(
    message: str, permission: Optional[str] = None
) -> VegaTestAuthorizationError:
    """Create authorization error"""
    return VegaTestAuthorizationError(message, permission)


def network_error(message: str, endpoint: Optional[str] = None) -> VegaTestNetworkError:
    """Create network error"""
    return VegaTestNetworkError(message, endpoint)


def database_error(message: str, table: Optional[str] = None) -> VegaTestDatabaseError:
    """Create database error"""
    return VegaTestDatabaseError(message, table=table)


def timeout_error(
    message: str, operation: Optional[str] = None
) -> VegaTestTimeoutError:
    """Create timeout error"""
    return VegaTestTimeoutError(message, operation=operation)


def not_found_error(
    message: str, resource_type: Optional[str] = None
) -> VegaTestResourceNotFoundError:
    """Create resource not found error"""
    return VegaTestResourceNotFoundError(message, resource_type)
