"""
Vega2.0 Custom Exceptions
========================

Custom exception hierarchy for better error handling and categorization.
"""

from typing import Optional, Dict, Any
from .error_handler import ErrorCode, ErrorSeverity, ErrorCategory


class VegaException(Exception):
    """Base exception for all Vega-specific errors"""

    def __init__(
        self,
        message: str,
        code: ErrorCode,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.severity = severity
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.context = context or {}


# Authentication & Authorization Exceptions
class AuthenticationError(VegaException):
    """Authentication failed"""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.INVALID_API_KEY),
            severity=ErrorSeverity.MEDIUM,
            recoverable=False,
            **kwargs,
        )


class AuthorizationError(VegaException):
    """Insufficient permissions"""

    def __init__(self, message: str = "Insufficient permissions", **kwargs):
        super().__init__(
            message,
            code=ErrorCode.INSUFFICIENT_PERMISSIONS,
            severity=ErrorSeverity.MEDIUM,
            recoverable=False,
            **kwargs,
        )


class InvalidAPIKeyError(AuthenticationError):
    """Invalid API key provided"""

    def __init__(self, **kwargs):
        super().__init__(
            "Invalid API key provided", code=ErrorCode.INVALID_API_KEY, **kwargs
        )


class MissingAPIKeyError(AuthenticationError):
    """API key not provided"""

    def __init__(self, **kwargs):
        super().__init__(
            "API key is required", code=ErrorCode.MISSING_API_KEY, **kwargs
        )


# Validation Exceptions
class ValidationError(VegaException):
    """Input validation failed"""

    def __init__(self, message: str = "Validation failed", **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.INVALID_INPUT),
            severity=ErrorSeverity.LOW,
            recoverable=True,
            **kwargs,
        )


class InvalidInputError(ValidationError):
    """Invalid input provided"""

    def __init__(self, field: str, value: Any = None, **kwargs):
        message = f"Invalid value for field '{field}'"
        if value is not None:
            message += f": {value}"

        super().__init__(
            message,
            code=ErrorCode.INVALID_INPUT,
            context={"field": field, "value": value},
            **kwargs,
        )


class MissingParameterError(ValidationError):
    """Required parameter missing"""

    def __init__(self, parameter: str, **kwargs):
        super().__init__(
            f"Required parameter '{parameter}' is missing",
            code=ErrorCode.MISSING_PARAMETER,
            context={"parameter": parameter},
            **kwargs,
        )


class OutOfRangeError(ValidationError):
    """Value out of acceptable range"""

    def __init__(
        self, field: str, value: Any, min_val: Any = None, max_val: Any = None, **kwargs
    ):
        message = f"Value for '{field}' is out of range: {value}"
        if min_val is not None and max_val is not None:
            message += f" (expected: {min_val}-{max_val})"

        super().__init__(
            message,
            code=ErrorCode.OUT_OF_RANGE,
            context={"field": field, "value": value, "min": min_val, "max": max_val},
            **kwargs,
        )


# LLM Provider Exceptions
class LLMProviderError(VegaException):
    """LLM provider error"""

    def __init__(self, message: str, provider: str, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.LLM_PROVIDER_ERROR),
            severity=ErrorSeverity.HIGH,
            context={"provider": provider},
            **kwargs,
        )


class LLMRateLimitError(LLMProviderError):
    """LLM rate limit exceeded"""

    def __init__(self, provider: str, retry_after: int = 60, **kwargs):
        super().__init__(
            f"Rate limit exceeded for {provider}",
            provider=provider,
            code=ErrorCode.LLM_RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after,
            **kwargs,
        )


class LLMTimeoutError(LLMProviderError):
    """LLM request timeout"""

    def __init__(self, provider: str, timeout: int, **kwargs):
        super().__init__(
            f"Request to {provider} timed out after {timeout}s",
            provider=provider,
            code=ErrorCode.LLM_TIMEOUT,
            context={"timeout": timeout},
            **kwargs,
        )


# Database Exceptions
class DatabaseError(VegaException):
    """Database operation failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.DATABASE_CONNECTION),
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class DatabaseConnectionError(DatabaseError):
    """Database connection failed"""

    def __init__(self, **kwargs):
        super().__init__(
            "Database connection failed", code=ErrorCode.DATABASE_CONNECTION, **kwargs
        )


class DatabaseTimeoutError(DatabaseError):
    """Database operation timed out"""

    def __init__(self, operation: str, timeout: int, **kwargs):
        super().__init__(
            f"Database operation '{operation}' timed out after {timeout}s",
            code=ErrorCode.DATABASE_TIMEOUT,
            context={"operation": operation, "timeout": timeout},
            **kwargs,
        )


# Process Management Exceptions
class ProcessError(VegaException):
    """Process management error"""

    def __init__(self, message: str, process_name: str = None, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.PROCESS_START_FAILED),
            severity=ErrorSeverity.HIGH,
            context={"process_name": process_name},
            **kwargs,
        )


class ProcessStartError(ProcessError):
    """Process failed to start"""

    def __init__(self, process_name: str, reason: str = None, **kwargs):
        message = f"Failed to start process '{process_name}'"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            process_name=process_name,
            code=ErrorCode.PROCESS_START_FAILED,
            context={"reason": reason},
            **kwargs,
        )


class ProcessCrashError(ProcessError):
    """Process crashed unexpectedly"""

    def __init__(self, process_name: str, exit_code: int = None, **kwargs):
        message = f"Process '{process_name}' crashed"
        if exit_code is not None:
            message += f" with exit code {exit_code}"

        super().__init__(
            message,
            process_name=process_name,
            code=ErrorCode.PROCESS_CRASHED,
            context={"exit_code": exit_code},
            **kwargs,
        )


class ResourceExhaustedError(ProcessError):
    """System resources exhausted"""

    def __init__(self, resource: str, current_usage: float = None, **kwargs):
        message = f"Resource '{resource}' exhausted"
        if current_usage is not None:
            message += f" (current usage: {current_usage}%)"

        super().__init__(
            message,
            code=ErrorCode.RESOURCE_EXHAUSTED,
            severity=ErrorSeverity.CRITICAL,
            context={"resource": resource, "usage": current_usage},
            **kwargs,
        )


# Configuration Exceptions
class ConfigurationError(VegaException):
    """Configuration error"""

    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.INVALID_CONFIG),
            severity=ErrorSeverity.HIGH,
            recoverable=False,
            context={"config_key": config_key},
            **kwargs,
        )


class InvalidConfigError(ConfigurationError):
    """Invalid configuration value"""

    def __init__(
        self, config_key: str, value: Any = None, expected: str = None, **kwargs
    ):
        message = f"Invalid configuration for '{config_key}'"
        if value is not None:
            message += f": {value}"
        if expected:
            message += f" (expected: {expected})"

        super().__init__(
            message,
            config_key=config_key,
            code=ErrorCode.INVALID_CONFIG,
            context={"value": value, "expected": expected},
            **kwargs,
        )


class MissingConfigError(ConfigurationError):
    """Required configuration missing"""

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Required configuration '{config_key}' is missing",
            config_key=config_key,
            code=ErrorCode.MISSING_CONFIG,
            **kwargs,
        )


# Network Exceptions
class NetworkError(VegaException):
    """Network operation failed"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.CONNECTION_FAILED),
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class ConnectionFailedError(NetworkError):
    """Network connection failed"""

    def __init__(self, host: str, port: int = None, **kwargs):
        message = f"Connection failed to {host}"
        if port:
            message += f":{port}"

        super().__init__(
            message,
            code=ErrorCode.CONNECTION_FAILED,
            context={"host": host, "port": port},
            **kwargs,
        )


class NetworkTimeoutError(NetworkError):
    """Network request timed out"""

    def __init__(self, url: str, timeout: int, **kwargs):
        super().__init__(
            f"Request to {url} timed out after {timeout}s",
            code=ErrorCode.NETWORK_TIMEOUT,
            context={"url": url, "timeout": timeout},
            **kwargs,
        )


# Integration Exceptions
class IntegrationError(VegaException):
    """External integration failed"""

    def __init__(self, message: str, service: str, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.INTEGRATION_FAILURE),
            severity=ErrorSeverity.MEDIUM,
            context={"service": service},
            **kwargs,
        )


class ServiceUnavailableError(IntegrationError):
    """External service unavailable"""

    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Service '{service}' is unavailable",
            service=service,
            code=ErrorCode.API_UNAVAILABLE,
            **kwargs,
        )


# Circuit Breaker Exception
class CircuitBreakerOpenError(VegaException):
    """Circuit breaker is open"""

    def __init__(self, service: str, retry_after: int = 60, **kwargs):
        super().__init__(
            f"Circuit breaker is open for service '{service}'",
            code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after,
            context={"service": service},
            **kwargs,
        )


# System Exceptions
class SystemError(VegaException):
    """System-level error"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            code=kwargs.pop("code", ErrorCode.INTERNAL_ERROR),
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class DiskFullError(SystemError):
    """Disk space exhausted"""

    def __init__(self, path: str, available: int = None, **kwargs):
        message = f"Disk full at '{path}'"
        if available is not None:
            message += f" ({available} bytes available)"

        super().__init__(
            message,
            code=ErrorCode.DISK_FULL,
            severity=ErrorSeverity.CRITICAL,
            context={"path": path, "available": available},
            **kwargs,
        )


class MemoryExhaustedError(SystemError):
    """System memory exhausted"""

    def __init__(self, current_usage: float = None, **kwargs):
        message = "System memory exhausted"
        if current_usage is not None:
            message += f" (current usage: {current_usage}%)"

        super().__init__(
            message,
            code=ErrorCode.MEMORY_EXHAUSTED,
            severity=ErrorSeverity.CRITICAL,
            context={"usage": current_usage},
            **kwargs,
        )


# Convenience functions for common errors
def raise_authentication_error(message: str = None):
    """Raise authentication error"""
    raise AuthenticationError(message or "Authentication failed")


def raise_validation_error(field: str, value: Any = None):
    """Raise validation error for specific field"""
    raise InvalidInputError(field, value)


def raise_missing_parameter(parameter: str):
    """Raise missing parameter error"""
    raise MissingParameterError(parameter)


def raise_llm_error(provider: str, message: str):
    """Raise LLM provider error"""
    raise LLMProviderError(message, provider)


def raise_process_error(process_name: str, message: str):
    """Raise process management error"""
    raise ProcessError(message, process_name)
