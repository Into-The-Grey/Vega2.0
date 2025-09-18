"""
Vega2.0 - Enhanced Error Handling and Logging System
===================================================

Comprehensive error handling with structured logging, error codes,
user-friendly messages, and recovery mechanisms.
"""

import logging
import logging.handlers
import json
import traceback
import sys
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import structlog
from fastapi import HTTPException


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    LLM_PROVIDER = "llm_provider"
    PROCESS_MANAGEMENT = "process_management"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    INTERNAL = "internal"


class ErrorCode(Enum):
    """Standard error codes for the system"""

    # Authentication & Authorization (1000-1099)
    INVALID_API_KEY = "VEGA-1001"
    MISSING_API_KEY = "VEGA-1002"
    EXPIRED_TOKEN = "VEGA-1003"
    INSUFFICIENT_PERMISSIONS = "VEGA-1004"

    # Validation Errors (1100-1199)
    INVALID_INPUT = "VEGA-1101"
    MISSING_PARAMETER = "VEGA-1102"
    INVALID_FORMAT = "VEGA-1103"
    OUT_OF_RANGE = "VEGA-1104"

    # External Service Errors (1200-1299)
    LLM_PROVIDER_ERROR = "VEGA-1201"
    LLM_RATE_LIMIT = "VEGA-1202"
    LLM_TIMEOUT = "VEGA-1203"
    INTEGRATION_FAILURE = "VEGA-1204"
    API_UNAVAILABLE = "VEGA-1205"

    # Database Errors (1300-1399)
    DATABASE_CONNECTION = "VEGA-1301"
    DATABASE_TIMEOUT = "VEGA-1302"
    DATA_CORRUPTION = "VEGA-1303"
    CONSTRAINT_VIOLATION = "VEGA-1304"

    # Process Management (1400-1499)
    PROCESS_START_FAILED = "VEGA-1401"
    PROCESS_CRASHED = "VEGA-1402"
    PROCESS_TIMEOUT = "VEGA-1403"
    RESOURCE_EXHAUSTED = "VEGA-1404"

    # Configuration Errors (1500-1599)
    INVALID_CONFIG = "VEGA-1501"
    MISSING_CONFIG = "VEGA-1502"
    CONFIG_PARSE_ERROR = "VEGA-1503"

    # Network Errors (1600-1699)
    CONNECTION_FAILED = "VEGA-1601"
    NETWORK_TIMEOUT = "VEGA-1602"
    DNS_RESOLUTION = "VEGA-1603"

    # System Errors (1700-1799)
    DISK_FULL = "VEGA-1701"
    MEMORY_EXHAUSTED = "VEGA-1702"
    FILE_NOT_FOUND = "VEGA-1703"
    PERMISSION_DENIED = "VEGA-1704"

    # Internal Errors (1800-1899)
    INTERNAL_ERROR = "VEGA-1801"
    UNEXPECTED_STATE = "VEGA-1802"
    CIRCUIT_BREAKER_OPEN = "VEGA-1803"


@dataclass
class ErrorContext:
    """Context information for errors"""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    stacktrace: Optional[str] = None


@dataclass
class VegaError:
    """Structured error representation"""

    error_id: str
    code: ErrorCode
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    timestamp: datetime
    context: ErrorContext
    recoverable: bool = True
    retry_after: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "error_id": self.error_id,
            "code": self.code.value,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "user_message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
            "context": asdict(self.context),
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
        }

    def to_http_exception(self) -> HTTPException:
        """Convert to FastAPI HTTPException"""
        status_code_map = {
            ErrorCategory.AUTHENTICATION: 401,
            ErrorCategory.AUTHORIZATION: 403,
            ErrorCategory.VALIDATION: 400,
            ErrorCategory.USER_INPUT: 400,
            ErrorCategory.EXTERNAL_SERVICE: 502,
            ErrorCategory.DATABASE: 503,
            ErrorCategory.LLM_PROVIDER: 502,
            ErrorCategory.PROCESS_MANAGEMENT: 503,
            ErrorCategory.CONFIGURATION: 500,
            ErrorCategory.NETWORK: 502,
            ErrorCategory.SYSTEM: 500,
            ErrorCategory.INTERNAL: 500,
        }

        status_code = status_code_map.get(self.category, 500)

        detail = {
            "error_id": self.error_id,
            "code": self.code.value,
            "message": self.user_message,
            "recoverable": self.recoverable,
        }

        if self.retry_after:
            detail["retry_after"] = self.retry_after

        return HTTPException(status_code=status_code, detail=detail)


class VegaErrorHandler:
    """Central error handler with logging and recovery"""

    def __init__(self, log_dir: str = "/tmp/vega_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.error_count = {}
        self.recovery_strategies = {}
        self._setup_logging()
        self._setup_recovery_strategies()

    def _setup_logging(self):
        """Setup structured logging with multiple handlers"""

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Setup main logger
        self.logger = structlog.get_logger("vega.error_handler")

        # Setup file handlers
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)

        debug_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "debug.log", maxBytes=50 * 1024 * 1024, backupCount=3  # 50MB
        )
        debug_handler.setLevel(logging.DEBUG)

        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(error_handler)
        root_logger.addHandler(debug_handler)
        root_logger.setLevel(logging.DEBUG)

    def _setup_recovery_strategies(self):
        """Setup automatic recovery strategies"""
        self.recovery_strategies = {
            ErrorCode.LLM_RATE_LIMIT: self._handle_rate_limit,
            ErrorCode.LLM_TIMEOUT: self._handle_timeout,
            ErrorCode.DATABASE_CONNECTION: self._handle_db_connection,
            ErrorCode.PROCESS_CRASHED: self._handle_process_crash,
            ErrorCode.RESOURCE_EXHAUSTED: self._handle_resource_exhaustion,
            ErrorCode.CIRCUIT_BREAKER_OPEN: self._handle_circuit_breaker,
        }

    def create_error(
        self,
        code: ErrorCode,
        message: str,
        user_message: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        severity: Optional[ErrorSeverity] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None,
    ) -> VegaError:
        """Create a structured error"""

        error_id = str(uuid.uuid4())

        # Auto-detect category from error code
        category = self._get_category_from_code(code)

        # Auto-detect severity if not provided
        if severity is None:
            severity = self._get_severity_from_code(code)

        # Generate user-friendly message if not provided
        if user_message is None:
            user_message = self._generate_user_message(code, message)

        # Create context if not provided
        if context is None:
            context = ErrorContext()

        # Add stack trace to context
        if context.stacktrace is None:
            context.stacktrace = traceback.format_exc()

        error = VegaError(
            error_id=error_id,
            code=code,
            category=category,
            severity=severity,
            message=message,
            user_message=user_message,
            timestamp=datetime.utcnow(),
            context=context,
            recoverable=recoverable,
            retry_after=retry_after,
        )

        return error

    def handle_error(
        self, error: Union[VegaError, Exception], context: Optional[ErrorContext] = None
    ) -> VegaError:
        """Handle an error with logging and recovery attempts"""

        # Convert exception to VegaError if needed
        if isinstance(error, Exception):
            error = self._exception_to_vega_error(error, context)

        # Log the error
        self._log_error(error)

        # Track error count
        self._track_error(error)

        # Attempt recovery
        self._attempt_recovery(error)

        return error

    def _get_category_from_code(self, code: ErrorCode) -> ErrorCategory:
        """Map error code to category"""
        code_value = int(code.value.split("-")[1])

        if 1000 <= code_value < 1100:
            return ErrorCategory.AUTHENTICATION
        elif 1100 <= code_value < 1200:
            return ErrorCategory.VALIDATION
        elif 1200 <= code_value < 1300:
            return ErrorCategory.EXTERNAL_SERVICE
        elif 1300 <= code_value < 1400:
            return ErrorCategory.DATABASE
        elif 1400 <= code_value < 1500:
            return ErrorCategory.PROCESS_MANAGEMENT
        elif 1500 <= code_value < 1600:
            return ErrorCategory.CONFIGURATION
        elif 1600 <= code_value < 1700:
            return ErrorCategory.NETWORK
        elif 1700 <= code_value < 1800:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.INTERNAL

    def _get_severity_from_code(self, code: ErrorCode) -> ErrorSeverity:
        """Auto-detect severity based on error code"""
        critical_codes = [
            ErrorCode.DATA_CORRUPTION,
            ErrorCode.MEMORY_EXHAUSTED,
            ErrorCode.DISK_FULL,
        ]

        high_codes = [
            ErrorCode.DATABASE_CONNECTION,
            ErrorCode.PROCESS_CRASHED,
            ErrorCode.RESOURCE_EXHAUSTED,
        ]

        low_codes = [
            ErrorCode.INVALID_INPUT,
            ErrorCode.MISSING_PARAMETER,
            ErrorCode.LLM_RATE_LIMIT,
        ]

        if code in critical_codes:
            return ErrorSeverity.CRITICAL
        elif code in high_codes:
            return ErrorSeverity.HIGH
        elif code in low_codes:
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM

    def _generate_user_message(self, code: ErrorCode, technical_message: str) -> str:
        """Generate user-friendly error messages"""
        user_messages = {
            ErrorCode.INVALID_API_KEY: "Invalid API key provided. Please check your credentials.",
            ErrorCode.MISSING_API_KEY: "API key is required. Please provide your API key.",
            ErrorCode.INVALID_INPUT: "Invalid input provided. Please check your request and try again.",
            ErrorCode.LLM_PROVIDER_ERROR: "AI service is temporarily unavailable. Please try again later.",
            ErrorCode.LLM_RATE_LIMIT: "Rate limit exceeded. Please wait before making another request.",
            ErrorCode.DATABASE_CONNECTION: "Database service is temporarily unavailable. Please try again later.",
            ErrorCode.PROCESS_CRASHED: "Background service encountered an error. It will be restarted automatically.",
            ErrorCode.NETWORK_TIMEOUT: "Network request timed out. Please check your connection and try again.",
            ErrorCode.INTERNAL_ERROR: "An internal error occurred. Our team has been notified.",
        }

        return user_messages.get(
            code, "An error occurred. Please try again or contact support."
        )

    def _exception_to_vega_error(
        self, exception: Exception, context: Optional[ErrorContext]
    ) -> VegaError:
        """Convert a generic exception to VegaError"""

        # Map common exception types to error codes
        exception_mapping = {
            ValueError: ErrorCode.INVALID_INPUT,
            KeyError: ErrorCode.MISSING_PARAMETER,
            ConnectionError: ErrorCode.NETWORK_TIMEOUT,
            TimeoutError: ErrorCode.NETWORK_TIMEOUT,
            PermissionError: ErrorCode.PERMISSION_DENIED,
            FileNotFoundError: ErrorCode.FILE_NOT_FOUND,
        }

        code = exception_mapping.get(type(exception), ErrorCode.INTERNAL_ERROR)

        return self.create_error(
            code=code, message=str(exception), context=context or ErrorContext()
        )

    def _log_error(self, error: VegaError):
        """Log error with structured data"""
        log_data = {
            "error_id": error.error_id,
            "code": error.code.value,
            "category": error.category.value,
            "severity": error.severity.value,
            "message": error.message,
            "context": asdict(error.context),
        }

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical("Critical error occurred", **log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error("High severity error", **log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning("Medium severity error", **log_data)
        else:
            self.logger.info("Low severity error", **log_data)

    def _track_error(self, error: VegaError):
        """Track error occurrences for monitoring"""
        key = f"{error.code.value}:{error.category.value}"
        self.error_count[key] = self.error_count.get(key, 0) + 1

        # Alert on high error counts
        if self.error_count[key] > 10:
            self.logger.warning(
                "High error count detected",
                error_code=error.code.value,
                count=self.error_count[key],
            )

    def _attempt_recovery(self, error: VegaError):
        """Attempt automatic recovery based on error type"""
        if not error.recoverable:
            return

        # Try built-in recovery strategies first
        recovery_func = self.recovery_strategies.get(error.code)
        if recovery_func:
            try:
                recovery_func(error)
                self.logger.info(
                    "Built-in recovery attempted",
                    error_id=error.error_id,
                    code=error.code.value,
                )
            except Exception as e:
                self.logger.error(
                    "Built-in recovery failed",
                    error_id=error.error_id,
                    recovery_error=str(e),
                )

        # Try advanced recovery manager if available
        try:
            from .recovery_manager import get_recovery_manager

            recovery_manager = get_recovery_manager()

            # Schedule recovery attempt (async)
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a task for recovery
                    loop.create_task(recovery_manager.recover_from_error(error))
                else:
                    # Run recovery synchronously if no loop
                    asyncio.run(recovery_manager.recover_from_error(error))
            except RuntimeError:
                # No event loop available, skip advanced recovery
                pass

        except ImportError:
            # Recovery manager not available
            pass

    # Recovery strategy implementations
    def _handle_rate_limit(self, error: VegaError):
        """Handle rate limit errors"""
        # Could implement exponential backoff, queue requests, etc.
        pass

    def _handle_timeout(self, error: VegaError):
        """Handle timeout errors"""
        # Could implement retry with longer timeout
        pass

    def _handle_db_connection(self, error: VegaError):
        """Handle database connection errors"""
        # Could implement connection pool restart
        pass

    def _handle_process_crash(self, error: VegaError):
        """Handle process crashes"""
        # Could trigger process restart
        pass

    def _handle_resource_exhaustion(self, error: VegaError):
        """Handle resource exhaustion"""
        # Could trigger cleanup, scale down operations
        pass

    def _handle_circuit_breaker(self, error: VegaError):
        """Handle circuit breaker open state"""
        # Could implement fallback mechanisms
        pass

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": sum(self.error_count.values()),
            "error_breakdown": dict(self.error_count),
            "timestamp": datetime.utcnow().isoformat(),
        }


# Global error handler instance
_error_handler = None


def get_error_handler() -> VegaErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = VegaErrorHandler()
    return _error_handler


@contextmanager
def error_context(
    component: str,
    operation: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None,
    **params,
):
    """Context manager for error handling"""
    context = ErrorContext(
        user_id=user_id,
        session_id=session_id,
        request_id=request_id,
        component=component,
        operation=operation,
        parameters=params,
    )

    try:
        yield context
    except Exception as e:
        handler = get_error_handler()
        vega_error = handler.handle_error(e, context)
        raise vega_error.to_http_exception()


def handle_error(
    code: ErrorCode,
    message: str,
    user_message: Optional[str] = None,
    context: Optional[ErrorContext] = None,
    severity: Optional[ErrorSeverity] = None,
    recoverable: bool = True,
) -> VegaError:
    """Convenience function for error handling"""
    handler = get_error_handler()
    error = handler.create_error(
        code=code,
        message=message,
        user_message=user_message,
        context=context,
        severity=severity,
        recoverable=recoverable,
    )
    return handler.handle_error(error)


def log_info(message: str, **kwargs):
    """Structured info logging"""
    logger = structlog.get_logger("vega")
    logger.info(message, **kwargs)


def log_warning(message: str, **kwargs):
    """Structured warning logging"""
    logger = structlog.get_logger("vega")
    logger.warning(message, **kwargs)


def log_error(message: str, **kwargs):
    """Structured error logging"""
    logger = structlog.get_logger("vega")
    logger.error(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Structured debug logging"""
    logger = structlog.get_logger("vega")
    logger.debug(message, **kwargs)
