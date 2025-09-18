#!/usr/bin/env python3
"""
Enhanced Error Handling Utilities
=================================

Provides robust error handling, logging, and recovery mechanisms
for the autonomous debugging system.

Features:
- Centralized exception handling
- Graceful degradation strategies
- Error context capture and reporting
- Recovery and retry mechanisms
- Circuit breaker patterns for external services
"""

import functools
import traceback
import logging
import time
import asyncio
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Captures error context for better debugging"""

    component: str
    operation: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: float
    traceback_text: str
    recovery_attempted: bool = False
    recovery_successful: bool = False


def with_error_handling(
    component_name: str,
    operation_name: str,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    retry_count: int = 0,
    fallback_value: Any = None,
):
    """Decorator for robust error handling with optional retry and fallback"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(retry_count + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Create error context
                    error_context = ErrorContext(
                        component=component_name,
                        operation=operation_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        severity=severity,
                        timestamp=time.time(),
                        traceback_text=traceback.format_exc(),
                    )

                    # Log error based on severity
                    if severity == ErrorSeverity.CRITICAL:
                        logger.critical(
                            f"CRITICAL ERROR in {component_name}.{operation_name}: {e}"
                        )
                    elif severity == ErrorSeverity.HIGH:
                        logger.error(f"ERROR in {component_name}.{operation_name}: {e}")
                    elif severity == ErrorSeverity.MEDIUM:
                        logger.warning(
                            f"WARNING in {component_name}.{operation_name}: {e}"
                        )
                    else:
                        logger.info(
                            f"Minor issue in {component_name}.{operation_name}: {e}"
                        )

                    # If this is not the last attempt, wait and retry
                    if attempt < retry_count:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.info(
                            f"Retrying {operation_name} in {wait_time} seconds (attempt {attempt + 2}/{retry_count + 1})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    # All retries exhausted
                    break

            # If we get here, all attempts failed
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                logger.error(
                    f"Operation {component_name}.{operation_name} failed after {retry_count + 1} attempts"
                )
                raise last_exception
            else:
                logger.warning(
                    f"Operation {component_name}.{operation_name} failed, using fallback value"
                )
                return fallback_value

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Create error context
                    error_context = ErrorContext(
                        component=component_name,
                        operation=operation_name,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        severity=severity,
                        timestamp=time.time(),
                        traceback_text=traceback.format_exc(),
                    )

                    # Log error based on severity
                    if severity == ErrorSeverity.CRITICAL:
                        logger.critical(
                            f"CRITICAL ERROR in {component_name}.{operation_name}: {e}"
                        )
                    elif severity == ErrorSeverity.HIGH:
                        logger.error(f"ERROR in {component_name}.{operation_name}: {e}")
                    elif severity == ErrorSeverity.MEDIUM:
                        logger.warning(
                            f"WARNING in {component_name}.{operation_name}: {e}"
                        )
                    else:
                        logger.info(
                            f"Minor issue in {component_name}.{operation_name}: {e}"
                        )

                    # If this is not the last attempt, wait and retry
                    if attempt < retry_count:
                        wait_time = 2**attempt  # Exponential backoff
                        logger.info(
                            f"Retrying {operation_name} in {wait_time} seconds (attempt {attempt + 2}/{retry_count + 1})"
                        )
                        time.sleep(wait_time)
                        continue

                    # All retries exhausted
                    break

            # If we get here, all attempts failed
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                logger.error(
                    f"Operation {component_name}.{operation_name} failed after {retry_count + 1} attempts"
                )
                raise last_exception
            else:
                logger.warning(
                    f"Operation {component_name}.{operation_name} failed, using fallback value"
                )
                return fallback_value

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for external service calls"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True

    def on_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"

    def on_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def safe_database_operation(operation_name: str):
    """Decorator for safe database operations with error handling"""
    return with_error_handling(
        component_name="database",
        operation_name=operation_name,
        severity=ErrorSeverity.HIGH,
        retry_count=3,
        fallback_value=None,
    )


def safe_file_operation(operation_name: str):
    """Decorator for safe file operations with error handling"""
    return with_error_handling(
        component_name="filesystem",
        operation_name=operation_name,
        severity=ErrorSeverity.MEDIUM,
        retry_count=2,
        fallback_value=None,
    )


def safe_network_operation(operation_name: str):
    """Decorator for safe network operations with error handling"""
    return with_error_handling(
        component_name="network",
        operation_name=operation_name,
        severity=ErrorSeverity.MEDIUM,
        retry_count=3,
        fallback_value={},
    )


class ErrorReporter:
    """Centralized error reporting and monitoring"""

    def __init__(self):
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[str, int] = {}

    def report_error(self, error_context: ErrorContext):
        """Report an error for monitoring"""
        self.error_history.append(error_context)

        # Update counts
        key = f"{error_context.component}.{error_context.operation}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1

        # Keep only recent errors (last 1000)
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        recent_errors = [
            e for e in self.error_history if time.time() - e.timestamp < 3600
        ]  # Last hour

        return {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_counts": self.error_counts,
            "critical_errors": len(
                [e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL]
            ),
            "high_errors": len(
                [e for e in recent_errors if e.severity == ErrorSeverity.HIGH]
            ),
        }


# Global error reporter instance
error_reporter = ErrorReporter()
