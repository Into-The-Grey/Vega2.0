#!/usr/bin/env python3
"""
Vega2.0 Hardened Core System
============================

This module provides bulletproof wrappers around all critical system operations.
Every function here is designed to:
1. Never crash under any circumstances
2. Always return a usable result (even if degraded)
3. Log all issues for debugging
4. Learn from failures to improve
5. Report serious issues to the operator
6. Suggest improvements based on patterns

Design Principles:
- Defense in depth: Multiple layers of protection
- Fail-safe defaults: Always have a fallback
- Observable: Everything is logged and measurable
- Self-healing: Automatic recovery when possible
- Learning: Build knowledge from experience
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Setup logging with fallback
try:
    from .logging_setup import get_core_logger  # type: ignore[import]

    logger = get_core_logger()
except Exception:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# =============================================================================
# Type Variables and Constants
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")

# Default timeouts (in seconds)
DEFAULT_TIMEOUT = 30.0
QUICK_TIMEOUT = 5.0
LONG_TIMEOUT = 120.0

# Retry configuration
MAX_IMMEDIATE_RETRIES = 2
MAX_BACKOFF_RETRIES = 5
BACKOFF_BASE = 1.0
BACKOFF_MAX = 60.0


# =============================================================================
# Severity and Status Enums
# =============================================================================


class Severity(Enum):
    """Issue severity levels"""

    DEBUG = "debug"  # Normal debugging info
    INFO = "info"  # Informational
    WARNING = "warning"  # Something unexpected but handled
    ERROR = "error"  # Something failed but recovered
    CRITICAL = "critical"  # Something failed, needs attention
    FATAL = "fatal"  # System cannot continue


class OperationStatus(Enum):
    """Status of an operation"""

    SUCCESS = "success"
    PARTIAL = "partial"  # Completed with degradation
    FAILED = "failed"  # Failed but recovered
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class OperationResult(Generic[T]):
    """
    Standard result wrapper for all operations.
    Always contains either a value or useful error info.
    """

    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    status: OperationStatus = OperationStatus.SUCCESS
    duration_ms: float = 0.0
    retries: int = 0
    degraded: bool = False
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, value: T, duration_ms: float = 0.0, **kwargs) -> "OperationResult[T]":
        """Create a successful result"""
        return cls(success=True, value=value, duration_ms=duration_ms, **kwargs)

    @classmethod
    def fail(
        cls,
        error: str,
        error_type: str = "UnknownError",
        status: OperationStatus = OperationStatus.FAILED,
        suggestions: Optional[List[str]] = None,
        **kwargs,
    ) -> "OperationResult[T]":
        """Create a failed result"""
        return cls(
            success=False,
            error=error,
            error_type=error_type,
            status=status,
            suggestions=suggestions or [],
            **kwargs,
        )

    @classmethod
    def partial(cls, value: T, error: str, **kwargs) -> "OperationResult[T]":
        """Create a partial success (degraded) result"""
        return cls(
            success=True,
            value=value,
            error=error,
            status=OperationStatus.PARTIAL,
            degraded=True,
            **kwargs,
        )


@dataclass
class IssueRecord:
    """Record of an issue for learning and reporting"""

    id: str
    timestamp: datetime
    operation: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    severity: Severity
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    resolution_time: Optional[datetime] = None
    occurrence_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "resolved": self.resolved,
            "resolution_strategy": self.resolution_strategy,
            "occurrence_count": self.occurrence_count,
        }


@dataclass
class ImprovementSuggestion:
    """Suggested improvement based on observed patterns"""

    id: str
    category: str  # "code", "config", "infrastructure", "feature"
    title: str
    description: str
    severity: Severity
    evidence: List[str]  # What patterns led to this suggestion
    implementation_hint: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "implementation_hint": self.implementation_hint,
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# Knowledge Base - Learning from Experience
# =============================================================================


class KnowledgeBase:
    """
    Stores and retrieves knowledge learned from system operation.
    This is the system's memory for what works and what doesn't.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/vega_knowledge.json")
        self._lock = Lock()

        # Error patterns and successful resolutions
        self.error_resolutions: Dict[str, Dict[str, Any]] = {}

        # Performance baselines
        self.performance_baselines: Dict[str, Dict[str, float]] = {}

        # Issue history
        self.issues: Dict[str, IssueRecord] = {}

        # Improvement suggestions
        self.suggestions: Dict[str, ImprovementSuggestion] = {}

        # Error frequency tracking
        self.error_frequency: Dict[str, int] = defaultdict(int)
        self.error_last_seen: Dict[str, datetime] = {}

        # Load existing knowledge
        self._load()

    def _load(self):
        """Load knowledge from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.error_resolutions = data.get("error_resolutions", {})
                    self.performance_baselines = data.get("performance_baselines", {})
                    self.error_frequency = defaultdict(int, data.get("error_frequency", {}))
                logger.debug(f"Loaded knowledge base with {len(self.error_resolutions)} resolutions")
        except Exception as e:
            logger.warning(f"Could not load knowledge base: {e}")

    def _save(self):
        """Save knowledge to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "error_resolutions": self.error_resolutions,
                        "performance_baselines": self.performance_baselines,
                        "error_frequency": dict(self.error_frequency),
                        "last_updated": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Could not save knowledge base: {e}")

    def get_error_signature(self, error_type: str, error_message: str) -> str:
        """Create a unique signature for an error pattern"""
        # Normalize the error message (remove variable parts like IDs, timestamps)
        import re

        normalized = re.sub(r"\b\d+\b", "<NUM>", error_message)  # Replace numbers
        normalized = re.sub(r"0x[0-9a-fA-F]+", "<ADDR>", normalized)  # Replace addresses
        normalized = re.sub(r"[\w-]+\.py:\d+", "<FILE:LINE>", normalized)  # Replace file:line

        content = f"{error_type}:{normalized[:200]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def record_error(self, operation: str, error: Exception, context: Dict[str, Any]) -> str:
        """Record an error and return its signature"""
        with self._lock:
            signature = self.get_error_signature(type(error).__name__, str(error))
            self.error_frequency[signature] += 1
            self.error_last_seen[signature] = datetime.utcnow()

            # Create issue record
            issue = IssueRecord(
                id=signature,
                timestamp=datetime.utcnow(),
                operation=operation,
                error_type=type(error).__name__,
                error_message=str(error)[:500],
                stack_trace=traceback.format_exc()[-2000:],
                context=context,
                severity=self._classify_severity(error),
                occurrence_count=self.error_frequency[signature],
            )
            self.issues[signature] = issue

            # Check if we should generate suggestions
            if self.error_frequency[signature] >= 3:
                self._generate_suggestions(signature, issue)

            self._save()
            return signature

    def record_resolution(
        self,
        error_signature: str,
        strategy: str,
        details: Dict[str, Any],
        success: bool,
    ):
        """Record whether a resolution strategy worked"""
        with self._lock:
            if error_signature not in self.error_resolutions:
                self.error_resolutions[error_signature] = {
                    "strategies": {},
                    "total_attempts": 0,
                }

            res = self.error_resolutions[error_signature]
            res["total_attempts"] += 1

            if strategy not in res["strategies"]:
                res["strategies"][strategy] = {"attempts": 0, "successes": 0}

            res["strategies"][strategy]["attempts"] += 1
            if success:
                res["strategies"][strategy]["successes"] += 1
                res["last_successful_strategy"] = strategy
                res["last_success_time"] = datetime.utcnow().isoformat()

                # Mark issue as resolved
                if error_signature in self.issues:
                    self.issues[error_signature].resolved = True
                    self.issues[error_signature].resolution_strategy = strategy
                    self.issues[error_signature].resolution_time = datetime.utcnow()

            self._save()

    def get_best_resolution(self, error_signature: str) -> Optional[str]:
        """Get the most successful resolution strategy for an error"""
        if error_signature not in self.error_resolutions:
            return None

        strategies = self.error_resolutions[error_signature].get("strategies", {})
        if not strategies:
            return None

        # Find strategy with highest success rate
        best = None
        best_rate = 0.0

        for name, stats in strategies.items():
            if stats["attempts"] > 0:
                rate = stats["successes"] / stats["attempts"]
                if rate > best_rate:
                    best_rate = rate
                    best = name

        return best

    def _classify_severity(self, error: Exception) -> Severity:
        """Classify the severity of an error"""
        error_type = type(error).__name__

        # Critical errors
        if any(t in error_type for t in ["Database", "Connection", "Auth", "Security"]):
            return Severity.CRITICAL

        # Standard errors
        if any(t in error_type for t in ["Timeout", "Network", "HTTP"]):
            return Severity.ERROR

        # Warnings
        if any(t in error_type for t in ["Validation", "Format", "Parse"]):
            return Severity.WARNING

        return Severity.ERROR

    def _generate_suggestions(self, signature: str, issue: IssueRecord):
        """Generate improvement suggestions based on error patterns"""
        suggestions = []

        # Frequent error suggestion
        if self.error_frequency[signature] >= 5:
            suggestions.append(
                ImprovementSuggestion(
                    id=f"freq_{signature}",
                    category="code",
                    title=f"Recurring error in {issue.operation}",
                    description=(
                        f"The error '{issue.error_type}' has occurred {self.error_frequency[signature]} times. "
                        f"Consider adding specific handling or prevention for this case."
                    ),
                    severity=Severity.WARNING,
                    evidence=[f"Occurred {self.error_frequency[signature]} times"],
                    implementation_hint=f"Add try/except for {issue.error_type} in {issue.operation}",
                )
            )

        # Timeout suggestion
        if "timeout" in issue.error_message.lower():
            suggestions.append(
                ImprovementSuggestion(
                    id=f"timeout_{signature}",
                    category="config",
                    title=f"Timeout issues in {issue.operation}",
                    description="Consider increasing timeout or adding retry logic.",
                    severity=Severity.INFO,
                    evidence=["Timeout error detected"],
                    implementation_hint="Increase LLM_TIMEOUT_SEC or add exponential backoff",
                )
            )

        for s in suggestions:
            self.suggestions[s.id] = s

    def get_pending_suggestions(self) -> List[ImprovementSuggestion]:
        """Get all pending improvement suggestions"""
        return list(self.suggestions.values())

    def get_unresolved_issues(self, severity_min: Severity = Severity.ERROR) -> List[IssueRecord]:
        """Get unresolved issues at or above a severity level"""
        severity_order = [
            Severity.DEBUG,
            Severity.INFO,
            Severity.WARNING,
            Severity.ERROR,
            Severity.CRITICAL,
            Severity.FATAL,
        ]
        min_index = severity_order.index(severity_min)

        return [
            issue
            for issue in self.issues.values()
            if not issue.resolved and severity_order.index(issue.severity) >= min_index
        ]


# Global knowledge base instance
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
    return _knowledge_base


# =============================================================================
# Core Decorators - Protection for Functions
# =============================================================================


def hardened(
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = MAX_IMMEDIATE_RETRIES,
    fallback: Optional[Callable[..., T]] = None,
    default: Optional[T] = None,
    critical: bool = False,
    operation_name: Optional[str] = None,
):
    """
    Decorator to make any async function bulletproof.

    Features:
    - Timeout protection
    - Automatic retries with backoff
    - Fallback function or default value
    - Full error logging
    - Knowledge base integration
    - Never crashes the caller

    Args:
        timeout: Maximum time for the operation
        retries: Number of retry attempts
        fallback: Fallback function to call on failure
        default: Default value to return on failure
        critical: If True, errors are logged as CRITICAL
        operation_name: Name for logging (defaults to function name)
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, OperationResult[T]]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> OperationResult[T]:
            op_name = operation_name or func.__name__
            kb = get_knowledge_base()
            start_time = time.perf_counter()
            last_error: Optional[Exception] = None
            retry_count = 0

            # Try the main function with retries
            for attempt in range(retries + 1):
                try:
                    # Execute with timeout
                    result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

                    duration = (time.perf_counter() - start_time) * 1000

                    # If we had to retry, note it
                    if attempt > 0:
                        logger.info(f"✅ {op_name} succeeded after {attempt} retries")

                    return OperationResult.ok(result, duration_ms=duration, retries=attempt)

                except asyncio.TimeoutError as e:
                    last_error = e
                    retry_count = attempt
                    logger.warning(f"⏱️ {op_name} timed out (attempt {attempt + 1}/{retries + 1})")

                except asyncio.CancelledError:
                    return OperationResult.fail(
                        "Operation cancelled",
                        "CancelledError",
                        status=OperationStatus.CANCELLED,
                    )

                except Exception as e:
                    last_error = e
                    retry_count = attempt
                    logger.warning(f"⚠️ {op_name} failed (attempt {attempt + 1}/{retries + 1}): {e}")

                # Backoff before retry
                if attempt < retries:
                    backoff = min(BACKOFF_BASE * (2**attempt), BACKOFF_MAX)
                    await asyncio.sleep(backoff)

            # All retries exhausted - record the error
            duration = (time.perf_counter() - start_time) * 1000
            # Ensure we have an error to record
            if last_error is None:
                last_error = RuntimeError("Operation failed with unknown error")
            error_signature = kb.record_error(
                op_name,
                last_error,
                {"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
            )

            # Check if we have a known resolution
            known_resolution = kb.get_best_resolution(error_signature)

            # Try fallback
            if fallback is not None:
                try:
                    logger.info(f"🔄 {op_name} trying fallback")
                    fallback_result = fallback(*args, **kwargs)
                    if asyncio.iscoroutine(fallback_result):
                        fallback_result = await fallback_result

                    kb.record_resolution(error_signature, "fallback", {}, True)

                    return OperationResult.partial(
                        fallback_result,
                        f"Used fallback after {last_error}",
                        duration_ms=duration,
                        retries=retry_count,
                        suggestions=[f"Consider investigating: {last_error}"],
                    )
                except Exception as fallback_error:
                    logger.error(f"❌ {op_name} fallback also failed: {fallback_error}")
                    kb.record_resolution(error_signature, "fallback", {}, False)

            # Return default if available
            if default is not None:
                return OperationResult.partial(
                    default,
                    f"Using default after {last_error}",
                    duration_ms=duration,
                    retries=retry_count,
                    suggestions=[
                        f"Error: {last_error}",
                        known_resolution and f"Try: {known_resolution}" or "No known resolution",
                    ],
                )

            # Log critical if configured
            if critical:
                logger.critical(
                    f"🚨 CRITICAL: {op_name} failed permanently after {retries + 1} attempts",
                    extra={
                        "error": str(last_error),
                        "error_signature": error_signature,
                    },
                )

            # Return failure result
            return OperationResult.fail(
                str(last_error),
                type(last_error).__name__,
                duration_ms=duration,
                retries=retry_count,
                suggestions=[
                    known_resolution and f"Known fix: {known_resolution}" or "No known resolution",
                    "Check logs for stack trace",
                ],
            )

        return wrapper

    return decorator


def hardened_sync(
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = MAX_IMMEDIATE_RETRIES,
    fallback: Optional[Callable[..., T]] = None,
    default: Optional[T] = None,
    critical: bool = False,
    operation_name: Optional[str] = None,
):
    """
    Decorator to make any sync function bulletproof.
    Similar to @hardened but for synchronous functions.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., OperationResult[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> OperationResult[T]:
            op_name = operation_name or func.__name__
            kb = get_knowledge_base()
            start_time = time.perf_counter()
            last_error: Optional[Exception] = None
            retry_count = 0

            for attempt in range(retries + 1):
                try:
                    result = func(*args, **kwargs)
                    duration = (time.perf_counter() - start_time) * 1000

                    if attempt > 0:
                        logger.info(f"✅ {op_name} succeeded after {attempt} retries")

                    return OperationResult.ok(result, duration_ms=duration, retries=attempt)

                except Exception as e:
                    last_error = e
                    retry_count = attempt
                    logger.warning(f"⚠️ {op_name} failed (attempt {attempt + 1}/{retries + 1}): {e}")

                if attempt < retries:
                    time.sleep(min(BACKOFF_BASE * (2**attempt), BACKOFF_MAX))

            # Record error
            duration = (time.perf_counter() - start_time) * 1000
            if last_error is None:
                last_error = RuntimeError(f"{op_name} failed with unknown error")
            error_signature = kb.record_error(op_name, last_error, {})

            # Try fallback
            if fallback is not None:
                try:
                    fallback_result = fallback(*args, **kwargs)
                    kb.record_resolution(error_signature, "fallback", {}, True)
                    return OperationResult.partial(
                        fallback_result,
                        f"Used fallback after {last_error}",
                        duration_ms=duration,
                        retries=retry_count,
                    )
                except Exception:
                    kb.record_resolution(error_signature, "fallback", {}, False)

            if default is not None:
                return OperationResult.partial(default, f"Using default after {last_error}")

            if critical:
                logger.critical(f"🚨 CRITICAL: {op_name} failed permanently")

            return OperationResult.fail(str(last_error), type(last_error).__name__)

        return wrapper

    return decorator


# =============================================================================
# Safe Execution Helpers
# =============================================================================


async def safe_execute(
    coro: Coroutine[Any, Any, T],
    default: T,
    operation_name: str = "operation",
    timeout: float = DEFAULT_TIMEOUT,
) -> T:
    """
    Safely execute a coroutine with timeout and fallback.
    Always returns a value, never raises.

    This is a simpler alternative to the @hardened decorator for one-off calls.
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"⏱️ {operation_name} timed out, using default")
        return default
    except asyncio.CancelledError:
        logger.info(f"🚫 {operation_name} cancelled, using default")
        return default
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}, using default")
        return default


def safe_execute_sync(
    func: Callable[..., T],
    args: tuple = (),
    kwargs: Optional[Dict] = None,
    default: T = None,
    operation_name: str = "operation",
) -> T:
    """
    Safely execute a sync function with fallback.
    Always returns a value, never raises.
    """
    kwargs = kwargs or {}
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}, using default")
        return default


@asynccontextmanager
async def safe_context(operation_name: str, critical: bool = False):
    """
    Async context manager that catches all errors and logs them.

    Usage:
        async with safe_context("database_query"):
            result = await db.query(...)
    """
    kb = get_knowledge_base()
    start = time.perf_counter()

    try:
        yield
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        signature = kb.record_error(operation_name, e, {"duration_ms": duration})

        if critical:
            logger.critical(f"🚨 CRITICAL in {operation_name}: {e}")
        else:
            logger.error(f"❌ Error in {operation_name}: {e}")

        # Don't re-raise - this is a safe context


@contextmanager
def safe_context_sync(operation_name: str, critical: bool = False):
    """
    Sync context manager that catches all errors and logs them.
    """
    kb = get_knowledge_base()
    start = time.perf_counter()

    try:
        yield
    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        signature = kb.record_error(operation_name, e, {"duration_ms": duration})

        if critical:
            logger.critical(f"🚨 CRITICAL in {operation_name}: {e}")
        else:
            logger.error(f"❌ Error in {operation_name}: {e}")


# =============================================================================
# Input Validation and Sanitization
# =============================================================================


def validate_input(
    value: Any,
    expected_type: type,
    name: str = "input",
    allow_none: bool = False,
    max_length: Optional[int] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> OperationResult[Any]:
    """
    Validate and sanitize input with detailed feedback.
    """
    # Check None
    if value is None:
        if allow_none:
            return OperationResult.ok(None)
        return OperationResult.fail(f"{name} cannot be None", "ValidationError")

    # Check type
    if not isinstance(value, expected_type):
        # Try to convert
        try:
            if expected_type == str:
                value = str(value)
            elif expected_type == int:
                value = int(value)
            elif expected_type == float:
                value = float(value)
            elif expected_type == bool:
                value = bool(value)
            else:
                return OperationResult.fail(
                    f"{name} must be {expected_type.__name__}, got {type(value).__name__}",
                    "TypeError",
                )
        except (ValueError, TypeError) as e:
            return OperationResult.fail(f"Cannot convert {name}: {e}", "ConversionError")

    # Check length for strings
    if isinstance(value, str) and max_length is not None:
        if len(value) > max_length:
            value = value[:max_length]
            logger.warning(f"Truncated {name} to {max_length} chars")

    # Check numeric bounds
    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            return OperationResult.fail(f"{name} must be >= {min_value}, got {value}", "RangeError")
        if max_value is not None and value > max_value:
            return OperationResult.fail(f"{name} must be <= {max_value}, got {value}", "RangeError")

    return OperationResult.ok(value)


def sanitize_string(
    value: str,
    max_length: int = 10000,
    remove_null: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """
    Sanitize a string to prevent crashes and security issues.
    """
    if not isinstance(value, str):
        value = str(value) if value is not None else ""

    # Remove null bytes (can cause issues)
    if remove_null:
        value = value.replace("\x00", "")

    # Strip whitespace
    if strip_whitespace:
        value = value.strip()

    # Truncate
    if len(value) > max_length:
        value = value[:max_length]

    return value


# =============================================================================
# Reporting and Alerting
# =============================================================================


class IssueReporter:
    """
    Handles reporting of serious issues to the operator.
    Batches reports to avoid spam and provides actionable information.
    """

    def __init__(self):
        self._pending_reports: List[IssueRecord] = []
        self._last_report_time: Optional[datetime] = None
        self._report_cooldown = timedelta(minutes=5)
        self._lock = Lock()

    def queue_report(self, issue: IssueRecord):
        """Queue an issue for reporting"""
        with self._lock:
            self._pending_reports.append(issue)

            # Check if we should report now
            if self._should_report_now(issue):
                self._send_report()

    def _should_report_now(self, issue: IssueRecord) -> bool:
        """Decide if we should report immediately"""
        # Always report critical/fatal immediately
        if issue.severity in (Severity.CRITICAL, Severity.FATAL):
            return True

        # Report if cooldown has passed and we have pending issues
        if self._last_report_time is None:
            return len(self._pending_reports) >= 3

        if datetime.utcnow() - self._last_report_time > self._report_cooldown:
            return len(self._pending_reports) >= 3

        return False

    def _send_report(self):
        """Send accumulated reports"""
        if not self._pending_reports:
            return

        # Group by severity
        by_severity = defaultdict(list)
        for issue in self._pending_reports:
            by_severity[issue.severity.value].append(issue)

        # Build report
        report_lines = [
            "=" * 60,
            "🚨 VEGA SYSTEM ISSUE REPORT",
            f"Time: {datetime.utcnow().isoformat()}",
            f"Issues: {len(self._pending_reports)}",
            "=" * 60,
        ]

        for severity in ["fatal", "critical", "error", "warning"]:
            issues = by_severity.get(severity, [])
            if issues:
                report_lines.append(f"\n{severity.upper()} ({len(issues)}):")
                for issue in issues[:5]:  # Limit per category
                    report_lines.append(f"  - [{issue.operation}] {issue.error_message[:100]}")

        # Add suggestions from knowledge base
        kb = get_knowledge_base()
        suggestions = kb.get_pending_suggestions()
        if suggestions:
            report_lines.append(f"\n💡 IMPROVEMENT SUGGESTIONS ({len(suggestions)}):")
            for s in suggestions[:3]:
                report_lines.append(f"  - {s.title}")

        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        # Log the report
        logger.critical(report)

        # Also print to console for visibility
        print(report)

        # Clear pending and update timestamp
        self._pending_reports = []
        self._last_report_time = datetime.utcnow()

    def get_summary(self) -> Dict[str, Any]:
        """Get current issue summary"""
        kb = get_knowledge_base()
        unresolved = kb.get_unresolved_issues()
        suggestions = kb.get_pending_suggestions()

        return {
            "pending_reports": len(self._pending_reports),
            "unresolved_issues": len(unresolved),
            "improvement_suggestions": len(suggestions),
            "last_report": (self._last_report_time.isoformat() if self._last_report_time else None),
        }


# Global reporter instance
_issue_reporter: Optional[IssueReporter] = None


def get_issue_reporter() -> IssueReporter:
    """Get or create the global issue reporter"""
    global _issue_reporter
    if _issue_reporter is None:
        _issue_reporter = IssueReporter()
    return _issue_reporter


def report_issue(issue: IssueRecord):
    """Convenience function to report an issue"""
    get_issue_reporter().queue_report(issue)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "Severity",
    "OperationStatus",
    # Data classes
    "OperationResult",
    "IssueRecord",
    "ImprovementSuggestion",
    # Decorators
    "hardened",
    "hardened_sync",
    # Safe execution
    "safe_execute",
    "safe_execute_sync",
    "safe_context",
    "safe_context_sync",
    # Validation
    "validate_input",
    "sanitize_string",
    # Knowledge base
    "KnowledgeBase",
    "get_knowledge_base",
    # Reporting
    "IssueReporter",
    "get_issue_reporter",
    "report_issue",
    # Constants
    "DEFAULT_TIMEOUT",
    "QUICK_TIMEOUT",
    "LONG_TIMEOUT",
]
