"""
Vega 2.0 Document Intelligence Base Classes

This module provides common base classes, patterns, and utilities for all document intelligence modules.
It establishes consistent async patterns, configuration management, error handling, and extensibility.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar


logger = logging.getLogger(__name__)


class DocumentIntelligenceError(Exception):
    """Base exception for all document intelligence errors"""

    pass


class ConfigurationError(DocumentIntelligenceError):
    """Configuration-related errors"""

    pass


class ProcessingError(DocumentIntelligenceError):
    """Processing-related errors"""

    pass


class ValidationError(DocumentIntelligenceError):
    """Input validation errors"""

    pass


@dataclass
class ProcessingContext:
    """Context information for document processing operations"""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "user_id": self.user_id,
            "metadata": self.metadata,
            "timeout_seconds": self.timeout_seconds,
        }

    def __str__(self) -> str:  # pragma: no cover - simple representation
        return (
            "ProcessingContext("
            f"context_id={self.context_id}, "
            f"session_id={self.session_id}, "
            f"user_id={self.user_id}, "
            f"created_at={self.created_at.isoformat()}"
            ")"
        )


@dataclass
class ProcessingResult:
    """Base class for processing results"""

    success: bool
    context: ProcessingContext
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        # Ensure single error attribute is always available for convenience access
        self.__dict__.setdefault("error", self.errors[0] if self.errors else None)

    def add_error(self, message: str) -> None:
        if message:
            self.errors.append(message)
        self.success = False
        self.data = None
        self.__dict__["error"] = self.errors[0] if self.errors else None

    def add_warning(self, message: str) -> None:
        if message:
            self.warnings.append(message)

    @property
    def processing_time(self) -> Optional[float]:
        return self.processing_time_ms / 1000 if self.processing_time_ms else None

    @property
    def results(self) -> Optional[Dict[str, Any]]:
        """Alias for data property for backwards compatibility"""
        return self.data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "context": self.context.to_dict(),
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }

    def __str__(self) -> str:
        return (
            f"ProcessingResult(success={self.success}, "
            f"context_id={self.context.context_id}, errors={self.errors}, "
            f"warnings={self.warnings})"
        )


T = TypeVar("T")
ConfigType = TypeVar("ConfigType")


def _processing_result_success(
    cls,
    data: Optional[Dict[str, Any]],
    context: ProcessingContext,
    processing_time_ms: Optional[float] = None,
) -> ProcessingResult:
    result = cls(success=True, context=context, data=data)
    result.processing_time_ms = (
        processing_time_ms if processing_time_ms is not None else 1.0
    )
    result.__dict__["error"] = None
    return result


def _processing_result_error(
    cls,
    message: str,
    context: ProcessingContext,
    processing_time_ms: Optional[float] = None,
) -> ProcessingResult:
    result = cls(success=False, context=context, data=None)
    if message:
        result.errors.append(message)
    result.processing_time_ms = (
        processing_time_ms if processing_time_ms is not None else 1.0
    )
    result.__dict__["error"] = result.errors[0] if result.errors else None
    return result


ProcessingResult.success = classmethod(_processing_result_success)  # type: ignore[attr-defined]
ProcessingResult.error = classmethod(_processing_result_error)  # type: ignore[attr-defined]


class _DefaultConfig:
    """A tiny forgiving default config used when no explicit config is provided.

    It returns sensible falsy defaults for attributes and exposes a validate_config()
    that returns an empty list so callers that expect a ConfigurableComponent still work.
    """

    def __getattr__(self, name: str):
        # Provide conservative defaults for commonly used names
        if name in ("min_confidence", "max_document_length", "timeout_seconds"):
            return 0.0
        if name in ("supported_languages", "regulations", "clause_patterns"):
            return []
        if name.startswith("use_") or name.startswith("enable_"):
            return False
        return None

    def validate_config(self) -> List[str]:
        return []


class BaseDocumentProcessor(ABC, Generic[ConfigType]):
    """
    Abstract base class for all document processors.
    Provides consistent async patterns, configuration management, and error handling.
    """

    def __init__(self, config: Optional[ConfigType] = None):
        # Accept None for tests or convenience and provide a forgiving default
        # configuration object so downstream code can access attributes safely.
        self.config = config if config is not None else _DefaultConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self.metrics = MetricsCollector()
        self._processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time_ms": 0.0,
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        """Initialize the processor (async setup, load models, etc.)"""
        if self._initialized:
            return

        try:
            await self._async_initialize()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise ConfigurationError(f"Initialization failed: {e}")

    @abstractmethod
    async def _async_initialize(self) -> None:
        """Subclass-specific async initialization"""
        pass

    async def process(
        self, input_data: Any, context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """
        Main processing entry point with consistent error handling and timing
        """
        if not self._initialized:
            await self.initialize()

        if context is None:
            context = ProcessingContext()

        self.metrics.increment_counter("requests_total")
        timer_name = f"{self.__class__.__name__}_process"
        self.metrics.start_timer(timer_name)

        start_time = asyncio.get_event_loop().time()
        result = ProcessingResult(success=True, context=context)

        try:
            # Input validation
            self._validate_input(input_data)

            # Apply timeout wrapper
            processed_data = await asyncio.wait_for(
                self._process_internal(input_data, context),
                timeout=context.timeout_seconds,
            )

            result.data = processed_data
            self._update_stats(
                True, (asyncio.get_event_loop().time() - start_time) * 1000
            )
            self.metrics.increment_counter("requests_success")

        except asyncio.TimeoutError:
            error_msg = f"Processing timeout after {context.timeout_seconds}s"
            result.add_error(error_msg)
            self.logger.error(error_msg)
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )
            self.metrics.increment_counter("requests_timeout")

        except ValidationError as e:
            result.add_error(f"Input validation failed: {e}")
            self.logger.error(f"Validation error: {e}")
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )
            self.metrics.increment_counter("requests_failed")

        except ProcessingError as e:
            result.add_error(f"Processing failed: {e}")
            self.logger.error(f"Processing error: {e}")
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )
            self.metrics.increment_counter("requests_failed")

        except Exception as e:
            result.add_error(f"Unexpected error: {e}")
            self.logger.exception(f"Unexpected error in {self.__class__.__name__}")
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )
            self.metrics.increment_counter("requests_failed")

        finally:
            result.processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000
            duration = self.metrics.end_timer(timer_name)
            self.metrics.record_metric("processing_time_seconds", duration)
            self.metrics.record_metric("processing_time_ms", result.processing_time_ms)

        return result

    @abstractmethod
    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Subclass-specific processing implementation"""
        pass

    def _validate_input(self, input_data: Any) -> None:
        """Validate input data - override in subclasses for specific validation"""
        if input_data is None:
            raise ValidationError("Input data cannot be None")

    def _update_stats(self, success: bool, processing_time_ms: float) -> None:
        """Update internal processing statistics"""
        self._processing_stats["total_processed"] += 1
        if success:
            self._processing_stats["successful"] += 1
        else:
            self._processing_stats["failed"] += 1

        # Update running average
        total = self._processing_stats["total_processed"]
        current_avg = self._processing_stats["avg_processing_time_ms"]
        self._processing_stats["avg_processing_time_ms"] = (
            current_avg * (total - 1) + processing_time_ms
        ) / total

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self._processing_stats.copy()

    async def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring"""
        status = "healthy" if self._initialized else "not_initialized"
        return {
            "status": status,
            "class_name": self.__class__.__name__,
            "initialized": self._initialized,
            "last_check": datetime.utcnow().isoformat(),
            "stats": self.get_stats(),
            "metrics": self.metrics.get_metrics(),
        }

    async def cleanup(self) -> None:
        """Cleanup resources - override in subclasses if needed"""
        self._initialized = False
        self.metrics.reset()
        self.logger.info(f"{self.__class__.__name__} cleaned up")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup"""
        await self.cleanup()
        return False  # Don't suppress exceptions


class ConfigurableComponent(ABC):
    """Base class for configurable components with validation"""

    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors"""
        pass

    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        if hasattr(self, "__dict__"):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return {}


async def batch_process(
    processor: BaseDocumentProcessor,
    inputs: List[Any],
    context: Optional[ProcessingContext] = None,
    max_concurrent: int = 5,
) -> List[ProcessingResult]:
    """
    Process multiple inputs concurrently with controlled parallelism
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_single(input_data: Any) -> ProcessingResult:
        async with semaphore:
            return await processor.process(input_data, context)

    tasks = [process_single(input_data) for input_data in inputs]
    results = await asyncio.gather(*tasks)
    return list(results)


def handle_import_error(module_name: str, optional: bool = True) -> bool:
    """
    Consistent handling of optional dependencies
    """
    try:
        __import__(module_name)
        return True
    except ImportError as e:
        if optional:
            logger.warning(f"Optional dependency '{module_name}' not available: {e}")
            return False
        else:
            logger.error(f"Required dependency '{module_name}' not available: {e}")
            raise ConfigurationError(f"Missing required dependency: {module_name}")


class MetricsCollector:
    """Simple metrics collector for document processing"""

    def __init__(self):
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self.timers: Dict[str, float] = {}

    def _ensure_metric(self, name: str) -> Dict[str, Any]:
        return self._metrics.setdefault(name, {})

    def start_timer(self, name: str) -> None:
        self.timers[name] = time.perf_counter()

    def end_timer(self, name: str) -> float:
        start = self.timers.pop(name, None)
        if start is None:
            return 0.0
        duration = time.perf_counter() - start
        metric = self._ensure_metric(name)
        metric["duration"] = duration
        return duration

    def record_metric(self, name: str, value: float) -> None:
        metric = self._ensure_metric(name)
        values = metric.setdefault("values", [])
        values.append(value)
        metric["count"] = len(values)
        metric["avg"] = sum(values) / len(values)
        metric["min"] = min(values)
        metric["max"] = max(values)

    def increment_counter(self, name: str, value: int = 1) -> int:
        metric = self._ensure_metric(name)
        metric["count"] = metric.get("count", 0) + value
        return metric["count"]

    def set_gauge(self, name: str, value: float) -> None:
        metric = self._ensure_metric(name)
        metric["value"] = value

    @contextmanager
    def timer(self, name: str):
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)

    def reset(self) -> None:
        self._metrics.clear()
        self.timers.clear()

    def get_metrics(self) -> Dict[str, Any]:
        return {name: metric.copy() for name, metric in self._metrics.items()}
