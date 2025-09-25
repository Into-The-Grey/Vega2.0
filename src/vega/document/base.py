"""
Vega 2.0 Document Intelligence Base Classes

This module provides common base classes, patterns, and utilities for all document intelligence modules.
It establishes consistent async patterns, configuration management, error handling, and extensibility.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from pathlib import Path
import uuid
from datetime import datetime


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
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 300.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "metadata": self.metadata,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass
class ProcessingResult:
    """Base class for processing results"""

    success: bool
    context: ProcessingContext
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: Optional[float] = None

    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "context": self.context.to_dict(),
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "processing_time_ms": self.processing_time_ms,
        }


T = TypeVar("T")
ConfigType = TypeVar("ConfigType")


class BaseDocumentProcessor(ABC, Generic[ConfigType]):
    """
    Abstract base class for all document processors.
    Provides consistent async patterns, configuration management, and error handling.
    """

    def __init__(self, config: ConfigType):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        self._processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "avg_processing_time_ms": 0.0,
        }

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

        except asyncio.TimeoutError:
            error_msg = f"Processing timeout after {context.timeout_seconds}s"
            result.add_error(error_msg)
            self.logger.error(error_msg)
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )

        except ValidationError as e:
            result.add_error(f"Input validation failed: {e}")
            self.logger.error(f"Validation error: {e}")
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )

        except ProcessingError as e:
            result.add_error(f"Processing failed: {e}")
            self.logger.error(f"Processing error: {e}")
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )

        except Exception as e:
            result.add_error(f"Unexpected error: {e}")
            self.logger.exception(f"Unexpected error in {self.__class__.__name__}")
            self._update_stats(
                False, (asyncio.get_event_loop().time() - start_time) * 1000
            )

        finally:
            result.processing_time_ms = (
                asyncio.get_event_loop().time() - start_time
            ) * 1000

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
        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "class_name": self.__class__.__name__,
            "stats": self.get_stats(),
        }


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
    return await asyncio.gather(*tasks, return_exceptions=True)


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
        self._metrics = {}
        self._start_times = {}

    def start_timer(self, name: str) -> None:
        self._start_times[name] = asyncio.get_event_loop().time()

    def end_timer(self, name: str) -> float:
        if name not in self._start_times:
            return 0.0
        duration = asyncio.get_event_loop().time() - self._start_times[name]
        self.record_metric(f"{name}_duration_ms", duration * 1000)
        del self._start_times[name]
        return duration

    def record_metric(self, name: str, value: float) -> None:
        if name not in self._metrics:
            self._metrics[name] = []
        self._metrics[name].append(value)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            name: {
                "count": len(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
            }
            for name, values in self._metrics.items()
        }
