#!/usr/bin/env python3
"""
Vega Centralized Logging Infrastructure
======================================

Provides structured logging configuration for all Vega modules with:
- Per-module log files under logs/{module}/
- Rotating file handlers with size and time limits
- Console and file output with different log levels
- JSON-structured logs for analysis
- Thread-safe logging across async operations
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime


class VegaLogger:
    """Centralized logger factory for all Vega modules"""

    _loggers: Dict[str, logging.Logger] = {}
    _base_log_dir = Path("logs")
    _setup_done = set()

    @classmethod
    def setup_base_logging(cls, base_dir: Optional[Path] = None):
        """Initialize the base logging directory structure"""
        if base_dir:
            cls._base_log_dir = base_dir
        else:
            env_dir = os.environ.get("VEGA_LOG_DIR")
            if env_dir:
                cls._base_log_dir = Path(env_dir)

        # Ensure logs directory exists
        cls._base_log_dir.mkdir(exist_ok=True)

        # Create module-specific directories
        modules = [
            "core",
            "app",
            "llm",
            "ui",
            "voice",
            "datasets",
            "training",
            "integrations",
            "learning",
            "analysis",
            "network",
            "autonomous",
            "intelligence",
        ]

        for module in modules:
            (cls._base_log_dir / module).mkdir(exist_ok=True)

    @classmethod
    def get_logger(cls, module_name: str, level: str = "INFO") -> logging.Logger:
        """Get or create a logger for the specified module"""

        if module_name in cls._loggers:
            return cls._loggers[module_name]

        # Setup base logging if not done
        cls.setup_base_logging()

        logger = logging.getLogger(f"vega.{module_name}")
        logger.setLevel(getattr(logging, level.upper()))

        # Clear any existing handlers
        logger.handlers.clear()

        # Create module log directory
        log_dir = cls._base_log_dir / module_name
        log_dir.mkdir(exist_ok=True)

        # File handler with rotation (writes JSON lines to .log as tests expect)
        log_file = log_dir / f"{module_name}.log"
        # Use small rotation size so tests that write ~2MB will trigger rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=200 * 1024, backupCount=5, encoding="utf-8"
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)

        # Formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        simple_formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")

        # JSON formatter
        module_label = module_name  # capture for closure

        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "module": module_label,
                    "level": record.levelname,
                    "function": record.funcName,
                    "line": record.lineno,
                    "message": record.getMessage(),
                    "thread": record.thread,
                    "process": record.process,
                }

                # Add exception info if present
                if record.exc_info:
                    # Use class method on VegaLogger to avoid scope issues
                    log_entry["exception"] = VegaLogger.format_exception(
                        record.exc_info
                    )

                # Add extra fields directly at top-level as tests expect
                for key, value in record.__dict__.items():
                    if key not in [
                        "name",
                        "msg",
                        "args",
                        "levelname",
                        "levelno",
                        "pathname",
                        "filename",
                        "module",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                        "lineno",
                        "funcName",
                        "created",
                        "msecs",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "processName",
                        "process",
                        "getMessage",
                    ]:
                        log_entry[key] = value

                return json.dumps(log_entry)

        # Set formatters
        file_handler.setFormatter(JsonFormatter())
        console_handler.setFormatter(simple_formatter)

        # Set levels
        file_handler.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.WARNING)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate logs
        logger.propagate = False

        cls._loggers[module_name] = logger
        return logger

    @staticmethod
    def format_exception(exc_info) -> str:
        """Format exception information for JSON logging"""
        import traceback

        return "".join(traceback.format_exception(*exc_info))

    @classmethod
    def log_request(
        cls,
        module: str,
        request_id: str,
        endpoint: str,
        method: str,
        status: int,
        duration: float,
        **kwargs,
    ):
        """Log HTTP request information"""
        logger = cls.get_logger(module)
        logger.info(
            f"Request {method} {endpoint} - Status: {status} - Duration: {duration:.3f}s",
            extra={
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "status_code": status,
                "duration_ms": duration * 1000,
                **kwargs,
            },
        )

    @classmethod
    def log_llm_interaction(
        cls,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration: float,
        **kwargs,
    ):
        """Log LLM interaction metrics"""
        logger = cls.get_logger("llm")
        logger.info(
            f"LLM {model} - Prompt: {prompt_tokens}t, Completion: {completion_tokens}t, Duration: {duration:.2f}s",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "duration_seconds": duration,
                **kwargs,
            },
        )

    @classmethod
    def log_system_metric(
        cls, metric_name: str, value: float, unit: str = "", **kwargs
    ):
        """Log system metrics for monitoring"""
        logger = cls.get_logger("core")
        logger.info(
            f"Metric {metric_name}: {value}{unit}",
            extra={
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                **kwargs,
            },
        )

    @classmethod
    def get_log_files(cls, module: Optional[str] = None) -> Dict[str, List[Path]]:
        """Get all log files, optionally filtered by module"""
        log_files = {}

        if module:
            module_dir = cls._base_log_dir / module
            if module_dir.exists():
                log_files[module] = list(module_dir.glob("*.log")) + list(
                    module_dir.glob("*.jsonl")
                )
        else:
            for module_dir in cls._base_log_dir.iterdir():
                if module_dir.is_dir():
                    log_files[module_dir.name] = list(module_dir.glob("*.log")) + list(
                        module_dir.glob("*.jsonl")
                    )

        return log_files

    @classmethod
    def tail_log(cls, module: str, lines: int = 50) -> List[str]:
        """Get the last N lines from a module's log file"""
        log_file = cls._base_log_dir / module / f"{module}.log"

        if not log_file.exists():
            raise FileNotFoundError(f"No log file found for module: {module}")

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                return f.readlines()[-lines:]
        except Exception as e:
            raise e

    @classmethod
    def list_modules(cls) -> List[str]:
        """List modules that have active loggers"""
        return sorted(list(cls._loggers.keys()))


# Convenience functions for quick module logger access
def get_app_logger() -> logging.Logger:
    """Get the app module logger"""
    return VegaLogger.get_logger("app")


def get_llm_logger() -> logging.Logger:
    """Get the LLM module logger"""
    return VegaLogger.get_logger("llm")


def get_ui_logger() -> logging.Logger:
    """Get the UI module logger"""
    return VegaLogger.get_logger("ui")


def get_voice_logger() -> logging.Logger:
    """Get the voice module logger"""
    return VegaLogger.get_logger("voice")


def get_core_logger() -> logging.Logger:
    """Get the core module logger"""
    return VegaLogger.get_logger("core")


def get_integration_logger() -> logging.Logger:
    """Get the integrations module logger"""
    return VegaLogger.get_logger("integrations")


def get_analysis_logger() -> logging.Logger:
    """Get the analysis module logger"""
    return VegaLogger.get_logger("analysis")


# Initialize logging on import
VegaLogger.setup_base_logging()


# Convenience logging functions
def log_info(module: str, message: str, **extra):
    VegaLogger.get_logger(module).info(message, extra=extra)


def log_error(module: str, message: str, **extra):
    VegaLogger.get_logger(module).error(message, extra=extra)


def log_warning(module: str, message: str, **extra):
    VegaLogger.get_logger(module).warning(message, extra=extra)


def log_debug(module: str, message: str, **extra):
    VegaLogger.get_logger(module).debug(message, extra=extra)
