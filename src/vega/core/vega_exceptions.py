#!/usr/bin/env python3
"""
vega_exceptions.py - Vega specific exception classes

These exception classes are used by the error handling system and tests.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(Enum):
    """Simple error codes for exceptions"""

    INVALID_INPUT = "INVALID_INPUT"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    PROCESS_CRASH = "PROCESS_CRASH"
    UNKNOWN = "UNKNOWN"


class VegaException(Exception):
    """Base exception for all Vega-specific errors"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[Any] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = ErrorCode(error_code) if error_code else ErrorCode.UNKNOWN
        self.details = details or {}
        self.context = context


class InvalidInputError(VegaException):
    """Raised when input validation fails"""

    def __init__(self, field: str, value: Any, message: Optional[str] = None):
        if message is None:
            message = f"Invalid value for field '{field}': {value}"
        super().__init__(message, "INVALID_INPUT")
        self.field = field
        self.value = value


class LLMRateLimitError(VegaException):
    """Raised when LLM service rate limit is exceeded"""

    def __init__(
        self,
        provider: str,
        retry_after: Optional[int] = None,
        message: Optional[str] = None,
    ):
        if message is None:
            message = f"Rate limit exceeded for {provider}"
            if retry_after:
                message += f", retry after {retry_after} seconds"
        super().__init__(message, "LLM_RATE_LIMIT")
        self.provider = provider
        self.retry_after = retry_after


class ProcessCrashError(VegaException):
    """Raised when a process crashes unexpectedly"""

    def __init__(
        self,
        process_name: str,
        exit_code: Optional[int] = None,
        message: Optional[str] = None,
    ):
        if message is None:
            message = f"Process '{process_name}' crashed"
            if exit_code is not None:
                message += f" with exit code {exit_code}"
        super().__init__(message, "PROCESS_CRASH")
        self.process_name = process_name
        self.exit_code = exit_code
