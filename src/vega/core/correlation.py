"""
correlation.py - Request correlation ID middleware for distributed tracing

Generates unique request IDs for tracking requests through the entire system.
Correlation IDs are:
- Generated for each incoming request
- Included in all log messages
- Returned in response headers
- Propagated to downstream services
"""

from __future__ import annotations

import uuid
import logging
from contextvars import ContextVar
from typing import Optional, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context variable for storing correlation ID across async boundaries
correlation_id_var: ContextVar[Optional[str]] = ContextVar(
    "correlation_id", default=None
)

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to generate and inject correlation IDs into requests.

    Features:
    - Generates UUID4 for each request if not provided
    - Accepts existing correlation ID from X-Correlation-ID header
    - Stores ID in context variable for access anywhere in request lifecycle
    - Injects ID into response headers
    - Adds ID to all log messages
    """

    def __init__(self, app, header_name: str = "X-Correlation-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation ID"""

        # Check if correlation ID exists in request headers
        correlation_id = request.headers.get(self.header_name)

        # Generate new ID if not provided
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        # Store in context variable for access throughout request
        correlation_id_var.set(correlation_id)

        # Add to request state for easy access
        request.state.correlation_id = correlation_id

        # Log request with correlation ID
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={"correlation_id": correlation_id},
        )

        # Process request
        try:
            response = await call_next(request)

            # Inject correlation ID into response headers
            response.headers[self.header_name] = correlation_id

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} -> {response.status_code}",
                extra={"correlation_id": correlation_id},
            )

            return response

        except Exception as e:
            logger.error(
                f"Request failed: {request.method} {request.url.path} -> {type(e).__name__}: {e}",
                extra={"correlation_id": correlation_id},
                exc_info=True,
            )
            raise

        finally:
            # Clear correlation ID from context
            correlation_id_var.set(None)


def get_correlation_id() -> Optional[str]:
    """
    Get the current request's correlation ID.

    Returns None if called outside of request context.

    Usage:
        correlation_id = get_correlation_id()
        logger.info("Processing item", extra={"correlation_id": correlation_id})
    """
    return correlation_id_var.get()


def log_with_correlation(
    logger_instance: logging.Logger, level: str, message: str, **kwargs
):
    """
    Convenience function to log with automatic correlation ID injection.

    Usage:
        log_with_correlation(logger, "info", "Processing started", user_id=123)
    """
    correlation_id = get_correlation_id()
    extra = kwargs.get("extra", {})

    if correlation_id:
        extra["correlation_id"] = correlation_id

    kwargs["extra"] = extra
    getattr(logger_instance, level)(message, **kwargs)


class CorrelationIdFilter(logging.Filter):
    """
    Logging filter to automatically add correlation ID to all log records.

    Usage in logging configuration:
        handler.addFilter(CorrelationIdFilter())
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record if available"""
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id if correlation_id else "N/A"
        return True


def configure_correlation_logging():
    """
    Configure root logger to include correlation IDs in all log messages.

    Should be called during application startup.
    """
    # Add filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(CorrelationIdFilter())

    # Update formatter to include correlation ID
    for handler in root_logger.handlers:
        if handler.formatter:
            # Get existing format string
            format_str = (
                handler.formatter._fmt if hasattr(handler.formatter, "_fmt") else None
            )

            if format_str and "correlation_id" not in format_str:
                # Inject correlation ID into format
                new_format = format_str.replace(
                    "%(message)s", "[%(correlation_id)s] %(message)s"
                )
                handler.setFormatter(logging.Formatter(new_format))
        else:
            # Create new formatter with correlation ID
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s [%(levelname)s] [%(correlation_id)s] %(name)s: %(message)s"
                )
            )

    logger.info("Correlation ID logging configured")


# Example usage and integration guide
"""
Integration with FastAPI:

    from fastapi import FastAPI
    from correlation import CorrelationIdMiddleware, configure_correlation_logging
    
    app = FastAPI()
    
    # Add middleware
    app.add_middleware(CorrelationIdMiddleware)
    
    # Configure logging (call once at startup)
    @app.on_event("startup")
    async def startup_event():
        configure_correlation_logging()

Using correlation IDs in code:

    from correlation import get_correlation_id, log_with_correlation
    import logging
    
    logger = logging.getLogger(__name__)
    
    async def my_function():
        # Automatic - just log normally, correlation ID is auto-added
        logger.info("Processing started")
        
        # Manual access if needed
        correlation_id = get_correlation_id()
        print(f"Request ID: {correlation_id}")
        
        # Convenience function
        log_with_correlation(logger, "info", "Custom log", user_id=123)

Client usage:

    import requests
    
    # Create new request with correlation ID
    response = requests.post(
        "http://localhost:8000/chat",
        headers={"X-Correlation-ID": "my-custom-id-123"},
        json={"prompt": "Hello"}
    )
    
    # Correlation ID is returned in response
    print(response.headers["X-Correlation-ID"])  # my-custom-id-123
"""
