"""
FastAPI Error Handling Middleware
===============================

Middleware to integrate VegaErrorHandler with FastAPI applications.
"""

import uuid
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint

from .error_handler import (
    get_error_handler,
    ErrorContext,
    ErrorCode,
    ErrorSeverity,
    log_info,
    log_error,
)
from .exceptions import VegaException


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())

        # Extract user/session info from headers if available
        user_id = request.headers.get("X-User-ID")
        session_id = request.headers.get("X-Session-ID")
        api_key = request.headers.get("X-API-Key")

        # Create error context
        context = ErrorContext(
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            component="api",
            operation=f"{request.method} {request.url.path}",
            parameters={
                "url": str(request.url),
                "method": request.method,
                "headers": dict(request.headers),
                "client": request.client.host if request.client else None,
            },
        )

        # Add request ID to headers for response
        request.state.request_id = request_id
        request.state.error_context = context

        # Log request start
        log_info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            user_id=user_id,
            session_id=session_id,
        )

        try:
            # Process request
            response = await call_next(request)

            # Log successful response
            log_info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                method=request.method,
                path=request.url.path,
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            # Handle error with structured logging
            handler = get_error_handler()

            # All exceptions go through the same error handling path
            # The error handler will properly detect VegaException and preserve error codes
            handled_error = handler.handle_error(e, context)

            # Log error
            log_error(
                "Request failed",
                request_id=request_id,
                error_id=handled_error.error_id,
                error_code=handled_error.code.value,
                method=request.method,
                path=request.url.path,
                error_message=str(e),
            )

            # Convert to HTTP response instead of raising exception
            http_exception = handled_error.to_http_exception()

            # Create a proper JSON response
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=http_exception.status_code,
                content=http_exception.detail,
                headers={"X-Request-ID": request_id},
            )


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware for request tracking and metrics"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Track request metrics
        if hasattr(request.app.state, "metrics"):
            request.app.state.metrics["requests_total"] += 1

        try:
            response = await call_next(request)

            # Track response metrics
            if hasattr(request.app.state, "metrics"):
                request.app.state.metrics["responses_total"] += 1

            return response

        except Exception as e:
            # Track error metrics
            if hasattr(request.app.state, "metrics"):
                request.app.state.metrics["errors_total"] += 1

            raise


def setup_error_middleware(app):
    """Setup error handling middleware on FastAPI app"""
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(RequestTrackingMiddleware)

    # Setup global exception handler
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors"""

        # Get context if available
        context = getattr(request.state, "error_context", None)

        if not context:
            context = ErrorContext(
                request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                component="api",
                operation=f"{request.method} {request.url.path}",
            )

        handler = get_error_handler()

        # Handle the error
        if isinstance(exc, HTTPException):
            # Re-raise HTTP exceptions as-is
            raise exc
        else:
            # Handle unexpected errors
            vega_error = handler.create_error(
                code=ErrorCode.INTERNAL_ERROR,
                message=str(exc),
                context=context,
                severity=ErrorSeverity.HIGH,
            )

            handled_error = handler.handle_error(vega_error)

            return JSONResponse(
                status_code=500,
                content={
                    "error_id": handled_error.error_id,
                    "code": handled_error.code.value,
                    "message": handled_error.user_message,
                    "recoverable": handled_error.recoverable,
                },
            )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handler for HTTP exceptions"""

        # Add request ID to error response if available
        request_id = getattr(request.state, "request_id", None)

        content = exc.detail
        if isinstance(content, dict) and request_id:
            content["request_id"] = request_id
        elif request_id:
            content = {
                "message": content if isinstance(content, str) else str(content),
                "request_id": request_id,
            }

        return JSONResponse(status_code=exc.status_code, content=content)
