"""
db_async.py - Async Database Utilities

Provides async-compatible database access patterns:
- Async session context managers
- Connection pool monitoring
- Query timing decorators
"""

from __future__ import annotations

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, TypeVar, Callable, Any
from functools import wraps

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar("T")


@asynccontextmanager
async def async_session() -> AsyncGenerator[Session, None]:
    """
    Async context manager for database sessions.

    Runs synchronous SQLAlchemy operations in a thread pool
    to avoid blocking the event loop.

    Usage:
        async with async_session() as session:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: session.query(Model).all()
            )
    """
    from .db import engine

    session = Session(engine)
    try:
        yield session
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def run_in_executor(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to run a synchronous function in an executor.

    Usage:
        @run_in_executor
        def blocking_db_operation():
            with Session(engine) as session:
                return session.query(Model).all()

        # Can now be awaited
        result = await blocking_db_operation()
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    return wrapper


class QueryTimer:
    """
    Context manager for timing database queries.

    Usage:
        with QueryTimer("get_users") as timer:
            result = session.query(User).all()
        print(f"Query took {timer.elapsed_ms:.2f}ms")
    """

    def __init__(self, operation_name: str, warn_threshold_ms: float = 100.0):
        self.operation_name = operation_name
        self.warn_threshold_ms = warn_threshold_ms
        self.start_time: float = 0
        self.end_time: float = 0

    def __enter__(self) -> "QueryTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()

        if self.elapsed_ms > self.warn_threshold_ms:
            logger.warning(
                f"Slow query detected: {self.operation_name} took {self.elapsed_ms:.2f}ms "
                f"(threshold: {self.warn_threshold_ms}ms)"
            )

    @property
    def elapsed_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class ConnectionPoolMonitor:
    """
    Monitor database connection pool health.

    Usage:
        monitor = ConnectionPoolMonitor()
        stats = monitor.get_pool_stats()
    """

    def __init__(self):
        from .db import engine

        self.engine = engine

    def get_pool_stats(self) -> dict:
        """Get current connection pool statistics"""
        pool = self.engine.pool

        return {
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
            "invalidated": pool.invalidatedcount() if hasattr(pool, "invalidatedcount") else 0,
        }

    def check_health(self) -> tuple[bool, str]:
        """
        Check pool health.

        Returns:
            Tuple of (is_healthy, message)
        """
        stats = self.get_pool_stats()

        # Warning if more than 80% of pool is checked out
        pool_size = stats.get("pool_size", 10)
        checked_out = stats.get("checked_out", 0)

        utilization = checked_out / pool_size if pool_size > 0 else 0

        if utilization > 0.9:
            return False, f"Connection pool nearly exhausted ({utilization:.0%} utilized)"
        elif utilization > 0.8:
            return True, f"Connection pool high utilization ({utilization:.0%})"
        else:
            return True, f"Connection pool healthy ({utilization:.0%} utilized)"


# Convenience function for common pattern
async def execute_with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    retry_delay: float = 0.5,
) -> T:
    """
    Execute a database function with automatic retry on failure.

    Args:
        func: Synchronous function to execute
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Result of the function

    Raises:
        Last exception if all retries fail
    """
    loop = asyncio.get_running_loop()
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await loop.run_in_executor(None, func)
        except Exception as e:
            last_error = e
            logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))

    raise last_error  # type: ignore
