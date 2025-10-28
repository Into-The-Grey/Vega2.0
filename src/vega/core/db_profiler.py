"""
db_profiler.py - Database Performance Monitoring for Vega2.0

Provides query timing, slow query logging, and connection pool metrics.
Integrates with correlation IDs for request tracing.
"""

from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
from functools import wraps
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution"""

    query: str
    duration_ms: float
    timestamp: datetime
    correlation_id: Optional[str] = None
    error: Optional[str] = None
    row_count: Optional[int] = None


@dataclass
class QueryStats:
    """Aggregated query statistics"""

    total_queries: int = 0
    total_duration_ms: float = 0.0
    slow_queries: int = 0
    failed_queries: int = 0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    recent_queries: deque = field(default_factory=lambda: deque(maxlen=100))
    slow_query_threshold_ms: float = 100.0

    def add_query(self, metrics: QueryMetrics) -> None:
        """Update stats with new query metrics"""
        self.total_queries += 1
        self.total_duration_ms += metrics.duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_queries

        if metrics.duration_ms > self.max_duration_ms:
            self.max_duration_ms = metrics.duration_ms
        if metrics.duration_ms < self.min_duration_ms:
            self.min_duration_ms = metrics.duration_ms

        if metrics.duration_ms >= self.slow_query_threshold_ms:
            self.slow_queries += 1

        if metrics.error:
            self.failed_queries += 1

        self.recent_queries.append(metrics)

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary for API responses"""
        return {
            "total_queries": self.total_queries,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "max_duration_ms": round(self.max_duration_ms, 2),
            "min_duration_ms": (
                round(self.min_duration_ms, 2)
                if self.min_duration_ms != float("inf")
                else 0.0
            ),
            "slow_queries": self.slow_queries,
            "failed_queries": self.failed_queries,
            "slow_query_threshold_ms": self.slow_query_threshold_ms,
            "recent_query_count": len(self.recent_queries),
        }


class DatabaseProfiler:
    """Singleton profiler for database performance monitoring"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.stats = QueryStats()
        self.enabled = True
        self._initialized = True

    def record_query(
        self,
        query: str,
        duration_ms: float,
        correlation_id: Optional[str] = None,
        error: Optional[str] = None,
        row_count: Optional[int] = None,
    ) -> None:
        """Record a query execution"""
        if not self.enabled:
            return

        metrics = QueryMetrics(
            query=query[:200],  # Truncate long queries
            duration_ms=duration_ms,
            timestamp=datetime.utcnow(),
            correlation_id=correlation_id,
            error=error,
            row_count=row_count,
        )

        self.stats.add_query(metrics)

        # Log slow queries
        if duration_ms >= self.stats.slow_query_threshold_ms:
            log_msg = f"SLOW QUERY ({duration_ms:.2f}ms): {query[:100]}"
            if correlation_id:
                log_msg = f"[{correlation_id}] {log_msg}"
            logger.warning(log_msg)

        # Log failed queries
        if error:
            log_msg = f"FAILED QUERY: {query[:100]} | Error: {error}"
            if correlation_id:
                log_msg = f"[{correlation_id}] {log_msg}"
            logger.error(log_msg)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.stats.to_dict()

    def get_recent_queries(self, limit: int = 20) -> list[Dict[str, Any]]:
        """Get recent query details"""
        queries = list(self.stats.recent_queries)[-limit:]
        return [
            {
                "query": q.query,
                "duration_ms": round(q.duration_ms, 2),
                "timestamp": q.timestamp.isoformat(),
                "correlation_id": q.correlation_id,
                "error": q.error,
                "row_count": q.row_count,
            }
            for q in reversed(queries)
        ]

    def get_slow_queries(self, limit: int = 20) -> list[Dict[str, Any]]:
        """Get recent slow queries"""
        slow = [
            q
            for q in self.stats.recent_queries
            if q.duration_ms >= self.stats.slow_query_threshold_ms
        ]
        return [
            {
                "query": q.query,
                "duration_ms": round(q.duration_ms, 2),
                "timestamp": q.timestamp.isoformat(),
                "correlation_id": q.correlation_id,
            }
            for q in reversed(list(slow)[-limit:])
        ]

    def reset_stats(self) -> None:
        """Reset all statistics"""
        self.stats = QueryStats()

    def set_slow_query_threshold(self, threshold_ms: float) -> None:
        """Update slow query threshold"""
        self.stats.slow_query_threshold_ms = threshold_ms
        logger.info(f"Slow query threshold set to {threshold_ms}ms")


# Global profiler instance
_profiler = DatabaseProfiler()


def get_profiler() -> DatabaseProfiler:
    """Get the global database profiler instance"""
    return _profiler


@contextmanager
def profile_query(query_name: str, correlation_id: Optional[str] = None):
    """Context manager for profiling database queries

    Usage:
        with profile_query("get_history", correlation_id):
            result = execute_query(...)
    """
    start = time.perf_counter()
    error = None

    try:
        yield
    except Exception as e:
        error = str(e)
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        _profiler.record_query(
            query=query_name,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
            error=error,
        )


def profile_db_function(func: Callable) -> Callable:
    """Decorator for profiling database functions

    Usage:
        @profile_db_function
        def get_history(limit: int = 50):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get correlation ID from context
        correlation_id = None
        try:
            from .correlation import get_correlation_id

            correlation_id = get_correlation_id()
        except Exception:
            pass

        query_name = f"{func.__module__}.{func.__name__}"
        start = time.perf_counter()
        error = None
        result = None

        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            # Try to determine row count from result
            row_count = None
            if result is not None:
                if isinstance(result, list):
                    row_count = len(result)
                elif isinstance(result, int):
                    row_count = result

            _profiler.record_query(
                query=query_name,
                duration_ms=duration_ms,
                correlation_id=correlation_id,
                error=error,
                row_count=row_count,
            )

    return wrapper


def get_connection_pool_stats() -> Dict[str, Any]:
    """Get SQLAlchemy connection pool statistics

    Returns pool size, connections in use, overflow, etc.
    """
    try:
        from .db import engine

        pool = engine.pool

        return {
            "pool_size": pool.size(),
            "checked_in_connections": pool.checkedin(),
            "checked_out_connections": pool.checkedout(),
            "overflow_connections": pool.overflow(),
            "total_connections": pool.checkedin() + pool.checkedout(),
            "status": "healthy" if pool.checkedout() < pool.size() else "near_capacity",
        }
    except Exception as e:
        logger.error(f"Failed to get connection pool stats: {e}")
        return {
            "error": str(e),
            "status": "unknown",
        }


def get_database_stats() -> Dict[str, Any]:
    """Get comprehensive database statistics including table sizes"""
    try:
        from .db import engine

        with engine.connect() as conn:
            # Get table sizes
            conversations_count = conn.exec_driver_sql(
                "SELECT COUNT(*) FROM conversations"
            ).scalar()

            memories_count = conn.exec_driver_sql(
                "SELECT COUNT(*) FROM memories"
            ).scalar()

            # Get index info
            index_list = conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='conversations'"
            ).fetchall()

            # Get database file size
            import os
            from .db import DB_PATH

            db_size_bytes = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0

            return {
                "conversations_count": conversations_count,
                "memories_count": memories_count,
                "db_size_mb": round(db_size_bytes / (1024 * 1024), 2),
                "indexes": [idx[0] for idx in index_list],
                "status": "healthy",
            }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            "error": str(e),
            "status": "error",
        }
