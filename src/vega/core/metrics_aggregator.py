"""
Comprehensive System Metrics Aggregator for Vega2.0

Provides unified access to all system performance metrics:
- Request coalescing statistics
- Connection pool metrics
- Query cache performance
- Rate limiter statistics
- Circuit breaker status
- Memory and resource usage
- Event loop health
- Database performance

This eliminates the need to query multiple endpoints and reduces
monitoring overhead while improving observability.
"""

from __future__ import annotations

import asyncio
import time
import logging
import psutil
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """Complete system metrics snapshot"""

    timestamp: str
    uptime_seconds: float

    # Resource metrics
    cpu_percent: float
    memory_rss_mb: float
    memory_percent: float

    # Performance metrics
    request_coalescing: Dict[str, Any] = field(default_factory=dict)
    connection_pool: Dict[str, Any] = field(default_factory=dict)
    query_cache: Dict[str, Any] = field(default_factory=dict)
    rate_limiter: Dict[str, Any] = field(default_factory=dict)

    # Integration health
    circuit_breakers: Dict[str, Any] = field(default_factory=dict)

    # Event loop health
    event_loop: Dict[str, Any] = field(default_factory=dict)

    # Database metrics
    database: Dict[str, Any] = field(default_factory=dict)

    # LLM metrics
    llm: Dict[str, Any] = field(default_factory=dict)


class SystemMetricsAggregator:
    """
    Aggregates metrics from all system components.

    Provides a unified view of system health and performance.
    """

    def __init__(self):
        self._start_time = time.time()
        self._process = psutil.Process(os.getpid())
        self._snapshot_history: List[SystemSnapshot] = []
        self._max_history = 100  # Keep last 100 snapshots

    @property
    def uptime_seconds(self) -> float:
        """System uptime in seconds"""
        return time.time() - self._start_time

    async def collect_metrics(self) -> SystemSnapshot:
        """
        Collect comprehensive metrics from all system components.

        Returns:
            Complete system metrics snapshot
        """
        timestamp = datetime.now().isoformat()

        # Collect resource metrics
        try:
            cpu_percent = self._process.cpu_percent(interval=0.1)
            memory_info = self._process.memory_info()
            memory_rss_mb = memory_info.rss / (1024 * 1024)
            memory_percent = self._process.memory_percent()
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            cpu_percent = 0.0
            memory_rss_mb = 0.0
            memory_percent = 0.0

        # Collect performance metrics
        request_coalescing_metrics = await self._collect_request_coalescing()
        connection_pool_metrics = await self._collect_connection_pool()
        query_cache_metrics = await self._collect_query_cache()
        rate_limiter_metrics = await self._collect_rate_limiter()

        # Collect integration health
        circuit_breaker_metrics = await self._collect_circuit_breakers()

        # Collect event loop health
        event_loop_metrics = await self._collect_event_loop()

        # Collect database metrics
        database_metrics = await self._collect_database()

        # Collect LLM metrics
        llm_metrics = await self._collect_llm()

        snapshot = SystemSnapshot(
            timestamp=timestamp,
            uptime_seconds=self.uptime_seconds,
            cpu_percent=cpu_percent,
            memory_rss_mb=memory_rss_mb,
            memory_percent=memory_percent,
            request_coalescing=request_coalescing_metrics,
            connection_pool=connection_pool_metrics,
            query_cache=query_cache_metrics,
            rate_limiter=rate_limiter_metrics,
            circuit_breakers=circuit_breaker_metrics,
            event_loop=event_loop_metrics,
            database=database_metrics,
            llm=llm_metrics,
        )

        # Store in history
        self._snapshot_history.append(snapshot)
        if len(self._snapshot_history) > self._max_history:
            self._snapshot_history.pop(0)

        return snapshot

    async def _collect_request_coalescing(self) -> Dict[str, Any]:
        """Collect request coalescing metrics"""
        try:
            from .request_coalescing import get_llm_coalescer

            coalescer = get_llm_coalescer()
            return coalescer.get_stats()
        except Exception as e:
            logger.debug(f"Could not collect request coalescing metrics: {e}")
            return {"error": str(e)}

    async def _collect_connection_pool(self) -> Dict[str, Any]:
        """Collect connection pool metrics"""
        try:
            from .connection_pool import get_connection_pool_manager

            manager = await get_connection_pool_manager()
            metrics = manager.get_metrics()

            return {
                "total_connections": metrics.total_connections,
                "active_connections": metrics.active_connections,
                "idle_connections": metrics.idle_connections,
                "reuse_rate": metrics.reuse_rate,
                "total_requests": metrics.total_requests,
                "total_errors": metrics.total_errors,
            }
        except Exception as e:
            logger.debug(f"Could not collect connection pool metrics: {e}")
            return {"error": str(e)}

    async def _collect_query_cache(self) -> Dict[str, Any]:
        """Collect query cache metrics"""
        try:
            from .query_cache import get_query_cache

            cache = await get_query_cache()
            return cache.get_metrics()
        except Exception as e:
            logger.debug(f"Could not collect query cache metrics: {e}")
            return {"error": str(e)}

    async def _collect_rate_limiter(self) -> Dict[str, Any]:
        """Collect rate limiter metrics"""
        try:
            from .adaptive_rate_limit import get_rate_limiter

            limiter = await get_rate_limiter()
            return limiter.get_metrics()
        except Exception as e:
            logger.debug(f"Could not collect rate limiter metrics: {e}")
            return {"error": str(e)}

    async def _collect_circuit_breakers(self) -> Dict[str, Any]:
        """Collect circuit breaker status"""
        try:
            from .integration_health import get_all_integration_health

            health_status = await get_all_integration_health()
            return health_status
        except Exception as e:
            logger.debug(f"Could not collect circuit breaker metrics: {e}")
            return {"error": str(e)}

    async def _collect_event_loop(self) -> Dict[str, Any]:
        """Collect event loop health metrics"""
        try:
            from .async_monitor import get_event_loop_monitor

            monitor = await get_event_loop_monitor()
            metrics = await monitor.get_metrics()
            return metrics
        except Exception as e:
            logger.debug(f"Could not collect event loop metrics: {e}")
            return {"error": str(e)}

    async def _collect_database(self) -> Dict[str, Any]:
        """Collect database performance metrics"""
        try:
            from .db_profiler import get_db_profiler

            profiler = get_db_profiler()
            stats = profiler.get_stats()

            return {
                "total_queries": stats.get("total_calls", 0),
                "avg_query_time_ms": stats.get("avg_duration_ms", 0),
                "slowest_query_ms": stats.get("max_duration_ms", 0),
                "total_time_ms": stats.get("total_duration_ms", 0),
            }
        except Exception as e:
            logger.debug(f"Could not collect database metrics: {e}")
            return {"error": str(e)}

    async def _collect_llm(self) -> Dict[str, Any]:
        """Collect LLM usage metrics"""
        try:
            from .llm import get_llm_manager

            manager = await get_llm_manager()

            return {
                "available_providers": manager.get_available_providers(),
                "usage_tracker": manager.usage_tracker,
                "cache_stats": (
                    manager.cache.stats() if hasattr(manager.cache, "stats") else {}
                ),
                "circuit_breaker": (
                    manager.circuit_breaker.status()
                    if hasattr(manager.circuit_breaker, "status")
                    else {}
                ),
            }
        except Exception as e:
            logger.debug(f"Could not collect LLM metrics: {e}")
            return {"error": str(e)}

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get high-level metrics summary"""
        if not self._snapshot_history:
            return {"error": "No metrics collected yet"}

        latest = self._snapshot_history[-1]

        return {
            "timestamp": latest.timestamp,
            "uptime_seconds": latest.uptime_seconds,
            "system": {
                "cpu_percent": latest.cpu_percent,
                "memory_rss_mb": latest.memory_rss_mb,
                "memory_percent": latest.memory_percent,
            },
            "cache_hit_rates": {
                "query_cache": latest.query_cache.get("hit_rate", 0),
                "request_coalescing": latest.request_coalescing.get("coalesce_rate", 0),
            },
            "rate_limiting": {
                "allow_rate": latest.rate_limiter.get("allow_rate", 100),
                "active_clients": latest.rate_limiter.get("active_clients", 0),
            },
            "connections": {
                "total": latest.connection_pool.get("total_connections", 0),
                "reuse_rate": latest.connection_pool.get("reuse_rate", 0),
            },
            "health_status": "healthy",  # Can be computed from various metrics
        }

    def get_trends(self, metric_path: str, samples: int = 10) -> List[float]:
        """
        Get trend data for a specific metric.

        Args:
            metric_path: Dot-separated path to metric (e.g., "cpu_percent", "query_cache.hit_rate")
            samples: Number of recent samples to return

        Returns:
            List of metric values over time
        """
        trends = []

        for snapshot in self._snapshot_history[-samples:]:
            # Navigate to the metric using dot notation
            parts = metric_path.split(".")
            value = snapshot

            try:
                for part in parts:
                    if isinstance(value, dict):
                        value = value[part]
                    else:
                        value = getattr(value, part)

                trends.append(float(value))
            except (KeyError, AttributeError, TypeError):
                trends.append(0.0)

        return trends


# Global singleton
_global_aggregator: Optional[SystemMetricsAggregator] = None
_aggregator_lock = asyncio.Lock()


async def get_metrics_aggregator() -> SystemMetricsAggregator:
    """Get or create global metrics aggregator"""
    global _global_aggregator

    if _global_aggregator is None:
        async with _aggregator_lock:
            if _global_aggregator is None:
                _global_aggregator = SystemMetricsAggregator()
                logger.info("System metrics aggregator initialized")

    return _global_aggregator
