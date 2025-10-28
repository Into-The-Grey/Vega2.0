"""
Intelligent Connection Pool Manager for Vega2.0

Advanced connection pool management with:
- Dynamic pool sizing based on load
- Connection health monitoring
- Automatic cleanup of stale connections
- Per-host connection limits
- Connection reuse metrics
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import Dict, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


@dataclass
class ConnectionStats:
    """Statistics for a single connection"""

    created_at: float
    last_used: float
    request_count: int = 0
    error_count: int = 0
    total_active_time: float = 0.0

    @property
    def age_seconds(self) -> float:
        """How long connection has existed"""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """How long since last use"""
        return time.time() - self.last_used

    @property
    def error_rate(self) -> float:
        """Percentage of requests that errored"""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100.0


@dataclass
class PoolMetrics:
    """Metrics for connection pool"""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    connections_created: int = 0
    connections_destroyed: int = 0
    connections_reused: int = 0
    max_concurrent_connections: int = 0

    @property
    def reuse_rate(self) -> float:
        """Percentage of requests that reused existing connections"""
        total = self.connections_created + self.connections_reused
        if total == 0:
            return 0.0
        return (self.connections_reused / total) * 100.0

    @property
    def error_rate(self) -> float:
        """Percentage of requests that errored"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_errors / self.total_requests) * 100.0


class ConnectionPoolManager:
    """
    Manages HTTP connection pools with intelligence.

    Features:
    - Tracks per-connection statistics
    - Automatically closes stale/unhealthy connections
    - Adjusts pool size based on usage patterns
    - Enforces per-host connection limits
    """

    def __init__(
        self,
        max_connections_per_host: int = 10,
        max_total_connections: int = 100,
        connection_ttl: float = 300.0,  # 5 minutes
        idle_timeout: float = 60.0,  # 1 minute
        cleanup_interval: float = 30.0,  # 30 seconds
    ):
        """
        Initialize connection pool manager.

        Args:
            max_connections_per_host: Max connections to single host
            max_total_connections: Max total connections across all hosts
            connection_ttl: Max age of connection before recycling
            idle_timeout: Max idle time before closing connection
            cleanup_interval: How often to run cleanup task
        """
        self._max_per_host = max_connections_per_host
        self._max_total = max_total_connections
        self._connection_ttl = connection_ttl
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval

        # Track connections by host
        self._connections: Dict[str, Set[int]] = defaultdict(set)
        self._connection_stats: Dict[int, ConnectionStats] = {}

        # Metrics
        self._metrics = PoolMetrics()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    def start_monitoring(self):
        """Start background monitoring and cleanup"""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Connection pool monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Connection pool monitoring stopped")

    async def _cleanup_loop(self):
        """Background task to clean up stale connections"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_stale_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")

    async def _cleanup_stale_connections(self):
        """Remove stale or unhealthy connections"""
        now = time.time()
        to_remove = []

        for conn_id, stats in self._connection_stats.items():
            # Remove if too old
            if stats.age_seconds > self._connection_ttl:
                to_remove.append((conn_id, "expired"))
                continue

            # Remove if idle too long
            if stats.idle_seconds > self._idle_timeout:
                to_remove.append((conn_id, "idle"))
                continue

            # Remove if error rate too high
            if stats.request_count >= 10 and stats.error_rate > 50.0:
                to_remove.append((conn_id, "unhealthy"))
                continue

        if to_remove:
            logger.info(
                f"Cleaning up {len(to_remove)} stale connections: "
                f"{', '.join(f'{reason}' for _, reason in to_remove)}"
            )

            for conn_id, reason in to_remove:
                self._remove_connection(conn_id)
                logger.debug(f"Removed connection {conn_id} ({reason})")

    def register_connection(self, conn_id: int, host: str):
        """Register a new connection"""
        if conn_id in self._connection_stats:
            # Already registered, just update usage
            self._connection_stats[conn_id].last_used = time.time()
            self._metrics.connections_reused += 1
            return

        # New connection
        now = time.time()
        self._connection_stats[conn_id] = ConnectionStats(created_at=now, last_used=now)
        self._connections[host].add(conn_id)

        self._metrics.total_connections += 1
        self._metrics.connections_created += 1

        # Track max concurrent
        if self._metrics.total_connections > self._metrics.max_concurrent_connections:
            self._metrics.max_concurrent_connections = self._metrics.total_connections

        logger.debug(
            f"Registered connection {conn_id} to {host} "
            f"(total: {self._metrics.total_connections})"
        )

    def unregister_connection(self, conn_id: int):
        """Unregister a connection (closed by client)"""
        self._remove_connection(conn_id)

    def _remove_connection(self, conn_id: int):
        """Internal: remove connection from tracking"""
        if conn_id not in self._connection_stats:
            return

        # Remove from host tracking
        for host, conn_set in self._connections.items():
            if conn_id in conn_set:
                conn_set.remove(conn_id)
                break

        # Remove stats
        del self._connection_stats[conn_id]

        self._metrics.total_connections -= 1
        self._metrics.connections_destroyed += 1

    def record_request(self, conn_id: int, success: bool = True):
        """Record a request on a connection"""
        if conn_id not in self._connection_stats:
            logger.warning(f"Request on untracked connection {conn_id}")
            return

        stats = self._connection_stats[conn_id]
        stats.request_count += 1
        stats.last_used = time.time()

        self._metrics.total_requests += 1

        if not success:
            stats.error_count += 1
            self._metrics.total_errors += 1

    def can_create_connection(self, host: str) -> bool:
        """Check if we can create a new connection to host"""
        # Check per-host limit
        if len(self._connections.get(host, set())) >= self._max_per_host:
            logger.debug(
                f"Per-host connection limit reached for {host} "
                f"({self._max_per_host})"
            )
            return False

        # Check total limit
        if self._metrics.total_connections >= self._max_total:
            logger.debug(f"Total connection limit reached ({self._max_total})")
            return False

        return True

    def get_host_stats(self, host: str) -> Dict[str, Any]:
        """Get statistics for a specific host"""
        conn_ids = self._connections.get(host, set())

        if not conn_ids:
            return {
                "host": host,
                "connections": 0,
                "total_requests": 0,
                "error_rate": 0.0,
            }

        total_requests = sum(
            self._connection_stats[cid].request_count for cid in conn_ids
        )
        total_errors = sum(self._connection_stats[cid].error_count for cid in conn_ids)

        return {
            "host": host,
            "connections": len(conn_ids),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (
                (total_errors / total_requests * 100.0) if total_requests > 0 else 0.0
            ),
            "oldest_connection_age": max(
                self._connection_stats[cid].age_seconds for cid in conn_ids
            ),
            "newest_connection_age": min(
                self._connection_stats[cid].age_seconds for cid in conn_ids
            ),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "metrics": {
                "total_connections": self._metrics.total_connections,
                "active_connections": self._metrics.active_connections,
                "idle_connections": self._metrics.idle_connections,
                "total_requests": self._metrics.total_requests,
                "total_errors": self._metrics.total_errors,
                "connections_created": self._metrics.connections_created,
                "connections_destroyed": self._metrics.connections_destroyed,
                "connections_reused": self._metrics.connections_reused,
                "max_concurrent_connections": self._metrics.max_concurrent_connections,
                "reuse_rate": self._metrics.reuse_rate,
                "error_rate": self._metrics.error_rate,
            },
            "limits": {
                "max_per_host": self._max_per_host,
                "max_total": self._max_total,
                "connection_ttl": self._connection_ttl,
                "idle_timeout": self._idle_timeout,
            },
            "hosts": {
                host: self.get_host_stats(host) for host in self._connections.keys()
            },
        }

    def get_metrics(self) -> PoolMetrics:
        """Get current metrics"""
        return self._metrics


# Global connection pool manager instance
_pool_manager: Optional[ConnectionPoolManager] = None


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get global connection pool manager instance"""
    global _pool_manager
    if _pool_manager is None:
        _pool_manager = ConnectionPoolManager()
        _pool_manager.start_monitoring()
    return _pool_manager
