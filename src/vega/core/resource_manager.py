"""
Resource Manager - Centralized resource lifecycle management for Vega2.0

Manages shared resources to prevent resource leaks and improve performance:
- Shared HTTP client pool (httpx.AsyncClient singleton)
- Configuration caching
- Graceful shutdown coordination
- Resource health monitoring

This module provides lasting value by:
- Reducing connection overhead across all HTTP integrations
- Preventing resource leaks
- Improving startup/shutdown reliability
- Enabling better observability
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from typing import Optional, Dict, Any, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ResourceStats:
    """Statistics for resource usage"""

    http_clients_created: int = 0
    http_requests_made: int = 0
    config_cache_hits: int = 0
    config_cache_misses: int = 0
    startup_time: Optional[float] = None
    shutdown_time: Optional[float] = None
    last_health_check: Optional[datetime] = None


@dataclass
class HTTPClientConfig:
    """Configuration for shared HTTP client"""

    timeout: float = 30.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 5.0
    http2: bool = True
    follow_redirects: bool = True
    verify_ssl: bool = True


class ResourceManager:
    """
    Centralized resource lifecycle manager.

    Provides:
    - Singleton HTTP client pool for all integrations
    - Configuration caching
    - Coordinated shutdown
    - Health monitoring
    """

    _instance: Optional[ResourceManager] = None
    _lock = asyncio.Lock()

    def __init__(self):
        # HTTP client management
        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_config = HTTPClientConfig()
        self._http_client_refs: Set[weakref.ref] = set()

        # Configuration caching
        self._config_cache: Optional[Any] = None
        self._config_cache_time: float = 0.0
        self._config_cache_ttl: float = 300.0  # 5 minutes

        # State tracking
        self._initialized = False
        self._shutting_down = False
        self._stats = ResourceStats()

        # Cleanup registry
        self._cleanup_tasks: list = []

    @classmethod
    async def get_instance(cls) -> ResourceManager:
        """Get singleton instance (thread-safe)"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = ResourceManager()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """Initialize resource manager"""
        if self._initialized:
            return

        start_time = time.time()

        try:
            # Pre-create HTTP client
            await self._ensure_http_client()

            self._initialized = True
            self._stats.startup_time = time.time() - start_time

            logger.info(
                "Resource manager initialized",
                extra={
                    "startup_time_ms": self._stats.startup_time * 1000,
                    "http_client_ready": self._http_client is not None,
                },
            )
        except Exception as e:
            logger.error(f"Resource manager initialization failed: {e}")
            raise

    async def _ensure_http_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client exists (lazy creation with lock)"""
        if self._http_client is not None and not self._http_client.is_closed:
            return self._http_client

        # Create new client
        limits = httpx.Limits(
            max_connections=self._http_config.max_connections,
            max_keepalive_connections=self._http_config.max_keepalive_connections,
            keepalive_expiry=self._http_config.keepalive_expiry,
        )

        timeout = httpx.Timeout(
            timeout=self._http_config.timeout,
            connect=10.0,
            read=30.0,
            write=10.0,
            pool=5.0,
        )

        self._http_client = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
            http2=self._http_config.http2,
            follow_redirects=self._http_config.follow_redirects,
            verify=self._http_config.verify_ssl,
        )

        self._stats.http_clients_created += 1

        logger.info(
            "HTTP client pool created",
            extra={
                "max_connections": self._http_config.max_connections,
                "timeout": self._http_config.timeout,
                "http2_enabled": self._http_config.http2,
            },
        )

        return self._http_client

    @asynccontextmanager
    async def get_http_client(self):
        """
        Get shared HTTP client as context manager.

        Example:
            async with resource_manager.get_http_client() as client:
                response = await client.get("https://example.com")
        """
        client = await self._ensure_http_client()
        self._stats.http_requests_made += 1

        try:
            yield client
        except Exception as e:
            logger.debug(f"HTTP request error: {e}")
            raise

    def get_http_client_direct(self) -> httpx.AsyncClient:
        """
        Get HTTP client directly (for cases where context manager isn't suitable).
        Caller must not close the client.

        Note: Prefer get_http_client() context manager when possible.
        """
        if self._http_client is None:
            # Synchronous fallback - create on first access
            # This is not ideal but handles edge cases
            self._http_client = httpx.AsyncClient(
                timeout=self._http_config.timeout,
                limits=httpx.Limits(
                    max_connections=self._http_config.max_connections,
                    max_keepalive_connections=self._http_config.max_keepalive_connections,
                ),
            )
            self._stats.http_clients_created += 1

        return self._http_client

    def cache_config(self, config: Any, ttl: Optional[float] = None):
        """Cache configuration object"""
        self._config_cache = config
        self._config_cache_time = time.time()
        if ttl is not None:
            self._config_cache_ttl = ttl
        self._stats.config_cache_misses += 1

    def get_cached_config(self) -> Optional[Any]:
        """Get cached configuration if still valid"""
        if self._config_cache is None:
            return None

        age = time.time() - self._config_cache_time
        if age > self._config_cache_ttl:
            # Cache expired
            self._config_cache = None
            return None

        self._stats.config_cache_hits += 1
        return self._config_cache

    def register_cleanup(self, cleanup_fn, name: str = "unnamed"):
        """Register cleanup function to be called on shutdown"""
        self._cleanup_tasks.append((name, cleanup_fn))
        logger.debug(f"Registered cleanup task: {name}")

    async def shutdown(self):
        """Gracefully shutdown all resources"""
        if self._shutting_down:
            return

        self._shutting_down = True
        start_time = time.time()

        logger.info("Resource manager shutdown initiated")

        # Close HTTP client
        if self._http_client is not None and not self._http_client.is_closed:
            try:
                await self._http_client.aclose()
                logger.info("HTTP client closed")
            except Exception as e:
                logger.error(f"Error closing HTTP client: {e}")

        # Run registered cleanup tasks
        for name, cleanup_fn in self._cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(cleanup_fn):
                    await cleanup_fn()
                else:
                    cleanup_fn()
                logger.debug(f"Cleanup task completed: {name}")
            except Exception as e:
                logger.error(f"Cleanup task failed ({name}): {e}")

        # Clear caches
        self._config_cache = None

        self._stats.shutdown_time = time.time() - start_time

        logger.info(
            "Resource manager shutdown complete",
            extra={
                "shutdown_time_ms": self._stats.shutdown_time * 1000,
                "cleanup_tasks": len(self._cleanup_tasks),
            },
        )

        self._initialized = False

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on managed resources"""
        self._stats.last_health_check = datetime.now()

        health = {
            "initialized": self._initialized,
            "shutting_down": self._shutting_down,
            "http_client_ready": (
                self._http_client is not None and not self._http_client.is_closed
            ),
            "config_cached": self._config_cache is not None,
            "cleanup_tasks_registered": len(self._cleanup_tasks),
        }

        return health

    def get_stats(self) -> Dict[str, Any]:
        """Get resource usage statistics"""
        return {
            "http_clients_created": self._stats.http_clients_created,
            "http_requests_made": self._stats.http_requests_made,
            "config_cache_hits": self._stats.config_cache_hits,
            "config_cache_misses": self._stats.config_cache_misses,
            "cache_hit_rate": (
                self._stats.config_cache_hits
                / max(
                    1, self._stats.config_cache_hits + self._stats.config_cache_misses
                )
            ),
            "startup_time_ms": (
                self._stats.startup_time * 1000 if self._stats.startup_time else None
            ),
            "shutdown_time_ms": (
                self._stats.shutdown_time * 1000 if self._stats.shutdown_time else None
            ),
            "last_health_check": (
                self._stats.last_health_check.isoformat()
                if self._stats.last_health_check
                else None
            ),
        }

    def get_pool_metrics(self) -> dict:
        """
        Get detailed HTTP connection pool metrics.

        Returns connection pool state, active connections, and queue status.
        Useful for monitoring pool exhaustion and connection lifecycle.
        """
        metrics = {
            "http_client_available": self._http_client is not None,
            "http_client_closed": False,
            "connection_pool": {
                "max_connections": self._http_config.max_connections,
                "max_keepalive": self._http_config.max_keepalive_connections,
                "keepalive_expiry_seconds": self._http_config.keepalive_expiry,
            },
            "usage_stats": {
                "clients_created": self._stats.http_clients_created,
                "requests_made": self._stats.http_requests_made,
            },
        }

        # Add detailed pool state if client exists
        if self._http_client is not None:
            metrics["http_client_closed"] = self._http_client.is_closed

            # httpx AsyncClient has internal connection pool
            # Access via private _transport (best-effort, may break in future versions)
            try:
                if (
                    hasattr(self._http_client, "_transport")
                    and self._http_client._transport
                ):
                    transport = self._http_client._transport

                    # Try to get connection pool stats
                    if hasattr(transport, "_pool"):
                        pool = transport._pool

                        # Connection pool state (varies by httpx version)
                        pool_state = {
                            "connections_in_pool": (
                                len(pool._connections)
                                if hasattr(pool, "_connections")
                                else "N/A"
                            ),
                            "pool_type": type(pool).__name__,
                        }

                        # Add request queue info if available
                        if hasattr(pool, "_requests"):
                            pool_state["requests_waiting"] = len(pool._requests)

                        metrics["pool_state"] = pool_state
                    else:
                        metrics["pool_state"] = {
                            "note": "Pool details unavailable in this httpx version"
                        }
                else:
                    metrics["pool_state"] = {"note": "Transport not initialized"}
            except Exception as e:
                metrics["pool_state"] = {
                    "error": f"Could not access pool details: {str(e)}"
                }

        return metrics


# Global singleton accessor (async-safe)
async def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance"""
    return await ResourceManager.get_instance()


# Synchronous accessor for compatibility (creates if not exists)
def get_resource_manager_sync() -> Optional[ResourceManager]:
    """
    Get resource manager instance synchronously.
    Returns None if not yet initialized.
    """
    return ResourceManager._instance


# Context manager for resource lifecycle
@asynccontextmanager
async def managed_resources():
    """
    Context manager for resource lifecycle.

    Usage in FastAPI lifespan:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            async with managed_resources():
                yield
    """
    manager = await get_resource_manager()
    try:
        yield manager
    finally:
        await manager.shutdown()
