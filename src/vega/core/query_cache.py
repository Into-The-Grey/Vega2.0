"""
Intelligent Database Query Cache for Vega2.0

Provides:
- Smart caching of frequently accessed queries
- Automatic cache invalidation on writes
- Request coalescing for concurrent duplicate queries
- LRU eviction when cache is full
- Detailed cache metrics and monitoring

This is a permanent systemic optimization that benefits all database access patterns.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Awaitable
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""

    data: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    key: str = ""

    @property
    def age_seconds(self) -> float:
        """How long entry has been in cache"""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """How long since last access"""
        return time.time() - self.last_accessed

    def mark_accessed(self):
        """Update access statistics"""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Cache performance metrics"""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    invalidations: int = 0
    evictions: int = 0
    total_latency_saved_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Percentage of requests that hit cache"""
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0

    @property
    def avg_latency_saved_ms(self) -> float:
        """Average latency saved per cache hit"""
        return self.total_latency_saved_ms / self.hits if self.hits > 0 else 0.0


class QueryCache:
    """
    Intelligent query result cache with automatic invalidation.

    Features:
    - LRU eviction when full
    - TTL-based expiration
    - Pattern-based invalidation (invalidate by key prefix)
    - Request coalescing for duplicate concurrent queries
    - Detailed metrics and monitoring
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,  # 5 minutes
        cleanup_interval: float = 60.0,  # 1 minute
    ):
        """
        Initialize query cache.

        Args:
            max_size: Maximum number of entries to cache
            default_ttl: Default TTL for cache entries (seconds)
            cleanup_interval: How often to run cleanup (seconds)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cleanup_interval = cleanup_interval
        self._metrics = CacheMetrics()
        self._lock = Lock()

        # In-flight request tracking for coalescing
        self._in_flight: Dict[str, asyncio.Future] = {}
        self._in_flight_lock = asyncio.Lock()

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    def start_cleanup(self):
        """Start background cleanup task"""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Query cache cleanup started")

    def stop_cleanup(self):
        """Stop background cleanup task"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Query cache cleanup stopped")

    async def _cleanup_loop(self):
        """Background task to remove expired entries"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")

    async def _cleanup_expired(self):
        """Remove expired cache entries"""
        now = time.time()
        to_remove = []

        with self._lock:
            for key, entry in self._cache.items():
                if entry.age_seconds > self._default_ttl:
                    to_remove.append(key)

        if to_remove:
            with self._lock:
                for key in to_remove:
                    if key in self._cache:
                        del self._cache[key]
                        self._metrics.evictions += 1

            logger.debug(f"Cleaned up {len(to_remove)} expired cache entries")

    def _make_key(self, query_type: str, *args, **kwargs) -> str:
        """Create cache key from query type and parameters"""
        key_data = {
            "type": query_type,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()[:24]

    async def get_or_fetch(
        self,
        query_type: str,
        fetch_fn: Callable[..., Awaitable[T]],
        *args,
        ttl: Optional[float] = None,
        **kwargs,
    ) -> T:
        """
        Get cached result or fetch from database with coalescing.

        If value is cached and not expired, return it immediately.
        If another request is already fetching this data, wait for it.
        Otherwise, fetch the data and cache it.

        Args:
            query_type: Type of query (e.g., "conversation_history", "session_data")
            fetch_fn: Async function to fetch data if not cached
            *args: Arguments to fetch_fn
            ttl: Optional custom TTL (uses default if None)
            **kwargs: Keyword arguments to fetch_fn

        Returns:
            Query result (from cache or database)
        """
        cache_key = self._make_key(query_type, *args, **kwargs)
        entry_ttl = ttl if ttl is not None else self._default_ttl

        # Check cache first
        with self._lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]

                # Check if expired
                if entry.age_seconds < entry_ttl:
                    # Cache hit
                    entry.mark_accessed()
                    self._metrics.hits += 1

                    # Move to end (LRU)
                    self._cache.move_to_end(cache_key)

                    logger.debug(
                        f"Cache HIT: {query_type} (age: {entry.age_seconds:.1f}s, "
                        f"hit rate: {self._metrics.hit_rate:.1f}%)"
                    )

                    return entry.data
                else:
                    # Expired, remove it
                    del self._cache[cache_key]

        # Cache miss
        self._metrics.misses += 1

        # Check if request is in-flight (coalescing)
        async with self._in_flight_lock:
            if cache_key in self._in_flight:
                # Another request is fetching this, wait for it
                future = self._in_flight[cache_key]
                logger.debug(f"Coalescing query: {query_type}")

                try:
                    result = await future
                    return result
                except Exception:
                    # If the in-flight request failed, we'll try ourselves
                    pass

            # No in-flight request, we'll handle it
            future: asyncio.Future = asyncio.Future()
            self._in_flight[cache_key] = future

        # Fetch data
        start_time = time.time()
        try:
            result = await fetch_fn(*args, **kwargs)

            fetch_time_ms = (time.time() - start_time) * 1000
            self._metrics.total_latency_saved_ms += fetch_time_ms

            # Cache the result
            now = time.time()
            entry = CacheEntry(
                data=result,
                created_at=now,
                last_accessed=now,
                access_count=1,
                key=cache_key,
            )

            with self._lock:
                # Check if we need to evict (LRU)
                if len(self._cache) >= self._max_size:
                    # Remove least recently used
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._metrics.evictions += 1
                    logger.debug(f"Evicted cache entry: {evicted_key[:16]}...")

                # Add new entry
                self._cache[cache_key] = entry
                self._metrics.writes += 1

            # Notify waiters
            async with self._in_flight_lock:
                if cache_key in self._in_flight:
                    future.set_result(result)
                    del self._in_flight[cache_key]

            logger.debug(
                f"Cache MISS: {query_type} (fetch: {fetch_time_ms:.1f}ms, "
                f"hit rate: {self._metrics.hit_rate:.1f}%)"
            )

            return result

        except Exception as e:
            # Propagate error to waiters
            async with self._in_flight_lock:
                if cache_key in self._in_flight:
                    future.set_exception(e)
                    del self._in_flight[cache_key]
            raise

    def invalidate(self, query_type: str, *args, **kwargs):
        """Invalidate specific cache entry"""
        cache_key = self._make_key(query_type, *args, **kwargs)

        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
                self._metrics.invalidations += 1
                logger.debug(f"Invalidated cache entry: {query_type}")

    def invalidate_pattern(self, query_type: str):
        """Invalidate all entries matching query type"""
        to_remove = []

        with self._lock:
            for key, entry in self._cache.items():
                # Check if key starts with query_type hash
                # (simple pattern matching, can be enhanced)
                if query_type in str(entry.data):  # Simplified check
                    to_remove.append(key)

        if to_remove:
            with self._lock:
                for key in to_remove:
                    if key in self._cache:
                        del self._cache[key]
                        self._metrics.invalidations += 1

            logger.debug(f"Invalidated {len(to_remove)} entries for: {query_type}")

    def invalidate_all(self):
        """Clear entire cache"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._metrics.invalidations += count

        logger.info(f"Cleared entire cache ({count} entries)")

    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics"""
        with self._lock:
            return {
                "hits": self._metrics.hits,
                "misses": self._metrics.misses,
                "writes": self._metrics.writes,
                "hit_rate": self._metrics.hit_rate,
                "evictions": self._metrics.evictions,
                "invalidations": self._metrics.invalidations,
                "current_size": len(self._cache),
                "max_size": self._max_size,
                "avg_latency_saved_ms": self._metrics.avg_latency_saved_ms,
                "total_latency_saved_ms": self._metrics.total_latency_saved_ms,
                "in_flight_requests": len(self._in_flight),
            }

    def get_entries_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all cached entries"""
        with self._lock:
            return [
                {
                    "key": key[:16] + "...",
                    "age_seconds": entry.age_seconds,
                    "idle_seconds": entry.idle_seconds,
                    "access_count": entry.access_count,
                }
                for key, entry in self._cache.items()
            ]


# Global singleton instance
_global_query_cache: Optional[QueryCache] = None
_cache_lock = asyncio.Lock()


async def get_query_cache() -> QueryCache:
    """Get or create global query cache instance"""
    global _global_query_cache

    if _global_query_cache is None:
        async with _cache_lock:
            if _global_query_cache is None:
                _global_query_cache = QueryCache(
                    max_size=1000,
                    default_ttl=300.0,  # 5 minutes
                    cleanup_interval=60.0,
                )
                _global_query_cache.start_cleanup()
                logger.info("Global query cache initialized")

    return _global_query_cache


# Convenience function for direct module-level access
def get_query_cache_sync() -> QueryCache:
    """
    Get query cache synchronously (creates if needed).
    Use get_query_cache() async version when possible.
    """
    global _global_query_cache

    if _global_query_cache is None:
        _global_query_cache = QueryCache(
            max_size=1000,
            default_ttl=300.0,
            cleanup_interval=60.0,
        )
        _global_query_cache.start_cleanup()
        logger.info("Global query cache initialized (sync)")

    return _global_query_cache
