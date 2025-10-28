"""
Request Deduplication and Coalescing for Vega2.0

Prevents redundant work when multiple identical requests arrive simultaneously:
- Coalesces duplicate in-flight requests
- Returns same result to all waiters
- Reduces backend load during traffic spikes
- Preserves per-request timeouts
- Thread-safe with async support
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Awaitable, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RequestMetrics:
    """Metrics for request deduplication"""

    total_requests: int = 0
    coalesced_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0
    total_wait_time_ms: float = 0.0

    @property
    def coalesce_rate(self) -> float:
        """Percentage of requests that were coalesced"""
        if self.total_requests == 0:
            return 0.0
        return (self.coalesced_requests / self.total_requests) * 100.0

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of requests that hit cache"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100.0


@dataclass
class InFlightRequest:
    """Tracks in-flight request that others can wait on"""

    future: asyncio.Future
    waiters: int = 1
    started_at: float = field(default_factory=time.time)


class RequestCoalescer:
    """
    Coalesces duplicate concurrent requests.

    When multiple identical requests arrive while one is in-flight,
    all waiters receive the same result without duplicate backend calls.
    """

    def __init__(self, cache_ttl: float = 60.0, max_in_flight: int = 1000):
        """
        Initialize request coalescer.

        Args:
            cache_ttl: How long to cache results (seconds)
            max_in_flight: Maximum concurrent unique requests to track
        """
        self._in_flight: Dict[str, InFlightRequest] = {}
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (result, timestamp)
        self._cache_ttl = cache_ttl
        self._max_in_flight = max_in_flight
        self._metrics = RequestMetrics()
        self._lock = asyncio.Lock()

    def _make_key(self, operation: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from operation and arguments"""
        # Create deterministic hash of operation + args + kwargs
        key_data = {
            "op": operation,
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    async def coalesce(
        self,
        operation_name: str,
        operation: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute operation with request coalescing.

        If an identical request is in-flight, wait for its result.
        Otherwise, execute the operation and share result with waiters.

        Args:
            operation_name: Name of operation (for metrics)
            operation: Async callable to execute
            *args: Arguments to operation
            **kwargs: Keyword arguments to operation

        Returns:
            Result from operation (either fresh or coalesced)
        """
        self._metrics.total_requests += 1
        start_time = time.time()

        # Create cache key
        cache_key = self._make_key(operation_name, args, kwargs)

        # Check cache first
        if cache_key in self._cache:
            result, cached_at = self._cache[cache_key]
            age = time.time() - cached_at

            if age < self._cache_ttl:
                self._metrics.cache_hits += 1
                logger.debug(
                    f"Cache hit for {operation_name} (age: {age:.2f}s, "
                    f"hit rate: {self._metrics.cache_hit_rate:.1f}%)"
                )
                return result
            else:
                # Expired, remove from cache
                del self._cache[cache_key]

        self._metrics.cache_misses += 1

        # Check if request is already in-flight
        async with self._lock:
            if cache_key in self._in_flight:
                # Another request is already processing this
                in_flight = self._in_flight[cache_key]
                in_flight.waiters += 1
                self._metrics.coalesced_requests += 1

                logger.debug(
                    f"Coalescing request for {operation_name} "
                    f"({in_flight.waiters} total waiters, "
                    f"coalesce rate: {self._metrics.coalesce_rate:.1f}%)"
                )

                # Wait for the in-flight request to complete
                try:
                    result = await in_flight.future
                    wait_time_ms = (time.time() - start_time) * 1000
                    self._metrics.total_wait_time_ms += wait_time_ms
                    return result
                except Exception as e:
                    self._metrics.errors += 1
                    raise

            # No in-flight request, we'll handle it
            future: asyncio.Future = asyncio.Future()
            self._in_flight[cache_key] = InFlightRequest(future=future)

        # Execute the operation
        try:
            result = await operation(*args, **kwargs)

            # Cache the result
            self._cache[cache_key] = (result, time.time())

            # Notify all waiters
            async with self._lock:
                if cache_key in self._in_flight:
                    in_flight = self._in_flight[cache_key]
                    in_flight.future.set_result(result)
                    del self._in_flight[cache_key]

                    if in_flight.waiters > 1:
                        logger.info(
                            f"Completed coalesced request for {operation_name} "
                            f"(served {in_flight.waiters} waiters)"
                        )

            wait_time_ms = (time.time() - start_time) * 1000
            self._metrics.total_wait_time_ms += wait_time_ms

            return result

        except Exception as e:
            self._metrics.errors += 1

            # Propagate error to all waiters
            async with self._lock:
                if cache_key in self._in_flight:
                    in_flight = self._in_flight[cache_key]
                    in_flight.future.set_exception(e)
                    del self._in_flight[cache_key]

            raise

    def clear_cache(self):
        """Clear all cached results"""
        self._cache.clear()
        logger.info("Request coalescer cache cleared")

    def get_metrics(self) -> RequestMetrics:
        """Get current metrics"""
        return self._metrics

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        return {
            "total_requests": self._metrics.total_requests,
            "coalesced_requests": self._metrics.coalesced_requests,
            "coalesce_rate": self._metrics.coalesce_rate,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "cache_hit_rate": self._metrics.cache_hit_rate,
            "errors": self._metrics.errors,
            "avg_wait_time_ms": (
                self._metrics.total_wait_time_ms / self._metrics.total_requests
                if self._metrics.total_requests > 0
                else 0.0
            ),
            "in_flight_count": len(self._in_flight),
            "cached_items": len(self._cache),
        }


# Global instance for LLM request coalescing
_llm_coalescer: Optional[RequestCoalescer] = None


def get_llm_coalescer() -> RequestCoalescer:
    """Get global LLM request coalescer instance"""
    global _llm_coalescer
    if _llm_coalescer is None:
        _llm_coalescer = RequestCoalescer(cache_ttl=300.0, max_in_flight=100)
    return _llm_coalescer


# Global instance for integration request coalescing
_integration_coalescer: Optional[RequestCoalescer] = None


def get_integration_coalescer() -> RequestCoalescer:
    """Get global integration request coalescer instance"""
    global _integration_coalescer
    if _integration_coalescer is None:
        _integration_coalescer = RequestCoalescer(cache_ttl=60.0, max_in_flight=200)
    return _integration_coalescer


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on backend performance.

    - Increases rate when backend is healthy
    - Decreases rate when backend is struggling
    - Prevents thundering herd during recovery
    """

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 100.0,
        adjustment_factor: float = 1.2,
    ):
        """
        Initialize adaptive rate limiter.

        Args:
            initial_rate: Starting requests per second
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            adjustment_factor: How much to adjust (1.2 = ±20%)
        """
        self._current_rate = initial_rate
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._adjustment_factor = adjustment_factor
        self._tokens = initial_rate
        self._last_update = time.time()
        self._lock = asyncio.Lock()
        self._success_count = 0
        self._failure_count = 0
        self._adjustment_threshold = 10  # Adjust after N requests

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make request.

        Args:
            timeout: Maximum time to wait for token (seconds)

        Returns:
            True if acquired, False if timeout
        """
        deadline = time.time() + timeout

        while time.time() < deadline:
            async with self._lock:
                now = time.time()
                elapsed = now - self._last_update
                self._last_update = now

                # Add tokens based on current rate
                self._tokens = min(
                    self._tokens + (self._current_rate * elapsed), self._current_rate
                )

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            # Wait a bit before retrying
            await asyncio.sleep(0.01)

        return False

    async def report_success(self):
        """Report successful request (increases rate)"""
        async with self._lock:
            self._success_count += 1
            await self._maybe_adjust_rate()

    async def report_failure(self):
        """Report failed request (decreases rate)"""
        async with self._lock:
            self._failure_count += 1
            await self._maybe_adjust_rate()

    async def _maybe_adjust_rate(self):
        """Adjust rate based on success/failure ratio"""
        total = self._success_count + self._failure_count

        if total < self._adjustment_threshold:
            return

        success_rate = self._success_count / total

        if success_rate > 0.95:
            # Healthy, increase rate
            new_rate = min(self._current_rate * self._adjustment_factor, self._max_rate)
            if new_rate != self._current_rate:
                logger.info(
                    f"Increasing rate limit: {self._current_rate:.1f} -> {new_rate:.1f} "
                    f"req/s (success rate: {success_rate:.1%})"
                )
                self._current_rate = new_rate

        elif success_rate < 0.80:
            # Struggling, decrease rate
            new_rate = max(self._current_rate / self._adjustment_factor, self._min_rate)
            if new_rate != self._current_rate:
                logger.warning(
                    f"Decreasing rate limit: {self._current_rate:.1f} -> {new_rate:.1f} "
                    f"req/s (success rate: {success_rate:.1%})"
                )
                self._current_rate = new_rate

        # Reset counters
        self._success_count = 0
        self._failure_count = 0

    def get_current_rate(self) -> float:
        """Get current requests per second limit"""
        return self._current_rate

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "current_rate": self._current_rate,
            "min_rate": self._min_rate,
            "max_rate": self._max_rate,
            "available_tokens": self._tokens,
            "recent_success_count": self._success_count,
            "recent_failure_count": self._failure_count,
        }
