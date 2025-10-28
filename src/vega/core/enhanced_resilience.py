"""
Enhanced resilience patterns for Vega2.0

Improvements over basic resilience.py:
- Exponential backoff with jitter for circuit breaker recovery
- Half-open state testing before full recovery
- Metrics tracking for circuit breaker state transitions
- Response caching with intelligent cache keys
- Async-safe implementations throughout
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
import logging
from typing import Optional, Any, Dict, Callable, TypeVar, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring"""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0  # Rejected due to open circuit
    state_transitions: Dict[str, int] = field(default_factory=dict)
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    recovery_attempts: int = 0
    recovery_successes: int = 0


class EnhancedCircuitBreaker:
    """
    Enhanced circuit breaker with exponential backoff and jitter.

    Features:
    - Exponential backoff: timeout doubles on repeated failures
    - Jitter: randomization prevents thundering herd
    - Half-open state: test recovery with single request
    - Metrics tracking: comprehensive monitoring
    """

    def __init__(
        self,
        fail_threshold: int = 5,
        base_timeout: float = 30.0,
        max_timeout: float = 300.0,
        half_open_test_count: int = 3,
    ):
        self.fail_threshold = max(1, int(fail_threshold))
        self.base_timeout = float(base_timeout)
        self.max_timeout = float(max_timeout)
        self.half_open_test_count = half_open_test_count

        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._half_open_successes = 0
        self._lock = asyncio.Lock()
        self._metrics = CircuitBreakerMetrics()

    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.

        Raises:
            RuntimeError: If circuit is open
        """
        async with self._lock:
            if not await self._allow_request():
                self._metrics.rejected_requests += 1
                raise RuntimeError(
                    f"Circuit breaker is {self._state.value}, "
                    f"open until {self._open_until - time.time():.1f}s"
                )

            self._metrics.total_requests += 1

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise

    async def _allow_request(self) -> bool:
        """Check if request is allowed based on current state"""
        now = time.time()

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if enough time has passed to try recovery
            if now >= self._open_until:
                await self._transition_to_half_open()
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            # Allow limited requests to test recovery
            return True

        return False

    async def _on_success(self):
        """Handle successful request"""
        async with self._lock:
            self._metrics.successful_requests += 1
            self._metrics.last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_test_count:
                    await self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._fail_count = 0
                self._consecutive_failures = 0

    async def _on_failure(self):
        """Handle failed request"""
        async with self._lock:
            self._metrics.failed_requests += 1
            self._metrics.last_failure_time = time.time()
            self._fail_count += 1
            self._consecutive_failures += 1

            if self._state == CircuitState.HALF_OPEN:
                # Failed recovery test, go back to open
                await self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                # Check if we should open the circuit
                if self._fail_count >= self.fail_threshold:
                    await self._transition_to_open()

    async def _transition_to_open(self):
        """Transition to OPEN state with exponential backoff"""
        if self._state != CircuitState.OPEN:
            self._metrics.state_transitions["closed_to_open"] = (
                self._metrics.state_transitions.get("closed_to_open", 0) + 1
            )

        self._state = CircuitState.OPEN

        # Calculate timeout with exponential backoff and jitter
        backoff_multiplier = min(2 ** (self._consecutive_failures - 1), 64)
        timeout = min(self.base_timeout * backoff_multiplier, self.max_timeout)

        # Add jitter (±20%)
        jitter = timeout * 0.2 * (random.random() * 2 - 1)
        timeout = timeout + jitter

        self._open_until = time.time() + timeout

        logger.warning(
            f"Circuit breaker opened after {self._fail_count} failures. "
            f"Will retry in {timeout:.1f}s (attempt {self._consecutive_failures})"
        )

    async def _transition_to_half_open(self):
        """Transition to HALF_OPEN state to test recovery"""
        self._state = CircuitState.HALF_OPEN
        self._half_open_successes = 0
        self._metrics.recovery_attempts += 1
        self._metrics.state_transitions["open_to_half_open"] = (
            self._metrics.state_transitions.get("open_to_half_open", 0) + 1
        )

        logger.info(
            f"Circuit breaker testing recovery "
            f"(attempt {self._metrics.recovery_attempts})"
        )

    async def _transition_to_closed(self):
        """Transition to CLOSED state after successful recovery"""
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._consecutive_failures = 0
        self._metrics.recovery_successes += 1
        self._metrics.state_transitions["half_open_to_closed"] = (
            self._metrics.state_transitions.get("half_open_to_closed", 0) + 1
        )

        logger.info(
            f"Circuit breaker closed after successful recovery "
            f"(success rate: {self._metrics.recovery_successes}/{self._metrics.recovery_attempts})"
        )

    async def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        async with self._lock:
            now = time.time()
            return {
                "state": self._state.value,
                "fail_count": self._fail_count,
                "consecutive_failures": self._consecutive_failures,
                "open_until": self._open_until,
                "seconds_until_retry": max(0, self._open_until - now),
                "half_open_successes": self._half_open_successes,
                "metrics": {
                    "total_requests": self._metrics.total_requests,
                    "successful_requests": self._metrics.successful_requests,
                    "failed_requests": self._metrics.failed_requests,
                    "rejected_requests": self._metrics.rejected_requests,
                    "success_rate": (
                        self._metrics.successful_requests
                        / max(1, self._metrics.total_requests)
                    ),
                    "recovery_success_rate": (
                        self._metrics.recovery_successes
                        / max(1, self._metrics.recovery_attempts)
                    ),
                    "state_transitions": self._metrics.state_transitions,
                },
            }

    async def reset(self):
        """Reset circuit breaker to closed state"""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._fail_count = 0
            self._consecutive_failures = 0
            self._open_until = 0.0
            self._half_open_successes = 0
            logger.info("Circuit breaker manually reset to CLOSED")


class ResponseCache:
    """
    Async-safe TTL cache for LLM responses.

    Features:
    - Intelligent cache keys based on prompt + model + params
    - TTL-based expiration
    - LRU eviction when full
    - Metrics tracking (hit rate, etc.)
    """

    def __init__(self, ttl_seconds: float = 300.0, maxsize: int = 1000):
        self.ttl = float(ttl_seconds)
        self.maxsize = int(maxsize)
        self._data: Dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _make_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters"""
        # Include key parameters that affect response
        key_params = {
            "prompt": prompt,
            "model": kwargs.get("model", ""),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }

        # Create deterministic hash
        key_str = json.dumps(key_params, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    async def get(self, prompt: str, **kwargs) -> Optional[str]:
        """Get cached response if available and not expired"""
        key = self._make_key(prompt, **kwargs)

        async with self._lock:
            item = self._data.get(key)
            if not item:
                self._misses += 1
                return None

            ts, value = item
            if time.time() - ts > self.ttl:
                # Expired
                self._data.pop(key, None)
                self._misses += 1
                return None

            self._hits += 1
            return value

    async def set(self, prompt: str, response: str, **kwargs):
        """Cache a response"""
        key = self._make_key(prompt, **kwargs)

        async with self._lock:
            # Check if we need to evict
            if len(self._data) >= self.maxsize:
                # Remove oldest entry (LRU)
                oldest_key = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest_key, None)
                self._evictions += 1

            self._data[key] = (time.time(), response)

    async def clear(self):
        """Clear all cached responses"""
        async with self._lock:
            self._data.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(1, total_requests)

            return {
                "size": len(self._data),
                "maxsize": self.maxsize,
                "ttl_seconds": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "total_requests": total_requests,
            }


def circuit_breaker(
    fail_threshold: int = 5,
    base_timeout: float = 30.0,
    max_timeout: float = 300.0,
):
    """
    Decorator to apply circuit breaker pattern to async functions.

    Usage:
        @circuit_breaker(fail_threshold=5, base_timeout=30.0)
        async def my_integration_call():
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        breaker = EnhancedCircuitBreaker(
            fail_threshold=fail_threshold,
            base_timeout=base_timeout,
            max_timeout=max_timeout,
        )

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await breaker.call(func, *args, **kwargs)

        # Attach breaker for inspection
        wrapper._circuit_breaker = breaker  # type: ignore

        return wrapper

    return decorator


def cached_response(ttl_seconds: float = 300.0, maxsize: int = 1000):
    """
    Decorator to cache async function responses.

    Usage:
        @cached_response(ttl_seconds=300)
        async def query_llm(prompt: str, **kwargs):
            ...
    """
    cache = ResponseCache(ttl_seconds=ttl_seconds, maxsize=maxsize)

    def decorator(func: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
        @wraps(func)
        async def wrapper(prompt: str, **kwargs) -> str:
            # Check cache first
            cached = await cache.get(prompt, **kwargs)
            if cached is not None:
                logger.debug(f"Cache hit for prompt (len={len(prompt)})")
                return cached

            # Call function
            result = await func(prompt, **kwargs)

            # Cache result if it's a string
            if isinstance(result, str):
                await cache.set(prompt, result, **kwargs)

            return result

        # Attach cache for inspection
        wrapper._response_cache = cache  # type: ignore

        return wrapper

    return decorator
