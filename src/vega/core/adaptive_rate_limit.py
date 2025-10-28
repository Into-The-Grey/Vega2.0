"""
Adaptive Rate Limiting for Vega2.0

Intelligent rate limiting with adaptive thresholds:
- Per-client rate limiting
- Adaptive thresholds based on system load
- Token bucket algorithm with burst allowance
- Automatic backoff for abusive clients
- Whitelist support for trusted clients
- Detailed metrics and monitoring

This provides lasting value by preventing resource exhaustion and ensuring
fair access under high load conditions.
"""

from __future__ import annotations

import time
import asyncio
import logging
from typing import Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""

    requests_per_second: float = 10.0
    burst_size: int = 20
    adaptive_enabled: bool = True
    adaptive_min_rate: float = 1.0  # Minimum adaptive rate
    adaptive_max_rate: float = 50.0  # Maximum adaptive rate
    ban_threshold: int = 1000  # Requests before temp ban
    ban_duration: float = 300.0  # Ban duration in seconds (5 min)
    whitelist_enabled: bool = True


@dataclass
class ClientState:
    """Per-client rate limit state"""

    tokens: float
    last_update: float
    request_count: int = 0
    violations: int = 0
    banned_until: float = 0.0

    def is_banned(self) -> bool:
        """Check if client is currently banned"""
        return time.time() < self.banned_until


@dataclass
class RateLimitMetrics:
    """Rate limiting metrics"""

    total_requests: int = 0
    allowed_requests: int = 0
    blocked_requests: int = 0
    adaptive_adjustments: int = 0
    bans_issued: int = 0
    whitelist_bypasses: int = 0

    @property
    def block_rate(self) -> float:
        """Percentage of requests blocked"""
        if self.total_requests == 0:
            return 0.0
        return (self.blocked_requests / self.total_requests) * 100.0

    @property
    def allow_rate(self) -> float:
        """Percentage of requests allowed"""
        if self.total_requests == 0:
            return 0.0
        return (self.allowed_requests / self.total_requests) * 100.0


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter using token bucket algorithm.

    Features:
    - Per-client token buckets
    - Adaptive rate adjustment based on system load
    - Temporary bans for abusive clients
    - Whitelist support
    - Burst allowance
    - Detailed metrics
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        """
        Initialize adaptive rate limiter.

        Args:
            config: Rate limiting configuration (uses defaults if None)
        """
        self.config = config or RateLimitConfig()

        # Client state tracking
        self._clients: Dict[str, ClientState] = {}

        # Whitelist (API keys or client IDs that bypass rate limits)
        self._whitelist: Set[str] = set()

        # Metrics
        self._metrics = RateLimitMetrics()

        # System load tracking for adaptive adjustment
        self._load_samples: deque = deque(maxlen=60)  # 60 seconds of samples
        self._current_rate = self.config.requests_per_second

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    def start_monitoring(self):
        """Start background monitoring and cleanup"""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Rate limiter monitoring started")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
        logger.info("Rate limiter monitoring stopped")

    async def _monitoring_loop(self):
        """Background task for cleanup and adaptive adjustment"""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Run every second

                # Cleanup old client states
                await self._cleanup_clients()

                # Adjust rate limits adaptively
                if self.config.adaptive_enabled:
                    await self._adjust_rate_adaptive()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rate limiter monitoring: {e}")

    async def _cleanup_clients(self):
        """Remove stale client states"""
        now = time.time()
        stale_threshold = 3600.0  # Remove clients inactive for 1 hour

        to_remove = [
            client_id
            for client_id, state in self._clients.items()
            if (now - state.last_update) > stale_threshold and not state.is_banned()
        ]

        for client_id in to_remove:
            del self._clients[client_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} stale client states")

    async def _adjust_rate_adaptive(self):
        """Adjust rate limits based on system load"""
        if len(self._load_samples) < 10:
            return  # Need more samples

        # Calculate recent request rate
        recent_rate = sum(self._load_samples) / len(self._load_samples)

        # Adaptive adjustment strategy:
        # - If load is high (> 80% of current rate), decrease rate
        # - If load is low (< 50% of current rate), increase rate

        high_threshold = self._current_rate * 0.8
        low_threshold = self._current_rate * 0.5

        if recent_rate > high_threshold:
            # System under load, reduce rate
            new_rate = max(
                self.config.adaptive_min_rate, self._current_rate * 0.9  # Reduce by 10%
            )
            if new_rate != self._current_rate:
                self._current_rate = new_rate
                self._metrics.adaptive_adjustments += 1
                logger.info(f"Adaptive rate limit decreased to {new_rate:.2f} req/s")

        elif recent_rate < low_threshold:
            # System underutilized, increase rate
            new_rate = min(
                self.config.adaptive_max_rate,
                self._current_rate * 1.1,  # Increase by 10%
            )
            if new_rate != self._current_rate:
                self._current_rate = new_rate
                self._metrics.adaptive_adjustments += 1
                logger.info(f"Adaptive rate limit increased to {new_rate:.2f} req/s")

    def _get_client_id(self, identifier: str) -> str:
        """Get stable client ID from identifier (IP, API key, etc.)"""
        # Hash the identifier for privacy
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def add_to_whitelist(self, identifier: str):
        """Add client to whitelist (bypasses rate limits)"""
        client_id = self._get_client_id(identifier)
        self._whitelist.add(client_id)
        logger.info(f"Added client to whitelist: {client_id}")

    def remove_from_whitelist(self, identifier: str):
        """Remove client from whitelist"""
        client_id = self._get_client_id(identifier)
        if client_id in self._whitelist:
            self._whitelist.discard(client_id)
            logger.info(f"Removed client from whitelist: {client_id}")

    async def check_rate_limit(self, identifier: str) -> Tuple[bool, str]:
        """
        Check if request should be allowed.

        Args:
            identifier: Client identifier (IP address, API key, etc.)

        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        self._metrics.total_requests += 1

        client_id = self._get_client_id(identifier)

        # Check whitelist
        if self.config.whitelist_enabled and client_id in self._whitelist:
            self._metrics.allowed_requests += 1
            self._metrics.whitelist_bypasses += 1
            return (True, "whitelisted")

        # Get or create client state
        if client_id not in self._clients:
            now = time.time()
            self._clients[client_id] = ClientState(
                tokens=float(self.config.burst_size),
                last_update=now,
            )

        state = self._clients[client_id]
        now = time.time()

        # Check if banned
        if state.is_banned():
            self._metrics.blocked_requests += 1
            ban_remaining = state.banned_until - now
            return (False, f"temporarily_banned ({ban_remaining:.0f}s remaining)")

        # Refill tokens based on time elapsed
        time_elapsed = now - state.last_update
        tokens_to_add = time_elapsed * self._current_rate
        state.tokens = min(float(self.config.burst_size), state.tokens + tokens_to_add)
        state.last_update = now

        # Check if tokens available
        if state.tokens >= 1.0:
            # Allow request
            state.tokens -= 1.0
            state.request_count += 1
            self._metrics.allowed_requests += 1

            # Record load sample
            self._load_samples.append(1.0)

            return (True, "allowed")
        else:
            # Rate limit exceeded
            state.violations += 1
            self._metrics.blocked_requests += 1

            # Check if should ban
            if state.violations >= self.config.ban_threshold:
                state.banned_until = now + self.config.ban_duration
                self._metrics.bans_issued += 1
                logger.warning(
                    f"Client {client_id} temporarily banned "
                    f"(violations: {state.violations})"
                )
                return (False, "temporarily_banned (abuse detected)")

            self._load_samples.append(0.0)

            return (False, "rate_limit_exceeded")

    def get_metrics(self) -> Dict[str, any]:
        """Get rate limiter metrics"""
        return {
            "total_requests": self._metrics.total_requests,
            "allowed_requests": self._metrics.allowed_requests,
            "blocked_requests": self._metrics.blocked_requests,
            "allow_rate": self._metrics.allow_rate,
            "block_rate": self._metrics.block_rate,
            "adaptive_adjustments": self._metrics.adaptive_adjustments,
            "bans_issued": self._metrics.bans_issued,
            "whitelist_bypasses": self._metrics.whitelist_bypasses,
            "current_rate_limit": self._current_rate,
            "active_clients": len(self._clients),
            "whitelisted_clients": len(self._whitelist),
            "banned_clients": sum(1 for s in self._clients.values() if s.is_banned()),
        }

    def get_client_info(self, identifier: str) -> Optional[Dict[str, any]]:
        """Get information about a specific client"""
        client_id = self._get_client_id(identifier)

        if client_id not in self._clients:
            return None

        state = self._clients[client_id]

        return {
            "client_id": client_id,
            "tokens_available": state.tokens,
            "request_count": state.request_count,
            "violations": state.violations,
            "is_banned": state.is_banned(),
            "banned_until": state.banned_until if state.is_banned() else None,
            "is_whitelisted": client_id in self._whitelist,
        }


# Global singleton instance
_global_rate_limiter: Optional[AdaptiveRateLimiter] = None
_limiter_lock = asyncio.Lock()


async def get_rate_limiter() -> AdaptiveRateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter

    if _global_rate_limiter is None:
        async with _limiter_lock:
            if _global_rate_limiter is None:
                config = RateLimitConfig(
                    requests_per_second=10.0,
                    burst_size=20,
                    adaptive_enabled=True,
                )
                _global_rate_limiter = AdaptiveRateLimiter(config)
                _global_rate_limiter.start_monitoring()
                logger.info("Global rate limiter initialized")

    return _global_rate_limiter
