"""
Optimized Configuration Management

Wraps config.py with intelligent caching to reduce repeated file I/O and parsing.
This provides lasting value by:
- Reducing configuration load overhead
- Supporting config hot-reload when needed
- Maintaining backward compatibility
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

# Cache state
_config_cache: Optional[Any] = None
_cache_timestamp: float = 0.0
_cache_ttl: float = 300.0  # 5 minutes default
_cache_lock = threading.Lock()
_cache_hits: int = 0
_cache_misses: int = 0


def get_config_cached(ttl: Optional[float] = None, force_reload: bool = False):
    """
    Get configuration with intelligent caching.

    Args:
        ttl: Cache TTL in seconds (None = use default)
        force_reload: Force reload from disk

    Returns:
        Config object (same as config.get_config())
    """
    global _config_cache, _cache_timestamp, _cache_hits, _cache_misses

    with _cache_lock:
        now = time.time()
        cache_age = now - _cache_timestamp
        effective_ttl = ttl if ttl is not None else _cache_ttl

        # Check if cache is valid
        if not force_reload and _config_cache is not None and cache_age < effective_ttl:
            _cache_hits += 1
            return _config_cache

        # Load fresh config
        try:
            from .config import get_config as _get_config_impl

            _config_cache = _get_config_impl()
            _cache_timestamp = now
            _cache_misses += 1

            logger.debug(
                f"Config loaded (cache age: {cache_age:.2f}s, "
                f"hits: {_cache_hits}, misses: {_cache_misses})"
            )

            return _config_cache

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Return stale cache if available, otherwise raise
            if _config_cache is not None:
                logger.warning("Returning stale config due to load failure")
                _cache_hits += 1
                return _config_cache
            raise


def invalidate_config_cache():
    """Invalidate config cache (force reload on next access)"""
    global _config_cache, _cache_timestamp

    with _cache_lock:
        _config_cache = None
        _cache_timestamp = 0.0
        logger.info("Config cache invalidated")


def get_cache_stats() -> dict:
    """Get cache statistics"""
    with _cache_lock:
        total = _cache_hits + _cache_misses
        hit_rate = (_cache_hits / total * 100) if total > 0 else 0.0

        return {
            "hits": _cache_hits,
            "misses": _cache_misses,
            "total_accesses": total,
            "hit_rate_percent": hit_rate,
            "cache_age_seconds": time.time() - _cache_timestamp,
            "cached": _config_cache is not None,
        }
