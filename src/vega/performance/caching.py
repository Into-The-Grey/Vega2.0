"""
Advanced Caching Layer

Provides multi-level caching with Redis integration, intelligent cache
invalidation, and local optimization for personal use.
"""

import asyncio
import json
import logging
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from functools import wraps
import aiofiles
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache levels in order of speed"""

    MEMORY = "memory"  # In-process memory cache
    REDIS = "redis"  # Redis cache
    DISK = "disk"  # Local disk cache
    DATABASE = "database"  # SQLite cache database


class CachePolicy(Enum):
    """Cache eviction policies"""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata"""

    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int]
    size_bytes: int
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds"""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class CacheStats:
    """Cache statistics"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    last_updated: datetime = None

    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()

    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryCache:
    """In-memory cache implementation"""

    def __init__(
        self, max_size_mb: int = 100, max_entries: int = 10000, default_ttl: int = 3600
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.entries: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache"""
        if key not in self.entries:
            self.stats.misses += 1
            return None

        entry = self.entries[key]

        if entry.is_expired:
            del self.entries[key]
            self.stats.misses += 1
            self.stats.evictions += 1
            return None

        # Update access statistics
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        self.stats.hits += 1

        return entry.value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache"""
        try:
            # Calculate size
            serialized = pickle.dumps(value)
            size_bytes = len(serialized)

            # Check if we need to evict
            await self._ensure_space(size_bytes)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                ttl_seconds=ttl or self.default_ttl,
                size_bytes=size_bytes,
            )

            self.entries[key] = entry
            self.stats.entry_count = len(self.entries)
            self.stats.size_bytes += size_bytes

            return True

        except Exception as e:
            logger.error(f"Failed to set memory cache entry {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from memory cache"""
        if key in self.entries:
            entry = self.entries[key]
            self.stats.size_bytes -= entry.size_bytes
            del self.entries[key]
            self.stats.entry_count = len(self.entries)
            return True
        return False

    async def clear(self):
        """Clear all entries"""
        self.entries.clear()
        self.stats = CacheStats()

    async def _ensure_space(self, needed_bytes: int):
        """Ensure enough space for new entry"""
        while (
            self.stats.size_bytes + needed_bytes > self.max_size_bytes
            or len(self.entries) >= self.max_entries
        ):

            # Find LRU entry
            lru_key = min(
                self.entries.keys(), key=lambda k: self.entries[k].last_accessed
            )

            entry = self.entries[lru_key]
            self.stats.size_bytes -= entry.size_bytes
            del self.entries[lru_key]
            self.stats.evictions += 1

            if not self.entries:
                break


class RedisCache:
    """Redis-based cache implementation"""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "vega:cache:",
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.redis_client: Optional[redis.Redis] = None
        self.stats = CacheStats()

    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            self.stats.misses += 1
            return None

        try:
            full_key = f"{self.key_prefix}{key}"
            data = await self.redis_client.get(full_key)

            if data is None:
                self.stats.misses += 1
                return None

            # Deserialize and update stats
            value = pickle.loads(data)
            self.stats.hits += 1

            # Update access time
            await self.redis_client.hset(
                f"{full_key}:meta", "last_accessed", datetime.now().isoformat()
            )

            return value

        except Exception as e:
            logger.error(f"Failed to get Redis cache entry {key}: {e}")
            self.stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False

        try:
            full_key = f"{self.key_prefix}{key}"
            ttl_seconds = ttl or self.default_ttl

            # Serialize and store
            data = pickle.dumps(value)
            await self.redis_client.setex(full_key, ttl_seconds, data)

            # Store metadata
            metadata = {
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size_bytes": len(data),
                "ttl_seconds": ttl_seconds,
            }

            await self.redis_client.hset(f"{full_key}:meta", mapping=metadata)
            await self.redis_client.expire(f"{full_key}:meta", ttl_seconds)

            return True

        except Exception as e:
            logger.error(f"Failed to set Redis cache entry {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from Redis cache"""
        if not self.redis_client:
            return False

        try:
            full_key = f"{self.key_prefix}{key}"
            result = await self.redis_client.delete(full_key, f"{full_key}:meta")
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete Redis cache entry {key}: {e}")
            return False

    async def clear(self):
        """Clear all cache entries"""
        if not self.redis_client:
            return

        try:
            pattern = f"{self.key_prefix}*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)

        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()


class DiskCache:
    """Disk-based cache implementation"""

    def __init__(
        self,
        cache_dir: str = "data/cache",
        max_size_mb: int = 1000,
        default_ttl: int = 86400,
    ):  # 24 hours default
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.stats = CacheStats()

        # Create cache directory
        import os

        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_file_path(self, key: str) -> str:
        """Get file path for cache key"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return f"{self.cache_dir}/{key_hash}.cache"

    def _get_meta_path(self, key: str) -> str:
        """Get metadata file path for cache key"""
        # Hash key to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return f"{self.cache_dir}/{key_hash}.meta"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache"""
        try:
            file_path = self._get_file_path(key)
            meta_path = self._get_meta_path(key)

            # Check if files exist
            import os

            if not os.path.exists(file_path) or not os.path.exists(meta_path):
                self.stats.misses += 1
                return None

            # Load metadata
            async with aiofiles.open(meta_path, "r") as f:
                metadata = json.loads(await f.read())

            # Check if expired
            created_at = datetime.fromisoformat(metadata["created_at"])
            ttl_seconds = metadata["ttl_seconds"]
            if (
                ttl_seconds
                and (datetime.now() - created_at).total_seconds() > ttl_seconds
            ):
                # Remove expired files
                os.remove(file_path)
                os.remove(meta_path)
                self.stats.misses += 1
                self.stats.evictions += 1
                return None

            # Load value
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
                value = pickle.loads(data)

            # Update access time
            metadata["last_accessed"] = datetime.now().isoformat()
            async with aiofiles.open(meta_path, "w") as f:
                await f.write(json.dumps(metadata))

            self.stats.hits += 1
            return value

        except Exception as e:
            logger.error(f"Failed to get disk cache entry {key}: {e}")
            self.stats.misses += 1
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache"""
        try:
            file_path = self._get_file_path(key)
            meta_path = self._get_meta_path(key)

            # Serialize value
            data = pickle.dumps(value)

            # Check space
            await self._ensure_space(len(data))

            # Write value
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(data)

            # Write metadata
            metadata = {
                "key": key,
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "size_bytes": len(data),
                "ttl_seconds": ttl or self.default_ttl,
            }

            async with aiofiles.open(meta_path, "w") as f:
                await f.write(json.dumps(metadata))

            return True

        except Exception as e:
            logger.error(f"Failed to set disk cache entry {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from disk cache"""
        try:
            file_path = self._get_file_path(key)
            meta_path = self._get_meta_path(key)

            import os

            deleted = False

            if os.path.exists(file_path):
                os.remove(file_path)
                deleted = True

            if os.path.exists(meta_path):
                os.remove(meta_path)
                deleted = True

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete disk cache entry {key}: {e}")
            return False

    async def clear(self):
        """Clear all cache entries"""
        try:
            import os
            import glob

            cache_files = glob.glob(f"{self.cache_dir}/*.cache")
            meta_files = glob.glob(f"{self.cache_dir}/*.meta")

            for file_path in cache_files + meta_files:
                os.remove(file_path)

        except Exception as e:
            logger.error(f"Failed to clear disk cache: {e}")

    async def _ensure_space(self, needed_bytes: int):
        """Ensure enough space for new entry"""
        try:
            import os
            import glob

            # Calculate current size
            total_size = 0
            files_info = []

            for file_path in glob.glob(f"{self.cache_dir}/*.cache"):
                stat = os.stat(file_path)
                total_size += stat.st_size
                files_info.append((file_path, stat.st_mtime, stat.st_size))

            # Remove oldest files if needed
            while total_size + needed_bytes > self.max_size_bytes and files_info:
                # Sort by modification time (oldest first)
                files_info.sort(key=lambda x: x[1])

                oldest_file, _, file_size = files_info.pop(0)
                meta_file = oldest_file.replace(".cache", ".meta")

                os.remove(oldest_file)
                if os.path.exists(meta_file):
                    os.remove(meta_file)

                total_size -= file_size
                self.stats.evictions += 1

        except Exception as e:
            logger.error(f"Failed to ensure disk cache space: {e}")


class MultiLevelCache:
    """
    Multi-level cache system with intelligent cache management
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}

        # Initialize cache levels
        self.memory_cache = MemoryCache(
            max_size_mb=config.get("memory_max_size_mb", 100),
            max_entries=config.get("memory_max_entries", 10000),
            default_ttl=config.get("memory_default_ttl", 3600),
        )

        self.redis_cache = RedisCache(
            redis_url=config.get("redis_url", "redis://localhost:6379/0"),
            default_ttl=config.get("redis_default_ttl", 3600),
            key_prefix=config.get("redis_key_prefix", "vega:cache:"),
        )

        self.disk_cache = DiskCache(
            cache_dir=config.get("disk_cache_dir", "data/cache"),
            max_size_mb=config.get("disk_max_size_mb", 1000),
            default_ttl=config.get("disk_default_ttl", 86400),
        )

        self.enabled_levels = config.get(
            "enabled_levels",
            [CacheLevel.MEMORY.value, CacheLevel.REDIS.value, CacheLevel.DISK.value],
        )

        self.global_stats = CacheStats()

    async def initialize(self):
        """Initialize cache system"""
        if CacheLevel.REDIS.value in self.enabled_levels:
            await self.redis_cache.connect()

        logger.info(f"Multi-level cache initialized with levels: {self.enabled_levels}")

    async def get(
        self, key: str, levels: Optional[List[CacheLevel]] = None
    ) -> Optional[Any]:
        """Get value from cache, checking levels in order"""

        if levels is None:
            levels = [CacheLevel(level) for level in self.enabled_levels]

        value = None
        found_level = None

        # Check each level in order
        for level in levels:
            if (
                level == CacheLevel.MEMORY
                and CacheLevel.MEMORY.value in self.enabled_levels
            ):
                value = await self.memory_cache.get(key)
                if value is not None:
                    found_level = level
                    break

            elif (
                level == CacheLevel.REDIS
                and CacheLevel.REDIS.value in self.enabled_levels
            ):
                value = await self.redis_cache.get(key)
                if value is not None:
                    found_level = level
                    break

            elif (
                level == CacheLevel.DISK
                and CacheLevel.DISK.value in self.enabled_levels
            ):
                value = await self.disk_cache.get(key)
                if value is not None:
                    found_level = level
                    break

        if value is not None:
            self.global_stats.hits += 1

            # Promote to higher levels (cache warming)
            await self._promote_cache_entry(key, value, found_level, levels)
        else:
            self.global_stats.misses += 1

        return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        levels: Optional[List[CacheLevel]] = None,
    ) -> bool:
        """Set value in specified cache levels"""

        if levels is None:
            levels = [CacheLevel(level) for level in self.enabled_levels]

        success = False

        for level in levels:
            if (
                level == CacheLevel.MEMORY
                and CacheLevel.MEMORY.value in self.enabled_levels
            ):
                if await self.memory_cache.set(key, value, ttl):
                    success = True

            elif (
                level == CacheLevel.REDIS
                and CacheLevel.REDIS.value in self.enabled_levels
            ):
                if await self.redis_cache.set(key, value, ttl):
                    success = True

            elif (
                level == CacheLevel.DISK
                and CacheLevel.DISK.value in self.enabled_levels
            ):
                if await self.disk_cache.set(key, value, ttl):
                    success = True

        return success

    async def delete(self, key: str, levels: Optional[List[CacheLevel]] = None) -> bool:
        """Delete entry from specified cache levels"""

        if levels is None:
            levels = [CacheLevel(level) for level in self.enabled_levels]

        success = False

        for level in levels:
            if (
                level == CacheLevel.MEMORY
                and CacheLevel.MEMORY.value in self.enabled_levels
            ):
                if await self.memory_cache.delete(key):
                    success = True

            elif (
                level == CacheLevel.REDIS
                and CacheLevel.REDIS.value in self.enabled_levels
            ):
                if await self.redis_cache.delete(key):
                    success = True

            elif (
                level == CacheLevel.DISK
                and CacheLevel.DISK.value in self.enabled_levels
            ):
                if await self.disk_cache.delete(key):
                    success = True

        return success

    async def _promote_cache_entry(
        self,
        key: str,
        value: Any,
        found_level: CacheLevel,
        target_levels: List[CacheLevel],
    ):
        """Promote cache entry to higher levels"""

        # Find levels above the found level
        level_order = [CacheLevel.MEMORY, CacheLevel.REDIS, CacheLevel.DISK]
        found_index = level_order.index(found_level)

        for level in level_order[:found_index]:
            if level in target_levels:
                await self.set(key, value, levels=[level])

    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated pattern matching
        logger.info(f"Invalidating cache entries matching pattern: {pattern}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "global": asdict(self.global_stats),
            "memory": (
                asdict(self.memory_cache.stats)
                if CacheLevel.MEMORY.value in self.enabled_levels
                else None
            ),
            "redis": (
                asdict(self.redis_cache.stats)
                if CacheLevel.REDIS.value in self.enabled_levels
                else None
            ),
            "disk": (
                asdict(self.disk_cache.stats)
                if CacheLevel.DISK.value in self.enabled_levels
                else None
            ),
            "enabled_levels": self.enabled_levels,
            "timestamp": datetime.now().isoformat(),
        }
        return stats

    async def clear_all(self):
        """Clear all cache levels"""
        if CacheLevel.MEMORY.value in self.enabled_levels:
            await self.memory_cache.clear()

        if CacheLevel.REDIS.value in self.enabled_levels:
            await self.redis_cache.clear()

        if CacheLevel.DISK.value in self.enabled_levels:
            await self.disk_cache.clear()

        self.global_stats = CacheStats()

    async def close(self):
        """Close cache connections"""
        if CacheLevel.REDIS.value in self.enabled_levels:
            await self.redis_cache.close()


# Cache decorators
def cached(ttl: int = 3600, cache_levels: Optional[List[CacheLevel]] = None):
    """Decorator to cache function results"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache from context (would be injected in real app)
            cache = kwargs.pop("_cache", None)
            if not cache:
                return await func(*args, **kwargs)

            # Generate cache key
            key_data = {"function": func.__name__, "args": args, "kwargs": kwargs}
            cache_key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()

            # Try to get from cache
            result = await cache.get(cache_key, cache_levels)
            if result is not None:
                return result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl, cache_levels)

            return result

        return wrapper

    return decorator


# Demo and testing functions
async def demo_advanced_caching():
    """Demonstrate advanced caching capabilities"""

    # Initialize cache system
    cache_config = {
        "memory_max_size_mb": 50,
        "memory_max_entries": 1000,
        "redis_url": "redis://localhost:6379/0",
        "disk_cache_dir": "data/demo_cache",
        "enabled_levels": ["memory", "disk"],  # Skip Redis for demo
    }

    cache = MultiLevelCache(cache_config)
    await cache.initialize()

    print("Advanced Caching Layer Demo")

    # Test basic operations
    test_data = {
        "document_123": "This is a test document content",
        "user_preferences": {"theme": "dark", "language": "en"},
        "computation_result": [1, 2, 3, 4, 5],
        "model_prediction": {"confidence": 0.95, "prediction": "positive"},
    }

    print("\nSetting cache entries...")
    for key, value in test_data.items():
        success = await cache.set(key, value, ttl=3600)
        print(f"Set {key}: {'✓' if success else '✗'}")

    print("\nGetting cache entries...")
    for key in test_data.keys():
        value = await cache.get(key)
        print(f"Get {key}: {'✓' if value is not None else '✗'}")

    # Test cache miss
    missing_value = await cache.get("nonexistent_key")
    print(f"Get nonexistent key: {'✗' if missing_value is None else '✓'}")

    # Get statistics
    stats = await cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"- Global hits: {stats['global']['hits']}")
    print(f"- Global misses: {stats['global']['misses']}")
    print(f"- Hit ratio: {stats['global']['hit_ratio']:.2%}")

    if stats["memory"]:
        print(f"- Memory entries: {stats['memory']['entry_count']}")
        print(f"- Memory size: {stats['memory']['size_bytes']} bytes")

    # Test cached function decorator
    @cached(ttl=1800)
    async def expensive_computation(n: int, _cache=None) -> int:
        """Simulate expensive computation"""
        await asyncio.sleep(0.1)  # Simulate work
        return n * n * n

    print("\nTesting cached function...")

    # First call (cache miss)
    start_time = time.time()
    result1 = await expensive_computation(5, _cache=cache)
    time1 = time.time() - start_time
    print(f"First call result: {result1} (took {time1:.3f}s)")

    # Second call (cache hit)
    start_time = time.time()
    result2 = await expensive_computation(5, _cache=cache)
    time2 = time.time() - start_time
    print(f"Second call result: {result2} (took {time2:.3f}s)")

    print(f"Speedup: {time1/time2:.1f}x faster")

    await cache.close()
    return cache


if __name__ == "__main__":
    asyncio.run(demo_advanced_caching())
