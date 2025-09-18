"""
resilience.py - Circuit breaker and TTL cache utilities
"""

from __future__ import annotations

import time
import threading
from typing import Optional, Callable, Any


class CircuitBreaker:
    def __init__(self, fail_threshold: int, reset_seconds: float):
        self.fail_threshold = max(1, int(fail_threshold))
        self.reset_seconds = float(reset_seconds)
        self.fail_count = 0
        self.open_until = 0.0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            if now < self.open_until:
                return False
            return True

    def on_success(self):
        with self._lock:
            self.fail_count = 0
            self.open_until = 0.0

    def on_failure(self):
        with self._lock:
            self.fail_count += 1
            if self.fail_count >= self.fail_threshold:
                self.open_until = time.time() + self.reset_seconds

    def status(self) -> dict:
        with self._lock:
            return {
                "fail_count": self.fail_count,
                "open_until": self.open_until,
                "is_open": time.time() < self.open_until,
                "fail_threshold": self.fail_threshold,
                "reset_seconds": self.reset_seconds,
            }


class TTLCache:
    def __init__(self, ttl_seconds: float, maxsize: int = 256):
        self.ttl = float(ttl_seconds)
        self.maxsize = int(maxsize)
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            ts, val = item
            if time.time() - ts > self.ttl:
                self._data.pop(key, None)
                return None
            return val

    def set(self, key: str, value: Any):
        with self._lock:
            if len(self._data) >= self.maxsize:
                # remove oldest
                oldest_key = min(self._data, key=lambda k: self._data[k][0])
                self._data.pop(oldest_key, None)
            self._data[key] = (time.time(), value)

    def clear(self):
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._data),
                "ttl": self.ttl,
                "maxsize": self.maxsize,
            }
