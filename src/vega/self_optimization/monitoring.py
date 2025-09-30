"""Performance monitoring utilities for Vega self-optimization."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

LOG_DIR = Path(__file__).resolve().parents[3] / "logs" / "self_optimization"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class MetricRecord:
    """Structure describing a single measurement."""

    task: str
    metric: str
    value: float
    timestamp: float
    context: Dict[str, Any]


class PerformanceMonitor:
    """Centralized recorder that emits JSON lines for historical analysis."""

    def __init__(self, logfile: Optional[Path] = None) -> None:
        self.logfile = logfile or LOG_DIR / "metrics.jsonl"
        self._lock = asyncio.Lock()

    async def record(
        self, task: str, metric: str, value: float, **context: Any
    ) -> None:
        record = MetricRecord(
            task=task,
            metric=metric,
            value=float(value),
            timestamp=time.time(),
            context=context,
        )
        async with self._lock:
            with self.logfile.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record)) + os.linesep)

    def observe_sync(
        self, task: str, metric: str, value: float, **context: Any
    ) -> None:
        """Synchronous variant for non-async clients."""

        record = MetricRecord(
            task=task,
            metric=metric,
            value=float(value),
            timestamp=time.time(),
            context=context,
        )
        with self.logfile.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + os.linesep)

    def monitor(
        self, task_name: str
    ) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
        """Decorator for async functions to measure runtime automatically."""

        def decorator(fn: Callable[..., Awaitable[Any]]):
            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start
                    await self.record(task_name, "duration_sec", duration)

            return wrapper

        return decorator


def monitor_task(
    task_name: str,
    monitor: Optional[PerformanceMonitor] = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Convenience decorator using a shared monitor instance."""

    _monitor = monitor or PerformanceMonitor()
    return _monitor.monitor(task_name)
