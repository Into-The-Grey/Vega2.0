"""
Async Event Loop Monitor for Vega2.0

Detects and reports async anti-patterns:
- Blocking operations in async functions
- Slow callbacks (>100ms)
- Excessive pending tasks
- Event loop stalls
- CPU-bound operations in event loop
"""

from __future__ import annotations

import asyncio
import functools
import time
import logging
import threading
import traceback
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SlowCallback:
    """Record of a slow callback execution"""

    function_name: str
    duration_ms: float
    timestamp: float
    stack_trace: str
    loop_delay_ms: float  # How much it delayed the event loop


@dataclass
class EventLoopMetrics:
    """Metrics for event loop health monitoring"""

    slow_callbacks: deque = field(default_factory=lambda: deque(maxlen=50))
    total_callbacks: int = 0
    blocked_callbacks: int = 0  # Callbacks > threshold
    max_callback_time: float = 0.0
    pending_tasks: List[int] = field(default_factory=list)  # Historical task counts
    max_pending_tasks: int = 0
    loop_stalls: int = 0  # Times loop was blocked
    total_monitoring_time: float = 0.0


class AsyncEventLoopMonitor:
    """
    Monitor async event loop for performance issues.

    Features:
    - Detect slow callbacks (configurable threshold)
    - Track pending task counts
    - Identify event loop blocking
    - Provide actionable diagnostics
    """

    def __init__(
        self,
        slow_callback_threshold_ms: float = 100.0,
        check_interval: float = 1.0,
        max_pending_tasks_warning: int = 100,
    ):
        self.slow_callback_threshold = slow_callback_threshold_ms / 1000.0
        self.check_interval = check_interval
        self.max_pending_tasks_warning = max_pending_tasks_warning

        self._metrics = EventLoopMetrics()
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._start_time = 0.0

    async def start(self):
        """Start monitoring the event loop"""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()
        self._loop = asyncio.get_running_loop()

        # Enable debug mode for better tracebacks
        self._loop.set_debug(True)
        self._loop.slow_callback_duration = self.slow_callback_threshold

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            f"Event loop monitor started "
            f"(slow callback threshold: {self.slow_callback_threshold * 1000:.0f}ms)"
        )

    async def stop(self):
        """Stop monitoring"""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._loop:
            self._loop.set_debug(False)

        self._metrics.total_monitoring_time = time.time() - self._start_time

        logger.info(
            f"Event loop monitor stopped "
            f"(found {self._metrics.blocked_callbacks} slow callbacks)"
        )

    async def _monitor_loop(self):
        """Background monitoring task"""
        last_check = time.time()

        while self._running:
            try:
                await asyncio.sleep(self.check_interval)

                # Check pending tasks
                if self._loop:
                    all_tasks = asyncio.all_tasks(self._loop)
                    pending_count = len([t for t in all_tasks if not t.done()])

                    self._metrics.pending_tasks.append(pending_count)
                    if len(self._metrics.pending_tasks) > 100:
                        self._metrics.pending_tasks.pop(0)

                    if pending_count > self._metrics.max_pending_tasks:
                        self._metrics.max_pending_tasks = pending_count

                    if pending_count > self.max_pending_tasks_warning:
                        logger.warning(
                            f"High number of pending tasks: {pending_count} "
                            f"(threshold: {self.max_pending_tasks_warning})"
                        )

                # Check for loop stalls (monitoring took too long)
                now = time.time()
                check_duration = now - last_check
                if check_duration > self.check_interval * 2:
                    self._metrics.loop_stalls += 1
                    logger.warning(
                        f"Event loop stall detected: "
                        f"check took {check_duration:.3f}s "
                        f"(expected {self.check_interval:.1f}s)"
                    )

                last_check = now

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event loop monitor error: {e}")

    def record_slow_callback(
        self, func_name: str, duration: float, stack_trace: str = ""
    ):
        """Record a slow callback for analysis"""
        self._metrics.total_callbacks += 1

        if duration > self.slow_callback_threshold:
            self._metrics.blocked_callbacks += 1

            if duration > self._metrics.max_callback_time:
                self._metrics.max_callback_time = duration

            slow_cb = SlowCallback(
                function_name=func_name,
                duration_ms=duration * 1000,
                timestamp=time.time(),
                stack_trace=stack_trace,
                loop_delay_ms=duration * 1000,  # Approximation
            )

            self._metrics.slow_callbacks.append(slow_cb)

            logger.warning(
                f"Slow callback detected: {func_name} took {duration * 1000:.1f}ms"
            )

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics"""
        current_pending = 0
        if self._loop:
            all_tasks = asyncio.all_tasks(self._loop)
            current_pending = len([t for t in all_tasks if not t.done()])

        slow_callbacks_list = [
            {
                "function": cb.function_name,
                "duration_ms": cb.duration_ms,
                "timestamp": cb.timestamp,
            }
            for cb in self._metrics.slow_callbacks
        ]

        return {
            "running": self._running,
            "monitoring_duration": time.time() - self._start_time,
            "total_callbacks": self._metrics.total_callbacks,
            "slow_callbacks_count": self._metrics.blocked_callbacks,
            "slow_callback_threshold_ms": self.slow_callback_threshold * 1000,
            "max_callback_time_ms": self._metrics.max_callback_time * 1000,
            "current_pending_tasks": current_pending,
            "max_pending_tasks": self._metrics.max_pending_tasks,
            "loop_stalls": self._metrics.loop_stalls,
            "recent_slow_callbacks": slow_callbacks_list[-10:],  # Last 10
            "pending_tasks_history": self._metrics.pending_tasks[
                -20:
            ],  # Last 20 checks
        }

    async def get_health_status(self) -> str:
        """Get overall event loop health status"""
        metrics = await self.get_metrics()

        # Determine health based on metrics
        if metrics["loop_stalls"] > 5:
            return "critical"

        if metrics["slow_callbacks_count"] > 20:
            return "warning"

        if metrics["max_pending_tasks"] > self.max_pending_tasks_warning * 2:
            return "warning"

        return "healthy"


# Global monitor instance
_monitor: Optional[AsyncEventLoopMonitor] = None


async def get_event_loop_monitor() -> AsyncEventLoopMonitor:
    """Get or create global event loop monitor"""
    global _monitor
    if _monitor is None:
        _monitor = AsyncEventLoopMonitor()
        await _monitor.start()
    return _monitor


def monitor_async_function(threshold_ms: float = 100.0):
    """
    Decorator to monitor async function execution time.

    Usage:
        @monitor_async_function(threshold_ms=50)
        async def my_function():
            ...
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                if duration > (threshold_ms / 1000.0):
                    stack = "".join(traceback.format_stack())
                    if _monitor:
                        _monitor.record_slow_callback(
                            func.__name__, duration, stack_trace=stack
                        )

        return wrapper

    return decorator


class BlockingCallDetector:
    """
    Detects blocking calls in async context.

    Monitors thread usage to identify when async code
    accidentally makes blocking calls.
    """

    def __init__(self):
        self._thread_times: Dict[int, float] = {}
        self._warnings_issued = 0

    def check_blocking(self, func_name: str):
        """Check if current call is potentially blocking"""
        thread_id = threading.get_ident()
        current_time = time.time()

        if thread_id in self._thread_times:
            elapsed = current_time - self._thread_times[thread_id]
            if elapsed > 0.1:  # 100ms threshold
                self._warnings_issued += 1
                logger.warning(
                    f"Potential blocking call detected in {func_name}: "
                    f"thread {thread_id} blocked for {elapsed * 1000:.1f}ms"
                )

        self._thread_times[thread_id] = current_time

    def get_stats(self) -> Dict[str, Any]:
        """Get blocking call statistics"""
        return {
            "monitored_threads": len(self._thread_times),
            "warnings_issued": self._warnings_issued,
        }


# Utility functions
async def diagnose_event_loop():
    """
    Run comprehensive event loop diagnostics.

    Returns detailed report on event loop health.
    """
    monitor = await get_event_loop_monitor()
    metrics = await monitor.get_metrics()
    health = await monitor.get_health_status()

    # Get current loop info
    loop = asyncio.get_running_loop()
    all_tasks = asyncio.all_tasks(loop)

    # Analyze tasks
    running_tasks = [t for t in all_tasks if not t.done()]
    completed_tasks = [t for t in all_tasks if t.done() and not t.cancelled()]
    cancelled_tasks = [t for t in all_tasks if t.cancelled()]

    # Task breakdown by name/type
    task_names = {}
    for task in running_tasks:
        name = task.get_name()
        task_names[name] = task_names.get(name, 0) + 1

    report = {
        "health_status": health,
        "metrics": metrics,
        "tasks": {
            "total": len(all_tasks),
            "running": len(running_tasks),
            "completed": len(completed_tasks),
            "cancelled": len(cancelled_tasks),
            "by_name": task_names,
        },
        "loop_info": {
            "is_running": loop.is_running(),
            "is_closed": loop.is_closed(),
            "debug_mode": loop.get_debug(),
        },
    }

    return report


async def wait_for_healthy_loop(
    timeout: float = 30.0, check_interval: float = 1.0
) -> bool:
    """
    Wait for event loop to become healthy.

    Useful during startup or recovery scenarios.

    Returns:
        True if loop became healthy, False if timeout
    """
    monitor = await get_event_loop_monitor()
    start_time = time.time()

    while time.time() - start_time < timeout:
        health = await monitor.get_health_status()
        if health == "healthy":
            return True

        await asyncio.sleep(check_interval)

    return False
