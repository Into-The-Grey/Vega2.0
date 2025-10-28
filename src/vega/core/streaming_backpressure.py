"""
Streaming backpressure control for Vega2.0

Prevents memory buildup when streaming responses to slow clients by:
- Buffering with configurable limits
- Flow control with pause/resume signaling
- Buffer size monitoring and metrics
- Automatic pressure relief strategies
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import AsyncGenerator, TypeVar, Optional, Callable
from dataclasses import dataclass
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackpressureState(Enum):
    """Stream backpressure states"""

    NORMAL = "normal"  # Buffer below thresholds
    WARNING = "warning"  # Approaching buffer limits
    THROTTLED = "throttled"  # Actively slowing production
    BLOCKED = "blocked"  # Buffer full, blocking producer


@dataclass
class StreamMetrics:
    """Metrics for stream monitoring"""

    chunks_produced: int = 0
    chunks_consumed: int = 0
    chunks_dropped: int = 0
    total_bytes_produced: int = 0
    total_bytes_consumed: int = 0
    buffer_overflow_count: int = 0
    max_buffer_size_seen: int = 0
    total_throttle_time: float = 0.0
    start_time: float = 0.0
    end_time: Optional[float] = None


class BufferedStream:
    """
    Async generator wrapper with backpressure control.

    Features:
    - Configurable buffer size limits
    - Automatic throttling when buffer fills
    - Metrics tracking for monitoring
    - Drop strategies for extreme cases
    """

    def __init__(
        self,
        source: AsyncGenerator[T, None],
        buffer_size: int = 100,
        warning_threshold: float = 0.7,
        throttle_threshold: float = 0.9,
        drop_on_overflow: bool = False,
        chunk_delay_ms: float = 0.0,
    ):
        self.source = source
        self.buffer_size = buffer_size
        self.warning_threshold = warning_threshold
        self.throttle_threshold = throttle_threshold
        self.drop_on_overflow = drop_on_overflow
        self.chunk_delay = chunk_delay_ms / 1000.0  # Convert to seconds

        self._buffer: deque = deque(maxlen=buffer_size)
        self._producer_task: Optional[asyncio.Task] = None
        self._finished = False
        self._error: Optional[Exception] = None
        self._metrics = StreamMetrics(start_time=time.time())
        self._state = BackpressureState.NORMAL
        self._lock = asyncio.Lock()

    async def _produce(self):
        """Background task to consume from source and fill buffer"""
        try:
            async for chunk in self.source:
                # Check buffer state and apply backpressure
                current_size = len(self._buffer)
                buffer_usage = current_size / self.buffer_size

                # Update state based on buffer usage
                if buffer_usage >= self.throttle_threshold:
                    self._state = BackpressureState.THROTTLED
                    # Slow down production
                    throttle_delay = 0.1 * (buffer_usage - self.throttle_threshold) * 10
                    await asyncio.sleep(throttle_delay)
                    self._metrics.total_throttle_time += throttle_delay
                elif buffer_usage >= self.warning_threshold:
                    self._state = BackpressureState.WARNING
                else:
                    self._state = BackpressureState.NORMAL

                # Wait for buffer space if full
                while len(self._buffer) >= self.buffer_size:
                    if self.drop_on_overflow:
                        self._metrics.chunks_dropped += 1
                        logger.warning(
                            f"Dropping chunk due to buffer overflow "
                            f"(buffer={len(self._buffer)}/{self.buffer_size})"
                        )
                        break

                    self._state = BackpressureState.BLOCKED
                    await asyncio.sleep(0.01)  # Small delay before retry

                # Add chunk to buffer
                async with self._lock:
                    self._buffer.append(chunk)
                    self._metrics.chunks_produced += 1

                    # Track chunk size if it's a string or bytes
                    if isinstance(chunk, (str, bytes)):
                        self._metrics.total_bytes_produced += len(chunk)

                    # Update max buffer size seen
                    if len(self._buffer) > self._metrics.max_buffer_size_seen:
                        self._metrics.max_buffer_size_seen = len(self._buffer)

                # Optional delay between chunks (for rate limiting)
                if self.chunk_delay > 0:
                    await asyncio.sleep(self.chunk_delay)

        except Exception as e:
            self._error = e
            logger.error(f"Stream producer error: {e}")
        finally:
            self._finished = True

    async def __aiter__(self):
        """Start producer and return iterator"""
        self._producer_task = asyncio.create_task(self._produce())
        return self

    async def __anext__(self) -> T:
        """Get next chunk with backpressure handling"""
        while True:
            # Check for errors
            if self._error:
                raise self._error

            # Try to get from buffer
            async with self._lock:
                if self._buffer:
                    chunk = self._buffer.popleft()
                    self._metrics.chunks_consumed += 1

                    if isinstance(chunk, (str, bytes)):
                        self._metrics.total_bytes_consumed += len(chunk)

                    return chunk

            # Check if producer finished
            if self._finished:
                self._metrics.end_time = time.time()
                raise StopAsyncIteration

            # Wait a bit for more data
            await asyncio.sleep(0.01)

    async def get_metrics(self) -> dict:
        """Get current stream metrics"""
        async with self._lock:
            duration = (
                self._metrics.end_time or time.time()
            ) - self._metrics.start_time

            chunks_in_flight = (
                self._metrics.chunks_produced - self._metrics.chunks_consumed
            )
            buffer_usage = len(self._buffer) / self.buffer_size

            return {
                "state": self._state.value,
                "buffer_size": len(self._buffer),
                "buffer_capacity": self.buffer_size,
                "buffer_usage_percent": buffer_usage * 100,
                "chunks_produced": self._metrics.chunks_produced,
                "chunks_consumed": self._metrics.chunks_consumed,
                "chunks_in_flight": chunks_in_flight,
                "chunks_dropped": self._metrics.chunks_dropped,
                "bytes_produced": self._metrics.total_bytes_produced,
                "bytes_consumed": self._metrics.total_bytes_consumed,
                "overflow_count": self._metrics.buffer_overflow_count,
                "max_buffer_size": self._metrics.max_buffer_size_seen,
                "throttle_time_seconds": self._metrics.total_throttle_time,
                "duration_seconds": duration,
                "throughput_chunks_per_sec": (
                    self._metrics.chunks_consumed / max(0.001, duration)
                ),
                "throughput_bytes_per_sec": (
                    self._metrics.total_bytes_consumed / max(0.001, duration)
                ),
            }

    async def cancel(self):
        """Cancel stream and cleanup"""
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass

        self._finished = True
        self._metrics.end_time = time.time()


def buffered_stream(
    buffer_size: int = 100,
    warning_threshold: float = 0.7,
    throttle_threshold: float = 0.9,
    drop_on_overflow: bool = False,
    chunk_delay_ms: float = 0.0,
):
    """
    Decorator to add backpressure control to async generators.

    Usage:
        @buffered_stream(buffer_size=50, throttle_threshold=0.8)
        async def stream_tokens():
            for token in tokens:
                yield token

    Args:
        buffer_size: Maximum number of chunks to buffer
        warning_threshold: Buffer usage % to trigger warning state (0.0-1.0)
        throttle_threshold: Buffer usage % to trigger throttling (0.0-1.0)
        drop_on_overflow: Whether to drop chunks when buffer is full
        chunk_delay_ms: Delay between chunks in milliseconds (for rate limiting)
    """

    def decorator(func: Callable[..., AsyncGenerator[T, None]]):
        async def wrapper(*args, **kwargs) -> AsyncGenerator[T, None]:
            source = func(*args, **kwargs)
            buffered = BufferedStream(
                source=source,
                buffer_size=buffer_size,
                warning_threshold=warning_threshold,
                throttle_threshold=throttle_threshold,
                drop_on_overflow=drop_on_overflow,
                chunk_delay_ms=chunk_delay_ms,
            )

            try:
                async for chunk in buffered:
                    yield chunk
            finally:
                # Log metrics on completion
                metrics = await buffered.get_metrics()
                if (
                    metrics["chunks_dropped"] > 0
                    or metrics["throttle_time_seconds"] > 1
                ):
                    logger.warning(
                        f"Stream completed with backpressure: "
                        f"throttled={metrics['throttle_time_seconds']:.1f}s, "
                        f"dropped={metrics['chunks_dropped']}, "
                        f"max_buffer={metrics['max_buffer_size']}"
                    )
                else:
                    logger.debug(
                        f"Stream completed: "
                        f"{metrics['chunks_consumed']} chunks, "
                        f"{metrics['duration_seconds']:.2f}s, "
                        f"{metrics['throughput_chunks_per_sec']:.1f} chunks/s"
                    )

        return wrapper

    return decorator


class AdaptiveStreamBuffer:
    """
    Self-tuning buffer that adjusts size based on consumption rate.

    Automatically increases buffer size if consumer is fast,
    decreases if consumer is slow and causing backpressure.
    """

    def __init__(
        self,
        initial_size: int = 50,
        min_size: int = 10,
        max_size: int = 500,
        adjustment_interval: float = 5.0,
    ):
        self.current_size = initial_size
        self.min_size = min_size
        self.max_size = max_size
        self.adjustment_interval = adjustment_interval

        self._last_adjustment = time.time()
        self._consumption_history: deque = deque(maxlen=10)

    def should_adjust(self) -> bool:
        """Check if it's time to adjust buffer size"""
        return time.time() - self._last_adjustment >= self.adjustment_interval

    def record_consumption(self, chunks_consumed: int, duration: float):
        """Record consumption rate for adjustment calculation"""
        rate = chunks_consumed / max(0.001, duration)
        self._consumption_history.append(rate)

    def adjust_size(self, current_buffer_usage: float) -> int:
        """Calculate new buffer size based on metrics"""
        if not self.should_adjust():
            return self.current_size

        # Calculate average consumption rate
        if not self._consumption_history:
            return self.current_size

        avg_rate = sum(self._consumption_history) / len(self._consumption_history)

        # Adjust based on buffer pressure and consumption rate
        if current_buffer_usage > 0.8:
            # High pressure, increase buffer
            new_size = min(int(self.current_size * 1.5), self.max_size)
        elif current_buffer_usage < 0.3 and avg_rate > 10:
            # Low pressure and fast consumer, decrease buffer
            new_size = max(int(self.current_size * 0.8), self.min_size)
        else:
            # Stable, no change
            new_size = self.current_size

        if new_size != self.current_size:
            logger.info(
                f"Adjusting stream buffer size: {self.current_size} → {new_size} "
                f"(usage={current_buffer_usage:.1%}, rate={avg_rate:.1f} chunks/s)"
            )

        self.current_size = new_size
        self._last_adjustment = time.time()

        return new_size
