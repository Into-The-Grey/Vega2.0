"""
Database Batch Operations for Vega2.0

Reduces database round-trips through:
- Bulk insert/update operations
- Automatic batching with configurable size
- Time-based flushing
- Transaction management
- Retry logic for transient failures
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class BatchMetrics:
    """Metrics for batch operations"""

    total_items_queued: int = 0
    total_items_inserted: int = 0
    total_batches: int = 0
    failed_batches: int = 0
    total_flush_time: float = 0.0
    avg_batch_size: float = 0.0
    max_queue_size: int = 0


class BatchedConversationLogger:
    """
    Batches conversation logging operations to reduce DB load.

    Instead of writing each conversation immediately,
    buffer them and write in batches for better performance.
    """

    def __init__(
        self,
        batch_size: int = 50,
        flush_interval: float = 5.0,  # Flush every 5 seconds
        max_queue_size: int = 1000,
    ):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size

        self._queue: deque = deque()
        self._metrics = BatchMetrics()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._last_flush = time.time()

    async def start(self):
        """Start batch processor"""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._auto_flush_loop())

        logger.info(
            f"Batch logger started "
            f"(batch_size={self.batch_size}, "
            f"flush_interval={self.flush_interval}s)"
        )

    async def stop(self):
        """Stop batch processor and flush remaining items"""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Final flush
        await self.flush()

        logger.info(
            f"Batch logger stopped "
            f"(processed {self._metrics.total_items_inserted} items "
            f"in {self._metrics.total_batches} batches)"
        )

    async def log_conversation(
        self,
        prompt: str,
        response: str,
        session_id: Optional[str] = None,
        **metadata,
    ):
        """
        Queue a conversation for batched insertion.

        Args:
            prompt: User prompt
            response: LLM response
            session_id: Optional session identifier
            **metadata: Additional fields to store
        """
        async with self._lock:
            # Check queue size limit
            if len(self._queue) >= self.max_queue_size:
                logger.warning(
                    f"Batch queue full ({self.max_queue_size}), "
                    f"forcing immediate flush"
                )
                await self._flush_batch()

            # Add to queue
            item = {
                "prompt": prompt,
                "response": response,
                "session_id": session_id,
                "timestamp": time.time(),
                **metadata,
            }

            self._queue.append(item)
            self._metrics.total_items_queued += 1

            # Update max queue size metric
            if len(self._queue) > self._metrics.max_queue_size:
                self._metrics.max_queue_size = len(self._queue)

            # Check if we should flush
            if len(self._queue) >= self.batch_size:
                await self._flush_batch()

    async def _auto_flush_loop(self):
        """Automatically flush based on time interval"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)

                # Check if flush is needed
                if self._queue and (
                    time.time() - self._last_flush >= self.flush_interval
                ):
                    await self.flush()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-flush error: {e}")

    async def flush(self):
        """Flush all pending items"""
        async with self._lock:
            await self._flush_batch()

    async def _flush_batch(self):
        """Internal flush implementation"""
        if not self._queue:
            return

        # Get batch to insert
        batch = []
        while self._queue and len(batch) < self.batch_size:
            batch.append(self._queue.popleft())

        if not batch:
            return

        start_time = time.time()

        try:
            # Import here to avoid circular dependency
            from .db import bulk_log_conversations

            # Perform bulk insert
            await bulk_log_conversations(batch)

            # Update metrics
            self._metrics.total_items_inserted += len(batch)
            self._metrics.total_batches += 1
            self._metrics.total_flush_time += time.time() - start_time

            # Update average batch size
            self._metrics.avg_batch_size = (
                self._metrics.total_items_inserted / self._metrics.total_batches
            )

            self._last_flush = time.time()

            logger.debug(
                f"Flushed batch of {len(batch)} conversations "
                f"in {(time.time() - start_time) * 1000:.1f}ms"
            )

        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            self._metrics.failed_batches += 1

            # Put items back in queue for retry
            for item in reversed(batch):
                self._queue.appendleft(item)

    async def get_metrics(self) -> Dict[str, Any]:
        """Get batch operation metrics"""
        async with self._lock:
            return {
                "queue_size": len(self._queue),
                "max_queue_size": self._metrics.max_queue_size,
                "batch_size": self.batch_size,
                "total_items_queued": self._metrics.total_items_queued,
                "total_items_inserted": self._metrics.total_items_inserted,
                "total_batches": self._metrics.total_batches,
                "failed_batches": self._metrics.failed_batches,
                "success_rate": (
                    (self._metrics.total_batches - self._metrics.failed_batches)
                    / max(1, self._metrics.total_batches)
                ),
                "avg_batch_size": self._metrics.avg_batch_size,
                "total_flush_time_seconds": self._metrics.total_flush_time,
                "avg_flush_time_ms": (
                    self._metrics.total_flush_time
                    / max(1, self._metrics.total_batches)
                    * 1000
                ),
            }


class BatchedUpdateExecutor:
    """
    Generic batched update executor.

    Batches any kind of database updates for efficiency.
    """

    def __init__(
        self,
        execute_func: Callable[[List[T]], Any],
        batch_size: int = 100,
        flush_interval: float = 2.0,
    ):
        self.execute_func = execute_func
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: deque = deque()
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()

    async def start(self):
        """Start batch executor"""
        if self._running:
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._auto_flush())

    async def stop(self):
        """Stop and flush"""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        await self.flush()

    async def queue_update(self, item: T):
        """Queue an item for batched execution"""
        async with self._lock:
            self._queue.append(item)

            if len(self._queue) >= self.batch_size:
                await self._flush_batch()

    async def _auto_flush(self):
        """Auto-flush timer"""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch executor auto-flush error: {e}")

    async def flush(self):
        """Flush pending items"""
        async with self._lock:
            await self._flush_batch()

    async def _flush_batch(self):
        """Execute batch"""
        if not self._queue:
            return

        batch = []
        while self._queue and len(batch) < self.batch_size:
            batch.append(self._queue.popleft())

        try:
            result = self.execute_func(batch)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error(f"Batch execution failed: {e}")
            # Put items back
            for item in reversed(batch):
                self._queue.appendleft(item)


# Singleton instance
_conversation_logger: Optional[BatchedConversationLogger] = None


async def get_batched_logger() -> BatchedConversationLogger:
    """Get or create global batched conversation logger"""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = BatchedConversationLogger()
        await _conversation_logger.start()
    return _conversation_logger


async def log_conversation_batched(
    prompt: str,
    response: str,
    session_id: Optional[str] = None,
    **metadata,
):
    """
    Convenience function to log conversation with batching.

    Drop-in replacement for direct db.log_conversation()
    but with automatic batching for better performance.
    """
    logger_instance = await get_batched_logger()
    await logger_instance.log_conversation(
        prompt=prompt, response=response, session_id=session_id, **metadata
    )


# Database helper functions that should be added to db.py


async def bulk_log_conversations(conversations: List[Dict[str, Any]]):
    """
    Bulk insert conversations into database.

    This should be added to db.py for actual implementation.

    Example implementation:
        INSERT INTO conversations (prompt, response, session_id, ts, ...)
        VALUES (%s, %s, %s, %s, ...) [repeated for each conversation]
    """
    # Import here to avoid circular dependency
    try:
        from .db import get_db

        # Get database session
        db = await get_db()

        # Prepare bulk insert
        # This is a placeholder - actual implementation depends on SQLAlchemy setup
        from .db import Conversation

        conversation_objects = []
        for conv in conversations:
            conversation_objects.append(
                Conversation(
                    prompt=conv["prompt"],
                    response=conv["response"],
                    session_id=conv.get("session_id"),
                    # Add other fields as needed
                )
            )

        # Bulk insert
        db.add_all(conversation_objects)
        await db.commit()

        logger.debug(f"Bulk inserted {len(conversations)} conversations")

    except Exception as e:
        logger.error(f"Bulk insert failed: {e}")
        raise


async def bulk_update_conversations(updates: List[Dict[str, Any]]):
    """
    Bulk update conversations.

    Example usage:
        updates = [
            {"id": 1, "feedback": "good"},
            {"id": 2, "feedback": "bad"},
        ]
    """
    try:
        from .db import get_db, Conversation

        db = await get_db()

        # Bulk update using SQLAlchemy
        for update in updates:
            conv_id = update.pop("id")
            stmt = (
                Conversation.__table__.update()
                .where(Conversation.id == conv_id)
                .values(**update)
            )

            await db.execute(stmt)

        await db.commit()

        logger.debug(f"Bulk updated {len(updates)} conversations")

    except Exception as e:
        logger.error(f"Bulk update failed: {e}")
        raise
