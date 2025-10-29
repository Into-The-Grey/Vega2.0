"""
Persistent Mode Memory Management for Vega
Ensures Vega never runs out of memory and maintains continuous context.
"""

import asyncio
import logging
import os
import psutil
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Monitors memory usage and proactively manages conversation history
    to prevent OOM errors while maintaining continuous context.
    """

    def __init__(
        self,
        warning_threshold_percent: float = 80.0,
        critical_threshold_percent: float = 90.0,
        check_interval_seconds: int = 30,
        max_context_entries: int = 50,
        compression_trigger: int = 100,
    ):
        self.warning_threshold = warning_threshold_percent
        self.critical_threshold = critical_threshold_percent
        self.check_interval = check_interval_seconds
        self.max_context_entries = max_context_entries
        self.compression_trigger = compression_trigger

        self.process = psutil.Process(os.getpid())
        self.last_cleanup = time.time()
        self.cleanup_count = 0
        self.running = False
        self._task: Optional[asyncio.Task] = None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        try:
            # Process memory
            mem_info = self.process.memory_info()
            process_mb = mem_info.rss / 1024 / 1024

            # System memory
            vm = psutil.virtual_memory()
            system_percent = vm.percent
            system_available_mb = vm.available / 1024 / 1024

            return {
                "process_memory_mb": round(process_mb, 2),
                "system_memory_percent": round(system_percent, 2),
                "system_available_mb": round(system_available_mb, 2),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}

    async def check_and_cleanup(self) -> bool:
        """
        Check memory usage and perform cleanup if needed.
        Returns True if cleanup was performed.
        """
        stats = self.get_memory_stats()
        if not stats:
            return False

        system_percent = stats.get("system_memory_percent", 0)

        # Log memory stats periodically
        logger.info(
            f"Memory: Process={stats['process_memory_mb']:.1f}MB, "
            f"System={system_percent:.1f}%, "
            f"Available={stats['system_available_mb']:.1f}MB"
        )

        # Critical threshold - aggressive cleanup
        if system_percent >= self.critical_threshold:
            logger.warning(
                f"CRITICAL: Memory usage at {system_percent:.1f}% "
                f"(threshold: {self.critical_threshold}%)"
            )
            await self._aggressive_cleanup()
            return True

        # Warning threshold - gentle cleanup
        elif system_percent >= self.warning_threshold:
            logger.warning(
                f"WARNING: Memory usage at {system_percent:.1f}% "
                f"(threshold: {self.warning_threshold}%)"
            )
            await self._gentle_cleanup()
            return True

        return False

    async def _gentle_cleanup(self):
        """Perform gentle cleanup - compress old context."""
        try:
            from .db import compress_old_context, get_db_size

            logger.info("Starting gentle cleanup: compressing old context...")

            # Compress conversations older than 1 hour
            cutoff = datetime.now() - timedelta(hours=1)
            compressed = compress_old_context(
                cutoff, keep_recent=self.max_context_entries
            )

            db_size_mb = get_db_size() / 1024 / 1024

            logger.info(
                f"Gentle cleanup complete: {compressed} entries compressed, "
                f"DB size: {db_size_mb:.2f}MB"
            )

            self.cleanup_count += 1
            self.last_cleanup = time.time()

        except Exception as e:
            logger.error(f"Error in gentle cleanup: {e}")

    async def _aggressive_cleanup(self):
        """Perform aggressive cleanup - summarize and remove old context."""
        try:
            from .db import (
                compress_old_context,
                summarize_and_archive_old,
                vacuum_database,
                get_db_size,
            )

            logger.warning("Starting AGGRESSIVE cleanup...")

            # 1. Compress conversations older than 30 minutes
            cutoff = datetime.now() - timedelta(minutes=30)
            compressed = compress_old_context(
                cutoff, keep_recent=self.max_context_entries
            )

            # 2. Summarize and archive conversations older than 2 hours
            archive_cutoff = datetime.now() - timedelta(hours=2)
            archived = summarize_and_archive_old(archive_cutoff, keep_recent=20)

            # 3. Vacuum database to reclaim space
            vacuum_database()

            db_size_mb = get_db_size() / 1024 / 1024

            logger.warning(
                f"Aggressive cleanup complete: {compressed} compressed, "
                f"{archived} archived, DB size: {db_size_mb:.2f}MB"
            )

            self.cleanup_count += 1
            self.last_cleanup = time.time()

        except Exception as e:
            logger.error(f"Error in aggressive cleanup: {e}")

    async def _monitoring_loop(self):
        """Background task that monitors memory continuously."""
        logger.info(
            f"Memory manager started: warning={self.warning_threshold}%, "
            f"critical={self.critical_threshold}%, interval={self.check_interval}s"
        )

        while self.running:
            try:
                await self.check_and_cleanup()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                logger.info("Memory manager monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    def start(self):
        """Start the memory monitoring background task."""
        if self.running:
            logger.warning("Memory manager already running")
            return

        self.running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory manager monitoring started")

    async def stop(self):
        """Stop the memory monitoring background task."""
        if not self.running:
            return

        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Memory manager stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory manager statistics."""
        return {
            "running": self.running,
            "cleanup_count": self.cleanup_count,
            "last_cleanup": (
                datetime.fromtimestamp(self.last_cleanup).isoformat()
                if self.last_cleanup
                else None
            ),
            "memory": self.get_memory_stats(),
        }


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager(
            warning_threshold_percent=80.0,
            critical_threshold_percent=90.0,
            check_interval_seconds=30,
            max_context_entries=50,
            compression_trigger=100,
        )
    return _memory_manager


async def start_memory_manager():
    """Start the global memory manager."""
    manager = get_memory_manager()
    manager.start()


async def stop_memory_manager():
    """Stop the global memory manager."""
    manager = get_memory_manager()
    await manager.stop()
