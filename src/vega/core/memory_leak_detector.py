"""
Memory Leak Detection System for Vega2.0

Tracks object lifecycles using weak references to detect:
- Unreleased conversation history
- Lingering LLM context
- Unclosed resources
- Memory growth patterns
"""

from __future__ import annotations

import asyncio
import gc
import sys
import time
import logging
import weakref
from typing import Any, Dict, List, Optional, Set, Type
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ObjectTracking:
    """Tracking info for a monitored object"""

    obj_id: int
    obj_type: str
    creation_time: float
    creation_stack: str
    size_bytes: int
    still_alive: bool = True


@dataclass
class LeakDetectionMetrics:
    """Metrics for memory leak detection"""

    objects_tracked: int = 0
    objects_freed: int = 0
    potential_leaks: int = 0
    total_memory_tracked_mb: float = 0.0
    gc_collections: int = 0
    last_check_time: float = 0.0


class MemoryLeakDetector:
    """
    Detect memory leaks using weak references.

    Monitors objects that should be garbage collected
    and alerts when they persist beyond expected lifetime.
    """

    def __init__(
        self,
        check_interval: float = 60.0,  # Check every minute
        leak_threshold_seconds: float = 300.0,  # 5 minutes
    ):
        self.check_interval = check_interval
        self.leak_threshold = leak_threshold_seconds

        # Weak reference tracking
        self._tracked_objects: Dict[int, ObjectTracking] = {}
        self._object_refs: Set[weakref.ref] = set()

        # Type-based tracking
        self._tracked_by_type: Dict[str, List[int]] = defaultdict(list)

        # Metrics
        self._metrics = LeakDetectionMetrics()

        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    def track_object(self, obj: Any, context: str = ""):
        """
        Start tracking an object for leak detection.

        Args:
            obj: Object to track
            context: Description of where object was created
        """
        obj_id = id(obj)

        # Skip if already tracked
        if obj_id in self._tracked_objects:
            return

        # Get object size
        size = sys.getsizeof(obj)

        # Create weak reference
        try:

            def callback(ref):
                """Called when object is garbage collected"""
                self._on_object_freed(obj_id)

            weak_ref = weakref.ref(obj, callback)
            self._object_refs.add(weak_ref)

        except TypeError:
            # Some objects don't support weak references
            logger.debug(f"Cannot create weak ref for {type(obj).__name__}")
            return

        # Record tracking info
        tracking = ObjectTracking(
            obj_id=obj_id,
            obj_type=type(obj).__name__,
            creation_time=time.time(),
            creation_stack=context,
            size_bytes=size,
        )

        self._tracked_objects[obj_id] = tracking
        self._tracked_by_type[tracking.obj_type].append(obj_id)
        self._metrics.objects_tracked += 1
        self._metrics.total_memory_tracked_mb += size / (1024 * 1024)

    def _on_object_freed(self, obj_id: int):
        """Callback when tracked object is freed"""
        if obj_id in self._tracked_objects:
            tracking = self._tracked_objects[obj_id]
            tracking.still_alive = False
            self._metrics.objects_freed += 1

            # Remove from type tracking
            if obj_id in self._tracked_by_type[tracking.obj_type]:
                self._tracked_by_type[tracking.obj_type].remove(obj_id)

            # Update memory tracking
            self._metrics.total_memory_tracked_mb -= tracking.size_bytes / (1024 * 1024)

    async def start(self):
        """Start background leak detection"""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info(
            f"Memory leak detector started "
            f"(check interval: {self.check_interval}s, "
            f"leak threshold: {self.leak_threshold}s)"
        )

    async def stop(self):
        """Stop leak detection"""
        self._running = False

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"Memory leak detector stopped "
            f"(tracked {self._metrics.objects_tracked} objects, "
            f"found {self._metrics.potential_leaks} potential leaks)"
        )

    async def _monitor_loop(self):
        """Background monitoring task"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                await self._check_for_leaks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory leak monitor error: {e}")

    async def _check_for_leaks(self):
        """Check for potential memory leaks"""
        # Force garbage collection
        gc.collect()
        self._metrics.gc_collections += 1
        self._metrics.last_check_time = time.time()

        now = time.time()
        potential_leaks = []

        # Check tracked objects
        for obj_id, tracking in list(self._tracked_objects.items()):
            if not tracking.still_alive:
                # Object was freed, remove from tracking
                del self._tracked_objects[obj_id]
                continue

            # Check if object has lived too long
            age = now - tracking.creation_time
            if age > self.leak_threshold:
                potential_leaks.append(tracking)

        # Report leaks
        if potential_leaks:
            self._metrics.potential_leaks += len(potential_leaks)

            # Group by type for reporting
            by_type = defaultdict(list)
            for leak in potential_leaks:
                by_type[leak.obj_type].append(leak)

            for obj_type, leaks in by_type.items():
                total_size_mb = sum(l.size_bytes for l in leaks) / (1024 * 1024)
                logger.warning(
                    f"Potential memory leak detected: "
                    f"{len(leaks)} {obj_type} objects "
                    f"({total_size_mb:.2f} MB) "
                    f"alive for >{self.leak_threshold}s"
                )

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current leak detection metrics"""
        # Get current memory usage
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        # Calculate stats by type
        by_type_stats = {}
        for obj_type, obj_ids in self._tracked_by_type.items():
            alive_objects = [
                self._tracked_objects[oid]
                for oid in obj_ids
                if oid in self._tracked_objects
                and self._tracked_objects[oid].still_alive
            ]

            if alive_objects:
                total_size = sum(o.size_bytes for o in alive_objects)
                avg_age = sum(
                    time.time() - o.creation_time for o in alive_objects
                ) / len(alive_objects)

                by_type_stats[obj_type] = {
                    "count": len(alive_objects),
                    "total_size_mb": total_size / (1024 * 1024),
                    "avg_age_seconds": avg_age,
                }

        return {
            "objects_tracked": self._metrics.objects_tracked,
            "objects_freed": self._metrics.objects_freed,
            "objects_currently_alive": len(
                [t for t in self._tracked_objects.values() if t.still_alive]
            ),
            "potential_leaks": self._metrics.potential_leaks,
            "memory_tracked_mb": self._metrics.total_memory_tracked_mb,
            "gc_collections": self._metrics.gc_collections,
            "last_check_time": datetime.fromtimestamp(
                self._metrics.last_check_time
            ).isoformat(),
            "by_type": by_type_stats,
            "process_memory": {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
            },
        }

    async def get_leaked_objects(self, obj_type: Optional[str] = None) -> List[Dict]:
        """Get list of potentially leaked objects"""
        now = time.time()
        leaked = []

        for tracking in self._tracked_objects.values():
            if not tracking.still_alive:
                continue

            age = now - tracking.creation_time
            if age < self.leak_threshold:
                continue

            if obj_type and tracking.obj_type != obj_type:
                continue

            leaked.append(
                {
                    "type": tracking.obj_type,
                    "age_seconds": age,
                    "size_bytes": tracking.size_bytes,
                    "creation_context": tracking.creation_stack,
                }
            )

        # Sort by age (oldest first)
        leaked.sort(key=lambda x: x["age_seconds"], reverse=True)

        return leaked

    async def force_cleanup(self):
        """Force garbage collection and cleanup"""
        # Multiple GC passes to handle reference cycles
        for _ in range(3):
            collected = gc.collect()
            logger.info(f"Garbage collection pass collected {collected} objects")
            await asyncio.sleep(0.1)

        # Update metrics
        await self._check_for_leaks()


# Global detector instance
_detector: Optional[MemoryLeakDetector] = None


async def get_memory_leak_detector() -> MemoryLeakDetector:
    """Get or create global memory leak detector"""
    global _detector
    if _detector is None:
        _detector = MemoryLeakDetector()
        await _detector.start()
    return _detector


def track_for_leaks(obj: Any, context: str = ""):
    """
    Convenience function to track an object for leaks.

    Usage:
        conversation_history = []
        track_for_leaks(conversation_history, "chat_session_123")
    """
    if _detector:
        _detector.track_object(obj, context)


class ConversationHistoryTracker:
    """
    Specialized tracker for conversation history objects.

    Ensures conversation history is properly cleaned up
    after sessions end.
    """

    def __init__(self):
        self._sessions: Dict[str, weakref.ref] = {}
        self._session_start_times: Dict[str, float] = {}

    def register_session(self, session_id: str, history: Any):
        """Register a conversation session for tracking"""
        self._sessions[session_id] = weakref.ref(history)
        self._session_start_times[session_id] = time.time()

        # Track with main detector
        track_for_leaks(history, f"conversation_session:{session_id}")

    def unregister_session(self, session_id: str):
        """Manually unregister a session"""
        self._sessions.pop(session_id, None)
        self._session_start_times.pop(session_id, None)

    def get_active_sessions(self) -> List[Dict]:
        """Get list of active sessions"""
        now = time.time()
        active = []

        for session_id, ref in list(self._sessions.items()):
            history = ref()
            if history is None:
                # Session was garbage collected
                self._sessions.pop(session_id, None)
                self._session_start_times.pop(session_id, None)
                continue

            age = now - self._session_start_times.get(session_id, now)
            active.append(
                {
                    "session_id": session_id,
                    "age_seconds": age,
                    "history_length": (
                        len(history) if hasattr(history, "__len__") else 0
                    ),
                }
            )

        return active


# Utility functions
async def diagnose_memory_leaks():
    """
    Run comprehensive memory leak diagnostics.

    Returns detailed report on potential memory issues.
    """
    detector = await get_memory_leak_detector()

    # Force cleanup first
    await detector.force_cleanup()

    # Get metrics
    metrics = await detector.get_metrics()

    # Get leaked objects by type
    leaked_by_type = {}
    for obj_type in metrics["by_type"].keys():
        leaked = await detector.get_leaked_objects(obj_type=obj_type)
        if leaked:
            leaked_by_type[obj_type] = leaked[:10]  # Top 10 oldest

    report = {
        "metrics": metrics,
        "leaked_objects_by_type": leaked_by_type,
        "gc_stats": {
            "counts": gc.get_count(),
            "threshold": gc.get_threshold(),
            "objects": len(gc.get_objects()),
        },
        "recommendations": _generate_recommendations(metrics),
    }

    return report


def _generate_recommendations(metrics: Dict) -> List[str]:
    """Generate recommendations based on metrics"""
    recommendations = []

    if metrics["potential_leaks"] > 10:
        recommendations.append(
            "High number of potential leaks detected. "
            "Review object lifecycle management."
        )

    if metrics["memory_tracked_mb"] > 100:
        recommendations.append(
            "Large amount of memory being tracked. "
            "Consider implementing object pooling or caching limits."
        )

    for obj_type, stats in metrics["by_type"].items():
        if stats["count"] > 100:
            recommendations.append(
                f"High count of {obj_type} objects ({stats['count']}). "
                f"Consider cleanup strategies."
            )

    return recommendations
