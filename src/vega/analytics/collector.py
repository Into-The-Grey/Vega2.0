"""
Analytics data collection and event tracking system
==================================================

This module provides comprehensive analytics data collection,
event tracking, and metrics gathering for the Vega 2.0 platform.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import asyncio
from collections import defaultdict, deque
import time


class EventType(Enum):
    """Types of events to track"""

    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTIVITY = "user_activity"

    # Collaboration events
    WORKSPACE_CREATED = "workspace_created"
    WORKSPACE_JOINED = "workspace_joined"
    WORKSPACE_LEFT = "workspace_left"
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_EDITED = "document_edited"
    DOCUMENT_VIEWED = "document_viewed"
    MESSAGE_SENT = "message_sent"
    ANNOTATION_CREATED = "annotation_created"

    # System events
    API_REQUEST = "api_request"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"

    # Federated learning events
    FL_TRAINING_STARTED = "fl_training_started"
    FL_TRAINING_COMPLETED = "fl_training_completed"
    FL_MODEL_UPDATED = "fl_model_updated"
    FL_PARTICIPANT_JOINED = "fl_participant_joined"
    FL_PARTICIPANT_LEFT = "fl_participant_left"


class MetricType(Enum):
    """Types of metrics to collect"""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Event:
    """Represents an analytics event"""

    id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    workspace_id: Optional[str]
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "properties": self.properties,
            "metadata": self.metadata,
        }


@dataclass
class Metric:
    """Represents a system metric"""

    name: str
    metric_type: MetricType
    value: Union[int, float]
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "metric_type": self.metric_type.value,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
        }


@dataclass
class PerformanceMetric:
    """Represents a performance measurement"""

    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error,
            "context": self.context,
        }


class EventBuffer:
    """Thread-safe buffer for events"""

    def __init__(self, max_size: int = 10000):
        self.events: deque = deque(maxlen=max_size)
        self.lock = asyncio.Lock()

    async def add_event(self, event: Event):
        """Add event to buffer"""
        async with self.lock:
            self.events.append(event)

    async def get_events(self, count: Optional[int] = None) -> List[Event]:
        """Get events from buffer"""
        async with self.lock:
            if count is None:
                return list(self.events)
            return list(self.events)[-count:]

    async def clear(self):
        """Clear buffer"""
        async with self.lock:
            self.events.clear()


class MetricsCollector:
    """Collects and aggregates metrics"""

    def __init__(self):
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()

    async def increment_counter(
        self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric"""
        async with self.lock:
            key = self._make_key(name, tags)
            self.counters[key] += value

    async def set_gauge(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """Set a gauge metric"""
        async with self.lock:
            key = self._make_key(name, tags)
            self.gauges[key] = value

    async def record_histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ):
        """Record a histogram value"""
        async with self.lock:
            key = self._make_key(name, tags)
            self.histograms[key].append(value)

            # Keep only recent values (last 1000)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]

    async def record_timer(
        self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None
    ):
        """Record a timer value"""
        async with self.lock:
            key = self._make_key(name, tags)
            self.timers[key].append(duration_ms)

            # Keep only recent values (last 1000)
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        async with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    k: {
                        "count": len(v),
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0,
                        "avg": sum(v) / len(v) if v else 0,
                        "sum": sum(v),
                    }
                    for k, v in self.histograms.items()
                },
                "timers": {
                    k: {
                        "count": len(v),
                        "min": min(v) if v else 0,
                        "max": max(v) if v else 0,
                        "avg": sum(v) / len(v) if v else 0,
                        "p50": self._percentile(v, 50) if v else 0,
                        "p95": self._percentile(v, 95) if v else 0,
                        "p99": self._percentile(v, 99) if v else 0,
                    }
                    for k, v in self.timers.items()
                },
            }

    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for metric with tags"""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}#{tag_str}"

    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]

        lower = sorted_values[int(index)]
        upper = sorted_values[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))


class AnalyticsCollector:
    """Main analytics collection system"""

    def __init__(self):
        self.event_buffer = EventBuffer()
        self.metrics_collector = MetricsCollector()
        self.performance_buffer = EventBuffer()

        # Session tracking
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_sessions: Dict[str, List[str]] = defaultdict(list)

        # Configuration
        self.enabled = True
        self.sampling_rate = 1.0  # Sample 100% of events by default

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._background_tasks_started = False

    def _start_background_tasks(self):
        """Start background collection tasks"""
        if self._background_tasks_started:
            return

        try:
            # Only start tasks if we have a running event loop
            loop = asyncio.get_running_loop()
            task = asyncio.create_task(self._collect_system_metrics())
            self._background_tasks.append(task)
            self._background_tasks_started = True
        except RuntimeError:
            # No event loop running, defer task creation
            pass

    def ensure_background_tasks(self):
        """Ensure background tasks are started (call this from async context)"""
        if not self._background_tasks_started:
            self._start_background_tasks()

    async def track_event(
        self,
        event_type: EventType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ):
        """Track an analytics event"""
        if not self.enabled or not self._should_sample():
            return

        # Ensure background tasks are started in async context
        self.ensure_background_tasks()

        event = Event(
            id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            workspace_id=workspace_id,
            properties=properties or {},
        )

        await self.event_buffer.add_event(event)

        # Update metrics based on event
        await self._update_metrics_from_event(event)

    async def track_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """Track a performance metric"""
        if not self.enabled:
            return

        perf_metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success,
            error=error,
            context=context or {},
        )

        # Track as event
        event = Event(
            id=str(uuid.uuid4()),
            event_type=EventType.PERFORMANCE_METRIC,
            timestamp=perf_metric.timestamp,
            user_id=None,
            session_id=None,
            workspace_id=None,
            properties=perf_metric.to_dict(),
        )

        await self.performance_buffer.add_event(event)

        # Track timing metrics
        await self.metrics_collector.record_timer(
            f"operation_duration_{operation}", duration_ms
        )

        # Track success/failure counters
        if success:
            await self.metrics_collector.increment_counter(
                f"operation_success_{operation}"
            )
        else:
            await self.metrics_collector.increment_counter(
                f"operation_error_{operation}"
            )

    async def start_session(
        self, user_id: str, session_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a user session"""
        session_id = str(uuid.uuid4())

        session_info = {
            "user_id": user_id,
            "start_time": datetime.now(),
            "last_activity": datetime.now(),
            "data": session_data or {},
        }

        self.active_sessions[session_id] = session_info
        self.user_sessions[user_id].append(session_id)

        await self.track_event(
            EventType.USER_LOGIN,
            user_id=user_id,
            session_id=session_id,
            properties=session_data,
        )

        return session_id

    async def end_session(self, session_id: str):
        """End a user session"""
        session_info = self.active_sessions.pop(session_id, None)
        if not session_info:
            return

        duration = (datetime.now() - session_info["start_time"]).total_seconds()

        await self.track_event(
            EventType.USER_LOGOUT,
            user_id=session_info["user_id"],
            session_id=session_id,
            properties={"session_duration_seconds": duration},
        )

    async def update_session_activity(
        self, session_id: str, activity_data: Optional[Dict[str, Any]] = None
    ):
        """Update session activity"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["last_activity"] = datetime.now()
            if activity_data:
                self.active_sessions[session_id]["data"].update(activity_data)

    async def get_events(
        self,
        event_type: Optional[EventType] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Event]:
        """Get events with filtering"""
        events = await self.event_buffer.get_events()

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if user_id:
            events = [e for e in events if e.user_id == user_id]

        if workspace_id:
            events = [e for e in events if e.workspace_id == workspace_id]

        if start_time:
            events = [e for e in events if e.timestamp >= start_time]

        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return await self.metrics_collector.get_metrics()

    async def get_analytics_summary(
        self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get analytics summary"""
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()

        events = await self.get_events(start_time=start_time, end_time=end_time)

        # Calculate summary statistics
        summary = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "total_events": len(events),
            "unique_users": len(set(e.user_id for e in events if e.user_id)),
            "unique_sessions": len(set(e.session_id for e in events if e.session_id)),
            "unique_workspaces": len(
                set(e.workspace_id for e in events if e.workspace_id)
            ),
            "event_types": {},
            "hourly_activity": defaultdict(int),
            "daily_activity": defaultdict(int),
        }

        # Count events by type
        for event in events:
            event_type = event.event_type.value
            summary["event_types"][event_type] = (
                summary["event_types"].get(event_type, 0) + 1
            )

            # Aggregate by hour and day
            hour_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            day_key = event.timestamp.strftime("%Y-%m-%d")
            summary["hourly_activity"][hour_key] += 1
            summary["daily_activity"][day_key] += 1

        # Add current metrics
        summary["current_metrics"] = await self.get_metrics()

        return summary

    async def _update_metrics_from_event(self, event: Event):
        """Update metrics based on event"""
        # Count events by type
        await self.metrics_collector.increment_counter(
            f"events_{event.event_type.value}"
        )

        # Track user activity
        if event.user_id:
            await self.metrics_collector.increment_counter(
                "user_activity", tags={"user_id": event.user_id}
            )

        # Track workspace activity
        if event.workspace_id:
            await self.metrics_collector.increment_counter(
                "workspace_activity", tags={"workspace_id": event.workspace_id}
            )

    async def _collect_system_metrics(self):
        """Collect system metrics periodically"""
        import psutil

        while True:
            try:
                # CPU usage
                await self.metrics_collector.set_gauge(
                    "system_cpu_percent", psutil.cpu_percent()
                )

                # Memory usage
                memory = psutil.virtual_memory()
                await self.metrics_collector.set_gauge(
                    "system_memory_percent", memory.percent
                )
                await self.metrics_collector.set_gauge(
                    "system_memory_used_bytes", memory.used
                )

                # Active sessions
                await self.metrics_collector.set_gauge(
                    "active_sessions", len(self.active_sessions)
                )

                # Event buffer size
                await self.metrics_collector.set_gauge(
                    "event_buffer_size", len(self.event_buffer.events)
                )

            except Exception as e:
                print(f"Error collecting system metrics: {e}")

            await asyncio.sleep(30)  # Collect every 30 seconds

    def _should_sample(self) -> bool:
        """Determine if event should be sampled"""
        import random

        return random.random() < self.sampling_rate

    async def shutdown(self):
        """Shutdown analytics collector"""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*self._background_tasks, return_exceptions=True)


# Context manager for performance tracking
class PerformanceTimer:
    """Context manager for tracking operation performance"""

    def __init__(
        self,
        operation: str,
        analytics: "AnalyticsCollector",
        context: Optional[Dict[str, Any]] = None,
    ):
        self.operation = operation
        self.analytics = analytics
        self.context = context or {}
        self.start_time = None
        self.success = True
        self.error = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000

            if exc_type:
                self.success = False
                self.error = str(exc_val)

            await self.analytics.track_performance(
                operation=self.operation,
                duration_ms=duration_ms,
                success=self.success,
                error=self.error,
                context=self.context,
            )


# Global analytics collector instance
analytics_collector = AnalyticsCollector()
