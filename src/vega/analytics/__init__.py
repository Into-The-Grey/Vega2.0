"""
Analytics module for Vega 2.0 platform
======================================

This module provides comprehensive analytics capabilities including:
- Event tracking and data collection
- Performance monitoring
- User activity analytics
- System metrics gathering
- Real-time dashboards and reporting
"""

from .collector import (
    analytics_collector,
    AnalyticsCollector,
    EventType,
    MetricType,
    Event,
    Metric,
    PerformanceMetric,
    PerformanceTimer,
)

__all__ = [
    "analytics_collector",
    "AnalyticsCollector",
    "EventType",
    "MetricType",
    "Event",
    "Metric",
    "PerformanceMetric",
    "PerformanceTimer",
]


# Convenience functions for common analytics operations
async def track_user_activity(user_id: str, activity: str, **properties):
    """Track user activity event"""
    await analytics_collector.track_event(
        EventType.USER_ACTIVITY,
        user_id=user_id,
        properties={"activity": activity, **properties},
    )


async def track_api_request(
    endpoint: str, method: str, user_id: str = None, **properties
):
    """Track API request event"""
    await analytics_collector.track_event(
        EventType.API_REQUEST,
        user_id=user_id,
        properties={"endpoint": endpoint, "method": method, **properties},
    )


async def track_error(
    error_type: str, error_message: str, user_id: str = None, **properties
):
    """Track error event"""
    await analytics_collector.track_event(
        EventType.ERROR_OCCURRED,
        user_id=user_id,
        properties={
            "error_type": error_type,
            "error_message": error_message,
            **properties,
        },
    )


def performance_timer(operation: str, **context):
    """Create performance timer context manager"""
    return PerformanceTimer(operation, analytics_collector, context)
