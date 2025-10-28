"""
Advanced Performance Monitoring Endpoints for Vega2.0

Exposes all new performance optimization systems through REST API:
- Enhanced circuit breakers
- Response caching
- Streaming backpressure
- Async event loop monitoring
- Memory leak detection
- Batch operations
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# This module should be imported in app.py and registered with:
# app.include_router(performance_router)

router = APIRouter(prefix="/admin/performance", tags=["performance"])


# Response models
class CircuitBreakerStatus(BaseModel):
    """Circuit breaker status response"""

    state: str
    fail_count: int
    consecutive_failures: int
    seconds_until_retry: float
    metrics: dict


class CacheStats(BaseModel):
    """Cache statistics response"""

    size: int
    maxsize: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int


class StreamMetrics(BaseModel):
    """Stream backpressure metrics"""

    state: str
    buffer_usage_percent: float
    chunks_produced: int
    chunks_consumed: int
    throughput_chunks_per_sec: float


class EventLoopHealth(BaseModel):
    """Event loop health status"""

    health_status: str
    slow_callbacks_count: int
    current_pending_tasks: int
    max_pending_tasks: int
    loop_stalls: int


class MemoryLeakReport(BaseModel):
    """Memory leak detection report"""

    objects_currently_alive: int
    potential_leaks: int
    memory_tracked_mb: float
    process_memory_rss_mb: float


# Circuit Breaker Endpoints
@router.get("/circuit-breaker/{integration_name}/status")
async def get_circuit_breaker_status(integration_name: str):
    """
    Get circuit breaker status for a specific integration.

    Available integrations: llm, search, fetch, osint, slack, homeassistant
    """
    try:
        from ..enhanced_resilience import EnhancedCircuitBreaker

        # Get circuit breaker for integration (implementation specific)
        # This is a placeholder - actual implementation depends on how
        # circuit breakers are registered per integration
        breaker = _get_circuit_breaker(integration_name)
        if not breaker:
            raise HTTPException(
                status_code=404, detail=f"Circuit breaker not found: {integration_name}"
            )

        status = await breaker.get_status()
        return status

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/circuit-breaker/{integration_name}/reset")
async def reset_circuit_breaker(integration_name: str):
    """
    Manually reset a circuit breaker to CLOSED state.

    Useful for recovery after fixing underlying issues.
    """
    try:
        breaker = _get_circuit_breaker(integration_name)
        if not breaker:
            raise HTTPException(
                status_code=404, detail=f"Circuit breaker not found: {integration_name}"
            )

        await breaker.reset()
        return {"status": "reset", "integration": integration_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/circuit-breakers/all")
async def get_all_circuit_breakers():
    """Get status of all circuit breakers in the system."""
    try:
        breakers = _get_all_circuit_breakers()
        statuses = {}

        for name, breaker in breakers.items():
            statuses[name] = await breaker.get_status()

        return {"circuit_breakers": statuses}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Response Cache Endpoints
@router.get("/cache/stats")
async def get_cache_stats():
    """Get LLM response cache statistics."""
    try:
        cache = _get_response_cache()
        if not cache:
            return {"error": "Cache not initialized"}

        stats = await cache.get_stats()
        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/clear")
async def clear_cache():
    """Clear all cached LLM responses."""
    try:
        cache = _get_response_cache()
        if not cache:
            return {"error": "Cache not initialized"}

        await cache.clear()
        return {"status": "cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Event Loop Monitoring Endpoints
@router.get("/event-loop/status")
async def get_event_loop_status():
    """Get comprehensive event loop health status."""
    try:
        from ..async_monitor import get_event_loop_monitor

        monitor = await get_event_loop_monitor()
        metrics = await monitor.get_metrics()
        health = await monitor.get_health_status()

        return {"health": health, "metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/event-loop/diagnostics")
async def get_event_loop_diagnostics():
    """Run comprehensive event loop diagnostics."""
    try:
        from ..async_monitor import diagnose_event_loop

        report = await diagnose_event_loop()
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Memory Leak Detection Endpoints
@router.get("/memory/leaks")
async def get_memory_leak_report():
    """Get memory leak detection report."""
    try:
        from ..memory_leak_detector import get_memory_leak_detector

        detector = await get_memory_leak_detector()
        metrics = await detector.get_metrics()

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/leaks/{object_type}")
async def get_leaked_objects(object_type: str):
    """Get details of potentially leaked objects of a specific type."""
    try:
        from ..memory_leak_detector import get_memory_leak_detector

        detector = await get_memory_leak_detector()
        leaked = await detector.get_leaked_objects(obj_type=object_type)

        return {"type": object_type, "leaked_objects": leaked, "count": len(leaked)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memory/gc")
async def force_garbage_collection():
    """Force garbage collection and memory cleanup."""
    try:
        from ..memory_leak_detector import get_memory_leak_detector

        detector = await get_memory_leak_detector()
        await detector.force_cleanup()

        metrics = await detector.get_metrics()
        return {
            "status": "completed",
            "objects_freed": metrics["objects_freed"],
            "memory_tracked_mb": metrics["memory_tracked_mb"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/memory/diagnostics")
async def get_memory_diagnostics():
    """Run comprehensive memory diagnostics."""
    try:
        from ..memory_leak_detector import diagnose_memory_leaks

        report = await diagnose_memory_leaks()
        return report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch Operations Endpoints
@router.get("/batch/conversation-logger/stats")
async def get_batch_logger_stats():
    """Get batched conversation logger statistics."""
    try:
        from ..batch_operations import get_batched_logger

        logger = await get_batched_logger()
        metrics = await logger.get_metrics()

        return metrics

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch/conversation-logger/flush")
async def flush_batch_logger():
    """Force flush of pending batched conversations."""
    try:
        from ..batch_operations import get_batched_logger

        logger = await get_batched_logger()
        await logger.flush()

        return {"status": "flushed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Comprehensive Health Check
@router.get("/health/comprehensive")
async def comprehensive_health_check():
    """
    Run comprehensive health check across all performance systems.

    Returns status for:
    - Circuit breakers
    - Response cache
    - Event loop
    - Memory
    - Batch operations
    """
    try:
        health_report = {
            "timestamp": __import__("time").time(),
            "overall_status": "healthy",
            "systems": {},
        }

        # Check circuit breakers
        try:
            breakers = _get_all_circuit_breakers()
            open_breakers = []
            for name, breaker in breakers.items():
                status = await breaker.get_status()
                if status["state"] == "open":
                    open_breakers.append(name)

            health_report["systems"]["circuit_breakers"] = {
                "status": "healthy" if not open_breakers else "degraded",
                "open_breakers": open_breakers,
                "total_breakers": len(breakers),
            }

            if open_breakers:
                health_report["overall_status"] = "degraded"

        except Exception as e:
            health_report["systems"]["circuit_breakers"] = {
                "status": "error",
                "error": str(e),
            }

        # Check event loop
        try:
            from ..async_monitor import get_event_loop_monitor

            monitor = await get_event_loop_monitor()
            loop_health = await monitor.get_health_status()

            health_report["systems"]["event_loop"] = {
                "status": loop_health,
            }

            if loop_health != "healthy":
                health_report["overall_status"] = "degraded"

        except Exception as e:
            health_report["systems"]["event_loop"] = {
                "status": "error",
                "error": str(e),
            }

        # Check memory
        try:
            from ..memory_leak_detector import get_memory_leak_detector

            detector = await get_memory_leak_detector()
            memory_metrics = await detector.get_metrics()

            potential_leaks = memory_metrics["potential_leaks"]
            health_report["systems"]["memory"] = {
                "status": "healthy" if potential_leaks < 10 else "warning",
                "potential_leaks": potential_leaks,
                "tracked_memory_mb": memory_metrics["memory_tracked_mb"],
            }

            if potential_leaks >= 10:
                health_report["overall_status"] = "degraded"

        except Exception as e:
            health_report["systems"]["memory"] = {"status": "error", "error": str(e)}

        # Check batch operations
        try:
            from ..batch_operations import get_batched_logger

            logger = await get_batched_logger()
            batch_metrics = await logger.get_metrics()

            queue_size = batch_metrics["queue_size"]
            max_queue = batch_metrics["max_queue_size"]

            health_report["systems"]["batch_operations"] = {
                "status": "healthy" if queue_size < max_queue * 0.8 else "warning",
                "queue_size": queue_size,
                "max_queue_size": max_queue,
                "success_rate": batch_metrics["success_rate"],
            }

        except Exception as e:
            health_report["systems"]["batch_operations"] = {
                "status": "error",
                "error": str(e),
            }

        # Check cache
        try:
            cache = _get_response_cache()
            if cache:
                cache_stats = await cache.get_stats()
                health_report["systems"]["cache"] = {
                    "status": "healthy",
                    "hit_rate": cache_stats["hit_rate"],
                    "size": cache_stats["size"],
                }
        except Exception as e:
            health_report["systems"]["cache"] = {"status": "error", "error": str(e)}

        return health_report

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions (these should be implemented based on actual integration)
def _get_circuit_breaker(name: str):
    """Get circuit breaker instance by name"""
    # Placeholder - actual implementation depends on circuit breaker registry
    # Could use a global registry dict or inspect decorated functions
    return None


def _get_all_circuit_breakers():
    """Get all registered circuit breakers"""
    # Placeholder - return dict of {name: breaker}
    return {}


def _get_response_cache():
    """Get response cache instance"""
    # Placeholder - should return the actual cache instance
    # from wherever it's initialized (e.g., in llm.py)
    return None
