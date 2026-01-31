"""
health_aggregator.py - Unified Health Check System

Aggregates health status from all subsystems into a single dashboard view.
Supports:
- Component-level health checks
- Dependency health (DB, LLM, cache)
- Self-healing triggers
- Prometheus-compatible metrics export
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component"""

    name: str
    status: HealthStatus
    message: str = ""
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Aggregated system health"""

    overall_status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime
    uptime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "components": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "last_check": c.last_check.isoformat(),
                    "response_time_ms": c.response_time_ms,
                    "details": c.details,
                }
                for c in self.components
            ],
        }


class HealthAggregator:
    """
    Central health check aggregator.

    Usage:
        aggregator = HealthAggregator()
        aggregator.register("database", check_database_health)
        aggregator.register("llm", check_llm_health)

        health = await aggregator.check_all()
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._async_checks: Dict[str, Callable[[], Any]] = {}
        self._start_time = time.time()
        self._last_results: Dict[str, ComponentHealth] = {}

    def register(self, name: str, check_fn: Callable[[], ComponentHealth], is_async: bool = False) -> None:
        """Register a health check function"""
        if is_async:
            self._async_checks[name] = check_fn
        else:
            self._checks[name] = check_fn

    def register_simple(
        self, name: str, check_fn: Callable[[], bool], healthy_msg: str = "OK", unhealthy_msg: str = "Check failed"
    ) -> None:
        """Register a simple boolean health check"""

        def wrapper() -> ComponentHealth:
            start = time.time()
            try:
                result = check_fn()
                elapsed = (time.time() - start) * 1000
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message=healthy_msg if result else unhealthy_msg,
                    response_time_ms=elapsed,
                )
            except Exception as e:
                elapsed = (time.time() - start) * 1000
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                    response_time_ms=elapsed,
                )

        self._checks[name] = wrapper

    async def check_all(self, timeout: float = 5.0) -> SystemHealth:
        """Run all health checks and aggregate results"""
        components: List[ComponentHealth] = []

        # Run sync checks
        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                components.append(result)
                self._last_results[name] = result
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                components.append(
                    ComponentHealth(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Check error: {e}",
                    )
                )

        # Run async checks with timeout
        for name, check_fn in self._async_checks.items():
            try:
                start = time.time()
                result = await asyncio.wait_for(check_fn(), timeout=timeout)
                elapsed = (time.time() - start) * 1000

                if isinstance(result, ComponentHealth):
                    components.append(result)
                else:
                    # Assume boolean result
                    components.append(
                        ComponentHealth(
                            name=name,
                            status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                            message="OK" if result else "Check failed",
                            response_time_ms=elapsed,
                        )
                    )
                self._last_results[name] = components[-1]
            except asyncio.TimeoutError:
                components.append(
                    ComponentHealth(
                        name=name,
                        status=HealthStatus.DEGRADED,
                        message=f"Check timed out after {timeout}s",
                    )
                )
            except Exception as e:
                logger.error(f"Async health check '{name}' failed: {e}")
                components.append(
                    ComponentHealth(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=f"Check error: {e}",
                    )
                )

        # Calculate overall status
        overall = self._calculate_overall_status(components)

        return SystemHealth(
            overall_status=overall,
            components=components,
            timestamp=datetime.utcnow(),
            uptime_seconds=time.time() - self._start_time,
        )

    def _calculate_overall_status(self, components: List[ComponentHealth]) -> HealthStatus:
        """Calculate overall system status from components"""
        if not components:
            return HealthStatus.UNKNOWN

        statuses = [c.status for c in components]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNKNOWN

    def get_prometheus_metrics(self) -> str:
        """Export health metrics in Prometheus format"""
        lines = [
            "# HELP vega_component_health Component health status (1=healthy, 0=unhealthy)",
            "# TYPE vega_component_health gauge",
        ]

        status_value = {
            HealthStatus.HEALTHY: 1,
            HealthStatus.DEGRADED: 0.5,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: -1,
        }

        for name, result in self._last_results.items():
            value = status_value.get(result.status, -1)
            lines.append(f'vega_component_health{{component="{name}"}} {value}')
            lines.append(f'vega_component_response_ms{{component="{name}"}} {result.response_time_ms:.2f}')

        return "\n".join(lines)


# Global aggregator instance
_health_aggregator: Optional[HealthAggregator] = None


def get_health_aggregator() -> HealthAggregator:
    """Get or create the global health aggregator"""
    global _health_aggregator
    if _health_aggregator is None:
        _health_aggregator = HealthAggregator()
    return _health_aggregator


# Standard health checks for core components
def register_core_health_checks(aggregator: HealthAggregator) -> None:
    """Register standard health checks for core Vega components"""

    # Database health check
    def check_database() -> ComponentHealth:
        start = time.time()
        try:
            from .db import engine

            with engine.connect() as conn:
                conn.execute("SELECT 1")
            elapsed = (time.time() - start) * 1000
            return ComponentHealth(
                name="database",
                status=HealthStatus.HEALTHY,
                message="SQLite connection OK",
                response_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time_ms=elapsed,
            )

    # Memory system health check
    def check_memory() -> ComponentHealth:
        start = time.time()
        try:
            from .memory import memory_engine

            with memory_engine.connect() as conn:
                conn.execute("SELECT 1")
            elapsed = (time.time() - start) * 1000
            return ComponentHealth(
                name="memory",
                status=HealthStatus.HEALTHY,
                message="Memory database OK",
                response_time_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return ComponentHealth(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time_ms=elapsed,
            )

    aggregator.register("database", check_database)
    aggregator.register("memory", check_memory)
