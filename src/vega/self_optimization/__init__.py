"""Self-optimization framework for Vega."""

from .monitoring import PerformanceMonitor, monitor_task
from .parameters import ParameterStateManager
from .optimizer import AutonomousSelfOptimizer

__all__ = [
    "PerformanceMonitor",
    "monitor_task",
    "ParameterStateManager",
    "AutonomousSelfOptimizer",
]
