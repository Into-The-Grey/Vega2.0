"""Entry points for fully autonomous self-optimization cycles."""

from __future__ import annotations

import asyncio
import signal
from typing import Awaitable, Callable, Mapping

from .ide_bridge import IDEActivityLogger, LiveViewServer
from .optimizer import AutonomousSelfOptimizer


async def run_self_optimization_loop(
    evaluator: Callable[[Mapping[str, str]], Awaitable[float]],
    cycle_interval_sec: float = 900.0,
    improvement_threshold: float = 0.01,
) -> None:
    """Continuously execute optimization cycles until process termination."""

    logger = IDEActivityLogger()
    live_server = LiveViewServer()
    await live_server.start()
    optimizer = AutonomousSelfOptimizer(evaluator)

    stop_event = asyncio.Event()

    def _handle_stop(*_: object) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_stop)

    logger.log("Self-optimization loop started", interval=cycle_interval_sec)
    await live_server.publish({"event": "startup", "interval_sec": cycle_interval_sec})

    while not stop_event.is_set():
        logger.log("Running optimization cycle")
        best_params = await optimizer.run_cycle(
            iterations=5, improvement_threshold=improvement_threshold
        )
        await live_server.publish({"event": "cycle_complete", "params": best_params})
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=cycle_interval_sec)
        except asyncio.TimeoutError:
            continue

    await live_server.publish({"event": "shutdown"})
    await live_server.stop()
    logger.log("Self-optimization loop stopped")
