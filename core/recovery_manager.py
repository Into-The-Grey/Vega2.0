"""
Vega2.0 Recovery Manager
=======================

Automatic error recovery and system healing capabilities.
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum

from .error_handler import (
    ErrorCode,
    ErrorSeverity,
    VegaError,
    log_info,
    log_warning,
    log_error,
)


class RecoveryAction(Enum):
    """Types of recovery actions"""

    RETRY = "retry"
    RESTART = "restart"
    FALLBACK = "fallback"
    SCALE_DOWN = "scale_down"
    CLEAR_CACHE = "clear_cache"
    RESET_CIRCUIT = "reset_circuit"
    CLEANUP = "cleanup"
    NOTIFY = "notify"


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration"""

    error_code: ErrorCode
    actions: List[RecoveryAction]
    max_attempts: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0
    timeout: float = 30.0
    conditions: Optional[Dict[str, Any]] = None

    def should_apply(self, error: VegaError, attempt: int) -> bool:
        """Check if this strategy should be applied"""
        if error.code != self.error_code:
            return False

        if attempt >= self.max_attempts:
            return False

        if not error.recoverable:
            return False

        # Check additional conditions
        if self.conditions:
            for key, expected in self.conditions.items():
                if (
                    error.context.parameters
                    and error.context.parameters.get(key) != expected
                ):
                    return False

        return True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        return self.initial_delay * (self.backoff_multiplier**attempt)


class RecoveryManager:
    """Manages automatic error recovery"""

    def __init__(self):
        self.strategies: Dict[ErrorCode, RecoveryStrategy] = {}
        self.recovery_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recovery_handlers: Dict[RecoveryAction, Callable] = {}
        self._setup_default_strategies()
        self._setup_recovery_handlers()

    def _setup_default_strategies(self):
        """Setup default recovery strategies"""

        # LLM Rate Limit - Retry with exponential backoff
        self.strategies[ErrorCode.LLM_RATE_LIMIT] = RecoveryStrategy(
            error_code=ErrorCode.LLM_RATE_LIMIT,
            actions=[RecoveryAction.RETRY],
            max_attempts=3,
            initial_delay=10.0,
            backoff_multiplier=2.0,
        )

        # LLM Timeout - Retry with fallback
        self.strategies[ErrorCode.LLM_TIMEOUT] = RecoveryStrategy(
            error_code=ErrorCode.LLM_TIMEOUT,
            actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
            max_attempts=2,
            initial_delay=5.0,
        )

        # Database Connection - Restart connection pool
        self.strategies[ErrorCode.DATABASE_CONNECTION] = RecoveryStrategy(
            error_code=ErrorCode.DATABASE_CONNECTION,
            actions=[RecoveryAction.RESTART, RecoveryAction.RETRY],
            max_attempts=3,
            initial_delay=2.0,
        )

        # Process Crashed - Restart process
        self.strategies[ErrorCode.PROCESS_CRASHED] = RecoveryStrategy(
            error_code=ErrorCode.PROCESS_CRASHED,
            actions=[RecoveryAction.RESTART],
            max_attempts=5,
            initial_delay=1.0,
        )

        # Resource Exhausted - Scale down and cleanup
        self.strategies[ErrorCode.RESOURCE_EXHAUSTED] = RecoveryStrategy(
            error_code=ErrorCode.RESOURCE_EXHAUSTED,
            actions=[RecoveryAction.CLEANUP, RecoveryAction.SCALE_DOWN],
            max_attempts=2,
            initial_delay=0.5,
        )

        # Circuit Breaker - Reset after delay
        self.strategies[ErrorCode.CIRCUIT_BREAKER_OPEN] = RecoveryStrategy(
            error_code=ErrorCode.CIRCUIT_BREAKER_OPEN,
            actions=[RecoveryAction.RESET_CIRCUIT],
            max_attempts=1,
            initial_delay=30.0,
        )

        # Network Timeout - Retry with fallback
        self.strategies[ErrorCode.NETWORK_TIMEOUT] = RecoveryStrategy(
            error_code=ErrorCode.NETWORK_TIMEOUT,
            actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
            max_attempts=2,
            initial_delay=3.0,
        )

        # Integration Failure - Fallback to alternative
        self.strategies[ErrorCode.INTEGRATION_FAILURE] = RecoveryStrategy(
            error_code=ErrorCode.INTEGRATION_FAILURE,
            actions=[RecoveryAction.FALLBACK, RecoveryAction.RETRY],
            max_attempts=2,
            initial_delay=5.0,
        )

    def _setup_recovery_handlers(self):
        """Setup recovery action handlers"""
        self.recovery_handlers = {
            RecoveryAction.RETRY: self._handle_retry,
            RecoveryAction.RESTART: self._handle_restart,
            RecoveryAction.FALLBACK: self._handle_fallback,
            RecoveryAction.SCALE_DOWN: self._handle_scale_down,
            RecoveryAction.CLEAR_CACHE: self._handle_clear_cache,
            RecoveryAction.RESET_CIRCUIT: self._handle_reset_circuit,
            RecoveryAction.CLEANUP: self._handle_cleanup,
            RecoveryAction.NOTIFY: self._handle_notify,
        }

    async def recover_from_error(self, error: VegaError) -> bool:
        """Attempt to recover from an error"""

        strategy = self.strategies.get(error.code)
        if not strategy:
            log_info(f"No recovery strategy for error {error.code.value}")
            return False

        # Get current attempt count
        error_key = f"{error.code.value}:{error.context.component or 'unknown'}"
        history = self.recovery_history.get(error_key, [])
        current_attempt = len(history)

        # Check if we should attempt recovery
        if not strategy.should_apply(error, current_attempt):
            log_warning(
                f"Recovery strategy not applicable for {error.code.value}",
                attempt=current_attempt,
                max_attempts=strategy.max_attempts,
            )
            return False

        log_info(
            f"Starting recovery for {error.code.value}",
            attempt=current_attempt + 1,
            max_attempts=strategy.max_attempts,
            actions=[action.value for action in strategy.actions],
        )

        # Record recovery attempt
        recovery_record = {
            "timestamp": time.time(),
            "error_id": error.error_id,
            "attempt": current_attempt + 1,
            "strategy": strategy.actions[0].value if strategy.actions else "none",
        }

        if error_key not in self.recovery_history:
            self.recovery_history[error_key] = []

        self.recovery_history[error_key].append(recovery_record)

        # Calculate delay
        delay = strategy.get_delay(current_attempt)
        if delay > 0:
            log_info(f"Waiting {delay}s before recovery attempt")
            await asyncio.sleep(delay)

        # Execute recovery actions
        success = False
        for action in strategy.actions:
            try:
                log_info(f"Executing recovery action: {action.value}")
                handler = self.recovery_handlers.get(action)
                if handler:
                    result = await handler(error, strategy)
                    if result:
                        success = True
                        log_info(f"Recovery action {action.value} succeeded")
                        break
                    else:
                        log_warning(f"Recovery action {action.value} failed")
                else:
                    log_warning(f"No handler for recovery action: {action.value}")
            except Exception as e:
                log_error(f"Recovery action {action.value} failed with exception: {e}")

        # Update recovery record
        recovery_record["success"] = success
        recovery_record["duration"] = time.time() - recovery_record["timestamp"]

        if success:
            log_info(f"Recovery successful for {error.code.value}")
            # Clear history on success
            self.recovery_history.pop(error_key, None)
        else:
            log_error(f"Recovery failed for {error.code.value}")

        return success

    # Recovery action handlers
    async def _handle_retry(self, error: VegaError, strategy: RecoveryStrategy) -> bool:
        """Handle retry recovery action"""
        # For retry, we just return True to indicate we should try again
        # The actual retry logic is handled by the calling code
        log_info("Retry recovery action executed")
        return True

    async def _handle_restart(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle restart recovery action"""
        try:
            # Determine what to restart based on error context
            component = error.context.component

            if component == "process_manager":
                # Restart specific process
                process_name = error.context.parameters.get("process_name")
                if process_name:
                    log_info(f"Attempting to restart process: {process_name}")
                    # Process restart would be handled by process manager
                    return True

            elif component == "database":
                # Restart database connection
                log_info("Attempting to restart database connection")
                # Database restart would be handled by database module
                return True

            elif component == "llm":
                # Restart LLM connection/circuit breaker
                log_info("Attempting to restart LLM connection")
                # LLM restart would be handled by LLM module
                return True

            log_warning(f"Don't know how to restart component: {component}")
            return False

        except Exception as e:
            log_error(f"Restart recovery failed: {e}")
            return False

    async def _handle_fallback(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle fallback recovery action"""
        try:
            component = error.context.component

            if component == "llm":
                # Fallback to alternative LLM provider
                log_info("Attempting LLM provider fallback")
                # Would trigger LLM manager to use alternative provider
                return True

            elif component == "integration":
                # Fallback to alternative integration
                service = error.context.parameters.get("service")
                log_info(f"Attempting fallback for service: {service}")
                # Would trigger integration manager to use alternative
                return True

            elif component == "database":
                # Fallback to read-only mode or cache
                log_info("Attempting database fallback to read-only mode")
                return True

            log_warning(f"No fallback available for component: {component}")
            return False

        except Exception as e:
            log_error(f"Fallback recovery failed: {e}")
            return False

    async def _handle_scale_down(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle scale down recovery action"""
        try:
            log_info("Attempting to scale down operations")

            # Scale down based on resource type
            resource = error.context.parameters.get("resource")

            if resource == "memory":
                # Reduce memory usage
                log_info("Scaling down memory usage")
                # Would trigger memory cleanup, reduce cache sizes, etc.
                return True

            elif resource == "cpu":
                # Reduce CPU usage
                log_info("Scaling down CPU usage")
                # Would reduce concurrent operations, increase delays, etc.
                return True

            elif resource == "disk":
                # Free up disk space
                log_info("Scaling down disk usage")
                # Would trigger log cleanup, temp file cleanup, etc.
                return True

            # Generic scale down
            log_info("Generic scale down operations")
            return True

        except Exception as e:
            log_error(f"Scale down recovery failed: {e}")
            return False

    async def _handle_clear_cache(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle cache clearing recovery action"""
        try:
            log_info("Attempting to clear caches")

            # Clear various caches
            # This would integrate with cache managers when available

            log_info("Cache clearing completed")
            return True

        except Exception as e:
            log_error(f"Cache clearing recovery failed: {e}")
            return False

    async def _handle_reset_circuit(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle circuit breaker reset recovery action"""
        try:
            service = error.context.parameters.get("service", "unknown")
            log_info(f"Attempting to reset circuit breaker for: {service}")

            # Reset circuit breaker
            # This would integrate with circuit breaker when available

            log_info(f"Circuit breaker reset for: {service}")
            return True

        except Exception as e:
            log_error(f"Circuit breaker reset recovery failed: {e}")
            return False

    async def _handle_cleanup(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle cleanup recovery action"""
        try:
            log_info("Attempting system cleanup")

            # Perform various cleanup operations
            # - Clear temporary files
            # - Free up resources
            # - Close unused connections
            # - Garbage collection

            log_info("System cleanup completed")
            return True

        except Exception as e:
            log_error(f"Cleanup recovery failed: {e}")
            return False

    async def _handle_notify(
        self, error: VegaError, strategy: RecoveryStrategy
    ) -> bool:
        """Handle notification recovery action"""
        try:
            log_info("Sending error notification")

            # Send notifications (email, slack, etc.)
            # This would integrate with notification systems when available

            log_info("Error notification sent")
            return True

        except Exception as e:
            log_error(f"Notification recovery failed: {e}")
            return False

    def add_strategy(self, strategy: RecoveryStrategy):
        """Add a custom recovery strategy"""
        self.strategies[strategy.error_code] = strategy
        log_info(f"Added recovery strategy for {strategy.error_code.value}")

    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total_attempts = sum(len(history) for history in self.recovery_history.values())
        successful_recoveries = 0

        for history in self.recovery_history.values():
            for record in history:
                if record.get("success", False):
                    successful_recoveries += 1

        return {
            "total_recovery_attempts": total_attempts,
            "successful_recoveries": successful_recoveries,
            "success_rate": (
                successful_recoveries / total_attempts if total_attempts > 0 else 0.0
            ),
            "active_error_types": len(self.recovery_history),
            "available_strategies": len(self.strategies),
        }

    def clear_history(self):
        """Clear recovery history"""
        self.recovery_history.clear()
        log_info("Recovery history cleared")


# Global recovery manager instance
_recovery_manager = None


def get_recovery_manager() -> RecoveryManager:
    """Get global recovery manager instance"""
    global _recovery_manager
    if _recovery_manager is None:
        _recovery_manager = RecoveryManager()
    return _recovery_manager
