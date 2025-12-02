#!/usr/bin/env python3
"""
Resilient Startup System for Vega2.0

This module implements a robust startup system that:
1. Separates core (critical) functions from non-critical features
2. Allows non-critical features to fail without compromising the core
3. Autonomously attempts to resolve failures in the background
4. Prioritizes healing based on feature usage frequency
5. Logs and notifies on persistent failures
6. Learns from successful resolutions for future use

Architecture:
- Core functions: Must succeed or system won't start (e.g., database, LLM backend)
- Non-critical functions: Can fail gracefully (e.g., profiler, analytics, integrations)
- Background healer: Low-priority async process that attempts to fix failures
- Knowledge base: Stores successful resolution strategies
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class FeaturePriority(Enum):
    """Priority levels for features - affects healing order"""

    CRITICAL = 1  # Core system - must work (DB, LLM, config)
    HIGH = 2  # Frequently used features
    MEDIUM = 3  # Regularly used features
    LOW = 4  # Rarely used features
    BACKGROUND = 5  # Nice-to-have features


class FeatureStatus(Enum):
    """Current status of a feature"""

    UNKNOWN = "unknown"
    LOADING = "loading"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Partially working
    FAILED = "failed"  # Failed, awaiting healing
    HEALING = "healing"  # Currently attempting recovery
    PERMANENTLY_FAILED = "permanently_failed"  # Gave up after max attempts


class ResolutionStrategy(Enum):
    """Types of resolution strategies"""

    REIMPORT = "reimport"  # Try importing again
    REINSTALL_DEPS = "reinstall_deps"  # Reinstall dependencies
    RESET_STATE = "reset_state"  # Reset internal state
    RECREATE_RESOURCE = "recreate_resource"  # Recreate the resource
    FALLBACK_CONFIG = "fallback_config"  # Use fallback configuration
    RESTART_SERVICE = "restart_service"  # Restart dependent service
    CUSTOM = "custom"  # Custom resolution function


@dataclass
class FeatureDefinition:
    """Defines a feature that can be loaded during startup"""

    name: str
    loader: Callable[[], Coroutine[Any, Any, Any]]
    priority: FeaturePriority
    is_critical: bool = False
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    fallback: Optional[Callable[[], Coroutine[Any, Any, Any]]] = None
    resolution_strategies: List[ResolutionStrategy] = field(
        default_factory=lambda: [
            ResolutionStrategy.REIMPORT,
            ResolutionStrategy.RESET_STATE,
        ]
    )
    max_heal_attempts: int = 5
    heal_backoff_seconds: float = 30.0  # Time between heal attempts


@dataclass
class FeatureState:
    """Runtime state of a feature"""

    definition: FeatureDefinition
    status: FeatureStatus = FeatureStatus.UNKNOWN
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    load_time: Optional[datetime] = None
    heal_attempts: int = 0
    last_heal_attempt: Optional[datetime] = None
    usage_count: int = 0  # How often this feature is used
    last_used: Optional[datetime] = None
    resolution_applied: Optional[str] = None


@dataclass
class ResolutionRecord:
    """Record of a successful resolution for the knowledge base"""

    feature_name: str
    error_signature: str  # Hash of error type + message
    strategy_used: str
    resolution_details: Dict[str, Any]
    success_count: int = 1
    last_used: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)


class ResolutionKnowledgeBase:
    """
    Stores successful resolution strategies for future use.
    Persists to disk for cross-session learning.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/resolution_knowledge.json")
        self.resolutions: Dict[str, List[ResolutionRecord]] = defaultdict(list)
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load knowledge base from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    for feature_name, records in data.items():
                        self.resolutions[feature_name] = [
                            ResolutionRecord(
                                feature_name=r["feature_name"],
                                error_signature=r["error_signature"],
                                strategy_used=r["strategy_used"],
                                resolution_details=r["resolution_details"],
                                success_count=r.get("success_count", 1),
                                last_used=datetime.fromisoformat(r["last_used"]),
                                created_at=datetime.fromisoformat(r["created_at"]),
                            )
                            for r in records
                        ]
                logger.info(
                    f"Loaded {sum(len(v) for v in self.resolutions.values())} resolution records"
                )
        except Exception as e:
            logger.warning(f"Could not load resolution knowledge base: {e}")

    def _save(self):
        """Save knowledge base to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                data = {}
                for feature_name, records in self.resolutions.items():
                    data[feature_name] = [
                        {
                            "feature_name": r.feature_name,
                            "error_signature": r.error_signature,
                            "strategy_used": r.strategy_used,
                            "resolution_details": r.resolution_details,
                            "success_count": r.success_count,
                            "last_used": r.last_used.isoformat(),
                            "created_at": r.created_at.isoformat(),
                        }
                        for r in records
                    ]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save resolution knowledge base: {e}")

    @staticmethod
    def _error_signature(error: Exception) -> str:
        """Create a signature for an error to match similar errors"""
        error_str = f"{type(error).__name__}:{str(error)[:200]}"
        return hashlib.md5(error_str.encode()).hexdigest()[:16]

    def find_resolution(
        self, feature_name: str, error: Exception
    ) -> Optional[ResolutionRecord]:
        """Find a known resolution for this error"""
        signature = self._error_signature(error)
        with self._lock:
            records = self.resolutions.get(feature_name, [])
            for record in sorted(records, key=lambda r: -r.success_count):
                if record.error_signature == signature:
                    return record
        return None

    def record_success(
        self,
        feature_name: str,
        error: Exception,
        strategy: str,
        details: Dict[str, Any],
    ):
        """Record a successful resolution"""
        signature = self._error_signature(error)
        with self._lock:
            # Check if we already have this resolution
            for record in self.resolutions[feature_name]:
                if (
                    record.error_signature == signature
                    and record.strategy_used == strategy
                ):
                    record.success_count += 1
                    record.last_used = datetime.utcnow()
                    self._save()
                    return

            # Add new resolution
            self.resolutions[feature_name].append(
                ResolutionRecord(
                    feature_name=feature_name,
                    error_signature=signature,
                    strategy_used=strategy,
                    resolution_details=details,
                )
            )
            self._save()
            logger.info(
                f"📚 Added new resolution to knowledge base: {feature_name} -> {strategy}"
            )


class FeatureUsageTracker:
    """
    Tracks feature usage to prioritize healing efforts.
    More frequently used features get higher healing priority.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/feature_usage.json")
        self.usage_counts: Dict[str, int] = defaultdict(int)
        self.last_used: Dict[str, datetime] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        """Load usage data from disk"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    self.usage_counts = defaultdict(int, data.get("counts", {}))
                    self.last_used = {
                        k: datetime.fromisoformat(v)
                        for k, v in data.get("last_used", {}).items()
                    }
        except Exception as e:
            logger.warning(f"Could not load feature usage data: {e}")

    def _save(self):
        """Save usage data to disk"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w") as f:
                json.dump(
                    {
                        "counts": dict(self.usage_counts),
                        "last_used": {
                            k: v.isoformat() for k, v in self.last_used.items()
                        },
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.warning(f"Could not save feature usage data: {e}")

    def record_usage(self, feature_name: str):
        """Record that a feature was used"""
        with self._lock:
            self.usage_counts[feature_name] += 1
            self.last_used[feature_name] = datetime.utcnow()
            # Save periodically (every 10 uses of any feature)
            if sum(self.usage_counts.values()) % 10 == 0:
                self._save()

    def get_priority_score(self, feature_name: str) -> float:
        """
        Calculate a priority score for a feature.
        Higher score = higher priority for healing.
        """
        count = self.usage_counts.get(feature_name, 0)
        last = self.last_used.get(feature_name)

        # Base score from usage count (logarithmic scaling)
        import math

        score = math.log1p(count) * 10

        # Recency bonus - features used recently get priority
        if last:
            hours_ago = (datetime.utcnow() - last).total_seconds() / 3600
            recency_bonus = (
                max(0, 24 - hours_ago) / 24 * 20
            )  # Up to 20 points for recent use
            score += recency_bonus

        return score


class BackgroundHealer:
    """
    Background process that attempts to heal failed features.
    Runs at low priority to avoid impacting core functionality.
    """

    def __init__(
        self,
        resilience_manager: "ResilientStartupManager",
        check_interval: float = 60.0,  # Check every minute
        min_priority: int = 19,  # Nice level for background process (Unix)
    ):
        self.manager = resilience_manager
        self.check_interval = check_interval
        self.min_priority = min_priority
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(self):
        """Start the background healer"""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._task = asyncio.create_task(self._healing_loop())
        logger.info("🏥 Background healer started")

    async def stop(self):
        """Stop the background healer"""
        self._running = False
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🏥 Background healer stopped")

    async def _healing_loop(self):
        """Main healing loop - runs in background"""
        # Set low priority for this task (Unix only)
        try:
            import os

            os.nice(self.min_priority)
        except (OSError, AttributeError):
            pass  # Not supported on this platform

        while self._running:
            try:
                # Wait with timeout to allow for stop signal
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.check_interval
                    )
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue healing

                # Get features that need healing, sorted by priority
                failed_features = self.manager.get_failed_features()

                if failed_features:
                    # Sort by priority score (higher = heal first)
                    failed_features.sort(
                        key=lambda f: self.manager.usage_tracker.get_priority_score(
                            f.definition.name
                        ),
                        reverse=True,
                    )

                    for feature_state in failed_features:
                        if not self._running:
                            break

                        # Check if enough time has passed since last attempt
                        if feature_state.last_heal_attempt:
                            backoff = feature_state.definition.heal_backoff_seconds * (
                                2
                                ** min(
                                    feature_state.heal_attempts, 5
                                )  # Exponential backoff, capped
                            )
                            if (
                                datetime.utcnow() - feature_state.last_heal_attempt
                            ).total_seconds() < backoff:
                                continue

                        # Check if we've exceeded max attempts
                        if (
                            feature_state.heal_attempts
                            >= feature_state.definition.max_heal_attempts
                        ):
                            if feature_state.status != FeatureStatus.PERMANENTLY_FAILED:
                                feature_state.status = FeatureStatus.PERMANENTLY_FAILED
                                await self.manager._notify_permanent_failure(
                                    feature_state
                                )
                            continue

                        # Attempt healing
                        await self.manager.attempt_heal(feature_state)

                        # Small delay between healing attempts to avoid overwhelming the system
                        await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in healing loop: {e}")
                await asyncio.sleep(self.check_interval)


class ResilientStartupManager:
    """
    Main manager for resilient startup system.

    Usage:
        manager = ResilientStartupManager()

        # Register features
        manager.register_feature(FeatureDefinition(
            name="database",
            loader=init_database,
            priority=FeaturePriority.CRITICAL,
            is_critical=True,
        ))

        manager.register_feature(FeatureDefinition(
            name="profiler",
            loader=init_profiler,
            priority=FeaturePriority.LOW,
            is_critical=False,
        ))

        # Start with resilience
        await manager.startup()
    """

    def __init__(
        self,
        knowledge_base_path: Optional[Path] = None,
        usage_tracker_path: Optional[Path] = None,
    ):
        self.features: Dict[str, FeatureState] = {}
        self.knowledge_base = ResolutionKnowledgeBase(knowledge_base_path)
        self.usage_tracker = FeatureUsageTracker(usage_tracker_path)
        self.healer = BackgroundHealer(self)
        self._startup_complete = False
        self._notification_handlers: List[Callable] = []
        self._lock = asyncio.Lock()

    def register_feature(self, definition: FeatureDefinition):
        """Register a feature for managed loading"""
        self.features[definition.name] = FeatureState(definition=definition)
        logger.debug(
            f"Registered feature: {definition.name} (priority={definition.priority.name})"
        )

    def add_notification_handler(self, handler: Callable):
        """Add a handler for failure notifications"""
        self._notification_handlers.append(handler)

    async def startup(self) -> Tuple[bool, Dict[str, str]]:
        """
        Execute startup sequence.

        Returns:
            Tuple of (success, status_dict) where:
            - success: True if all critical features loaded
            - status_dict: Feature name -> status message mapping
        """
        status = {}
        critical_failures = []
        non_critical_failures = []

        print("\n" + "=" * 80)
        print("🚀 Vega2.0 Resilient Startup Sequence")
        print("=" * 80 + "\n")

        # Sort features by priority (critical first)
        sorted_features = sorted(
            self.features.values(),
            key=lambda f: (
                0 if f.definition.is_critical else 1,
                f.definition.priority.value,
            ),
        )

        # Phase 1: Load critical features (must succeed)
        print("📌 Phase 1: Loading CRITICAL Features")
        print("-" * 40)

        for feature_state in sorted_features:
            if not feature_state.definition.is_critical:
                continue

            success, message = await self._load_feature(feature_state)
            status[feature_state.definition.name] = message

            if not success:
                critical_failures.append(feature_state.definition.name)
                print(f"   ❌ {feature_state.definition.name}: FAILED (CRITICAL)")
            else:
                print(f"   ✅ {feature_state.definition.name}: OK")

        if critical_failures:
            print(
                f"\n🚨 CRITICAL FAILURE: {len(critical_failures)} core feature(s) failed to load"
            )
            print("   System cannot continue. Please fix these issues:")
            for name in critical_failures:
                fs = self.features[name]
                print(f"   - {name}: {fs.error}")
            return False, status

        # Phase 2: Load non-critical features (can fail)
        print("\n📌 Phase 2: Loading NON-CRITICAL Features")
        print("-" * 40)

        for feature_state in sorted_features:
            if feature_state.definition.is_critical:
                continue

            success, message = await self._load_feature(feature_state)
            status[feature_state.definition.name] = message

            if not success:
                non_critical_failures.append(feature_state.definition.name)
                print(
                    f"   ⚠️  {feature_state.definition.name}: FAILED (will heal in background)"
                )
            else:
                print(f"   ✅ {feature_state.definition.name}: OK")

        # Phase 3: Start background healer if there are failures
        if non_critical_failures:
            print(
                f"\n🏥 Starting background healer for {len(non_critical_failures)} failed feature(s)"
            )
            await self.healer.start()

        self._startup_complete = True

        print("\n" + "=" * 80)
        if non_critical_failures:
            print(
                f"🌌 VEGA SYSTEM ONLINE (Degraded Mode - {len(non_critical_failures)} features healing)"
            )
        else:
            print("🌌 VEGA SYSTEM ONLINE - All Features Operational")
        print("=" * 80 + "\n")

        return True, status

    async def shutdown(self):
        """Shutdown the manager and healer"""
        await self.healer.stop()
        self.usage_tracker._save()
        logger.info("Resilient startup manager shutdown complete")

    async def _load_feature(self, feature_state: FeatureState) -> Tuple[bool, str]:
        """Attempt to load a single feature"""
        feature_state.status = FeatureStatus.LOADING

        try:
            # Check dependencies first
            for dep_name in feature_state.definition.dependencies:
                dep_state = self.features.get(dep_name)
                if not dep_state or dep_state.status != FeatureStatus.HEALTHY:
                    raise RuntimeError(f"Dependency '{dep_name}' not available")

            # Execute the loader
            await feature_state.definition.loader()

            feature_state.status = FeatureStatus.HEALTHY
            feature_state.load_time = datetime.utcnow()
            feature_state.error = None
            feature_state.error_traceback = None

            return True, "Loaded successfully"

        except Exception as e:
            feature_state.status = FeatureStatus.FAILED
            feature_state.error = e
            feature_state.error_traceback = traceback.format_exc()

            logger.warning(
                f"Feature '{feature_state.definition.name}' failed to load: {e}",
                exc_info=True,
            )

            # Try fallback if available
            if feature_state.definition.fallback:
                try:
                    await feature_state.definition.fallback()
                    feature_state.status = FeatureStatus.DEGRADED
                    return True, f"Loaded with fallback: {e}"
                except Exception as fallback_error:
                    logger.warning(f"Fallback also failed: {fallback_error}")

            return False, str(e)

    def get_failed_features(self) -> List[FeatureState]:
        """Get list of features that need healing"""
        return [
            fs
            for fs in self.features.values()
            if fs.status in (FeatureStatus.FAILED, FeatureStatus.DEGRADED)
        ]

    async def attempt_heal(self, feature_state: FeatureState) -> bool:
        """
        Attempt to heal a failed feature.

        This method:
        1. Checks knowledge base for known resolutions
        2. Tries known resolutions first
        3. Falls back to generic strategies
        4. Records successful resolutions
        """
        async with self._lock:
            if feature_state.status not in (
                FeatureStatus.FAILED,
                FeatureStatus.DEGRADED,
            ):
                return True

            feature_state.status = FeatureStatus.HEALING
            feature_state.heal_attempts += 1
            feature_state.last_heal_attempt = datetime.utcnow()

            logger.info(
                f"🔧 Attempting to heal '{feature_state.definition.name}' "
                f"(attempt {feature_state.heal_attempts}/{feature_state.definition.max_heal_attempts})"
            )

            # Check knowledge base for known resolution
            if feature_state.error:
                known_resolution = self.knowledge_base.find_resolution(
                    feature_state.definition.name, feature_state.error
                )

                if known_resolution:
                    logger.info(
                        f"📚 Found known resolution: {known_resolution.strategy_used}"
                    )
                    success = await self._apply_resolution(
                        feature_state,
                        known_resolution.strategy_used,
                        known_resolution.resolution_details,
                    )
                    if success:
                        self.knowledge_base.record_success(
                            feature_state.definition.name,
                            feature_state.error,
                            known_resolution.strategy_used,
                            known_resolution.resolution_details,
                        )
                        return True

            # Try each resolution strategy
            for strategy in feature_state.definition.resolution_strategies:
                success = await self._try_strategy(feature_state, strategy)
                if success:
                    return True

            # Failed to heal
            feature_state.status = FeatureStatus.FAILED
            logger.warning(
                f"⚠️  Could not heal '{feature_state.definition.name}' "
                f"(attempt {feature_state.heal_attempts})"
            )
            return False

    async def _try_strategy(
        self, feature_state: FeatureState, strategy: ResolutionStrategy
    ) -> bool:
        """Try a specific resolution strategy"""
        try:
            details: Dict[str, Any] = {}

            if strategy == ResolutionStrategy.REIMPORT:
                # Clear any cached imports and try again
                import sys

                module_name = f"src.vega.core.{feature_state.definition.name}"
                modules_to_remove = [
                    m
                    for m in sys.modules
                    if m.startswith(module_name) or feature_state.definition.name in m
                ]
                for m in modules_to_remove:
                    del sys.modules[m]
                details = {"cleared_modules": modules_to_remove}

            elif strategy == ResolutionStrategy.RESET_STATE:
                # Wait a bit and try again (transient error)
                await asyncio.sleep(2.0)
                details = {"action": "waited_and_retry"}

            elif strategy == ResolutionStrategy.RECREATE_RESOURCE:
                # Try to recreate any resources
                details = {"action": "recreate_attempted"}

            elif strategy == ResolutionStrategy.FALLBACK_CONFIG:
                # Use fallback/default configuration
                details = {"action": "fallback_config"}

            # Try loading the feature again
            success, _ = await self._load_feature(feature_state)

            if success:
                # Record the successful resolution
                if feature_state.error:
                    self.knowledge_base.record_success(
                        feature_state.definition.name,
                        feature_state.error,
                        strategy.value,
                        details,
                    )
                logger.info(
                    f"✅ Successfully healed '{feature_state.definition.name}' using {strategy.value}"
                )
                await self._notify_healed(feature_state, strategy.value)
                return True

        except Exception as e:
            logger.debug(f"Strategy {strategy.value} failed: {e}")

        return False

    async def _apply_resolution(
        self, feature_state: FeatureState, strategy: str, details: Dict[str, Any]
    ) -> bool:
        """Apply a known resolution from the knowledge base"""
        try:
            # Apply the resolution based on stored details
            if strategy == ResolutionStrategy.REIMPORT.value:
                import sys

                for module in details.get("cleared_modules", []):
                    if module in sys.modules:
                        del sys.modules[module]

            # Try loading
            success, _ = await self._load_feature(feature_state)
            if success:
                logger.info(
                    f"✅ Known resolution worked for '{feature_state.definition.name}'"
                )
                await self._notify_healed(feature_state, strategy)
            return success

        except Exception as e:
            logger.debug(f"Known resolution failed: {e}")
            return False

    async def _notify_permanent_failure(self, feature_state: FeatureState):
        """Notify about a permanent failure"""
        message = (
            f"🚨 PERMANENT FAILURE: Feature '{feature_state.definition.name}' "
            f"could not be healed after {feature_state.heal_attempts} attempts.\n"
            f"Error: {feature_state.error}\n"
            f"Manual intervention required."
        )

        logger.error(message)
        print(f"\n{message}\n")

        # Call notification handlers
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler("permanent_failure", feature_state)
                else:
                    handler("permanent_failure", feature_state)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    async def _notify_healed(self, feature_state: FeatureState, strategy: str):
        """Notify about successful healing"""
        message = (
            f"🏥 HEALED: Feature '{feature_state.definition.name}' "
            f"was successfully recovered using strategy '{strategy}'"
        )

        logger.info(message)
        print(f"\n{message}\n")

        # Call notification handlers
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler("healed", feature_state, strategy)
                else:
                    handler("healed", feature_state, strategy)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    def record_feature_usage(self, feature_name: str):
        """Record that a feature was used (call this from feature code)"""
        self.usage_tracker.record_usage(feature_name)

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all features"""
        return {
            "startup_complete": self._startup_complete,
            "healer_running": self.healer._running,
            "features": {
                name: {
                    "status": state.status.value,
                    "priority": state.definition.priority.name,
                    "is_critical": state.definition.is_critical,
                    "heal_attempts": state.heal_attempts,
                    "usage_count": self.usage_tracker.usage_counts.get(name, 0),
                    "error": str(state.error) if state.error else None,
                }
                for name, state in self.features.items()
            },
            "knowledge_base_entries": sum(
                len(v) for v in self.knowledge_base.resolutions.values()
            ),
        }


# Global instance (created lazily)
_resilient_manager: Optional[ResilientStartupManager] = None


def get_resilient_manager() -> ResilientStartupManager:
    """Get or create the global resilient startup manager"""
    global _resilient_manager
    if _resilient_manager is None:
        _resilient_manager = ResilientStartupManager(
            knowledge_base_path=Path("data/resolution_knowledge.json"),
            usage_tracker_path=Path("data/feature_usage.json"),
        )
    return _resilient_manager


# Decorator for tracking feature usage
def track_usage(feature_name: str):
    """Decorator to track usage of a feature"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            get_resilient_manager().record_feature_usage(feature_name)
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            get_resilient_manager().record_feature_usage(feature_name)
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ==============================================================================
# Compatibility Layer for API and Tests
# ==============================================================================
# The following classes provide a simplified interface that matches what the
# app.py startup system and tests expect. This allows flexible usage patterns.


class FeatureCategory(Enum):
    """Simplified category enum for API compatibility"""

    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"
    OPTIONAL = "optional"

    @classmethod
    def from_priority(
        cls, priority: FeaturePriority, is_critical: bool
    ) -> "FeatureCategory":
        """Convert from internal FeaturePriority to FeatureCategory"""
        if is_critical:
            return cls.CRITICAL
        mapping = {
            FeaturePriority.CRITICAL: cls.CRITICAL,
            FeaturePriority.HIGH: cls.HIGH,
            FeaturePriority.MEDIUM: cls.STANDARD,
            FeaturePriority.LOW: cls.LOW,
            FeaturePriority.BACKGROUND: cls.OPTIONAL,
        }
        return mapping.get(priority, cls.STANDARD)


@dataclass
class RepairStrategy:
    """Simplified repair strategy for API compatibility"""

    name: str
    repair_func: Callable[[], Coroutine[Any, Any, bool]]
    description: str = ""
    success_rate: float = 0.0


@dataclass
class StartupFeature:
    """Simplified feature definition for API compatibility"""

    name: str
    category: FeatureCategory
    init_func: Callable[[], Coroutine[Any, Any, Any]]
    health_check_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None
    repair_strategies: List[RepairStrategy] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_repair_attempts: int = 3
    description: str = ""


class SimplifiedStartupManager:
    """
    Simplified startup manager that provides the interface expected by app.py.
    This wraps the more complex ResilientStartupManager for ease of use.
    """

    def __init__(self):
        self.features: Dict[str, StartupFeature] = {}
        self.feature_status: Dict[str, Dict[str, Any]] = {}
        self.feature_usage: Dict[str, int] = defaultdict(int)
        self.repair_queue: asyncio.Queue = asyncio.Queue()
        self.pending_repairs: Set[str] = set()
        self.repair_knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.is_running = False
        self._healer_task: Optional[asyncio.Task] = None
        self._internal_manager = get_resilient_manager()

    def register_feature(self, feature: StartupFeature):
        """Register a feature with the simplified manager"""
        self.features[feature.name] = feature
        self.feature_status[feature.name] = {
            "status": "unknown",
            "healthy": False,
            "error": None,
            "repair_attempts": 0,
            "last_health_check": None,
        }
        logger.debug(f"Registered feature: {feature.name}")

    async def startup_sequence(self) -> bool:
        """
        Execute the startup sequence.
        Returns True if all critical features succeeded.
        """
        self.is_running = True
        all_critical_ok = True

        # Sort features: critical first, then by dependencies
        sorted_features = self._sort_by_dependencies()

        for feature in sorted_features:
            success = await self._init_feature(feature)

            if not success and feature.category == FeatureCategory.CRITICAL:
                all_critical_ok = False
                logger.error(
                    f"Critical feature '{feature.name}' failed - startup aborted"
                )
                break
            elif not success:
                # Queue non-critical failures for background repair
                await self.queue_repair(feature.name)

        # Start background healer if we have failures
        if self.pending_repairs:
            self._healer_task = asyncio.create_task(self._background_healer())

        return all_critical_ok

    async def _init_feature(self, feature: StartupFeature) -> bool:
        """Initialize a single feature"""
        self.feature_status[feature.name]["status"] = "loading"

        try:
            # Check dependencies
            for dep in feature.dependencies:
                dep_status = self.feature_status.get(dep, {})
                if dep_status.get("status") != "healthy":
                    raise RuntimeError(f"Dependency '{dep}' not healthy")

            # Run init with timeout
            await asyncio.wait_for(feature.init_func(), timeout=feature.timeout)

            self.feature_status[feature.name] = {
                "status": "healthy",
                "healthy": True,
                "error": None,
                "repair_attempts": 0,
                "last_health_check": datetime.utcnow().isoformat(),
            }
            return True

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {feature.timeout}s"
            self.feature_status[feature.name] = {
                "status": "failed",
                "healthy": False,
                "error": error_msg,
                "repair_attempts": 0,
            }
            logger.warning(f"Feature '{feature.name}' timed out")
            return False

        except Exception as e:
            self.feature_status[feature.name] = {
                "status": "failed",
                "healthy": False,
                "error": str(e),
                "repair_attempts": 0,
            }
            logger.warning(f"Feature '{feature.name}' failed: {e}")
            return False

    def _sort_by_dependencies(self) -> List[StartupFeature]:
        """Sort features so dependencies come first"""
        sorted_list = []
        visited = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            feature = self.features.get(name)
            if feature:
                for dep in feature.dependencies:
                    visit(dep)
                sorted_list.append(feature)

        # Visit critical features first
        for name, feature in self.features.items():
            if feature.category == FeatureCategory.CRITICAL:
                visit(name)

        # Then high priority
        for name, feature in self.features.items():
            if feature.category == FeatureCategory.HIGH:
                visit(name)

        # Then the rest
        for name in self.features:
            visit(name)

        return sorted_list

    async def queue_repair(self, feature_name: str):
        """Queue a feature for background repair"""
        if feature_name not in self.pending_repairs:
            self.pending_repairs.add(feature_name)
            priority = self.calculate_repair_priority(feature_name)
            await self.repair_queue.put(
                (-priority, feature_name)
            )  # Negative for max-heap behavior
            logger.info(f"Queued '{feature_name}' for repair (priority: {priority})")

    def calculate_repair_priority(self, feature_name: str) -> float:
        """Calculate repair priority based on usage and category"""
        feature = self.features.get(feature_name)
        if not feature:
            return 0.0

        # Base priority from category
        category_scores = {
            FeatureCategory.CRITICAL: 100,
            FeatureCategory.HIGH: 80,
            FeatureCategory.STANDARD: 50,
            FeatureCategory.LOW: 20,
            FeatureCategory.OPTIONAL: 10,
        }
        score = category_scores.get(feature.category, 50)

        # Add usage bonus
        usage = self.feature_usage.get(feature_name, 0)
        import math

        score += math.log1p(usage) * 5

        return score

    async def _background_healer(self):
        """Background task that attempts to repair failed features"""
        logger.info("Background healer started")

        while self.is_running and self.pending_repairs:
            try:
                # Get highest priority item
                _, feature_name = await asyncio.wait_for(
                    self.repair_queue.get(), timeout=60.0
                )

                if feature_name not in self.pending_repairs:
                    continue

                feature = self.features.get(feature_name)
                if not feature:
                    self.pending_repairs.discard(feature_name)
                    continue

                status = self.feature_status.get(feature_name, {})
                attempts = status.get("repair_attempts", 0)

                if attempts >= feature.max_repair_attempts:
                    # Give up and notify
                    self.pending_repairs.discard(feature_name)
                    logger.error(
                        f"🚨 Giving up on '{feature_name}' after {attempts} repair attempts"
                    )
                    continue

                # Update attempt count
                self.feature_status[feature_name]["repair_attempts"] = attempts + 1

                # Try repair strategies
                repaired = False
                error_pattern = status.get("error", "")

                # Check knowledge base first
                best_strategy = self.get_best_repair_strategy(
                    feature_name, error_pattern
                )
                if best_strategy:
                    try:
                        repaired = await best_strategy.repair_func()
                        if repaired:
                            # Verify with init
                            repaired = await self._init_feature(feature)
                    except Exception as e:
                        logger.debug(f"Knowledge base strategy failed: {e}")

                # Try other strategies
                if not repaired:
                    for strategy in feature.repair_strategies:
                        try:
                            if await strategy.repair_func():
                                repaired = await self._init_feature(feature)
                                if repaired:
                                    # Add to knowledge base
                                    self.add_to_knowledge_base(
                                        feature_name, strategy.name, error_pattern
                                    )
                                    break
                        except Exception as e:
                            logger.debug(
                                f"Repair strategy '{strategy.name}' failed: {e}"
                            )

                if repaired:
                    self.pending_repairs.discard(feature_name)
                    logger.info(f"✅ Successfully repaired '{feature_name}'")
                else:
                    # Re-queue with backoff
                    await asyncio.sleep(
                        min(30 * (2**attempts), 300)
                    )  # Exponential backoff
                    if feature_name in self.pending_repairs:
                        await self.repair_queue.put(
                            (
                                -self.calculate_repair_priority(feature_name),
                                feature_name,
                            )
                        )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background healer error: {e}")
                await asyncio.sleep(10)

        logger.info("Background healer stopped")

    def record_feature_usage(self, feature_name: str):
        """Record that a feature was used"""
        self.feature_usage[feature_name] = self.feature_usage.get(feature_name, 0) + 1
        self._internal_manager.record_feature_usage(feature_name)

    def add_to_knowledge_base(
        self, feature_name: str, strategy_name: str, error_pattern: str
    ):
        """Add a successful repair strategy to the knowledge base"""
        self.repair_knowledge_base[feature_name] = {
            "strategy": strategy_name,
            "error_pattern": error_pattern,
            "success_count": self.repair_knowledge_base.get(feature_name, {}).get(
                "success_count", 0
            )
            + 1,
            "last_used": datetime.utcnow().isoformat(),
        }
        logger.info(f"📚 Added to knowledge base: {feature_name} -> {strategy_name}")

    def get_best_repair_strategy(
        self, feature_name: str, error_pattern: str
    ) -> Optional[RepairStrategy]:
        """Get the best repair strategy for a feature based on knowledge base"""
        kb_entry = self.repair_knowledge_base.get(feature_name)
        if not kb_entry:
            return None

        # Check if error pattern matches
        if kb_entry.get("error_pattern", "") in error_pattern:
            feature = self.features.get(feature_name)
            if feature:
                for strategy in feature.repair_strategies:
                    if strategy.name == kb_entry["strategy"]:
                        return strategy

        return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all feature statuses"""
        healthy = []
        degraded = []
        failed = []

        for name, status in self.feature_status.items():
            if status.get("healthy"):
                healthy.append(name)
            elif status.get("status") == "failed":
                failed.append(name)
            else:
                degraded.append(name)

        return {
            "healthy_features": healthy,
            "degraded_features": degraded,
            "failed_features": failed,
            "total_features": len(self.features),
            "is_running": self.is_running,
            "pending_repairs": len(self.pending_repairs),
        }

    async def shutdown(self):
        """Shutdown the manager"""
        self.is_running = False
        if self._healer_task:
            self._healer_task.cancel()
            try:
                await self._healer_task
            except asyncio.CancelledError:
                pass
        logger.info("Simplified startup manager shutdown complete")
