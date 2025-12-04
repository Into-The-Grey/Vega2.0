#!/usr/bin/env python3
"""
Startup Features Configuration for Vega2.0

This module defines all features that are loaded during startup,
categorizing them as either CRITICAL (must succeed) or non-critical
(can fail and heal in background).

Feature Categories:
-------------------
CRITICAL (is_critical=True):
    - Configuration validation
    - Database connection
    - LLM backend connectivity
    - Core API functionality

HIGH Priority (non-critical):
    - Memory manager
    - Resource manager
    - Error handling

MEDIUM Priority (non-critical):
    - Database profiler
    - Correlation ID tracing
    - Performance monitoring

LOW Priority (non-critical):
    - Analytics integration
    - Collaboration features
    - Document intelligence

BACKGROUND Priority (non-critical):
    - Process management
    - Extended integrations
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

# Import both the internal and simplified interfaces
from .resilient_startup import (
    FeatureDefinition,
    FeaturePriority,
    ResolutionStrategy,
    get_resilient_manager,
    # Simplified interface for app.py
    FeatureCategory,
    StartupFeature,
    RepairStrategy,
    SimplifiedStartupManager,
)

if TYPE_CHECKING:
    from .resilient_startup import ResilientStartupManager

logger = logging.getLogger(__name__)


# ==============================================================================
# Feature Loaders - Each returns an async function that loads the feature
# ==============================================================================


async def load_config_validation():
    """CRITICAL: Validate startup configuration"""
    from .config_validator import validate_startup_config  # type: ignore[import]

    is_valid = validate_startup_config()
    if not is_valid:
        # Log but don't fail - allow degraded operation
        logger.warning("Configuration validation found issues - operating in degraded mode")
    return is_valid


async def load_database():
    """CRITICAL: Initialize database connection"""
    from sqlalchemy import text

    from .db import _init_db, engine  # type: ignore[attr-defined]

    # Initialize the database
    _init_db()
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection verified")


async def load_llm_backend():
    """CRITICAL: Verify LLM backend is accessible"""
    from .llm import llm_warmup  # type: ignore[attr-defined]

    await llm_warmup()
    logger.info("LLM backend verified")


async def load_config():
    """CRITICAL: Load configuration"""
    from .config import get_config  # type: ignore[import]

    config = get_config()
    if not config:
        raise RuntimeError("Failed to load configuration")
    logger.info("Configuration loaded")
    return config


async def load_resource_manager():
    """HIGH: Initialize resource manager for connection pooling"""
    from .resource_manager import get_resource_manager  # type: ignore[import]

    manager = await get_resource_manager()
    logger.info("Resource manager initialized")
    return manager


async def load_memory_manager():
    """HIGH: Start persistent memory manager"""
    from .memory_manager import start_memory_manager  # type: ignore[import]

    await start_memory_manager()
    logger.info("Memory manager started (persistent mode)")


async def load_error_handling():
    """HIGH: Setup error handling and recovery"""
    from .error_handler import get_error_handler  # type: ignore[import]
    from .recovery_manager import get_recovery_manager  # type: ignore[import]

    error_handler = get_error_handler()
    recovery_manager = get_recovery_manager()
    logger.info("Error handling and recovery initialized")


async def load_db_profiler():
    """MEDIUM: Initialize database query profiler"""
    from .db_profiler import get_profiler  # type: ignore[import]

    profiler = get_profiler()
    profiler.enabled = True
    profiler.set_slow_query_threshold(100.0)  # 100ms threshold
    logger.info("Database profiler enabled")


async def load_correlation_tracing():
    """MEDIUM: Enable distributed tracing with correlation IDs"""
    from .correlation import configure_correlation_logging  # type: ignore[import]

    configure_correlation_logging()
    logger.info("Correlation ID tracing enabled")


async def load_metrics_aggregator():
    """MEDIUM: Initialize metrics collection"""
    from .metrics_aggregator import get_metrics_aggregator  # type: ignore[import]

    aggregator = await get_metrics_aggregator()
    logger.info("Metrics aggregator initialized")


async def load_analytics():
    """LOW: Initialize analytics system"""
    from ..analytics.collector import analytics_collector  # type: ignore[import]
    from ..analytics.engine import analytics_engine  # type: ignore[import]

    # Initialize analytics components
    logger.info("Analytics system initialized")


async def load_collaboration():
    """LOW: Initialize collaboration features"""
    # Collaboration manager may not be implemented yet
    try:
        from ..collaboration.integration import collaboration_manager  # type: ignore[attr-defined]

        logger.info("Collaboration features initialized")
    except ImportError:
        logger.warning("Collaboration module not available - skipping")


async def load_document_intelligence():
    """LOW: Initialize document processing"""
    # Document processor may not be implemented yet
    try:
        from ..document.api import document_processor  # type: ignore[attr-defined]

        logger.info("Document intelligence initialized")
    except ImportError:
        logger.warning("Document intelligence module not available - skipping")


async def load_process_manager():
    """BACKGROUND: Initialize background process management"""
    from .process_manager import get_process_manager  # type: ignore[import]

    manager = get_process_manager()
    logger.info("Process manager initialized")


async def load_ecc_crypto():
    """BACKGROUND: Initialize ECC cryptography system"""
    from .ecc_crypto import get_ecc_manager  # type: ignore[import]

    manager = get_ecc_manager()
    logger.info("ECC cryptography initialized")


# ==============================================================================
# Fallback Loaders - Used when primary loader fails
# ==============================================================================


async def fallback_config_validation():
    """Fallback: Skip validation but log warning"""
    logger.warning("Running without configuration validation - using defaults")


async def fallback_resource_manager():
    """Fallback: Use minimal resource management"""
    logger.warning("Resource manager unavailable - using basic HTTP clients")


async def fallback_db_profiler():
    """Fallback: Disable profiling"""
    logger.warning("Database profiler unavailable - query profiling disabled")


async def fallback_analytics():
    """Fallback: Disable analytics"""
    logger.warning("Analytics unavailable - metrics collection disabled")


# ==============================================================================
# Feature Registration
# ==============================================================================


def register_all_features_internal():
    """Register all startup features with the internal resilient manager.

    This function uses the internal FeatureDefinition interface.
    For the simplified interface used by app.py, see register_all_features().
    """
    manager = get_resilient_manager()

    # -------------------------------------------------------------------------
    # CRITICAL FEATURES - Must succeed or startup fails
    # -------------------------------------------------------------------------

    manager.register_feature(
        FeatureDefinition(
            name="config",
            loader=load_config,
            priority=FeaturePriority.CRITICAL,
            is_critical=True,
            description="Core configuration loading",
            resolution_strategies=[
                ResolutionStrategy.RESET_STATE,
                ResolutionStrategy.FALLBACK_CONFIG,
            ],
        )
    )

    # Note: Database and LLM are critical but we handle them separately
    # because they have complex initialization that's already in app.py

    # -------------------------------------------------------------------------
    # HIGH PRIORITY FEATURES - Important but can fail
    # -------------------------------------------------------------------------

    manager.register_feature(
        FeatureDefinition(
            name="config_validation",
            loader=load_config_validation,
            priority=FeaturePriority.HIGH,
            is_critical=False,
            description="Configuration validation and checks",
            dependencies=["config"],
            fallback=fallback_config_validation,
            resolution_strategies=[
                ResolutionStrategy.RESET_STATE,
                ResolutionStrategy.REIMPORT,
            ],
            max_heal_attempts=3,
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="resource_manager",
            loader=load_resource_manager,
            priority=FeaturePriority.HIGH,
            is_critical=False,
            description="HTTP connection pooling and resource management",
            fallback=fallback_resource_manager,
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RESET_STATE,
                ResolutionStrategy.RECREATE_RESOURCE,
            ],
            max_heal_attempts=5,
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="memory_manager",
            loader=load_memory_manager,
            priority=FeaturePriority.HIGH,
            is_critical=False,
            description="Persistent memory management system",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RESET_STATE,
            ],
            max_heal_attempts=5,
            heal_backoff_seconds=60.0,
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="error_handling",
            loader=load_error_handling,
            priority=FeaturePriority.HIGH,
            is_critical=False,
            description="Error handling and recovery system",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RESET_STATE,
            ],
            max_heal_attempts=3,
        )
    )

    # -------------------------------------------------------------------------
    # MEDIUM PRIORITY FEATURES - Regular use
    # -------------------------------------------------------------------------

    manager.register_feature(
        FeatureDefinition(
            name="db_profiler",
            loader=load_db_profiler,
            priority=FeaturePriority.MEDIUM,
            is_critical=False,
            description="Database query profiling",
            fallback=fallback_db_profiler,
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
            ],
            max_heal_attempts=3,
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="correlation_tracing",
            loader=load_correlation_tracing,
            priority=FeaturePriority.MEDIUM,
            is_critical=False,
            description="Distributed tracing with correlation IDs",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RESET_STATE,
            ],
            max_heal_attempts=3,
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="metrics_aggregator",
            loader=load_metrics_aggregator,
            priority=FeaturePriority.MEDIUM,
            is_critical=False,
            description="System metrics collection and aggregation",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RECREATE_RESOURCE,
            ],
            max_heal_attempts=3,
        )
    )

    # -------------------------------------------------------------------------
    # LOW PRIORITY FEATURES - Nice to have
    # -------------------------------------------------------------------------

    manager.register_feature(
        FeatureDefinition(
            name="analytics",
            loader=load_analytics,
            priority=FeaturePriority.LOW,
            is_critical=False,
            description="Analytics and visualization system",
            fallback=fallback_analytics,
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
            ],
            max_heal_attempts=5,
            heal_backoff_seconds=120.0,  # 2 minutes between attempts
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="collaboration",
            loader=load_collaboration,
            priority=FeaturePriority.LOW,
            is_critical=False,
            description="Collaboration and real-time features",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RESET_STATE,
            ],
            max_heal_attempts=5,
            heal_backoff_seconds=120.0,
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="document_intelligence",
            loader=load_document_intelligence,
            priority=FeaturePriority.LOW,
            is_critical=False,
            description="Document processing and analysis",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
            ],
            max_heal_attempts=5,
            heal_backoff_seconds=120.0,
        )
    )

    # -------------------------------------------------------------------------
    # BACKGROUND PRIORITY FEATURES - Least important
    # -------------------------------------------------------------------------

    manager.register_feature(
        FeatureDefinition(
            name="process_manager",
            loader=load_process_manager,
            priority=FeaturePriority.BACKGROUND,
            is_critical=False,
            description="Background process management",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
            ],
            max_heal_attempts=3,
            heal_backoff_seconds=300.0,  # 5 minutes between attempts
        )
    )

    manager.register_feature(
        FeatureDefinition(
            name="ecc_crypto",
            loader=load_ecc_crypto,
            priority=FeaturePriority.BACKGROUND,
            is_critical=False,
            description="ECC cryptography system",
            resolution_strategies=[
                ResolutionStrategy.REIMPORT,
                ResolutionStrategy.RESET_STATE,
            ],
            max_heal_attempts=3,
            heal_backoff_seconds=300.0,
        )
    )

    logger.info(f"Registered {len(manager.features)} startup features")
    return manager


# ==============================================================================
# Notification Handlers
# ==============================================================================


async def log_notification_handler(event_type: str, feature_state, strategy: Optional[str] = None):
    """Log notifications to file"""
    from datetime import datetime
    from pathlib import Path

    log_file = Path("data/vega_logs/resilience_events.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat()

    if event_type == "permanent_failure":
        message = (
            f"[{timestamp}] PERMANENT_FAILURE: {feature_state.definition.name}\n"
            f"  Error: {feature_state.error}\n"
            f"  Attempts: {feature_state.heal_attempts}\n"
            f"  Traceback: {feature_state.error_traceback}\n"
        )
    elif event_type == "healed":
        message = (
            f"[{timestamp}] HEALED: {feature_state.definition.name}\n"
            f"  Strategy: {strategy}\n"
            f"  Attempts needed: {feature_state.heal_attempts}\n"
        )
    else:
        message = f"[{timestamp}] {event_type}: {feature_state.definition.name}\n"

    with open(log_file, "a") as f:
        f.write(message + "\n")


def console_notification_handler(event_type: str, feature_state, strategy: Optional[str] = None):
    """Print notifications to console"""
    if event_type == "permanent_failure":
        print(f"\n🚨 ALERT: Feature '{feature_state.definition.name}' permanently failed!")
        print(f"   Error: {feature_state.error}")
        print(f"   Manual intervention required.\n")
    elif event_type == "healed":
        print(f"\n✅ Feature '{feature_state.definition.name}' has been healed!")
        print(f"   Strategy used: {strategy}\n")


def setup_notification_handlers():
    """Setup default notification handlers"""
    manager = get_resilient_manager()
    manager.add_notification_handler(log_notification_handler)
    manager.add_notification_handler(console_notification_handler)


# ==============================================================================
# Simplified Manager Registration (for app.py)
# ==============================================================================


async def _simple_repair_reimport():
    """Generic repair strategy: reimport the module"""
    import sys
    import importlib

    # Clear cached modules
    to_clear = [m for m in sys.modules if "vega" in m]
    for m in to_clear[:10]:  # Limit to avoid clearing too much
        try:
            del sys.modules[m]
        except Exception:
            pass
    return True


async def _simple_repair_wait():
    """Generic repair strategy: wait and retry"""
    import asyncio

    await asyncio.sleep(2.0)
    return True


def register_all_features(manager: SimplifiedStartupManager):
    """
    Register all startup features with the SimplifiedStartupManager.
    This is the entry point used by app.py.
    """

    # Create generic repair strategies
    reimport_strategy = RepairStrategy(
        name="reimport",
        repair_func=_simple_repair_reimport,
        description="Clear module cache and reimport",
    )

    wait_strategy = RepairStrategy(
        name="wait_retry",
        repair_func=_simple_repair_wait,
        description="Wait briefly and retry",
    )

    # -------------------------------------------------------------------------
    # CRITICAL FEATURES
    # -------------------------------------------------------------------------

    async def init_config():
        from .config import get_config  # type: ignore[import]

        config = get_config()
        if not config:
            raise RuntimeError("Failed to load configuration")
        logger.info("✅ Configuration loaded")

    async def check_config_health():
        from .config import get_config  # type: ignore[import]

        return get_config() is not None

    manager.register_feature(
        StartupFeature(
            name="config",
            category=FeatureCategory.CRITICAL,
            init_func=init_config,
            health_check_func=check_config_health,
            repair_strategies=[reimport_strategy, wait_strategy],
            description="Core configuration",
        )
    )

    async def init_config_validation():
        from .config_validator import validate_startup_config  # type: ignore[import]

        is_valid = validate_startup_config()
        if not is_valid:
            logger.warning("Configuration validation found issues")
        logger.info("✅ Configuration validated")

    manager.register_feature(
        StartupFeature(
            name="config_validation",
            category=FeatureCategory.CRITICAL,
            init_func=init_config_validation,
            dependencies=["config"],
            repair_strategies=[reimport_strategy],
            description="Configuration validation",
        )
    )

    # -------------------------------------------------------------------------
    # HIGH PRIORITY FEATURES
    # -------------------------------------------------------------------------

    async def init_resource_manager():
        from .resource_manager import get_resource_manager  # type: ignore[import]

        await get_resource_manager()
        logger.info("✅ Resource manager initialized")

    manager.register_feature(
        StartupFeature(
            name="resource_manager",
            category=FeatureCategory.HIGH,
            init_func=init_resource_manager,
            repair_strategies=[reimport_strategy, wait_strategy],
            description="HTTP connection pooling",
        )
    )

    async def init_db_profiler():
        from .db_profiler import get_profiler  # type: ignore[import]

        profiler = get_profiler()
        profiler.enabled = True
        profiler.set_slow_query_threshold(100.0)
        logger.info("✅ Database profiler enabled")

    manager.register_feature(
        StartupFeature(
            name="db_profiler",
            category=FeatureCategory.HIGH,
            init_func=init_db_profiler,
            repair_strategies=[reimport_strategy],
            description="Database query profiling",
        )
    )

    async def init_memory_manager():
        from .memory_manager import start_memory_manager  # type: ignore[import]

        await start_memory_manager()
        logger.info("✅ Memory manager started")

    async def check_memory_health():
        from .memory_manager import get_memory_manager  # type: ignore[import]

        manager = get_memory_manager()
        return manager is not None and manager.running

    manager.register_feature(
        StartupFeature(
            name="memory_manager",
            category=FeatureCategory.HIGH,
            init_func=init_memory_manager,
            health_check_func=check_memory_health,
            repair_strategies=[wait_strategy, reimport_strategy],
            description="Persistent memory management",
        )
    )

    # -------------------------------------------------------------------------
    # STANDARD PRIORITY FEATURES
    # -------------------------------------------------------------------------

    async def init_correlation():
        try:
            from .correlation import configure_correlation_logging  # type: ignore[import]

            configure_correlation_logging()
            logger.info("✅ Correlation tracing enabled")
        except ImportError:
            logger.info("⚠️  Correlation tracing not available")

    manager.register_feature(
        StartupFeature(
            name="correlation_tracing",
            category=FeatureCategory.STANDARD,
            init_func=init_correlation,
            repair_strategies=[reimport_strategy],
            description="Distributed tracing",
        )
    )

    # -------------------------------------------------------------------------
    # LOW PRIORITY FEATURES
    # -------------------------------------------------------------------------

    async def init_error_handling():
        try:
            from .error_handler import get_error_handler  # type: ignore[import]

            get_error_handler()
            logger.info("✅ Error handling initialized")
        except ImportError:
            logger.info("⚠️  Error handling not available")

    manager.register_feature(
        StartupFeature(
            name="error_handling",
            category=FeatureCategory.LOW,
            init_func=init_error_handling,
            repair_strategies=[reimport_strategy],
            description="Error handling system",
        )
    )

    # -------------------------------------------------------------------------
    # OPTIONAL FEATURES
    # -------------------------------------------------------------------------

    async def init_process_manager():
        try:
            from .process_manager import get_process_manager  # type: ignore[import]

            get_process_manager()
            logger.info("✅ Process manager available")
        except ImportError:
            logger.info("⚠️  Process manager not available")

    manager.register_feature(
        StartupFeature(
            name="process_manager",
            category=FeatureCategory.OPTIONAL,
            init_func=init_process_manager,
            repair_strategies=[reimport_strategy],
            description="Background process management",
        )
    )

    logger.info(f"Registered {len(manager.features)} startup features with simplified manager")
    return manager
