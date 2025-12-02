"""
Tests for the Resilient Startup System

Tests the self-healing startup manager that allows non-critical features
to fail gracefully and be repaired in the background.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


class TestFeatureCategory:
    """Test the FeatureCategory enum"""

    def test_category_values(self):
        """Test that all expected categories exist"""
        from src.vega.core.resilient_startup import FeatureCategory

        assert FeatureCategory.CRITICAL.value == "critical"
        assert FeatureCategory.HIGH.value == "high"
        assert FeatureCategory.STANDARD.value == "standard"
        assert FeatureCategory.LOW.value == "low"
        assert FeatureCategory.OPTIONAL.value == "optional"

    def test_category_comparison(self):
        """Test category ordering for priority"""
        from src.vega.core.resilient_startup import FeatureCategory

        # Categories have implicit ordering based on enum definition
        categories = list(FeatureCategory)
        assert categories[0] == FeatureCategory.CRITICAL
        assert categories[-1] == FeatureCategory.OPTIONAL


class TestStartupFeature:
    """Test the StartupFeature dataclass"""

    def test_create_feature(self):
        """Test creating a startup feature"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="test_feature", category=FeatureCategory.STANDARD, init_func=init_func
        )

        assert feature.name == "test_feature"
        assert feature.category == FeatureCategory.STANDARD
        assert feature.timeout == 30.0  # default
        assert feature.max_repair_attempts == 3  # default
        assert feature.dependencies == []

    def test_feature_with_dependencies(self):
        """Test creating a feature with dependencies"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="dependent_feature",
            category=FeatureCategory.HIGH,
            init_func=init_func,
            dependencies=["base_feature", "config"],
        )

        assert "base_feature" in feature.dependencies
        assert "config" in feature.dependencies


class TestRepairStrategy:
    """Test the RepairStrategy dataclass"""

    def test_create_repair_strategy(self):
        """Test creating a repair strategy"""
        from src.vega.core.resilient_startup import RepairStrategy

        async def repair_func():
            return True

        strategy = RepairStrategy(
            name="restart_service",
            repair_func=repair_func,
            description="Restart the underlying service",
        )

        assert strategy.name == "restart_service"
        assert strategy.success_rate == 0.0  # default

    def test_repair_strategy_with_success_rate(self):
        """Test repair strategy tracks success rate"""
        from src.vega.core.resilient_startup import RepairStrategy

        async def repair_func():
            return True

        strategy = RepairStrategy(
            name="reconnect",
            repair_func=repair_func,
            description="Reconnect to service",
            success_rate=0.75,
        )

        assert strategy.success_rate == 0.75


class TestSimplifiedStartupManager:
    """Test the SimplifiedStartupManager class (used by app.py)"""

    @pytest.fixture
    def manager(self):
        """Create a fresh startup manager for each test"""
        from src.vega.core.resilient_startup import SimplifiedStartupManager

        return SimplifiedStartupManager()

    def test_manager_initialization(self, manager):
        """Test manager initializes with empty state"""
        assert manager.features == {}
        assert manager.feature_status == {}
        assert manager.is_running is False

    def test_register_feature(self, manager):
        """Test registering a feature"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="test_feature", category=FeatureCategory.STANDARD, init_func=init_func
        )

        manager.register_feature(feature)

        assert "test_feature" in manager.features
        assert manager.features["test_feature"] == feature

    def test_register_duplicate_feature_overwrites(self, manager):
        """Test that registering duplicate feature overwrites"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature1 = StartupFeature(
            name="duplicate", category=FeatureCategory.STANDARD, init_func=init_func
        )
        feature2 = StartupFeature(
            name="duplicate", category=FeatureCategory.HIGH, init_func=init_func
        )

        manager.register_feature(feature1)
        manager.register_feature(feature2)

        # Second registration should overwrite
        assert manager.features["duplicate"].category == FeatureCategory.HIGH

    @pytest.mark.asyncio
    async def test_init_feature_success(self, manager):
        """Test successful feature initialization"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        init_called = False

        async def init_func():
            nonlocal init_called
            init_called = True
            return True

        feature = StartupFeature(
            name="success_feature",
            category=FeatureCategory.STANDARD,
            init_func=init_func,
        )

        manager.register_feature(feature)
        success = await manager._init_feature(feature)  # pylint: disable=protected-access

        assert success is True
        assert init_called is True
        assert manager.feature_status["success_feature"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_init_feature_failure(self, manager):
        """Test feature initialization failure"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def failing_init():
            raise RuntimeError("Init failed!")

        feature = StartupFeature(
            name="failing_feature",
            category=FeatureCategory.STANDARD,
            init_func=failing_init,
        )

        manager.register_feature(feature)
        success = await manager._init_feature(feature)  # pylint: disable=protected-access

        assert success is False
        assert manager.feature_status["failing_feature"]["status"] == "failed"
        assert "Init failed!" in manager.feature_status["failing_feature"]["error"]

    @pytest.mark.asyncio
    async def test_critical_feature_blocks_startup(self, manager):
        """Test that critical feature failure blocks startup"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def failing_critical():
            raise RuntimeError("Critical failure!")

        feature = StartupFeature(
            name="critical_feature",
            category=FeatureCategory.CRITICAL,
            init_func=failing_critical,
        )

        manager.register_feature(feature)

        # startup_sequence should return False when critical feature fails
        success = await manager.startup_sequence()

        assert success is False

    @pytest.mark.asyncio
    async def test_non_critical_feature_does_not_block(self, manager):
        """Test that non-critical failure doesn't block startup"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def working_critical():
            return True

        async def failing_optional():
            raise RuntimeError("Optional failure!")

        critical = StartupFeature(
            name="critical",
            category=FeatureCategory.CRITICAL,
            init_func=working_critical,
        )
        optional = StartupFeature(
            name="optional",
            category=FeatureCategory.OPTIONAL,
            init_func=failing_optional,
        )

        manager.register_feature(critical)
        manager.register_feature(optional)

        success = await manager.startup_sequence()

        # Startup should succeed even with optional failure
        assert success is True
        assert manager.feature_status["critical"]["status"] == "healthy"
        assert manager.feature_status["optional"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_dependency_order(self, manager):
        """Test that features are initialized in dependency order"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        init_order = []

        async def init_base():
            init_order.append("base")
            return True

        async def init_dependent():
            init_order.append("dependent")
            return True

        base = StartupFeature(
            name="base", category=FeatureCategory.CRITICAL, init_func=init_base
        )
        dependent = StartupFeature(
            name="dependent",
            category=FeatureCategory.CRITICAL,
            init_func=init_dependent,
            dependencies=["base"],
        )

        # Register in reverse order to test sorting
        manager.register_feature(dependent)
        manager.register_feature(base)

        await manager.startup_sequence()

        # Base should be initialized before dependent
        assert init_order.index("base") < init_order.index("dependent")

    @pytest.mark.asyncio
    async def test_queue_repair(self, manager):
        """Test queuing a feature for repair"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="repairable", category=FeatureCategory.STANDARD, init_func=init_func
        )

        manager.register_feature(feature)
        manager.feature_status["repairable"] = {"status": "failed"}

        await manager.queue_repair("repairable")

        assert "repairable" in manager.pending_repairs

    def test_usage_tracking(self, manager):
        """Test that feature usage is tracked"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="tracked_feature",
            category=FeatureCategory.STANDARD,
            init_func=init_func,
        )

        manager.register_feature(feature)

        # Simulate usage tracking
        manager.record_feature_usage("tracked_feature")
        manager.record_feature_usage("tracked_feature")
        manager.record_feature_usage("tracked_feature")

        assert manager.feature_usage["tracked_feature"] == 3

    def test_get_status_summary(self, manager):
        """Test getting status summary"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="test", category=FeatureCategory.STANDARD, init_func=init_func
        )

        manager.register_feature(feature)
        manager.feature_status["test"] = {"status": "healthy", "healthy": True}

        summary = manager.get_status_summary()

        assert "healthy_features" in summary
        assert "test" in summary["healthy_features"]

    @pytest.mark.asyncio
    async def test_shutdown(self, manager):
        """Test graceful shutdown"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature = StartupFeature(
            name="test", category=FeatureCategory.STANDARD, init_func=init_func
        )

        manager.register_feature(feature)
        await manager.startup_sequence()

        # Should not raise
        await manager.shutdown()

        assert manager.is_running is False


class TestRepairKnowledgeBase:
    """Test the repair knowledge base functionality"""

    @pytest.fixture
    def manager(self):
        from src.vega.core.resilient_startup import SimplifiedStartupManager

        return SimplifiedStartupManager()

    def test_add_to_knowledge_base(self, manager):
        """Test adding successful repair to knowledge base"""
        manager.add_to_knowledge_base(
            feature_name="test_feature",
            strategy_name="restart",
            error_pattern="connection refused",
        )

        assert "test_feature" in manager.repair_knowledge_base
        assert manager.repair_knowledge_base["test_feature"]["strategy"] == "restart"

    def test_get_best_repair_strategy(self, manager):
        """Test getting best repair strategy from knowledge base"""
        from src.vega.core.resilient_startup import (
            StartupFeature,
            FeatureCategory,
            RepairStrategy,
        )

        async def repair_func():
            return True

        async def init_func():
            return True

        strategy = RepairStrategy(
            name="known_fix",
            repair_func=repair_func,
            description="Known working fix",
            success_rate=0.9,
        )

        feature = StartupFeature(
            name="test_feature",
            category=FeatureCategory.STANDARD,
            init_func=init_func,
            repair_strategies=[strategy],
        )

        manager.register_feature(feature)
        manager.add_to_knowledge_base(
            feature_name="test_feature",
            strategy_name="known_fix",
            error_pattern="specific error",
        )

        best = manager.get_best_repair_strategy(
            "test_feature", "specific error occurred"
        )

        # Should prefer the known working strategy
        assert best is not None
        assert best.name == "known_fix"


class TestPriorityBasedRepair:
    """Test priority-based repair scheduling"""

    @pytest.fixture
    def manager(self):
        from src.vega.core.resilient_startup import SimplifiedStartupManager

        return SimplifiedStartupManager()

    def test_high_usage_features_prioritized(self, manager):
        """Test that frequently used features get higher repair priority"""
        from src.vega.core.resilient_startup import StartupFeature, FeatureCategory

        async def init_func():
            return True

        feature1 = StartupFeature(
            name="low_usage", category=FeatureCategory.STANDARD, init_func=init_func
        )
        feature2 = StartupFeature(
            name="high_usage", category=FeatureCategory.STANDARD, init_func=init_func
        )

        manager.register_feature(feature1)
        manager.register_feature(feature2)

        # Simulate different usage patterns
        manager.record_feature_usage("low_usage")
        for _ in range(10):
            manager.record_feature_usage("high_usage")

        # High usage feature should have higher priority score
        priority1 = manager.calculate_repair_priority("low_usage")
        priority2 = manager.calculate_repair_priority("high_usage")

        # Higher priority value = higher priority (will be sorted first)
        assert priority2 > priority1


# Integration test with app.py endpoints
class TestStartupEndpoints:
    """Test the startup management API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from src.vega.core.app import app

        return TestClient(app)

    def test_startup_status_endpoint(self, client):
        """Test /system/startup/status endpoint"""
        response = client.get("/system/startup/status")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "mode" in data

    def test_startup_features_endpoint(self, client):
        """Test /system/startup/features endpoint"""
        response = client.get("/system/startup/features")
        assert response.status_code == 200
        data = response.json()
        # Either has features or error (depending on manager state)
        assert "features" in data or "error" in data

    def test_repairs_endpoint(self, client):
        """Test /system/startup/repairs endpoint"""
        response = client.get("/system/startup/repairs")
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data or "error" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
