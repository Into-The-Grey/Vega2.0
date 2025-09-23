"""
Personal Testing Framework for ML Models

Provides automated model validation, A/B testing for personal models,
statistical significance testing, and model comparison tools.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager
import aiosqlite
from abc import ABC, abstractmethod
import warnings
import tempfile

logger = logging.getLogger(__name__)

try:
    import scipy.stats as stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - statistical tests disabled")

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        roc_auc_score,
        confusion_matrix,
        classification_report,
    )
    from sklearn.model_selection import cross_val_score, train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - some metrics disabled")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Matplotlib/Seaborn not available - plotting disabled")


class TestType(Enum):
    """Types of model tests"""

    UNIT_TEST = "unit_test"
    INTEGRATION_TEST = "integration_test"
    PERFORMANCE_TEST = "performance_test"
    AB_TEST = "ab_test"
    REGRESSION_TEST = "regression_test"
    STRESS_TEST = "stress_test"
    ACCURACY_TEST = "accuracy_test"
    BIAS_TEST = "bias_test"


class TestStatus(Enum):
    """Test execution status"""

    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class MetricType(Enum):
    """Types of metrics"""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    MSE = "mse"
    MAE = "mae"
    R2_SCORE = "r2_score"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CUSTOM = "custom"


@dataclass
class TestResult:
    """Test result information"""

    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    model_id: str
    model_version: str
    dataset_name: str
    metrics: Dict[str, float]
    execution_time: float
    created_at: datetime
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.metrics is None:
            self.metrics = {}
        if self.details is None:
            self.details = {}


@dataclass
class ABTestConfig:
    """A/B test configuration"""

    test_name: str
    model_a_id: str
    model_b_id: str
    traffic_split: float  # Percentage for model A (0.0 to 1.0)
    sample_size: int
    confidence_level: float
    minimum_effect_size: float
    max_duration_days: int
    success_metric: MetricType

    def __post_init__(self):
        if not (0.0 <= self.traffic_split <= 1.0):
            raise ValueError("Traffic split must be between 0.0 and 1.0")
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError("Confidence level must be between 0.0 and 1.0")


@dataclass
class TestSuite:
    """Collection of related tests"""

    suite_id: str
    name: str
    description: str
    tests: List[str]  # Test IDs
    created_at: datetime
    author: str
    tags: List[str]

    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if self.tests is None:
            self.tests = []
        if self.tags is None:
            self.tags = []


class ModelTester(Protocol):
    """Protocol for model testing interface"""

    def predict(self, data: Any) -> Any:
        """Make predictions on data"""
        ...

    def predict_proba(self, data: Any) -> Any:
        """Get prediction probabilities (for classification)"""
        ...


class BaseModelTest(ABC):
    """Base class for model tests"""

    def __init__(self, test_name: str, test_type: TestType):
        self.test_name = test_name
        self.test_type = test_type
        self.test_id = f"{test_name}_{int(time.time())}"

    @abstractmethod
    async def run_test(
        self, model: ModelTester, test_data: Any, ground_truth: Any = None
    ) -> TestResult:
        """Run the test and return results"""
        pass

    def _create_result(
        self,
        model_id: str,
        model_version: str,
        dataset_name: str,
        status: TestStatus,
        metrics: Dict[str, float],
        execution_time: float,
        error_message: str = None,
        details: Dict[str, Any] = None,
    ) -> TestResult:
        """Helper to create test result"""
        return TestResult(
            test_id=self.test_id,
            test_name=self.test_name,
            test_type=self.test_type,
            status=status,
            model_id=model_id,
            model_version=model_version,
            dataset_name=dataset_name,
            metrics=metrics or {},
            execution_time=execution_time,
            created_at=datetime.now(),
            error_message=error_message,
            details=details or {},
        )


class AccuracyTest(BaseModelTest):
    """Test model accuracy against ground truth"""

    def __init__(self, minimum_accuracy: float = 0.8):
        super().__init__("accuracy_test", TestType.ACCURACY_TEST)
        self.minimum_accuracy = minimum_accuracy

    async def run_test(
        self, model: ModelTester, test_data: Any, ground_truth: Any = None
    ) -> TestResult:
        start_time = time.time()

        try:
            # Make predictions
            predictions = model.predict(test_data)

            if ground_truth is None:
                raise ValueError("Ground truth required for accuracy test")

            # Calculate metrics
            metrics = {}

            if SKLEARN_AVAILABLE:
                # Classification metrics
                try:
                    metrics["accuracy"] = accuracy_score(ground_truth, predictions)
                    metrics["precision"] = precision_score(
                        ground_truth, predictions, average="weighted", zero_division=0
                    )
                    metrics["recall"] = recall_score(
                        ground_truth, predictions, average="weighted", zero_division=0
                    )
                    metrics["f1_score"] = f1_score(
                        ground_truth, predictions, average="weighted", zero_division=0
                    )

                    # ROC-AUC for binary classification
                    if len(np.unique(ground_truth)) == 2:
                        try:
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(test_data)
                                if proba.shape[1] == 2:
                                    metrics["roc_auc"] = roc_auc_score(
                                        ground_truth, proba[:, 1]
                                    )
                        except Exception:
                            pass

                except ValueError:
                    # Regression metrics
                    metrics["mse"] = mean_squared_error(ground_truth, predictions)
                    metrics["mae"] = mean_absolute_error(ground_truth, predictions)
                    metrics["r2_score"] = r2_score(ground_truth, predictions)

            else:
                # Basic accuracy calculation
                if hasattr(ground_truth, "__len__") and hasattr(predictions, "__len__"):
                    correct = sum(
                        1 for i, j in zip(ground_truth, predictions) if i == j
                    )
                    metrics["accuracy"] = correct / len(ground_truth)

            # Determine test status
            main_metric = metrics.get("accuracy", metrics.get("r2_score", 0))
            status = (
                TestStatus.PASSED
                if main_metric >= self.minimum_accuracy
                else TestStatus.FAILED
            )

            execution_time = time.time() - start_time

            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=status,
                metrics=metrics,
                execution_time=execution_time,
                details={"minimum_accuracy": self.minimum_accuracy},
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=TestStatus.ERROR,
                metrics={},
                execution_time=execution_time,
                error_message=str(e),
            )


class PerformanceTest(BaseModelTest):
    """Test model performance (latency, throughput)"""

    def __init__(self, max_latency_ms: float = 1000, min_throughput: float = 10):
        super().__init__("performance_test", TestType.PERFORMANCE_TEST)
        self.max_latency_ms = max_latency_ms
        self.min_throughput = min_throughput

    async def run_test(
        self, model: ModelTester, test_data: Any, ground_truth: Any = None
    ) -> TestResult:
        start_time = time.time()

        try:
            # Test latency with single predictions
            latencies = []
            for i in range(
                min(10, len(test_data) if hasattr(test_data, "__len__") else 10)
            ):
                sample = (
                    test_data[i] if hasattr(test_data, "__getitem__") else test_data
                )

                lat_start = time.time()
                _ = model.predict(sample)
                lat_end = time.time()

                latencies.append((lat_end - lat_start) * 1000)  # Convert to ms

            avg_latency = np.mean(latencies)

            # Test throughput with batch prediction
            batch_size = min(
                100, len(test_data) if hasattr(test_data, "__len__") else 100
            )
            batch_data = (
                test_data[:batch_size]
                if hasattr(test_data, "__getitem__")
                else test_data
            )

            throughput_start = time.time()
            _ = model.predict(batch_data)
            throughput_end = time.time()

            throughput = batch_size / (throughput_end - throughput_start)

            metrics = {
                "avg_latency_ms": avg_latency,
                "throughput_per_sec": throughput,
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
            }

            # Determine test status
            latency_ok = avg_latency <= self.max_latency_ms
            throughput_ok = throughput >= self.min_throughput
            status = (
                TestStatus.PASSED if latency_ok and throughput_ok else TestStatus.FAILED
            )

            execution_time = time.time() - start_time

            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=status,
                metrics=metrics,
                execution_time=execution_time,
                details={
                    "max_latency_ms": self.max_latency_ms,
                    "min_throughput": self.min_throughput,
                    "latency_ok": latency_ok,
                    "throughput_ok": throughput_ok,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=TestStatus.ERROR,
                metrics={},
                execution_time=execution_time,
                error_message=str(e),
            )


class RegressionTest(BaseModelTest):
    """Test for model regression (degradation from baseline)"""

    def __init__(self, baseline_metrics: Dict[str, float], tolerance: float = 0.05):
        super().__init__("regression_test", TestType.REGRESSION_TEST)
        self.baseline_metrics = baseline_metrics
        self.tolerance = tolerance

    async def run_test(
        self, model: ModelTester, test_data: Any, ground_truth: Any = None
    ) -> TestResult:
        start_time = time.time()

        try:
            # Run accuracy test to get current metrics
            accuracy_test = AccuracyTest()
            current_result = await accuracy_test.run_test(
                model, test_data, ground_truth
            )
            current_metrics = current_result.metrics

            # Compare with baseline
            regression_details = {}
            has_regression = False

            for metric_name, baseline_value in self.baseline_metrics.items():
                if metric_name in current_metrics:
                    current_value = current_metrics[metric_name]
                    diff = baseline_value - current_value
                    relative_diff = diff / baseline_value if baseline_value != 0 else 0

                    regression_details[metric_name] = {
                        "baseline": baseline_value,
                        "current": current_value,
                        "absolute_diff": diff,
                        "relative_diff": relative_diff,
                        "regression": relative_diff > self.tolerance,
                    }

                    if relative_diff > self.tolerance:
                        has_regression = True

            status = TestStatus.FAILED if has_regression else TestStatus.PASSED
            execution_time = time.time() - start_time

            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=status,
                metrics=current_metrics,
                execution_time=execution_time,
                details={
                    "baseline_metrics": self.baseline_metrics,
                    "tolerance": self.tolerance,
                    "regression_analysis": regression_details,
                    "has_regression": has_regression,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=TestStatus.ERROR,
                metrics={},
                execution_time=execution_time,
                error_message=str(e),
            )


class BiasTest(BaseModelTest):
    """Test for model bias across different groups"""

    def __init__(self, protected_attribute: str, fairness_threshold: float = 0.1):
        super().__init__("bias_test", TestType.BIAS_TEST)
        self.protected_attribute = protected_attribute
        self.fairness_threshold = fairness_threshold

    async def run_test(
        self, model: ModelTester, test_data: Any, ground_truth: Any = None
    ) -> TestResult:
        start_time = time.time()

        try:
            if (
                not hasattr(test_data, "columns")
                or self.protected_attribute not in test_data.columns
            ):
                raise ValueError(
                    f"Protected attribute '{self.protected_attribute}' not found in test data"
                )

            predictions = model.predict(
                test_data.drop(columns=[self.protected_attribute])
            )

            # Calculate metrics for each group
            groups = test_data[self.protected_attribute].unique()
            group_metrics = {}

            for group in groups:
                group_mask = test_data[self.protected_attribute] == group
                group_pred = predictions[group_mask]
                group_truth = (
                    ground_truth[group_mask] if ground_truth is not None else None
                )

                if SKLEARN_AVAILABLE and group_truth is not None:
                    group_accuracy = accuracy_score(group_truth, group_pred)
                    group_metrics[str(group)] = {
                        "accuracy": group_accuracy,
                        "sample_size": sum(group_mask),
                    }

            # Calculate fairness metrics
            fairness_metrics = {}
            group_accuracies = [
                metrics["accuracy"] for metrics in group_metrics.values()
            ]

            if len(group_accuracies) >= 2:
                fairness_metrics["accuracy_range"] = max(group_accuracies) - min(
                    group_accuracies
                )
                fairness_metrics["accuracy_std"] = np.std(group_accuracies)

                # Demographic parity (difference in positive prediction rates)
                group_positive_rates = []
                for group in groups:
                    group_mask = test_data[self.protected_attribute] == group
                    group_pred = predictions[group_mask]
                    positive_rate = np.mean(group_pred) if len(group_pred) > 0 else 0
                    group_positive_rates.append(positive_rate)

                fairness_metrics["demographic_parity_diff"] = max(
                    group_positive_rates
                ) - min(group_positive_rates)

            # Determine bias
            has_bias = (
                fairness_metrics.get("accuracy_range", 0) > self.fairness_threshold
                or fairness_metrics.get("demographic_parity_diff", 0)
                > self.fairness_threshold
            )

            status = TestStatus.FAILED if has_bias else TestStatus.PASSED
            execution_time = time.time() - start_time

            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=status,
                metrics=fairness_metrics,
                execution_time=execution_time,
                details={
                    "protected_attribute": self.protected_attribute,
                    "fairness_threshold": self.fairness_threshold,
                    "group_metrics": group_metrics,
                    "has_bias": has_bias,
                },
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return self._create_result(
                model_id="unknown",
                model_version="unknown",
                dataset_name="test_data",
                status=TestStatus.ERROR,
                metrics={},
                execution_time=execution_time,
                error_message=str(e),
            )


class ABTestFramework:
    """A/B testing framework for model comparison"""

    def __init__(self):
        self.active_tests = {}
        self.test_results = {}

    def create_ab_test(self, config: ABTestConfig) -> str:
        """Create new A/B test"""
        test_id = f"ab_test_{config.test_name}_{int(time.time())}"

        self.active_tests[test_id] = {
            "config": config,
            "model_a_results": [],
            "model_b_results": [],
            "start_time": datetime.now(),
            "status": "active",
        }

        logger.info(f"Created A/B test: {config.test_name}")
        return test_id

    def record_result(
        self,
        test_id: str,
        model_type: str,
        prediction: Any,
        ground_truth: Any,
        latency: float,
    ):
        """Record result for A/B test"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_data = self.active_tests[test_id]

        result = {
            "prediction": prediction,
            "ground_truth": ground_truth,
            "latency": latency,
            "timestamp": datetime.now(),
        }

        if model_type == "A":
            test_data["model_a_results"].append(result)
        elif model_type == "B":
            test_data["model_b_results"].append(result)
        else:
            raise ValueError("Model type must be 'A' or 'B'")

    def analyze_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze A/B test results"""
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")

        test_data = self.active_tests[test_id]
        config = test_data["config"]

        model_a_results = test_data["model_a_results"]
        model_b_results = test_data["model_b_results"]

        analysis = {
            "test_id": test_id,
            "config": asdict(config),
            "sample_sizes": {
                "model_a": len(model_a_results),
                "model_b": len(model_b_results),
            },
            "metrics": {},
            "statistical_tests": {},
            "recommendation": None,
        }

        if len(model_a_results) == 0 or len(model_b_results) == 0:
            analysis["recommendation"] = "Insufficient data for analysis"
            return analysis

        # Calculate metrics for both models
        if SKLEARN_AVAILABLE:
            for model_type, results in [
                ("model_a", model_a_results),
                ("model_b", model_b_results),
            ]:
                predictions = [r["prediction"] for r in results]
                ground_truths = [
                    r["ground_truth"] for r in results if r["ground_truth"] is not None
                ]
                latencies = [r["latency"] for r in results]

                metrics = {
                    "avg_latency": np.mean(latencies),
                    "latency_std": np.std(latencies),
                }

                if ground_truths and len(ground_truths) == len(predictions):
                    try:
                        metrics["accuracy"] = accuracy_score(ground_truths, predictions)
                        metrics["precision"] = precision_score(
                            ground_truths,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                        metrics["recall"] = recall_score(
                            ground_truths,
                            predictions,
                            average="weighted",
                            zero_division=0,
                        )
                    except:
                        # Regression metrics
                        metrics["mse"] = mean_squared_error(ground_truths, predictions)
                        metrics["mae"] = mean_absolute_error(ground_truths, predictions)

                analysis["metrics"][model_type] = metrics

        # Statistical significance testing
        if (
            SCIPY_AVAILABLE
            and len(model_a_results) >= 10
            and len(model_b_results) >= 10
        ):
            # Get success metric values
            success_metric = config.success_metric.value

            if (
                success_metric in analysis["metrics"]["model_a"]
                and success_metric in analysis["metrics"]["model_b"]
            ):
                metric_a = analysis["metrics"]["model_a"][success_metric]
                metric_b = analysis["metrics"]["model_b"][success_metric]

                # For accuracy-type metrics, perform proportion test
                if success_metric in ["accuracy", "precision", "recall"]:
                    # Convert to counts for proportion test
                    successes_a = int(metric_a * len(model_a_results))
                    successes_b = int(metric_b * len(model_b_results))

                    # Two-proportion z-test
                    p1 = successes_a / len(model_a_results)
                    p2 = successes_b / len(model_b_results)

                    pooled_p = (successes_a + successes_b) / (
                        len(model_a_results) + len(model_b_results)
                    )
                    se = np.sqrt(
                        pooled_p
                        * (1 - pooled_p)
                        * (1 / len(model_a_results) + 1 / len(model_b_results))
                    )

                    if se > 0:
                        z_score = (p1 - p2) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                        analysis["statistical_tests"] = {
                            "test_type": "two_proportion_z_test",
                            "z_score": z_score,
                            "p_value": p_value,
                            "significant": p_value < (1 - config.confidence_level),
                            "effect_size": abs(p1 - p2),
                        }

                # For continuous metrics, perform t-test
                else:
                    values_a = [analysis["metrics"]["model_a"][success_metric]] * len(
                        model_a_results
                    )
                    values_b = [analysis["metrics"]["model_b"][success_metric]] * len(
                        model_b_results
                    )

                    t_stat, p_value = stats.ttest_ind(values_a, values_b)

                    analysis["statistical_tests"] = {
                        "test_type": "independent_t_test",
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "significant": p_value < (1 - config.confidence_level),
                        "effect_size": abs(metric_a - metric_b),
                    }

        # Generate recommendation
        if "statistical_tests" in analysis and analysis["statistical_tests"]:
            stats_test = analysis["statistical_tests"]

            if (
                stats_test["significant"]
                and stats_test["effect_size"] >= config.minimum_effect_size
            ):
                # Determine winner based on success metric
                if config.success_metric.value in analysis["metrics"]["model_a"]:
                    metric_a = analysis["metrics"]["model_a"][
                        config.success_metric.value
                    ]
                    metric_b = analysis["metrics"]["model_b"][
                        config.success_metric.value
                    ]

                    if metric_a > metric_b:
                        analysis["recommendation"] = "Model A is significantly better"
                    else:
                        analysis["recommendation"] = "Model B is significantly better"
                else:
                    analysis["recommendation"] = (
                        "Significant difference detected, manual review needed"
                    )
            else:
                analysis["recommendation"] = "No significant difference between models"
        else:
            analysis["recommendation"] = "Insufficient data for statistical analysis"

        return analysis

    def stop_ab_test(self, test_id: str) -> Dict[str, Any]:
        """Stop A/B test and get final analysis"""
        if test_id in self.active_tests:
            self.active_tests[test_id]["status"] = "completed"
            final_analysis = self.analyze_ab_test(test_id)
            self.test_results[test_id] = final_analysis

            logger.info(f"Stopped A/B test: {test_id}")
            return final_analysis

        raise ValueError(f"Test {test_id} not found")


class TestingFramework:
    """Main testing framework for ML models"""

    def __init__(self, db_path: str = "data/testing_framework.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.ab_framework = ABTestFramework()
        self.test_suites = {}

        # Initialize database
        asyncio.create_task(self._initialize_database())

    async def _initialize_database(self):
        """Initialize SQLite database for test results"""
        async with aiosqlite.connect(self.db_path) as db:
            # Test results table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS test_results (
                    test_id TEXT PRIMARY KEY,
                    test_name TEXT,
                    test_type TEXT,
                    status TEXT,
                    model_id TEXT,
                    model_version TEXT,
                    dataset_name TEXT,
                    metrics TEXT,
                    execution_time REAL,
                    created_at TEXT,
                    error_message TEXT,
                    details TEXT
                )
            """
            )

            # Test suites table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS test_suites (
                    suite_id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    tests TEXT,
                    created_at TEXT,
                    author TEXT,
                    tags TEXT
                )
            """
            )

            # A/B test results table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS ab_test_results (
                    test_id TEXT PRIMARY KEY,
                    test_name TEXT,
                    model_a_id TEXT,
                    model_b_id TEXT,
                    config TEXT,
                    analysis TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )
            """
            )

            # Create indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_results_model ON test_results(model_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_results_type ON test_results(test_type)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_test_results_status ON test_results(status)"
            )

            await db.commit()

    async def run_test(
        self,
        test: BaseModelTest,
        model: ModelTester,
        test_data: Any,
        ground_truth: Any = None,
        model_id: str = "unknown",
        model_version: str = "unknown",
        dataset_name: str = "test_data",
    ) -> TestResult:
        """Run a single test"""

        result = await test.run_test(model, test_data, ground_truth)

        # Update result with provided metadata
        result.model_id = model_id
        result.model_version = model_version
        result.dataset_name = dataset_name

        # Store result in database
        await self._store_test_result(result)

        logger.info(f"Test {test.test_name} completed: {result.status.value}")
        return result

    async def _store_test_result(self, result: TestResult):
        """Store test result in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO test_results 
                (test_id, test_name, test_type, status, model_id, model_version, 
                 dataset_name, metrics, execution_time, created_at, error_message, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.test_id,
                    result.test_name,
                    result.test_type.value,
                    result.status.value,
                    result.model_id,
                    result.model_version,
                    result.dataset_name,
                    json.dumps(result.metrics),
                    result.execution_time,
                    result.created_at.isoformat(),
                    result.error_message,
                    json.dumps(result.details),
                ),
            )
            await db.commit()

    async def run_test_suite(
        self,
        suite: TestSuite,
        model: ModelTester,
        test_data: Any,
        ground_truth: Any = None,
        model_id: str = "unknown",
        model_version: str = "unknown",
    ) -> List[TestResult]:
        """Run all tests in a test suite"""

        results = []

        # Load test definitions (this would be expanded with actual test loading)
        for test_id in suite.tests:
            # For demo, create basic tests
            if "accuracy" in test_id.lower():
                test = AccuracyTest()
            elif "performance" in test_id.lower():
                test = PerformanceTest()
            elif "regression" in test_id.lower():
                test = RegressionTest({})  # Would load baseline from DB
            else:
                continue

            result = await self.run_test(
                test, model, test_data, ground_truth, model_id, model_version
            )
            results.append(result)

        logger.info(f"Test suite {suite.name} completed: {len(results)} tests")
        return results

    async def create_test_suite(self, suite: TestSuite) -> str:
        """Create new test suite"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO test_suites 
                (suite_id, name, description, tests, created_at, author, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    suite.suite_id,
                    suite.name,
                    suite.description,
                    json.dumps(suite.tests),
                    suite.created_at.isoformat(),
                    suite.author,
                    json.dumps(suite.tags),
                ),
            )
            await db.commit()

        self.test_suites[suite.suite_id] = suite
        logger.info(f"Created test suite: {suite.name}")
        return suite.suite_id

    async def get_test_results(
        self,
        model_id: Optional[str] = None,
        test_type: Optional[TestType] = None,
        status: Optional[TestStatus] = None,
        limit: int = 100,
    ) -> List[TestResult]:
        """Get test results with optional filtering"""

        query = "SELECT * FROM test_results WHERE 1=1"
        params = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        if test_type:
            query += " AND test_type = ?"
            params.append(test_type.value)

        if status:
            query += " AND status = ?"
            params.append(status.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            results = []
            for row in rows:
                columns = [description[0] for description in cursor.description]
                data = dict(zip(columns, row))

                # Parse JSON fields
                data["metrics"] = json.loads(data["metrics"])
                data["details"] = json.loads(data["details"]) if data["details"] else {}
                data["test_type"] = TestType(data["test_type"])
                data["status"] = TestStatus(data["status"])

                results.append(TestResult(**data))

            return results

    async def generate_test_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive test report for a model"""

        results = await self.get_test_results(model_id=model_id)

        report = {
            "model_id": model_id,
            "total_tests": len(results),
            "test_summary": {},
            "metrics_summary": {},
            "recent_tests": [],
            "trend_analysis": {},
            "generated_at": datetime.now().isoformat(),
        }

        # Test summary by status
        for status in TestStatus:
            count = sum(1 for r in results if r.status == status)
            report["test_summary"][status.value] = count

        # Metrics summary
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        for metric_name, values in all_metrics.items():
            if values:
                report["metrics_summary"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        # Recent tests
        report["recent_tests"] = [
            {
                "test_name": r.test_name,
                "test_type": r.test_type.value,
                "status": r.status.value,
                "created_at": r.created_at.isoformat(),
                "key_metrics": {
                    k: v
                    for k, v in r.metrics.items()
                    if k in ["accuracy", "f1_score", "mse"]
                },
            }
            for r in results[:10]
        ]

        return report

    def create_custom_test(
        self, test_name: str, test_function: Callable
    ) -> BaseModelTest:
        """Create custom test from function"""

        class CustomTest(BaseModelTest):
            def __init__(self, name: str, func: Callable):
                super().__init__(name, TestType.UNIT_TEST)
                self.test_func = func

            async def run_test(
                self, model: ModelTester, test_data: Any, ground_truth: Any = None
            ) -> TestResult:
                start_time = time.time()

                try:
                    result = self.test_func(model, test_data, ground_truth)

                    if isinstance(result, dict):
                        metrics = result.get("metrics", {})
                        status = (
                            TestStatus.PASSED
                            if result.get("passed", True)
                            else TestStatus.FAILED
                        )
                        details = result.get("details", {})
                    else:
                        metrics = {}
                        status = TestStatus.PASSED if result else TestStatus.FAILED
                        details = {}

                    execution_time = time.time() - start_time

                    return self._create_result(
                        model_id="unknown",
                        model_version="unknown",
                        dataset_name="test_data",
                        status=status,
                        metrics=metrics,
                        execution_time=execution_time,
                        details=details,
                    )

                except Exception as e:
                    execution_time = time.time() - start_time
                    return self._create_result(
                        model_id="unknown",
                        model_version="unknown",
                        dataset_name="test_data",
                        status=TestStatus.ERROR,
                        metrics={},
                        execution_time=execution_time,
                        error_message=str(e),
                    )

        return CustomTest(test_name, test_function)


# Demo and testing functions
class DummyModel:
    """Dummy model for testing"""

    def __init__(self, accuracy: float = 0.85):
        self.accuracy = accuracy
        self.latency_ms = 50

    def predict(self, data):
        """Make predictions"""
        if hasattr(data, "__len__"):
            # Random predictions based on desired accuracy
            predictions = []
            for _ in range(len(data)):
                pred = 1 if np.random.random() < self.accuracy else 0
                predictions.append(pred)
            return np.array(predictions)
        else:
            return 1 if np.random.random() < self.accuracy else 0

    def predict_proba(self, data):
        """Get prediction probabilities"""
        predictions = self.predict(data)
        probabilities = []

        for pred in predictions:
            if pred == 1:
                prob = [0.2, 0.8]  # High confidence positive
            else:
                prob = [0.8, 0.2]  # High confidence negative
            probabilities.append(prob)

        return np.array(probabilities)


async def demo_testing_framework():
    """Demonstrate testing framework capabilities"""

    print("Personal Testing Framework Demo")

    # Initialize framework
    framework = TestingFramework("data/testing_demo.db")
    await framework._initialize_database()

    # Create dummy models
    model_a = DummyModel(accuracy=0.85)
    model_b = DummyModel(accuracy=0.82)

    # Generate test data
    np.random.seed(42)
    test_data = np.random.randn(100, 10)
    ground_truth = np.random.randint(0, 2, 100)

    print("\n1. Running Individual Tests")

    # Test model A
    print("\nTesting Model A:")

    # Accuracy test
    accuracy_test = AccuracyTest(minimum_accuracy=0.8)
    result = await framework.run_test(
        accuracy_test,
        model_a,
        test_data,
        ground_truth,
        "model_a",
        "1.0.0",
        "test_dataset",
    )
    print(f"Accuracy Test: {result.status.value}")
    print(f"Metrics: {result.metrics}")

    # Performance test
    performance_test = PerformanceTest(max_latency_ms=100, min_throughput=10)
    result = await framework.run_test(
        performance_test, model_a, test_data, None, "model_a", "1.0.0", "test_dataset"
    )
    print(f"Performance Test: {result.status.value}")
    print(f"Latency: {result.metrics.get('avg_latency_ms', 'N/A'):.2f}ms")
    print(f"Throughput: {result.metrics.get('throughput_per_sec', 'N/A'):.1f}/sec")

    # Regression test (comparing to baseline)
    baseline_metrics = {"accuracy": 0.8, "f1_score": 0.75}
    regression_test = RegressionTest(baseline_metrics, tolerance=0.05)
    result = await framework.run_test(
        regression_test,
        model_a,
        test_data,
        ground_truth,
        "model_a",
        "1.0.0",
        "test_dataset",
    )
    print(f"Regression Test: {result.status.value}")

    print("\n2. A/B Testing")

    # Create A/B test
    ab_config = ABTestConfig(
        test_name="model_comparison",
        model_a_id="model_a",
        model_b_id="model_b",
        traffic_split=0.5,
        sample_size=50,
        confidence_level=0.95,
        minimum_effect_size=0.05,
        max_duration_days=7,
        success_metric=MetricType.ACCURACY,
    )

    ab_test_id = framework.ab_framework.create_ab_test(ab_config)
    print(f"Created A/B test: {ab_test_id}")

    # Simulate A/B test data collection
    for i in range(50):
        # Model A predictions
        pred_a = model_a.predict(test_data[i])
        framework.ab_framework.record_result(
            ab_test_id, "A", pred_a, ground_truth[i], 45.0
        )

        # Model B predictions
        pred_b = model_b.predict(test_data[i])
        framework.ab_framework.record_result(
            ab_test_id, "B", pred_b, ground_truth[i], 42.0
        )

    # Analyze A/B test
    analysis = framework.ab_framework.analyze_ab_test(ab_test_id)
    print(f"\nA/B Test Analysis:")
    print(
        f"Model A Accuracy: {analysis['metrics']['model_a'].get('accuracy', 'N/A'):.3f}"
    )
    print(
        f"Model B Accuracy: {analysis['metrics']['model_b'].get('accuracy', 'N/A'):.3f}"
    )
    print(f"Recommendation: {analysis['recommendation']}")

    if "statistical_tests" in analysis:
        stats_test = analysis["statistical_tests"]
        print(f"Statistical Test: {stats_test['test_type']}")
        print(f"P-value: {stats_test['p_value']:.4f}")
        print(f"Significant: {stats_test['significant']}")

    print("\n3. Test Suite Creation")

    # Create test suite
    test_suite = TestSuite(
        suite_id="model_validation_suite",
        name="Model Validation Suite",
        description="Comprehensive validation tests for ML models",
        tests=["accuracy_test", "performance_test", "regression_test"],
        created_at=datetime.now(),
        author="Vega2.0",
        tags=["validation", "accuracy", "performance"],
    )

    suite_id = await framework.create_test_suite(test_suite)
    print(f"Created test suite: {test_suite.name}")

    # Run test suite
    suite_results = await framework.run_test_suite(
        test_suite, model_a, test_data, ground_truth, "model_a", "1.0.0"
    )
    print(f"Test suite completed: {len(suite_results)} tests")

    passed_tests = sum(1 for r in suite_results if r.status == TestStatus.PASSED)
    print(f"Passed: {passed_tests}/{len(suite_results)} tests")

    print("\n4. Custom Test Creation")

    # Create custom test
    def custom_prediction_range_test(model, test_data, ground_truth):
        """Custom test to check prediction range"""
        predictions = model.predict(test_data)

        min_pred = min(predictions)
        max_pred = max(predictions)

        # Check if predictions are in expected range
        valid_range = min_pred >= 0 and max_pred <= 1

        return {
            "passed": valid_range,
            "metrics": {
                "min_prediction": float(min_pred),
                "max_prediction": float(max_pred),
                "prediction_range": float(max_pred - min_pred),
            },
            "details": {
                "expected_min": 0,
                "expected_max": 1,
                "valid_range": valid_range,
            },
        }

    custom_test = framework.create_custom_test(
        "prediction_range_test", custom_prediction_range_test
    )
    result = await framework.run_test(
        custom_test,
        model_a,
        test_data,
        ground_truth,
        "model_a",
        "1.0.0",
        "test_dataset",
    )
    print(f"Custom Test: {result.status.value}")
    print(f"Prediction Range: {result.metrics.get('prediction_range', 'N/A'):.3f}")

    print("\n5. Test Report Generation")

    # Generate test report
    report = await framework.generate_test_report("model_a")
    print(f"\nTest Report for model_a:")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Test Summary: {report['test_summary']}")

    if report["metrics_summary"]:
        print("Metrics Summary:")
        for metric, stats in report["metrics_summary"].items():
            print(f"  {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

    # Stop A/B test
    final_analysis = framework.ab_framework.stop_ab_test(ab_test_id)
    print(f"\nA/B Test Final Recommendation: {final_analysis['recommendation']}")

    return framework


if __name__ == "__main__":
    asyncio.run(demo_testing_framework())
