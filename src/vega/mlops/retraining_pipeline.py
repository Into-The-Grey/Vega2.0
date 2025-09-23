"""
Automated Retraining Pipeline

Provides data drift detection, model performance monitoring,
automated retraining triggers, and pipeline orchestration.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import time
import threading
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager
import aiosqlite
from abc import ABC, abstractmethod
import schedule
import warnings

logger = logging.getLogger(__name__)

try:
    import scipy.stats as stats
    from scipy.spatial.distance import wasserstein_distance

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available - some drift detection methods disabled")

try:
    from sklearn.metrics import accuracy_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - some monitoring features disabled")

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some model operations disabled")


class DriftType(Enum):
    """Types of data drift"""

    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CONCEPT_DRIFT = "concept_drift"
    COVARIATE_SHIFT = "covariate_shift"


class DriftSeverity(Enum):
    """Severity levels for drift detection"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriggerType(Enum):
    """Types of retraining triggers"""

    DRIFT_DETECTED = "drift_detected"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TIME_BASED = "time_based"
    DATA_VOLUME = "data_volume"
    MANUAL = "manual"


class PipelineStatus(Enum):
    """Pipeline execution status"""

    IDLE = "idle"
    MONITORING = "monitoring"
    RETRAINING = "retraining"
    VALIDATING = "validating"
    DEPLOYING = "deploying"
    FAILED = "failed"


class RetrainingStrategy(Enum):
    """Retraining strategies"""

    FULL_RETRAIN = "full_retrain"
    INCREMENTAL = "incremental"
    TRANSFER_LEARNING = "transfer_learning"
    ENSEMBLE_UPDATE = "ensemble_update"


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis"""

    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    drift_score: float
    threshold: float
    feature_drifts: Dict[str, float]
    detection_method: str
    timestamp: datetime
    details: Dict[str, Any]

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.feature_drifts is None:
            self.feature_drifts = {}
        if self.details is None:
            self.details = {}


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""

    model_id: str
    model_version: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    custom_metrics: Dict[str, float] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class RetrainingTrigger:
    """Retraining trigger information"""

    trigger_id: str
    trigger_type: TriggerType
    model_id: str
    severity: DriftSeverity
    reason: str
    metadata: Dict[str, Any]
    timestamp: datetime
    acknowledged: bool = False

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrainingJob:
    """Retraining job information"""

    job_id: str
    model_id: str
    trigger_id: str
    strategy: RetrainingStrategy
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    progress: float = 0.0
    logs: List[str] = None
    error_message: Optional[str] = None
    new_model_metrics: Optional[PerformanceMetrics] = None

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if self.end_time and isinstance(self.end_time, str):
            self.end_time = datetime.fromisoformat(self.end_time)
        if self.logs is None:
            self.logs = []


class DriftDetector(ABC):
    """Base class for drift detection methods"""

    @abstractmethod
    def detect_drift(
        self, reference_data: np.ndarray, current_data: np.ndarray, **kwargs
    ) -> DriftDetectionResult:
        """Detect drift between reference and current data"""
        pass


class StatisticalDriftDetector(DriftDetector):
    """Statistical drift detection using various tests"""

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def detect_drift(
        self, reference_data: np.ndarray, current_data: np.ndarray, **kwargs
    ) -> DriftDetectionResult:
        """Detect drift using statistical tests"""

        if not SCIPY_AVAILABLE:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.FEATURE_DRIFT,
                severity=DriftSeverity.LOW,
                drift_score=0.0,
                threshold=self.significance_level,
                feature_drifts={},
                detection_method="statistical_test",
                timestamp=datetime.now(),
                details={"error": "SciPy not available"},
            )

        feature_drifts = {}
        p_values = []

        # Ensure data is 2D
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        # Test each feature
        for i in range(reference_data.shape[1]):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]

            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(ref_feature, cur_feature)
            feature_drifts[f"feature_{i}"] = p_value
            p_values.append(p_value)

        # Overall drift score (minimum p-value)
        drift_score = min(p_values) if p_values else 1.0
        drift_detected = drift_score < self.significance_level

        # Determine severity
        if drift_score < 0.001:
            severity = DriftSeverity.CRITICAL
        elif drift_score < 0.01:
            severity = DriftSeverity.HIGH
        elif drift_score < 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.FEATURE_DRIFT,
            severity=severity,
            drift_score=drift_score,
            threshold=self.significance_level,
            feature_drifts=feature_drifts,
            detection_method="kolmogorov_smirnov",
            timestamp=datetime.now(),
            details={
                "ks_statistics": {
                    f"feature_{i}": stats.ks_2samp(
                        reference_data[:, i], current_data[:, i]
                    )[0]
                    for i in range(reference_data.shape[1])
                },
                "p_values": p_values,
            },
        )


class DistributionDriftDetector(DriftDetector):
    """Drift detection using distribution comparison"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold

    def detect_drift(
        self, reference_data: np.ndarray, current_data: np.ndarray, **kwargs
    ) -> DriftDetectionResult:
        """Detect drift using distribution comparison"""

        feature_drifts = {}
        drift_scores = []

        # Ensure data is 2D
        if reference_data.ndim == 1:
            reference_data = reference_data.reshape(-1, 1)
        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        # Compare distributions for each feature
        for i in range(reference_data.shape[1]):
            ref_feature = reference_data[:, i]
            cur_feature = current_data[:, i]

            # Calculate Wasserstein distance (Earth Mover's Distance)
            if SCIPY_AVAILABLE:
                distance = wasserstein_distance(ref_feature, cur_feature)
            else:
                # Fallback: use simple mean/std comparison
                ref_mean, ref_std = np.mean(ref_feature), np.std(ref_feature)
                cur_mean, cur_std = np.mean(cur_feature), np.std(cur_feature)

                mean_diff = abs(ref_mean - cur_mean) / (ref_std + 1e-8)
                std_diff = abs(ref_std - cur_std) / (ref_std + 1e-8)
                distance = (mean_diff + std_diff) / 2

            feature_drifts[f"feature_{i}"] = distance
            drift_scores.append(distance)

        # Overall drift score (maximum distance)
        drift_score = max(drift_scores) if drift_scores else 0.0
        drift_detected = drift_score > self.threshold

        # Determine severity
        if drift_score > self.threshold * 3:
            severity = DriftSeverity.CRITICAL
        elif drift_score > self.threshold * 2:
            severity = DriftSeverity.HIGH
        elif drift_score > self.threshold:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=DriftType.COVARIATE_SHIFT,
            severity=severity,
            drift_score=drift_score,
            threshold=self.threshold,
            feature_drifts=feature_drifts,
            detection_method=(
                "wasserstein_distance" if SCIPY_AVAILABLE else "mean_std_comparison"
            ),
            timestamp=datetime.now(),
            details={
                "method": (
                    "wasserstein_distance" if SCIPY_AVAILABLE else "mean_std_comparison"
                ),
                "drift_scores": drift_scores,
            },
        )


class ModelDriftDetector(DriftDetector):
    """Model-based drift detection using prediction confidence"""

    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold

    def detect_drift(
        self, reference_data: np.ndarray, current_data: np.ndarray, model=None, **kwargs
    ) -> DriftDetectionResult:
        """Detect drift using model prediction confidence"""

        if model is None:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.LOW,
                drift_score=0.0,
                threshold=self.confidence_threshold,
                feature_drifts={},
                detection_method="model_confidence",
                timestamp=datetime.now(),
                details={"error": "Model not provided"},
            )

        try:
            # Get prediction confidences
            if hasattr(model, "predict_proba"):
                ref_proba = model.predict_proba(reference_data)
                cur_proba = model.predict_proba(current_data)

                # Calculate average confidence (max probability)
                ref_confidence = np.mean(np.max(ref_proba, axis=1))
                cur_confidence = np.mean(np.max(cur_proba, axis=1))
            else:
                # Fallback: use prediction variance as confidence proxy
                ref_pred = model.predict(reference_data)
                cur_pred = model.predict(current_data)

                ref_confidence = 1.0 - np.std(ref_pred)
                cur_confidence = 1.0 - np.std(cur_pred)

            # Calculate drift score as confidence difference
            confidence_diff = abs(ref_confidence - cur_confidence)
            drift_score = confidence_diff

            # Check if confidence dropped significantly
            confidence_drop = ref_confidence - cur_confidence
            drift_detected = (confidence_drop > (1 - self.confidence_threshold)) or (
                confidence_diff > 0.2
            )

            # Determine severity
            if confidence_diff > 0.3:
                severity = DriftSeverity.CRITICAL
            elif confidence_diff > 0.2:
                severity = DriftSeverity.HIGH
            elif confidence_diff > 0.1:
                severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.LOW

            return DriftDetectionResult(
                drift_detected=drift_detected,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                drift_score=drift_score,
                threshold=1 - self.confidence_threshold,
                feature_drifts={},
                detection_method="model_confidence",
                timestamp=datetime.now(),
                details={
                    "reference_confidence": ref_confidence,
                    "current_confidence": cur_confidence,
                    "confidence_difference": confidence_diff,
                    "confidence_drop": confidence_drop,
                },
            )

        except Exception as e:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=DriftSeverity.LOW,
                drift_score=0.0,
                threshold=self.confidence_threshold,
                feature_drifts={},
                detection_method="model_confidence",
                timestamp=datetime.now(),
                details={"error": str(e)},
            )


class PerformanceMonitor:
    """Monitor model performance and detect degradation"""

    def __init__(
        self, baseline_metrics: PerformanceMetrics, degradation_threshold: float = 0.05
    ):
        self.baseline_metrics = baseline_metrics
        self.degradation_threshold = degradation_threshold
        self.performance_history = []

    def evaluate_performance(
        self, model, test_data: np.ndarray, test_labels: np.ndarray
    ) -> PerformanceMetrics:
        """Evaluate current model performance"""

        try:
            predictions = model.predict(test_data)

            metrics = PerformanceMetrics(
                model_id=self.baseline_metrics.model_id,
                model_version="current",
                timestamp=datetime.now(),
            )

            if SKLEARN_AVAILABLE:
                # Classification metrics
                try:
                    metrics.accuracy = accuracy_score(test_labels, predictions)
                except:
                    # Regression metrics
                    metrics.mse = mean_squared_error(test_labels, predictions)
                    metrics.mae = np.mean(np.abs(test_labels - predictions))

                    # R² score
                    ss_res = np.sum((test_labels - predictions) ** 2)
                    ss_tot = np.sum((test_labels - np.mean(test_labels)) ** 2)
                    metrics.r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            self.performance_history.append(metrics)
            return metrics

        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return PerformanceMetrics(
                model_id=self.baseline_metrics.model_id, model_version="current"
            )

    def detect_performance_degradation(
        self, current_metrics: PerformanceMetrics
    ) -> bool:
        """Detect if performance has degraded significantly"""

        # Compare primary metric
        if (
            self.baseline_metrics.accuracy is not None
            and current_metrics.accuracy is not None
        ):
            degradation = self.baseline_metrics.accuracy - current_metrics.accuracy
            return degradation > self.degradation_threshold

        elif self.baseline_metrics.mse is not None and current_metrics.mse is not None:
            # For MSE, lower is better, so degradation is increase
            degradation = current_metrics.mse - self.baseline_metrics.mse
            baseline_relative = (
                degradation / self.baseline_metrics.mse
                if self.baseline_metrics.mse != 0
                else 0
            )
            return baseline_relative > self.degradation_threshold

        elif (
            self.baseline_metrics.r2_score is not None
            and current_metrics.r2_score is not None
        ):
            degradation = self.baseline_metrics.r2_score - current_metrics.r2_score
            return degradation > self.degradation_threshold

        return False

    def get_performance_trend(self, window_size: int = 10) -> Dict[str, Any]:
        """Analyze performance trend over recent evaluations"""

        if len(self.performance_history) < 2:
            return {"trend": "insufficient_data"}

        recent_metrics = self.performance_history[-window_size:]

        # Extract primary metric values
        values = []
        for metrics in recent_metrics:
            if metrics.accuracy is not None:
                values.append(metrics.accuracy)
            elif metrics.r2_score is not None:
                values.append(metrics.r2_score)
            elif metrics.mse is not None:
                values.append(
                    -metrics.mse
                )  # Negative for trend analysis (higher is better)

        if not values:
            return {"trend": "no_metrics"}

        # Simple trend analysis
        if len(values) >= 3:
            # Linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]

            if slope > 0.001:
                trend = "improving"
            elif slope < -0.001:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            # Simple comparison
            if values[-1] > values[0]:
                trend = "improving"
            elif values[-1] < values[0]:
                trend = "degrading"
            else:
                trend = "stable"

        return {
            "trend": trend,
            "slope": slope if len(values) >= 3 else None,
            "recent_values": values,
            "mean_recent": np.mean(values),
            "std_recent": np.std(values),
            "num_evaluations": len(values),
        }


class RetrainingOrchestrator:
    """Orchestrate automated retraining pipeline"""

    def __init__(self, model_registry, training_config: Dict[str, Any]):
        self.model_registry = model_registry
        self.training_config = training_config
        self.drift_detectors = {}
        self.performance_monitors = {}
        self.active_jobs = {}
        self.trigger_queue = []

        # Default drift detectors
        self.drift_detectors["statistical"] = StatisticalDriftDetector()
        self.drift_detectors["distribution"] = DistributionDriftDetector()
        self.drift_detectors["model_based"] = ModelDriftDetector()

    def register_model_for_monitoring(
        self,
        model_id: str,
        reference_data: np.ndarray,
        baseline_metrics: PerformanceMetrics,
    ):
        """Register model for monitoring"""

        # Store reference data for drift detection
        reference_path = Path(f"data/monitoring/{model_id}_reference.pkl")
        reference_path.parent.mkdir(parents=True, exist_ok=True)

        with open(reference_path, "wb") as f:
            pickle.dump(reference_data, f)

        # Create performance monitor
        self.performance_monitors[model_id] = PerformanceMonitor(
            baseline_metrics,
            degradation_threshold=self.training_config.get(
                "degradation_threshold", 0.05
            ),
        )

        logger.info(f"Registered model {model_id} for monitoring")

    def check_for_drift(
        self, model_id: str, current_data: np.ndarray, model=None
    ) -> List[DriftDetectionResult]:
        """Check for drift using all available detectors"""

        results = []

        # Load reference data
        reference_path = Path(f"data/monitoring/{model_id}_reference.pkl")
        if not reference_path.exists():
            logger.warning(f"No reference data found for model {model_id}")
            return results

        with open(reference_path, "rb") as f:
            reference_data = pickle.load(f)

        # Run all drift detectors
        for detector_name, detector in self.drift_detectors.items():
            try:
                if detector_name == "model_based":
                    result = detector.detect_drift(
                        reference_data, current_data, model=model
                    )
                else:
                    result = detector.detect_drift(reference_data, current_data)

                results.append(result)

                # Create trigger if significant drift detected
                if result.drift_detected and result.severity in [
                    DriftSeverity.HIGH,
                    DriftSeverity.CRITICAL,
                ]:
                    trigger = RetrainingTrigger(
                        trigger_id=f"drift_{model_id}_{int(time.time())}",
                        trigger_type=TriggerType.DRIFT_DETECTED,
                        model_id=model_id,
                        severity=result.severity,
                        reason=f"Drift detected by {detector_name}: {result.drift_score:.4f}",
                        metadata={
                            "detector": detector_name,
                            "drift_score": result.drift_score,
                            "drift_type": result.drift_type.value,
                        },
                        timestamp=datetime.now(),
                    )
                    self.trigger_queue.append(trigger)

            except Exception as e:
                logger.error(f"Drift detection failed for {detector_name}: {e}")

        return results

    def check_performance_degradation(
        self, model_id: str, model, test_data: np.ndarray, test_labels: np.ndarray
    ) -> bool:
        """Check for performance degradation"""

        if model_id not in self.performance_monitors:
            logger.warning(f"No performance monitor for model {model_id}")
            return False

        monitor = self.performance_monitors[model_id]

        # Evaluate current performance
        current_metrics = monitor.evaluate_performance(model, test_data, test_labels)

        # Check for degradation
        degraded = monitor.detect_performance_degradation(current_metrics)

        if degraded:
            # Determine severity based on degradation magnitude
            if (
                monitor.baseline_metrics.accuracy is not None
                and current_metrics.accuracy is not None
            ):
                degradation = (
                    monitor.baseline_metrics.accuracy - current_metrics.accuracy
                )
                if degradation > 0.1:
                    severity = DriftSeverity.CRITICAL
                elif degradation > 0.05:
                    severity = DriftSeverity.HIGH
                else:
                    severity = DriftSeverity.MEDIUM
            else:
                severity = DriftSeverity.MEDIUM

            # Create retraining trigger
            trigger = RetrainingTrigger(
                trigger_id=f"perf_{model_id}_{int(time.time())}",
                trigger_type=TriggerType.PERFORMANCE_DEGRADATION,
                model_id=model_id,
                severity=severity,
                reason=f"Performance degradation detected",
                metadata={
                    "current_metrics": asdict(current_metrics),
                    "baseline_metrics": asdict(monitor.baseline_metrics),
                },
                timestamp=datetime.now(),
            )
            self.trigger_queue.append(trigger)

        return degraded

    async def process_retraining_triggers(self):
        """Process pending retraining triggers"""

        while self.trigger_queue:
            trigger = self.trigger_queue.pop(0)

            if not trigger.acknowledged:
                # Auto-acknowledge high and critical severity triggers
                if trigger.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    trigger.acknowledged = True
                    await self.start_retraining_job(trigger)
                else:
                    # Queue for manual review
                    logger.info(
                        f"Retraining trigger queued for review: {trigger.trigger_id}"
                    )

    async def start_retraining_job(self, trigger: RetrainingTrigger) -> str:
        """Start automated retraining job"""

        job_id = f"job_{trigger.model_id}_{int(time.time())}"

        # Determine retraining strategy
        if trigger.severity == DriftSeverity.CRITICAL:
            strategy = RetrainingStrategy.FULL_RETRAIN
        elif trigger.trigger_type == TriggerType.DRIFT_DETECTED:
            strategy = RetrainingStrategy.INCREMENTAL
        else:
            strategy = RetrainingStrategy.FULL_RETRAIN

        job = RetrainingJob(
            job_id=job_id,
            model_id=trigger.model_id,
            trigger_id=trigger.trigger_id,
            strategy=strategy,
            status=PipelineStatus.RETRAINING,
            start_time=datetime.now(),
        )

        self.active_jobs[job_id] = job

        # Start retraining in background
        asyncio.create_task(self._execute_retraining_job(job))

        logger.info(f"Started retraining job {job_id} for model {trigger.model_id}")
        return job_id

    async def _execute_retraining_job(self, job: RetrainingJob):
        """Execute retraining job"""

        try:
            job.logs.append(f"Starting retraining with strategy: {job.strategy.value}")
            job.progress = 0.1

            # Simulate retraining process
            await asyncio.sleep(1)  # Data preparation
            job.logs.append("Data preparation completed")
            job.progress = 0.3

            await asyncio.sleep(2)  # Model training
            job.logs.append("Model training completed")
            job.progress = 0.7

            await asyncio.sleep(1)  # Validation
            job.logs.append("Model validation completed")
            job.progress = 0.9

            # Create dummy new model metrics
            if job.strategy == RetrainingStrategy.FULL_RETRAIN:
                accuracy_improvement = 0.02
            else:
                accuracy_improvement = 0.01

            # Get baseline metrics
            if job.model_id in self.performance_monitors:
                baseline = self.performance_monitors[job.model_id].baseline_metrics
                new_accuracy = (baseline.accuracy or 0.8) + accuracy_improvement
            else:
                new_accuracy = 0.85

            job.new_model_metrics = PerformanceMetrics(
                model_id=job.model_id,
                model_version="retrained",
                accuracy=new_accuracy,
                timestamp=datetime.now(),
            )

            job.status = PipelineStatus.VALIDATING
            await asyncio.sleep(1)  # Validation

            job.status = PipelineStatus.DEPLOYING
            await asyncio.sleep(1)  # Deployment

            job.logs.append("Retraining completed successfully")
            job.progress = 1.0
            job.end_time = datetime.now()
            job.status = PipelineStatus.IDLE

            logger.info(f"Retraining job {job.job_id} completed successfully")

        except Exception as e:
            job.status = PipelineStatus.FAILED
            job.error_message = str(e)
            job.end_time = datetime.now()
            job.logs.append(f"Retraining failed: {e}")

            logger.error(f"Retraining job {job.job_id} failed: {e}")

    def get_job_status(self, job_id: str) -> Optional[RetrainingJob]:
        """Get status of retraining job"""
        return self.active_jobs.get(job_id)

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get monitoring summary for all models"""

        summary = {
            "monitored_models": list(self.performance_monitors.keys()),
            "active_jobs": len(
                [
                    j
                    for j in self.active_jobs.values()
                    if j.status != PipelineStatus.IDLE
                ]
            ),
            "pending_triggers": len(self.trigger_queue),
            "recent_triggers": [],
            "job_history": [],
        }

        # Recent triggers
        for trigger in self.trigger_queue[-5:]:
            summary["recent_triggers"].append(
                {
                    "trigger_id": trigger.trigger_id,
                    "type": trigger.trigger_type.value,
                    "model_id": trigger.model_id,
                    "severity": trigger.severity.value,
                    "timestamp": trigger.timestamp.isoformat(),
                }
            )

        # Job history
        for job in list(self.active_jobs.values())[-5:]:
            summary["job_history"].append(
                {
                    "job_id": job.job_id,
                    "model_id": job.model_id,
                    "status": job.status.value,
                    "strategy": job.strategy.value,
                    "progress": job.progress,
                    "start_time": job.start_time.isoformat(),
                }
            )

        return summary


class AutomatedRetrainingPipeline:
    """Main automated retraining pipeline"""

    def __init__(self, model_registry, config: Dict[str, Any]):
        self.model_registry = model_registry
        self.config = config
        self.orchestrator = RetrainingOrchestrator(model_registry, config)
        self.scheduler_thread = None
        self.running = False

        # Database for persistent storage
        self.db_path = Path(config.get("db_path", "data/retraining_pipeline.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        asyncio.create_task(self._initialize_database())

    async def _initialize_database(self):
        """Initialize database for pipeline data"""
        async with aiosqlite.connect(self.db_path) as db:
            # Drift detection results table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS drift_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    drift_detected BOOLEAN,
                    drift_type TEXT,
                    severity TEXT,
                    drift_score REAL,
                    detection_method TEXT,
                    timestamp TEXT,
                    details TEXT
                )
            """
            )

            # Performance metrics table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT,
                    model_version TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    mse REAL,
                    mae REAL,
                    r2_score REAL,
                    timestamp TEXT,
                    custom_metrics TEXT
                )
            """
            )

            # Retraining triggers table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    trigger_type TEXT,
                    model_id TEXT,
                    severity TEXT,
                    reason TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    acknowledged BOOLEAN
                )
            """
            )

            # Retraining jobs table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS retraining_jobs (
                    job_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    trigger_id TEXT,
                    strategy TEXT,
                    status TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    progress REAL,
                    logs TEXT,
                    error_message TEXT,
                    new_model_metrics TEXT
                )
            """
            )

            await db.commit()

    async def register_model(
        self,
        model_id: str,
        reference_data: np.ndarray,
        baseline_metrics: PerformanceMetrics,
    ):
        """Register model for automated monitoring"""
        self.orchestrator.register_model_for_monitoring(
            model_id, reference_data, baseline_metrics
        )

        # Store in database
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO performance_metrics 
                (model_id, model_version, accuracy, precision, recall, f1_score, 
                 mse, mae, r2_score, timestamp, custom_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    baseline_metrics.model_id,
                    baseline_metrics.model_version,
                    baseline_metrics.accuracy,
                    baseline_metrics.precision,
                    baseline_metrics.recall,
                    baseline_metrics.f1_score,
                    baseline_metrics.mse,
                    baseline_metrics.mae,
                    baseline_metrics.r2_score,
                    baseline_metrics.timestamp.isoformat(),
                    json.dumps(baseline_metrics.custom_metrics),
                ),
            )
            await db.commit()

    async def check_model_health(
        self,
        model_id: str,
        model,
        current_data: np.ndarray,
        test_data: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Comprehensive model health check"""

        health_report = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "drift_results": [],
            "performance_degraded": False,
            "triggers_created": 0,
            "overall_status": "healthy",
        }

        # Check for drift
        drift_results = self.orchestrator.check_for_drift(model_id, current_data, model)
        health_report["drift_results"] = [asdict(result) for result in drift_results]

        # Store drift results
        for result in drift_results:
            await self._store_drift_result(result)

        # Check performance if test data provided
        if test_data is not None and test_labels is not None:
            degraded = self.orchestrator.check_performance_degradation(
                model_id, model, test_data, test_labels
            )
            health_report["performance_degraded"] = degraded

            if degraded:
                health_report["overall_status"] = "degraded"

        # Check for critical drift
        critical_drift = any(
            r.severity == DriftSeverity.CRITICAL for r in drift_results
        )
        if critical_drift:
            health_report["overall_status"] = "critical"

        # Count new triggers
        health_report["triggers_created"] = len(self.orchestrator.trigger_queue)

        # Process triggers
        await self.orchestrator.process_retraining_triggers()

        return health_report

    async def _store_drift_result(self, result: DriftDetectionResult):
        """Store drift detection result in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO drift_results 
                (model_id, drift_detected, drift_type, severity, drift_score, 
                 detection_method, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "unknown",  # Model ID would be passed from caller
                    result.drift_detected,
                    result.drift_type.value,
                    result.severity.value,
                    result.drift_score,
                    result.detection_method,
                    result.timestamp.isoformat(),
                    json.dumps(result.details),
                ),
            )
            await db.commit()

    def start_monitoring(self, check_interval_minutes: int = 60):
        """Start automated monitoring"""
        if self.running:
            logger.warning("Monitoring already running")
            return

        self.running = True

        # Schedule periodic checks
        schedule.every(check_interval_minutes).minutes.do(self._scheduled_check)

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self.scheduler_thread.start()

        logger.info(
            f"Started automated monitoring (check interval: {check_interval_minutes} minutes)"
        )

    def _run_scheduler(self):
        """Run the scheduler in background thread"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _scheduled_check(self):
        """Scheduled health check for all monitored models"""
        # This would be implemented to check all registered models
        logger.info("Running scheduled model health checks")

        # In a real implementation, this would:
        # 1. Get latest data for each monitored model
        # 2. Run health checks
        # 3. Process any triggers

    def stop_monitoring(self):
        """Stop automated monitoring"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)

        logger.info("Stopped automated monitoring")

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status"""
        return {
            "running": self.running,
            "monitoring_summary": self.orchestrator.get_monitoring_summary(),
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }


# Demo and testing functions
class DummyModel:
    """Dummy model for testing pipeline"""

    def __init__(self, accuracy: float = 0.85, stability: float = 0.95):
        self.accuracy = accuracy
        self.stability = stability  # How stable predictions are
        self.call_count = 0

    def predict(self, data):
        """Make predictions with optional degradation over time"""
        self.call_count += 1

        # Simulate gradual degradation
        degradation_factor = max(0.5, 1.0 - (self.call_count * 0.001))
        current_accuracy = self.accuracy * degradation_factor * self.stability

        if hasattr(data, "__len__"):
            predictions = []
            for _ in range(len(data)):
                pred = 1 if np.random.random() < current_accuracy else 0
                predictions.append(pred)
            return np.array(predictions)
        else:
            return 1 if np.random.random() < current_accuracy else 0

    def predict_proba(self, data):
        """Get prediction probabilities"""
        predictions = self.predict(data)
        probabilities = []

        confidence = self.stability * (1.0 - (self.call_count * 0.0005))

        for pred in predictions:
            if pred == 1:
                prob = [1 - confidence, confidence]
            else:
                prob = [confidence, 1 - confidence]
            probabilities.append(prob)

        return np.array(probabilities)


async def demo_retraining_pipeline():
    """Demonstrate automated retraining pipeline"""

    print("Automated Retraining Pipeline Demo")

    # Configuration
    config = {
        "degradation_threshold": 0.05,
        "drift_threshold": 0.1,
        "db_path": "data/retraining_demo.db",
    }

    # Initialize pipeline (mock model registry)
    pipeline = AutomatedRetrainingPipeline(None, config)
    await pipeline._initialize_database()

    # Create dummy model
    model = DummyModel(accuracy=0.85, stability=0.95)

    # Generate reference data
    np.random.seed(42)
    reference_data = np.random.randn(1000, 5)
    reference_labels = np.random.randint(0, 2, 1000)

    # Create baseline metrics
    baseline_metrics = PerformanceMetrics(
        model_id="demo_model",
        model_version="1.0.0",
        accuracy=0.85,
        precision=0.83,
        recall=0.87,
        f1_score=0.85,
    )

    # Register model for monitoring
    await pipeline.register_model("demo_model", reference_data, baseline_metrics)
    print("Registered model for monitoring")

    print("\n1. Testing Normal Data (No Drift)")

    # Generate similar data (no drift)
    normal_data = np.random.randn(500, 5) + 0.1  # Slight variation
    normal_labels = np.random.randint(0, 2, 500)

    health_report = await pipeline.check_model_health(
        "demo_model", model, normal_data, normal_data[:100], normal_labels[:100]
    )

    print(f"Overall Status: {health_report['overall_status']}")
    print(f"Drift Results: {len(health_report['drift_results'])}")
    print(f"Performance Degraded: {health_report['performance_degraded']}")
    print(f"Triggers Created: {health_report['triggers_created']}")

    # Show drift details
    for i, drift_result in enumerate(health_report["drift_results"]):
        print(f"  Drift Detector {i+1}: {drift_result['detection_method']}")
        print(f"    Detected: {drift_result['drift_detected']}")
        print(f"    Score: {drift_result['drift_score']:.4f}")
        print(f"    Severity: {drift_result['severity']}")

    print("\n2. Testing Drifted Data")

    # Generate drifted data
    drifted_data = np.random.randn(500, 5) * 2 + 1  # Different distribution
    drifted_labels = np.random.randint(0, 2, 500)

    health_report = await pipeline.check_model_health(
        "demo_model", model, drifted_data, drifted_data[:100], drifted_labels[:100]
    )

    print(f"Overall Status: {health_report['overall_status']}")
    print(f"Drift Results: {len(health_report['drift_results'])}")
    print(f"Performance Degraded: {health_report['performance_degraded']}")
    print(f"Triggers Created: {health_report['triggers_created']}")

    for i, drift_result in enumerate(health_report["drift_results"]):
        print(f"  Drift Detector {i+1}: {drift_result['detection_method']}")
        print(f"    Detected: {drift_result['drift_detected']}")
        print(f"    Score: {drift_result['drift_score']:.4f}")
        print(f"    Severity: {drift_result['severity']}")

    print("\n3. Simulating Performance Degradation")

    # Use model many times to simulate degradation
    for _ in range(1000):
        _ = model.predict(normal_data[:10])

    health_report = await pipeline.check_model_health(
        "demo_model", model, normal_data, normal_data[:100], normal_labels[:100]
    )

    print(f"Overall Status: {health_report['overall_status']}")
    print(f"Performance Degraded: {health_report['performance_degraded']}")
    print(f"Triggers Created: {health_report['triggers_created']}")

    print("\n4. Pipeline Status and Job Management")

    # Get pipeline status
    status = await pipeline.get_pipeline_status()
    print(f"Pipeline Running: {status['running']}")

    monitoring_summary = status["monitoring_summary"]
    print(f"Monitored Models: {monitoring_summary['monitored_models']}")
    print(f"Active Jobs: {monitoring_summary['active_jobs']}")
    print(f"Pending Triggers: {monitoring_summary['pending_triggers']}")

    # Show recent triggers
    if monitoring_summary["recent_triggers"]:
        print("\nRecent Triggers:")
        for trigger in monitoring_summary["recent_triggers"]:
            print(
                f"  {trigger['trigger_id']}: {trigger['type']} ({trigger['severity']})"
            )

    # Show job history
    if monitoring_summary["job_history"]:
        print("\nJob History:")
        for job in monitoring_summary["job_history"]:
            print(f"  {job['job_id']}: {job['status']} ({job['progress']:.0%})")

    print("\n5. Manual Trigger Creation")

    # Create manual trigger
    manual_trigger = RetrainingTrigger(
        trigger_id="manual_trigger_001",
        trigger_type=TriggerType.MANUAL,
        model_id="demo_model",
        severity=DriftSeverity.MEDIUM,
        reason="Manual retraining request for model improvement",
        metadata={"requested_by": "user", "strategy": "full_retrain"},
        timestamp=datetime.now(),
    )

    pipeline.orchestrator.trigger_queue.append(manual_trigger)
    manual_trigger.acknowledged = True

    # Start retraining job
    job_id = await pipeline.orchestrator.start_retraining_job(manual_trigger)
    print(f"Started manual retraining job: {job_id}")

    # Wait for job to complete
    await asyncio.sleep(6)

    # Check job status
    job = pipeline.orchestrator.get_job_status(job_id)
    if job:
        print(f"Job Status: {job.status.value}")
        print(f"Progress: {job.progress:.0%}")
        print(f"Duration: {(job.end_time - job.start_time).total_seconds():.1f}s")

        if job.new_model_metrics:
            print(f"New Model Accuracy: {job.new_model_metrics.accuracy:.3f}")

        print("Job Logs:")
        for log in job.logs[-3:]:
            print(f"  {log}")

    return pipeline


if __name__ == "__main__":
    asyncio.run(demo_retraining_pipeline())
