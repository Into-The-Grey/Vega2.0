"""
Advanced ML Monitoring System

Provides model explainability, bias detection, performance degradation alerts,
and interpretability dashboards for comprehensive ML model monitoring.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import json
import time
import threading
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Protocol
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager
import aiosqlite
from abc import ABC, abstractmethod
import tempfile

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("Plotting libraries not available - dashboards disabled")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available - some explainability features disabled")

try:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report,
        roc_curve,
        auc,
    )
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import cross_val_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available - some monitoring features disabled")

try:
    import lime
    import lime.lime_text
    import lime.lime_image
    import lime.lime_tabular

    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available - some explainability features disabled")


class ExplainabilityMethod(Enum):
    """Methods for model explainability"""

    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    FEATURE_IMPORTANCE = "feature_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    GRADIENTS = "gradients"


class BiasType(Enum):
    """Types of bias to detect"""

    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    INDIVIDUAL_FAIRNESS = "individual_fairness"
    COUNTERFACTUAL_FAIRNESS = "counterfactual_fairness"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MonitoringMetric(Enum):
    """Metrics to monitor"""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PREDICTION_CONFIDENCE = "prediction_confidence"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    BIAS_SCORE = "bias_score"
    EXPLAINABILITY_SCORE = "explainability_score"


@dataclass
class ExplanationResult:
    """Result of model explanation analysis"""

    method: ExplainabilityMethod
    model_id: str
    sample_id: str
    feature_importance: Dict[str, float]
    explanation_text: str
    confidence_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.feature_importance is None:
            self.feature_importance = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class BiasAssessment:
    """Result of bias assessment"""

    bias_type: BiasType
    model_id: str
    protected_attribute: str
    bias_score: float
    threshold: float
    is_biased: bool
    group_metrics: Dict[str, Dict[str, float]]
    fairness_metrics: Dict[str, float]
    timestamp: datetime
    recommendations: List[str]

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.group_metrics is None:
            self.group_metrics = {}
        if self.fairness_metrics is None:
            self.fairness_metrics = {}
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class PerformanceAlert:
    """Performance degradation alert"""

    alert_id: str
    model_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percentage: float
    severity: AlertSeverity
    message: str
    timestamp: datetime
    acknowledged: bool = False

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class InterpretabilityReport:
    """Comprehensive interpretability report"""

    model_id: str
    report_id: str
    global_explanations: Dict[str, Any]
    local_explanations: List[ExplanationResult]
    bias_assessments: List[BiasAssessment]
    performance_summary: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime

    def __post_init__(self):
        if isinstance(self.generated_at, str):
            self.generated_at = datetime.fromisoformat(self.generated_at)
        if self.global_explanations is None:
            self.global_explanations = {}
        if self.local_explanations is None:
            self.local_explanations = []
        if self.bias_assessments is None:
            self.bias_assessments = []
        if self.performance_summary is None:
            self.performance_summary = {}
        if self.recommendations is None:
            self.recommendations = []


class ExplainabilityEngine:
    """Engine for generating model explanations"""

    def __init__(self):
        self.explainers = {}
        self._initialize_explainers()

    def _initialize_explainers(self):
        """Initialize available explainers"""
        if SHAP_AVAILABLE:
            self.explainers[ExplainabilityMethod.SHAP] = self._shap_explain

        if LIME_AVAILABLE:
            self.explainers[ExplainabilityMethod.LIME] = self._lime_explain

        if SKLEARN_AVAILABLE:
            self.explainers[ExplainabilityMethod.PERMUTATION_IMPORTANCE] = (
                self._permutation_explain
            )
            self.explainers[ExplainabilityMethod.FEATURE_IMPORTANCE] = (
                self._feature_importance_explain
            )

    async def explain_prediction(
        self,
        model,
        sample_data: np.ndarray,
        training_data: np.ndarray = None,
        feature_names: List[str] = None,
        method: ExplainabilityMethod = ExplainabilityMethod.SHAP,
        model_id: str = "unknown",
    ) -> ExplanationResult:
        """Generate explanation for a single prediction"""

        if method not in self.explainers:
            raise ValueError(f"Explainability method {method.value} not available")

        try:
            result = await self.explainers[method](
                model, sample_data, training_data, feature_names, model_id
            )
            return result

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return ExplanationResult(
                method=method,
                model_id=model_id,
                sample_id="error",
                feature_importance={},
                explanation_text=f"Explanation failed: {str(e)}",
                confidence_score=0.0,
                timestamp=datetime.now(),
                metadata={"error": str(e)},
            )

    async def _shap_explain(
        self,
        model,
        sample_data: np.ndarray,
        training_data: np.ndarray,
        feature_names: List[str],
        model_id: str,
    ) -> ExplanationResult:
        """Generate SHAP explanation"""

        try:
            # Choose appropriate SHAP explainer
            if hasattr(model, "predict_proba"):
                # Classification model
                if training_data is not None:
                    explainer = shap.KernelExplainer(
                        model.predict_proba, training_data[:100]
                    )
                else:
                    # Use sample data as background
                    explainer = shap.KernelExplainer(
                        model.predict_proba, sample_data.reshape(1, -1)
                    )
            else:
                # Regression model
                if training_data is not None:
                    explainer = shap.KernelExplainer(model.predict, training_data[:100])
                else:
                    explainer = shap.KernelExplainer(
                        model.predict, sample_data.reshape(1, -1)
                    )

            # Get SHAP values
            shap_values = explainer.shap_values(sample_data.reshape(1, -1))

            # Handle multi-class case
            if isinstance(shap_values, list):
                # Take the class with highest prediction
                pred_proba = model.predict_proba(sample_data.reshape(1, -1))[0]
                class_idx = np.argmax(pred_proba)
                shap_values = shap_values[class_idx][0]
                confidence = pred_proba[class_idx]
            else:
                shap_values = shap_values[0]
                confidence = 0.8  # Default confidence for regression

            # Create feature importance dictionary
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(shap_values))]

            feature_importance = {
                name: float(value) for name, value in zip(feature_names, shap_values)
            }

            # Generate explanation text
            top_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            explanation_parts = []

            for feature, importance in top_features:
                direction = "increases" if importance > 0 else "decreases"
                explanation_parts.append(
                    f"{feature} {direction} prediction by {abs(importance):.3f}"
                )

            explanation_text = "Top contributing features: " + "; ".join(
                explanation_parts
            )

            return ExplanationResult(
                method=ExplainabilityMethod.SHAP,
                model_id=model_id,
                sample_id=f"sample_{int(time.time())}",
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence_score=confidence,
                timestamp=datetime.now(),
                metadata={
                    "shap_values_sum": float(np.sum(np.abs(shap_values))),
                    "explainer_type": type(explainer).__name__,
                },
            )

        except Exception as e:
            raise Exception(f"SHAP explanation failed: {e}")

    async def _lime_explain(
        self,
        model,
        sample_data: np.ndarray,
        training_data: np.ndarray,
        feature_names: List[str],
        model_id: str,
    ) -> ExplanationResult:
        """Generate LIME explanation"""

        try:
            # Create LIME explainer for tabular data
            if training_data is not None:
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data,
                    feature_names=feature_names
                    or [f"feature_{i}" for i in range(sample_data.shape[0])],
                    class_names=(
                        ["class_0", "class_1"]
                        if hasattr(model, "predict_proba")
                        else None
                    ),
                    mode=(
                        "classification"
                        if hasattr(model, "predict_proba")
                        else "regression"
                    ),
                )
            else:
                # Create dummy training data
                dummy_data = np.random.randn(100, len(sample_data))
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    dummy_data,
                    feature_names=feature_names
                    or [f"feature_{i}" for i in range(sample_data.shape[0])],
                    mode=(
                        "classification"
                        if hasattr(model, "predict_proba")
                        else "regression"
                    ),
                )

            # Generate explanation
            if hasattr(model, "predict_proba"):
                explanation = explainer.explain_instance(
                    sample_data, model.predict_proba, num_features=len(sample_data)
                )
                # Get prediction confidence
                proba = model.predict_proba(sample_data.reshape(1, -1))[0]
                confidence = np.max(proba)
            else:
                explanation = explainer.explain_instance(
                    sample_data, model.predict, num_features=len(sample_data)
                )
                confidence = 0.8  # Default for regression

            # Extract feature importance
            feature_importance = {}
            for feature_idx, importance in explanation.as_list():
                if feature_names and feature_idx < len(feature_names):
                    feature_name = feature_names[feature_idx]
                else:
                    feature_name = f"feature_{feature_idx}"
                feature_importance[feature_name] = importance

            # Generate explanation text
            top_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            explanation_parts = []

            for feature, importance in top_features:
                direction = "supports" if importance > 0 else "opposes"
                explanation_parts.append(
                    f"{feature} {direction} prediction (weight: {importance:.3f})"
                )

            explanation_text = "LIME analysis: " + "; ".join(explanation_parts)

            return ExplanationResult(
                method=ExplainabilityMethod.LIME,
                model_id=model_id,
                sample_id=f"sample_{int(time.time())}",
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence_score=confidence,
                timestamp=datetime.now(),
                metadata={
                    "explanation_score": explanation.score,
                    "local_pred": (
                        explanation.local_pred[0] if explanation.local_pred else None
                    ),
                },
            )

        except Exception as e:
            raise Exception(f"LIME explanation failed: {e}")

    async def _permutation_explain(
        self,
        model,
        sample_data: np.ndarray,
        training_data: np.ndarray,
        feature_names: List[str],
        model_id: str,
    ) -> ExplanationResult:
        """Generate permutation importance explanation"""

        try:
            if training_data is None:
                raise ValueError("Training data required for permutation importance")

            # Use subset of training data for efficiency
            X_test = training_data[:200]

            # Generate dummy labels for permutation importance
            if hasattr(model, "predict_proba"):
                y_test = model.predict(X_test)
                # Convert to binary labels
                y_test = (
                    (y_test > 0.5).astype(int)
                    if len(y_test.shape) == 1
                    else np.argmax(y_test, axis=1)
                )
            else:
                y_test = model.predict(X_test)

            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats=5, random_state=42
            )

            # Create feature importance dictionary
            if feature_names is None:
                feature_names = [
                    f"feature_{i}" for i in range(len(perm_importance.importances_mean))
                ]

            feature_importance = {
                name: float(importance)
                for name, importance in zip(
                    feature_names, perm_importance.importances_mean
                )
            }

            # Calculate confidence based on prediction
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample_data.reshape(1, -1))[0]
                confidence = np.max(proba)
            else:
                confidence = 0.8

            # Generate explanation text
            top_features = sorted(
                feature_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            explanation_parts = []

            for feature, importance in top_features:
                explanation_parts.append(f"{feature}: {importance:.3f}")

            explanation_text = "Permutation importance ranking: " + "; ".join(
                explanation_parts
            )

            return ExplanationResult(
                method=ExplainabilityMethod.PERMUTATION_IMPORTANCE,
                model_id=model_id,
                sample_id=f"sample_{int(time.time())}",
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence_score=confidence,
                timestamp=datetime.now(),
                metadata={
                    "importance_std": [
                        float(x) for x in perm_importance.importances_std
                    ],
                    "n_repeats": 5,
                },
            )

        except Exception as e:
            raise Exception(f"Permutation importance failed: {e}")

    async def _feature_importance_explain(
        self,
        model,
        sample_data: np.ndarray,
        training_data: np.ndarray,
        feature_names: List[str],
        model_id: str,
    ) -> ExplanationResult:
        """Generate feature importance explanation (for tree-based models)"""

        try:
            # Check if model has feature_importances_ attribute
            if not hasattr(model, "feature_importances_"):
                raise ValueError("Model does not have feature_importances_ attribute")

            importances = model.feature_importances_

            # Create feature importance dictionary
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            feature_importance = {
                name: float(importance)
                for name, importance in zip(feature_names, importances)
            }

            # Calculate confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample_data.reshape(1, -1))[0]
                confidence = np.max(proba)
            else:
                confidence = 0.8

            # Generate explanation text
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:5]
            explanation_parts = []

            for feature, importance in top_features:
                explanation_parts.append(f"{feature}: {importance:.3f}")

            explanation_text = "Model feature importance: " + "; ".join(
                explanation_parts
            )

            return ExplanationResult(
                method=ExplainabilityMethod.FEATURE_IMPORTANCE,
                model_id=model_id,
                sample_id=f"sample_{int(time.time())}",
                feature_importance=feature_importance,
                explanation_text=explanation_text,
                confidence_score=confidence,
                timestamp=datetime.now(),
                metadata={
                    "total_importance": float(np.sum(importances)),
                    "max_importance": float(np.max(importances)),
                },
            )

        except Exception as e:
            raise Exception(f"Feature importance explanation failed: {e}")


class BiasDetector:
    """Detect and assess model bias"""

    def __init__(self):
        self.bias_thresholds = {
            BiasType.DEMOGRAPHIC_PARITY: 0.1,
            BiasType.EQUAL_OPPORTUNITY: 0.1,
            BiasType.EQUALIZED_ODDS: 0.1,
        }

    async def assess_bias(
        self,
        model,
        test_data: pd.DataFrame,
        test_labels: np.ndarray,
        protected_attribute: str,
        bias_type: BiasType = BiasType.DEMOGRAPHIC_PARITY,
        model_id: str = "unknown",
    ) -> BiasAssessment:
        """Assess model bias for protected attribute"""

        if protected_attribute not in test_data.columns:
            raise ValueError(
                f"Protected attribute '{protected_attribute}' not found in data"
            )

        # Get model predictions
        X_test = test_data.drop(columns=[protected_attribute])
        predictions = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            prediction_probs = model.predict_proba(X_test)
        else:
            prediction_probs = None

        # Get unique groups
        groups = test_data[protected_attribute].unique()

        # Calculate metrics for each group
        group_metrics = {}
        for group in groups:
            group_mask = test_data[protected_attribute] == group
            group_pred = predictions[group_mask]
            group_true = test_labels[group_mask]

            if SKLEARN_AVAILABLE:
                metrics = {
                    "accuracy": accuracy_score(group_true, group_pred),
                    "precision": precision_score(
                        group_true, group_pred, average="binary", zero_division=0
                    ),
                    "recall": recall_score(
                        group_true, group_pred, average="binary", zero_division=0
                    ),
                    "f1_score": f1_score(
                        group_true, group_pred, average="binary", zero_division=0
                    ),
                    "positive_rate": np.mean(group_pred),
                    "sample_size": len(group_pred),
                }
            else:
                metrics = {
                    "accuracy": np.mean(group_true == group_pred),
                    "positive_rate": np.mean(group_pred),
                    "sample_size": len(group_pred),
                }

            group_metrics[str(group)] = metrics

        # Calculate fairness metrics based on bias type
        fairness_metrics = {}
        bias_score = 0.0

        if bias_type == BiasType.DEMOGRAPHIC_PARITY:
            # Demographic parity: positive prediction rates should be similar
            positive_rates = [
                metrics["positive_rate"] for metrics in group_metrics.values()
            ]
            bias_score = max(positive_rates) - min(positive_rates)
            fairness_metrics["demographic_parity_difference"] = bias_score

        elif bias_type == BiasType.EQUAL_OPPORTUNITY:
            # Equal opportunity: true positive rates should be similar
            tpr_values = []
            for group in groups:
                group_mask = test_data[protected_attribute] == group
                group_pred = predictions[group_mask]
                group_true = test_labels[group_mask]

                # True positive rate
                if np.sum(group_true == 1) > 0:
                    tpr = np.sum((group_pred == 1) & (group_true == 1)) / np.sum(
                        group_true == 1
                    )
                else:
                    tpr = 0.0
                tpr_values.append(tpr)

            bias_score = max(tpr_values) - min(tpr_values)
            fairness_metrics["equal_opportunity_difference"] = bias_score

        elif bias_type == BiasType.EQUALIZED_ODDS:
            # Equalized odds: both TPR and FPR should be similar
            tpr_values = []
            fpr_values = []

            for group in groups:
                group_mask = test_data[protected_attribute] == group
                group_pred = predictions[group_mask]
                group_true = test_labels[group_mask]

                # True positive rate
                if np.sum(group_true == 1) > 0:
                    tpr = np.sum((group_pred == 1) & (group_true == 1)) / np.sum(
                        group_true == 1
                    )
                else:
                    tpr = 0.0
                tpr_values.append(tpr)

                # False positive rate
                if np.sum(group_true == 0) > 0:
                    fpr = np.sum((group_pred == 1) & (group_true == 0)) / np.sum(
                        group_true == 0
                    )
                else:
                    fpr = 0.0
                fpr_values.append(fpr)

            tpr_diff = max(tpr_values) - min(tpr_values)
            fpr_diff = max(fpr_values) - min(fpr_values)
            bias_score = max(tpr_diff, fpr_diff)

            fairness_metrics["tpr_difference"] = tpr_diff
            fairness_metrics["fpr_difference"] = fpr_diff
            fairness_metrics["equalized_odds_difference"] = bias_score

        # Determine if model is biased
        threshold = self.bias_thresholds.get(bias_type, 0.1)
        is_biased = bias_score > threshold

        # Generate recommendations
        recommendations = []
        if is_biased:
            recommendations.append(f"Model shows significant bias in {bias_type.value}")
            recommendations.append("Consider rebalancing training data")
            recommendations.append("Apply fairness constraints during training")
            recommendations.append("Use bias mitigation techniques")
        else:
            recommendations.append(
                f"Model shows acceptable fairness for {bias_type.value}"
            )

        return BiasAssessment(
            bias_type=bias_type,
            model_id=model_id,
            protected_attribute=protected_attribute,
            bias_score=bias_score,
            threshold=threshold,
            is_biased=is_biased,
            group_metrics=group_metrics,
            fairness_metrics=fairness_metrics,
            timestamp=datetime.now(),
            recommendations=recommendations,
        )


class PerformanceAlertSystem:
    """System for monitoring performance and generating alerts"""

    def __init__(self, db_path: str = "data/monitoring_alerts.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.alert_thresholds = {
            MonitoringMetric.ACCURACY: {"warning": 0.05, "critical": 0.1},
            MonitoringMetric.PRECISION: {"warning": 0.05, "critical": 0.1},
            MonitoringMetric.RECALL: {"warning": 0.05, "critical": 0.1},
            MonitoringMetric.F1_SCORE: {"warning": 0.05, "critical": 0.1},
            MonitoringMetric.PREDICTION_CONFIDENCE: {"warning": 0.1, "critical": 0.2},
        }

        self.baseline_metrics = {}
        self.active_alerts = {}

        asyncio.create_task(self._initialize_database())

    async def _initialize_database(self):
        """Initialize alerts database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    alert_id TEXT PRIMARY KEY,
                    model_id TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    baseline_value REAL,
                    degradation_percentage REAL,
                    severity TEXT,
                    message TEXT,
                    timestamp TEXT,
                    acknowledged BOOLEAN
                )
            """
            )

            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS baseline_metrics (
                    model_id TEXT PRIMARY KEY,
                    metrics TEXT,
                    updated_at TEXT
                )
            """
            )

            await db.commit()

    def set_baseline_metrics(self, model_id: str, metrics: Dict[str, float]):
        """Set baseline metrics for a model"""
        self.baseline_metrics[model_id] = {
            "metrics": metrics,
            "updated_at": datetime.now(),
        }

    async def check_performance_degradation(
        self, model_id: str, current_metrics: Dict[str, float]
    ) -> List[PerformanceAlert]:
        """Check for performance degradation and generate alerts"""

        alerts = []

        if model_id not in self.baseline_metrics:
            logger.warning(f"No baseline metrics found for model {model_id}")
            return alerts

        baseline = self.baseline_metrics[model_id]["metrics"]

        for metric_name, current_value in current_metrics.items():
            if metric_name not in baseline:
                continue

            baseline_value = baseline[metric_name]

            # Calculate degradation percentage
            if baseline_value != 0:
                degradation_pct = (baseline_value - current_value) / baseline_value
            else:
                degradation_pct = 0.0

            # Check thresholds
            metric_enum = None
            try:
                metric_enum = MonitoringMetric(metric_name)
            except ValueError:
                continue

            if metric_enum not in self.alert_thresholds:
                continue

            thresholds = self.alert_thresholds[metric_enum]
            severity = None

            if abs(degradation_pct) >= thresholds["critical"]:
                severity = AlertSeverity.CRITICAL
            elif abs(degradation_pct) >= thresholds["warning"]:
                severity = AlertSeverity.WARNING

            if severity:
                alert_id = f"alert_{model_id}_{metric_name}_{int(time.time())}"

                message = (
                    f"Performance degradation detected in {metric_name}: "
                    f"{current_value:.3f} vs baseline {baseline_value:.3f} "
                    f"({degradation_pct:.1%} change)"
                )

                alert = PerformanceAlert(
                    alert_id=alert_id,
                    model_id=model_id,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    degradation_percentage=degradation_pct,
                    severity=severity,
                    message=message,
                    timestamp=datetime.now(),
                )

                alerts.append(alert)
                self.active_alerts[alert_id] = alert

                # Store in database
                await self._store_alert(alert)

        return alerts

    async def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO performance_alerts 
                (alert_id, model_id, metric_name, current_value, baseline_value,
                 degradation_percentage, severity, message, timestamp, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.alert_id,
                    alert.model_id,
                    alert.metric_name,
                    alert.current_value,
                    alert.baseline_value,
                    alert.degradation_percentage,
                    alert.severity.value,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.acknowledged,
                ),
            )
            await db.commit()

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True

            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE performance_alerts SET acknowledged = ? WHERE alert_id = ?",
                    (True, alert_id),
                )
                await db.commit()

            return True
        return False

    async def get_active_alerts(
        self, model_id: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """Get active (unacknowledged) alerts"""
        query = "SELECT * FROM performance_alerts WHERE acknowledged = ?"
        params = [False]

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)

        query += " ORDER BY timestamp DESC"

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

            alerts = []
            for row in rows:
                columns = [description[0] for description in cursor.description]
                data = dict(zip(columns, row))
                data["severity"] = AlertSeverity(data["severity"])
                alerts.append(PerformanceAlert(**data))

            return alerts


class MLMonitoringDashboard:
    """Dashboard for ML monitoring and interpretability"""

    def __init__(
        self,
        explainability_engine: ExplainabilityEngine,
        bias_detector: BiasDetector,
        alert_system: PerformanceAlertSystem,
    ):
        self.explainability_engine = explainability_engine
        self.bias_detector = bias_detector
        self.alert_system = alert_system

    async def generate_model_report(
        self,
        model_id: str,
        model,
        test_data: pd.DataFrame,
        test_labels: np.ndarray,
        training_data: np.ndarray = None,
        feature_names: List[str] = None,
        protected_attributes: List[str] = None,
    ) -> InterpretabilityReport:
        """Generate comprehensive interpretability report"""

        report_id = f"report_{model_id}_{int(time.time())}"

        # Global explanations
        global_explanations = {}

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            if feature_names is None:
                feature_names = [
                    f"feature_{i}" for i in range(len(model.feature_importances_))
                ]

            global_explanations["feature_importance"] = {
                name: float(importance)
                for name, importance in zip(feature_names, model.feature_importances_)
            }

        # Local explanations for sample predictions
        local_explanations = []
        sample_indices = np.random.choice(
            len(test_data), min(10, len(test_data)), replace=False
        )

        for idx in sample_indices:
            sample = test_data.iloc[idx].values

            # Try different explanation methods
            for method in [ExplainabilityMethod.SHAP, ExplainabilityMethod.LIME]:
                try:
                    explanation = await self.explainability_engine.explain_prediction(
                        model, sample, training_data, feature_names, method, model_id
                    )
                    local_explanations.append(explanation)
                    break  # Use first successful method
                except:
                    continue

        # Bias assessments
        bias_assessments = []
        if protected_attributes:
            for attr in protected_attributes:
                if attr in test_data.columns:
                    try:
                        bias_assessment = await self.bias_detector.assess_bias(
                            model,
                            test_data,
                            test_labels,
                            attr,
                            BiasType.DEMOGRAPHIC_PARITY,
                            model_id,
                        )
                        bias_assessments.append(bias_assessment)
                    except Exception as e:
                        logger.warning(f"Bias assessment failed for {attr}: {e}")

        # Performance summary
        X_test = test_data.drop(columns=protected_attributes or [])
        predictions = model.predict(X_test)

        performance_summary = {}
        if SKLEARN_AVAILABLE:
            try:
                performance_summary["accuracy"] = accuracy_score(
                    test_labels, predictions
                )
                performance_summary["precision"] = precision_score(
                    test_labels, predictions, average="weighted", zero_division=0
                )
                performance_summary["recall"] = recall_score(
                    test_labels, predictions, average="weighted", zero_division=0
                )
                performance_summary["f1_score"] = f1_score(
                    test_labels, predictions, average="weighted", zero_division=0
                )
            except:
                # Regression metrics
                performance_summary["mse"] = np.mean((test_labels - predictions) ** 2)
                performance_summary["mae"] = np.mean(np.abs(test_labels - predictions))

        # Generate recommendations
        recommendations = []

        # Model performance recommendations
        if performance_summary.get("accuracy", 0) < 0.8:
            recommendations.append(
                "Consider improving model accuracy through feature engineering or algorithm selection"
            )

        # Bias recommendations
        if any(assessment.is_biased for assessment in bias_assessments):
            recommendations.append(
                "Address detected bias through data rebalancing or fairness constraints"
            )

        # Explainability recommendations
        if not local_explanations:
            recommendations.append("Consider using simpler, more interpretable models")

        if not global_explanations:
            recommendations.append(
                "Use models with built-in feature importance (e.g., tree-based models)"
            )

        return InterpretabilityReport(
            model_id=model_id,
            report_id=report_id,
            global_explanations=global_explanations,
            local_explanations=local_explanations,
            bias_assessments=bias_assessments,
            performance_summary=performance_summary,
            recommendations=recommendations,
            generated_at=datetime.now(),
        )

    def create_visualization_dashboard(
        self, report: InterpretabilityReport
    ) -> Dict[str, Any]:
        """Create visualization dashboard data"""

        if not PLOTTING_AVAILABLE:
            return {"error": "Plotting libraries not available"}

        dashboard_data = {
            "model_id": report.model_id,
            "report_id": report.report_id,
            "visualizations": {},
        }

        # Feature importance visualization
        if "feature_importance" in report.global_explanations:
            importance_data = report.global_explanations["feature_importance"]
            sorted_features = sorted(
                importance_data.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]

            features, importances = zip(*sorted_features)

            dashboard_data["visualizations"]["feature_importance"] = {
                "type": "bar",
                "data": {"features": list(features), "importances": list(importances)},
                "title": "Top 10 Feature Importances",
            }

        # Performance metrics visualization
        if report.performance_summary:
            metrics = list(report.performance_summary.keys())
            values = list(report.performance_summary.values())

            dashboard_data["visualizations"]["performance_metrics"] = {
                "type": "radar",
                "data": {"metrics": metrics, "values": values},
                "title": "Model Performance Metrics",
            }

        # Bias assessment visualization
        if report.bias_assessments:
            bias_data = []
            for assessment in report.bias_assessments:
                bias_data.append(
                    {
                        "attribute": assessment.protected_attribute,
                        "bias_score": assessment.bias_score,
                        "threshold": assessment.threshold,
                        "is_biased": assessment.is_biased,
                    }
                )

            dashboard_data["visualizations"]["bias_assessment"] = {
                "type": "scatter",
                "data": bias_data,
                "title": "Bias Assessment Results",
            }

        # Local explanations heatmap
        if report.local_explanations:
            explanation_matrix = []
            feature_names = []

            for explanation in report.local_explanations[:5]:  # Top 5 explanations
                if not feature_names:
                    feature_names = list(explanation.feature_importance.keys())

                importance_values = [
                    explanation.feature_importance.get(feature, 0)
                    for feature in feature_names
                ]
                explanation_matrix.append(importance_values)

            dashboard_data["visualizations"]["local_explanations"] = {
                "type": "heatmap",
                "data": {
                    "matrix": explanation_matrix,
                    "features": feature_names,
                    "samples": [
                        f"Sample {i+1}" for i in range(len(explanation_matrix))
                    ],
                },
                "title": "Local Explanation Heatmap",
            }

        return dashboard_data

    async def get_monitoring_summary(self, model_id: str) -> Dict[str, Any]:
        """Get monitoring summary for a model"""

        # Get active alerts
        alerts = await self.alert_system.get_active_alerts(model_id)

        # Categorize alerts by severity
        alert_summary = {
            "critical": len(
                [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
            ),
            "warning": len([a for a in alerts if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in alerts if a.severity == AlertSeverity.INFO]),
            "total": len(alerts),
        }

        # Recent alerts
        recent_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:5]

        return {
            "model_id": model_id,
            "alert_summary": alert_summary,
            "recent_alerts": [
                {
                    "alert_id": alert.alert_id,
                    "metric_name": alert.metric_name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                }
                for alert in recent_alerts
            ],
            "monitoring_status": (
                "critical"
                if alert_summary["critical"] > 0
                else "warning" if alert_summary["warning"] > 0 else "healthy"
            ),
            "last_updated": datetime.now().isoformat(),
        }


class AdvancedMLMonitoring:
    """Main advanced ML monitoring system"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # Initialize components
        self.explainability_engine = ExplainabilityEngine()
        self.bias_detector = BiasDetector()
        self.alert_system = PerformanceAlertSystem(
            self.config.get("alerts_db_path", "data/monitoring_alerts.db")
        )
        self.dashboard = MLMonitoringDashboard(
            self.explainability_engine, self.bias_detector, self.alert_system
        )

        # Monitoring state
        self.monitored_models = {}
        self.monitoring_active = False

    async def register_model_for_monitoring(
        self,
        model_id: str,
        model,
        baseline_metrics: Dict[str, float],
        training_data: np.ndarray = None,
        feature_names: List[str] = None,
        protected_attributes: List[str] = None,
    ):
        """Register a model for advanced monitoring"""

        self.monitored_models[model_id] = {
            "model": model,
            "training_data": training_data,
            "feature_names": feature_names,
            "protected_attributes": protected_attributes,
            "registered_at": datetime.now(),
        }

        # Set baseline metrics for alerts
        self.alert_system.set_baseline_metrics(model_id, baseline_metrics)

        logger.info(f"Registered model {model_id} for advanced monitoring")

    async def run_comprehensive_monitoring(
        self, model_id: str, test_data: pd.DataFrame, test_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Run comprehensive monitoring for a model"""

        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered for monitoring")

        model_info = self.monitored_models[model_id]
        model = model_info["model"]

        monitoring_results = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "performance_alerts": [],
            "interpretability_report": None,
            "monitoring_summary": {},
            "recommendations": [],
        }

        # Check performance and generate alerts
        X_test = test_data.drop(columns=model_info["protected_attributes"] or [])
        predictions = model.predict(X_test)

        # Calculate current metrics
        current_metrics = {}
        if SKLEARN_AVAILABLE:
            try:
                current_metrics["accuracy"] = accuracy_score(test_labels, predictions)
                current_metrics["precision"] = precision_score(
                    test_labels, predictions, average="weighted", zero_division=0
                )
                current_metrics["recall"] = recall_score(
                    test_labels, predictions, average="weighted", zero_division=0
                )
                current_metrics["f1_score"] = f1_score(
                    test_labels, predictions, average="weighted", zero_division=0
                )
            except:
                # Regression metrics
                current_metrics["mse"] = np.mean((test_labels - predictions) ** 2)
                current_metrics["mae"] = np.mean(np.abs(test_labels - predictions))

        # Check for performance degradation
        alerts = await self.alert_system.check_performance_degradation(
            model_id, current_metrics
        )
        monitoring_results["performance_alerts"] = [asdict(alert) for alert in alerts]

        # Generate interpretability report
        try:
            report = await self.dashboard.generate_model_report(
                model_id,
                model,
                test_data,
                test_labels,
                model_info["training_data"],
                model_info["feature_names"],
                model_info["protected_attributes"],
            )
            monitoring_results["interpretability_report"] = asdict(report)
        except Exception as e:
            logger.error(f"Failed to generate interpretability report: {e}")
            monitoring_results["interpretability_report"] = {"error": str(e)}

        # Get monitoring summary
        monitoring_results["monitoring_summary"] = (
            await self.dashboard.get_monitoring_summary(model_id)
        )

        # Generate overall recommendations
        recommendations = []

        if alerts:
            critical_alerts = [
                a for a in alerts if a.severity == AlertSeverity.CRITICAL
            ]
            if critical_alerts:
                recommendations.append(
                    "URGENT: Critical performance degradation detected - immediate attention required"
                )
            else:
                recommendations.append(
                    "Performance degradation detected - monitor closely"
                )

        if monitoring_results["interpretability_report"] and not isinstance(
            monitoring_results["interpretability_report"], dict
        ):
            report_recommendations = monitoring_results["interpretability_report"].get(
                "recommendations", []
            )
            recommendations.extend(report_recommendations)

        if not recommendations:
            recommendations.append(
                "Model monitoring looks healthy - continue regular monitoring"
            )

        monitoring_results["recommendations"] = recommendations

        return monitoring_results

    async def get_dashboard_data(self, model_id: str) -> Dict[str, Any]:
        """Get dashboard visualization data for a model"""

        if model_id not in self.monitored_models:
            raise ValueError(f"Model {model_id} not registered for monitoring")

        # This would typically load the latest interpretability report
        # For demo purposes, we'll return basic dashboard structure
        return {
            "model_id": model_id,
            "dashboard_available": PLOTTING_AVAILABLE,
            "last_updated": datetime.now().isoformat(),
            "status": "Dashboard data would be available with latest monitoring results",
        }


# Demo and testing functions
class DummyMLModel:
    """Dummy ML model for testing monitoring"""

    def __init__(self, n_features: int = 10, accuracy: float = 0.85):
        self.n_features = n_features
        self.accuracy = accuracy
        self.feature_importances_ = np.random.dirichlet(np.ones(n_features))
        self.call_count = 0

    def predict(self, X):
        """Make predictions"""
        self.call_count += 1

        # Simulate gradual accuracy degradation
        current_accuracy = self.accuracy * (1.0 - self.call_count * 0.001)

        predictions = []
        for i in range(len(X)):
            pred = 1 if np.random.random() < current_accuracy else 0
            predictions.append(pred)

        return np.array(predictions)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        predictions = self.predict(X)
        probabilities = []

        for pred in predictions:
            if pred == 1:
                prob = [0.2, 0.8]
            else:
                prob = [0.8, 0.2]
            probabilities.append(prob)

        return np.array(probabilities)


async def demo_advanced_monitoring():
    """Demonstrate advanced ML monitoring capabilities"""

    print("Advanced ML Monitoring Demo")

    # Initialize monitoring system
    config = {"alerts_db_path": "data/monitoring_demo.db"}
    monitoring = AdvancedMLMonitoring(config)

    # Create dummy model and data
    model = DummyMLModel(n_features=5, accuracy=0.87)

    # Generate test data
    np.random.seed(42)
    n_samples = 1000

    test_data = pd.DataFrame(
        {
            "feature_0": np.random.randn(n_samples),
            "feature_1": np.random.randn(n_samples),
            "feature_2": np.random.randn(n_samples),
            "feature_3": np.random.randn(n_samples),
            "feature_4": np.random.randn(n_samples),
            "gender": np.random.choice(["M", "F"], n_samples),  # Protected attribute
            "age_group": np.random.choice(
                ["young", "old"], n_samples
            ),  # Another protected attribute
        }
    )

    test_labels = np.random.randint(0, 2, n_samples)
    training_data = np.random.randn(500, 5)
    feature_names = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]

    # Register model for monitoring
    baseline_metrics = {
        "accuracy": 0.87,
        "precision": 0.85,
        "recall": 0.88,
        "f1_score": 0.86,
    }

    await monitoring.register_model_for_monitoring(
        "demo_model",
        model,
        baseline_metrics,
        training_data,
        feature_names,
        ["gender", "age_group"],
    )

    print("Registered model for advanced monitoring")

    print("\n1. Initial Monitoring Run")

    results = await monitoring.run_comprehensive_monitoring(
        "demo_model", test_data, test_labels
    )

    print(f"Monitoring completed for model: {results['model_id']}")
    print(f"Performance alerts: {len(results['performance_alerts'])}")
    print(
        f"Monitoring status: {results['monitoring_summary'].get('monitoring_status', 'unknown')}"
    )

    # Show interpretability report summary
    if (
        results["interpretability_report"]
        and "error" not in results["interpretability_report"]
    ):
        report = results["interpretability_report"]
        print(f"\nInterpretability Report:")
        print(f"- Global explanations: {len(report.get('global_explanations', {}))}")
        print(f"- Local explanations: {len(report.get('local_explanations', []))}")
        print(f"- Bias assessments: {len(report.get('bias_assessments', []))}")
        print(
            f"- Performance metrics: {list(report.get('performance_summary', {}).keys())}"
        )

        # Show bias assessment results
        for assessment in report.get("bias_assessments", [])[:2]:
            print(f"\nBias Assessment for {assessment['protected_attribute']}:")
            print(f"- Bias detected: {assessment['is_biased']}")
            print(f"- Bias score: {assessment['bias_score']:.3f}")
            print(f"- Bias type: {assessment['bias_type']}")

    print(f"\nRecommendations:")
    for rec in results["recommendations"]:
        print(f"- {rec}")

    print("\n2. Simulating Performance Degradation")

    # Use model many times to trigger degradation
    for _ in range(500):
        _ = model.predict(training_data[:10])

    # Run monitoring again
    results = await monitoring.run_comprehensive_monitoring(
        "demo_model", test_data, test_labels
    )

    print(f"Performance alerts after degradation: {len(results['performance_alerts'])}")

    # Show alert details
    for alert in results["performance_alerts"]:
        print(f"\nAlert: {alert['alert_id']}")
        print(f"- Metric: {alert['metric_name']}")
        print(f"- Severity: {alert['severity']}")
        print(f"- Degradation: {alert['degradation_percentage']:.1%}")
        print(f"- Message: {alert['message']}")

    print("\n3. Testing Individual Components")

    # Test explainability engine
    print("\nTesting Explainability Engine:")
    sample_data = test_data.iloc[0][feature_names].values

    for method in [
        ExplainabilityMethod.SHAP,
        ExplainabilityMethod.PERMUTATION_IMPORTANCE,
    ]:
        try:
            explanation = await monitoring.explainability_engine.explain_prediction(
                model, sample_data, training_data, feature_names, method, "demo_model"
            )
            print(f"- {method.value}: {explanation.explanation_text[:100]}...")
            print(f"  Confidence: {explanation.confidence_score:.3f}")
        except Exception as e:
            print(f"- {method.value}: Failed ({str(e)[:50]}...)")

    # Test bias detector
    print("\nTesting Bias Detector:")
    bias_assessment = await monitoring.bias_detector.assess_bias(
        model,
        test_data,
        test_labels,
        "gender",
        BiasType.DEMOGRAPHIC_PARITY,
        "demo_model",
    )

    print(f"- Bias detected: {bias_assessment.is_biased}")
    print(f"- Bias score: {bias_assessment.bias_score:.3f}")
    print(f"- Group metrics: {list(bias_assessment.group_metrics.keys())}")

    print("\n4. Alert System Testing")

    # Get monitoring summary
    summary = await monitoring.dashboard.get_monitoring_summary("demo_model")
    print(f"\nMonitoring Summary:")
    print(f"- Total alerts: {summary['alert_summary']['total']}")
    print(f"- Critical alerts: {summary['alert_summary']['critical']}")
    print(f"- Warning alerts: {summary['alert_summary']['warning']}")
    print(f"- Overall status: {summary['monitoring_status']}")

    # Acknowledge alerts
    if results["performance_alerts"]:
        first_alert_id = results["performance_alerts"][0]["alert_id"]
        acknowledged = await monitoring.alert_system.acknowledge_alert(first_alert_id)
        print(f"- Acknowledged alert {first_alert_id}: {acknowledged}")

    print("\n5. Dashboard Data")

    dashboard_data = await monitoring.get_dashboard_data("demo_model")
    print(f"Dashboard available: {dashboard_data['dashboard_available']}")
    print(f"Status: {dashboard_data['status']}")

    return monitoring


if __name__ == "__main__":
    asyncio.run(demo_advanced_monitoring())
