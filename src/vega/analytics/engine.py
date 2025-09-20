"""
Analytics engine for data processing and analysis
=================================================

This module provides advanced analytics processing including
statistical analysis, anomaly detection, and trend analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict
import statistics
import json


class AnomalyType(Enum):
    """Types of anomalies that can be detected"""

    SPIKE = "spike"
    DROP = "drop"
    TREND = "trend"
    OUTLIER = "outlier"
    PATTERN = "pattern"


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Anomaly:
    """Represents a detected anomaly"""

    id: str
    anomaly_type: AnomalyType
    metric_name: str
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    confidence: float
    description: str
    alert_level: AlertLevel
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "anomaly_type": self.anomaly_type.value,
            "metric_name": self.metric_name,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "confidence": self.confidence,
            "description": self.description,
            "alert_level": self.alert_level.value,
            "metadata": self.metadata,
        }


@dataclass
class Trend:
    """Represents a detected trend"""

    metric_name: str
    start_time: datetime
    end_time: datetime
    direction: str  # "increasing", "decreasing", "stable"
    slope: float
    correlation: float
    confidence: float
    significance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "direction": self.direction,
            "slope": self.slope,
            "correlation": self.correlation,
            "confidence": self.confidence,
            "significance": self.significance,
        }


@dataclass
class StatisticalSummary:
    """Statistical summary of a metric"""

    metric_name: str
    time_range: Tuple[datetime, datetime]
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentiles: Dict[int, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "metric_name": self.metric_name,
            "time_range": [
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat(),
            ],
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "percentiles": self.percentiles,
        }


class TimeSeriesAnalyzer:
    """Analyzes time series data for patterns and anomalies"""

    def __init__(self):
        self.historical_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(
            list
        )
        self.baseline_windows: Dict[str, int] = {}  # metric -> window size in hours
        self.sensitivity_thresholds: Dict[str, float] = {}  # metric -> threshold

    def add_data_point(self, metric_name: str, timestamp: datetime, value: float):
        """Add a data point for analysis"""
        self.historical_data[metric_name].append((timestamp, value))

        # Keep only recent data (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        self.historical_data[metric_name] = [
            (ts, val) for ts, val in self.historical_data[metric_name] if ts > cutoff
        ]

    def detect_anomalies(
        self,
        metric_name: str,
        current_value: float,
        timestamp: datetime,
        window_hours: int = 24,
    ) -> List[Anomaly]:
        """Detect anomalies in metric data"""
        anomalies = []

        if metric_name not in self.historical_data:
            return anomalies

        # Get historical data for baseline
        baseline_start = timestamp - timedelta(hours=window_hours)
        baseline_data = [
            value
            for ts, value in self.historical_data[metric_name]
            if baseline_start <= ts < timestamp
        ]

        if len(baseline_data) < 10:  # Need sufficient data
            return anomalies

        # Calculate baseline statistics
        mean_value = statistics.mean(baseline_data)
        std_dev = statistics.stdev(baseline_data) if len(baseline_data) > 1 else 0

        if std_dev == 0:
            return anomalies

        # Calculate z-score
        z_score = abs(current_value - mean_value) / std_dev

        # Detect spike or drop
        if z_score > 3:  # 3 sigma threshold
            anomaly_type = (
                AnomalyType.SPIKE if current_value > mean_value else AnomalyType.DROP
            )
            deviation = abs(current_value - mean_value)
            confidence = min(z_score / 5.0, 1.0)  # Normalize confidence

            alert_level = AlertLevel.CRITICAL if z_score > 5 else AlertLevel.WARNING

            anomaly = Anomaly(
                id=f"anomaly_{metric_name}_{timestamp.isoformat()}",
                anomaly_type=anomaly_type,
                metric_name=metric_name,
                timestamp=timestamp,
                value=current_value,
                expected_value=mean_value,
                deviation=deviation,
                confidence=confidence,
                description=f"{anomaly_type.value.title()} detected in {metric_name}: "
                f"value {current_value:.2f} vs expected {mean_value:.2f} "
                f"(z-score: {z_score:.2f})",
                alert_level=alert_level,
                metadata={"z_score": z_score, "baseline_std": std_dev},
            )

            anomalies.append(anomaly)

        return anomalies

    def analyze_trend(
        self, metric_name: str, window_hours: int = 24
    ) -> Optional[Trend]:
        """Analyze trend in metric data"""
        if metric_name not in self.historical_data:
            return None

        # Get recent data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)

        data_points = [
            (ts, value)
            for ts, value in self.historical_data[metric_name]
            if start_time <= ts <= end_time
        ]

        if len(data_points) < 10:
            return None

        # Sort by timestamp
        data_points.sort(key=lambda x: x[0])

        # Calculate linear regression
        x_values = [(ts - data_points[0][0]).total_seconds() for ts, _ in data_points]
        y_values = [value for _, value in data_points]

        slope, correlation = self._calculate_linear_regression(x_values, y_values)

        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for stable
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Calculate confidence based on correlation
        confidence = abs(correlation)

        # Calculate statistical significance
        significance = self._calculate_significance(x_values, y_values, slope)

        return Trend(
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            direction=direction,
            slope=slope,
            correlation=correlation,
            confidence=confidence,
            significance=significance,
        )

    def calculate_statistics(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[StatisticalSummary]:
        """Calculate statistical summary for metric"""
        if metric_name not in self.historical_data:
            return None

        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()

        # Filter data by time range
        values = [
            value
            for ts, value in self.historical_data[metric_name]
            if start_time <= ts <= end_time
        ]

        if not values:
            return None

        # Calculate statistics
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        min_val = min(values)
        max_val = max(values)

        # Calculate percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90, 95, 99]:
            try:
                percentiles[p] = (
                    statistics.quantiles(values, n=100)[p - 1]
                    if len(values) > 1
                    else values[0]
                )
            except (statistics.StatisticsError, IndexError):
                percentiles[p] = median_val

        return StatisticalSummary(
            metric_name=metric_name,
            time_range=(start_time, end_time),
            count=len(values),
            mean=mean_val,
            median=median_val,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            percentiles=percentiles,
        )

    def _calculate_linear_regression(
        self, x_values: List[float], y_values: List[float]
    ) -> Tuple[float, float]:
        """Calculate linear regression slope and correlation"""
        n = len(x_values)
        if n < 2:
            return 0.0, 0.0

        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        # Calculate slope
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        slope = numerator / denominator if denominator != 0 else 0

        # Calculate correlation coefficient
        x_std = statistics.stdev(x_values) if n > 1 else 0
        y_std = statistics.stdev(y_values) if n > 1 else 0

        if x_std == 0 or y_std == 0:
            correlation = 0
        else:
            correlation = numerator / (n * x_std * y_std)

        return slope, correlation

    def _calculate_significance(
        self, x_values: List[float], y_values: List[float], slope: float
    ) -> float:
        """Calculate statistical significance of trend"""
        n = len(x_values)
        if n < 3:
            return 0.0

        # Simple t-test approximation
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        # Calculate residual sum of squares
        predicted_y = [y_mean + slope * (x - x_mean) for x in x_values]
        rss = sum(
            (actual - predicted) ** 2
            for actual, predicted in zip(y_values, predicted_y)
        )

        # Calculate standard error
        se = (rss / (n - 2)) ** 0.5 if n > 2 else 1.0

        # Calculate t-statistic
        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        se_slope = se / (x_variance**0.5) if x_variance > 0 else 1.0
        t_stat = abs(slope / se_slope) if se_slope > 0 else 0

        # Convert to significance (simplified)
        significance = min(t_stat / 2.0, 1.0)

        return significance


class AnalyticsEngine:
    """Main analytics processing engine"""

    def __init__(self):
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.detected_anomalies: List[Anomaly] = []
        self.alert_subscribers: List[callable] = []

        # Performance monitoring
        self.performance_baselines: Dict[str, Dict[str, float]] = {}

        # User behavior analysis
        self.user_patterns: Dict[str, Dict[str, Any]] = {}

        # Background processing
        self._processing_tasks: List[asyncio.Task] = []
        self._start_background_processing()

    def _start_background_processing(self):
        """Start background analysis tasks"""
        # Anomaly detection task - runs every 5 minutes
        task = asyncio.create_task(self._periodic_anomaly_detection())
        self._processing_tasks.append(task)

        # Performance analysis task - runs every 10 minutes
        task = asyncio.create_task(self._periodic_performance_analysis())
        self._processing_tasks.append(task)

    async def process_metric_update(
        self, metric_name: str, value: float, timestamp: datetime
    ):
        """Process a metric update and check for anomalies"""
        # Add to time series data
        self.time_series_analyzer.add_data_point(metric_name, timestamp, value)

        # Check for anomalies
        anomalies = self.time_series_analyzer.detect_anomalies(
            metric_name, value, timestamp
        )

        for anomaly in anomalies:
            await self._handle_anomaly(anomaly)

    async def analyze_user_behavior(
        self, user_id: str, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        if not events:
            return {}

        analysis = {
            "user_id": user_id,
            "total_events": len(events),
            "event_types": defaultdict(int),
            "activity_patterns": {},
            "engagement_score": 0.0,
        }

        # Count event types
        for event in events:
            event_type = event.get("event_type", "unknown")
            analysis["event_types"][event_type] += 1

        # Analyze activity patterns by hour
        hourly_activity = defaultdict(int)
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                hour = timestamp.hour
                hourly_activity[hour] += 1
            except (KeyError, ValueError):
                continue

        analysis["activity_patterns"]["hourly"] = dict(hourly_activity)

        # Calculate engagement score (simplified)
        unique_event_types = len(analysis["event_types"])
        avg_events_per_hour = len(events) / 24 if events else 0
        analysis["engagement_score"] = min(
            unique_event_types * avg_events_per_hour / 10, 100
        )

        # Store patterns for future reference
        self.user_patterns[user_id] = analysis

        return analysis

    async def analyze_workspace_activity(
        self, workspace_id: str, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze workspace activity and collaboration patterns"""
        if not events:
            return {}

        analysis = {
            "workspace_id": workspace_id,
            "total_events": len(events),
            "unique_users": len(
                set(e.get("user_id") for e in events if e.get("user_id"))
            ),
            "collaboration_score": 0.0,
            "activity_timeline": defaultdict(int),
            "user_contributions": defaultdict(int),
        }

        # Analyze timeline
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                day = timestamp.strftime("%Y-%m-%d")
                analysis["activity_timeline"][day] += 1

                user_id = event.get("user_id")
                if user_id:
                    analysis["user_contributions"][user_id] += 1
            except (KeyError, ValueError):
                continue

        # Calculate collaboration score
        if analysis["unique_users"] > 1:
            # Higher score for more balanced contributions
            contributions = list(analysis["user_contributions"].values())
            if contributions:
                std_dev = (
                    statistics.stdev(contributions) if len(contributions) > 1 else 0
                )
                mean_contrib = statistics.mean(contributions)
                balance_factor = 1 - (std_dev / mean_contrib) if mean_contrib > 0 else 0
                analysis["collaboration_score"] = (
                    balance_factor * analysis["unique_users"] * 10
                )

        return analysis

    async def generate_performance_report(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
            },
            "anomalies": [],
            "trends": [],
            "performance_metrics": {},
            "recommendations": [],
        }

        # Get anomalies in time range
        report["anomalies"] = [
            a.to_dict()
            for a in self.detected_anomalies
            if start_time <= a.timestamp <= end_time
        ]

        # Analyze trends for key metrics
        key_metrics = ["api_requests", "response_time", "error_rate", "user_activity"]
        for metric in key_metrics:
            trend = self.time_series_analyzer.analyze_trend(metric, window_hours=24)
            if trend:
                report["trends"].append(trend.to_dict())

        # Generate recommendations based on findings
        report["recommendations"] = self._generate_recommendations(report)

        return report

    async def _handle_anomaly(self, anomaly: Anomaly):
        """Handle detected anomaly"""
        self.detected_anomalies.append(anomaly)

        # Notify subscribers
        for subscriber in self.alert_subscribers:
            try:
                await subscriber(anomaly)
            except Exception as e:
                print(f"Error notifying anomaly subscriber: {e}")

    async def _periodic_anomaly_detection(self):
        """Periodic anomaly detection task"""
        while True:
            try:
                # Check key metrics for anomalies
                key_metrics = [
                    "cpu_usage",
                    "memory_usage",
                    "response_time",
                    "error_rate",
                ]

                for metric in key_metrics:
                    # This would integrate with the metrics collector
                    # For now, we'll skip actual detection
                    pass

            except Exception as e:
                print(f"Error in anomaly detection: {e}")

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _periodic_performance_analysis(self):
        """Periodic performance analysis task"""
        while True:
            try:
                # Analyze performance trends
                # This would integrate with performance data
                pass

            except Exception as e:
                print(f"Error in performance analysis: {e}")

            await asyncio.sleep(600)  # Analyze every 10 minutes

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Check for critical anomalies
        critical_anomalies = [
            a for a in report["anomalies"] if a["alert_level"] == "critical"
        ]

        if critical_anomalies:
            recommendations.append(
                f"Investigate {len(critical_anomalies)} critical anomalies detected"
            )

        # Check for concerning trends
        negative_trends = [
            t
            for t in report["trends"]
            if t["direction"] == "decreasing" and t["confidence"] > 0.7
        ]

        if negative_trends:
            recommendations.append(
                "Monitor declining metrics for potential performance issues"
            )

        return recommendations

    def subscribe_to_alerts(self, callback: callable):
        """Subscribe to anomaly alerts"""
        self.alert_subscribers.append(callback)

    async def shutdown(self):
        """Shutdown analytics engine"""
        # Cancel background tasks
        for task in self._processing_tasks:
            task.cancel()

        await asyncio.gather(*self._processing_tasks, return_exceptions=True)


# Global analytics engine instance
analytics_engine = AnalyticsEngine()
