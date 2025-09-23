"""
Personal Analytics Dashboard

Provides personal dashboards, usage analytics, and optimization
recommendations for individual productivity tracking.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, Counter
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""

    USAGE = "usage"
    PERFORMANCE = "performance"
    PRODUCTIVITY = "productivity"
    FEATURE = "feature"
    CONTENT = "content"
    TIME = "time"
    QUALITY = "quality"
    GOAL = "goal"


class ChartType(Enum):
    """Types of charts for visualization"""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TREEMAP = "treemap"


@dataclass
class Metric:
    """Individual metric data point"""

    metric_id: str
    metric_type: MetricType
    name: str
    value: Union[int, float, str]
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Dashboard:
    """Dashboard configuration"""

    dashboard_id: str
    name: str
    description: str
    widgets: List[Dict[str, Any]]
    layout: Dict[str, Any]
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Widget:
    """Dashboard widget configuration"""

    widget_id: str
    title: str
    chart_type: ChartType
    metric_query: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    style: Dict[str, Any] = None

    def __post_init__(self):
        if self.style is None:
            self.style = {}


@dataclass
class AnalyticsReport:
    """Analytics report with insights"""

    report_id: str
    title: str
    summary: str
    insights: List[str]
    recommendations: List[str]
    metrics: List[Metric]
    charts: List[Dict[str, Any]]
    generated_at: datetime
    period_start: datetime
    period_end: datetime


class PersonalAnalyticsDashboard:
    """
    Personal analytics dashboard for tracking productivity and usage
    """

    def __init__(self, data_manager=None):
        self.data_manager = data_manager
        self.metrics: List[Metric] = []
        self.dashboards: Dict[str, Dashboard] = {}
        self.active_dashboard: Optional[str] = None

        # Initialize default dashboards
        self._initialize_default_dashboards()

    def _initialize_default_dashboards(self):
        """Initialize default dashboard configurations"""

        # Productivity Overview Dashboard
        productivity_widgets = [
            {
                "widget_id": "daily_activity",
                "title": "Daily Activity",
                "chart_type": ChartType.LINE.value,
                "metric_query": {"metric_type": "productivity", "timeframe": "7d"},
                "position": {"x": 0, "y": 0, "width": 6, "height": 4},
            },
            {
                "widget_id": "feature_usage",
                "title": "Feature Usage",
                "chart_type": ChartType.PIE.value,
                "metric_query": {"metric_type": "feature", "timeframe": "30d"},
                "position": {"x": 6, "y": 0, "width": 6, "height": 4},
            },
            {
                "widget_id": "content_creation",
                "title": "Content Creation",
                "chart_type": ChartType.BAR.value,
                "metric_query": {"metric_type": "content", "timeframe": "30d"},
                "position": {"x": 0, "y": 4, "width": 12, "height": 4},
            },
        ]

        productivity_dashboard = Dashboard(
            dashboard_id="productivity",
            name="Productivity Overview",
            description="Track your daily productivity and content creation",
            widgets=productivity_widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
        )

        # Usage Analytics Dashboard
        usage_widgets = [
            {
                "widget_id": "session_duration",
                "title": "Session Duration",
                "chart_type": ChartType.HISTOGRAM.value,
                "metric_query": {
                    "metric_type": "usage",
                    "metric_name": "session_duration",
                },
                "position": {"x": 0, "y": 0, "width": 6, "height": 4},
            },
            {
                "widget_id": "api_usage",
                "title": "API Usage",
                "chart_type": ChartType.LINE.value,
                "metric_query": {"metric_type": "usage", "metric_name": "api_calls"},
                "position": {"x": 6, "y": 0, "width": 6, "height": 4},
            },
            {
                "widget_id": "error_rate",
                "title": "Error Rate",
                "chart_type": ChartType.GAUGE.value,
                "metric_query": {
                    "metric_type": "performance",
                    "metric_name": "error_rate",
                },
                "position": {"x": 0, "y": 4, "width": 4, "height": 4},
            },
            {
                "widget_id": "response_time",
                "title": "Response Time",
                "chart_type": ChartType.LINE.value,
                "metric_query": {
                    "metric_type": "performance",
                    "metric_name": "response_time",
                },
                "position": {"x": 4, "y": 4, "width": 8, "height": 4},
            },
        ]

        usage_dashboard = Dashboard(
            dashboard_id="usage",
            name="Usage Analytics",
            description="Monitor system usage and performance metrics",
            widgets=usage_widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
        )

        # Goals & Progress Dashboard
        goals_widgets = [
            {
                "widget_id": "goal_progress",
                "title": "Goal Progress",
                "chart_type": ChartType.BAR.value,
                "metric_query": {"metric_type": "goal", "timeframe": "30d"},
                "position": {"x": 0, "y": 0, "width": 12, "height": 4},
            },
            {
                "widget_id": "quality_score",
                "title": "Quality Score",
                "chart_type": ChartType.GAUGE.value,
                "metric_query": {
                    "metric_type": "quality",
                    "metric_name": "overall_quality",
                },
                "position": {"x": 0, "y": 4, "width": 6, "height": 4},
            },
            {
                "widget_id": "time_distribution",
                "title": "Time Distribution",
                "chart_type": ChartType.TREEMAP.value,
                "metric_query": {"metric_type": "time", "timeframe": "7d"},
                "position": {"x": 6, "y": 4, "width": 6, "height": 4},
            },
        ]

        goals_dashboard = Dashboard(
            dashboard_id="goals",
            name="Goals & Progress",
            description="Track progress towards personal goals and quality metrics",
            widgets=goals_widgets,
            layout={"grid_columns": 12, "grid_rows": 8},
        )

        # Register dashboards
        for dashboard in [productivity_dashboard, usage_dashboard, goals_dashboard]:
            self.dashboards[dashboard.dashboard_id] = dashboard

        # Set default active dashboard
        self.active_dashboard = "productivity"

    async def track_metric(
        self,
        metric_type: MetricType,
        name: str,
        value: Union[int, float, str],
        unit: str = "",
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Track a new metric data point"""

        metric_id = f"{metric_type.value}_{name}_{datetime.now().timestamp()}"

        metric = Metric(
            metric_id=metric_id,
            metric_type=metric_type,
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self.metrics.append(metric)

        # Store in database if available
        if self.data_manager:
            await self._store_metric_in_db(metric)

        logger.debug(f"Tracked metric: {name} = {value} {unit}")
        return metric_id

    async def _store_metric_in_db(self, metric: Metric):
        """Store metric in database"""
        try:
            data = {
                "metric_id": metric.metric_id,
                "metric_type": metric.metric_type.value,
                "name": metric.name,
                "value": str(metric.value),
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "metadata": json.dumps(metric.metadata),
            }

            await self.data_manager.insert_data("personal_analytics", data)

        except Exception as e:
            logger.error(f"Failed to store metric in database: {e}")

    async def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        metric_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Metric]:
        """Get metrics with optional filtering"""

        # Try to load from database first
        if self.data_manager:
            db_metrics = await self._load_metrics_from_db(
                metric_type, metric_name, start_date, end_date, limit
            )
            if db_metrics:
                return db_metrics

        # Filter in-memory metrics
        filtered_metrics = self.metrics

        if metric_type:
            filtered_metrics = [
                m for m in filtered_metrics if m.metric_type == metric_type
            ]

        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.name == metric_name]

        if start_date:
            filtered_metrics = [
                m for m in filtered_metrics if m.timestamp >= start_date
            ]

        if end_date:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_date]

        # Sort by timestamp
        filtered_metrics.sort(key=lambda m: m.timestamp)

        if limit:
            filtered_metrics = filtered_metrics[-limit:]

        return filtered_metrics

    async def _load_metrics_from_db(
        self,
        metric_type: Optional[MetricType] = None,
        metric_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Metric]:
        """Load metrics from database"""
        try:
            filters = {}

            if metric_type:
                filters["metric_type"] = metric_type.value
            if metric_name:
                filters["name"] = metric_name

            # Add date range to SQL query if needed
            order_by = "timestamp DESC"

            rows = await self.data_manager.query_data(
                "personal_analytics", filters, order_by, limit
            )

            metrics = []
            for row in rows:
                # Apply date filtering
                timestamp = datetime.fromisoformat(row["timestamp"])
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue

                metric = Metric(
                    metric_id=row["metric_id"],
                    metric_type=MetricType(row["metric_type"]),
                    name=row["name"],
                    value=self._parse_value(row["value"]),
                    unit=row["unit"],
                    timestamp=timestamp,
                    metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
                )
                metrics.append(metric)

            return metrics

        except Exception as e:
            logger.error(f"Failed to load metrics from database: {e}")
            return []

    def _parse_value(self, value_str: str) -> Union[int, float, str]:
        """Parse value from string storage"""
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            return value_str

    async def generate_chart(self, widget_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart based on widget configuration"""

        chart_type = ChartType(widget_config["chart_type"])
        query = widget_config["metric_query"]

        # Get metrics based on query
        metrics = await self._query_metrics(query)

        if not metrics:
            return {"error": "No data available"}

        # Generate chart based on type
        if chart_type == ChartType.LINE:
            return self._create_line_chart(metrics, widget_config["title"])
        elif chart_type == ChartType.BAR:
            return self._create_bar_chart(metrics, widget_config["title"])
        elif chart_type == ChartType.PIE:
            return self._create_pie_chart(metrics, widget_config["title"])
        elif chart_type == ChartType.GAUGE:
            return self._create_gauge_chart(metrics, widget_config["title"])
        elif chart_type == ChartType.HISTOGRAM:
            return self._create_histogram_chart(metrics, widget_config["title"])
        elif chart_type == ChartType.TREEMAP:
            return self._create_treemap_chart(metrics, widget_config["title"])
        else:
            return {"error": f"Unsupported chart type: {chart_type.value}"}

    async def _query_metrics(self, query: Dict[str, Any]) -> List[Metric]:
        """Query metrics based on configuration"""

        metric_type = None
        if "metric_type" in query:
            metric_type = MetricType(query["metric_type"])

        metric_name = query.get("metric_name")

        # Parse timeframe
        end_date = datetime.now()
        start_date = None

        timeframe = query.get("timeframe", "7d")
        if timeframe.endswith("d"):
            days = int(timeframe[:-1])
            start_date = end_date - timedelta(days=days)
        elif timeframe.endswith("h"):
            hours = int(timeframe[:-1])
            start_date = end_date - timedelta(hours=hours)

        return await self.get_metrics(metric_type, metric_name, start_date, end_date)

    def _create_line_chart(self, metrics: List[Metric], title: str) -> Dict[str, Any]:
        """Create line chart from metrics"""

        # Group by name if multiple metric names
        grouped = defaultdict(list)
        for metric in metrics:
            grouped[metric.name].append(metric)

        fig = go.Figure()

        for name, metric_list in grouped.items():
            x_values = [m.timestamp for m in metric_list]
            y_values = [
                (
                    float(m.value)
                    if isinstance(m.value, (int, float, str))
                    and str(m.value).replace(".", "").isdigit()
                    else 0
                )
                for m in metric_list
            ]

            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode="lines+markers",
                    name=name,
                    line=dict(width=2),
                    marker=dict(size=6),
                )
            )

        fig.update_layout(
            title=title, xaxis_title="Time", yaxis_title="Value", hovermode="x unified"
        )

        return {"chart": fig.to_dict(), "type": "line"}

    def _create_bar_chart(self, metrics: List[Metric], title: str) -> Dict[str, Any]:
        """Create bar chart from metrics"""

        # Aggregate by name
        aggregated = defaultdict(list)
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                aggregated[metric.name].append(float(metric.value))

        names = list(aggregated.keys())
        values = [sum(aggregated[name]) for name in names]

        fig = go.Figure(
            data=[go.Bar(x=names, y=values, marker_color="rgb(55, 83, 109)")]
        )

        fig.update_layout(title=title, xaxis_title="Category", yaxis_title="Value")

        return {"chart": fig.to_dict(), "type": "bar"}

    def _create_pie_chart(self, metrics: List[Metric], title: str) -> Dict[str, Any]:
        """Create pie chart from metrics"""

        # Count occurrences or sum values by name
        aggregated = defaultdict(float)
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                aggregated[metric.name] += float(metric.value)
            else:
                aggregated[metric.name] += 1

        names = list(aggregated.keys())
        values = list(aggregated.values())

        fig = go.Figure(data=[go.Pie(labels=names, values=values, hole=0.3)])

        fig.update_layout(title=title)

        return {"chart": fig.to_dict(), "type": "pie"}

    def _create_gauge_chart(self, metrics: List[Metric], title: str) -> Dict[str, Any]:
        """Create gauge chart from metrics"""

        if not metrics:
            value = 0
        else:
            # Use latest metric value
            latest_metric = max(metrics, key=lambda m: m.timestamp)
            value = (
                float(latest_metric.value)
                if isinstance(latest_metric.value, (int, float))
                else 0
            )

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=value,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": title},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "gray"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            )
        )

        return {"chart": fig.to_dict(), "type": "gauge"}

    def _create_histogram_chart(
        self, metrics: List[Metric], title: str
    ) -> Dict[str, Any]:
        """Create histogram from metrics"""

        values = []
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                values.append(float(metric.value))

        if not values:
            return {"error": "No numeric values for histogram"}

        fig = go.Figure(data=[go.Histogram(x=values, nbinsx=20)])

        fig.update_layout(title=title, xaxis_title="Value", yaxis_title="Frequency")

        return {"chart": fig.to_dict(), "type": "histogram"}

    def _create_treemap_chart(
        self, metrics: List[Metric], title: str
    ) -> Dict[str, Any]:
        """Create treemap chart from metrics"""

        # Aggregate by name
        aggregated = defaultdict(float)
        for metric in metrics:
            if isinstance(metric.value, (int, float)):
                aggregated[metric.name] += float(metric.value)
            else:
                aggregated[metric.name] += 1

        names = list(aggregated.keys())
        values = list(aggregated.values())

        fig = go.Figure(
            go.Treemap(labels=names, values=values, parents=[""] * len(names))
        )

        fig.update_layout(title=title)

        return {"chart": fig.to_dict(), "type": "treemap"}

    async def render_dashboard(
        self, dashboard_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Render complete dashboard with all widgets"""

        if dashboard_id is None:
            dashboard_id = self.active_dashboard

        if dashboard_id not in self.dashboards:
            return {"error": f"Dashboard {dashboard_id} not found"}

        dashboard = self.dashboards[dashboard_id]

        rendered_widgets = []
        for widget_config in dashboard.widgets:
            chart_data = await self.generate_chart(widget_config)

            widget = {
                "widget_id": widget_config["widget_id"],
                "title": widget_config["title"],
                "position": widget_config["position"],
                "chart_data": chart_data,
            }
            rendered_widgets.append(widget)

        return {
            "dashboard_id": dashboard_id,
            "name": dashboard.name,
            "description": dashboard.description,
            "layout": dashboard.layout,
            "widgets": rendered_widgets,
            "generated_at": datetime.now().isoformat(),
        }

    async def generate_insights_report(self, days: int = 30) -> AnalyticsReport:
        """Generate insights and recommendations report"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get all metrics for the period
        metrics = await self.get_metrics(start_date=start_date, end_date=end_date)

        insights = []
        recommendations = []

        # Analyze productivity trends
        productivity_metrics = [
            m for m in metrics if m.metric_type == MetricType.PRODUCTIVITY
        ]
        if productivity_metrics:
            daily_counts = defaultdict(int)
            for metric in productivity_metrics:
                date_key = metric.timestamp.date()
                if isinstance(metric.value, (int, float)):
                    daily_counts[date_key] += float(metric.value)

            if len(daily_counts) > 1:
                values = list(daily_counts.values())
                avg_productivity = statistics.mean(values)
                trend = "increasing" if values[-1] > avg_productivity else "decreasing"

                insights.append(
                    f"Your productivity is {trend} over the last {days} days"
                )
                insights.append(
                    f"Average daily productivity score: {avg_productivity:.1f}"
                )

                if trend == "decreasing":
                    recommendations.append(
                        "Consider reviewing your daily routines and identifying productivity bottlenecks"
                    )
                else:
                    recommendations.append(
                        "Great job maintaining high productivity! Keep up the good work"
                    )

        # Analyze feature usage patterns
        feature_metrics = [m for m in metrics if m.metric_type == MetricType.FEATURE]
        if feature_metrics:
            feature_counts = Counter([m.name for m in feature_metrics])
            most_used = feature_counts.most_common(3)
            least_used = (
                feature_counts.most_common()[-3:] if len(feature_counts) > 3 else []
            )

            insights.append(
                f"Most used features: {', '.join([f[0] for f in most_used])}"
            )

            if least_used:
                underutilized = [
                    f[0]
                    for f in least_used
                    if f[1] < statistics.mean(feature_counts.values())
                ]
                if underutilized:
                    recommendations.append(
                        f"Consider exploring underutilized features: {', '.join(underutilized)}"
                    )

        # Analyze performance metrics
        performance_metrics = [
            m for m in metrics if m.metric_type == MetricType.PERFORMANCE
        ]
        if performance_metrics:
            response_times = [
                float(m.value)
                for m in performance_metrics
                if m.name == "response_time" and isinstance(m.value, (int, float))
            ]

            if response_times:
                avg_response = statistics.mean(response_times)
                insights.append(f"Average response time: {avg_response:.2f}ms")

                if avg_response > 1000:  # More than 1 second
                    recommendations.append(
                        "System performance could be improved - consider optimizing slow operations"
                    )

        # Generate summary
        summary = f"Analytics report for {days}-day period ending {end_date.strftime('%Y-%m-%d')}"

        report = AnalyticsReport(
            report_id=f"report_{datetime.now().timestamp()}",
            title=f"{days}-Day Personal Analytics Report",
            summary=summary,
            insights=insights,
            recommendations=recommendations,
            metrics=metrics,
            charts=[],  # Could add chart data here
            generated_at=datetime.now(),
            period_start=start_date,
            period_end=end_date,
        )

        return report

    async def set_goal(
        self,
        name: str,
        target_value: float,
        metric_type: MetricType,
        unit: str = "",
        deadline: Optional[datetime] = None,
    ) -> str:
        """Set a personal goal"""

        goal_id = f"goal_{name}_{datetime.now().timestamp()}"

        goal_metric = Metric(
            metric_id=goal_id,
            metric_type=MetricType.GOAL,
            name=name,
            value=target_value,
            unit=unit,
            timestamp=datetime.now(),
            metadata={
                "target_metric_type": metric_type.value,
                "deadline": deadline.isoformat() if deadline else None,
                "status": "active",
            },
        )

        await self.track_metric(
            MetricType.GOAL, name, target_value, unit, goal_metric.metadata
        )

        logger.info(f"Set goal: {name} = {target_value} {unit}")
        return goal_id

    async def check_goal_progress(self, goal_name: str) -> Dict[str, Any]:
        """Check progress towards a goal"""

        # Get goal definition
        goal_metrics = await self.get_metrics(
            metric_type=MetricType.GOAL, metric_name=goal_name
        )

        if not goal_metrics:
            return {"error": f"Goal {goal_name} not found"}

        goal = goal_metrics[-1]  # Latest goal definition
        target_value = float(goal.value)
        target_metric_type = MetricType(
            goal.metadata.get("target_metric_type", "productivity")
        )

        # Get actual progress metrics
        progress_metrics = await self.get_metrics(
            metric_type=target_metric_type, metric_name=goal_name
        )

        if not progress_metrics:
            current_value = 0.0
        else:
            current_value = sum(
                float(m.value)
                for m in progress_metrics
                if isinstance(m.value, (int, float))
            )

        progress_percentage = (
            (current_value / target_value * 100) if target_value > 0 else 0
        )

        return {
            "goal_name": goal_name,
            "target_value": target_value,
            "current_value": current_value,
            "progress_percentage": min(progress_percentage, 100),
            "unit": goal.unit,
            "status": "completed" if progress_percentage >= 100 else "in_progress",
        }


# Demo and testing functions
async def demo_personal_analytics():
    """Demonstrate personal analytics dashboard capabilities"""

    dashboard = PersonalAnalyticsDashboard()

    print("Personal Analytics Dashboard Demo")

    # Track some sample metrics
    await dashboard.track_metric(
        MetricType.PRODUCTIVITY, "documents_created", 5, "count"
    )
    await dashboard.track_metric(
        MetricType.PRODUCTIVITY, "words_written", 1250, "words"
    )
    await dashboard.track_metric(MetricType.FEATURE, "search_used", 1, "count")
    await dashboard.track_metric(MetricType.FEATURE, "export_used", 1, "count")
    await dashboard.track_metric(MetricType.PERFORMANCE, "response_time", 342, "ms")
    await dashboard.track_metric(MetricType.USAGE, "session_duration", 45, "minutes")

    # Set a goal
    await dashboard.set_goal("documents_created", 10, MetricType.PRODUCTIVITY, "count")

    # Check goal progress
    progress = await dashboard.check_goal_progress("documents_created")
    print(f"Goal Progress: {progress}")

    # Generate insights report
    report = await dashboard.generate_insights_report(7)
    print(f"\nInsights Report:")
    print(f"- Title: {report.title}")
    print(f"- Insights: {len(report.insights)}")
    print(f"- Recommendations: {len(report.recommendations)}")

    for insight in report.insights:
        print(f"  • {insight}")

    for rec in report.recommendations:
        print(f"  → {rec}")

    # Render dashboard
    rendered = await dashboard.render_dashboard()
    print(
        f"\nDashboard '{rendered['name']}' rendered with {len(rendered['widgets'])} widgets"
    )

    return dashboard


if __name__ == "__main__":
    asyncio.run(demo_personal_analytics())
