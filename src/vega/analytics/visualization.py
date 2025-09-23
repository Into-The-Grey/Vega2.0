"""
Analytics visualization and dashboard system
===========================================

This module provides interactive dashboards, charts, and reporting
capabilities for the analytics data.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse


class ChartType(Enum):
    """Types of charts available"""

    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"


@dataclass
class ChartConfig:
    """Configuration for a chart"""

    chart_id: str
    title: str
    chart_type: ChartType
    data_source: str
    x_axis: str
    y_axis: str
    filters: Dict[str, Any]
    refresh_interval: int = 30  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "chart_type": self.chart_type.value,
            "data_source": self.data_source,
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "filters": self.filters,
            "refresh_interval": self.refresh_interval,
        }


@dataclass
class Dashboard:
    """Represents a dashboard configuration"""

    id: str
    name: str
    description: str
    charts: List[ChartConfig]
    layout: Dict[str, Any]
    created_by: str
    created_at: datetime
    is_public: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "charts": [chart.to_dict() for chart in self.charts],
            "layout": self.layout,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "is_public": self.is_public,
        }


class VisualizationManager:
    """Manages dashboards and visualizations"""

    def __init__(self):
        self.dashboards: Dict[str, Dashboard] = {}
        self.chart_configs: Dict[str, ChartConfig] = {}
        self._setup_default_dashboards()

    def _setup_default_dashboards(self):
        """Setup default system dashboards"""
        # System Overview Dashboard
        system_charts = [
            ChartConfig(
                chart_id="system_metrics",
                title="System Performance",
                chart_type=ChartType.LINE,
                data_source="system_metrics",
                x_axis="timestamp",
                y_axis="value",
                filters={"metrics": ["cpu_usage", "memory_usage"]},
            ),
            ChartConfig(
                chart_id="event_volume",
                title="Event Volume",
                chart_type=ChartType.BAR,
                data_source="events",
                x_axis="hour",
                y_axis="count",
                filters={},
            ),
            ChartConfig(
                chart_id="user_activity",
                title="User Activity",
                chart_type=ChartType.AREA,
                data_source="user_events",
                x_axis="timestamp",
                y_axis="active_users",
                filters={},
            ),
            ChartConfig(
                chart_id="error_rates",
                title="Error Rates",
                chart_type=ChartType.LINE,
                data_source="errors",
                x_axis="timestamp",
                y_axis="error_rate",
                filters={},
            ),
        ]

        system_dashboard = Dashboard(
            id="system_overview",
            name="System Overview",
            description="Overall system health and performance metrics",
            charts=system_charts,
            layout={
                "grid": [
                    [
                        {"chart": "system_metrics", "span": 6},
                        {"chart": "event_volume", "span": 6},
                    ],
                    [
                        {"chart": "user_activity", "span": 6},
                        {"chart": "error_rates", "span": 6},
                    ],
                ]
            },
            created_by="system",
            created_at=datetime.now(),
            is_public=True,
        )

        self.dashboards["system_overview"] = system_dashboard

        # Personal Workspace Analytics Dashboard
        collab_charts = [
            ChartConfig(
                chart_id="workspace_activity",
                title="Workspace Activity",
                chart_type=ChartType.BAR,
                data_source="workspace_events",
                x_axis="workspace_id",
                y_axis="event_count",
                filters={},
            ),
            ChartConfig(
                chart_id="document_edits",
                title="Document Edits Over Time",
                chart_type=ChartType.LINE,
                data_source="document_events",
                x_axis="timestamp",
                y_axis="edit_count",
                filters={"event_type": "document_edited"},
            ),
            ChartConfig(
                chart_id="personal_workspace",
                title="Personal Workspace Activity",
                chart_type=ChartType.SCATTER,
                data_source="personal_workspace_matrix",
                x_axis="user1",
                y_axis="user2",
                filters={},
            ),
            ChartConfig(
                chart_id="message_volume",
                title="Chat Message Volume",
                chart_type=ChartType.AREA,
                data_source="chat_events",
                x_axis="timestamp",
                y_axis="message_count",
                filters={},
            ),
        ]

        collab_dashboard = Dashboard(
            id="personal_workspace_analytics",
            name="Personal Workspace Analytics",
            description="Team collaboration and workspace activity metrics",
            charts=collab_charts,
            layout={
                "grid": [
                    [{"chart": "workspace_activity", "span": 12}],
                    [
                        {"chart": "document_edits", "span": 6},
                        {"chart": "message_volume", "span": 6},
                    ],
                    [{"chart": "user_collaboration", "span": 12}],
                ]
            },
            created_by="system",
            created_at=datetime.now(),
            is_public=True,
        )

        self.dashboards["collaboration_analytics"] = collab_dashboard

    def create_dashboard(
        self,
        name: str,
        description: str,
        created_by: str,
        charts: Optional[List[ChartConfig]] = None,
        layout: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new dashboard"""
        dashboard_id = (
            f"dashboard_{len(self.dashboards)}_{int(datetime.now().timestamp())}"
        )

        dashboard = Dashboard(
            id=dashboard_id,
            name=name,
            description=description,
            charts=charts or [],
            layout=layout or {"grid": []},
            created_by=created_by,
            created_at=datetime.now(),
        )

        self.dashboards[dashboard_id] = dashboard
        return dashboard_id

    def add_chart_to_dashboard(self, dashboard_id: str, chart: ChartConfig) -> bool:
        """Add a chart to a dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False

        dashboard.charts.append(chart)
        self.chart_configs[chart.chart_id] = chart
        return True

    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get a dashboard by ID"""
        return self.dashboards.get(dashboard_id)

    def list_dashboards(self, user_id: Optional[str] = None) -> List[Dashboard]:
        """List available dashboards"""
        dashboards = list(self.dashboards.values())

        if user_id:
            # Filter to public dashboards and user's own dashboards
            dashboards = [
                d for d in dashboards if d.is_public or d.created_by == user_id
            ]
        else:
            # Return only public dashboards if no user specified
            dashboards = [d for d in dashboards if d.is_public]

        return dashboards


def create_visualization_router(analytics_collector, analytics_engine) -> APIRouter:
    """Create FastAPI router for analytics visualization"""
    router = APIRouter(prefix="/analytics", tags=["analytics"])
    visualization_manager = VisualizationManager()

    @router.get("/dashboards")
    async def list_dashboards(user_id: Optional[str] = None):
        """List available dashboards"""
        dashboards = visualization_manager.list_dashboards(user_id)
        return {"dashboards": [d.to_dict() for d in dashboards]}

    @router.get("/dashboards/{dashboard_id}")
    async def get_dashboard(dashboard_id: str):
        """Get dashboard configuration"""
        dashboard = visualization_manager.get_dashboard(dashboard_id)
        if not dashboard:
            raise HTTPException(status_code=404, detail="Dashboard not found")

        return dashboard.to_dict()

    @router.get("/data/events")
    async def get_events_data(
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 1000,
    ):
        """Get events data for visualization"""
        # Parse datetime strings
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        # Get events from analytics collector
        from ..analytics.collector import EventType

        event_type_enum = None
        if event_type:
            try:
                event_type_enum = EventType(event_type)
            except ValueError:
                pass

        events = await analytics_collector.get_events(
            event_type=event_type_enum,
            user_id=user_id,
            workspace_id=workspace_id,
            start_time=start_dt,
            end_time=end_dt,
            limit=limit,
        )

        return {"events": [event.to_dict() for event in events]}

    @router.get("/data/metrics")
    async def get_metrics_data():
        """Get current metrics data"""
        metrics = await analytics_collector.get_metrics()
        return {"metrics": metrics}

    @router.get("/data/summary")
    async def get_analytics_summary(
        start_time: Optional[str] = None, end_time: Optional[str] = None
    ):
        """Get analytics summary"""
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        summary = await analytics_collector.get_analytics_summary(start_dt, end_dt)
        return summary

    @router.get("/data/anomalies")
    async def get_anomalies(
        start_time: Optional[str] = None, end_time: Optional[str] = None
    ):
        """Get detected anomalies"""
        start_dt = (
            datetime.fromisoformat(start_time)
            if start_time
            else datetime.now() - timedelta(days=7)
        )
        end_dt = datetime.fromisoformat(end_time) if end_time else datetime.now()

        anomalies = [
            a.to_dict()
            for a in analytics_engine.detected_anomalies
            if start_dt <= a.timestamp <= end_dt
        ]

        return {"anomalies": anomalies}

    @router.get("/dashboard", response_class=HTMLResponse)
    async def analytics_dashboard():
        """Serve interactive analytics dashboard"""
        return HTMLResponse(
            content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vega 2.0 Analytics Dashboard</title>
            <meta charset="utf-8">
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .header {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .dashboard-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 20px;
                }
                .chart-card {
                    background: white;
                    border-radius: 8px;
                    padding: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    min-height: 300px;
                }
                .chart-title {
                    font-size: 18px;
                    font-weight: 600;
                    margin-bottom: 15px;
                    color: #333;
                }
                .chart-container {
                    position: relative;
                    height: 250px;
                }
                .metrics-summary {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .metric-item {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #007bff;
                }
                .metric-label {
                    font-size: 12px;
                    color: #666;
                    margin-top: 5px;
                }
                .controls {
                    display: flex;
                    gap: 10px;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .control-group {
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }
                label {
                    font-size: 12px;
                    color: #666;
                }
                select, input {
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }
                button {
                    padding: 8px 16px;
                    background: #007bff;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                }
                button:hover {
                    background: #0056b3;
                }
                .status-indicator {
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    margin-right: 8px;
                }
                .status-online { background: #28a745; }
                .status-warning { background: #ffc107; }
                .status-error { background: #dc3545; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Vega 2.0 Analytics Dashboard</h1>
                <div class="controls">
                    <div class="control-group">
                        <label>Time Range</label>
                        <select id="timeRange">
                            <option value="1h">Last Hour</option>
                            <option value="6h">Last 6 Hours</option>
                            <option value="24h" selected>Last 24 Hours</option>
                            <option value="7d">Last 7 Days</option>
                            <option value="30d">Last 30 Days</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Auto Refresh</label>
                        <select id="autoRefresh">
                            <option value="0">Off</option>
                            <option value="30" selected>30 seconds</option>
                            <option value="60">1 minute</option>
                            <option value="300">5 minutes</option>
                        </select>
                    </div>
                    <button onclick="refreshDashboard()">Refresh Now</button>
                </div>
                
                <div class="metrics-summary" id="metricsSummary">
                    <div class="metric-item">
                        <div class="metric-value" id="totalEvents">-</div>
                        <div class="metric-label">Total Events</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="activeUsers">-</div>
                        <div class="metric-label">Active Users</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="systemStatus">
                            <span class="status-indicator status-online"></span>Online
                        </div>
                        <div class="metric-label">System Status</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value" id="anomalies">-</div>
                        <div class="metric-label">Anomalies</div>
                    </div>
                </div>
            </div>
            
            <div class="dashboard-grid">
                <div class="chart-card">
                    <div class="chart-title">Event Volume Over Time</div>
                    <div class="chart-container">
                        <canvas id="eventVolumeChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <div class="chart-title">User Activity</div>
                    <div class="chart-container">
                        <canvas id="userActivityChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <div class="chart-title">Event Types Distribution</div>
                    <div class="chart-container">
                        <canvas id="eventTypesChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <div class="chart-title">System Performance</div>
                    <div class="chart-container">
                        <canvas id="performanceChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <div class="chart-title">Workspace Activity</div>
                    <div class="chart-container">
                        <canvas id="workspaceChart"></canvas>
                    </div>
                </div>
                
                <div class="chart-card">
                    <div class="chart-title">Recent Anomalies</div>
                    <div id="anomaliesList" style="height: 250px; overflow-y: auto;">
                        <p style="color: #666; text-align: center; margin-top: 100px;">Loading...</p>
                    </div>
                </div>
            </div>
            
            <script>
                let charts = {};
                let refreshInterval = null;
                
                // Initialize charts
                function initializeCharts() {
                    // Event Volume Chart
                    const eventVolumeCtx = document.getElementById('eventVolumeChart').getContext('2d');
                    charts.eventVolume = new Chart(eventVolumeCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Events',
                                data: [],
                                borderColor: '#007bff',
                                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                    
                    // User Activity Chart
                    const userActivityCtx = document.getElementById('userActivityChart').getContext('2d');
                    charts.userActivity = new Chart(userActivityCtx, {
                        type: 'area',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Active Users',
                                data: [],
                                borderColor: '#28a745',
                                backgroundColor: 'rgba(40, 167, 69, 0.2)',
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                    
                    // Event Types Chart
                    const eventTypesCtx = document.getElementById('eventTypesChart').getContext('2d');
                    charts.eventTypes = new Chart(eventTypesCtx, {
                        type: 'doughnut',
                        data: {
                            labels: [],
                            datasets: [{
                                data: [],
                                backgroundColor: [
                                    '#007bff', '#28a745', '#ffc107', '#dc3545',
                                    '#6f42c1', '#fd7e14', '#20c997', '#6c757d'
                                ]
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false
                        }
                    });
                    
                    // Performance Chart
                    const performanceCtx = document.getElementById('performanceChart').getContext('2d');
                    charts.performance = new Chart(performanceCtx, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [
                                {
                                    label: 'Response Time (ms)',
                                    data: [],
                                    borderColor: '#ffc107',
                                    backgroundColor: 'rgba(255, 193, 7, 0.1)'
                                },
                                {
                                    label: 'Error Rate (%)',
                                    data: [],
                                    borderColor: '#dc3545',
                                    backgroundColor: 'rgba(220, 53, 69, 0.1)'
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                    
                    // Workspace Chart
                    const workspaceCtx = document.getElementById('workspaceChart').getContext('2d');
                    charts.workspace = new Chart(workspaceCtx, {
                        type: 'bar',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Activity',
                                data: [],
                                backgroundColor: '#6f42c1'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                async function loadDashboardData() {
                    try {
                        // Load summary data
                        const summaryResponse = await fetch('/analytics/data/summary');
                        const summary = await summaryResponse.json();
                        
                        // Update summary metrics
                        document.getElementById('totalEvents').textContent = summary.total_events || 0;
                        document.getElementById('activeUsers').textContent = summary.unique_users || 0;
                        
                        // Update event volume chart
                        if (summary.hourly_activity) {
                            const hourlyData = Object.entries(summary.hourly_activity).sort();
                            charts.eventVolume.data.labels = hourlyData.map(([hour, _]) => 
                                new Date(hour).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
                            );
                            charts.eventVolume.data.datasets[0].data = hourlyData.map(([_, count]) => count);
                            charts.eventVolume.update();
                        }
                        
                        // Update event types chart
                        if (summary.event_types) {
                            const eventTypes = Object.entries(summary.event_types);
                            charts.eventTypes.data.labels = eventTypes.map(([type, _]) => type);
                            charts.eventTypes.data.datasets[0].data = eventTypes.map(([_, count]) => count);
                            charts.eventTypes.update();
                        }
                        
                        // Load anomalies
                        const anomaliesResponse = await fetch('/analytics/data/anomalies');
                        const anomaliesData = await anomaliesResponse.json();
                        
                        document.getElementById('anomalies').textContent = anomaliesData.anomalies.length;
                        
                        // Update anomalies list
                        const anomaliesList = document.getElementById('anomaliesList');
                        if (anomaliesData.anomalies.length > 0) {
                            anomaliesList.innerHTML = anomaliesData.anomalies.map(anomaly => `
                                <div style="border-left: 4px solid ${anomaly.alert_level === 'critical' ? '#dc3545' : '#ffc107'}; 
                                           padding: 10px; margin-bottom: 10px; background: #f8f9fa;">
                                    <strong>${anomaly.metric_name}</strong>
                                    <div style="font-size: 12px; color: #666;">
                                        ${new Date(anomaly.timestamp).toLocaleString()}
                                    </div>
                                    <div style="font-size: 14px; margin-top: 5px;">
                                        ${anomaly.description}
                                    </div>
                                </div>
                            `).join('');
                        } else {
                            anomaliesList.innerHTML = '<p style="color: #666; text-align: center; margin-top: 100px;">No anomalies detected</p>';
                        }
                        
                    } catch (error) {
                        console.error('Error loading dashboard data:', error);
                        document.getElementById('systemStatus').innerHTML = 
                            '<span class="status-indicator status-error"></span>Error';
                    }
                }
                
                function refreshDashboard() {
                    loadDashboardData();
                }
                
                function setupAutoRefresh() {
                    const interval = parseInt(document.getElementById('autoRefresh').value);
                    
                    if (refreshInterval) {
                        clearInterval(refreshInterval);
                    }
                    
                    if (interval > 0) {
                        refreshInterval = setInterval(refreshDashboard, interval * 1000);
                    }
                }
                
                // Event listeners
                document.getElementById('timeRange').addEventListener('change', refreshDashboard);
                document.getElementById('autoRefresh').addEventListener('change', setupAutoRefresh);
                
                // Initialize dashboard
                initializeCharts();
                loadDashboardData();
                setupAutoRefresh();
            </script>
        </body>
        </html>
        """
        )

    return router


# Export the visualization manager
visualization_manager = VisualizationManager()
