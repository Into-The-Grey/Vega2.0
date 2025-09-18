"""
Real-Time Autonomous AI Dashboard System

This module provides a comprehensive web dashboard for monitoring and visualizing
the autonomous AI improvement system. It offers real-time insights into:
- Live performance metrics and system health
- Autonomous improvement cycles and actions
- Knowledge extraction and graph evolution
- Skill versioning and capability tracking
- Cross-phase synthesis and global insights

The dashboard integrates with all 7 phases of the autonomous system to provide
a unified view of the AI's self-improvement capabilities.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """Real-time metrics for dashboard display"""

    timestamp: str
    system_health: float
    active_improvements: int
    total_skills: int
    knowledge_items: int
    performance_baseline: float
    recent_improvements: List[Dict[str, Any]]
    system_load: Dict[str, float]
    autonomous_cycles: int
    evaluation_score: float


@dataclass
class ImprovementEvent:
    """Real-time improvement event for dashboard streaming"""

    timestamp: str
    phase: str
    action: str
    impact: float
    details: Dict[str, Any]
    status: str


class DashboardDataCollector:
    """Collects real-time data from all autonomous AI phases"""

    def __init__(self):
        self.base_path = Path("/home/ncacord/Vega2.0")
        self.db_connections = {}
        self.cache = {}
        self.cache_expiry = {}
        self.cache_ttl = 30  # 30 seconds cache TTL

    def _get_db_connection(self, db_name: str) -> sqlite3.Connection:
        """Get cached database connection"""
        if db_name not in self.db_connections:
            db_path = self.base_path / f"{db_name}.db"
            if db_path.exists():
                self.db_connections[db_name] = sqlite3.connect(str(db_path))
                self.db_connections[db_name].row_factory = sqlite3.Row
        return self.db_connections.get(db_name)

    def _get_cached_data(self, key: str, fetch_func):
        """Get data with caching to avoid excessive DB queries"""
        now = datetime.now()
        if key in self.cache and key in self.cache_expiry:
            if now < self.cache_expiry[key]:
                return self.cache[key]

        # Cache expired or doesn't exist, fetch new data
        try:
            data = fetch_func()
            self.cache[key] = data
            self.cache_expiry[key] = now + timedelta(seconds=self.cache_ttl)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {key}: {e}")
            return self.cache.get(key, {})

    def get_system_health(self) -> float:
        """Calculate overall system health score"""

        def fetch_health():
            health_factors = []

            # Check telemetry system health
            conn = self._get_db_connection("telemetry")
            if conn:
                cursor = conn.execute(
                    """
                    SELECT AVG(cpu_percent), AVG(memory_percent) 
                    FROM system_metrics 
                    WHERE timestamp > datetime('now', '-5 minutes')
                """
                )
                row = cursor.fetchone()
                if row and row[0] is not None:
                    cpu_health = max(
                        0, 1.0 - (row[0] / 100.0)
                    )  # Lower CPU = better health
                    memory_health = max(
                        0, 1.0 - (row[1] / 100.0)
                    )  # Lower memory = better health
                    health_factors.extend([cpu_health, memory_health])

            # Check improvement system activity
            conn = self._get_db_connection("self_improvement")
            if conn:
                cursor = conn.execute(
                    """
                    SELECT COUNT(*) FROM improvement_cycles 
                    WHERE timestamp > datetime('now', '-10 minutes')
                """
                )
                row = cursor.fetchone()
                if row and row[0] > 0:
                    health_factors.append(1.0)  # Active improvements = good health
                else:
                    health_factors.append(
                        0.7
                    )  # No recent improvements = moderate health

            # Check evaluation scores
            conn = self._get_db_connection("evaluations")
            if conn:
                cursor = conn.execute(
                    """
                    SELECT AVG(overall_score) FROM conversation_evaluations 
                    WHERE timestamp > datetime('now', '-1 hour')
                """
                )
                row = cursor.fetchone()
                if row and row[0] is not None:
                    health_factors.append(row[0])

            return sum(health_factors) / len(health_factors) if health_factors else 0.5

        return self._get_cached_data("system_health", fetch_health)

    def get_active_improvements(self) -> int:
        """Get count of currently active improvement processes"""

        def fetch_improvements():
            conn = self._get_db_connection("self_improvement")
            if not conn:
                return 0

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM improvement_actions 
                WHERE status = 'active' AND timestamp > datetime('now', '-1 hour')
            """
            )
            row = cursor.fetchone()
            return row[0] if row else 0

        return self._get_cached_data("active_improvements", fetch_improvements)

    def get_total_skills(self) -> int:
        """Get total number of registered skills"""

        def fetch_skills():
            conn = self._get_db_connection("skill_registry")
            if not conn:
                return 0

            cursor = conn.execute("SELECT COUNT(*) FROM skills")
            row = cursor.fetchone()
            return row[0] if row else 0

        return self._get_cached_data("total_skills", fetch_skills)

    def get_knowledge_items(self) -> int:
        """Get total number of knowledge items in graph"""

        def fetch_knowledge():
            conn = self._get_db_connection("knowledge_graph")
            if not conn:
                return 0

            cursor = conn.execute("SELECT COUNT(*) FROM knowledge_items")
            row = cursor.fetchone()
            return row[0] if row else 0

        return self._get_cached_data("knowledge_items", fetch_knowledge)

    def get_performance_baseline(self) -> float:
        """Get current performance baseline"""

        def fetch_baseline():
            conn = self._get_db_connection("variants")
            if not conn:
                return 0.0

            cursor = conn.execute(
                """
                SELECT AVG(performance_score) FROM variant_tests 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC LIMIT 10
            """
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else 0.0

        return self._get_cached_data("performance_baseline", fetch_baseline)

    def get_recent_improvements(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent improvement actions"""

        def fetch_improvements():
            conn = self._get_db_connection("self_improvement")
            if not conn:
                return []

            cursor = conn.execute(
                """
                SELECT phase, action_type, description, impact_score, timestamp, status
                FROM improvement_actions 
                ORDER BY timestamp DESC LIMIT ?
            """,
                (limit,),
            )

            improvements = []
            for row in cursor.fetchall():
                improvements.append(
                    {
                        "phase": row[0],
                        "action": row[1],
                        "description": row[2],
                        "impact": row[3],
                        "timestamp": row[4],
                        "status": row[5],
                    }
                )
            return improvements

        return self._get_cached_data("recent_improvements", fetch_improvements)

    def get_system_load(self) -> Dict[str, float]:
        """Get current system resource usage"""

        def fetch_load():
            conn = self._get_db_connection("telemetry")
            if not conn:
                return {"cpu": 0.0, "memory": 0.0, "disk": 0.0}

            cursor = conn.execute(
                """
                SELECT cpu_percent, memory_percent, disk_percent 
                FROM system_metrics 
                ORDER BY timestamp DESC LIMIT 1
            """
            )
            row = cursor.fetchone()

            if row:
                return {
                    "cpu": row[0] or 0.0,
                    "memory": row[1] or 0.0,
                    "disk": row[2] or 0.0,
                }
            return {"cpu": 0.0, "memory": 0.0, "disk": 0.0}

        return self._get_cached_data("system_load", fetch_load)

    def get_autonomous_cycles(self) -> int:
        """Get count of autonomous improvement cycles"""

        def fetch_cycles():
            conn = self._get_db_connection("self_improvement")
            if not conn:
                return 0

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM improvement_cycles 
                WHERE timestamp > datetime('now', '-24 hours')
            """
            )
            row = cursor.fetchone()
            return row[0] if row else 0

        return self._get_cached_data("autonomous_cycles", fetch_cycles)

    def get_evaluation_score(self) -> float:
        """Get recent average evaluation score"""

        def fetch_score():
            conn = self._get_db_connection("evaluations")
            if not conn:
                return 0.0

            cursor = conn.execute(
                """
                SELECT AVG(overall_score) FROM conversation_evaluations 
                WHERE timestamp > datetime('now', '-1 hour')
            """
            )
            row = cursor.fetchone()
            return row[0] if row and row[0] else 0.0

        return self._get_cached_data("evaluation_score", fetch_score)

    async def get_real_time_metrics(self) -> DashboardMetrics:
        """Collect all real-time metrics for dashboard"""
        return DashboardMetrics(
            timestamp=datetime.now().isoformat(),
            system_health=self.get_system_health(),
            active_improvements=self.get_active_improvements(),
            total_skills=self.get_total_skills(),
            knowledge_items=self.get_knowledge_items(),
            performance_baseline=self.get_performance_baseline(),
            recent_improvements=self.get_recent_improvements(),
            system_load=self.get_system_load(),
            autonomous_cycles=self.get_autonomous_cycles(),
            evaluation_score=self.get_evaluation_score(),
        )

    def get_improvement_timeline(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get improvement timeline for visualization"""
        conn = self._get_db_connection("self_improvement")
        if not conn:
            return []

        cursor = conn.execute(
            """
            SELECT timestamp, phase, action_type, impact_score, description
            FROM improvement_actions 
            WHERE timestamp > datetime('now', '-{} hours')
            ORDER BY timestamp DESC
        """.format(
                hours
            )
        )

        timeline = []
        for row in cursor.fetchall():
            timeline.append(
                {
                    "timestamp": row[0],
                    "phase": row[1],
                    "action": row[2],
                    "impact": row[3],
                    "description": row[4],
                }
            )

        return timeline

    def get_skill_evolution(self) -> Dict[str, Any]:
        """Get skill version history and evolution"""
        conn = self._get_db_connection("skill_registry")
        if not conn:
            return {}

        cursor = conn.execute(
            """
            SELECT skill_name, version, performance_score, timestamp
            FROM skill_versions 
            ORDER BY skill_name, timestamp DESC
        """
        )

        skills = defaultdict(list)
        for row in cursor.fetchall():
            skills[row[0]].append(
                {"version": row[1], "performance": row[2], "timestamp": row[3]}
            )

        return dict(skills)

    def get_knowledge_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        conn = self._get_db_connection("knowledge_graph")
        if not conn:
            return {}

        # Get knowledge counts by type
        cursor = conn.execute(
            """
            SELECT knowledge_type, COUNT(*) 
            FROM knowledge_items 
            GROUP BY knowledge_type
        """
        )

        type_counts = {}
        for row in cursor.fetchall():
            type_counts[row[0]] = row[1]

        # Get recent extractions
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM knowledge_items 
            WHERE timestamp > datetime('now', '-24 hours')
        """
        )
        recent_count = cursor.fetchone()[0] if cursor.fetchone() else 0

        # Get relationship count
        cursor = conn.execute("SELECT COUNT(*) FROM knowledge_relationships")
        relationship_count = cursor.fetchone()[0] if cursor.fetchone() else 0

        return {
            "type_counts": type_counts,
            "recent_extractions": recent_count,
            "total_relationships": relationship_count,
            "graph_density": relationship_count / max(sum(type_counts.values()), 1),
        }

    def close_connections(self):
        """Close all database connections"""
        for conn in self.db_connections.values():
            if conn:
                conn.close()
        self.db_connections.clear()


class DashboardEventStream:
    """Real-time event streaming for dashboard updates"""

    def __init__(self):
        self.subscribers = set()
        self.collector = DashboardDataCollector()
        self.running = False

    async def subscribe(self, websocket):
        """Subscribe a WebSocket connection to real-time updates"""
        self.subscribers.add(websocket)
        logger.info(f"Dashboard subscriber added. Total: {len(self.subscribers)}")

    async def unsubscribe(self, websocket):
        """Unsubscribe a WebSocket connection"""
        self.subscribers.discard(websocket)
        logger.info(f"Dashboard subscriber removed. Total: {len(self.subscribers)}")

    async def broadcast_metrics(self):
        """Broadcast current metrics to all subscribers"""
        if not self.subscribers:
            return

        try:
            metrics = await self.collector.get_real_time_metrics()
            message = {"type": "metrics_update", "data": asdict(metrics)}

            # Send to all subscribers
            disconnected = set()
            for websocket in self.subscribers:
                try:
                    await websocket.send(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Failed to send to subscriber: {e}")
                    disconnected.add(websocket)

            # Remove disconnected subscribers
            for websocket in disconnected:
                self.subscribers.discard(websocket)

        except Exception as e:
            logger.error(f"Error broadcasting metrics: {e}")

    async def broadcast_improvement_event(self, event: ImprovementEvent):
        """Broadcast improvement event to all subscribers"""
        if not self.subscribers:
            return

        message = {"type": "improvement_event", "data": asdict(event)}

        disconnected = set()
        for websocket in self.subscribers:
            try:
                await websocket.send(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send event to subscriber: {e}")
                disconnected.add(websocket)

        # Remove disconnected subscribers
        for websocket in disconnected:
            self.subscribers.discard(websocket)

    async def start_streaming(self):
        """Start real-time metric streaming"""
        self.running = True
        logger.info("Dashboard event streaming started")

        while self.running:
            await self.broadcast_metrics()
            await asyncio.sleep(5)  # Update every 5 seconds

    def stop_streaming(self):
        """Stop real-time streaming"""
        self.running = False
        self.collector.close_connections()
        logger.info("Dashboard event streaming stopped")


# Global dashboard instance
dashboard_stream = DashboardEventStream()


async def trigger_manual_improvement():
    """Manually trigger an improvement cycle for testing"""
    try:
        # Import the global orchestrator
        from global_self_improvement import global_orchestrator

        if global_orchestrator:
            await global_orchestrator.run_improvement_cycle()

            # Broadcast improvement event
            event = ImprovementEvent(
                timestamp=datetime.now().isoformat(),
                phase="Manual",
                action="manual_trigger",
                impact=1.0,
                details={"source": "dashboard_manual_trigger"},
                status="completed",
            )
            await dashboard_stream.broadcast_improvement_event(event)

            return {
                "status": "success",
                "message": "Manual improvement cycle triggered",
            }
        else:
            return {"status": "error", "message": "Global orchestrator not available"}

    except Exception as e:
        logger.error(f"Error triggering manual improvement: {e}")
        return {"status": "error", "message": str(e)}


def get_dashboard_data_collector() -> DashboardDataCollector:
    """Get dashboard data collector instance"""
    return DashboardDataCollector()


def get_dashboard_event_stream() -> DashboardEventStream:
    """Get dashboard event stream instance"""
    return dashboard_stream
