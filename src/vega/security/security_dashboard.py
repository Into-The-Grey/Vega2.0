#!/usr/bin/env python3
"""
security_dashboard.py - Security Monitoring Dashboard

Real-time security monitoring and alerting for Vega 2.0 platform including:
- Security metrics collection and aggregation
- Real-time threat monitoring
- Automated incident response
- Security KPI tracking
- Alert management and escalation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import json
import logging
import asyncio
import sqlite3
from pathlib import Path
import time


class AlertSeverity(Enum):
    """Security alert severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Security alert status"""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class SecurityAlert:
    """Security alert record"""

    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    source: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.OPEN
    assigned_to: Optional[str] = None
    escalated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityMetrics:
    """Security metrics snapshot"""

    timestamp: datetime
    vulnerabilities_total: int
    vulnerabilities_critical: int
    vulnerabilities_high: int
    failed_logins_24h: int
    active_sessions: int
    security_events_24h: int
    compliance_score: float
    threat_level: str
    system_uptime: float


class SecurityDashboard:
    """Comprehensive security monitoring dashboard"""

    def __init__(self, db_path: str = "security/security_dashboard.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("security_dashboard")

        # Alert handlers
        self.alert_handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }

        # Monitoring state
        self.monitoring_active = False
        self.last_metrics_update = None

        # Initialize database
        self._init_database()

        # Register default alert handlers
        self._register_default_handlers()

    def _init_database(self) -> None:
        """Initialize security dashboard database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS security_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                severity TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                assigned_to TEXT,
                escalated BOOLEAN DEFAULT FALSE,
                metadata TEXT,
                response_actions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS security_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP NOT NULL,
                vulnerabilities_total INTEGER,
                vulnerabilities_critical INTEGER,
                vulnerabilities_high INTEGER,
                failed_logins_24h INTEGER,
                active_sessions INTEGER,
                security_events_24h INTEGER,
                compliance_score REAL,
                threat_level TEXT,
                system_uptime REAL,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id TEXT UNIQUE NOT NULL,
                alert_id TEXT,
                response_type TEXT NOT NULL,
                action_taken TEXT,
                timestamp TIMESTAMP NOT NULL,
                success BOOLEAN,
                metadata TEXT
            )
        """
        )

        conn.commit()
        conn.close()

    def _register_default_handlers(self) -> None:
        """Register default alert handlers"""
        # Critical alert handler
        self.register_alert_handler(AlertSeverity.CRITICAL, self._handle_critical_alert)

        # High severity handler
        self.register_alert_handler(AlertSeverity.HIGH, self._handle_high_alert)

        # General logging handler
        for severity in AlertSeverity:
            self.register_alert_handler(severity, self._log_alert)

    def register_alert_handler(
        self, severity: AlertSeverity, handler: Callable[[SecurityAlert], None]
    ) -> None:
        """Register alert handler for specific severity"""
        self.alert_handlers[severity].append(handler)

    async def start_monitoring(self) -> None:
        """Start real-time security monitoring"""
        self.monitoring_active = True
        self.logger.info("Starting security monitoring")

        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._monitor_vulnerabilities()),
            asyncio.create_task(self._monitor_system_health()),
            asyncio.create_task(self._monitor_compliance()),
            asyncio.create_task(self._process_alerts()),
            asyncio.create_task(self._update_metrics()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring_active = False

    def stop_monitoring(self) -> None:
        """Stop security monitoring"""
        self.monitoring_active = False
        self.logger.info("Security monitoring stopped")

    async def _monitor_vulnerabilities(self) -> None:
        """Monitor for new vulnerabilities"""
        while self.monitoring_active:
            try:
                # Check for new critical vulnerabilities
                critical_vulns = await self._check_critical_vulnerabilities()

                if critical_vulns > 0:
                    alert = SecurityAlert(
                        alert_id=f"vuln-critical-{int(time.time())}",
                        title=f"Critical Vulnerabilities Detected",
                        description=f"{critical_vulns} critical vulnerabilities require immediate attention",
                        severity=AlertSeverity.CRITICAL,
                        source="vulnerability_monitor",
                        timestamp=datetime.now(),
                        metadata={"vulnerability_count": critical_vulns},
                    )
                    await self._trigger_alert(alert)

                # Check for aging vulnerabilities
                aging_vulns = await self._check_aging_vulnerabilities()
                if aging_vulns > 0:
                    alert = SecurityAlert(
                        alert_id=f"vuln-aging-{int(time.time())}",
                        title=f"Aging Vulnerabilities",
                        description=f"{aging_vulns} vulnerabilities are overdue for remediation",
                        severity=AlertSeverity.HIGH,
                        source="vulnerability_monitor",
                        timestamp=datetime.now(),
                        metadata={"aging_vulnerability_count": aging_vulns},
                    )
                    await self._trigger_alert(alert)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Vulnerability monitoring error: {e}")
                await asyncio.sleep(60)

    async def _monitor_system_health(self) -> None:
        """Monitor system health and security indicators"""
        while self.monitoring_active:
            try:
                # Check system uptime
                uptime = await self._get_system_uptime()

                # Check for unusual activity
                failed_logins = await self._get_failed_login_count()
                if failed_logins > 50:  # Threshold for failed logins
                    alert = SecurityAlert(
                        alert_id=f"auth-fail-{int(time.time())}",
                        title="Excessive Failed Login Attempts",
                        description=f"{failed_logins} failed login attempts in the last 24 hours",
                        severity=AlertSeverity.HIGH,
                        source="auth_monitor",
                        timestamp=datetime.now(),
                        metadata={"failed_login_count": failed_logins},
                    )
                    await self._trigger_alert(alert)

                # Check active sessions
                active_sessions = await self._get_active_session_count()
                if active_sessions > 1000:  # Threshold for concurrent sessions
                    alert = SecurityAlert(
                        alert_id=f"session-high-{int(time.time())}",
                        title="High Number of Active Sessions",
                        description=f"{active_sessions} active sessions detected",
                        severity=AlertSeverity.MEDIUM,
                        source="session_monitor",
                        timestamp=datetime.now(),
                        metadata={"active_session_count": active_sessions},
                    )
                    await self._trigger_alert(alert)

                await asyncio.sleep(180)  # Check every 3 minutes

            except Exception as e:
                self.logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(60)

    async def _monitor_compliance(self) -> None:
        """Monitor compliance status"""
        while self.monitoring_active:
            try:
                # Check compliance score
                compliance_score = await self._get_compliance_score()

                if compliance_score < 0.8:  # Below 80% compliance
                    alert = SecurityAlert(
                        alert_id=f"compliance-low-{int(time.time())}",
                        title="Low Compliance Score",
                        description=f"Compliance score dropped to {compliance_score:.1%}",
                        severity=AlertSeverity.HIGH,
                        source="compliance_monitor",
                        timestamp=datetime.now(),
                        metadata={"compliance_score": compliance_score},
                    )
                    await self._trigger_alert(alert)

                await asyncio.sleep(3600)  # Check every hour

            except Exception as e:
                self.logger.error(f"Compliance monitoring error: {e}")
                await asyncio.sleep(300)

    async def _process_alerts(self) -> None:
        """Process and escalate alerts"""
        while self.monitoring_active:
            try:
                # Check for unacknowledged critical alerts
                critical_alerts = self._get_unacknowledged_alerts(
                    AlertSeverity.CRITICAL
                )

                for alert in critical_alerts:
                    if not alert.escalated:
                        # Escalate after 15 minutes
                        if datetime.now() - alert.timestamp > timedelta(minutes=15):
                            await self._escalate_alert(alert)

                # Check for aging high-severity alerts
                high_alerts = self._get_unacknowledged_alerts(AlertSeverity.HIGH)

                for alert in high_alerts:
                    if not alert.escalated:
                        # Escalate after 1 hour
                        if datetime.now() - alert.timestamp > timedelta(hours=1):
                            await self._escalate_alert(alert)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)

    async def _update_metrics(self) -> None:
        """Update security metrics"""
        while self.monitoring_active:
            try:
                metrics = SecurityMetrics(
                    timestamp=datetime.now(),
                    vulnerabilities_total=await self._get_vulnerability_count(),
                    vulnerabilities_critical=await self._check_critical_vulnerabilities(),
                    vulnerabilities_high=await self._get_high_vulnerability_count(),
                    failed_logins_24h=await self._get_failed_login_count(),
                    active_sessions=await self._get_active_session_count(),
                    security_events_24h=await self._get_security_event_count(),
                    compliance_score=await self._get_compliance_score(),
                    threat_level=await self._assess_threat_level(),
                    system_uptime=await self._get_system_uptime(),
                )

                self._store_metrics(metrics)
                self.last_metrics_update = datetime.now()

                await asyncio.sleep(600)  # Update every 10 minutes

            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(60)

    async def _trigger_alert(self, alert: SecurityAlert) -> None:
        """Trigger security alert and execute handlers"""
        self.logger.warning(f"Security Alert: {alert.title} [{alert.severity.value}]")

        # Store alert
        self._store_alert(alert)

        # Execute handlers
        handlers = self.alert_handlers.get(alert.severity, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")

    async def _escalate_alert(self, alert: SecurityAlert) -> None:
        """Escalate unresolved alert"""
        self.logger.critical(f"ESCALATED: {alert.title}")

        # Mark as escalated
        alert.escalated = True
        self._update_alert(alert)

        # Trigger escalation notifications
        escalation_alert = SecurityAlert(
            alert_id=f"escalation-{alert.alert_id}",
            title=f"ESCALATED: {alert.title}",
            description=f"Alert {alert.alert_id} has been escalated due to lack of response",
            severity=AlertSeverity.CRITICAL,
            source="escalation_system",
            timestamp=datetime.now(),
            metadata={"original_alert_id": alert.alert_id},
        )

        await self._trigger_alert(escalation_alert)

    def _handle_critical_alert(self, alert: SecurityAlert) -> None:
        """Handle critical security alerts"""
        self.logger.critical(f"CRITICAL ALERT: {alert.title}")

        # Implement automatic response actions
        if "vulnerability" in alert.source:
            self._initiate_vulnerability_response(alert)
        elif "auth" in alert.source:
            self._initiate_auth_security_response(alert)
        elif "compliance" in alert.source:
            self._initiate_compliance_response(alert)

    def _handle_high_alert(self, alert: SecurityAlert) -> None:
        """Handle high-severity security alerts"""
        self.logger.error(f"HIGH ALERT: {alert.title}")

        # Schedule response actions
        self._schedule_response_action(alert, "security_review")

    def _log_alert(self, alert: SecurityAlert) -> None:
        """Log security alert"""
        log_level = {
            AlertSeverity.CRITICAL: logging.CRITICAL,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.INFO: logging.INFO,
        }

        self.logger.log(
            log_level[alert.severity],
            f"Security Alert [{alert.severity.value}]: {alert.title} - {alert.description}",
        )

    def _initiate_vulnerability_response(self, alert: SecurityAlert) -> None:
        """Initiate automated vulnerability response"""
        actions = [
            "vulnerability_scan_triggered",
            "security_team_notified",
            "patch_assessment_initiated",
        ]

        alert.response_actions.extend(actions)
        self._update_alert(alert)

        # Log incident response
        self._log_incident_response(alert.alert_id, "vulnerability_response", actions)

    def _initiate_auth_security_response(self, alert: SecurityAlert) -> None:
        """Initiate authentication security response"""
        actions = [
            "ip_blocking_review",
            "account_security_audit",
            "security_monitoring_enhanced",
        ]

        alert.response_actions.extend(actions)
        self._update_alert(alert)

        self._log_incident_response(alert.alert_id, "auth_security_response", actions)

    def _initiate_compliance_response(self, alert: SecurityAlert) -> None:
        """Initiate compliance response"""
        actions = [
            "compliance_audit_triggered",
            "control_assessment_scheduled",
            "remediation_plan_required",
        ]

        alert.response_actions.extend(actions)
        self._update_alert(alert)

        self._log_incident_response(alert.alert_id, "compliance_response", actions)

    def _schedule_response_action(self, alert: SecurityAlert, action_type: str) -> None:
        """Schedule response action for alert"""
        alert.response_actions.append(f"scheduled_{action_type}")
        self._update_alert(alert)

    # Mock data collection methods (would integrate with real systems)
    async def _check_critical_vulnerabilities(self) -> int:
        """Check for critical vulnerabilities"""
        try:
            from .vulnerability_manager import VulnerabilityManager

            manager = VulnerabilityManager()
            vulns = manager.get_vulnerabilities()
            return len([v for v in vulns if v.severity.value == "critical"])
        except:
            return 0

    async def _check_aging_vulnerabilities(self) -> int:
        """Check for aging vulnerabilities"""
        try:
            from .vulnerability_manager import VulnerabilityManager

            manager = VulnerabilityManager()
            vulns = manager.get_vulnerabilities()
            # Count vulnerabilities older than 30 days
            cutoff = datetime.now() - timedelta(days=30)
            return len([v for v in vulns if v.discovered_date < cutoff])
        except:
            return 0

    async def _get_vulnerability_count(self) -> int:
        """Get total vulnerability count"""
        try:
            from .vulnerability_manager import VulnerabilityManager

            manager = VulnerabilityManager()
            return len(manager.get_vulnerabilities())
        except:
            return 0

    async def _get_high_vulnerability_count(self) -> int:
        """Get high-severity vulnerability count"""
        try:
            from .vulnerability_manager import VulnerabilityManager

            manager = VulnerabilityManager()
            vulns = manager.get_vulnerabilities()
            return len([v for v in vulns if v.severity.value == "high"])
        except:
            return 0

    async def _get_failed_login_count(self) -> int:
        """Get failed login count in last 24 hours"""
        # Mock implementation - would integrate with auth logs
        return 5

    async def _get_active_session_count(self) -> int:
        """Get active session count"""
        # Mock implementation - would integrate with session store
        return 42

    async def _get_security_event_count(self) -> int:
        """Get security event count in last 24 hours"""
        # Mock implementation - would integrate with SIEM
        return 128

    async def _get_compliance_score(self) -> float:
        """Get current compliance score"""
        try:
            from .compliance_reporter import ComplianceReporter, ComplianceFramework

            reporter = ComplianceReporter()
            assessment = reporter.conduct_compliance_assessment(
                ComplianceFramework.SOC2, "Automated"
            )
            return max(0.0, 1.0 - (assessment.risk_score / 10.0))
        except:
            return 0.85  # Default score

    async def _assess_threat_level(self) -> str:
        """Assess current threat level"""
        critical_vulns = await self._check_critical_vulnerabilities()
        failed_logins = await self._get_failed_login_count()

        if critical_vulns > 5 or failed_logins > 100:
            return "HIGH"
        elif critical_vulns > 0 or failed_logins > 50:
            return "MEDIUM"
        else:
            return "LOW"

    async def _get_system_uptime(self) -> float:
        """Get system uptime percentage"""
        # Mock implementation - would integrate with monitoring
        return 99.9

    # Database operations
    def _store_alert(self, alert: SecurityAlert) -> None:
        """Store security alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO security_alerts 
            (alert_id, title, description, severity, source, timestamp, status,
             assigned_to, escalated, metadata, response_actions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                alert.alert_id,
                alert.title,
                alert.description,
                alert.severity.value,
                alert.source,
                alert.timestamp,
                alert.status.value,
                alert.assigned_to,
                alert.escalated,
                json.dumps(alert.metadata),
                json.dumps(alert.response_actions),
            ),
        )

        conn.commit()
        conn.close()

    def _update_alert(self, alert: SecurityAlert) -> None:
        """Update security alert in database"""
        self._store_alert(alert)

    def _store_metrics(self, metrics: SecurityMetrics) -> None:
        """Store security metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO security_metrics 
            (timestamp, vulnerabilities_total, vulnerabilities_critical, vulnerabilities_high,
             failed_logins_24h, active_sessions, security_events_24h, compliance_score,
             threat_level, system_uptime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metrics.timestamp,
                metrics.vulnerabilities_total,
                metrics.vulnerabilities_critical,
                metrics.vulnerabilities_high,
                metrics.failed_logins_24h,
                metrics.active_sessions,
                metrics.security_events_24h,
                metrics.compliance_score,
                metrics.threat_level,
                metrics.system_uptime,
            ),
        )

        conn.commit()
        conn.close()

    def _get_unacknowledged_alerts(
        self, severity: AlertSeverity
    ) -> List[SecurityAlert]:
        """Get unacknowledged alerts of specific severity"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM security_alerts 
            WHERE severity = ? AND status = 'open'
            ORDER BY timestamp ASC
        """,
            (severity.value,),
        )

        rows = cursor.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            alert = SecurityAlert(
                alert_id=row[1],
                title=row[2],
                description=row[3],
                severity=AlertSeverity(row[4]),
                source=row[5],
                timestamp=datetime.fromisoformat(row[6]),
                status=AlertStatus(row[7]),
                assigned_to=row[8],
                escalated=bool(row[9]),
                metadata=json.loads(row[10]) if row[10] else {},
                response_actions=json.loads(row[11]) if row[11] else [],
            )
            alerts.append(alert)

        return alerts

    def _log_incident_response(
        self, alert_id: str, response_type: str, actions: List[str]
    ) -> None:
        """Log incident response actions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        incident_id = f"incident-{int(time.time())}"

        cursor.execute(
            """
            INSERT INTO incident_responses 
            (incident_id, alert_id, response_type, action_taken, timestamp, success, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                incident_id,
                alert_id,
                response_type,
                json.dumps(actions),
                datetime.now(),
                True,
                json.dumps({"automated": True}),
            ),
        )

        conn.commit()
        conn.close()

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        # Get latest metrics
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM security_metrics 
            ORDER BY timestamp DESC LIMIT 1
        """
        )
        metrics_row = cursor.fetchone()

        # Get recent alerts
        cursor.execute(
            """
            SELECT COUNT(*) FROM security_alerts 
            WHERE timestamp > datetime('now', '-24 hours')
        """
        )
        recent_alerts = cursor.fetchone()[0]

        # Get open alerts by severity
        cursor.execute(
            """
            SELECT severity, COUNT(*) FROM security_alerts 
            WHERE status = 'open'
            GROUP BY severity
        """
        )
        open_alerts = dict(cursor.fetchall())

        conn.close()

        dashboard_data = {
            "last_updated": (
                self.last_metrics_update.isoformat()
                if self.last_metrics_update
                else None
            ),
            "monitoring_active": self.monitoring_active,
            "recent_alerts_24h": recent_alerts,
            "open_alerts": open_alerts,
            "metrics": {},
        }

        if metrics_row:
            dashboard_data["metrics"] = {
                "vulnerabilities_total": metrics_row[2],
                "vulnerabilities_critical": metrics_row[3],
                "vulnerabilities_high": metrics_row[4],
                "failed_logins_24h": metrics_row[5],
                "active_sessions": metrics_row[6],
                "security_events_24h": metrics_row[7],
                "compliance_score": metrics_row[8],
                "threat_level": metrics_row[9],
                "system_uptime": metrics_row[10],
            }

        return dashboard_data


async def main():
    """Main function for running security dashboard"""
    import argparse

    parser = argparse.ArgumentParser(description="Vega 2.0 Security Dashboard")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring")
    parser.add_argument("--status", action="store_true", help="Show dashboard status")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Monitoring duration in seconds (0 = infinite)",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dashboard = SecurityDashboard()

    if args.status:
        data = dashboard.get_dashboard_data()
        print("Security Dashboard Status:")
        print(f"  Monitoring Active: {data['monitoring_active']}")
        print(f"  Last Updated: {data['last_updated']}")
        print(f"  Recent Alerts (24h): {data['recent_alerts_24h']}")
        print(f"  Open Alerts: {data['open_alerts']}")
        if data["metrics"]:
            print("  Current Metrics:")
            for key, value in data["metrics"].items():
                print(f"    {key}: {value}")

    if args.monitor:
        print("Starting security monitoring...")
        if args.duration > 0:
            print(f"Monitoring for {args.duration} seconds")
            try:
                await asyncio.wait_for(
                    dashboard.start_monitoring(), timeout=args.duration
                )
            except asyncio.TimeoutError:
                print("Monitoring duration completed")
                dashboard.stop_monitoring()
        else:
            print("Monitoring indefinitely (Ctrl+C to stop)")
            try:
                await dashboard.start_monitoring()
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                dashboard.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())
