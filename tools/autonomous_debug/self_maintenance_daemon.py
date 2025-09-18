#!/usr/bin/env python3
"""
Self-Maintenance Daemon + Automation
===================================

Autonomous hourly error resolution cycles, daily health reports, 
and proactive system monitoring. Orchestrates the entire 
autonomous debugging pipeline.

Features:
- Autonomous hourly error resolution cycles
- Daily system health reports
- Proactive monitoring and alerting
- Integration with all debugging components
- Configurable automation policies
- Performance tracking and optimization
- Failure recovery and circuit breakers
- Email/webhook notifications
"""

import os
import sys
import asyncio
import signal
import json
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
import threading
import time

# Try to import optional dependencies
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    logger.warning("Schedule module not available - daemon scheduling disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Local imports
from error_tracker import ErrorDatabase, ErrorRecord
from self_debugger import SelfDebugger
from error_web_resolver import WebErrorResolver
from code_sandbox import SandboxValidator
from patch_manager import PatchManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_debug/logs/daemon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class AutomationConfig:
    """Configuration for automation policies"""
    enabled: bool = True
    hourly_enabled: bool = True
    daily_reports: bool = True
    max_fixes_per_hour: int = 5
    max_fixes_per_day: int = 20
    confidence_threshold: float = 0.7
    safety_threshold: float = 0.8
    auto_apply_enabled: bool = False  # Start with manual review
    notification_enabled: bool = True
    email_recipients: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    working_hours_only: bool = True
    working_hours_start: int = 9  # 9 AM
    working_hours_end: int = 17   # 5 PM

@dataclass
class DaemonStats:
    """Statistics for daemon operations"""
    started_at: datetime
    total_cycles: int = 0
    errors_processed: int = 0
    fixes_applied: int = 0
    fixes_rejected: int = 0
    rollbacks_performed: int = 0
    notifications_sent: int = 0
    last_cycle_time: Optional[datetime] = None
    last_health_report: Optional[datetime] = None
    uptime_seconds: float = 0.0

class NotificationManager:
    """Manages notifications and alerts"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.email_config = self._load_email_config()
    
    def _load_email_config(self) -> Dict[str, str]:
        """Load email configuration from environment or config file"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_user': os.getenv('SMTP_USER', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('FROM_EMAIL', '')
        }
    
    async def send_notification(self, subject: str, message: str, priority: str = "normal"):
        """Send notification via configured channels"""
        try:
            if not self.config.notification_enabled:
                return
            
            # Send email notification
            if self.config.email_recipients and self.email_config.get('smtp_user'):
                await self._send_email(subject, message)
            
            # Send webhook notification
            if self.config.webhook_url:
                await self._send_webhook(subject, message, priority)
            
            logger.info(f"Sent notification: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    async def _send_email(self, subject: str, message: str):
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[Vega2.0 AutoDebug] {subject}"
            
            body = f"""
Vega2.0 Autonomous Debugging System

{message}

---
Timestamp: {datetime.now().isoformat()}
System: Autonomous Debug Daemon
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['smtp_user'], self.email_config['smtp_password'])
            text = msg.as_string()
            server.sendmail(self.email_config['from_email'], self.config.email_recipients, text)
            server.quit()
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    async def _send_webhook(self, subject: str, message: str, priority: str):
        """Send webhook notification"""
        try:
            payload = {
                'subject': subject,
                'message': message,
                'priority': priority,
                'timestamp': datetime.now().isoformat(),
                'system': 'vega2.0-autodebug'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.config.webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Webhook failed with status {response.status}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

class HealthMonitor:
    """Monitors system health and performance"""
    
    def __init__(self):
        self.metrics = {
            'error_rate': 0.0,
            'fix_success_rate': 0.0,
            'average_resolution_time': 0.0,
            'system_load': 0.0,
            'disk_usage': 0.0,
            'memory_usage': 0.0
        }
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # Database metrics
            error_db = ErrorDatabase()
            stats = error_db.get_error_statistics()
            
            self.metrics['error_rate'] = stats.get('unresolved_errors', 0)
            self.metrics['fix_success_rate'] = stats.get('resolution_rate', 0)
            
            error_db.close()
            
            # System metrics
            self.metrics['system_load'] = await self._get_system_load()
            self.metrics['disk_usage'] = await self._get_disk_usage()
            self.metrics['memory_usage'] = await self._get_memory_usage()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return self.metrics
    
    async def _get_system_load(self) -> float:
        """Get system load average"""
        try:
            if hasattr(os, 'getloadavg'):
                return os.getloadavg()[0]  # 1-minute average
            return 0.0
        except:
            return 0.0
    
    async def _get_disk_usage(self) -> float:
        """Get disk usage percentage"""
        try:
            stat = os.statvfs('.')
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_available * stat.f_frsize
            used = total - free
            return (used / total) * 100 if total > 0 else 0.0
        except:
            return 0.0
    
    async def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            # Simple memory check using /proc/meminfo on Linux
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    lines = f.readlines()
                
                mem_total = 0
                mem_available = 0
                
                for line in lines:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1])
                    elif line.startswith('MemAvailable:'):
                        mem_available = int(line.split()[1])
                
                if mem_total > 0:
                    used = mem_total - mem_available
                    return (used / mem_total) * 100
            
            return 0.0
        except:
            return 0.0
    
    def assess_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system health"""
        health_score = 100.0
        issues = []
        warnings = []
        
        # Error rate assessment
        if metrics['error_rate'] > 50:
            health_score -= 30
            issues.append(f"High error rate: {metrics['error_rate']} unresolved errors")
        elif metrics['error_rate'] > 20:
            health_score -= 15
            warnings.append(f"Elevated error rate: {metrics['error_rate']} unresolved errors")
        
        # Fix success rate assessment
        if metrics['fix_success_rate'] < 50:
            health_score -= 25
            issues.append(f"Low fix success rate: {metrics['fix_success_rate']:.1f}%")
        elif metrics['fix_success_rate'] < 70:
            health_score -= 10
            warnings.append(f"Moderate fix success rate: {metrics['fix_success_rate']:.1f}%")
        
        # System resource assessment
        if metrics['disk_usage'] > 90:
            health_score -= 20
            issues.append(f"High disk usage: {metrics['disk_usage']:.1f}%")
        elif metrics['disk_usage'] > 80:
            health_score -= 10
            warnings.append(f"Elevated disk usage: {metrics['disk_usage']:.1f}%")
        
        if metrics['memory_usage'] > 90:
            health_score -= 15
            issues.append(f"High memory usage: {metrics['memory_usage']:.1f}%")
        elif metrics['memory_usage'] > 80:
            health_score -= 5
            warnings.append(f"Elevated memory usage: {metrics['memory_usage']:.1f}%")
        
        if metrics['system_load'] > 5.0:
            health_score -= 15
            issues.append(f"High system load: {metrics['system_load']:.2f}")
        elif metrics['system_load'] > 2.0:
            health_score -= 5
            warnings.append(f"Elevated system load: {metrics['system_load']:.2f}")
        
        # Determine health status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        elif health_score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            'health_score': max(0, health_score),
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'metrics': metrics
        }

class AutomationEngine:
    """Main automation engine that orchestrates debugging cycles"""
    
    def __init__(self, config: AutomationConfig):
        self.config = config
        self.stats = DaemonStats(started_at=datetime.now())
        self.notification_manager = NotificationManager(config)
        self.health_monitor = HealthMonitor()
        
        # Initialize components
        self.error_db = ErrorDatabase()
        self.self_debugger = SelfDebugger()
        self.web_resolver = WebErrorResolver()
        self.sandbox_validator = SandboxValidator("/home/ncacord/Vega2.0")
        self.patch_manager = PatchManager()
        
        # Runtime state
        self.running = False
        self.fixes_applied_today = 0
        self.fixes_applied_this_hour = 0
        self.last_hour_reset = datetime.now().hour
        self.circuit_breaker_open = False
    
    async def start_daemon(self):
        """Start the autonomous debugging daemon"""
        try:
            self.running = True
            logger.info("🤖 Starting Autonomous Debugging Daemon")
            
            # Send startup notification
            await self.notification_manager.send_notification(
                "Daemon Started",
                "Autonomous debugging daemon has started successfully"
            )
            
            # Schedule periodic tasks
            schedule.every().hour.at(":00").do(self._schedule_hourly_cycle)
            schedule.every().day.at("08:00").do(self._schedule_daily_report)
            schedule.every().day.at("00:00").do(self._schedule_daily_reset)
            
            # Main daemon loop
            while self.running:
                try:
                    # Run scheduled tasks
                    schedule.run_pending()
                    
                    # Update stats
                    self.stats.uptime_seconds = (datetime.now() - self.stats.started_at).total_seconds()
                    
                    # Short sleep to prevent high CPU usage
                    await asyncio.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Error in daemon loop: {e}")
                    await asyncio.sleep(60)  # Longer sleep on error
            
        except Exception as e:
            logger.error(f"Fatal error in daemon: {e}")
            await self.notification_manager.send_notification(
                "Daemon Error",
                f"Fatal error in autonomous debugging daemon: {e}",
                "critical"
            )
        finally:
            await self.cleanup()
    
    def stop_daemon(self):
        """Stop the daemon gracefully"""
        logger.info("🛑 Stopping Autonomous Debugging Daemon")
        self.running = False
    
    async def _schedule_hourly_cycle(self):
        """Schedule hourly debugging cycle"""
        asyncio.create_task(self.hourly_cycle())
    
    async def _schedule_daily_report(self):
        """Schedule daily health report"""
        asyncio.create_task(self.daily_health_report())
    
    async def _schedule_daily_reset(self):
        """Schedule daily counter reset"""
        self.fixes_applied_today = 0
        logger.info("🔄 Daily counters reset")
    
    async def hourly_cycle(self):
        """Execute hourly autonomous debugging cycle"""
        if not self.config.hourly_enabled or self.circuit_breaker_open:
            return
        
        # Check working hours
        if self.config.working_hours_only:
            current_hour = datetime.now().hour
            if not (self.config.working_hours_start <= current_hour < self.config.working_hours_end):
                logger.debug("Outside working hours, skipping cycle")
                return
        
        # Reset hourly counter
        current_hour = datetime.now().hour
        if current_hour != self.last_hour_reset:
            self.fixes_applied_this_hour = 0
            self.last_hour_reset = current_hour
        
        # Check limits
        if (self.fixes_applied_this_hour >= self.config.max_fixes_per_hour or
            self.fixes_applied_today >= self.config.max_fixes_per_day):
            logger.info("⏸️ Fix limits reached, skipping cycle")
            return
        
        try:
            logger.info("🔄 Starting hourly debugging cycle")
            cycle_start = datetime.now()
            
            # Get unresolved errors
            unresolved_errors = self.error_db.get_unresolved_errors(limit=10)
            
            if not unresolved_errors:
                logger.info("✅ No unresolved errors found")
                return
            
            cycle_results = []
            
            for error_row in unresolved_errors:
                try:
                    # Convert to ErrorRecord
                    error = self._row_to_error_record(error_row)
                    
                    # Process error
                    result = await self.process_error_autonomously(error)
                    cycle_results.append(result)
                    
                    # Update counters
                    if result.get('action_taken') == 'applied':
                        self.fixes_applied_this_hour += 1
                        self.fixes_applied_today += 1
                        self.stats.fixes_applied += 1
                    elif result.get('action_taken') == 'rejected':
                        self.stats.fixes_rejected += 1
                    
                    self.stats.errors_processed += 1
                    
                    # Check limits after each fix
                    if (self.fixes_applied_this_hour >= self.config.max_fixes_per_hour or
                        self.fixes_applied_today >= self.config.max_fixes_per_day):
                        logger.info("⏸️ Fix limits reached during cycle")
                        break
                
                except Exception as e:
                    logger.error(f"Error processing {error_row['id']}: {e}")
            
            # Update cycle stats
            self.stats.total_cycles += 1
            self.stats.last_cycle_time = datetime.now()
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            
            # Summary notification
            fixes_applied = sum(1 for r in cycle_results if r.get('action_taken') == 'applied')
            fixes_rejected = sum(1 for r in cycle_results if r.get('action_taken') == 'rejected')
            
            if fixes_applied > 0 or fixes_rejected > 2:
                await self.notification_manager.send_notification(
                    "Hourly Cycle Complete",
                    f"Processed {len(cycle_results)} errors in {cycle_duration:.1f}s. "
                    f"Applied: {fixes_applied}, Rejected: {fixes_rejected}"
                )
            
            logger.info(f"✅ Hourly cycle complete: {len(cycle_results)} errors processed")
            
        except Exception as e:
            logger.error(f"Failed hourly cycle: {e}")
            await self.notification_manager.send_notification(
                "Hourly Cycle Failed",
                f"Hourly debugging cycle failed: {e}",
                "high"
            )
    
    async def process_error_autonomously(self, error: ErrorRecord) -> Dict[str, Any]:
        """Process a single error through the autonomous pipeline"""
        result = {
            'error_id': error.id,
            'action_taken': 'none',
            'confidence': 0.0,
            'safety_score': 0.0,
            'reason': '',
            'details': {}
        }
        
        try:
            logger.info(f"🔍 Processing error {error.id[:8]}: {error.error_type}")
            
            # Step 1: Self-debug with LLM
            debug_result = await self.self_debugger.debug_error(error.id)
            
            if not debug_result['success'] or not debug_result.get('fixes'):
                result['reason'] = "No fixes generated"
                return result
            
            best_fix = debug_result['best_fix']['fix']
            validation_result = debug_result['best_fix']['validation']
            
            # Check confidence threshold
            if best_fix.confidence_score < self.config.confidence_threshold:
                result['action_taken'] = 'rejected'
                result['reason'] = f"Low confidence: {best_fix.confidence_score:.2f}"
                result['confidence'] = best_fix.confidence_score
                return result
            
            # Step 2: Web research (for additional validation)
            web_result = await self.web_resolver.resolve_error(error)
            
            if web_result['success'] and web_result.get('best_solution'):
                # Boost confidence if web solution confirms our approach
                web_confidence = web_result['best_solution'].confidence_score
                if web_confidence > 0.7:
                    best_fix.confidence_score = min(1.0, best_fix.confidence_score * 1.1)
            
            # Step 3: Sandbox validation
            sandbox_result = await self.sandbox_validator.validate_fix(best_fix, error)
            
            result['safety_score'] = sandbox_result.safety_score
            
            # Check safety threshold
            if sandbox_result.safety_score < self.config.safety_threshold:
                result['action_taken'] = 'rejected'
                result['reason'] = f"Low safety score: {sandbox_result.safety_score:.2f}"
                result['confidence'] = best_fix.confidence_score
                return result
            
            # Check for regressions
            if sandbox_result.regression_detected:
                result['action_taken'] = 'rejected'
                result['reason'] = "Regression detected in sandbox"
                result['confidence'] = best_fix.confidence_score
                return result
            
            # Step 4: Apply patch (if auto-apply is enabled)
            if self.config.auto_apply_enabled and sandbox_result.recommendation == "apply":
                patch_metadata = self.patch_manager.apply_patch(best_fix, sandbox_result, "autonomous")
                
                if patch_metadata.status == "applied":
                    # Mark error as resolved
                    self.error_db.mark_error_resolved(error.id, patch_metadata.patch_id)
                    
                    result['action_taken'] = 'applied'
                    result['reason'] = f"Fix applied successfully: {patch_metadata.patch_id}"
                    result['details'] = {
                        'patch_id': patch_metadata.patch_id,
                        'files_modified': len(patch_metadata.files_modified)
                    }
                else:
                    result['action_taken'] = 'rejected'
                    result['reason'] = "Patch application failed"
            else:
                result['action_taken'] = 'manual_review'
                result['reason'] = "Requires manual review"
            
            result['confidence'] = best_fix.confidence_score
            
        except Exception as e:
            logger.error(f"Failed to process error {error.id}: {e}")
            result['action_taken'] = 'error'
            result['reason'] = str(e)
        
        return result
    
    async def daily_health_report(self):
        """Generate and send daily health report"""
        try:
            logger.info("📊 Generating daily health report")
            
            # Collect metrics
            metrics = await self.health_monitor.collect_metrics()
            health_assessment = self.health_monitor.assess_health(metrics)
            
            # Generate report
            report = self._generate_health_report(health_assessment)
            
            # Send notification
            priority = "high" if health_assessment['status'] in ['poor', 'critical'] else "normal"
            await self.notification_manager.send_notification(
                f"Daily Health Report - {health_assessment['status'].title()}",
                report,
                priority
            )
            
            self.stats.last_health_report = datetime.now()
            logger.info("✅ Daily health report sent")
            
        except Exception as e:
            logger.error(f"Failed to generate health report: {e}")
    
    def _generate_health_report(self, health_assessment: Dict[str, Any]) -> str:
        """Generate formatted health report"""
        status_icons = {
            'excellent': '🟢',
            'good': '🟡',
            'fair': '🟠',
            'poor': '🔴',
            'critical': '🆘'
        }
        
        icon = status_icons.get(health_assessment['status'], '❓')
        
        report = f"""
{icon} DAILY HEALTH REPORT - {health_assessment['status'].upper()}

Health Score: {health_assessment['health_score']:.1f}/100

📊 SYSTEM METRICS:
• Error Rate: {health_assessment['metrics']['error_rate']} unresolved
• Fix Success Rate: {health_assessment['metrics']['fix_success_rate']:.1f}%
• System Load: {health_assessment['metrics']['system_load']:.2f}
• Disk Usage: {health_assessment['metrics']['disk_usage']:.1f}%
• Memory Usage: {health_assessment['metrics']['memory_usage']:.1f}%

🤖 DAEMON STATISTICS:
• Uptime: {self.stats.uptime_seconds / 3600:.1f} hours
• Total Cycles: {self.stats.total_cycles}
• Errors Processed: {self.stats.errors_processed}
• Fixes Applied: {self.stats.fixes_applied}
• Fixes Rejected: {self.stats.fixes_rejected}
• Rollbacks: {self.stats.rollbacks_performed}

"""
        
        if health_assessment['issues']:
            report += "🚨 ISSUES:\n"
            for issue in health_assessment['issues']:
                report += f"• {issue}\n"
            report += "\n"
        
        if health_assessment['warnings']:
            report += "⚠️ WARNINGS:\n"
            for warning in health_assessment['warnings']:
                report += f"• {warning}\n"
            report += "\n"
        
        return report
    
    def _row_to_error_record(self, row) -> ErrorRecord:
        """Convert database row to ErrorRecord"""
        context_data = json.loads(row['context_data']) if row['context_data'] else {}
        
        return ErrorRecord(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            file_path=row['file_path'],
            line_number=row['line_number'] or 0,
            error_type=row['error_type'],
            message=row['message'],
            traceback_hash=row['traceback_hash'],
            frequency=row['frequency'],
            snippet=row['snippet'] or "",
            first_seen=datetime.fromisoformat(row['first_seen']),
            last_seen=datetime.fromisoformat(row['last_seen']),
            severity=row['severity'],
            resolved=bool(row['resolved']),
            resolution_attempts=row['resolution_attempts'],
            full_traceback=row['full_traceback'] or "",
            context_data=context_data
        )
    
    async def cleanup(self):
        """Clean up resources"""
        try:
            if self.error_db:
                self.error_db.close()
            if self.self_debugger:
                self.self_debugger.close()
            if self.sandbox_validator:
                self.sandbox_validator.close()
            if self.patch_manager:
                self.patch_manager.close()
            
            logger.info("🧹 Daemon cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

class DaemonController:
    """Controls daemon lifecycle and signal handling"""
    
    def __init__(self, config_path: str = "autonomous_debug/daemon_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.engine = AutomationEngine(self.config)
        self.setup_signal_handlers()
    
    def _load_config(self) -> AutomationConfig:
        """Load daemon configuration"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                    return AutomationConfig(**config_data)
            else:
                # Create default config
                default_config = AutomationConfig()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return AutomationConfig()
    
    def _save_config(self, config: AutomationConfig):
        """Save daemon configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(asdict(config), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.engine.stop_daemon()
    
    async def run(self):
        """Run the daemon"""
        await self.engine.start_daemon()

async def main():
    """Main function for daemon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-Maintenance Daemon + Automation")
    parser.add_argument("--start", action="store_true", help="Start the daemon")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--test-cycle", action="store_true", help="Run test cycle")
    parser.add_argument("--health-report", action="store_true", help="Generate health report")
    parser.add_argument("--status", action="store_true", help="Show daemon status")
    
    args = parser.parse_args()
    
    try:
        if args.start:
            print("🤖 Starting Autonomous Debugging Daemon...")
            controller = DaemonController(args.config or "autonomous_debug/daemon_config.json")
            await controller.run()
        
        elif args.test_cycle:
            print("🔄 Running test automation cycle...")
            config = AutomationConfig()
            engine = AutomationEngine(config)
            await engine.hourly_cycle()
            await engine.cleanup()
            print("✅ Test cycle complete")
        
        elif args.health_report:
            print("📊 Generating health report...")
            config = AutomationConfig()
            engine = AutomationEngine(config)
            await engine.daily_health_report()
            await engine.cleanup()
            print("✅ Health report sent")
        
        elif args.status:
            print("📊 Daemon Status:")
            # This would check if daemon is running and show status
            print("  Status: Not implemented yet")
        
        else:
            print("Specify --start, --test-cycle, --health-report, or --status")
    
    except KeyboardInterrupt:
        print("\n👋 Shutdown requested")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("autonomous_debug/logs", exist_ok=True)
    
    # Run main
    asyncio.run(main())