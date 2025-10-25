"""
Vega Daemon - Continuous monitoring and maintenance service

Runs in the background to:
- Keep server alive and healthy
- Perform automatic updates
- Run periodic maintenance
- Generate AI suggestions
"""

import time
import schedule
from pathlib import Path
from datetime import datetime

from src.vega.daemon.system_manager import VegaSystemManager


class VegaDaemon:
    """Daemon for continuous Vega system management"""

    def __init__(self):
        self.manager = VegaSystemManager()
        self.running = True

    def start(self):
        """Start the daemon"""
        self.manager.logger.info("=" * 80)
        self.manager.logger.info("Vega Daemon Starting")
        self.manager.logger.info("=" * 80)

        self.manager.add_comment(
            "Vega Daemon started. Continuous monitoring active.", "DAEMON_START"
        )

        # Ensure server is running
        if not self.manager.is_server_running():
            self.manager.logger.warning("Server not running, starting...")
            self.manager.start_server()

        # Schedule tasks
        self.schedule_tasks()

        # Main loop
        self.manager.logger.info("Entering main monitoring loop")
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            self.manager.logger.info("Daemon stopped by user")
            self.stop()
        except Exception as e:
            self.manager.logger.error(f"Daemon error: {e}")
            self.stop()

    def stop(self):
        """Stop the daemon"""
        self.running = False
        self.manager.logger.info("Vega Daemon Stopping")
        self.manager.add_comment("Vega Daemon stopped.", "DAEMON_STOP")

    def schedule_tasks(self):
        """Schedule periodic tasks"""
        # Health check every 5 minutes
        schedule.every(5).minutes.do(self.health_check)

        # Update check every 6 hours
        schedule.every(6).hours.do(self.check_updates)

        # Cleanup once per day at 3 AM
        schedule.every().day.at("03:00").do(self.daily_cleanup)

        # Weekly full update on Sunday at 2 AM
        schedule.every().sunday.at("02:00").do(self.weekly_update)

        # Generate weekly report on Monday at 9 AM
        schedule.every().monday.at("09:00").do(self.weekly_report)

        self.manager.logger.info("Scheduled tasks configured:")
        self.manager.logger.info("  - Health check: Every 5 minutes")
        self.manager.logger.info("  - Update check: Every 6 hours")
        self.manager.logger.info("  - Cleanup: Daily at 3 AM")
        self.manager.logger.info("  - Full update: Sunday at 2 AM")
        self.manager.logger.info("  - Weekly report: Monday at 9 AM")

    def health_check(self):
        """Perform health check"""
        self.manager.logger.debug("Running health check...")
        health = self.manager.monitor_health()

        # Log critical issues
        if health["cpu_percent"] > 90:
            self.manager.logger.warning(f"Critical CPU usage: {health['cpu_percent']}%")
        if health["memory_percent"] > 90:
            self.manager.logger.warning(
                f"Critical memory usage: {health['memory_percent']}%"
            )
        if health["disk_percent"] > 95:
            self.manager.logger.error(f"Critical disk usage: {health['disk_percent']}%")
            # Auto-cleanup on critical disk usage
            self.manager.cleanup_system()

        # Ensure server is running
        if not health["server_running"]:
            self.manager.logger.error("Server is down! Attempting restart...")
            self.manager.start_server()

    def check_updates(self):
        """Check for available updates"""
        self.manager.logger.info("Checking for updates...")
        updates = self.manager.check_for_updates()

        # Log available updates
        if any(updates.values()):
            self.manager.logger.info("Updates available:")
            if updates["system"]:
                self.manager.logger.info(
                    f"  - System: {len(updates['system'])} packages"
                )
            if updates["python_packages"]:
                self.manager.logger.info(
                    f"  - Python: {len(updates['python_packages'])} packages"
                )
            if updates["vega_updates"]:
                self.manager.logger.info("  - Vega: Application updates available")

    def daily_cleanup(self):
        """Perform daily maintenance"""
        self.manager.logger.info("Running daily cleanup...")
        results = self.manager.cleanup_system()

        self.manager.logger.info("Daily cleanup completed:")
        self.manager.logger.info(f"  - Space freed: {results['space_freed_mb']} MB")
        self.manager.logger.info(f"  - Logs rotated: {results['logs_rotated']}")

    def weekly_update(self):
        """Perform weekly system update"""
        self.manager.logger.info("Running weekly system update...")

        # Check what needs updating
        updates = self.manager.check_for_updates()

        # Update system packages
        if updates["system"]:
            self.manager.logger.info("Updating system packages...")
            self.manager.update_system()

        # Update Python packages
        if updates["python_packages"]:
            self.manager.logger.info("Updating Python packages...")
            self.manager.update_python_packages()

        # Update Vega application
        if updates["vega_updates"]:
            self.manager.logger.info("Updating Vega application...")
            self.manager.update_vega()

        # Restart server after updates
        self.manager.logger.info("Restarting server after updates...")
        self.manager.restart_server()

        # Run cleanup
        self.manager.logger.info("Running post-update cleanup...")
        self.manager.cleanup_system()

    def weekly_report(self):
        """Generate weekly report"""
        self.manager.logger.info("Generating weekly report...")

        report = f"""
Weekly Vega System Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

System Statistics:
- Total Restarts: {self.manager.state.restart_count}
- Total Updates: {self.manager.state.update_count}
- Total Cleanups: {self.manager.state.cleanup_count}
- Last Update Check: {self.manager.state.last_update_check or 'Never'}
- Last Cleanup: {self.manager.state.last_cleanup or 'Never'}
- Last Restart: {self.manager.state.last_restart or 'Never'}

Current Health:
"""

        health = self.manager.monitor_health()
        report += f"- CPU Usage: {health['cpu_percent']}%\n"
        report += f"- Memory Usage: {health['memory_percent']}%\n"
        report += f"- Disk Usage: {health['disk_percent']}%\n"
        report += f"- Server Running: {health['server_running']}\n"

        if health["suggestions"]:
            report += "\nRecommendations:\n"
            for suggestion in health["suggestions"]:
                report += f"  - {suggestion}\n"

        self.manager.add_comment(report, "WEEKLY_REPORT")


def main():
    """Main entry point for daemon"""
    daemon = VegaDaemon()
    daemon.start()


if __name__ == "__main__":
    main()
