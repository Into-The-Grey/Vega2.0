"""
Vega System Manager - Auto-maintenance and server management

Handles:
- Server lifecycle management (start, stop, restart, update)
- System updates and upgrades
- Cleanup and maintenance
- Comprehensive logging
- AI-powered suggestions and comments
"""

import os
import subprocess
import shutil
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import psutil


# ============================================================================
# Configuration
# ============================================================================

HOME_DIR = Path.home()
LOG_FILE = HOME_DIR / "vega_system.log"
COMMENTS_FILE = HOME_DIR / "VEGA_COMMENTS.txt"
STATE_FILE = HOME_DIR / ".vega" / "system_state.json"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SystemAction:
    """Record of system action taken"""

    timestamp: str
    action_type: str
    description: str
    status: str  # success, failure, warning
    details: Optional[Dict[str, Any]] = None

    def to_log_entry(self) -> str:
        """Format as log entry"""
        return f"[{self.timestamp}] {self.action_type.upper()}: {self.description} - {self.status}"


@dataclass
class SystemState:
    """Current system state"""

    last_update_check: Optional[str] = None
    last_cleanup: Optional[str] = None
    last_restart: Optional[str] = None
    uptime_seconds: int = 0
    restart_count: int = 0
    update_count: int = 0
    cleanup_count: int = 0


# ============================================================================
# System Manager
# ============================================================================


class VegaSystemManager:
    """Manages Vega system lifecycle and maintenance"""

    def __init__(self):
        self.logger = self._setup_logger()
        self.state = self._load_state()
        self.vega_process: Optional[psutil.Process] = None

    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("VegaSystemManager")
        logger.setLevel(logging.DEBUG)

        # File handler with rotation
        handler = logging.FileHandler(LOG_FILE)
        handler.setLevel(logging.DEBUG)

        # Detailed format with timestamp
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        return logger

    def _load_state(self) -> SystemState:
        """Load system state from file"""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        if STATE_FILE.exists():
            try:
                with open(STATE_FILE) as f:
                    data = json.load(f)
                    return SystemState(**data)
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")

        return SystemState()

    def _save_state(self):
        """Save system state to file"""
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(asdict(self.state), f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def _log_action(self, action: SystemAction):
        """Log action to both logger and file"""
        # Log to logger
        if action.status == "success":
            self.logger.info(action.to_log_entry())
        elif action.status == "failure":
            self.logger.error(action.to_log_entry())
        else:
            self.logger.warning(action.to_log_entry())

        # Append to log file (redundant but explicit)
        try:
            with open(LOG_FILE, "a") as f:
                f.write(action.to_log_entry() + "\n")
                if action.details:
                    f.write(f"  Details: {json.dumps(action.details, indent=2)}\n")
        except Exception as e:
            self.logger.error(f"Failed to write to log file: {e}")

    def add_comment(self, comment: str, category: str = "GENERAL"):
        """Add AI-generated comment or suggestion to comments file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        entry = f"""
{'='*80}
[{timestamp}] {category}
{'='*80}
{comment}

"""

        try:
            with open(COMMENTS_FILE, "a") as f:
                f.write(entry)

            self.logger.info(f"Added comment: {category}")
        except Exception as e:
            self.logger.error(f"Failed to write comment: {e}")

    # ========================================================================
    # Server Management
    # ========================================================================

    def start_server(self) -> bool:
        """Start Vega server"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="start_server",
            description="Starting Vega server",
            status="pending",
        )

        try:
            # Check if already running
            if self.is_server_running():
                action.status = "warning"
                action.description = "Server already running"
                self._log_action(action)
                return True

            # Start server using systemd
            result = subprocess.run(
                ["sudo", "systemctl", "start", "vega"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                action.status = "success"
                action.description = "Vega server started successfully"
                self.state.last_restart = datetime.now().isoformat()
                self.state.restart_count += 1
                self._save_state()
                self._log_action(action)
                return True
            else:
                action.status = "failure"
                action.details = {"error": result.stderr}
                self._log_action(action)
                return False

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return False

    def stop_server(self) -> bool:
        """Stop Vega server"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="stop_server",
            description="Stopping Vega server",
            status="pending",
        )

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "stop", "vega"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                action.status = "success"
                action.description = "Vega server stopped successfully"
                self._log_action(action)
                return True
            else:
                action.status = "failure"
                action.details = {"error": result.stderr}
                self._log_action(action)
                return False

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return False

    def restart_server(self) -> bool:
        """Restart Vega server"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="restart_server",
            description="Restarting Vega server",
            status="pending",
        )

        try:
            result = subprocess.run(
                ["sudo", "systemctl", "restart", "vega"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                action.status = "success"
                action.description = "Vega server restarted successfully"
                self.state.last_restart = datetime.now().isoformat()
                self.state.restart_count += 1
                self._save_state()
                self._log_action(action)

                self.add_comment(
                    f"Server restarted successfully. Total restarts: {self.state.restart_count}",
                    "SERVER_MANAGEMENT",
                )
                return True
            else:
                action.status = "failure"
                action.details = {"error": result.stderr}
                self._log_action(action)
                return False

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return False

    def is_server_running(self) -> bool:
        """Check if Vega server is running"""
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "vega"], capture_output=True, text=True
            )
            return result.stdout.strip() == "active"
        except Exception as e:
            self.logger.error(f"Failed to check server status: {e}")
            return False

    def get_server_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        try:
            result = subprocess.run(
                ["systemctl", "status", "vega"], capture_output=True, text=True
            )

            return {
                "running": self.is_server_running(),
                "status_output": result.stdout,
                "uptime_seconds": self.state.uptime_seconds,
                "restart_count": self.state.restart_count,
            }
        except Exception as e:
            self.logger.error(f"Failed to get server status: {e}")
            return {"running": False, "error": str(e)}

    # ========================================================================
    # System Updates
    # ========================================================================

    def check_for_updates(self) -> Dict[str, Any]:
        """Check for system and package updates"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="check_updates",
            description="Checking for updates",
            status="pending",
        )

        updates_available = {"system": [], "python_packages": [], "vega_updates": False}

        try:
            # Check system updates
            result = subprocess.run(
                ["apt", "list", "--upgradable"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                updates_available["system"] = [
                    line.split("/")[0] for line in lines if line
                ]

            # Check Python packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                updates_available["python_packages"] = json.loads(result.stdout)

            # Check for Vega updates (git)
            vega_dir = Path(__file__).parent.parent.parent.parent
            if (vega_dir / ".git").exists():
                result = subprocess.run(
                    ["git", "fetch"],
                    cwd=vega_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                result = subprocess.run(
                    ["git", "status", "-uno"],
                    cwd=vega_dir,
                    capture_output=True,
                    text=True,
                )

                updates_available["vega_updates"] = "behind" in result.stdout.lower()

            action.status = "success"
            action.details = updates_available
            self.state.last_update_check = datetime.now().isoformat()
            self._save_state()
            self._log_action(action)

            # Generate comment if updates available
            if any(updates_available.values()):
                comment = f"Updates available:\n"
                if updates_available["system"]:
                    comment += (
                        f"  - System packages: {len(updates_available['system'])}\n"
                    )
                if updates_available["python_packages"]:
                    comment += f"  - Python packages: {len(updates_available['python_packages'])}\n"
                if updates_available["vega_updates"]:
                    comment += f"  - Vega application updates available\n"

                self.add_comment(comment, "UPDATE_CHECK")

            return updates_available

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return updates_available

    def update_system(self) -> bool:
        """Update system packages using apt"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="update_system",
            description="Updating system packages",
            status="pending",
        )

        try:
            # Update package list
            result = subprocess.run(
                ["sudo", "apt", "update"], capture_output=True, text=True, timeout=300
            )

            if result.returncode != 0:
                action.status = "failure"
                action.details = {"error": result.stderr}
                self._log_action(action)
                return False

            # Upgrade packages
            result = subprocess.run(
                ["sudo", "apt", "upgrade", "-y"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode == 0:
                action.status = "success"
                action.description = "System packages updated successfully"
                self.state.update_count += 1
                self._save_state()
                self._log_action(action)

                self.add_comment(
                    f"System packages updated successfully. Total updates: {self.state.update_count}",
                    "SYSTEM_UPDATE",
                )
                return True
            else:
                action.status = "failure"
                action.details = {"error": result.stderr}
                self._log_action(action)
                return False

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return False

    def update_python_packages(self) -> bool:
        """Update Python packages"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="update_python",
            description="Updating Python packages",
            status="pending",
        )

        try:
            # Get outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                outdated = json.loads(result.stdout)

                if not outdated:
                    action.status = "success"
                    action.description = "All Python packages up to date"
                    self._log_action(action)
                    return True

                # Update each package
                for package in outdated:
                    pkg_name = package["name"]
                    result = subprocess.run(
                        ["pip", "install", "--upgrade", pkg_name],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode != 0:
                        self.logger.warning(f"Failed to update {pkg_name}")

                action.status = "success"
                action.description = f"Updated {len(outdated)} Python packages"
                action.details = {"packages": [p["name"] for p in outdated]}
                self._log_action(action)

                self.add_comment(
                    f"Updated {len(outdated)} Python packages:\n"
                    + "\n".join(
                        f"  - {p['name']}: {p['version']} → {p['latest_version']}"
                        for p in outdated
                    ),
                    "PYTHON_UPDATE",
                )
                return True

            return False

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return False

    def update_vega(self) -> bool:
        """Update Vega application from git"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="update_vega",
            description="Updating Vega application",
            status="pending",
        )

        try:
            vega_dir = Path(__file__).parent.parent.parent.parent

            if not (vega_dir / ".git").exists():
                action.status = "failure"
                action.description = "Vega directory is not a git repository"
                self._log_action(action)
                return False

            # Pull latest changes
            result = subprocess.run(
                ["git", "pull"],
                cwd=vega_dir,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                action.status = "success"
                action.description = "Vega application updated successfully"
                action.details = {"output": result.stdout}
                self._log_action(action)

                # Restart server after update
                self.restart_server()

                self.add_comment(
                    f"Vega updated from git:\n{result.stdout}", "VEGA_UPDATE"
                )
                return True
            else:
                action.status = "failure"
                action.details = {"error": result.stderr}
                self._log_action(action)
                return False

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return False

    # ========================================================================
    # System Cleanup
    # ========================================================================

    def cleanup_system(self) -> Dict[str, Any]:
        """Perform comprehensive system cleanup"""
        action = SystemAction(
            timestamp=datetime.now().isoformat(),
            action_type="cleanup_system",
            description="Performing system cleanup",
            status="pending",
        )

        cleanup_results = {
            "apt_cleaned": False,
            "logs_rotated": False,
            "temp_files_removed": False,
            "space_freed_mb": 0,
        }

        try:
            # Get initial disk usage
            initial_usage = shutil.disk_usage("/")

            # Clean apt cache
            result = subprocess.run(
                ["sudo", "apt", "autoremove", "-y"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            cleanup_results["apt_cleaned"] = result.returncode == 0

            result = subprocess.run(
                ["sudo", "apt", "clean"], capture_output=True, text=True, timeout=120
            )

            # Rotate logs if they're too large
            if LOG_FILE.exists() and LOG_FILE.stat().st_size > 10 * 1024 * 1024:  # 10MB
                backup = LOG_FILE.with_suffix(".log.old")
                shutil.copy2(LOG_FILE, backup)
                with open(LOG_FILE, "w") as f:
                    f.write(f"[{datetime.now().isoformat()}] Log rotated\n")
                cleanup_results["logs_rotated"] = True

            # Clean temp files in Vega
            vega_temp = Path.home() / ".vega" / "temp"
            if vega_temp.exists():
                for item in vega_temp.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                cleanup_results["temp_files_removed"] = True

            # Calculate space freed
            final_usage = shutil.disk_usage("/")
            space_freed = (initial_usage.used - final_usage.used) / (1024 * 1024)
            cleanup_results["space_freed_mb"] = round(space_freed, 2)

            action.status = "success"
            action.details = cleanup_results
            self.state.last_cleanup = datetime.now().isoformat()
            self.state.cleanup_count += 1
            self._save_state()
            self._log_action(action)

            self.add_comment(
                f"System cleanup completed:\n"
                + f"  - APT cleaned: {cleanup_results['apt_cleaned']}\n"
                + f"  - Logs rotated: {cleanup_results['logs_rotated']}\n"
                + f"  - Temp files removed: {cleanup_results['temp_files_removed']}\n"
                + f"  - Space freed: {cleanup_results['space_freed_mb']} MB\n"
                + f"  - Total cleanups: {self.state.cleanup_count}",
                "SYSTEM_CLEANUP",
            )

            return cleanup_results

        except Exception as e:
            action.status = "failure"
            action.details = {"exception": str(e)}
            self._log_action(action)
            return cleanup_results

    # ========================================================================
    # Health Monitoring
    # ========================================================================

    def monitor_health(self) -> Dict[str, Any]:
        """Monitor system health and generate suggestions"""
        health_status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage("/").percent,
            "server_running": self.is_server_running(),
            "suggestions": [],
        }

        # Generate suggestions
        if health_status["cpu_percent"] > 80:
            health_status["suggestions"].append(
                "High CPU usage detected. Consider optimizing resource-intensive tasks."
            )

        if health_status["memory_percent"] > 85:
            health_status["suggestions"].append(
                "High memory usage detected. Consider restarting server to free memory."
            )

        if health_status["disk_percent"] > 90:
            health_status["suggestions"].append(
                "Disk space critically low. Run cleanup_system() immediately."
            )

        if not health_status["server_running"]:
            health_status["suggestions"].append(
                "Vega server is not running. Starting server..."
            )
            self.start_server()

        # Log suggestions
        if health_status["suggestions"]:
            self.add_comment(
                "\n".join(health_status["suggestions"]), "HEALTH_MONITORING"
            )

        return health_status


# ============================================================================
# CLI Interface
# ============================================================================


def main():
    """Main entry point for system manager"""
    manager = VegaSystemManager()

    import sys

    if len(sys.argv) < 2:
        print("Usage: python system_manager.py <command>")
        print("Commands: start, stop, restart, status, update, cleanup, health")
        return

    command = sys.argv[1].lower()

    if command == "start":
        success = manager.start_server()
        print(f"Server start: {'Success' if success else 'Failed'}")

    elif command == "stop":
        success = manager.stop_server()
        print(f"Server stop: {'Success' if success else 'Failed'}")

    elif command == "restart":
        success = manager.restart_server()
        print(f"Server restart: {'Success' if success else 'Failed'}")

    elif command == "status":
        status = manager.get_server_status()
        print(json.dumps(status, indent=2))

    elif command == "update":
        print("Checking for updates...")
        updates = manager.check_for_updates()
        print(json.dumps(updates, indent=2))

        if updates["system"]:
            print("\nUpdating system packages...")
            manager.update_system()

        if updates["python_packages"]:
            print("\nUpdating Python packages...")
            manager.update_python_packages()

        if updates["vega_updates"]:
            print("\nUpdating Vega application...")
            manager.update_vega()

    elif command == "cleanup":
        print("Performing system cleanup...")
        results = manager.cleanup_system()
        print(json.dumps(results, indent=2))

    elif command == "health":
        health = manager.monitor_health()
        print(json.dumps(health, indent=2))

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
