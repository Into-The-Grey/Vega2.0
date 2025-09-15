"""
System Autonomy Core (SAC) - Phase 2: Active Monitoring & System Health Watchdog

This module provides real-time system monitoring with autonomous threat detection,
threshold breach monitoring, and automated mitigation strategies. It operates as
a persistent background daemon that continuously watches system health.

Key Features:
- Real-time resource monitoring (CPU, memory, disk, GPU, network)
- Configurable threshold management with escalation levels
- Automatic mitigation strategies for common issues
- Emergency response protocols
- Persistent daemon operation with auto-restart
- Integration with system probe for detailed analysis

Author: Vega2.0 Autonomous AI System
"""

import asyncio
import json
import time
import threading
import signal
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import logging
from enum import Enum

try:
    import psutil
except ImportError:
    print(
        "WARNING: psutil required for system monitoring. Install with: pip install psutil"
    )
    psutil = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/ncacord/Vega2.0/sac/logs/system_watchdog.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of monitored metrics"""

    CPU_USAGE = "cpu_usage"
    CPU_TEMP = "cpu_temp"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    GPU_TEMP = "gpu_temp"
    GPU_USAGE = "gpu_usage"
    NETWORK_ERROR = "network_error"
    PROCESS_COUNT = "process_count"
    LOAD_AVERAGE = "load_average"
    THERMAL_ZONE = "thermal_zone"


@dataclass
class Alert:
    """System alert structure"""

    timestamp: str
    metric_type: MetricType
    level: AlertLevel
    message: str
    current_value: float
    threshold_value: float
    hostname: str
    mitigation_action: Optional[str] = None
    resolved: bool = False
    resolution_timestamp: Optional[str] = None


@dataclass
class MitigationAction:
    """Mitigation action configuration"""

    name: str
    description: str
    command: Optional[str]
    function: Optional[Callable]
    emergency_only: bool = False
    requires_confirmation: bool = True


@dataclass
class SystemMetrics:
    """Current system metrics snapshot"""

    timestamp: str
    cpu_usage: float
    cpu_temp: Optional[float]
    memory_usage: float
    memory_available: int
    disk_usage: Dict[str, float]  # device -> usage %
    gpu_temps: List[Optional[float]]
    gpu_usage: List[Optional[float]]
    load_average: List[float]
    process_count: int
    network_errors: Dict[str, int]  # interface -> error count
    thermal_zones: Dict[str, float]


class SystemWatchdog:
    """
    Advanced system health monitoring daemon with autonomous response capabilities.

    Provides real-time monitoring, threshold management, anomaly detection,
    and automated mitigation strategies for system health maintenance.
    """

    def __init__(self, config_path: str = "/home/ncacord/Vega2.0/sac/config"):
        self.config_path = Path(config_path)
        self.logs_path = Path("/home/ncacord/Vega2.0/sac/logs")
        self.alerts_file = self.logs_path / "system_alerts.jsonl"

        # Runtime state
        self.running = False
        self.monitoring_thread = None
        self.last_metrics = None
        self.active_alerts = {}  # metric_type -> Alert
        self.alert_history = []
        self.mitigation_cooldown = {}  # action_name -> last_executed_time

        # Load configuration and mitigation actions
        self.config = self._load_config()
        self.mitigations = self._setup_mitigations()

        # Ensure directories exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Load alert history
        self._load_alert_history()

        logger.info("SystemWatchdog initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load watchdog configuration"""
        config_file = self.config_path / "watchdog_config.json"

        default_config = {
            "monitoring": {
                "interval_seconds": 60,
                "enable_auto_mitigation": True,
                "emergency_mode_threshold": 3,  # Number of critical alerts to trigger emergency
                "cooldown_minutes": 5,  # Minimum time between same mitigations
                "max_alerts_per_hour": 10,
            },
            "thresholds": {
                "cpu_usage_warning": 80.0,
                "cpu_usage_critical": 95.0,
                "cpu_temp_warning": 70.0,
                "cpu_temp_critical": 85.0,
                "memory_usage_warning": 85.0,
                "memory_usage_critical": 95.0,
                "disk_usage_warning": 80.0,
                "disk_usage_critical": 90.0,
                "gpu_temp_warning": 75.0,
                "gpu_temp_critical": 90.0,
                "gpu_usage_warning": 90.0,
                "gpu_usage_critical": 98.0,
                "load_average_warning": 20.0,  # For 24-core system
                "load_average_critical": 30.0,
                "process_count_warning": 1000,
                "process_count_critical": 2000,
                "network_errors_warning": 100,
                "network_errors_critical": 1000,
            },
            "mitigation": {
                "enable_process_killing": False,  # Safety: disabled by default
                "enable_cache_clearing": True,
                "enable_service_restart": False,  # Safety: disabled by default
                "enable_thermal_throttling": True,
                "emergency_actions_enabled": False,  # Safety: disabled by default
            },
            "notifications": {
                "enable_daily_digest": True,
                "enable_immediate_alerts": True,
                "alert_retention_days": 30,
            },
        }

        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if subkey not in config[key]:
                                config[key][subkey] = subvalue
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                config = default_config
        else:
            config = default_config
            self._save_config(config)

        return config

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        config_file = self.config_path / "watchdog_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _setup_mitigations(self) -> Dict[str, MitigationAction]:
        """Setup available mitigation actions"""
        return {
            "clear_page_cache": MitigationAction(
                name="clear_page_cache",
                description="Clear system page cache to free memory",
                command="sync && echo 1 | sudo tee /proc/sys/vm/drop_caches",
                function=None,
                emergency_only=False,
                requires_confirmation=False,
            ),
            "clear_all_caches": MitigationAction(
                name="clear_all_caches",
                description="Clear all system caches (page, dentry, inode)",
                command="sync && echo 3 | sudo tee /proc/sys/vm/drop_caches",
                function=None,
                emergency_only=False,
                requires_confirmation=False,
            ),
            "kill_memory_hogs": MitigationAction(
                name="kill_memory_hogs",
                description="Terminate processes consuming excessive memory",
                command=None,
                function=self._kill_memory_intensive_processes,
                emergency_only=True,
                requires_confirmation=True,
            ),
            "reduce_cpu_freq": MitigationAction(
                name="reduce_cpu_freq",
                description="Reduce CPU frequency to lower temperatures",
                command="echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor",
                function=None,
                emergency_only=False,
                requires_confirmation=False,
            ),
            "restart_high_cpu_processes": MitigationAction(
                name="restart_high_cpu_processes",
                description="Restart processes with abnormally high CPU usage",
                command=None,
                function=self._restart_high_cpu_processes,
                emergency_only=True,
                requires_confirmation=True,
            ),
            "emergency_shutdown": MitigationAction(
                name="emergency_shutdown",
                description="Emergency system shutdown to prevent hardware damage",
                command="sudo shutdown -h +1 'Emergency shutdown triggered by SystemWatchdog'",
                function=None,
                emergency_only=True,
                requires_confirmation=True,
            ),
            "flush_disk_buffers": MitigationAction(
                name="flush_disk_buffers",
                description="Flush disk write buffers to improve I/O performance",
                command="sync",
                function=None,
                emergency_only=False,
                requires_confirmation=False,
            ),
            "lower_swappiness": MitigationAction(
                name="lower_swappiness",
                description="Reduce swappiness to minimize disk thrashing",
                command="echo 10 | sudo tee /proc/sys/vm/swappiness",
                function=None,
                emergency_only=False,
                requires_confirmation=False,
            ),
        }

    def _load_alert_history(self):
        """Load alert history from file"""
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, "r") as f:
                    for line in f:
                        if line.strip():
                            alert_data = json.loads(line)
                            alert = Alert(**alert_data)
                            self.alert_history.append(alert)
            except Exception as e:
                logger.error(f"Error loading alert history: {e}")

    def _save_alert(self, alert: Alert):
        """Save alert to persistent log"""
        try:
            with open(self.alerts_file, "a") as f:
                f.write(json.dumps(asdict(alert)) + "\n")
        except Exception as e:
            logger.error(f"Error saving alert: {e}")

    def _run_command_safe(self, command: str) -> bool:
        """Execute command safely with timeout"""
        try:
            import subprocess

            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"Command failed: {command} - {result.stderr}")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"Command timeout: {command}")
            return False
        except Exception as e:
            logger.error(f"Command error: {command} - {e}")
            return False

    def _kill_memory_intensive_processes(self) -> bool:
        """Kill processes using excessive memory (emergency mitigation)"""
        if not psutil:
            return False

        try:
            # Get processes sorted by memory usage
            processes = []
            for proc in psutil.process_iter(["pid", "name", "memory_percent"]):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by memory usage (highest first)
            processes.sort(key=lambda x: x["memory_percent"], reverse=True)

            # Kill top memory consumers (excluding system processes)
            killed_count = 0
            system_processes = {
                "kernel",
                "init",
                "kthreadd",
                "systemd",
                "ssh",
                "watchdog",
            }

            for proc_info in processes[:5]:  # Only consider top 5
                if proc_info["memory_percent"] > 10.0:  # Using more than 10% memory
                    proc_name = proc_info["name"].lower()

                    # Skip system-critical processes
                    if any(sys_proc in proc_name for sys_proc in system_processes):
                        continue

                    try:
                        proc = psutil.Process(proc_info["pid"])
                        proc.terminate()
                        logger.warning(
                            f"Terminated high-memory process: {proc_info['name']} (PID: {proc_info['pid']}, Memory: {proc_info['memory_percent']:.1f}%)"
                        )
                        killed_count += 1

                        if killed_count >= 3:  # Limit to 3 processes
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.warning(
                            f"Could not terminate process {proc_info['pid']}: {e}"
                        )

            return killed_count > 0
        except Exception as e:
            logger.error(f"Error in memory process killer: {e}")
            return False

    def _restart_high_cpu_processes(self) -> bool:
        """Restart processes with abnormally high CPU usage"""
        if not psutil:
            return False

        try:
            # Monitor CPU usage for a short period
            high_cpu_procs = []

            for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
                try:
                    # Get CPU usage over 1 second interval
                    cpu_percent = proc.info["cpu_percent"]
                    if cpu_percent > 50.0:  # High CPU usage
                        high_cpu_procs.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Restart high CPU processes (carefully)
            restarted_count = 0
            safe_to_restart = {"firefox", "chrome", "code", "python"}

            for proc_info in high_cpu_procs:
                proc_name = proc_info["name"].lower()

                # Only restart known-safe applications
                if any(safe_name in proc_name for safe_name in safe_to_restart):
                    try:
                        proc = psutil.Process(proc_info["pid"])
                        proc.terminate()
                        logger.warning(
                            f"Restarted high-CPU process: {proc_info['name']} (PID: {proc_info['pid']}, CPU: {proc_info['cpu_percent']:.1f}%)"
                        )
                        restarted_count += 1

                        if restarted_count >= 2:  # Limit restarts
                            break
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.warning(
                            f"Could not restart process {proc_info['pid']}: {e}"
                        )

            return restarted_count > 0
        except Exception as e:
            logger.error(f"Error in CPU process restarter: {e}")
            return False

    def _collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        if not psutil:
            # Return empty metrics if psutil not available
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_usage=0.0,
                cpu_temp=None,
                memory_usage=0.0,
                memory_available=0,
                disk_usage={},
                gpu_temps=[],
                gpu_usage=[],
                load_average=[0, 0, 0],
                process_count=0,
                network_errors={},
                thermal_zones={},
            )

        # CPU metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        cpu_temp = None

        # Try to get CPU temperature
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    # Try different temperature sensor names
                    for sensor_name in ["coretemp", "k10temp", "cpu_thermal"]:
                        if sensor_name in temps:
                            cpu_temp = temps[sensor_name][0].current
                            break
        except Exception:
            pass

        # Memory metrics
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available

        # Disk metrics
        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.device] = (usage.used / usage.total) * 100
            except PermissionError:
                continue

        # GPU metrics (if nvidia-smi available)
        gpu_temps = []
        gpu_usage = []
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split(",")
                        if len(parts) >= 2:
                            try:
                                temp = float(parts[0].strip())
                                util = float(parts[1].strip())
                                gpu_temps.append(temp)
                                gpu_usage.append(util)
                            except ValueError:
                                gpu_temps.append(None)
                                gpu_usage.append(None)
        except Exception:
            pass

        # Load average
        load_average = list(os.getloadavg())

        # Process count
        process_count = len(psutil.pids())

        # Network errors
        network_errors = {}
        try:
            net_io = psutil.net_io_counters(pernic=True)
            for interface, stats in net_io.items():
                if interface != "lo":  # Skip loopback
                    network_errors[interface] = stats.errin + stats.errout
        except Exception:
            pass

        # Thermal zones
        thermal_zones = {}
        try:
            thermal_path = Path("/sys/class/thermal")
            if thermal_path.exists():
                for zone_dir in thermal_path.glob("thermal_zone*"):
                    temp_file = zone_dir / "temp"
                    type_file = zone_dir / "type"

                    if temp_file.exists():
                        try:
                            with open(temp_file) as f:
                                temp_milli = int(f.read().strip())
                                temp_celsius = temp_milli / 1000.0

                            zone_type = zone_dir.name
                            if type_file.exists():
                                with open(type_file) as f:
                                    zone_type = f.read().strip()

                            thermal_zones[zone_type] = temp_celsius
                        except (ValueError, IOError):
                            pass
        except Exception:
            pass

        return SystemMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            cpu_temp=cpu_temp,
            memory_usage=memory_usage,
            memory_available=memory_available,
            disk_usage=disk_usage,
            gpu_temps=gpu_temps,
            gpu_usage=gpu_usage,
            load_average=load_average,
            process_count=process_count,
            network_errors=network_errors,
            thermal_zones=thermal_zones,
        )

    def _check_thresholds(self, metrics: SystemMetrics) -> List[Alert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        thresholds = self.config["thresholds"]
        hostname = os.uname().nodename

        # CPU usage alerts
        if metrics.cpu_usage >= thresholds["cpu_usage_critical"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.CPU_USAGE,
                level=AlertLevel.CRITICAL,
                message=f"CPU usage critical: {metrics.cpu_usage:.1f}%",
                current_value=metrics.cpu_usage,
                threshold_value=thresholds["cpu_usage_critical"],
                hostname=hostname,
            )
            alerts.append(alert)
        elif metrics.cpu_usage >= thresholds["cpu_usage_warning"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.CPU_USAGE,
                level=AlertLevel.WARNING,
                message=f"CPU usage high: {metrics.cpu_usage:.1f}%",
                current_value=metrics.cpu_usage,
                threshold_value=thresholds["cpu_usage_warning"],
                hostname=hostname,
            )
            alerts.append(alert)

        # CPU temperature alerts
        if metrics.cpu_temp:
            if metrics.cpu_temp >= thresholds["cpu_temp_critical"]:
                alert = Alert(
                    timestamp=metrics.timestamp,
                    metric_type=MetricType.CPU_TEMP,
                    level=AlertLevel.CRITICAL,
                    message=f"CPU temperature critical: {metrics.cpu_temp:.1f}°C",
                    current_value=metrics.cpu_temp,
                    threshold_value=thresholds["cpu_temp_critical"],
                    hostname=hostname,
                )
                alerts.append(alert)
            elif metrics.cpu_temp >= thresholds["cpu_temp_warning"]:
                alert = Alert(
                    timestamp=metrics.timestamp,
                    metric_type=MetricType.CPU_TEMP,
                    level=AlertLevel.WARNING,
                    message=f"CPU temperature high: {metrics.cpu_temp:.1f}°C",
                    current_value=metrics.cpu_temp,
                    threshold_value=thresholds["cpu_temp_warning"],
                    hostname=hostname,
                )
                alerts.append(alert)

        # Memory usage alerts
        if metrics.memory_usage >= thresholds["memory_usage_critical"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.MEMORY_USAGE,
                level=AlertLevel.CRITICAL,
                message=f"Memory usage critical: {metrics.memory_usage:.1f}%",
                current_value=metrics.memory_usage,
                threshold_value=thresholds["memory_usage_critical"],
                hostname=hostname,
            )
            alerts.append(alert)
        elif metrics.memory_usage >= thresholds["memory_usage_warning"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.MEMORY_USAGE,
                level=AlertLevel.WARNING,
                message=f"Memory usage high: {metrics.memory_usage:.1f}%",
                current_value=metrics.memory_usage,
                threshold_value=thresholds["memory_usage_warning"],
                hostname=hostname,
            )
            alerts.append(alert)

        # Disk usage alerts
        for device, usage_percent in metrics.disk_usage.items():
            if usage_percent >= thresholds["disk_usage_critical"]:
                alert = Alert(
                    timestamp=metrics.timestamp,
                    metric_type=MetricType.DISK_USAGE,
                    level=AlertLevel.CRITICAL,
                    message=f"Disk usage critical on {device}: {usage_percent:.1f}%",
                    current_value=usage_percent,
                    threshold_value=thresholds["disk_usage_critical"],
                    hostname=hostname,
                )
                alerts.append(alert)
            elif usage_percent >= thresholds["disk_usage_warning"]:
                alert = Alert(
                    timestamp=metrics.timestamp,
                    metric_type=MetricType.DISK_USAGE,
                    level=AlertLevel.WARNING,
                    message=f"Disk usage high on {device}: {usage_percent:.1f}%",
                    current_value=usage_percent,
                    threshold_value=thresholds["disk_usage_warning"],
                    hostname=hostname,
                )
                alerts.append(alert)

        # GPU temperature alerts
        for i, temp in enumerate(metrics.gpu_temps):
            if temp:
                if temp >= thresholds["gpu_temp_critical"]:
                    alert = Alert(
                        timestamp=metrics.timestamp,
                        metric_type=MetricType.GPU_TEMP,
                        level=AlertLevel.CRITICAL,
                        message=f"GPU {i} temperature critical: {temp:.1f}°C",
                        current_value=temp,
                        threshold_value=thresholds["gpu_temp_critical"],
                        hostname=hostname,
                    )
                    alerts.append(alert)
                elif temp >= thresholds["gpu_temp_warning"]:
                    alert = Alert(
                        timestamp=metrics.timestamp,
                        metric_type=MetricType.GPU_TEMP,
                        level=AlertLevel.WARNING,
                        message=f"GPU {i} temperature high: {temp:.1f}°C",
                        current_value=temp,
                        threshold_value=thresholds["gpu_temp_warning"],
                        hostname=hostname,
                    )
                    alerts.append(alert)

        # Load average alerts (for 24-core system)
        load_1min = metrics.load_average[0]
        if load_1min >= thresholds["load_average_critical"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.LOAD_AVERAGE,
                level=AlertLevel.CRITICAL,
                message=f"Load average critical: {load_1min:.2f}",
                current_value=load_1min,
                threshold_value=thresholds["load_average_critical"],
                hostname=hostname,
            )
            alerts.append(alert)
        elif load_1min >= thresholds["load_average_warning"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.LOAD_AVERAGE,
                level=AlertLevel.WARNING,
                message=f"Load average high: {load_1min:.2f}",
                current_value=load_1min,
                threshold_value=thresholds["load_average_warning"],
                hostname=hostname,
            )
            alerts.append(alert)

        # Process count alerts
        if metrics.process_count >= thresholds["process_count_critical"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.PROCESS_COUNT,
                level=AlertLevel.CRITICAL,
                message=f"Process count critical: {metrics.process_count}",
                current_value=metrics.process_count,
                threshold_value=thresholds["process_count_critical"],
                hostname=hostname,
            )
            alerts.append(alert)
        elif metrics.process_count >= thresholds["process_count_warning"]:
            alert = Alert(
                timestamp=metrics.timestamp,
                metric_type=MetricType.PROCESS_COUNT,
                level=AlertLevel.WARNING,
                message=f"Process count high: {metrics.process_count}",
                current_value=metrics.process_count,
                threshold_value=thresholds["process_count_warning"],
                hostname=hostname,
            )
            alerts.append(alert)

        return alerts

    def _execute_mitigation(self, alert: Alert) -> bool:
        """Execute appropriate mitigation for an alert"""
        if not self.config["monitoring"]["enable_auto_mitigation"]:
            logger.info(f"Auto-mitigation disabled, skipping for: {alert.message}")
            return False

        # Determine appropriate mitigation based on alert type and level
        mitigation_name = None

        if alert.metric_type == MetricType.MEMORY_USAGE:
            if alert.level == AlertLevel.CRITICAL:
                if self.config["mitigation"]["enable_process_killing"]:
                    mitigation_name = "kill_memory_hogs"
                else:
                    mitigation_name = "clear_all_caches"
            else:
                mitigation_name = "clear_page_cache"

        elif alert.metric_type == MetricType.CPU_TEMP:
            if alert.level == AlertLevel.CRITICAL:
                mitigation_name = "reduce_cpu_freq"
            else:
                mitigation_name = "reduce_cpu_freq"

        elif alert.metric_type == MetricType.CPU_USAGE:
            if (
                alert.level == AlertLevel.CRITICAL
                and self.config["mitigation"]["enable_process_killing"]
            ):
                mitigation_name = "restart_high_cpu_processes"

        elif alert.metric_type == MetricType.DISK_USAGE:
            mitigation_name = "flush_disk_buffers"

        elif alert.metric_type == MetricType.LOAD_AVERAGE:
            mitigation_name = "lower_swappiness"

        if not mitigation_name:
            logger.info(f"No mitigation available for: {alert.message}")
            return False

        # Check cooldown
        cooldown_minutes = self.config["monitoring"]["cooldown_minutes"]
        last_executed = self.mitigation_cooldown.get(mitigation_name)
        if last_executed:
            time_since = datetime.now() - datetime.fromisoformat(last_executed)
            if time_since.total_seconds() < cooldown_minutes * 60:
                logger.info(f"Mitigation {mitigation_name} on cooldown, skipping")
                return False

        # Execute mitigation
        mitigation = self.mitigations.get(mitigation_name)
        if not mitigation:
            logger.error(f"Unknown mitigation: {mitigation_name}")
            return False

        # Check if emergency action is required
        if (
            mitigation.emergency_only
            and not self.config["mitigation"]["emergency_actions_enabled"]
        ):
            logger.warning(f"Emergency mitigation {mitigation_name} disabled, skipping")
            return False

        logger.info(f"Executing mitigation: {mitigation.description}")

        success = False
        try:
            if mitigation.command:
                success = self._run_command_safe(mitigation.command)
            elif mitigation.function:
                success = mitigation.function()

            if success:
                self.mitigation_cooldown[mitigation_name] = datetime.now().isoformat()
                alert.mitigation_action = mitigation_name
                logger.info(f"Mitigation {mitigation_name} executed successfully")
            else:
                logger.error(f"Mitigation {mitigation_name} failed")

        except Exception as e:
            logger.error(f"Error executing mitigation {mitigation_name}: {e}")

        return success

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("System monitoring started")

        while self.running:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.last_metrics = metrics

                # Check thresholds and generate alerts
                new_alerts = self._check_thresholds(metrics)

                # Process new alerts
                for alert in new_alerts:
                    # Check if this is a new alert or update to existing
                    alert_key = f"{alert.metric_type.value}_{alert.level.value}"

                    if alert_key not in self.active_alerts:
                        # New alert
                        self.active_alerts[alert_key] = alert
                        self.alert_history.append(alert)
                        self._save_alert(alert)

                        logger.warning(f"NEW ALERT: {alert.message}")

                        # Execute mitigation if enabled
                        if alert.level in [AlertLevel.CRITICAL, AlertLevel.WARNING]:
                            mitigation_success = self._execute_mitigation(alert)
                            if mitigation_success:
                                logger.info(f"Mitigation executed for: {alert.message}")
                    else:
                        # Update existing alert
                        self.active_alerts[alert_key] = alert

                # Check for resolved alerts
                resolved_alerts = []
                for alert_key, active_alert in self.active_alerts.items():
                    metric_type = active_alert.metric_type

                    # Check if alert condition is resolved
                    resolved = False
                    if metric_type == MetricType.CPU_USAGE:
                        resolved = (
                            metrics.cpu_usage
                            < self.config["thresholds"]["cpu_usage_warning"]
                        )
                    elif metric_type == MetricType.CPU_TEMP:
                        resolved = (
                            not metrics.cpu_temp
                            or metrics.cpu_temp
                            < self.config["thresholds"]["cpu_temp_warning"]
                        )
                    elif metric_type == MetricType.MEMORY_USAGE:
                        resolved = (
                            metrics.memory_usage
                            < self.config["thresholds"]["memory_usage_warning"]
                        )
                    # Add more resolution checks as needed

                    if resolved:
                        active_alert.resolved = True
                        active_alert.resolution_timestamp = datetime.now().isoformat()
                        self._save_alert(active_alert)
                        resolved_alerts.append(alert_key)
                        logger.info(f"RESOLVED: {active_alert.message}")

                # Remove resolved alerts from active list
                for alert_key in resolved_alerts:
                    del self.active_alerts[alert_key]

                # Check for emergency conditions
                critical_count = sum(
                    1
                    for alert in self.active_alerts.values()
                    if alert.level == AlertLevel.CRITICAL
                )
                if (
                    critical_count
                    >= self.config["monitoring"]["emergency_mode_threshold"]
                ):
                    self._handle_emergency_condition(critical_count)

                # Sleep until next monitoring cycle
                time.sleep(self.config["monitoring"]["interval_seconds"])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Short sleep on error

    def _handle_emergency_condition(self, critical_alert_count: int):
        """Handle emergency system condition"""
        logger.critical(f"EMERGENCY: {critical_alert_count} critical alerts active")

        if self.config["mitigation"]["emergency_actions_enabled"]:
            # Consider emergency shutdown if thermal issues
            thermal_alerts = [
                a
                for a in self.active_alerts.values()
                if a.metric_type in [MetricType.CPU_TEMP, MetricType.GPU_TEMP]
                and a.level == AlertLevel.CRITICAL
            ]

            if len(thermal_alerts) >= 2:  # Multiple thermal alerts
                logger.critical(
                    "Multiple thermal alerts - considering emergency shutdown"
                )
                # Emergency shutdown would be executed here if configured

    def start(self):
        """Start the monitoring daemon"""
        if self.running:
            logger.warning("Watchdog already running")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("SystemWatchdog started")

    def stop(self):
        """Stop the monitoring daemon"""
        if not self.running:
            return

        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("SystemWatchdog stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current watchdog status"""
        return {
            "running": self.running,
            "last_check": self.last_metrics.timestamp if self.last_metrics else None,
            "active_alerts": len(self.active_alerts),
            "total_alerts_today": len(
                [
                    a
                    for a in self.alert_history
                    if datetime.fromisoformat(a.timestamp).date()
                    == datetime.now().date()
                ]
            ),
            "current_metrics": asdict(self.last_metrics) if self.last_metrics else None,
            "active_alert_details": [
                asdict(alert) for alert in self.active_alerts.values()
            ],
        }

    def generate_daily_digest(self) -> str:
        """Generate daily system health digest"""
        today = datetime.now().date()
        today_alerts = [
            a
            for a in self.alert_history
            if datetime.fromisoformat(a.timestamp).date() == today
        ]

        critical_alerts = [a for a in today_alerts if a.level == AlertLevel.CRITICAL]
        warning_alerts = [a for a in today_alerts if a.level == AlertLevel.WARNING]

        digest = f"""# Daily System Health Digest - {today}

## Summary
- **Total Alerts**: {len(today_alerts)}
- **Critical Alerts**: {len(critical_alerts)}
- **Warning Alerts**: {len(warning_alerts)}
- **Active Alerts**: {len(self.active_alerts)}

## Current System Status
"""

        if self.last_metrics:
            digest += f"""
- **CPU Usage**: {self.last_metrics.cpu_usage:.1f}%
- **CPU Temperature**: {self.last_metrics.cpu_temp:.1f}°C" if self.last_metrics.cpu_temp else "N/A"
- **Memory Usage**: {self.last_metrics.memory_usage:.1f}%
- **Load Average**: {self.last_metrics.load_average[0]:.2f}
- **Process Count**: {self.last_metrics.process_count}
"""

        if critical_alerts:
            digest += "\n## Critical Alerts Today\n"
            for alert in critical_alerts[-5:]:  # Last 5
                digest += f"- {alert.timestamp}: {alert.message}\n"

        if self.active_alerts:
            digest += "\n## Currently Active Alerts\n"
            for alert in self.active_alerts.values():
                digest += f"- {alert.level.value.upper()}: {alert.message}\n"

        return digest


# Global watchdog instance
system_watchdog = SystemWatchdog()


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down watchdog")
    system_watchdog.stop()
    sys.exit(0)


if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(
        description="System Autonomy Core - Health Watchdog"
    )
    parser.add_argument("--start", action="store_true", help="Start monitoring daemon")
    parser.add_argument("--stop", action="store_true", help="Stop monitoring daemon")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--digest", action="store_true", help="Generate daily digest")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")

    args = parser.parse_args()

    if args.start or args.daemon:
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        system_watchdog.start()

        if args.daemon:
            logger.info("Running in daemon mode")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                system_watchdog.stop()
        else:
            print("✅ System monitoring started")
            print("Press Ctrl+C to stop")
            try:
                while system_watchdog.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                system_watchdog.stop()

    elif args.stop:
        system_watchdog.stop()
        print("✅ System monitoring stopped")

    elif args.status:
        status = system_watchdog.get_status()
        print("📊 System Watchdog Status:")
        print(f"   Running: {status['running']}")
        print(f"   Last Check: {status['last_check']}")
        print(f"   Active Alerts: {status['active_alerts']}")
        print(f"   Alerts Today: {status['total_alerts_today']}")

        if status["active_alert_details"]:
            print("\n🚨 Active Alerts:")
            for alert in status["active_alert_details"]:
                print(f"   - {alert['level'].upper()}: {alert['message']}")

    elif args.digest:
        digest = system_watchdog.generate_daily_digest()
        print(digest)

    else:
        print("Usage: python system_watchdog.py [--start|--status|--digest|--daemon]")
