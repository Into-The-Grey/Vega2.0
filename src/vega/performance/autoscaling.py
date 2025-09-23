"""
Personal Autoscaling System

Provides intelligent resource scaling, process management, and dynamic
optimization for single-user environments and local workloads.
"""

import asyncio
import logging
import time
import threading
import os
import signal
import subprocess
import psutil
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import json
import statistics
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


class ScalingPolicy(Enum):
    """Scaling policies"""

    REACTIVE = "reactive"  # Scale based on current metrics
    PREDICTIVE = "predictive"  # Scale based on predicted demand
    SCHEDULED = "scheduled"  # Scale based on time schedules
    THRESHOLD_BASED = "threshold"  # Scale based on metric thresholds
    INTELLIGENT = "intelligent"  # ML-based scaling decisions


class ScaleDirection(Enum):
    """Scaling directions"""

    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"


class ProcessState(Enum):
    """Process states"""

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    SCALING = "scaling"


class ResourceType(Enum):
    """Types of resources to monitor"""

    CPU = "cpu"
    MEMORY = "memory"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CONNECTIONS = "connections"
    QUEUE_SIZE = "queue_size"
    RESPONSE_TIME = "response_time"
    CUSTOM = "custom"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""

    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb_s: float
    disk_write_mb_s: float
    network_recv_mb_s: float
    network_sent_mb_s: float
    connections: int = 0
    queue_size: int = 0
    response_time_ms: Optional[float] = None
    custom_metrics: Dict[str, float] = None

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class ScalingRule:
    """Scaling rule definition"""

    rule_id: str
    name: str
    resource_type: ResourceType
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_adjustment: int
    scale_down_adjustment: int
    cooldown_seconds: int = 300
    evaluation_periods: int = 2
    datapoints_to_alarm: int = 2
    enabled: bool = True


@dataclass
class ProcessConfig:
    """Process configuration for scaling"""

    process_id: str
    name: str
    command: List[str]
    working_directory: str
    environment: Dict[str, str]
    min_instances: int = 1
    max_instances: int = 10
    desired_instances: int = 1
    health_check_command: Optional[List[str]] = None
    health_check_interval: int = 30
    startup_grace_period: int = 60
    shutdown_timeout: int = 30
    port_range_start: Optional[int] = None
    resource_limits: Dict[str, Any] = None

    def __post_init__(self):
        if self.environment is None:
            self.environment = {}
        if self.resource_limits is None:
            self.resource_limits = {}


@dataclass
class ProcessInstance:
    """Running process instance"""

    instance_id: str
    process_config_id: str
    pid: Optional[int]
    port: Optional[int]
    state: ProcessState
    start_time: Optional[datetime]
    last_health_check: Optional[datetime]
    health_status: bool = True
    resource_usage: Optional[ResourceMetrics] = None
    restart_count: int = 0

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.last_health_check, str):
            self.last_health_check = datetime.fromisoformat(self.last_health_check)


@dataclass
class ScalingEvent:
    """Scaling event record"""

    event_id: str
    process_config_id: str
    timestamp: datetime
    direction: ScaleDirection
    reason: str
    rule_id: Optional[str]
    old_count: int
    new_count: int
    success: bool
    error_message: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class AutoscalerStats:
    """Autoscaler statistics"""

    total_scaling_events: int
    scale_up_events: int
    scale_down_events: int
    successful_events: int
    failed_events: int
    active_processes: int
    total_instances: int
    average_cpu_usage: float
    average_memory_usage: float
    last_scaling_event: Optional[datetime]
    last_updated: datetime

    def __post_init__(self):
        if isinstance(self.last_scaling_event, str):
            self.last_scaling_event = datetime.fromisoformat(self.last_scaling_event)
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)


class ResourceMonitor:
    """
    System resource monitoring
    """

    def __init__(self, monitoring_interval: float = 10.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.process_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

        # Network baseline for calculating rates
        self.last_network_stats = None
        self.last_disk_stats = None
        self.last_measurement_time = None

    async def start_monitoring(self):
        """Start resource monitoring"""
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Resource monitoring started")

    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_system_metrics(self) -> ResourceMetrics:
        """Collect system resource metrics"""
        current_time = time.time()

        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_mb = (memory.total - memory.available) / (1024 * 1024)
        memory_percent = memory.percent

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb_s = 0.0
        disk_write_mb_s = 0.0

        if self.last_disk_stats and self.last_measurement_time:
            time_delta = current_time - self.last_measurement_time
            if time_delta > 0:
                read_delta = disk_io.read_bytes - self.last_disk_stats.read_bytes
                write_delta = disk_io.write_bytes - self.last_disk_stats.write_bytes
                disk_read_mb_s = (read_delta / time_delta) / (1024 * 1024)
                disk_write_mb_s = (write_delta / time_delta) / (1024 * 1024)

        self.last_disk_stats = disk_io

        # Network I/O
        network_io = psutil.net_io_counters()
        network_recv_mb_s = 0.0
        network_sent_mb_s = 0.0

        if self.last_network_stats and self.last_measurement_time:
            time_delta = current_time - self.last_measurement_time
            if time_delta > 0:
                recv_delta = network_io.bytes_recv - self.last_network_stats.bytes_recv
                sent_delta = network_io.bytes_sent - self.last_network_stats.bytes_sent
                network_recv_mb_s = (recv_delta / time_delta) / (1024 * 1024)
                network_sent_mb_s = (sent_delta / time_delta) / (1024 * 1024)

        self.last_network_stats = network_io
        self.last_measurement_time = current_time

        # Connection count (approximate)
        connections = len(psutil.net_connections())

        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            disk_read_mb_s=disk_read_mb_s,
            disk_write_mb_s=disk_write_mb_s,
            network_recv_mb_s=network_recv_mb_s,
            network_sent_mb_s=network_sent_mb_s,
            connections=connections,
        )

    async def collect_process_metrics(self, pid: int) -> Optional[ResourceMetrics]:
        """Collect metrics for specific process"""
        try:
            process = psutil.Process(pid)

            # CPU and memory usage
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_percent = process.memory_percent()

            # Get system memory for percentage calculation
            system_memory = psutil.virtual_memory()
            memory_percent = (memory_info.rss / system_memory.total) * 100

            # I/O stats
            try:
                io_counters = process.io_counters()
                # Note: These are cumulative, not rates
                disk_read_mb_s = io_counters.read_bytes / (1024 * 1024)
                disk_write_mb_s = io_counters.write_bytes / (1024 * 1024)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                disk_read_mb_s = 0.0
                disk_write_mb_s = 0.0

            # Connection count
            try:
                connections = len(process.connections())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                connections = 0

            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_read_mb_s=disk_read_mb_s,
                disk_write_mb_s=disk_write_mb_s,
                network_recv_mb_s=0.0,  # Process-level network I/O not easily available
                network_sent_mb_s=0.0,
                connections=connections,
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not collect metrics for PID {pid}: {e}")
            return None

    def get_recent_metrics(self, seconds: int = 300) -> List[ResourceMetrics]:
        """Get metrics from last N seconds"""
        cutoff_time = datetime.now() - timedelta(seconds=seconds)
        return [
            metrics
            for metrics in self.metrics_history
            if metrics.timestamp >= cutoff_time
        ]

    def get_average_metrics(self, seconds: int = 300) -> Optional[ResourceMetrics]:
        """Get average metrics over time period"""
        recent_metrics = self.get_recent_metrics(seconds)
        if not recent_metrics:
            return None

        return ResourceMetrics(
            timestamp=datetime.now(),
            cpu_percent=statistics.mean(m.cpu_percent for m in recent_metrics),
            memory_mb=statistics.mean(m.memory_mb for m in recent_metrics),
            memory_percent=statistics.mean(m.memory_percent for m in recent_metrics),
            disk_read_mb_s=statistics.mean(m.disk_read_mb_s for m in recent_metrics),
            disk_write_mb_s=statistics.mean(m.disk_write_mb_s for m in recent_metrics),
            network_recv_mb_s=statistics.mean(
                m.network_recv_mb_s for m in recent_metrics
            ),
            network_sent_mb_s=statistics.mean(
                m.network_sent_mb_s for m in recent_metrics
            ),
            connections=int(statistics.mean(m.connections for m in recent_metrics)),
        )


class ProcessManager:
    """
    Manages process lifecycle and scaling
    """

    def __init__(self):
        self.process_configs: Dict[str, ProcessConfig] = {}
        self.process_instances: Dict[str, ProcessInstance] = {}
        self.next_port: Dict[str, int] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.resource_monitor = ResourceMonitor()

    def add_process_config(self, config: ProcessConfig):
        """Add process configuration"""
        self.process_configs[config.process_id] = config

        # Initialize port allocation
        if config.port_range_start:
            self.next_port[config.process_id] = config.port_range_start

        logger.info(f"Added process config: {config.name}")

    def remove_process_config(self, process_id: str):
        """Remove process configuration"""
        if process_id in self.process_configs:
            # Stop all instances first
            instances_to_stop = [
                instance_id
                for instance_id, instance in self.process_instances.items()
                if instance.process_config_id == process_id
            ]

            for instance_id in instances_to_stop:
                asyncio.create_task(self.stop_instance(instance_id))

            del self.process_configs[process_id]
            logger.info(f"Removed process config: {process_id}")

    async def start_instance(self, process_id: str) -> Optional[str]:
        """Start new process instance"""
        if process_id not in self.process_configs:
            logger.error(f"Process config not found: {process_id}")
            return None

        config = self.process_configs[process_id]

        # Generate instance ID
        instance_id = f"{process_id}_{int(time.time() * 1000)}"

        # Allocate port if needed
        port = None
        if config.port_range_start:
            port = self.next_port[process_id]
            self.next_port[process_id] += 1

        # Create instance record
        instance = ProcessInstance(
            instance_id=instance_id,
            process_config_id=process_id,
            pid=None,
            port=port,
            state=ProcessState.STARTING,
            start_time=datetime.now(),
        )

        self.process_instances[instance_id] = instance

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(config.environment)

            if port:
                env["PORT"] = str(port)

            # Start process
            process = await asyncio.create_subprocess_exec(
                *config.command,
                cwd=config.working_directory,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            instance.pid = process.pid
            instance.state = ProcessState.RUNNING

            logger.info(f"Started instance {instance_id} with PID {process.pid}")

            # Start health checking
            if config.health_check_command:
                health_task = asyncio.create_task(self._health_check_loop(instance_id))
                self.health_check_tasks[instance_id] = health_task

            return instance_id

        except Exception as e:
            instance.state = ProcessState.FAILED
            logger.error(f"Failed to start instance {instance_id}: {e}")
            return None

    async def stop_instance(self, instance_id: str):
        """Stop process instance"""
        if instance_id not in self.process_instances:
            return

        instance = self.process_instances[instance_id]
        instance.state = ProcessState.STOPPING

        # Stop health checking
        if instance_id in self.health_check_tasks:
            self.health_check_tasks[instance_id].cancel()
            del self.health_check_tasks[instance_id]

        # Stop process
        if instance.pid:
            try:
                process = psutil.Process(instance.pid)

                # Try graceful shutdown first
                process.terminate()

                # Wait for graceful shutdown
                config = self.process_configs[instance.process_config_id]
                try:
                    process.wait(timeout=config.shutdown_timeout)
                except psutil.TimeoutExpired:
                    # Force kill if graceful shutdown failed
                    process.kill()
                    process.wait()

                logger.info(f"Stopped instance {instance_id}")

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logger.warning(f"Error stopping instance {instance_id}: {e}")

        instance.state = ProcessState.STOPPED
        instance.pid = None

    async def scale_process(self, process_id: str, target_instances: int) -> bool:
        """Scale process to target instance count"""
        if process_id not in self.process_configs:
            return False

        config = self.process_configs[process_id]

        # Validate target within limits
        target_instances = max(
            config.min_instances, min(config.max_instances, target_instances)
        )

        # Get current instances
        current_instances = [
            instance
            for instance in self.process_instances.values()
            if (
                instance.process_config_id == process_id
                and instance.state in [ProcessState.RUNNING, ProcessState.STARTING]
            )
        ]

        current_count = len(current_instances)

        if target_instances > current_count:
            # Scale up
            for _ in range(target_instances - current_count):
                await self.start_instance(process_id)

        elif target_instances < current_count:
            # Scale down
            instances_to_stop = current_instances[target_instances:]
            for instance in instances_to_stop:
                await self.stop_instance(instance.instance_id)

        # Update desired count
        config.desired_instances = target_instances

        logger.info(
            f"Scaled {config.name} from {current_count} to {target_instances} instances"
        )
        return True

    async def _health_check_loop(self, instance_id: str):
        """Health check loop for process instance"""
        while instance_id in self.process_instances:
            try:
                instance = self.process_instances[instance_id]
                config = self.process_configs[instance.process_config_id]

                if not config.health_check_command:
                    break

                # Run health check
                try:
                    result = await asyncio.create_subprocess_exec(
                        *config.health_check_command,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    await result.wait()
                    instance.health_status = result.returncode == 0
                    instance.last_health_check = datetime.now()

                    if not instance.health_status:
                        logger.warning(
                            f"Health check failed for instance {instance_id}"
                        )

                        # Restart unhealthy instance
                        await self.stop_instance(instance_id)
                        await self.start_instance(instance.process_config_id)
                        break

                except Exception as e:
                    logger.error(f"Health check error for {instance_id}: {e}")
                    instance.health_status = False

                await asyncio.sleep(config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {instance_id}: {e}")
                await asyncio.sleep(config.health_check_interval)

    def get_process_instances(self, process_id: str) -> List[ProcessInstance]:
        """Get all instances for a process"""
        return [
            instance
            for instance in self.process_instances.values()
            if instance.process_config_id == process_id
        ]

    def get_running_instances(self, process_id: str) -> List[ProcessInstance]:
        """Get running instances for a process"""
        return [
            instance
            for instance in self.process_instances.values()
            if (
                instance.process_config_id == process_id
                and instance.state == ProcessState.RUNNING
            )
        ]


class AutoscalerEngine:
    """
    Main autoscaling engine
    """

    def __init__(self, scaling_policy: ScalingPolicy = ScalingPolicy.INTELLIGENT):
        self.scaling_policy = scaling_policy
        self.process_manager = ProcessManager()
        self.resource_monitor = ResourceMonitor()
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_events: List[ScalingEvent] = []
        self.last_scaling_time: Dict[str, datetime] = {}

        self.stats = AutoscalerStats(
            total_scaling_events=0,
            scale_up_events=0,
            scale_down_events=0,
            successful_events=0,
            failed_events=0,
            active_processes=0,
            total_instances=0,
            average_cpu_usage=0.0,
            average_memory_usage=0.0,
            last_scaling_event=None,
            last_updated=datetime.now(),
        )

        self.is_running = False
        self.autoscale_task: Optional[asyncio.Task] = None
        self.evaluation_interval = 30.0  # seconds

    async def start(self):
        """Start autoscaler engine"""
        await self.resource_monitor.start_monitoring()

        self.is_running = True
        self.autoscale_task = asyncio.create_task(self._autoscaling_loop())

        logger.info("Autoscaler engine started")

    async def stop(self):
        """Stop autoscaler engine"""
        self.is_running = False

        if self.autoscale_task:
            self.autoscale_task.cancel()
            try:
                await self.autoscale_task
            except asyncio.CancelledError:
                pass

        await self.resource_monitor.stop_monitoring()

        logger.info("Autoscaler engine stopped")

    def add_scaling_rule(self, rule: ScalingRule):
        """Add scaling rule"""
        self.scaling_rules[rule.rule_id] = rule
        logger.info(f"Added scaling rule: {rule.name}")

    def remove_scaling_rule(self, rule_id: str):
        """Remove scaling rule"""
        if rule_id in self.scaling_rules:
            del self.scaling_rules[rule_id]
            logger.info(f"Removed scaling rule: {rule_id}")

    def add_process_config(self, config: ProcessConfig):
        """Add process configuration"""
        self.process_manager.add_process_config(config)

    async def _autoscaling_loop(self):
        """Main autoscaling evaluation loop"""
        while self.is_running:
            try:
                await self._evaluate_scaling_decisions()
                self._update_stats()

                await asyncio.sleep(self.evaluation_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in autoscaling loop: {e}")
                await asyncio.sleep(self.evaluation_interval)

    async def _evaluate_scaling_decisions(self):
        """Evaluate and make scaling decisions"""
        for process_id, config in self.process_manager.process_configs.items():
            try:
                decision = await self._make_scaling_decision(process_id)

                if decision != ScaleDirection.MAINTAIN:
                    await self._execute_scaling_decision(process_id, decision)

            except Exception as e:
                logger.error(f"Error evaluating scaling for {process_id}: {e}")

    async def _make_scaling_decision(self, process_id: str) -> ScaleDirection:
        """Make scaling decision for process"""
        # Check cooldown period
        if process_id in self.last_scaling_time:
            time_since_last = datetime.now() - self.last_scaling_time[process_id]
            if time_since_last.total_seconds() < 300:  # 5 minute cooldown
                return ScaleDirection.MAINTAIN

        # Get current metrics
        avg_metrics = self.resource_monitor.get_average_metrics(300)  # 5 minutes
        if not avg_metrics:
            return ScaleDirection.MAINTAIN

        # Apply scaling policy
        if self.scaling_policy == ScalingPolicy.THRESHOLD_BASED:
            return await self._threshold_based_decision(process_id, avg_metrics)
        elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
            return await self._predictive_decision(process_id, avg_metrics)
        elif self.scaling_policy == ScalingPolicy.INTELLIGENT:
            return await self._intelligent_decision(process_id, avg_metrics)
        else:
            return await self._reactive_decision(process_id, avg_metrics)

    async def _threshold_based_decision(
        self, process_id: str, metrics: ResourceMetrics
    ) -> ScaleDirection:
        """Threshold-based scaling decision"""
        for rule in self.scaling_rules.values():
            if not rule.enabled:
                continue

            metric_value = self._get_metric_value(metrics, rule.resource_type)
            if metric_value is None:
                continue

            if metric_value >= rule.scale_up_threshold:
                return ScaleDirection.UP
            elif metric_value <= rule.scale_down_threshold:
                return ScaleDirection.DOWN

        return ScaleDirection.MAINTAIN

    async def _reactive_decision(
        self, process_id: str, metrics: ResourceMetrics
    ) -> ScaleDirection:
        """Reactive scaling decision"""
        # Simple reactive policy
        if metrics.cpu_percent > 80 or metrics.memory_percent > 85:
            return ScaleDirection.UP
        elif metrics.cpu_percent < 20 and metrics.memory_percent < 30:
            return ScaleDirection.DOWN

        return ScaleDirection.MAINTAIN

    async def _predictive_decision(
        self, process_id: str, metrics: ResourceMetrics
    ) -> ScaleDirection:
        """Predictive scaling decision"""
        # Simple trend analysis
        recent_metrics = self.resource_monitor.get_recent_metrics(600)  # 10 minutes
        if len(recent_metrics) < 5:
            return ScaleDirection.MAINTAIN

        # Calculate trend for CPU usage
        cpu_values = [m.cpu_percent for m in recent_metrics[-5:]]
        cpu_trend = statistics.mean(cpu_values[-3:]) - statistics.mean(cpu_values[:2])

        # Predict future load
        predicted_cpu = metrics.cpu_percent + (cpu_trend * 2)  # 2x trend extrapolation

        if predicted_cpu > 75:
            return ScaleDirection.UP
        elif predicted_cpu < 25 and metrics.cpu_percent < 40:
            return ScaleDirection.DOWN

        return ScaleDirection.MAINTAIN

    async def _intelligent_decision(
        self, process_id: str, metrics: ResourceMetrics
    ) -> ScaleDirection:
        """Intelligent scaling decision using multiple factors"""
        score = 0

        # CPU factor
        if metrics.cpu_percent > 70:
            score += 2
        elif metrics.cpu_percent > 50:
            score += 1
        elif metrics.cpu_percent < 20:
            score -= 1
        elif metrics.cpu_percent < 10:
            score -= 2

        # Memory factor
        if metrics.memory_percent > 80:
            score += 2
        elif metrics.memory_percent > 60:
            score += 1
        elif metrics.memory_percent < 30:
            score -= 1
        elif metrics.memory_percent < 20:
            score -= 2

        # I/O factor
        total_io = metrics.disk_read_mb_s + metrics.disk_write_mb_s
        if total_io > 50:  # High I/O
            score += 1

        # Network factor
        total_network = metrics.network_recv_mb_s + metrics.network_sent_mb_s
        if total_network > 10:  # High network usage
            score += 1

        # Connection factor
        if metrics.connections > 100:
            score += 1
        elif metrics.connections < 10:
            score -= 1

        # Time-based factor (scale up during peak hours)
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 17:  # Business hours
            score += 1

        # Make decision based on score
        if score >= 3:
            return ScaleDirection.UP
        elif score <= -3:
            return ScaleDirection.DOWN
        else:
            return ScaleDirection.MAINTAIN

    def _get_metric_value(
        self, metrics: ResourceMetrics, resource_type: ResourceType
    ) -> Optional[float]:
        """Get metric value by type"""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent
        elif resource_type == ResourceType.DISK_IO:
            return metrics.disk_read_mb_s + metrics.disk_write_mb_s
        elif resource_type == ResourceType.NETWORK_IO:
            return metrics.network_recv_mb_s + metrics.network_sent_mb_s
        elif resource_type == ResourceType.CONNECTIONS:
            return metrics.connections
        elif resource_type == ResourceType.RESPONSE_TIME:
            return metrics.response_time_ms
        else:
            return None

    async def _execute_scaling_decision(
        self, process_id: str, direction: ScaleDirection
    ):
        """Execute scaling decision"""
        config = self.process_manager.process_configs[process_id]
        running_instances = self.process_manager.get_running_instances(process_id)
        current_count = len(running_instances)

        if direction == ScaleDirection.UP:
            new_count = min(current_count + 1, config.max_instances)
        else:  # ScaleDirection.DOWN
            new_count = max(current_count - 1, config.min_instances)

        if new_count == current_count:
            return  # No change needed

        # Record scaling event
        event_id = f"scale_{int(time.time() * 1000)}"
        event = ScalingEvent(
            event_id=event_id,
            process_config_id=process_id,
            timestamp=datetime.now(),
            direction=direction,
            reason=f"Scaling based on {self.scaling_policy.value} policy",
            rule_id=None,
            old_count=current_count,
            new_count=new_count,
            success=False,
        )

        try:
            # Execute scaling
            success = await self.process_manager.scale_process(process_id, new_count)

            event.success = success
            self.last_scaling_time[process_id] = datetime.now()

            logger.info(
                f"Scaling {direction.value}: {config.name} "
                f"from {current_count} to {new_count} instances"
            )

        except Exception as e:
            event.error_message = str(e)
            logger.error(f"Scaling failed for {process_id}: {e}")

        self.scaling_events.append(event)

        # Keep only recent events
        cutoff_time = datetime.now() - timedelta(days=7)
        self.scaling_events = [
            e for e in self.scaling_events if e.timestamp >= cutoff_time
        ]

    def _update_stats(self):
        """Update autoscaler statistics"""
        # Count scaling events
        total_events = len(self.scaling_events)
        scale_up_events = sum(
            1 for e in self.scaling_events if e.direction == ScaleDirection.UP
        )
        scale_down_events = sum(
            1 for e in self.scaling_events if e.direction == ScaleDirection.DOWN
        )
        successful_events = sum(1 for e in self.scaling_events if e.success)

        # Count processes and instances
        active_processes = len(self.process_manager.process_configs)
        total_instances = len(self.process_manager.process_instances)

        # Get average resource usage
        avg_metrics = self.resource_monitor.get_average_metrics(300)
        avg_cpu = avg_metrics.cpu_percent if avg_metrics else 0.0
        avg_memory = avg_metrics.memory_percent if avg_metrics else 0.0

        # Get last scaling event
        last_event = None
        if self.scaling_events:
            last_event = max(self.scaling_events, key=lambda e: e.timestamp).timestamp

        self.stats = AutoscalerStats(
            total_scaling_events=total_events,
            scale_up_events=scale_up_events,
            scale_down_events=scale_down_events,
            successful_events=successful_events,
            failed_events=total_events - successful_events,
            active_processes=active_processes,
            total_instances=total_instances,
            average_cpu_usage=avg_cpu,
            average_memory_usage=avg_memory,
            last_scaling_event=last_event,
            last_updated=datetime.now(),
        )

    def get_autoscaler_stats(self) -> Dict[str, Any]:
        """Get comprehensive autoscaler statistics"""
        stats_dict = asdict(self.stats)

        # Add process details
        stats_dict["processes"] = {}
        for process_id, config in self.process_manager.process_configs.items():
            instances = self.process_manager.get_process_instances(process_id)
            running_instances = self.process_manager.get_running_instances(process_id)

            stats_dict["processes"][process_id] = {
                "name": config.name,
                "desired_instances": config.desired_instances,
                "min_instances": config.min_instances,
                "max_instances": config.max_instances,
                "total_instances": len(instances),
                "running_instances": len(running_instances),
                "instances": [asdict(instance) for instance in instances],
            }

        # Add scaling rules
        stats_dict["scaling_rules"] = {
            rule_id: asdict(rule) for rule_id, rule in self.scaling_rules.items()
        }

        # Add recent scaling events
        stats_dict["recent_events"] = [
            asdict(event) for event in self.scaling_events[-10:]  # Last 10 events
        ]

        # Add current resource metrics
        current_metrics = self.resource_monitor.get_average_metrics(60)  # Last minute
        if current_metrics:
            stats_dict["current_metrics"] = asdict(current_metrics)

        return stats_dict


# Demo and testing functions
async def demo_autoscaling():
    """Demonstrate autoscaling capabilities"""

    print("Personal Autoscaling Demo")

    # Create autoscaler
    autoscaler = AutoscalerEngine(ScalingPolicy.INTELLIGENT)

    # Add process configuration
    process_config = ProcessConfig(
        process_id="demo-service",
        name="Demo Service",
        command=[
            "python3",
            "-c",
            "import time; print('Service running...'); time.sleep(3600)",
        ],
        working_directory="/tmp",
        environment={"SERVICE_NAME": "demo"},
        min_instances=1,
        max_instances=5,
        desired_instances=2,
    )

    autoscaler.add_process_config(process_config)

    # Add scaling rules
    cpu_rule = ScalingRule(
        rule_id="cpu-rule",
        name="CPU Scaling Rule",
        resource_type=ResourceType.CPU,
        scale_up_threshold=70.0,
        scale_down_threshold=30.0,
        scale_up_adjustment=1,
        scale_down_adjustment=1,
        cooldown_seconds=180,
    )

    memory_rule = ScalingRule(
        rule_id="memory-rule",
        name="Memory Scaling Rule",
        resource_type=ResourceType.MEMORY,
        scale_up_threshold=80.0,
        scale_down_threshold=40.0,
        scale_up_adjustment=1,
        scale_down_adjustment=1,
        cooldown_seconds=180,
    )

    autoscaler.add_scaling_rule(cpu_rule)
    autoscaler.add_scaling_rule(memory_rule)

    print(f"Added process config: {process_config.name}")
    print(f"Added 2 scaling rules")

    # Start autoscaler
    await autoscaler.start()

    # Start initial instances
    print("\nStarting initial instances...")
    for _ in range(process_config.desired_instances):
        instance_id = await autoscaler.process_manager.start_instance(
            process_config.process_id
        )
        if instance_id:
            print(f"Started instance: {instance_id}")

    # Monitor for a short period
    print("\nMonitoring autoscaling behavior...")
    for i in range(10):
        await asyncio.sleep(5)

        # Get current stats
        stats = autoscaler.get_autoscaler_stats()

        print(f"\nIteration {i+1}:")
        print(f"- CPU Usage: {stats['average_cpu_usage']:.1f}%")
        print(f"- Memory Usage: {stats['average_memory_usage']:.1f}%")
        print(
            f"- Running Instances: {stats['processes']['demo-service']['running_instances']}"
        )
        print(f"- Total Scaling Events: {stats['total_scaling_events']}")

        if stats["recent_events"]:
            latest_event = stats["recent_events"][-1]
            print(
                f"- Latest Event: {latest_event['direction']} scaling at {latest_event['timestamp']}"
            )

    # Get final statistics
    final_stats = autoscaler.get_autoscaler_stats()
    print(f"\nFinal Autoscaler Statistics:")
    print(f"- Total scaling events: {final_stats['total_scaling_events']}")
    print(f"- Scale up events: {final_stats['scale_up_events']}")
    print(f"- Scale down events: {final_stats['scale_down_events']}")
    print(
        f"- Success rate: {final_stats['successful_events']}/{final_stats['total_scaling_events']}"
    )
    print(f"- Active processes: {final_stats['active_processes']}")
    print(f"- Total instances: {final_stats['total_instances']}")

    # Stop autoscaler
    await autoscaler.stop()

    return autoscaler


if __name__ == "__main__":
    asyncio.run(demo_autoscaling())
