"""
System Autonomy Core (SAC) - Phase 1: System Introspection and Hardware Enumeration

This module provides comprehensive hardware scanning and system state analysis.
It creates detailed snapshots of the system's hardware, performance metrics,
and operational status for baseline establishment and anomaly detection.

Hardware Target: AMD Ryzen 9 3900X, 128GB DDR4, GTX 1660 Super + Quadro P1000
Environment: Linux Ubuntu rackmount server, local-only access

Author: Vega2.0 Autonomous AI System
"""

import json
import os
import subprocess
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

try:
    import psutil
except ImportError:
    print("WARNING: psutil not installed. Install with: pip install psutil")
    psutil = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/ncacord/Vega2.0/sac/logs/system_probe.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class CPUInfo:
    """CPU information structure"""

    model: str
    cores_physical: int
    cores_logical: int
    frequency_current: float
    frequency_max: float
    frequency_min: float
    temperature: Optional[float]
    usage_percent: float
    architecture: str
    cache_l1: Optional[str]
    cache_l2: Optional[str]
    cache_l3: Optional[str]
    virtualization: bool
    features: List[str]


@dataclass
class GPUInfo:
    """GPU information structure"""

    name: str
    memory_total: int
    memory_used: int
    memory_free: int
    temperature: Optional[float]
    power_draw: Optional[float]
    utilization: Optional[float]
    driver_version: str
    cuda_version: Optional[str]
    compute_capability: Optional[str]
    pci_bus: str


@dataclass
class MemoryInfo:
    """Memory information structure"""

    total: int
    available: int
    used: int
    free: int
    buffers: int
    cached: int
    swap_total: int
    swap_used: int
    swap_free: int
    ecc_enabled: Optional[bool]
    dimm_count: int
    speed_mhz: Optional[int]


@dataclass
class DiskInfo:
    """Disk information structure"""

    device: str
    mountpoint: str
    filesystem: str
    total: int
    used: int
    free: int
    percent_used: float
    smart_health: str
    temperature: Optional[float]
    read_iops: Optional[float]
    write_iops: Optional[float]
    model: str
    serial: str
    interface: str


@dataclass
class NetworkInfo:
    """Network interface information"""

    interface: str
    ip_address: str
    netmask: str
    broadcast: str
    speed_mbps: Optional[int]
    duplex: str
    status: str
    packets_sent: int
    packets_recv: int
    bytes_sent: int
    bytes_recv: int
    errors_in: int
    errors_out: int
    drops_in: int
    drops_out: int


@dataclass
class SystemSnapshot:
    """Complete system snapshot structure"""

    timestamp: str
    hostname: str
    uptime: float
    boot_time: str
    os_info: Dict[str, str]
    cpu: CPUInfo
    gpus: List[GPUInfo]
    memory: MemoryInfo
    disks: List[DiskInfo]
    network: List[NetworkInfo]
    processes_count: int
    load_average: Tuple[float, float, float]
    thermal_zones: Dict[str, float]
    anomalies: List[str]
    health_score: float
    snapshot_hash: str


class SystemProbe:
    """
    Advanced system introspection and hardware enumeration engine.

    Provides comprehensive hardware scanning, baseline establishment,
    anomaly detection, and automated reporting capabilities.
    """

    def __init__(self, config_path: str = "/home/ncacord/Vega2.0/sac/config"):
        self.config_path = Path(config_path)
        self.snapshots_path = Path("/home/ncacord/Vega2.0/sac/snapshots")
        self.baseline_path = self.config_path / "baseline_profile.json"

        # Ensure directories exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.snapshots_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()
        self.baseline = self._load_baseline()

        logger.info("SystemProbe initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load system probe configuration"""
        config_file = self.config_path / "system_probe_config.json"

        default_config = {
            "thresholds": {
                "cpu_temp_warning": 70.0,
                "cpu_temp_critical": 85.0,
                "cpu_usage_warning": 80.0,
                "cpu_usage_critical": 95.0,
                "memory_usage_warning": 85.0,
                "memory_usage_critical": 95.0,
                "disk_usage_warning": 80.0,
                "disk_usage_critical": 90.0,
                "gpu_temp_warning": 75.0,
                "gpu_temp_critical": 90.0,
            },
            "monitoring": {
                "enable_smart": True,
                "enable_thermal": True,
                "enable_network_stats": True,
                "snapshot_retention_days": 30,
            },
            "hardware": {
                "expected_cpu_cores": 24,  # Ryzen 9 3900X
                "expected_memory_gb": 128,
                "expected_gpu_count": 2,
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
        config_file = self.config_path / "system_probe_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline system profile"""
        if self.baseline_path.exists():
            try:
                with open(self.baseline_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading baseline: {e}")
        return None

    def _save_baseline(self, snapshot: SystemSnapshot):
        """Save current snapshot as baseline"""
        try:
            baseline_data = {
                "created": snapshot.timestamp,
                "cpu_model": snapshot.cpu.model,
                "cpu_cores": snapshot.cpu.cores_physical,
                "memory_total": snapshot.memory.total,
                "gpu_count": len(snapshot.gpus),
                "gpus": [
                    {"name": gpu.name, "memory": gpu.memory_total}
                    for gpu in snapshot.gpus
                ],
                "disk_count": len(snapshot.disks),
                "network_interfaces": len(snapshot.network),
                "os_info": snapshot.os_info,
            }

            with open(self.baseline_path, "w") as f:
                json.dump(baseline_data, f, indent=2)

            logger.info("Baseline profile saved")
        except Exception as e:
            logger.error(f"Error saving baseline: {e}")

    def _run_command(self, command: str) -> Tuple[bool, str]:
        """Execute system command safely"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timeout: {command}")
            return False, ""
        except Exception as e:
            logger.error(f"Command error: {command} - {e}")
            return False, ""

    def _get_cpu_info(self) -> CPUInfo:
        """Collect comprehensive CPU information"""
        if not psutil:
            logger.warning("psutil not available, using basic CPU info")
            return CPUInfo(
                model="Unknown",
                cores_physical=0,
                cores_logical=0,
                frequency_current=0,
                frequency_max=0,
                frequency_min=0,
                temperature=None,
                usage_percent=0,
                architecture="Unknown",
                cache_l1=None,
                cache_l2=None,
                cache_l3=None,
                virtualization=False,
                features=[],
            )

        # Basic CPU info
        cpu_freq = psutil.cpu_freq()
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get detailed CPU info from /proc/cpuinfo
        cpu_model = "Unknown"
        cache_info = {}
        features = []

        success, cpuinfo = self._run_command("cat /proc/cpuinfo")
        if success:
            for line in cpuinfo.split("\n"):
                if "model name" in line and cpu_model == "Unknown":
                    cpu_model = line.split(":")[1].strip()
                elif "flags" in line:
                    features = line.split(":")[1].strip().split()
                    break

        # Get cache info from lscpu
        success, lscpu_output = self._run_command("lscpu")
        if success:
            for line in lscpu_output.split("\n"):
                if "L1d cache" in line:
                    cache_info["l1"] = line.split(":")[1].strip()
                elif "L2 cache" in line:
                    cache_info["l2"] = line.split(":")[1].strip()
                elif "L3 cache" in line:
                    cache_info["l3"] = line.split(":")[1].strip()

        # Get CPU temperature
        temperature = None
        try:
            # Try multiple temperature sources
            temp_sources = [
                "sensors | grep 'Tctl:' | awk '{print $2}' | sed 's/+//g' | sed 's/°C//g'",
                "sensors | grep 'Core 0:' | awk '{print $3}' | sed 's/+//g' | sed 's/°C//g'",
                "cat /sys/class/thermal/thermal_zone0/temp",
            ]

            for cmd in temp_sources:
                success, temp_str = self._run_command(cmd)
                if success and temp_str:
                    try:
                        if "thermal_zone" in cmd:
                            temperature = (
                                float(temp_str) / 1000.0
                            )  # Convert from milli-celsius
                        else:
                            temperature = float(temp_str)
                        break
                    except ValueError:
                        continue
        except Exception as e:
            logger.warning(f"Could not get CPU temperature: {e}")

        # Check virtualization support
        virtualization = "vmx" in features or "svm" in features

        # Get architecture
        success, arch = self._run_command("uname -m")
        architecture = arch if success else "Unknown"

        return CPUInfo(
            model=cpu_model,
            cores_physical=psutil.cpu_count(logical=False),
            cores_logical=psutil.cpu_count(logical=True),
            frequency_current=cpu_freq.current if cpu_freq else 0,
            frequency_max=cpu_freq.max if cpu_freq else 0,
            frequency_min=cpu_freq.min if cpu_freq else 0,
            temperature=temperature,
            usage_percent=cpu_percent,
            architecture=architecture,
            cache_l1=cache_info.get("l1"),
            cache_l2=cache_info.get("l2"),
            cache_l3=cache_info.get("l3"),
            virtualization=virtualization,
            features=features[:20],  # Limit features list
        )

    def _get_gpu_info(self) -> List[GPUInfo]:
        """Collect GPU information from nvidia-smi"""
        gpus = []

        # Try nvidia-smi for NVIDIA GPUs
        success, nvidia_output = self._run_command(
            "nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,"
            "temperature.gpu,power.draw,utilization.gpu,driver_version,"
            "pci.bus_id --format=csv,noheader,nounits"
        )

        if success and nvidia_output:
            for line in nvidia_output.split("\n"):
                if not line.strip():
                    continue

                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    try:
                        # Get CUDA version
                        cuda_success, cuda_version = self._run_command(
                            "nvidia-smi | grep 'CUDA Version' | awk '{print $9}'"
                        )

                        # Get compute capability
                        compute_success, compute_cap = self._run_command(
                            f"nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id={len(gpus)}"
                        )

                        gpu = GPUInfo(
                            name=parts[0],
                            memory_total=int(parts[1])
                            * 1024
                            * 1024,  # Convert MB to bytes
                            memory_used=int(parts[2]) * 1024 * 1024,
                            memory_free=int(parts[3]) * 1024 * 1024,
                            temperature=(
                                float(parts[4])
                                if parts[4] != "[Not Supported]"
                                else None
                            ),
                            power_draw=(
                                float(parts[5])
                                if parts[5] != "[Not Supported]"
                                else None
                            ),
                            utilization=(
                                float(parts[6])
                                if parts[6] != "[Not Supported]"
                                else None
                            ),
                            driver_version=parts[7],
                            cuda_version=cuda_version if cuda_success else None,
                            compute_capability=compute_cap if compute_success else None,
                            pci_bus=parts[8] if len(parts) > 8 else "Unknown",
                        )
                        gpus.append(gpu)
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error parsing GPU info: {e}")

        # Try lspci for additional GPU detection
        success, lspci_output = self._run_command("lspci | grep -i vga")
        if success and not gpus:  # Only if nvidia-smi didn't find anything
            for line in lspci_output.split("\n"):
                if "VGA" in line or "Display" in line:
                    gpu = GPUInfo(
                        name=line.split(":")[-1].strip(),
                        memory_total=0,
                        memory_used=0,
                        memory_free=0,
                        temperature=None,
                        power_draw=None,
                        utilization=None,
                        driver_version="Unknown",
                        cuda_version=None,
                        compute_capability=None,
                        pci_bus=line.split()[0],
                    )
                    gpus.append(gpu)

        return gpus

    def _get_memory_info(self) -> MemoryInfo:
        """Collect memory information"""
        if not psutil:
            return MemoryInfo(
                total=0,
                available=0,
                used=0,
                free=0,
                buffers=0,
                cached=0,
                swap_total=0,
                swap_used=0,
                swap_free=0,
                ecc_enabled=None,
                dimm_count=0,
                speed_mhz=None,
            )

        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Get memory speed and DIMM count
        speed_mhz = None
        dimm_count = 0
        ecc_enabled = None

        success, dmidecode_output = self._run_command("sudo dmidecode -t memory")
        if success:
            for line in dmidecode_output.split("\n"):
                if "Speed:" in line and "MHz" in line:
                    try:
                        speed_str = line.split(":")[1].strip()
                        speed_mhz = int(speed_str.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif "Size:" in line and "MB" in line:
                    dimm_count += 1
                elif "Error Correction Type:" in line:
                    ecc_type = line.split(":")[1].strip()
                    ecc_enabled = ecc_type != "None"

        return MemoryInfo(
            total=mem.total,
            available=mem.available,
            used=mem.used,
            free=mem.free,
            buffers=getattr(mem, "buffers", 0),
            cached=getattr(mem, "cached", 0),
            swap_total=swap.total,
            swap_used=swap.used,
            swap_free=swap.free,
            ecc_enabled=ecc_enabled,
            dimm_count=dimm_count,
            speed_mhz=speed_mhz,
        )

    def _get_disk_info(self) -> List[DiskInfo]:
        """Collect disk information"""
        disks = []

        if not psutil:
            return disks

        # Get basic disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)

                # Get SMART info if available
                smart_health = "Unknown"
                temperature = None
                model = "Unknown"
                serial = "Unknown"
                interface = "Unknown"

                # Try to get device info
                device_name = partition.device.split("/")[-1]
                if device_name.startswith("sd") or device_name.startswith("nvme"):
                    # Get SMART health
                    success, smart_output = self._run_command(
                        f"sudo smartctl -H {partition.device}"
                    )
                    if success:
                        if "PASSED" in smart_output:
                            smart_health = "PASSED"
                        elif "FAILED" in smart_output:
                            smart_health = "FAILED"

                    # Get device info
                    success, info_output = self._run_command(
                        f"sudo smartctl -i {partition.device}"
                    )
                    if success:
                        for line in info_output.split("\n"):
                            if "Device Model:" in line or "Model Number:" in line:
                                model = line.split(":")[1].strip()
                            elif "Serial Number:" in line:
                                serial = line.split(":")[1].strip()
                            elif "Transport protocol:" in line:
                                interface = line.split(":")[1].strip()

                    # Get temperature
                    success, temp_output = self._run_command(
                        f"sudo smartctl -A {partition.device} | grep Temperature_Celsius"
                    )
                    if success and temp_output:
                        try:
                            temp_parts = temp_output.split()
                            if len(temp_parts) >= 10:
                                temperature = float(temp_parts[9])
                        except (ValueError, IndexError):
                            pass

                # Get I/O stats
                read_iops = None
                write_iops = None
                try:
                    io_stats = psutil.disk_io_counters(perdisk=True)
                    if device_name in io_stats:
                        stats = io_stats[device_name]
                        # These are cumulative, would need baseline for rate
                        read_iops = stats.read_count
                        write_iops = stats.write_count
                except Exception:
                    pass

                disk = DiskInfo(
                    device=partition.device,
                    mountpoint=partition.mountpoint,
                    filesystem=partition.fstype,
                    total=usage.total,
                    used=usage.used,
                    free=usage.free,
                    percent_used=(
                        (usage.used / usage.total * 100) if usage.total > 0 else 0
                    ),
                    smart_health=smart_health,
                    temperature=temperature,
                    read_iops=read_iops,
                    write_iops=write_iops,
                    model=model,
                    serial=serial,
                    interface=interface,
                )
                disks.append(disk)

            except PermissionError:
                logger.warning(f"Permission denied accessing {partition.mountpoint}")
            except Exception as e:
                logger.warning(f"Error getting disk info for {partition.device}: {e}")

        return disks

    def _get_network_info(self) -> List[NetworkInfo]:
        """Collect network interface information"""
        interfaces = []

        if not psutil:
            return interfaces

        # Get network interfaces
        net_if_addrs = psutil.net_if_addrs()
        net_if_stats = psutil.net_if_stats()
        net_io_counters = psutil.net_io_counters(pernic=True)

        for interface_name, addresses in net_if_addrs.items():
            if interface_name == "lo":  # Skip loopback
                continue

            # Find IPv4 address
            ip_address = ""
            netmask = ""
            broadcast = ""

            for addr in addresses:
                if addr.family == 2:  # AF_INET (IPv4)
                    ip_address = addr.address
                    netmask = addr.netmask
                    broadcast = addr.broadcast or ""
                    break

            # Get interface stats
            stats = net_if_stats.get(interface_name)
            io_stats = net_io_counters.get(interface_name)

            if stats and io_stats:
                # Get additional interface info
                speed_mbps = stats.speed if stats.speed != 0 else None
                duplex = (
                    "full"
                    if stats.duplex == 2
                    else "half" if stats.duplex == 1 else "unknown"
                )
                status = "up" if stats.isup else "down"

                interface = NetworkInfo(
                    interface=interface_name,
                    ip_address=ip_address,
                    netmask=netmask,
                    broadcast=broadcast,
                    speed_mbps=speed_mbps,
                    duplex=duplex,
                    status=status,
                    packets_sent=io_stats.packets_sent,
                    packets_recv=io_stats.packets_recv,
                    bytes_sent=io_stats.bytes_sent,
                    bytes_recv=io_stats.bytes_recv,
                    errors_in=io_stats.errin,
                    errors_out=io_stats.errout,
                    drops_in=io_stats.dropin,
                    drops_out=io_stats.dropout,
                )
                interfaces.append(interface)

        return interfaces

    def _get_thermal_zones(self) -> Dict[str, float]:
        """Get thermal zone temperatures"""
        thermal_zones = {}

        try:
            # Get thermal zones from /sys
            thermal_path = Path("/sys/class/thermal")
            if thermal_path.exists():
                for zone_dir in thermal_path.glob("thermal_zone*"):
                    zone_name = zone_dir.name
                    temp_file = zone_dir / "temp"
                    type_file = zone_dir / "type"

                    if temp_file.exists():
                        try:
                            with open(temp_file) as f:
                                temp_milli = int(f.read().strip())
                                temp_celsius = temp_milli / 1000.0

                            # Get zone type if available
                            zone_type = zone_name
                            if type_file.exists():
                                with open(type_file) as f:
                                    zone_type = f.read().strip()

                            thermal_zones[zone_type] = temp_celsius
                        except (ValueError, IOError):
                            pass
        except Exception as e:
            logger.warning(f"Error reading thermal zones: {e}")

        return thermal_zones

    def _detect_anomalies(self, snapshot: SystemSnapshot) -> List[str]:
        """Detect system anomalies based on thresholds and baseline"""
        anomalies = []
        thresholds = self.config["thresholds"]

        # CPU anomalies
        if (
            snapshot.cpu.temperature
            and snapshot.cpu.temperature > thresholds["cpu_temp_critical"]
        ):
            anomalies.append(
                f"CRITICAL: CPU temperature {snapshot.cpu.temperature:.1f}°C exceeds threshold"
            )
        elif (
            snapshot.cpu.temperature
            and snapshot.cpu.temperature > thresholds["cpu_temp_warning"]
        ):
            anomalies.append(
                f"WARNING: CPU temperature {snapshot.cpu.temperature:.1f}°C exceeds warning threshold"
            )

        if snapshot.cpu.usage_percent > thresholds["cpu_usage_critical"]:
            anomalies.append(
                f"CRITICAL: CPU usage {snapshot.cpu.usage_percent:.1f}% exceeds threshold"
            )
        elif snapshot.cpu.usage_percent > thresholds["cpu_usage_warning"]:
            anomalies.append(
                f"WARNING: CPU usage {snapshot.cpu.usage_percent:.1f}% exceeds warning threshold"
            )

        # Memory anomalies
        memory_percent = (snapshot.memory.used / snapshot.memory.total) * 100
        if memory_percent > thresholds["memory_usage_critical"]:
            anomalies.append(
                f"CRITICAL: Memory usage {memory_percent:.1f}% exceeds threshold"
            )
        elif memory_percent > thresholds["memory_usage_warning"]:
            anomalies.append(
                f"WARNING: Memory usage {memory_percent:.1f}% exceeds warning threshold"
            )

        # GPU anomalies
        for i, gpu in enumerate(snapshot.gpus):
            if gpu.temperature and gpu.temperature > thresholds["gpu_temp_critical"]:
                anomalies.append(
                    f"CRITICAL: GPU {i} temperature {gpu.temperature:.1f}°C exceeds threshold"
                )
            elif gpu.temperature and gpu.temperature > thresholds["gpu_temp_warning"]:
                anomalies.append(
                    f"WARNING: GPU {i} temperature {gpu.temperature:.1f}°C exceeds warning threshold"
                )

        # Disk anomalies
        for disk in snapshot.disks:
            if disk.percent_used > thresholds["disk_usage_critical"]:
                anomalies.append(
                    f"CRITICAL: Disk {disk.device} usage {disk.percent_used:.1f}% exceeds threshold"
                )
            elif disk.percent_used > thresholds["disk_usage_warning"]:
                anomalies.append(
                    f"WARNING: Disk {disk.device} usage {disk.percent_used:.1f}% exceeds warning threshold"
                )

            if disk.smart_health == "FAILED":
                anomalies.append(
                    f"CRITICAL: Disk {disk.device} SMART health check failed"
                )

        # Baseline comparison anomalies
        if self.baseline:
            expected_hw = self.config["hardware"]

            if snapshot.cpu.cores_physical != expected_hw["expected_cpu_cores"]:
                anomalies.append(
                    f"WARNING: CPU core count mismatch (expected {expected_hw['expected_cpu_cores']}, got {snapshot.cpu.cores_physical})"
                )

            memory_gb = snapshot.memory.total / (1024**3)
            if abs(memory_gb - expected_hw["expected_memory_gb"]) > 1:
                anomalies.append(
                    f"WARNING: Memory size mismatch (expected {expected_hw['expected_memory_gb']}GB, got {memory_gb:.1f}GB)"
                )

            if len(snapshot.gpus) != expected_hw["expected_gpu_count"]:
                anomalies.append(
                    f"WARNING: GPU count mismatch (expected {expected_hw['expected_gpu_count']}, got {len(snapshot.gpus)})"
                )

        return anomalies

    def _calculate_health_score(self, snapshot: SystemSnapshot) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0

        # CPU health impact
        if snapshot.cpu.temperature:
            if snapshot.cpu.temperature > 85:
                score -= 30
            elif snapshot.cpu.temperature > 70:
                score -= 15

        if snapshot.cpu.usage_percent > 95:
            score -= 20
        elif snapshot.cpu.usage_percent > 80:
            score -= 10

        # Memory health impact
        memory_percent = (snapshot.memory.used / snapshot.memory.total) * 100
        if memory_percent > 95:
            score -= 25
        elif memory_percent > 85:
            score -= 10

        # Disk health impact
        for disk in snapshot.disks:
            if disk.smart_health == "FAILED":
                score -= 50
            elif disk.percent_used > 90:
                score -= 15
            elif disk.percent_used > 80:
                score -= 5

        # GPU health impact
        for gpu in snapshot.gpus:
            if gpu.temperature and gpu.temperature > 90:
                score -= 20
            elif gpu.temperature and gpu.temperature > 75:
                score -= 10

        # Anomaly impact
        critical_anomalies = sum(1 for a in snapshot.anomalies if "CRITICAL" in a)
        warning_anomalies = sum(1 for a in snapshot.anomalies if "WARNING" in a)

        score -= critical_anomalies * 15
        score -= warning_anomalies * 5

        return max(0.0, min(100.0, score))

    def create_snapshot(self) -> SystemSnapshot:
        """Create a complete system snapshot"""
        logger.info("Creating system snapshot...")
        start_time = time.time()

        # Collect all system information
        timestamp = datetime.now().isoformat()

        # Basic system info
        hostname = os.uname().nodename
        boot_time = (
            datetime.fromtimestamp(psutil.boot_time()).isoformat()
            if psutil
            else "Unknown"
        )
        uptime = time.time() - psutil.boot_time() if psutil else 0

        # OS information
        success, os_release = self._run_command("cat /etc/os-release")
        os_info = {}
        if success:
            for line in os_release.split("\n"):
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os_info[key] = value.strip('"')

        # Kernel info
        success, kernel = self._run_command("uname -r")
        if success:
            os_info["KERNEL"] = kernel

        # Hardware collection
        cpu = self._get_cpu_info()
        gpus = self._get_gpu_info()
        memory = self._get_memory_info()
        disks = self._get_disk_info()
        network = self._get_network_info()

        # System stats
        processes_count = len(psutil.pids()) if psutil else 0
        load_average = os.getloadavg()
        thermal_zones = self._get_thermal_zones()

        # Create preliminary snapshot for anomaly detection
        preliminary_snapshot = SystemSnapshot(
            timestamp=timestamp,
            hostname=hostname,
            uptime=uptime,
            boot_time=boot_time,
            os_info=os_info,
            cpu=cpu,
            gpus=gpus,
            memory=memory,
            disks=disks,
            network=network,
            processes_count=processes_count,
            load_average=load_average,
            thermal_zones=thermal_zones,
            anomalies=[],
            health_score=0.0,
            snapshot_hash="",
        )

        # Detect anomalies
        anomalies = self._detect_anomalies(preliminary_snapshot)

        # Calculate health score
        health_score = self._calculate_health_score(preliminary_snapshot)

        # Create final snapshot with anomalies and health score
        snapshot_data = asdict(preliminary_snapshot)
        snapshot_data["anomalies"] = anomalies
        snapshot_data["health_score"] = health_score

        # Calculate hash
        snapshot_json = json.dumps(snapshot_data, sort_keys=True)
        snapshot_hash = hashlib.sha256(snapshot_json.encode()).hexdigest()[:16]

        final_snapshot = SystemSnapshot(
            timestamp=timestamp,
            hostname=hostname,
            uptime=uptime,
            boot_time=boot_time,
            os_info=os_info,
            cpu=cpu,
            gpus=gpus,
            memory=memory,
            disks=disks,
            network=network,
            processes_count=processes_count,
            load_average=load_average,
            thermal_zones=thermal_zones,
            anomalies=anomalies,
            health_score=health_score,
            snapshot_hash=snapshot_hash,
        )

        duration = time.time() - start_time
        logger.info(
            f"System snapshot completed in {duration:.2f}s - Health: {health_score:.1f}% - Anomalies: {len(anomalies)}"
        )

        return final_snapshot

    def save_snapshot(self, snapshot: SystemSnapshot) -> str:
        """Save snapshot to file and return filename"""
        timestamp = datetime.fromisoformat(snapshot.timestamp)
        filename = timestamp.strftime("%Y-%m-%d-%H.json")
        filepath = self.snapshots_path / filename

        try:
            with open(filepath, "w") as f:
                json.dump(asdict(snapshot), f, indent=2)

            logger.info(f"Snapshot saved: {filepath}")

            # Clean old snapshots
            self._cleanup_old_snapshots()

            return str(filepath)
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            raise

    def _cleanup_old_snapshots(self):
        """Remove snapshots older than retention period"""
        try:
            retention_days = self.config["monitoring"]["snapshot_retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            for snapshot_file in self.snapshots_path.glob("*.json"):
                try:
                    # Parse date from filename
                    date_str = snapshot_file.stem  # YYYY-MM-DD-HH
                    file_date = datetime.strptime(date_str, "%Y-%m-%d-%H")

                    if file_date < cutoff_date:
                        snapshot_file.unlink()
                        logger.debug(f"Removed old snapshot: {snapshot_file}")

                except ValueError:
                    # Skip files that don't match expected format
                    continue

        except Exception as e:
            logger.warning(f"Error cleaning old snapshots: {e}")

    def generate_report(self, snapshot: SystemSnapshot) -> str:
        """Generate markdown report from snapshot"""
        report = f"""# System Health Report
Generated: {snapshot.timestamp}
Hostname: {snapshot.hostname}
Health Score: {snapshot.health_score:.1f}/100

## System Overview
- **Uptime**: {snapshot.uptime/3600:.1f} hours
- **Boot Time**: {snapshot.boot_time}
- **Processes**: {snapshot.processes_count}
- **Load Average**: {snapshot.load_average[0]:.2f}, {snapshot.load_average[1]:.2f}, {snapshot.load_average[2]:.2f}

## CPU Information
- **Model**: {snapshot.cpu.model}
- **Cores**: {snapshot.cpu.cores_physical} physical, {snapshot.cpu.cores_logical} logical
- **Frequency**: {snapshot.cpu.frequency_current:.0f} MHz (max: {snapshot.cpu.frequency_max:.0f} MHz)
- **Usage**: {snapshot.cpu.usage_percent:.1f}%
- **Temperature**: {snapshot.cpu.temperature:.1f}°C" if snapshot.cpu.temperature else "N/A"
- **Architecture**: {snapshot.cpu.architecture}
- **Virtualization**: {'Enabled' if snapshot.cpu.virtualization else 'Disabled'}

## Memory Information
- **Total**: {snapshot.memory.total / (1024**3):.1f} GB
- **Used**: {snapshot.memory.used / (1024**3):.1f} GB ({(snapshot.memory.used/snapshot.memory.total)*100:.1f}%)
- **Available**: {snapshot.memory.available / (1024**3):.1f} GB
- **Swap**: {snapshot.memory.swap_used / (1024**3):.1f} GB / {snapshot.memory.swap_total / (1024**3):.1f} GB
- **DIMMs**: {snapshot.memory.dimm_count}
- **Speed**: {snapshot.memory.speed_mhz} MHz" if snapshot.memory.speed_mhz else "N/A"
- **ECC**: {'Enabled' if snapshot.memory.ecc_enabled else 'Disabled' if snapshot.memory.ecc_enabled is not None else 'Unknown'}

## GPU Information
"""

        if snapshot.gpus:
            for i, gpu in enumerate(snapshot.gpus):
                report += f"""
### GPU {i}: {gpu.name}
- **Memory**: {gpu.memory_used / (1024**3):.1f} GB / {gpu.memory_total / (1024**3):.1f} GB ({(gpu.memory_used/gpu.memory_total)*100:.1f}% used)
- **Temperature**: {gpu.temperature:.1f}°C" if gpu.temperature else "N/A"
- **Power Draw**: {gpu.power_draw:.1f}W" if gpu.power_draw else "N/A"
- **Utilization**: {gpu.utilization:.1f}%" if gpu.utilization else "N/A"
- **Driver**: {gpu.driver_version}
- **CUDA**: {gpu.cuda_version or 'N/A'}
- **PCI Bus**: {gpu.pci_bus}
"""
        else:
            report += "No GPUs detected\n"

        report += "\n## Storage Information\n"
        for disk in snapshot.disks:
            report += f"""
### {disk.device} ({disk.mountpoint})
- **Filesystem**: {disk.filesystem}
- **Capacity**: {disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB ({disk.percent_used:.1f}% used)
- **Model**: {disk.model}
- **Interface**: {disk.interface}
- **SMART Health**: {disk.smart_health}
- **Temperature**: {disk.temperature:.1f}°C" if disk.temperature else "N/A"
"""

        report += "\n## Network Information\n"
        for interface in snapshot.network:
            report += f"""
### {interface.interface}
- **IP Address**: {interface.ip_address}
- **Status**: {interface.status}
- **Speed**: {interface.speed_mbps} Mbps" if interface.speed_mbps else "Unknown"
- **Duplex**: {interface.duplex}
- **Packets**: {interface.packets_recv:,} received, {interface.packets_sent:,} sent
- **Bytes**: {interface.bytes_recv / (1024**3):.2f} GB received, {interface.bytes_sent / (1024**3):.2f} GB sent
- **Errors**: {interface.errors_in} in, {interface.errors_out} out
"""

        if snapshot.thermal_zones:
            report += "\n## Thermal Zones\n"
            for zone, temp in snapshot.thermal_zones.items():
                report += f"- **{zone}**: {temp:.1f}°C\n"

        if snapshot.anomalies:
            report += "\n## ⚠️ Anomalies Detected\n"
            for anomaly in snapshot.anomalies:
                report += f"- {anomaly}\n"
        else:
            report += "\n## ✅ No Anomalies Detected\n"

        report += f"\n---\n*Report generated by Vega2.0 System Autonomy Core*\n*Snapshot hash: {snapshot.snapshot_hash}*\n"

        return report

    def run_full_scan(self) -> Tuple[SystemSnapshot, str]:
        """Run complete system scan and generate report"""
        snapshot = self.create_snapshot()
        filepath = self.save_snapshot(snapshot)

        # Generate and save report
        report = self.generate_report(snapshot)
        report_path = filepath.replace(".json", "_report.md")

        try:
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved: {report_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")

        # Save as baseline if none exists
        if not self.baseline:
            self._save_baseline(snapshot)
            logger.info("First scan - saved as baseline profile")

        return snapshot, report_path


# Global system probe instance
system_probe = SystemProbe()

if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(
        description="System Autonomy Core - Hardware Enumeration"
    )
    parser.add_argument("--scan", action="store_true", help="Run full system scan")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument(
        "--baseline", action="store_true", help="Save current scan as baseline"
    )

    args = parser.parse_args()

    if args.scan:
        snapshot, report_path = system_probe.run_full_scan()
        print(f"✅ System scan completed")
        print(f"📊 Health Score: {snapshot.health_score:.1f}%")
        print(f"⚠️  Anomalies: {len(snapshot.anomalies)}")
        print(f"📄 Report: {report_path}")

        if snapshot.anomalies:
            print("\n🚨 Anomalies detected:")
            for anomaly in snapshot.anomalies:
                print(f"   - {anomaly}")
    else:
        print("Usage: python system_probe.py --scan")
