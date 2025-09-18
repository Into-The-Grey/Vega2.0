"""
System Autonomy Core (SAC) - Phase 4: Network Monitoring & Traffic Firewall

This module provides comprehensive network security monitoring with autonomous
threat detection, traffic analysis, connection monitoring, and automated
firewall management for complete network sovereignty.

Key Features:
- Real-time network traffic monitoring and analysis
- Intrusion detection and threat pattern recognition
- Automated firewall rule management
- Network interface monitoring and health checking
- Bandwidth usage tracking and anomaly detection
- Connection state analysis and port monitoring
- Autonomous threat response and mitigation
- Network topology discovery and mapping

Author: Vega2.0 Autonomous AI System
"""

import subprocess
import json
import time
import threading
import socket
import struct
import ipaddress
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging

try:
    import psutil
except ImportError:
    print(
        "WARNING: psutil required for network monitoring. Install with: pip install psutil"
    )
    psutil = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/home/ncacord/Vega2.0/sac/logs/net_guard.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Network threat severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConnectionState(Enum):
    """Network connection states"""

    ESTABLISHED = "established"
    LISTENING = "listening"
    TIME_WAIT = "time_wait"
    CLOSE_WAIT = "close_wait"
    FIN_WAIT = "fin_wait"
    SYN_SENT = "syn_sent"
    SYN_RECV = "syn_recv"


class TrafficDirection(Enum):
    """Traffic flow direction"""

    INBOUND = "inbound"
    OUTBOUND = "outbound"
    BIDIRECTIONAL = "bidirectional"


class FirewallAction(Enum):
    """Firewall rule actions"""

    ALLOW = "allow"
    DENY = "deny"
    DROP = "drop"
    REJECT = "reject"


@dataclass
class NetworkInterface:
    """Network interface information"""

    name: str
    status: str
    mtu: int
    ip_addresses: List[str]
    mac_address: str
    speed: Optional[int]
    duplex: Optional[str]
    rx_bytes: int
    tx_bytes: int
    rx_packets: int
    tx_packets: int
    rx_errors: int
    tx_errors: int
    rx_dropped: int
    tx_dropped: int


@dataclass
class NetworkConnection:
    """Active network connection"""

    local_address: str
    local_port: int
    remote_address: str
    remote_port: int
    status: ConnectionState
    pid: Optional[int]
    process_name: Optional[str]
    protocol: str
    family: str
    created_time: Optional[str]


@dataclass
class ThreatEvent:
    """Network security threat event"""

    timestamp: str
    threat_id: str
    threat_level: ThreatLevel
    source_ip: str
    destination_ip: str
    source_port: Optional[int]
    destination_port: Optional[int]
    protocol: str
    description: str
    evidence: Dict[str, Any]
    mitigation_action: Optional[str]
    resolved: bool = False


@dataclass
class FirewallRule:
    """Firewall rule definition"""

    rule_id: str
    action: FirewallAction
    protocol: Optional[str]
    source_ip: Optional[str]
    source_port: Optional[str]
    destination_ip: Optional[str]
    destination_port: Optional[str]
    direction: TrafficDirection
    description: str
    created_time: str
    expires_at: Optional[str]
    hit_count: int = 0


@dataclass
class NetworkStats:
    """Network statistics snapshot"""

    timestamp: str
    interfaces: List[NetworkInterface]
    connections: List[NetworkConnection]
    total_bandwidth_usage: Dict[str, int]  # interface -> bytes per second
    suspicious_connections: int
    blocked_attempts: int
    active_threats: int
    firewall_rules_count: int


class NetworkGuard:
    """
    Advanced network security monitoring and firewall management system
    with autonomous threat detection and response capabilities.
    """

    def __init__(self, config_path: str = "/home/ncacord/Vega2.0/sac/config"):
        self.config_path = Path(config_path)
        self.logs_path = Path("/home/ncacord/Vega2.0/sac/logs")
        self.threats_log = self.logs_path / "network_threats.jsonl"
        self.firewall_log = self.logs_path / "firewall_rules.jsonl"

        # Ensure directories exist
        self.config_path.mkdir(parents=True, exist_ok=True)
        self.logs_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_config()

        # Runtime state
        self.running = False
        self.monitoring_thread = None
        self.last_stats = None
        self.active_threats = {}  # threat_id -> ThreatEvent
        self.firewall_rules = {}  # rule_id -> FirewallRule
        self.connection_history = deque(maxlen=10000)  # Recent connections
        self.bandwidth_history = defaultdict(
            lambda: deque(maxlen=60)
        )  # Per interface, last 60 seconds
        self.suspicious_ips = set()  # IPs flagged as suspicious
        self.blocked_ips = set()  # IPs currently blocked

        # Threat detection patterns
        self.threat_patterns = self._load_threat_patterns()

        # Load existing firewall rules
        self._load_firewall_rules()

        logger.info("NetworkGuard initialized with threat detection enabled")

    def _load_config(self) -> Dict[str, Any]:
        """Load network guard configuration"""
        config_file = self.config_path / "net_guard_config.json"

        default_config = {
            "monitoring": {
                "enable_deep_packet_inspection": False,  # Requires special permissions
                "interface_polling_interval": 5,
                "connection_polling_interval": 2,
                "threat_detection_interval": 10,
                "max_connections_per_ip": 100,
                "bandwidth_threshold_mbps": 1000,
                "enable_port_scan_detection": True,
                "enable_dos_detection": True,
            },
            "firewall": {
                "enable_auto_blocking": True,
                "auto_block_duration_minutes": 60,
                "max_firewall_rules": 1000,
                "enable_whitelist_bypass": True,
                "default_action": "allow",
                "enable_rate_limiting": True,
            },
            "threat_detection": {
                "port_scan_threshold": 20,  # Connections to different ports in short time
                "connection_rate_threshold": 50,  # Connections per minute
                "failed_connection_threshold": 10,
                "suspicious_port_ranges": [
                    [1, 1023],  # System ports
                    [1433, 1434],  # SQL Server
                    [3389, 3389],  # RDP
                    [22, 22],  # SSH
                    [23, 23],  # Telnet
                    [21, 21],  # FTP
                    [3306, 3306],  # MySQL
                    [5432, 5432],  # PostgreSQL
                    [6379, 6379],  # Redis
                    [27017, 27017],  # MongoDB
                ],
            },
            "whitelist": {
                "trusted_ips": [
                    "127.0.0.1",  # Localhost
                    "::1",  # IPv6 localhost
                    "10.0.0.0/8",  # Private networks
                    "172.16.0.0/12",
                    "192.168.0.0/16",
                ],
                "trusted_services": ["ssh", "dns", "ntp", "dhcp"],
            },
            "notifications": {
                "enable_threat_alerts": True,
                "enable_firewall_alerts": True,
                "alert_threshold": "medium",
                "max_alerts_per_hour": 50,
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
        config_file = self.config_path / "net_guard_config.json"
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def _load_threat_patterns(self) -> Dict[str, Any]:
        """Load threat detection patterns"""
        patterns_file = self.config_path / "threat_patterns.json"

        default_patterns = {
            "port_scan_signatures": [
                {
                    "name": "nmap_tcp_scan",
                    "pattern": r"SYN.*(?:RST|timeout)",
                    "severity": "medium",
                },
                {
                    "name": "stealth_scan",
                    "pattern": r"FIN.*no_response",
                    "severity": "high",
                },
                {"name": "xmas_scan", "pattern": r"FIN.*PSH.*URG", "severity": "high"},
            ],
            "dos_signatures": [
                {
                    "name": "syn_flood",
                    "pattern": r"SYN.*rate_exceeded",
                    "severity": "critical",
                },
                {
                    "name": "icmp_flood",
                    "pattern": r"ICMP.*rate_exceeded",
                    "severity": "high",
                },
                {
                    "name": "udp_flood",
                    "pattern": r"UDP.*rate_exceeded",
                    "severity": "high",
                },
            ],
            "intrusion_signatures": [
                {
                    "name": "bruteforce_ssh",
                    "pattern": r"ssh.*failed_auth.*rate_exceeded",
                    "severity": "high",
                },
                {
                    "name": "web_exploit",
                    "pattern": r"HTTP.*(union|select|drop|script)",
                    "severity": "critical",
                },
                {
                    "name": "malware_beacon",
                    "pattern": r"DNS.*suspicious_tld",
                    "severity": "high",
                },
            ],
        }

        if patterns_file.exists():
            try:
                with open(patterns_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading threat patterns: {e}")
                return default_patterns
        else:
            self._save_threat_patterns(default_patterns)
            return default_patterns

    def _save_threat_patterns(self, patterns: Dict[str, Any]):
        """Save threat patterns to file"""
        patterns_file = self.config_path / "threat_patterns.json"
        try:
            with open(patterns_file, "w") as f:
                json.dump(patterns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving threat patterns: {e}")

    def _load_firewall_rules(self):
        """Load existing firewall rules from log"""
        if self.firewall_log.exists():
            try:
                with open(self.firewall_log, "r") as f:
                    for line in f:
                        if line.strip():
                            rule_data = json.loads(line)
                            rule = FirewallRule(**rule_data)

                            # Only load active rules (not expired)
                            if rule.expires_at:
                                expire_time = datetime.fromisoformat(rule.expires_at)
                                if datetime.now() > expire_time:
                                    continue

                            self.firewall_rules[rule.rule_id] = rule
            except Exception as e:
                logger.error(f"Error loading firewall rules: {e}")

    def _save_threat_event(self, threat: ThreatEvent):
        """Save threat event to persistent log"""
        try:
            with open(self.threats_log, "a") as f:
                f.write(json.dumps(asdict(threat)) + "\n")
        except Exception as e:
            logger.error(f"Error saving threat event: {e}")

    def _save_firewall_rule(self, rule: FirewallRule):
        """Save firewall rule to persistent log"""
        try:
            with open(self.firewall_log, "a") as f:
                f.write(json.dumps(asdict(rule)) + "\n")
        except Exception as e:
            logger.error(f"Error saving firewall rule: {e}")

    def _get_network_interfaces(self) -> List[NetworkInterface]:
        """Get detailed network interface information"""
        interfaces = []

        if not psutil:
            return interfaces

        try:
            # Get interface statistics
            stats = psutil.net_io_counters(pernic=True)
            # Get interface addresses
            addrs = psutil.net_if_addrs()
            # Get interface status
            if hasattr(psutil, "net_if_stats"):
                if_stats = psutil.net_if_stats()
            else:
                if_stats = {}

            for interface_name, stat in stats.items():
                if interface_name == "lo":  # Skip loopback
                    continue

                # Get IP addresses
                ip_addresses = []
                mac_address = ""
                if interface_name in addrs:
                    for addr in addrs[interface_name]:
                        if addr.family == socket.AF_INET:
                            ip_addresses.append(addr.address)
                        elif addr.family == psutil.AF_LINK:
                            mac_address = addr.address

                # Get interface stats
                status = "up"
                mtu = 1500
                speed = None
                duplex = None

                if interface_name in if_stats:
                    status = "up" if if_stats[interface_name].isup else "down"
                    mtu = if_stats[interface_name].mtu
                    speed = getattr(if_stats[interface_name], "speed", None)
                    duplex = getattr(if_stats[interface_name], "duplex", None)
                    if duplex is not None:
                        duplex = str(duplex)

                interface = NetworkInterface(
                    name=interface_name,
                    status=status,
                    mtu=mtu,
                    ip_addresses=ip_addresses,
                    mac_address=mac_address,
                    speed=speed,
                    duplex=duplex,
                    rx_bytes=stat.bytes_recv,
                    tx_bytes=stat.bytes_sent,
                    rx_packets=stat.packets_recv,
                    tx_packets=stat.packets_sent,
                    rx_errors=stat.errin,
                    tx_errors=stat.errout,
                    rx_dropped=stat.dropin,
                    tx_dropped=stat.dropout,
                )
                interfaces.append(interface)

        except Exception as e:
            logger.error(f"Error getting network interfaces: {e}")

        return interfaces

    def _get_network_connections(self) -> List[NetworkConnection]:
        """Get active network connections"""
        connections = []

        if not psutil:
            return connections

        try:
            net_connections = psutil.net_connections(kind="inet")

            for conn in net_connections:
                # Skip connections without remote address
                if not conn.raddr:
                    continue

                # Get process information
                process_name = None
                if conn.pid:
                    try:
                        process = psutil.Process(conn.pid)
                        process_name = process.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Map psutil status to our enum
                status_map = {
                    psutil.CONN_ESTABLISHED: ConnectionState.ESTABLISHED,
                    psutil.CONN_LISTEN: ConnectionState.LISTENING,
                    psutil.CONN_TIME_WAIT: ConnectionState.TIME_WAIT,
                    psutil.CONN_CLOSE_WAIT: ConnectionState.CLOSE_WAIT,
                    psutil.CONN_FIN_WAIT1: ConnectionState.FIN_WAIT,
                    psutil.CONN_FIN_WAIT2: ConnectionState.FIN_WAIT,
                    psutil.CONN_SYN_SENT: ConnectionState.SYN_SENT,
                    psutil.CONN_SYN_RECV: ConnectionState.SYN_RECV,
                }

                status = status_map.get(conn.status, ConnectionState.ESTABLISHED)

                connection = NetworkConnection(
                    local_address=conn.laddr.ip,
                    local_port=conn.laddr.port,
                    remote_address=conn.raddr.ip,
                    remote_port=conn.raddr.port,
                    status=status,
                    pid=conn.pid,
                    process_name=process_name,
                    protocol="tcp" if conn.type == socket.SOCK_STREAM else "udp",
                    family="ipv4" if conn.family == socket.AF_INET else "ipv6",
                    created_time=datetime.now().isoformat(),
                )
                connections.append(connection)

        except Exception as e:
            logger.error(f"Error getting network connections: {e}")

        return connections

    def _is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP address is in whitelist"""
        trusted_ips = self.config["whitelist"]["trusted_ips"]

        try:
            ip_addr = ipaddress.ip_address(ip)
            for trusted in trusted_ips:
                if "/" in trusted:
                    # CIDR notation
                    if ip_addr in ipaddress.ip_network(trusted):
                        return True
                else:
                    # Single IP
                    if str(ip_addr) == trusted:
                        return True
        except ValueError:
            # Invalid IP address
            pass

        return False

    def _detect_port_scan(
        self, connections: List[NetworkConnection]
    ) -> List[ThreatEvent]:
        """Detect port scanning attempts"""
        threats = []

        if not self.config["monitoring"]["enable_port_scan_detection"]:
            return threats

        # Group connections by source IP
        ip_connections = defaultdict(list)
        for conn in connections:
            if not self._is_ip_whitelisted(conn.remote_address):
                ip_connections[conn.remote_address].append(conn)

        # Check for port scan patterns
        threshold = self.config["threat_detection"]["port_scan_threshold"]

        for source_ip, conns in ip_connections.items():
            if source_ip in self.blocked_ips:
                continue  # Already blocked

            # Count unique destination ports
            unique_ports = set(conn.local_port for conn in conns)

            if len(unique_ports) >= threshold:
                threat_id = f"port_scan_{source_ip}_{int(time.time())}"

                threat = ThreatEvent(
                    timestamp=datetime.now().isoformat(),
                    threat_id=threat_id,
                    threat_level=ThreatLevel.HIGH,
                    source_ip=source_ip,
                    destination_ip="multiple",
                    source_port=None,
                    destination_port=None,
                    protocol="tcp",
                    description=f"Port scan detected from {source_ip} targeting {len(unique_ports)} ports",
                    evidence={
                        "unique_ports": len(unique_ports),
                        "total_connections": len(conns),
                        "ports_scanned": sorted(list(unique_ports))[
                            :20
                        ],  # First 20 ports
                    },
                )
                threats.append(threat)
                self.suspicious_ips.add(source_ip)

        return threats

    def _detect_dos_attacks(
        self, connections: List[NetworkConnection]
    ) -> List[ThreatEvent]:
        """Detect Denial of Service attacks"""
        threats = []

        if not self.config["monitoring"]["enable_dos_detection"]:
            return threats

        # Group connections by source IP
        ip_connections = defaultdict(list)
        for conn in connections:
            if not self._is_ip_whitelisted(conn.remote_address):
                ip_connections[conn.remote_address].append(conn)

        # Check for excessive connections
        threshold = self.config["threat_detection"]["connection_rate_threshold"]

        for source_ip, conns in ip_connections.items():
            if source_ip in self.blocked_ips:
                continue  # Already blocked

            if len(conns) >= threshold:
                threat_id = f"dos_attack_{source_ip}_{int(time.time())}"

                threat = ThreatEvent(
                    timestamp=datetime.now().isoformat(),
                    threat_id=threat_id,
                    threat_level=ThreatLevel.CRITICAL,
                    source_ip=source_ip,
                    destination_ip="multiple",
                    source_port=None,
                    destination_port=None,
                    protocol="multiple",
                    description=f"DoS attack detected from {source_ip} with {len(conns)} concurrent connections",
                    evidence={
                        "connection_count": len(conns),
                        "connection_states": {
                            state.value: sum(1 for c in conns if c.status == state)
                            for state in ConnectionState
                        },
                        "target_ports": list(set(conn.local_port for conn in conns))[
                            :10
                        ],
                    },
                )
                threats.append(threat)
                self.suspicious_ips.add(source_ip)

        return threats

    def _detect_suspicious_connections(
        self, connections: List[NetworkConnection]
    ) -> List[ThreatEvent]:
        """Detect suspicious connection patterns"""
        threats = []

        suspicious_port_ranges = self.config["threat_detection"][
            "suspicious_port_ranges"
        ]

        for conn in connections:
            if self._is_ip_whitelisted(conn.remote_address):
                continue

            # Check for connections to suspicious ports
            for port_range in suspicious_port_ranges:
                start_port, end_port = port_range
                if start_port <= conn.local_port <= end_port:
                    # This could be legitimate, so only flag as low threat
                    threat_id = f"suspicious_port_{conn.remote_address}_{conn.local_port}_{int(time.time())}"

                    threat = ThreatEvent(
                        timestamp=datetime.now().isoformat(),
                        threat_id=threat_id,
                        threat_level=ThreatLevel.LOW,
                        source_ip=conn.remote_address,
                        destination_ip=conn.local_address,
                        source_port=conn.remote_port,
                        destination_port=conn.local_port,
                        protocol=conn.protocol,
                        description=f"Connection to suspicious port {conn.local_port} from {conn.remote_address}",
                        evidence={
                            "port_category": (
                                "system" if start_port <= 1023 else "service"
                            ),
                            "process_name": conn.process_name,
                            "connection_state": conn.status.value,
                        },
                    )
                    threats.append(threat)
                    break

        return threats

    def _create_firewall_rule(
        self,
        action: FirewallAction,
        source_ip: str,
        description: str,
        duration_minutes: Optional[int] = None,
        protocol: Optional[str] = None,
        port: Optional[int] = None,
    ) -> FirewallRule:
        """Create a new firewall rule"""

        rule_id = f"auto_{action.value}_{source_ip}_{int(time.time())}"

        expires_at = None
        if duration_minutes:
            expires_at = (
                datetime.now() + timedelta(minutes=duration_minutes)
            ).isoformat()

        rule = FirewallRule(
            rule_id=rule_id,
            action=action,
            protocol=protocol,
            source_ip=source_ip,
            source_port=None,
            destination_ip=None,
            destination_port=str(port) if port else None,
            direction=TrafficDirection.INBOUND,
            description=description,
            created_time=datetime.now().isoformat(),
            expires_at=expires_at,
        )

        return rule

    def _apply_firewall_rule(self, rule: FirewallRule) -> bool:
        """Apply firewall rule using iptables"""
        try:
            # Build iptables command
            cmd = ["iptables", "-I", "INPUT"]

            if rule.protocol:
                cmd.extend(["-p", rule.protocol])

            if rule.source_ip:
                cmd.extend(["-s", rule.source_ip])

            if rule.destination_port:
                cmd.extend(["--dport", rule.destination_port])

            # Add action
            if rule.action == FirewallAction.DENY or rule.action == FirewallAction.DROP:
                cmd.extend(["-j", "DROP"])
            elif rule.action == FirewallAction.REJECT:
                cmd.extend(["-j", "REJECT"])
            else:
                cmd.extend(["-j", "ACCEPT"])

            # Add comment for identification
            cmd.extend(["-m", "comment", "--comment", f"SAC_{rule.rule_id}"])

            # Execute command (would require sudo)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info(f"Applied firewall rule: {rule.description}")
                return True
            else:
                logger.error(f"Failed to apply firewall rule: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Error applying firewall rule: {e}")
            return False

    def _remove_firewall_rule(self, rule: FirewallRule) -> bool:
        """Remove firewall rule using iptables"""
        try:
            # Find and remove rule by comment
            cmd = [
                "iptables",
                "-D",
                "INPUT",
                "-m",
                "comment",
                "--comment",
                f"SAC_{rule.rule_id}",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                logger.info(f"Removed firewall rule: {rule.rule_id}")
                return True
            else:
                logger.warning(
                    f"Failed to remove firewall rule (may not exist): {result.stderr}"
                )
                return False

        except Exception as e:
            logger.error(f"Error removing firewall rule: {e}")
            return False

    def _respond_to_threat(self, threat: ThreatEvent) -> bool:
        """Respond to detected threat with appropriate mitigation"""
        if not self.config["firewall"]["enable_auto_blocking"]:
            logger.info(
                f"Auto-blocking disabled, threat logged only: {threat.description}"
            )
            return False

        if self._is_ip_whitelisted(threat.source_ip):
            logger.info(f"Source IP {threat.source_ip} is whitelisted, skipping block")
            return False

        # Determine response based on threat level
        action = FirewallAction.DROP
        duration = self.config["firewall"]["auto_block_duration_minutes"]

        if threat.threat_level == ThreatLevel.CRITICAL:
            duration = duration * 2  # Longer block for critical threats
        elif threat.threat_level == ThreatLevel.LOW:
            duration = max(duration // 2, 5)  # Shorter block for low threats

        # Create firewall rule
        rule = self._create_firewall_rule(
            action=action,
            source_ip=threat.source_ip,
            description=f"Auto-block for threat: {threat.description}",
            duration_minutes=duration,
            protocol=threat.protocol if threat.protocol != "multiple" else None,
            port=threat.destination_port,
        )

        # Apply rule (in production this would need proper sudo access)
        # For demonstration, we'll just log the action
        logger.warning(
            f"WOULD BLOCK: {threat.source_ip} for {duration} minutes due to: {threat.description}"
        )

        # Store rule
        self.firewall_rules[rule.rule_id] = rule
        self._save_firewall_rule(rule)

        # Update threat with mitigation action
        threat.mitigation_action = f"blocked_ip_{duration}min"

        # Add to blocked IPs
        self.blocked_ips.add(threat.source_ip)

        return True

    def _cleanup_expired_rules(self):
        """Remove expired firewall rules"""
        current_time = datetime.now()
        expired_rules = []

        for rule_id, rule in self.firewall_rules.items():
            if rule.expires_at:
                expire_time = datetime.fromisoformat(rule.expires_at)
                if current_time > expire_time:
                    expired_rules.append(rule_id)

        for rule_id in expired_rules:
            rule = self.firewall_rules.pop(rule_id)
            # Remove from iptables (in production)
            logger.info(f"Expired firewall rule removed: {rule.description}")

            # Remove from blocked IPs if it was a block rule
            if (
                rule.action in [FirewallAction.DROP, FirewallAction.DENY]
                and rule.source_ip
            ):
                self.blocked_ips.discard(rule.source_ip)

    def _monitoring_loop(self):
        """Main network monitoring loop"""
        logger.info("Network monitoring started")

        last_interface_check = 0
        last_threat_detection = 0

        while self.running:
            try:
                current_time = time.time()

                # Get current network state
                connections = self._get_network_connections()

                # Update connection history
                self.connection_history.extend(connections)

                # Get interface stats periodically
                interfaces = []
                if (
                    current_time - last_interface_check
                    >= self.config["monitoring"]["interface_polling_interval"]
                ):
                    interfaces = self._get_network_interfaces()
                    last_interface_check = current_time

                # Calculate bandwidth usage
                bandwidth_usage = {}
                for interface in interfaces:
                    if interface.name in self.bandwidth_history:
                        # Calculate bytes per second since last measurement
                        history = self.bandwidth_history[interface.name]
                        if history:
                            last_measurement = history[-1]
                            time_diff = current_time - last_measurement[0]
                            bytes_diff = (
                                interface.rx_bytes + interface.tx_bytes
                            ) - last_measurement[1]
                            if time_diff > 0:
                                bandwidth_usage[interface.name] = int(
                                    bytes_diff / time_diff
                                )

                    # Store current measurement
                    self.bandwidth_history[interface.name].append(
                        (current_time, interface.rx_bytes + interface.tx_bytes)
                    )

                # Run threat detection periodically
                if (
                    current_time - last_threat_detection
                    >= self.config["monitoring"]["threat_detection_interval"]
                ):
                    new_threats = []

                    # Run threat detection algorithms
                    new_threats.extend(self._detect_port_scan(connections))
                    new_threats.extend(self._detect_dos_attacks(connections))
                    new_threats.extend(self._detect_suspicious_connections(connections))

                    # Process new threats
                    for threat in new_threats:
                        self.active_threats[threat.threat_id] = threat
                        self._save_threat_event(threat)

                        logger.warning(f"THREAT DETECTED: {threat.description}")

                        # Respond to threat
                        if threat.threat_level in [
                            ThreatLevel.HIGH,
                            ThreatLevel.CRITICAL,
                        ]:
                            self._respond_to_threat(threat)

                    last_threat_detection = current_time

                # Create current stats snapshot
                self.last_stats = NetworkStats(
                    timestamp=datetime.now().isoformat(),
                    interfaces=interfaces,
                    connections=connections,
                    total_bandwidth_usage=bandwidth_usage,
                    suspicious_connections=len(
                        [
                            c
                            for c in connections
                            if c.remote_address in self.suspicious_ips
                        ]
                    ),
                    blocked_attempts=len(self.blocked_ips),
                    active_threats=len(self.active_threats),
                    firewall_rules_count=len(self.firewall_rules),
                )

                # Cleanup expired rules
                self._cleanup_expired_rules()

                # Sleep until next monitoring cycle
                time.sleep(self.config["monitoring"]["connection_polling_interval"])

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)

    def start(self):
        """Start network monitoring"""
        if self.running:
            logger.warning("NetworkGuard already running")
            return

        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("NetworkGuard started")

    def stop(self):
        """Stop network monitoring"""
        if not self.running:
            return

        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("NetworkGuard stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current network security status"""
        active_threat_count = len(self.active_threats)
        blocked_ips_count = len(self.blocked_ips)

        return {
            "running": self.running,
            "last_check": self.last_stats.timestamp if self.last_stats else None,
            "active_threats": active_threat_count,
            "blocked_ips": blocked_ips_count,
            "firewall_rules": len(self.firewall_rules),
            "suspicious_ips": len(self.suspicious_ips),
            "recent_connections": len(self.connection_history),
            "monitoring_config": {
                "auto_blocking": self.config["firewall"]["enable_auto_blocking"],
                "port_scan_detection": self.config["monitoring"][
                    "enable_port_scan_detection"
                ],
                "dos_detection": self.config["monitoring"]["enable_dos_detection"],
            },
        }

    def get_active_threats(self) -> List[Dict[str, Any]]:
        """Get all active threat events"""
        return [asdict(threat) for threat in self.active_threats.values()]

    def get_firewall_rules(self) -> List[Dict[str, Any]]:
        """Get all active firewall rules"""
        return [asdict(rule) for rule in self.firewall_rules.values()]

    def get_network_summary(self) -> Dict[str, Any]:
        """Get comprehensive network security summary"""
        if not self.last_stats:
            return {"error": "No statistics available"}

        # Calculate threat summary
        threat_levels = defaultdict(int)
        for threat in self.active_threats.values():
            threat_levels[threat.threat_level.value] += 1

        # Calculate top suspicious IPs
        ip_threat_count = defaultdict(int)
        for threat in self.active_threats.values():
            ip_threat_count[threat.source_ip] += 1

        top_suspicious = sorted(
            ip_threat_count.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "timestamp": self.last_stats.timestamp,
            "network_interfaces": len(self.last_stats.interfaces),
            "active_connections": len(self.last_stats.connections),
            "bandwidth_usage": self.last_stats.total_bandwidth_usage,
            "security_status": {
                "threat_levels": dict(threat_levels),
                "total_threats": len(self.active_threats),
                "blocked_ips": len(self.blocked_ips),
                "firewall_rules": len(self.firewall_rules),
            },
            "top_suspicious_ips": top_suspicious,
            "recent_activity": {
                "connections_last_hour": len(
                    [
                        c
                        for c in self.connection_history
                        if (
                            datetime.now() - datetime.fromisoformat(c.created_time)
                        ).total_seconds()
                        < 3600
                    ]
                ),
                "threats_last_hour": len(
                    [
                        t
                        for t in self.active_threats.values()
                        if (
                            datetime.now() - datetime.fromisoformat(t.timestamp)
                        ).total_seconds()
                        < 3600
                    ]
                ),
            },
        }

    def block_ip(
        self, ip: str, duration_minutes: int = 60, reason: str = "Manual block"
    ) -> bool:
        """Manually block an IP address"""
        if self._is_ip_whitelisted(ip):
            logger.warning(f"Cannot block whitelisted IP: {ip}")
            return False

        rule = self._create_firewall_rule(
            action=FirewallAction.DROP,
            source_ip=ip,
            description=f"Manual block: {reason}",
            duration_minutes=duration_minutes,
        )

        self.firewall_rules[rule.rule_id] = rule
        self._save_firewall_rule(rule)
        self.blocked_ips.add(ip)

        logger.info(
            f"Manually blocked IP {ip} for {duration_minutes} minutes: {reason}"
        )
        return True

    def unblock_ip(self, ip: str) -> bool:
        """Manually unblock an IP address"""
        # Find and remove all rules for this IP
        rules_to_remove = []
        for rule_id, rule in self.firewall_rules.items():
            if rule.source_ip == ip and rule.action in [
                FirewallAction.DROP,
                FirewallAction.DENY,
            ]:
                rules_to_remove.append(rule_id)

        for rule_id in rules_to_remove:
            rule = self.firewall_rules.pop(rule_id)
            # Remove from iptables (in production)
            logger.info(f"Removed block rule for IP {ip}: {rule.description}")

        self.blocked_ips.discard(ip)
        self.suspicious_ips.discard(ip)

        return len(rules_to_remove) > 0


# Global network guard instance
network_guard = NetworkGuard()
net_guard = network_guard  # Alias for compatibility

if __name__ == "__main__":
    # CLI interface
    import argparse

    parser = argparse.ArgumentParser(description="System Autonomy Core - Network Guard")
    parser.add_argument("--start", action="store_true", help="Start network monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop network monitoring")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--threats", action="store_true", help="Show active threats")
    parser.add_argument("--summary", action="store_true", help="Show network summary")
    parser.add_argument("--block-ip", help="Manually block an IP address")
    parser.add_argument("--unblock-ip", help="Manually unblock an IP address")
    parser.add_argument(
        "--duration", type=int, default=60, help="Block duration in minutes"
    )
    parser.add_argument(
        "--reason", default="Manual intervention", help="Reason for blocking"
    )

    args = parser.parse_args()

    if args.start:
        network_guard.start()
        print("🛡️ Network monitoring started")
        print("Press Ctrl+C to stop")
        try:
            while network_guard.running:
                time.sleep(1)
        except KeyboardInterrupt:
            network_guard.stop()

    elif args.stop:
        network_guard.stop()
        print("🛡️ Network monitoring stopped")

    elif args.status:
        status = network_guard.get_status()
        print("🛡️ NetworkGuard Status:")
        print(f"   Running: {status['running']}")
        print(f"   Last Check: {status['last_check']}")
        print(f"   Active Threats: {status['active_threats']}")
        print(f"   Blocked IPs: {status['blocked_ips']}")
        print(f"   Firewall Rules: {status['firewall_rules']}")
        print(f"   Auto-blocking: {status['monitoring_config']['auto_blocking']}")

    elif args.threats:
        threats = network_guard.get_active_threats()
        if threats:
            print("🚨 Active Threats:")
            for threat in threats[-10:]:  # Last 10 threats
                print(
                    f"   {threat['timestamp']}: {threat['description']} ({threat['threat_level']})"
                )
        else:
            print("✅ No active threats")

    elif args.summary:
        summary = network_guard.get_network_summary()
        if "error" not in summary:
            print("📊 Network Security Summary:")
            print(f"   Interfaces: {summary['network_interfaces']}")
            print(f"   Active Connections: {summary['active_connections']}")
            print(f"   Total Threats: {summary['security_status']['total_threats']}")
            print(f"   Blocked IPs: {summary['security_status']['blocked_ips']}")
            print(f"   Firewall Rules: {summary['security_status']['firewall_rules']}")

            if summary["top_suspicious_ips"]:
                print("   Top Suspicious IPs:")
                for ip, count in summary["top_suspicious_ips"][:5]:
                    print(f"     {ip}: {count} threats")
        else:
            print(f"❌ {summary['error']}")

    elif args.block_ip:
        success = network_guard.block_ip(args.block_ip, args.duration, args.reason)
        if success:
            print(f"🚫 Blocked IP {args.block_ip} for {args.duration} minutes")
        else:
            print(f"❌ Failed to block IP {args.block_ip}")

    elif args.unblock_ip:
        success = network_guard.unblock_ip(args.unblock_ip)
        if success:
            print(f"✅ Unblocked IP {args.unblock_ip}")
        else:
            print(f"❌ IP {args.unblock_ip} was not blocked or failed to unblock")

    else:
        status = network_guard.get_status()
        summary = network_guard.get_network_summary()

        print("🛡️ NetworkGuard Quick Status:")
        print(f"   Status: {'Running' if status['running'] else 'Stopped'}")
        print(f"   Active Threats: {status['active_threats']}")
        print(f"   Blocked IPs: {status['blocked_ips']}")

        if "error" not in summary:
            print(f"   Active Connections: {summary['active_connections']}")
            print(
                f"   Recent Activity: {summary['recent_activity']['connections_last_hour']} connections in last hour"
            )
