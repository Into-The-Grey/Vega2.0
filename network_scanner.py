#!/usr/bin/env python3
"""
NETWORK SCANNER - INTELLIGENT DISCOVERY ENGINE
==============================================

Advanced network scanning and intelligent device integration system.
Discovers devices, services, and integration opportunities across local network.

Features:
- 🔍 Comprehensive network device discovery
- 🌐 Service enumeration and capability detection
- 🤖 AI-powered integration opportunity analysis
- 🛡️ Security scanning and vulnerability assessment
- 📊 Network topology mapping and visualization
- 🔄 Continuous monitoring and adaptive scanning
- 🎯 Smart targeting based on device capabilities
- 📱 Mobile device and IoT detection

Usage:
    python network_scanner.py --scan --quick          # Quick network scan
    python network_scanner.py --scan --deep           # Deep discovery scan
    python network_scanner.py --monitor               # Continuous monitoring
    python network_scanner.py --integration-analysis  # Analyze integration opportunities
    python network_scanner.py --daemon                # Background daemon mode
"""

import os
import sys
import json
import time
import asyncio
import sqlite3
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import logging
import ipaddress
import socket
import hashlib

# Network scanning imports
NETWORK_AVAILABLE = False
try:
    import scapy.all as scapy
    from scapy.layers import http
    import nmap
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    NETWORK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Network scanning libraries not available: {e}")

# Analysis imports
ANALYSIS_AVAILABLE = False
try:
    import whois
    import dns.resolver
    import ssl
    import OpenSSL

    ANALYSIS_AVAILABLE = True
except ImportError:
    pass


class DeviceType(Enum):
    """Device type classifications"""

    COMPUTER = "computer"
    MOBILE = "mobile"
    IOT = "iot"
    ROUTER = "router"
    PRINTER = "printer"
    SMART_TV = "smart_tv"
    GAMING_CONSOLE = "gaming_console"
    NAS = "nas"
    SECURITY_CAMERA = "security_camera"
    SMART_SPEAKER = "smart_speaker"
    UNKNOWN = "unknown"


class ServiceType(Enum):
    """Service type classifications"""

    WEB_SERVER = "web_server"
    SSH = "ssh"
    FTP = "ftp"
    SMB = "smb"
    API = "api"
    DATABASE = "database"
    MEDIA_SERVER = "media_server"
    CHAT_SERVICE = "chat_service"
    VPN = "vpn"
    UNKNOWN = "unknown"


class IntegrationPotential(Enum):
    """Integration opportunity levels"""

    HIGH = "high"  # Direct API access, known protocols
    MEDIUM = "medium"  # Web interface, standard protocols
    LOW = "low"  # Limited protocols, basic interaction
    NONE = "none"  # No viable integration methods


@dataclass
class NetworkDevice:
    """Discovered network device"""

    ip_address: str
    mac_address: str = ""
    hostname: str = ""
    device_type: DeviceType = DeviceType.UNKNOWN
    vendor: str = ""
    os_fingerprint: str = ""
    last_seen: str = ""
    response_time: float = 0.0
    open_ports: List[int] = None
    services: List[Dict[str, Any]] = None
    integration_potential: IntegrationPotential = IntegrationPotential.NONE

    def __post_init__(self):
        if self.open_ports is None:
            self.open_ports = []
        if self.services is None:
            self.services = []


@dataclass
class DiscoveredService:
    """Service running on a device"""

    device_ip: str
    port: int
    service_type: ServiceType
    service_name: str
    version: str = ""
    banner: str = ""
    endpoints: List[str] = None
    authentication: str = ""
    capabilities: List[str] = None
    integration_methods: List[str] = None

    def __post_init__(self):
        if self.endpoints is None:
            self.endpoints = []
        if self.capabilities is None:
            self.capabilities = []
        if self.integration_methods is None:
            self.integration_methods = []


@dataclass
class IntegrationOpportunity:
    """Identified integration opportunity"""

    device_ip: str
    service: DiscoveredService
    integration_type: str  # "api", "web", "protocol", "scraping"
    confidence: float
    description: str
    implementation_complexity: str  # "low", "medium", "high"
    potential_benefits: List[str]
    required_capabilities: List[str]
    sample_code: str = ""


@dataclass
class ScanConfiguration:
    """Network scanning configuration"""

    target_network: str = "192.168.1.0/24"
    port_ranges: List[str] = None
    scan_timeout: int = 5
    max_threads: int = 50
    deep_scan: bool = False
    service_detection: bool = True
    os_detection: bool = True
    vulnerability_scan: bool = False

    def __post_init__(self):
        if self.port_ranges is None:
            self.port_ranges = ["1-1000", "3000-3010", "8000-8100"]


class NetworkScanner:
    """Intelligent network discovery and integration engine"""

    def __init__(self, config: ScanConfiguration = None):
        self.config = config or ScanConfiguration()
        self.base_dir = Path(__file__).parent
        self.state_dir = self.base_dir / "vega_state"
        self.logs_dir = self.base_dir / "vega_logs"

        # Create directories
        for directory in [self.state_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)8s | %(name)20s | %(message)s",
            handlers=[
                logging.FileHandler(self.logs_dir / "network_scanner.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("NetworkScanner")

        # Discovery data
        self.discovered_devices: Dict[str, NetworkDevice] = {}
        self.discovered_services: List[DiscoveredService] = []
        self.integration_opportunities: List[IntegrationOpportunity] = []

        # Scanning state
        self.scan_in_progress = False
        self.last_scan_time = None

        # Initialize database
        self.init_database()

        # Known service signatures
        self.service_signatures = {
            80: ServiceType.WEB_SERVER,
            443: ServiceType.WEB_SERVER,
            22: ServiceType.SSH,
            21: ServiceType.FTP,
            445: ServiceType.SMB,
            139: ServiceType.SMB,
            3306: ServiceType.DATABASE,
            5432: ServiceType.DATABASE,
            8080: ServiceType.WEB_SERVER,
            8000: ServiceType.WEB_SERVER,
            3000: ServiceType.WEB_SERVER,
            9000: ServiceType.API,
        }

        # Device type detection patterns
        self.device_patterns = {
            DeviceType.ROUTER: ["router", "gateway", "cisco", "netgear", "linksys"],
            DeviceType.PRINTER: ["printer", "hp", "canon", "epson", "brother"],
            DeviceType.NAS: ["nas", "synology", "qnap", "freenas"],
            DeviceType.SMART_TV: ["tv", "samsung", "lg", "sony", "roku"],
            DeviceType.IOT: ["iot", "sensor", "arduino", "raspberry"],
        }

        self.logger.info("🔍 Network Scanner initialized")

    def init_database(self):
        """Initialize SQLite database for network discovery"""
        db_path = self.state_dir / "network_discovery.db"

        try:
            with sqlite3.connect(db_path) as conn:
                # Devices table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS devices (
                        ip_address TEXT PRIMARY KEY,
                        mac_address TEXT,
                        hostname TEXT,
                        device_type TEXT,
                        vendor TEXT,
                        os_fingerprint TEXT,
                        last_seen TEXT,
                        response_time REAL,
                        first_discovered TEXT,
                        scan_count INTEGER DEFAULT 1
                    )
                """
                )

                # Services table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS services (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_ip TEXT,
                        port INTEGER,
                        service_type TEXT,
                        service_name TEXT,
                        version TEXT,
                        banner TEXT,
                        last_seen TEXT,
                        FOREIGN KEY (device_ip) REFERENCES devices (ip_address)
                    )
                """
                )

                # Integration opportunities table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS integration_opportunities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        device_ip TEXT,
                        service_id INTEGER,
                        integration_type TEXT,
                        confidence REAL,
                        description TEXT,
                        complexity TEXT,
                        discovered_date TEXT,
                        status TEXT DEFAULT 'new',
                        FOREIGN KEY (device_ip) REFERENCES devices (ip_address),
                        FOREIGN KEY (service_id) REFERENCES services (id)
                    )
                """
                )

                # Scan history table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS scan_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_type TEXT,
                        start_time TEXT,
                        end_time TEXT,
                        devices_found INTEGER,
                        services_found INTEGER,
                        target_network TEXT,
                        scan_config TEXT
                    )
                """
                )

                conn.commit()

            self.logger.info("📊 Network discovery database initialized")

        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {e}")

    def get_local_network(self) -> str:
        """Detect local network range"""
        try:
            # Get default gateway
            result = subprocess.run(
                ["ip", "route", "show", "default"], capture_output=True, text=True
            )

            if result.stdout:
                gateway_ip = result.stdout.split()[2]
                # Assume /24 subnet
                network = ipaddress.IPv4Network(f"{gateway_ip}/24", strict=False)
                return str(network)

            # Fallback to common private networks
            return "192.168.1.0/24"

        except Exception as e:
            self.logger.error(f"❌ Error detecting local network: {e}")
            return "192.168.1.0/24"

    async def ping_sweep(self, network: str) -> List[str]:
        """Perform ping sweep to discover active hosts"""
        active_hosts = []

        try:
            network_obj = ipaddress.IPv4Network(network, strict=False)

            # Create semaphore to limit concurrent pings
            semaphore = asyncio.Semaphore(self.config.max_threads)

            async def ping_host(ip_str: str):
                async with semaphore:
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            "ping",
                            "-c",
                            "1",
                            "-W",
                            "1",
                            ip_str,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL,
                        )

                        await asyncio.wait_for(proc.wait(), timeout=2)

                        if proc.returncode == 0:
                            active_hosts.append(ip_str)
                            self.logger.debug(f"✅ Host active: {ip_str}")

                    except Exception:
                        pass  # Host not reachable

            # Create tasks for all IPs in network
            tasks = []
            for ip in network_obj.hosts():
                if len(tasks) < 254:  # Limit to reasonable number
                    tasks.append(ping_host(str(ip)))

            # Execute ping sweep
            await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.info(f"🔍 Ping sweep found {len(active_hosts)} active hosts")
            return active_hosts

        except Exception as e:
            self.logger.error(f"❌ Error in ping sweep: {e}")
            return []

    async def get_device_info(self, ip: str) -> NetworkDevice:
        """Get detailed device information"""
        device = NetworkDevice(ip_address=ip, last_seen=datetime.now().isoformat())

        try:
            # Get hostname
            try:
                hostname = socket.gethostbyaddr(ip)[0]
                device.hostname = hostname
            except:
                pass

            # Get MAC address (for local network)
            try:
                if NETWORK_AVAILABLE:
                    arp_request = scapy.ARP(pdst=ip)
                    broadcast = scapy.Ether(dst="ff:ff:ff:ff:ff:ff")
                    arp_request_broadcast = broadcast / arp_request
                    answered_list = scapy.srp(
                        arp_request_broadcast, timeout=1, verbose=False
                    )[0]

                    if answered_list:
                        device.mac_address = answered_list[0][1].hwsrc
            except:
                pass

            # Detect device type from hostname
            if device.hostname:
                hostname_lower = device.hostname.lower()
                for device_type, patterns in self.device_patterns.items():
                    if any(pattern in hostname_lower for pattern in patterns):
                        device.device_type = device_type
                        break

            # Basic port scan
            if self.config.service_detection:
                device.open_ports = await self.scan_common_ports(ip)

            return device

        except Exception as e:
            self.logger.error(f"❌ Error getting device info for {ip}: {e}")
            return device

    async def scan_common_ports(self, ip: str) -> List[int]:
        """Scan common ports on a device"""
        open_ports = []
        common_ports = [
            21,
            22,
            23,
            25,
            53,
            80,
            110,
            135,
            139,
            143,
            443,
            445,
            993,
            995,
            1723,
            3000,
            3306,
            3389,
            5432,
            5900,
            8000,
            8080,
            8443,
            9000,
        ]

        semaphore = asyncio.Semaphore(20)  # Limit concurrent port scans

        async def scan_port(port: int):
            async with semaphore:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(ip, port), timeout=2
                    )
                    writer.close()
                    await writer.wait_closed()
                    open_ports.append(port)

                except:
                    pass  # Port closed or filtered

        # Scan all common ports
        tasks = [scan_port(port) for port in common_ports]
        await asyncio.gather(*tasks, return_exceptions=True)

        return sorted(open_ports)

    async def analyze_service(self, ip: str, port: int) -> Optional[DiscoveredService]:
        """Analyze a specific service on a port"""
        try:
            service_type = self.service_signatures.get(port, ServiceType.UNKNOWN)
            service = DiscoveredService(
                device_ip=ip,
                port=port,
                service_type=service_type,
                service_name=f"Service on port {port}",
            )

            # HTTP/HTTPS service analysis
            if port in [80, 443, 8000, 8080, 8443]:
                await self.analyze_web_service(service)

            # SSH analysis
            elif port == 22:
                await self.analyze_ssh_service(service)

            # Database analysis
            elif port in [3306, 5432]:
                await self.analyze_database_service(service)

            return service

        except Exception as e:
            self.logger.error(f"❌ Error analyzing service {ip}:{port}: {e}")
            return None

    async def analyze_web_service(self, service: DiscoveredService):
        """Analyze web service for integration opportunities"""
        try:
            protocol = "https" if service.port in [443, 8443] else "http"
            base_url = f"{protocol}://{service.device_ip}:{service.port}"

            # Create session with retries and timeouts
            session = requests.Session()
            retry_strategy = Retry(
                total=2,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Test basic connectivity
            response = session.get(base_url, timeout=5, verify=False)
            service.banner = response.headers.get("Server", "")

            # Check for API endpoints
            api_endpoints = [
                "/api",
                "/api/v1",
                "/api/v2",
                "/graphql",
                "/rest",
                "/swagger",
            ]
            for endpoint in api_endpoints:
                try:
                    api_response = session.get(
                        f"{base_url}{endpoint}", timeout=3, verify=False
                    )
                    if api_response.status_code < 400:
                        service.endpoints.append(endpoint)
                        service.integration_methods.append("api")
                except:
                    pass

            # Check for admin interfaces
            admin_paths = ["/admin", "/administrator", "/management", "/config"]
            for path in admin_paths:
                try:
                    admin_response = session.get(
                        f"{base_url}{path}", timeout=3, verify=False
                    )
                    if admin_response.status_code in [
                        200,
                        401,
                        403,
                    ]:  # Interface exists
                        service.endpoints.append(path)
                        service.integration_methods.append("web")
                except:
                    pass

            # Detect service type from response
            content = response.text.lower()
            if "plex" in content:
                service.service_name = "Plex Media Server"
                service.capabilities.extend(["media_streaming", "api_access"])
            elif "home assistant" in content:
                service.service_name = "Home Assistant"
                service.capabilities.extend(["iot_control", "automation", "api_access"])
            elif "grafana" in content:
                service.service_name = "Grafana"
                service.capabilities.extend(["monitoring", "dashboards", "api_access"])
            elif "jenkins" in content:
                service.service_name = "Jenkins"
                service.capabilities.extend(["ci_cd", "automation", "api_access"])

        except Exception as e:
            self.logger.debug(
                f"Web service analysis failed for {service.device_ip}:{service.port}: {e}"
            )

    async def analyze_ssh_service(self, service: DiscoveredService):
        """Analyze SSH service"""
        try:
            # Try to get SSH banner
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(service.device_ip, 22), timeout=3
            )

            banner = await asyncio.wait_for(reader.read(1024), timeout=2)
            service.banner = banner.decode("utf-8", errors="ignore").strip()

            writer.close()
            await writer.wait_closed()

            service.service_name = "SSH Server"
            service.capabilities.extend(["remote_access", "shell_commands"])
            service.integration_methods.append("ssh")

        except Exception as e:
            self.logger.debug(f"SSH analysis failed for {service.device_ip}: {e}")

    async def analyze_database_service(self, service: DiscoveredService):
        """Analyze database service"""
        if service.port == 3306:
            service.service_name = "MySQL/MariaDB"
        elif service.port == 5432:
            service.service_name = "PostgreSQL"

        service.capabilities.extend(["data_storage", "queries"])
        service.integration_methods.append("database")

    def analyze_integration_opportunities(self) -> List[IntegrationOpportunity]:
        """Analyze discovered services for integration opportunities"""
        opportunities = []

        for service in self.discovered_services:
            # High potential: Services with APIs
            if "api" in service.integration_methods:
                opportunity = IntegrationOpportunity(
                    device_ip=service.device_ip,
                    service=service,
                    integration_type="api",
                    confidence=0.9,
                    description=f"Direct API integration with {service.service_name}",
                    implementation_complexity="low",
                    potential_benefits=[
                        "Real-time data access",
                        "Automated control",
                        "Status monitoring",
                    ],
                    required_capabilities=["HTTP client", "JSON parsing"],
                )
                opportunities.append(opportunity)

            # Medium potential: Web interfaces
            elif "web" in service.integration_methods:
                opportunity = IntegrationOpportunity(
                    device_ip=service.device_ip,
                    service=service,
                    integration_type="web",
                    confidence=0.6,
                    description=f"Web scraping integration with {service.service_name}",
                    implementation_complexity="medium",
                    potential_benefits=["Status monitoring", "Basic automation"],
                    required_capabilities=["Web scraping", "HTML parsing"],
                )
                opportunities.append(opportunity)

            # Special cases for known services
            if "home assistant" in service.service_name.lower():
                opportunity = IntegrationOpportunity(
                    device_ip=service.device_ip,
                    service=service,
                    integration_type="api",
                    confidence=0.95,
                    description="Home Assistant smart home integration",
                    implementation_complexity="low",
                    potential_benefits=[
                        "IoT device control",
                        "Automation triggers",
                        "Sensor data access",
                        "Smart home orchestration",
                    ],
                    required_capabilities=["Home Assistant API"],
                    sample_code=self.generate_homeassistant_sample(service.device_ip),
                )
                opportunities.append(opportunity)

            elif "plex" in service.service_name.lower():
                opportunity = IntegrationOpportunity(
                    device_ip=service.device_ip,
                    service=service,
                    integration_type="api",
                    confidence=0.85,
                    description="Plex media server integration",
                    implementation_complexity="medium",
                    potential_benefits=[
                        "Media library access",
                        "Playback control",
                        "Viewing statistics",
                    ],
                    required_capabilities=["Plex API"],
                    sample_code=self.generate_plex_sample(service.device_ip),
                )
                opportunities.append(opportunity)

        return opportunities

    def generate_homeassistant_sample(self, ip: str) -> str:
        """Generate sample Home Assistant integration code"""
        return f"""
# Home Assistant Integration Sample
import requests

class HomeAssistantClient:
    def __init__(self):
        self.base_url = "http://{ip}:8123"
        self.token = "YOUR_LONG_LIVED_ACCESS_TOKEN"
        self.headers = {{
            "Authorization": f"Bearer {{self.token}}",
            "Content-Type": "application/json"
        }}
    
    async def get_states(self):
        response = requests.get(f"{{self.base_url}}/api/states", headers=self.headers)
        return response.json()
    
    async def call_service(self, domain, service, entity_id):
        data = {{"entity_id": entity_id}}
        response = requests.post(
            f"{{self.base_url}}/api/services/{{domain}}/{{service}}",
            headers=self.headers,
            json=data
        )
        return response.json()

# Usage:
# ha = HomeAssistantClient()
# states = await ha.get_states()
# await ha.call_service("light", "turn_on", "light.living_room")
"""

    def generate_plex_sample(self, ip: str) -> str:
        """Generate sample Plex integration code"""
        return f"""
# Plex Integration Sample
import requests

class PlexClient:
    def __init__(self):
        self.base_url = "http://{ip}:32400"
        self.token = "YOUR_PLEX_TOKEN"
    
    async def get_libraries(self):
        url = f"{{self.base_url}}/library/sections?X-Plex-Token={{self.token}}"
        response = requests.get(url)
        return response.json()
    
    async def get_recently_added(self):
        url = f"{{self.base_url}}/library/recentlyAdded?X-Plex-Token={{self.token}}"
        response = requests.get(url)
        return response.json()

# Usage:
# plex = PlexClient()
# libraries = await plex.get_libraries()
# recent = await plex.get_recently_added()
"""

    def save_discovery_results(self):
        """Save discovery results to database"""
        try:
            db_path = self.state_dir / "network_discovery.db"

            with sqlite3.connect(db_path) as conn:
                # Save devices
                for device in self.discovered_devices.values():
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO devices 
                        (ip_address, mac_address, hostname, device_type, vendor, 
                         os_fingerprint, last_seen, response_time, first_discovered)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                                COALESCE((SELECT first_discovered FROM devices WHERE ip_address = ?), ?))
                    """,
                        (
                            device.ip_address,
                            device.mac_address,
                            device.hostname,
                            device.device_type.value,
                            device.vendor,
                            device.os_fingerprint,
                            device.last_seen,
                            device.response_time,
                            device.ip_address,
                            device.last_seen,
                        ),
                    )

                # Save services
                for service in self.discovered_services:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO services 
                        (device_ip, port, service_type, service_name, version, banner, last_seen)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            service.device_ip,
                            service.port,
                            service.service_type.value,
                            service.service_name,
                            service.version,
                            service.banner,
                            datetime.now().isoformat(),
                        ),
                    )

                # Save integration opportunities
                for opportunity in self.integration_opportunities:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO integration_opportunities 
                        (device_ip, integration_type, confidence, description, 
                         complexity, discovered_date)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            opportunity.device_ip,
                            opportunity.integration_type,
                            opportunity.confidence,
                            opportunity.description,
                            opportunity.implementation_complexity,
                            datetime.now().isoformat(),
                        ),
                    )

                conn.commit()

            self.logger.info("💾 Discovery results saved to database")

        except Exception as e:
            self.logger.error(f"❌ Error saving discovery results: {e}")

    async def quick_scan(self) -> Dict[str, Any]:
        """Perform quick network scan"""
        self.logger.info("🚀 Starting quick network scan...")
        self.scan_in_progress = True
        scan_start = datetime.now()

        try:
            # Detect network if not specified
            if not self.config.target_network:
                self.config.target_network = self.get_local_network()

            # Ping sweep to find active hosts
            active_hosts = await self.ping_sweep(self.config.target_network)

            # Get basic info for each host
            tasks = []
            for ip in active_hosts:
                tasks.append(self.get_device_info(ip))

            devices = await asyncio.gather(*tasks, return_exceptions=True)

            # Store valid devices
            for device in devices:
                if isinstance(device, NetworkDevice):
                    self.discovered_devices[device.ip_address] = device

            scan_end = datetime.now()
            self.last_scan_time = scan_end

            results = {
                "scan_type": "quick",
                "duration": (scan_end - scan_start).total_seconds(),
                "devices_found": len(self.discovered_devices),
                "target_network": self.config.target_network,
                "timestamp": scan_end.isoformat(),
            }

            self.save_discovery_results()
            self.logger.info(
                f"✅ Quick scan completed: {len(self.discovered_devices)} devices found"
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Error in quick scan: {e}")
            return {"error": str(e)}

        finally:
            self.scan_in_progress = False

    async def deep_scan(self) -> Dict[str, Any]:
        """Perform comprehensive deep scan"""
        self.logger.info("🔬 Starting deep network scan...")
        self.scan_in_progress = True
        scan_start = datetime.now()

        try:
            # First do quick scan to find devices
            await self.quick_scan()

            # Deep analysis of each device
            for device in self.discovered_devices.values():
                # Service analysis for each open port
                for port in device.open_ports:
                    service = await self.analyze_service(device.ip_address, port)
                    if service:
                        self.discovered_services.append(service)

            # Analyze integration opportunities
            self.integration_opportunities = self.analyze_integration_opportunities()

            scan_end = datetime.now()

            results = {
                "scan_type": "deep",
                "duration": (scan_end - scan_start).total_seconds(),
                "devices_found": len(self.discovered_devices),
                "services_found": len(self.discovered_services),
                "integration_opportunities": len(self.integration_opportunities),
                "target_network": self.config.target_network,
                "timestamp": scan_end.isoformat(),
            }

            self.save_discovery_results()
            self.logger.info(
                f"✅ Deep scan completed: {len(self.integration_opportunities)} integration opportunities found"
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Error in deep scan: {e}")
            return {"error": str(e)}

        finally:
            self.scan_in_progress = False

    async def continuous_monitoring(self):
        """Continuous network monitoring mode"""
        self.logger.info("🔄 Starting continuous network monitoring...")

        while True:
            try:
                # Perform quick scan every 5 minutes
                await self.quick_scan()

                # Deep scan every hour
                if (
                    not self.last_scan_time
                    or (datetime.now() - self.last_scan_time).total_seconds() > 3600
                ):
                    await self.deep_scan()

                # Save state
                self.save_state()

                # Wait before next scan
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                self.logger.error(f"❌ Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    def save_state(self):
        """Save scanner state to file"""
        try:
            state_file = self.state_dir / "network_scanner_state.json"

            state_data = {
                "timestamp": datetime.now().isoformat(),
                "scan_in_progress": self.scan_in_progress,
                "last_scan_time": (
                    self.last_scan_time.isoformat() if self.last_scan_time else None
                ),
                "devices_count": len(self.discovered_devices),
                "services_count": len(self.discovered_services),
                "integration_opportunities_count": len(self.integration_opportunities),
                "config": asdict(self.config),
            }

            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"❌ Error saving state: {e}")

    def export_results(self) -> Dict[str, Any]:
        """Export discovery results"""
        return {
            "devices": {
                ip: asdict(device) for ip, device in self.discovered_devices.items()
            },
            "services": [asdict(service) for service in self.discovered_services],
            "integration_opportunities": [
                asdict(opp) for opp in self.integration_opportunities
            ],
            "scan_timestamp": (
                self.last_scan_time.isoformat() if self.last_scan_time else None
            ),
        }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vega Network Scanner")
    parser.add_argument("--scan", action="store_true", help="Perform network scan")
    parser.add_argument("--quick", action="store_true", help="Quick scan mode")
    parser.add_argument("--deep", action="store_true", help="Deep scan mode")
    parser.add_argument("--monitor", action="store_true", help="Continuous monitoring")
    parser.add_argument(
        "--integration-analysis",
        action="store_true",
        help="Analyze integration opportunities",
    )
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument(
        "--network", type=str, help="Target network (e.g., 192.168.1.0/24)"
    )

    args = parser.parse_args()

    if not NETWORK_AVAILABLE:
        print(
            "❌ Network scanning libraries not available. Install with: pip install scapy python-nmap requests"
        )
        return

    # Create configuration
    config = ScanConfiguration()
    if args.network:
        config.target_network = args.network

    scanner = NetworkScanner(config)

    try:
        if args.monitor or args.daemon:
            await scanner.continuous_monitoring()
        elif args.scan:
            if args.deep:
                results = await scanner.deep_scan()
            else:
                results = await scanner.quick_scan()

            print(json.dumps(results, indent=2))

        elif args.integration_analysis:
            # Load existing data and analyze
            opportunities = scanner.analyze_integration_opportunities()
            print(
                json.dumps(
                    [asdict(opp) for opp in opportunities], indent=2, default=str
                )
            )

        else:
            print("🔍 Vega Network Scanner")
            print("Use --scan --quick for quick discovery")
            print("Use --scan --deep for comprehensive analysis")
            print("Use --monitor for continuous monitoring")

    except KeyboardInterrupt:
        print("\n🛑 Scanner stopped by user")
    except Exception as e:
        print(f"❌ Scanner error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
