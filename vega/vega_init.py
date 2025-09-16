#!/usr/bin/env python3
"""
VEGA INIT - MASTER SYSTEM BOOTSTRAPPER
======================================

The single source of truth entry point for the entire Vega Ambient AI ecosystem.
This module serves as the conductor, bootloader, and steward of local intelligence.

When you run this, Vega is ALIVE.

Features:
- 🚀 Complete system initialization and daemon management
- 🎙️ Audio-visual presence with voice reactive visualizations  
- 🌐 Dynamic network scanning and device discovery
- 🧠 Intelligent integration decision engine
- 🛡️ Ethical decision management with user fallback
- 📊 Comprehensive logging a    def save_system_state(self):
        """Save current system state to disk"""
        state_file = self.state_dir / "system_state.json"
        
        # Convert system state to serializable format
        system_state_dict = asdict(self.system_state)
        # Convert VegaMode enum to string
        if 'mode' in system_state_dict:
            system_state_dict['mode'] = system_state_dict['mode'].value if hasattr(system_state_dict['mode'], 'value') else str(system_state_dict['mode'])
        
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "system_state": system_state_dict,
            "boot_stats": asdict(self.boot_stats)
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)alth monitoring
- 🔄 Self-aware execution tracking and thermal management

Usage:
    python vega_init.py --mode=normal --log=info
    python vega_init.py --mode=safe --quiet
    python vega_init.py --mode=diagnostic --verbose
"""

import os
import sys
import json
import time
import signal
import asyncio
import logging
import threading
import subprocess
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import argparse

# Core system imports
try:
    import schedule
    import sqlite3
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.live import Live
    from rich.status import Status
    from rich import box
    RICH_AVAILABLE = True
except ImportError as e:
    RICH_AVAILABLE = False
    print(f"Warning: Some dependencies missing: {e}")

# Audio/Visual imports (will be checked dynamically)
AUDIO_AVAILABLE = False
try:
    import pyaudio
    import numpy as np
    import matplotlib.pyplot as plt
    import pygame
    AUDIO_AVAILABLE = True
except ImportError:
    pass

# Network scanning imports (will be checked dynamically)
NETWORK_AVAILABLE = False
try:
    import scapy.all as scapy
    import nmap
    import requests
    NETWORK_AVAILABLE = True
except ImportError:
    pass

class VegaMode(Enum):
    """Vega operation modes"""
    NORMAL = "normal"          # Full functionality
    SAFE = "safe"              # Reduced activity, no autonomous patches
    QUIET = "quiet"            # No verbal/spoken output
    DIAGNOSTIC = "diagnostic"  # Log-only, no execution

class SystemHealth(Enum):
    """System health states"""
    OPTIMAL = "optimal"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class BootStats:
    """Boot cycle statistics"""
    start_time: str
    mode: str
    subsystems_loaded: List[str]
    subsystems_failed: List[str]
    load_times: Dict[str, float]
    total_boot_time: float
    health_score: float
    warnings: List[str]
    errors: List[str]

@dataclass
class SystemState:
    """Current system state"""
    mode: VegaMode
    health: SystemHealth
    uptime_seconds: float
    cpu_temp: float
    gpu_temp: float
    memory_usage: float
    cpu_usage: float
    active_daemons: List[str]
    network_devices: int
    last_scan_time: str
    audio_enabled: bool
    visual_enabled: bool
    ethics_queue_size: int

class VegaBootCore:
    """Master system initializer and orchestrator"""
    
    def __init__(self, mode: VegaMode = VegaMode.NORMAL, log_level: str = "INFO"):
        self.mode = mode
        self.start_time = datetime.now()
        self.base_dir = Path(__file__).parent
        self.state_dir = self.base_dir / "vega_state"
        self.logs_dir = self.base_dir / "vega_logs"
        self.tools_dir = self.base_dir / "vega_integrations" / "tools"
        
        # Create directories
        for directory in [self.state_dir, self.logs_dir, self.tools_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.setup_logging(log_level)
        self.logger = logging.getLogger("VegaBootCore")
        
        # Initialize console
        self.console = Console() if RICH_AVAILABLE else None
        
        # System state
        self.boot_stats = BootStats(
            start_time=self.start_time.isoformat(),
            mode=mode.value,
            subsystems_loaded=[],
            subsystems_failed=[],
            load_times={},
            total_boot_time=0.0,
            health_score=100.0,
            warnings=[],
            errors=[]
        )
        
        self.system_state = SystemState(
            mode=mode,
            health=SystemHealth.OPTIMAL,
            uptime_seconds=0.0,
            cpu_temp=0.0,
            gpu_temp=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            active_daemons=[],
            network_devices=0,
            last_scan_time="never",
            audio_enabled=AUDIO_AVAILABLE,
            visual_enabled=AUDIO_AVAILABLE,  # Audio required for visualizer
            ethics_queue_size=0
        )
        
        # Daemon processes
        self.daemons = {}
        self.shutdown_requested = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Ethics and decision queues
        self.ethics_queue = []
        self.decision_patterns = {}
        
        self.logger.info(f"🤖 Vega Boot Core initialized in {mode.value} mode")
    
    def setup_logging(self, level: str):
        """Configure comprehensive logging system"""
        log_format = "%(asctime)s | %(levelname)8s | %(name)20s | %(message)s"
        
        # Main log file
        main_log = self.logs_dir / "vega_main.log"
        
        # Boot log file  
        boot_log = self.logs_dir / "startup_log.jsonl"
        
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(main_log),
                logging.StreamHandler(sys.stdout) if not self.mode == VegaMode.QUIET else logging.NullHandler()
            ]
        )
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"📡 Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def print_banner(self):
        """Display startup banner"""
        if self.console and self.mode != VegaMode.QUIET:
            banner_text = f"""
🤖 VEGA AMBIENT AI SYSTEM CORE
===============================

Mode: {self.mode.value.upper()}
Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}

Initializing complete ecosystem...
"""
            panel = Panel(
                banner_text.strip(),
                title="🚀 SYSTEM BOOT",
                style="blue bold",
                box=box.DOUBLE
            )
            self.console.print(panel)
        else:
            print(f"🤖 Vega System Core - {self.mode.value} mode - {self.start_time}")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check system dependencies and capabilities"""
        self.logger.info("🔍 Checking system dependencies...")
        
        dependencies = {
            "rich_ui": RICH_AVAILABLE,
            "audio_system": AUDIO_AVAILABLE,
            "network_tools": NETWORK_AVAILABLE,
            "schedule": 'schedule' in sys.modules,
            "sqlite": 'sqlite3' in sys.modules,
        }
        
        # Check for existing Vega modules
        vega_modules = [
            "vega_loop.py",
            "idle_personality_engine.py", 
            "user_presence.py",
            "interaction_log.py",
            "spontaneous_thought_engine.py",
            "advanced_augmentations.py"
        ]
        
        for module in vega_modules:
            module_path = self.base_dir / module
            dependencies[module.replace('.py', '')] = module_path.exists()
        
        # Log dependency status
        for dep, available in dependencies.items():
            status = "✅" if available else "❌"
            self.logger.info(f"  {status} {dep}: {'Available' if available else 'Missing'}")
            
            if not available:
                self.boot_stats.warnings.append(f"Missing dependency: {dep}")
        
        return dependencies
    
    def measure_system_health(self) -> SystemHealth:
        """Assess current system health"""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Temperature monitoring (if available)
            cpu_temp = 0.0
            gpu_temp = 0.0
            
            try:
                temps = psutil.sensors_temperatures()
                if 'coretemp' in temps:
                    cpu_temp = max([sensor.current for sensor in temps['coretemp']])
                elif 'cpu_thermal' in temps:
                    cpu_temp = temps['cpu_thermal'][0].current
            except:
                pass
            
            # Update system state
            self.system_state.cpu_usage = cpu_usage
            self.system_state.memory_usage = memory.percent
            self.system_state.cpu_temp = cpu_temp
            self.system_state.gpu_temp = gpu_temp
            
            # Calculate health score
            health_score = 100.0
            
            # Penalize high resource usage
            if cpu_usage > 90:
                health_score -= 30
            elif cpu_usage > 75:
                health_score -= 15
            
            if memory.percent > 90:
                health_score -= 25
            elif memory.percent > 75:
                health_score -= 10
            
            # Penalize high temperatures
            if cpu_temp > 80:
                health_score -= 20
            elif cpu_temp > 70:
                health_score -= 10
            
            # Determine health status
            if health_score >= 90:
                health = SystemHealth.OPTIMAL
            elif health_score >= 75:
                health = SystemHealth.GOOD
            elif health_score >= 60:
                health = SystemHealth.WARNING
            elif health_score >= 40:
                health = SystemHealth.CRITICAL
            else:
                health = SystemHealth.EMERGENCY
            
            self.system_state.health = health
            self.boot_stats.health_score = health_score
            
            # Log health status
            self.logger.info(f"💚 System Health: {health.value} (Score: {health_score:.1f})")
            self.logger.info(f"📊 CPU: {cpu_usage:.1f}% | Memory: {memory.percent:.1f}% | Temp: {cpu_temp:.1f}°C")
            
            return health
            
        except Exception as e:
            self.logger.error(f"❌ Error measuring system health: {e}")
            return SystemHealth.CRITICAL
    
    def load_subsystem(self, name: str, loader_func) -> bool:
        """Load a subsystem with timing and error handling"""
        start_time = time.time()
        
        try:
            if self.console:
                with Status(f"Loading {name}...", spinner="dots"):
                    result = loader_func()
            else:
                self.logger.info(f"⚙️ Loading {name}...")
                result = loader_func()
            
            load_time = time.time() - start_time
            self.boot_stats.load_times[name] = load_time
            
            if result:
                self.boot_stats.subsystems_loaded.append(name)
                self.logger.info(f"✅ {name} loaded successfully ({load_time:.2f}s)")
                return True
            else:
                self.boot_stats.subsystems_failed.append(name)
                self.logger.warning(f"⚠️ {name} failed to load ({load_time:.2f}s)")
                return False
                
        except Exception as e:
            load_time = time.time() - start_time
            self.boot_stats.load_times[name] = load_time
            self.boot_stats.subsystems_failed.append(name)
            self.boot_stats.errors.append(f"{name}: {str(e)}")
            self.logger.error(f"❌ Error loading {name}: {e}")
            return False
    
    def start_daemon(self, name: str, command: List[str], cwd: Optional[Path] = None) -> bool:
        """Start a background daemon process"""
        try:
            if self.mode == VegaMode.DIAGNOSTIC:
                self.logger.info(f"🔍 [DIAGNOSTIC] Would start daemon: {name}")
                return True
            
            cwd = cwd or self.base_dir
            
            proc = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True
            )
            
            # Give daemon time to start
            time.sleep(2)
            
            if proc.poll() is None:  # Process still running
                self.daemons[name] = proc
                self.system_state.active_daemons.append(name)
                self.logger.info(f"🔄 Daemon {name} started (PID: {proc.pid})")
                return True
            else:
                stderr = proc.stderr.read().decode() if proc.stderr else "Unknown error"
                self.logger.error(f"❌ Daemon {name} failed to start: {stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting daemon {name}: {e}")
            return False
    
    def initialize_core_daemons(self) -> bool:
        """Initialize core Vega daemon processes"""
        self.logger.info("🔄 Starting core daemon processes...")
        
        # Core daemons to start
        daemons = [
            ("vega_loop", [sys.executable, "vega_loop.py", "--daemon"]),
            ("voice_visualizer", [sys.executable, "voice_visualizer.py", "--daemon"]),
            ("network_scanner", [sys.executable, "network_scanner.py", "--daemon"]),
            ("integration_engine", [sys.executable, "integration_engine.py", "--daemon"]),
        ]
        
        success_count = 0
        for name, command in daemons:
            if self.load_subsystem(f"daemon_{name}", lambda cmd=command, n=name: self.start_daemon(n, cmd)):
                success_count += 1
        
        self.logger.info(f"✅ Started {success_count}/{len(daemons)} core daemons")
        return success_count > 0
    
    def initialize_ui_services(self) -> bool:
        """Initialize web UI and dashboard services"""
        if self.mode == VegaMode.QUIET:
            self.logger.info("🔇 Skipping UI services in quiet mode")
            return True
        
        self.logger.info("🌐 Starting UI services...")
        
        # Check if UI dependencies are available
        ui_env = self.base_dir / "vega_ui_env" / "bin" / "python"
        if not ui_env.exists():
            self.logger.warning("⚠️ UI environment not found, skipping UI services")
            return False
        
        # Start dashboard service
        dashboard_started = self.start_daemon(
            "dashboard",
            [str(ui_env), "vega_dashboard.py", "--port", "8080"]
        )
        
        if dashboard_started:
            self.logger.info("🌐 Dashboard available at http://127.0.0.1:8080")
        
        return dashboard_started
    
    def save_boot_log(self):
        """Save boot statistics to startup log"""
        self.boot_stats.total_boot_time = (datetime.now() - self.start_time).total_seconds()
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "boot_stats": asdict(self.boot_stats),
            "system_state": asdict(self.system_state)
        }
        
        boot_log_file = self.logs_dir / "startup_log.jsonl"
        with open(boot_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        self.logger.info(f"📄 Boot log saved to {boot_log_file}")
    
    def display_boot_summary(self):
        """Display boot summary"""
        if self.console and self.mode != VegaMode.QUIET:
            # Create summary table
            table = Table(title="🚀 Vega Boot Summary", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Status", justify="center")
            table.add_column("Load Time", justify="right", style="dim")
            
            # Add loaded systems
            for system in self.boot_stats.subsystems_loaded:
                load_time = self.boot_stats.load_times.get(system, 0)
                table.add_row(system, "✅ Loaded", f"{load_time:.2f}s")
            
            # Add failed systems
            for system in self.boot_stats.subsystems_failed:
                load_time = self.boot_stats.load_times.get(system, 0)
                table.add_row(system, "❌ Failed", f"{load_time:.2f}s")
            
            self.console.print(table)
            
            # Health summary
            health_color = {
                SystemHealth.OPTIMAL: "green",
                SystemHealth.GOOD: "green", 
                SystemHealth.WARNING: "yellow",
                SystemHealth.CRITICAL: "red",
                SystemHealth.EMERGENCY: "red bold"
            }.get(self.system_state.health, "white")
            
            health_panel = Panel(
                f"Health: {self.system_state.health.value.upper()}\n"
                f"Score: {self.boot_stats.health_score:.1f}/100\n"
                f"Boot Time: {self.boot_stats.total_boot_time:.2f}s\n"
                f"Active Daemons: {len(self.system_state.active_daemons)}",
                title="💚 System Status",
                style=health_color
            )
            self.console.print(health_panel)
    
    async def main_loop(self):
        """Main system monitoring and coordination loop"""
        self.logger.info("🔄 Entering main system loop...")
        
        last_health_check = time.time()
        last_network_scan = 0
        
        while not self.shutdown_requested:
            try:
                current_time = time.time()
                
                # Update uptime
                self.system_state.uptime_seconds = current_time - self.start_time.timestamp()
                
                # Periodic health checks (every 60 seconds)
                if current_time - last_health_check >= 60:
                    health = self.measure_system_health()
                    last_health_check = current_time
                    
                    # Emergency thermal management
                    if self.system_state.cpu_temp > 75 or self.system_state.gpu_temp > 80:
                        self.logger.warning("🌡️ High temperature detected, reducing system load")
                        self.thermal_protection()
                
                # Periodic network scanning (every hour)
                if current_time - last_network_scan >= 3600:  # 1 hour
                    if NETWORK_AVAILABLE and self.mode != VegaMode.SAFE:
                        await self.trigger_network_scan()
                        last_network_scan = current_time
                
                # Process ethics queue
                if self.ethics_queue:
                    await self.process_ethics_queue()
                
                # Check daemon health
                self.check_daemon_health()
                
                # Save state periodically
                self.save_system_state()
                
                # Sleep before next cycle
                await asyncio.sleep(30)  # 30 second cycle
                
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    def thermal_protection(self):
        """Activate thermal protection measures"""
        self.logger.warning("🛡️ Activating thermal protection...")
        
        # Reduce scan intervals
        # Suspend heavy computations
        # Notify user if temperature is critical
        
        if self.system_state.cpu_temp > 80:
            self.logger.critical("🚨 Critical temperature! Consider system shutdown.")
    
    async def trigger_network_scan(self):
        """Trigger network scanning daemon"""
        try:
            # Signal network scanner to perform scan
            scanner_log = self.state_dir / "network_scan_trigger.json"
            trigger_data = {
                "timestamp": datetime.now().isoformat(),
                "requested_by": "main_loop",
                "scan_type": "full"
            }
            
            with open(scanner_log, 'w') as f:
                json.dump(trigger_data, f)
            
            self.logger.info("🌐 Network scan triggered")
            
        except Exception as e:
            self.logger.error(f"❌ Error triggering network scan: {e}")
    
    async def process_ethics_queue(self):
        """Process pending ethical decisions"""
        for decision in self.ethics_queue[:]:  # Copy to avoid modification during iteration
            await self.handle_ethical_decision(decision)
    
    async def handle_ethical_decision(self, decision: Dict[str, Any]):
        """Handle a single ethical decision"""
        # This would integrate with user notification system
        self.logger.info(f"🤔 Processing ethical decision: {decision['type']}")
        # Implementation would depend on user interface preferences
    
    def check_daemon_health(self):
        """Check health of running daemons"""
        dead_daemons = []
        
        for name, proc in self.daemons.items():
            if proc.poll() is not None:  # Process has terminated
                dead_daemons.append(name)
                self.logger.warning(f"💀 Daemon {name} has died (exit code: {proc.returncode})")
        
        # Remove dead daemons and restart if needed
        for name in dead_daemons:
            del self.daemons[name]
            if name in self.system_state.active_daemons:
                self.system_state.active_daemons.remove(name)
            
            # Attempt restart in safe mode
            if self.mode != VegaMode.DIAGNOSTIC:
                self.logger.info(f"🔄 Attempting to restart daemon: {name}")
                # Restart logic would go here
    
    def save_system_state(self):
        """Save current system state"""
        state_file = self.state_dir / "system_state.json"
        
        # Convert system state to serializable format
        system_state_dict = asdict(self.system_state)
        # Convert VegaMode enum to string value
        if 'mode' in system_state_dict:
            system_state_dict['mode'] = system_state_dict['mode'].value if hasattr(system_state_dict['mode'], 'value') else str(system_state_dict['mode'])
        
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "system_state": system_state_dict,
            "boot_stats": asdict(self.boot_stats)
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def cleanup_and_shutdown(self):
        """Graceful system shutdown"""
        self.logger.info("🛑 Beginning graceful shutdown...")
        
        # Stop all daemons
        for name, proc in self.daemons.items():
            self.logger.info(f"🔄 Stopping daemon: {name}")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"⏰ Force killing daemon: {name}")
                proc.kill()
        
        # Save final state
        self.save_system_state()
        self.save_boot_log()
        
        self.logger.info("✅ Vega system shutdown complete")
    
    async def run(self):
        """Main entry point - orchestrate the complete system"""
        try:
            # Phase 1: Boot sequence
            self.print_banner()
            
            # Check dependencies
            dependencies = self.check_dependencies()
            
            # Measure initial system health
            initial_health = self.measure_system_health()
            
            if initial_health == SystemHealth.EMERGENCY:
                self.logger.critical("🚨 System health critical! Aborting startup.")
                return False
            
            # Initialize core systems
            success = True
            
            # Load core daemons
            if not self.load_subsystem("core_daemons", self.initialize_core_daemons):
                success = False
            
            # Load UI services (if not in quiet mode)
            if not self.load_subsystem("ui_services", self.initialize_ui_services):
                if self.mode != VegaMode.QUIET:
                    self.boot_stats.warnings.append("UI services failed to start")
            
            # Display boot summary
            self.display_boot_summary()
            self.save_boot_log()
            
            if not success and self.mode != VegaMode.DIAGNOSTIC:
                self.logger.error("❌ Critical systems failed to load")
                return False
            
            # Phase 2: Enter main loop
            if self.mode != VegaMode.DIAGNOSTIC:
                self.logger.info("🚀 Vega is now ALIVE and entering main coordination loop")
                await self.main_loop()
            else:
                self.logger.info("🔍 Diagnostic mode complete - no execution performed")
            
            return True
            
        except KeyboardInterrupt:
            self.logger.info("⌨️ Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"💥 Fatal error during execution: {e}", exc_info=True)
            return False
        finally:
            self.cleanup_and_shutdown()

def main():
    """Command line entry point"""
    parser = argparse.ArgumentParser(
        description="Vega Ambient AI System Core - Master Initializer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🤖 VEGA SYSTEM CORE

Examples:
  python vega_init.py                          # Normal mode
  python vega_init.py --mode=safe              # Safe mode (no autonomous patches)
  python vega_init.py --mode=quiet --log=warn # Quiet mode with minimal logging  
  python vega_init.py --mode=diagnostic       # Diagnostic mode (no execution)
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['normal', 'safe', 'quiet', 'diagnostic'],
        default='normal',
        help='Operating mode (default: normal)'
    )
    
    parser.add_argument(
        '--log',
        choices=['debug', 'info', 'warn', 'error'],
        default='info',
        help='Logging level (default: info)'
    )
    
    parser.add_argument(
        '--no-ui',
        action='store_true',
        help='Skip UI services startup'
    )
    
    args = parser.parse_args()
    
    # Convert string mode to enum
    mode = VegaMode(args.mode)
    
    # Override mode for no-ui
    if args.no_ui and mode != VegaMode.QUIET:
        mode = VegaMode.QUIET
    
    # Create and run the boot core
    boot_core = VegaBootCore(mode=mode, log_level=args.log.upper())
    
    # Run the system
    try:
        success = asyncio.run(boot_core.run())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Vega shutdown by user")
        sys.exit(0)

if __name__ == "__main__":
    main()