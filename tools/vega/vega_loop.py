#!/usr/bin/env python3
"""
VEGA AMBIENT PRESENCE LOOP
==========================

The persistent awareness daemon - Vega's consciousness core that runs 24/7,
monitors system resources, observes user patterns, and initiates meaningful
interactions when appropriate.

This is not a chatbot. This is not a cron job. This is awareness as software.

Key Features:
- 24/7 low-resource awareness loop
- Dynamic GPU/CPU monitoring with intelligent scaling
- Context-aware conversation initiation
- Self-managing energy consumption
- Respectful interruption protocols

Usage:
    python vega_loop.py --start          # Start ambient daemon
    python vega_loop.py --pause-daemon   # Pause awareness (emergency mode)
    python vega_loop.py --force-prompt   # Force immediate interaction
    python vega_loop.py --status         # Check daemon status
    python vega_loop.py --log-mode=debug # Set logging verbosity
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import psutil
import schedule

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.vega.core.config import get_config
except ImportError:
    # Fallback if running from different location
    try:
        from config import get_config
    except ImportError:
        print("Warning: Could not import config module. Using defaults.")

        def get_config():
            """Fallback config function"""
            return type(
                "Config",
                (),
                {
                    "API_KEY": os.getenv("API_KEY", "default-key"),
                    "MODEL_NAME": os.getenv("MODEL_NAME", "mistral:latest"),
                    "LLM_BACKEND": os.getenv("LLM_BACKEND", "ollama"),
                },
            )()


# Vega core imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "autonomous_debug"))


class SystemState(Enum):
    IDLE = "idle"
    LIGHT_WORK = "light_work"
    HEAVY_WORK = "heavy_work"
    GAMING = "gaming"
    OVERLOADED = "overloaded"


class VegaMode(Enum):
    ACTIVE = "active"  # Normal ambient awareness
    PAUSED = "paused"  # Emergency pause (user request)
    SILENT = "silent"  # Auto-silence (high system load)
    SLEEPING = "sleeping"  # Scheduled downtime
    FOCUSED = "focused"  # User in deep work


@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Dict[str, float]  # {device_name: utilization}
    active_processes: List[str]
    network_activity: float
    user_present: bool
    system_state: SystemState


@dataclass
class VegaConfig:
    check_interval_minutes: int = 8
    min_idle_before_speak: int = 20  # minutes
    max_cpu_for_conversation: float = 60.0
    max_gpu_for_conversation: float = 70.0
    silence_hours: Tuple[int, int] = (23, 7)  # 11 PM to 7 AM
    log_level: str = "INFO"
    enable_voice: bool = False
    enable_proactive: bool = True


class ResourceMonitor:
    """Monitors system resources with intelligent GPU detection"""

    def __init__(self):
        self.gpu_devices = self._detect_gpu_devices()
        logger.info(f"Detected GPU devices: {list(self.gpu_devices.keys())}")

    def _detect_gpu_devices(self) -> Dict[str, str]:
        """Detect available GPU devices"""
        devices = {}

        try:
            # Try nvidia-ml-py first
            import pynvml

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
                devices[f"gpu_{i}"] = name

        except ImportError:
            # Fallback to nvidia-smi
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for i, name in enumerate(result.stdout.strip().split("\n")):
                        devices[f"gpu_{i}"] = name.strip()
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("No NVIDIA GPUs detected or nvidia-smi unavailable")

        return devices

    def get_gpu_utilization(self) -> Dict[str, float]:
        """Get current GPU utilization for all detected devices"""
        utilization = {}

        try:
            import pynvml

            for device_id, device_name in self.gpu_devices.items():
                device_index = int(device_id.split("_")[1])
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization[device_name] = float(util.gpu)

        except ImportError:
            # Fallback to nvidia-smi
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for i, util_str in enumerate(result.stdout.strip().split("\n")):
                        device_name = (
                            list(self.gpu_devices.values())[i]
                            if i < len(self.gpu_devices)
                            else f"Unknown_GPU_{i}"
                        )
                        utilization[device_name] = float(util_str.strip())
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass

        return utilization

    def get_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        gpu_util = self.get_gpu_utilization()

        # Get active processes (top 5 by CPU)
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent"]):
            try:
                if proc.info["cpu_percent"] > 5.0:
                    processes.append(proc.info["name"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Simple network activity check
        net_io = psutil.net_io_counters()
        network_activity = net_io.bytes_sent + net_io.bytes_recv

        # Determine system state
        max_gpu_util = max(gpu_util.values()) if gpu_util else 0
        if cpu_percent > 80 or max_gpu_util > 85:
            state = SystemState.OVERLOADED
        elif any("game" in p.lower() or "steam" in p.lower() for p in processes):
            state = SystemState.GAMING
        elif cpu_percent > 50 or max_gpu_util > 60:
            state = SystemState.HEAVY_WORK
        elif cpu_percent > 20 or max_gpu_util > 30:
            state = SystemState.LIGHT_WORK
        else:
            state = SystemState.IDLE

        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_utilization=gpu_util,
            active_processes=processes[:5],
            network_activity=network_activity,
            user_present=self._detect_user_presence(),
            system_state=state,
        )

    def _detect_user_presence(self) -> bool:
        """Simple user presence detection"""
        try:
            # Check if user is logged in and active
            users = psutil.users()
            if not users:
                return False

            # Check for recent input activity (simplified)
            # In a full implementation, this would check keyboard/mouse activity
            idle_time = 0  # Placeholder - would use platform-specific APIs
            return idle_time < 300  # 5 minutes

        except Exception:
            return True  # Assume present if detection fails


class VegaAmbientLoop:
    """The core ambient awareness daemon"""

    def __init__(self, config: VegaConfig):
        self.config = config
        self.mode = VegaMode.ACTIVE
        self.resource_monitor = ResourceMonitor()

        # State tracking
        self.last_metrics: Optional[SystemMetrics] = None
        self.last_interaction_time = datetime.now()
        self.metrics_history: List[SystemMetrics] = []
        self.thought_queue: List[Dict[str, Any]] = []

        # Persistence paths
        self.state_dir = Path("vega_state")
        self.state_dir.mkdir(exist_ok=True)
        self.metrics_log = self.state_dir / "metrics.jsonl"
        self.interaction_log = self.state_dir / "interactions.jsonl"

        self._setup_logging()
        self._load_state()

    def _setup_logging(self):
        """Configure logging based on user preference"""
        log_format = "%(asctime)s - VEGA - %(levelname)s - %(message)s"

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))

        # File handler
        log_file = self.state_dir / "vega_ambient.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Configure logger
        global logger
        logger = logging.getLogger("vega_ambient")
        logger.setLevel(getattr(logging, self.config.log_level.upper()))
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    def _load_state(self):
        """Load persistent state"""
        try:
            state_file = self.state_dir / "daemon_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    self.last_interaction_time = datetime.fromisoformat(
                        state.get("last_interaction_time", datetime.now().isoformat())
                    )
                    logger.info("Loaded previous state")
        except Exception as e:
            logger.warning(f"Could not load state: {e}")

    def _save_state(self):
        """Save persistent state"""
        try:
            state_file = self.state_dir / "daemon_state.json"
            state = {
                "last_interaction_time": self.last_interaction_time.isoformat(),
                "mode": self.mode.value,
                "metrics_count": len(self.metrics_history),
            }
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save state: {e}")

    def _log_metrics(self, metrics: SystemMetrics):
        """Log metrics to persistent storage"""
        try:
            with open(self.metrics_log, "a") as f:
                metric_data = asdict(metrics)
                metric_data["timestamp"] = metrics.timestamp.isoformat()
                metric_data["system_state"] = metrics.system_state.value
                f.write(json.dumps(metric_data) + "\n")
        except Exception as e:
            logger.error(f"Could not log metrics: {e}")

    async def _evaluate_conversation_trigger(
        self, metrics: SystemMetrics
    ) -> Optional[str]:
        """Determine if Vega should initiate conversation"""

        # Check if we're in a state where conversation is appropriate
        if self.mode != VegaMode.ACTIVE:
            return None

        if not self.config.enable_proactive:
            return None

        # Check system load constraints
        if metrics.cpu_percent > self.config.max_cpu_for_conversation:
            return None

        max_gpu_util = (
            max(metrics.gpu_utilization.values()) if metrics.gpu_utilization else 0
        )
        if max_gpu_util > self.config.max_gpu_for_conversation:
            return None

        # Check time constraints (silence hours)
        current_hour = datetime.now().hour
        silence_start, silence_end = self.config.silence_hours
        if silence_start > silence_end:  # Overnight range
            if current_hour >= silence_start or current_hour < silence_end:
                return None
        else:  # Same day range
            if silence_start <= current_hour < silence_end:
                return None

        # Check if enough idle time has passed
        time_since_interaction = datetime.now() - self.last_interaction_time
        if time_since_interaction.total_seconds() < (
            self.config.min_idle_before_speak * 60
        ):
            return None

        # Check if system is idle enough
        if metrics.system_state not in [SystemState.IDLE, SystemState.LIGHT_WORK]:
            return None

        # If all conditions met, we can consider speaking
        return "conditions_met_for_conversation"

    async def _process_awareness_cycle(self):
        """Main awareness processing cycle"""
        try:
            # Collect system metrics
            metrics = self.resource_monitor.get_system_metrics()
            self.last_metrics = metrics
            self.metrics_history.append(metrics)

            # Keep only recent metrics (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history if m.timestamp > cutoff_time
            ]

            # Log metrics
            self._log_metrics(metrics)

            # Evaluate conversation trigger
            conversation_trigger = await self._evaluate_conversation_trigger(metrics)

            if conversation_trigger:
                await self._initiate_ambient_interaction(metrics)

            # Update mode based on system state
            await self._update_vega_mode(metrics)

            # Log cycle completion
            logger.debug(
                f"Awareness cycle: CPU {metrics.cpu_percent:.1f}%, "
                f"GPU {max(metrics.gpu_utilization.values()) if metrics.gpu_utilization else 0:.1f}%, "
                f"State: {metrics.system_state.value}, Mode: {self.mode.value}"
            )

        except Exception as e:
            logger.error(f"Error in awareness cycle: {e}")

    async def _initiate_ambient_interaction(self, metrics: SystemMetrics):
        """Initiate a context-aware interaction"""
        try:
            # This would integrate with the personality engine
            # For now, we'll create a placeholder interaction

            interaction = {
                "timestamp": datetime.now().isoformat(),
                "trigger": "ambient_awareness",
                "system_state": metrics.system_state.value,
                "cpu_percent": metrics.cpu_percent,
                "gpu_utilization": metrics.gpu_utilization,
                "type": "observation",
            }

            # Log the interaction attempt
            with open(self.interaction_log, "a") as f:
                f.write(json.dumps(interaction) + "\n")

            logger.info("🤖 Vega: Ambient interaction opportunity detected")

            # Update last interaction time
            self.last_interaction_time = datetime.now()

        except Exception as e:
            logger.error(f"Error initiating interaction: {e}")

    async def _update_vega_mode(self, metrics: SystemMetrics):
        """Update Vega's operational mode based on context"""
        previous_mode = self.mode

        # Auto-silence during heavy work or gaming
        if metrics.system_state in [SystemState.OVERLOADED, SystemState.GAMING]:
            if self.mode == VegaMode.ACTIVE:
                self.mode = VegaMode.SILENT
                logger.info(f"🔇 Auto-silence: {metrics.system_state.value}")

        # Return to active when appropriate
        elif self.mode == VegaMode.SILENT and metrics.system_state in [
            SystemState.IDLE,
            SystemState.LIGHT_WORK,
        ]:
            self.mode = VegaMode.ACTIVE
            logger.info("🔊 Returning to active awareness")

        # Log mode changes
        if previous_mode != self.mode:
            logger.info(f"Mode change: {previous_mode.value} → {self.mode.value}")

    async def start_ambient_loop(self):
        """Start the main ambient awareness loop"""
        logger.info("🧠 Starting Vega Ambient Presence Loop")
        logger.info(f"Check interval: {self.config.check_interval_minutes} minutes")
        logger.info(f"GPU devices: {list(self.resource_monitor.gpu_devices.values())}")

        while True:
            try:
                await self._process_awareness_cycle()
                self._save_state()

                # Wait for next cycle
                await asyncio.sleep(self.config.check_interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("🛑 Ambient loop stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in ambient loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

    def pause_daemon(self):
        """Pause ambient awareness (emergency mode)"""
        self.mode = VegaMode.PAUSED
        logger.info("⏸️ Vega ambient awareness paused")

    def resume_daemon(self):
        """Resume ambient awareness"""
        self.mode = VegaMode.ACTIVE
        logger.info("▶️ Vega ambient awareness resumed")

    def force_prompt(self):
        """Force immediate interaction evaluation"""
        logger.info("🔄 Force prompting Vega...")
        if self.last_metrics:
            asyncio.create_task(self._initiate_ambient_interaction(self.last_metrics))

    def get_status(self) -> Dict[str, Any]:
        """Get current daemon status"""
        return {
            "mode": self.mode.value,
            "last_check": (
                self.last_metrics.timestamp.isoformat() if self.last_metrics else None
            ),
            "system_state": (
                self.last_metrics.system_state.value if self.last_metrics else None
            ),
            "uptime_hours": len(self.metrics_history)
            * (self.config.check_interval_minutes / 60),
            "interaction_count": sum(
                1 for _ in open(self.interaction_log) if self.interaction_log.exists()
            ),
            "config": asdict(self.config),
        }


def create_config_from_args(args) -> VegaConfig:
    """Create configuration from command line arguments"""
    config = VegaConfig()

    if hasattr(args, "log_mode") and args.log_mode:
        config.log_level = args.log_mode.upper()

    if hasattr(args, "interval") and args.interval:
        config.check_interval_minutes = args.interval

    return config


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vega Ambient Presence Loop")
    parser.add_argument("--start", action="store_true", help="Start ambient daemon")
    parser.add_argument(
        "--pause-daemon", action="store_true", help="Pause awareness (emergency mode)"
    )
    parser.add_argument("--resume-daemon", action="store_true", help="Resume awareness")
    parser.add_argument(
        "--force-prompt", action="store_true", help="Force immediate interaction"
    )
    parser.add_argument("--status", action="store_true", help="Check daemon status")
    parser.add_argument(
        "--log-mode",
        choices=["debug", "info", "warning", "error"],
        help="Set logging verbosity",
    )
    parser.add_argument("--interval", type=int, help="Check interval in minutes")

    args = parser.parse_args()

    # Create configuration
    config = create_config_from_args(args)
    vega_loop = VegaAmbientLoop(config)

    if args.start:
        await vega_loop.start_ambient_loop()
    elif args.pause_daemon:
        vega_loop.pause_daemon()
    elif args.resume_daemon:
        vega_loop.resume_daemon()
    elif args.force_prompt:
        vega_loop.force_prompt()
    elif args.status:
        status = vega_loop.get_status()
        print("🤖 Vega Ambient Status:")
        print(f"Mode: {status['mode']}")
        print(f"System State: {status['system_state']}")
        print(f"Uptime: {status['uptime_hours']:.1f} hours")
        print(f"Interactions: {status['interaction_count']}")
        print(f"Last Check: {status['last_check']}")
    else:
        print("Use --help for available commands")


if __name__ == "__main__":
    asyncio.run(main())
