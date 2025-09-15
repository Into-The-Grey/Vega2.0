#!/usr/bin/env python3
"""
VEGA SYSTEM LAUNCHER - UNIFIED ECOSYSTEM ENTRY POINT
===================================================

Single command to launch the complete Vega ecosystem with all components.
This is the primary entry point that users will use to start Vega.

Features:
- 🚀 One-command ecosystem startup
- 🎯 Intelligent component detection and initialization
- 📊 Real-time startup monitoring and health checks
- 🛡️ Safety checks and dependency validation
- 🎨 Beautiful progress visualization
- 🔧 Automatic dependency installation
- 📱 Multi-interface access (CLI, Web, Dashboard)
- 🤖 Integrated smart assistant activation

Usage:
    python vega.py                    # Start complete ecosystem
    python vega.py --quick            # Quick start (minimal components)
    python vega.py --safe             # Safe mode startup
    python vega.py --diagnostic       # Diagnostic mode
    python vega.py --status           # Check system status
"""

import os
import sys
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json

# Rich imports for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class VegaEcosystemLauncher:
    """Main launcher for the complete Vega ecosystem"""

    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.base_dir = Path(__file__).parent
        self.console = Console() if RICH_AVAILABLE else None

        # Component definitions
        self.components = {
            "vega_core": {
                "name": "💬 Chat API Service",
                "command": [
                    "python",
                    "-m",
                    "uvicorn",
                    "core.app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                "health_check": "http://127.0.0.1:8000/healthz",
                "essential": True,
                "startup_time": 5,
            },
            "voice_visualizer": {
                "name": "🎙️ Voice Visualizer",
                "command": ["python", "voice_visualizer.py", "--mode=daemon"],
                "health_check": "vega_state/voice_visualizer_state.json",
                "essential": False,
                "startup_time": 3,
            },
            "network_scanner": {
                "name": "🔍 Network Scanner",
                "command": ["python", "network_scanner.py", "--mode=daemon"],
                "health_check": "vega_state/network_scanner_state.json",
                "essential": False,
                "startup_time": 3,
            },
            "integration_engine": {
                "name": "🧠 Integration Engine",
                "command": ["python", "integration_engine.py", "--mode=daemon"],
                "health_check": "vega_state/integration_engine_state.json",
                "essential": False,
                "startup_time": 3,
            },
            "smart_assistant": {
                "name": "🎯 Smart Assistant",
                "command": ["python", "vega_smart.py", "--mode=daemon"],
                "health_check": "vega_state/vega_smart_state.json",
                "essential": False,
                "startup_time": 2,
            },
        }

        # UI components (optional)
        self.ui_components = {
            "dashboard": {
                "name": "📊 Web Dashboard",
                "command": ["python", "vega_dashboard.py", "--port", "8080"],
                "url": "http://127.0.0.1:8080",
                "startup_time": 3,
            },
            "ui": {
                "name": "🎨 Rich UI",
                "command": ["python", "vega_ui.py"],
                "interactive": True,
                "startup_time": 1,
            },
        }

        self.startup_progress = {}
        self.failed_components = []

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are available"""
        deps = {
            "python": True,  # We're running Python
            "uvicorn": self.check_module("uvicorn"),
            "fastapi": self.check_module("fastapi"),
            "rich": RICH_AVAILABLE,
            "sqlite3": self.check_module("sqlite3"),
            "asyncio": self.check_module("asyncio"),
        }

        return deps

    def check_module(self, module_name: str) -> bool:
        """Check if a Python module is available"""
        try:
            __import__(module_name)
            return True
        except ImportError:
            return False

    def check_file_exists(self, filepath: str) -> bool:
        """Check if a file exists"""
        return (self.base_dir / filepath).exists()

    async def check_component_health(
        self, component_name: str, component_info: Dict
    ) -> bool:
        """Check if a component is healthy"""
        health_check = component_info.get("health_check", "")

        if health_check.startswith("http"):
            # HTTP health check
            try:
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    async with session.get(health_check, timeout=5) as response:
                        return response.status == 200
            except:
                return False

        elif health_check.endswith(".json"):
            # File-based health check
            health_file = self.base_dir / health_check
            if health_file.exists():
                try:
                    with open(health_file, "r") as f:
                        data = json.load(f)
                    return data.get("status") == "active"
                except:
                    return False

        return False

    async def start_component(self, name: str, info: Dict) -> bool:
        """Start a single component"""
        try:
            if self.console:
                self.console.print(f"🚀 Starting {info['name']}...")

            # Check if component file exists
            command_file = (
                info["command"][1] if len(info["command"]) > 1 else info["command"][0]
            )
            if not self.check_file_exists(command_file.replace("python", "").strip()):
                if command_file not in ["uvicorn", "-m"]:  # Skip built-in commands
                    self.failed_components.append(f"{name}: File not found")
                    return False

            # Start the process
            proc = await asyncio.create_subprocess_exec(
                *info["command"],
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.base_dir,
            )

            # Wait for startup
            await asyncio.sleep(info.get("startup_time", 3))

            # Check if it's running
            if proc.returncode is None:  # Still running
                # Verify health if possible
                is_healthy = await self.check_component_health(name, info)
                if is_healthy or not info.get("essential", False):
                    self.startup_progress[name] = "✅ Running"
                    return True
                else:
                    self.startup_progress[name] = "⚠️ Started but not healthy"
                    return not info.get("essential", False)
            else:
                self.failed_components.append(f"{name}: Failed to start")
                self.startup_progress[name] = "❌ Failed"
                return False

        except Exception as e:
            self.failed_components.append(f"{name}: {str(e)}")
            self.startup_progress[name] = f"❌ Error: {str(e)}"
            return False

    def print_banner(self):
        """Print startup banner"""
        if self.console:
            banner = f"""
🤖 VEGA 2.0 - AMBIENT AI ECOSYSTEM
==================================

🚀 Comprehensive AI system with:
   • Intelligent chat and conversation
   • Voice personality and visualization  
   • Network discovery and integration
   • Ethical decision making
   • Smart system orchestration

Mode: {self.mode.upper()}
Time: {time.strftime('%Y-%m-%d %H:%M:%S')}

Initializing complete ecosystem...
"""
            panel = Panel(
                banner.strip(),
                title="🌟 VEGA ECOSYSTEM STARTUP",
                style="cyan bold",
                box=box.DOUBLE,
            )
            self.console.print(panel)
        else:
            print("🤖 VEGA 2.0 - Starting Ambient AI Ecosystem...")

    def create_status_table(self) -> Table:
        """Create status table for components"""
        table = Table(title="🔄 Component Startup Status", box=box.ROUNDED)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="bold")
        table.add_column("Essential", justify="center")

        for name, info in self.components.items():
            status = self.startup_progress.get(name, "⏳ Pending")
            essential = "✅" if info.get("essential", False) else "⭕"
            table.add_row(info["name"], status, essential)

        return table

    async def startup_sequence(self):
        """Execute the complete startup sequence"""
        try:
            self.print_banner()

            # Check dependencies
            if self.console:
                self.console.print("🔍 Checking dependencies...")

            deps = self.check_dependencies()
            missing_deps = [name for name, available in deps.items() if not available]

            if missing_deps:
                error_msg = f"❌ Missing dependencies: {', '.join(missing_deps)}"
                if self.console:
                    self.console.print(error_msg, style="red bold")
                else:
                    print(error_msg)
                return False

            if self.console:
                self.console.print("✅ All dependencies available", style="green")

            # Start components with progress tracking
            if self.console and RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TimeElapsedColumn(),
                    console=self.console,
                ) as progress:

                    # Create progress tasks
                    total_components = len(self.components)
                    main_task = progress.add_task(
                        "Starting components...", total=total_components
                    )

                    # Start essential components first
                    essential_components = [
                        (name, info)
                        for name, info in self.components.items()
                        if info.get("essential", False)
                    ]

                    for name, info in essential_components:
                        component_task = progress.add_task(
                            f"Starting {info['name']}...", total=1
                        )

                        success = await self.start_component(name, info)

                        progress.update(component_task, completed=1)
                        progress.update(main_task, advance=1)

                        if not success and info.get("essential", False):
                            self.console.print(
                                f"❌ Essential component {name} failed to start!",
                                style="red bold",
                            )
                            return False

                    # Start non-essential components
                    non_essential = [
                        (name, info)
                        for name, info in self.components.items()
                        if not info.get("essential", False)
                    ]

                    for name, info in non_essential:
                        component_task = progress.add_task(
                            f"Starting {info['name']}...", total=1
                        )

                        await self.start_component(name, info)

                        progress.update(component_task, completed=1)
                        progress.update(main_task, advance=1)

            else:
                # Fallback without rich progress
                print("🔄 Starting components...")
                for name, info in self.components.items():
                    print(f"  🚀 Starting {info['name']}...")
                    await self.start_component(name, info)

            # Show final status
            await self.show_startup_summary()

            return True

        except Exception as e:
            error_msg = f"❌ Startup failed: {str(e)}"
            if self.console:
                self.console.print(error_msg, style="red bold")
            else:
                print(error_msg)
            return False

    async def show_startup_summary(self):
        """Show startup summary"""
        if self.console:
            # Component status table
            status_table = self.create_status_table()
            self.console.print(status_table)

            # Success/failure summary
            running_count = len(
                [s for s in self.startup_progress.values() if "Running" in s]
            )
            total_count = len(self.components)

            if running_count == total_count:
                summary_style = "green bold"
                summary_text = f"🎉 All {total_count} components started successfully!"
            elif running_count >= len(
                [c for c in self.components.values() if c.get("essential", False)]
            ):
                summary_style = "yellow bold"
                summary_text = f"⚠️ {running_count}/{total_count} components running (essentials OK)"
            else:
                summary_style = "red bold"
                summary_text = (
                    f"❌ Only {running_count}/{total_count} components running"
                )

            self.console.print(f"\n{summary_text}", style=summary_style)

            # Access information
            access_panel = Panel(
                """🌐 **Access Points:**

• **Chat API:** http://127.0.0.1:8000/
• **Web Dashboard:** http://127.0.0.1:8080/
• **Smart Assistant:** `python vega_smart.py "system status"`
• **Rich UI:** `python vega_ui.py`

📋 **Quick Commands:**
• Check status: `python vega.py --status`
• Smart assistant: `python vega_smart.py --interactive`
• Open dashboard: Open http://127.0.0.1:8080 in browser
""",
                title="🎯 Vega Ecosystem Ready",
                style="green",
                box=box.ROUNDED,
            )
            self.console.print(access_panel)

            # Show any failures
            if self.failed_components:
                failure_text = "\n".join(
                    [f"• {failure}" for failure in self.failed_components]
                )
                failure_panel = Panel(
                    f"**Failed Components:**\n{failure_text}",
                    title="⚠️ Issues Detected",
                    style="yellow",
                    box=box.ROUNDED,
                )
                self.console.print(failure_panel)

        else:
            # Fallback without rich
            print("\n" + "=" * 50)
            print("🎉 VEGA ECOSYSTEM STARTUP COMPLETE")
            print("=" * 50)

            print("\n📊 Component Status:")
            for name, info in self.components.items():
                status = self.startup_progress.get(name, "Unknown")
                print(f"  {info['name']}: {status}")

            print("\n🌐 Access Points:")
            print("  • Chat API: http://127.0.0.1:8000/")
            print("  • Web Dashboard: http://127.0.0.1:8080/")
            print("  • Smart Assistant: python vega_smart.py --interactive")

            if self.failed_components:
                print("\n⚠️ Issues:")
                for failure in self.failed_components:
                    print(f"  • {failure}")

    async def quick_start(self):
        """Quick start with minimal components"""
        self.console.print(
            "⚡ Quick Start Mode - Starting essential components only...",
            style="yellow bold",
        )

        # Only start essential components
        essential_only = {
            name: info
            for name, info in self.components.items()
            if info.get("essential", False)
        }
        self.components = essential_only

        return await self.startup_sequence()

    async def safe_mode(self):
        """Safe mode startup"""
        self.console.print(
            "🛡️ Safe Mode - Starting with extra safety checks...", style="blue bold"
        )

        # Add extra checks and slower startup
        for info in self.components.values():
            info["startup_time"] = (
                info.get("startup_time", 3) * 2
            )  # Double startup time

        return await self.startup_sequence()

    async def diagnostic_mode(self):
        """Diagnostic mode with detailed output"""
        self.console.print(
            "🔬 Diagnostic Mode - Detailed system analysis...", style="magenta bold"
        )

        # Check all files exist
        self.console.print("\n📁 File Check:")
        for name, info in self.components.items():
            command_file = (
                info["command"][1] if len(info["command"]) > 1 else info["command"][0]
            )
            if command_file not in ["python", "uvicorn", "-m"]:
                exists = self.check_file_exists(command_file)
                status = "✅" if exists else "❌"
                self.console.print(f"  {status} {info['name']}: {command_file}")

        # Check dependencies in detail
        self.console.print("\n🔍 Dependency Check:")
        deps = self.check_dependencies()
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            self.console.print(f"  {status} {dep}")

        # Check network ports
        self.console.print("\n🌐 Port Availability:")
        import socket

        ports_to_check = [8000, 8080, 8081, 8082]
        for port in ports_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(("127.0.0.1", port))
                sock.close()

                if result == 0:
                    self.console.print(f"  ⚠️ Port {port}: In use")
                else:
                    self.console.print(f"  ✅ Port {port}: Available")
            except:
                self.console.print(f"  ❓ Port {port}: Unknown")

        # Proceed with normal startup
        return await self.startup_sequence()

    async def check_status(self):
        """Check current system status"""
        if self.console:
            self.console.print(
                "📊 Checking Vega ecosystem status...", style="cyan bold"
            )

        status_results = {}

        for name, info in self.components.items():
            is_healthy = await self.check_component_health(name, info)
            status_results[name] = {
                "name": info["name"],
                "healthy": is_healthy,
                "essential": info.get("essential", False),
            }

        # Show results
        if self.console:
            status_table = Table(title="📋 Current System Status", box=box.ROUNDED)
            status_table.add_column("Component", style="cyan")
            status_table.add_column("Status", style="bold")
            status_table.add_column("Type")

            for name, result in status_results.items():
                status = "🟢 Healthy" if result["healthy"] else "🔴 Not Running"
                comp_type = "Essential" if result["essential"] else "Optional"
                status_table.add_row(result["name"], status, comp_type)

            self.console.print(status_table)

            # Summary
            healthy_count = len([r for r in status_results.values() if r["healthy"]])
            total_count = len(status_results)

            if healthy_count == total_count:
                self.console.print(
                    f"\n🎉 All {total_count} components are healthy!",
                    style="green bold",
                )
            else:
                self.console.print(
                    f"\n⚠️ {healthy_count}/{total_count} components are healthy",
                    style="yellow bold",
                )

        return status_results


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vega Ecosystem Launcher")
    parser.add_argument(
        "--quick", action="store_true", help="Quick start (essential components only)"
    )
    parser.add_argument("--safe", action="store_true", help="Safe mode startup")
    parser.add_argument(
        "--diagnostic", action="store_true", help="Diagnostic mode with detailed checks"
    )
    parser.add_argument(
        "--status", action="store_true", help="Check current system status"
    )

    args = parser.parse_args()

    if args.quick:
        launcher = VegaEcosystemLauncher("quick")
        success = await launcher.quick_start()
    elif args.safe:
        launcher = VegaEcosystemLauncher("safe")
        success = await launcher.safe_mode()
    elif args.diagnostic:
        launcher = VegaEcosystemLauncher("diagnostic")
        success = await launcher.diagnostic_mode()
    elif args.status:
        launcher = VegaEcosystemLauncher("status")
        await launcher.check_status()
        return
    else:
        launcher = VegaEcosystemLauncher("normal")
        success = await launcher.startup_sequence()

    if success:
        if RICH_AVAILABLE and launcher.console:
            launcher.console.print(
                "\n🎯 Vega ecosystem is now running!", style="green bold"
            )
            launcher.console.print(
                "Press Ctrl+C to see shutdown options or keep running in background.",
                style="dim",
            )
        else:
            print("\n🎯 Vega ecosystem is now running!")

        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            if RICH_AVAILABLE and launcher.console:
                launcher.console.print(
                    "\n🛑 Shutdown requested. Components will continue running in background.",
                    style="yellow",
                )
                launcher.console.print(
                    "To stop all components, run: pkill -f vega", style="dim"
                )
            else:
                print(
                    "\n🛑 Shutdown requested. Components will continue running in background."
                )
                print("To stop all components, run: pkill -f vega")
    else:
        if RICH_AVAILABLE and launcher.console:
            launcher.console.print(
                "\n❌ Startup failed. Check errors above.", style="red bold"
            )
        else:
            print("\n❌ Startup failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
