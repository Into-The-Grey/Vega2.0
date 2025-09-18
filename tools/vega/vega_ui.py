#!/usr/bin/env python3
"""
VEGA MODERN CLI INTERFACE
========================

Beautiful, rich terminal interface for the Vega Ambient AI system.
Features real-time monitoring, interactive controls, and gorgeous visuals.

Requirements:
    pip install rich textual psutil

Usage:
    python vega_ui.py                    # Launch interactive dashboard
    python vega_ui.py --monitor          # Real-time monitoring mode
    python vega_ui.py --quick-status     # Quick status check
    python vega_ui.py --interactive      # Interactive command mode
"""

import os
import sys
import time
import json
import asyncio
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

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
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich.tree import Tree
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    from rich import box
    from rich.status import Status
    from rich.rule import Rule
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal, Vertical
    from textual.widgets import (
        Header,
        Footer,
        Button,
        Static,
        Label,
        ProgressBar,
        DataTable,
        Log,
        Input,
        Switch,
        Select,
        Checkbox,
    )
    from textual.reactive import reactive
    from textual.message import Message
    from textual.binding import Binding
except ImportError:
    print("❌ Rich/Textual not installed. Run: pip install rich textual")
    sys.exit(1)

import psutil

# Global console for rich output
console = Console()


class VegaStatus:
    """Current status of Vega system"""

    def __init__(self):
        self.state_dir = Path.cwd() / "vega_state"
        self.is_running = False
        self.mode = "unknown"
        self.uptime = "0s"
        self.last_interaction = "never"
        self.system_health = 0.0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.gpu_usage = {}
        self.user_presence = "unknown"
        self.recent_thoughts = []
        self.interaction_count = 0
        self.energy_level = 1.0
        self.silence_protocol = "standard"

    def refresh(self):
        """Refresh status from system"""
        try:
            # Check if loop is running
            self.is_running = self._check_daemon_running()

            # Load state from files
            self._load_loop_state()
            self._load_presence_state()
            self._load_personality_state()
            self._load_system_metrics()

        except Exception as e:
            console.print(f"[red]Error refreshing status: {e}[/red]")

    def _check_daemon_running(self) -> bool:
        """Check if vega_loop.py is running"""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if "python" in proc.info["name"].lower():
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    if "vega_loop.py" in cmdline:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _load_loop_state(self):
        """Load state from vega loop"""
        state_file = self.state_dir / "loop_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self.mode = data.get("vega_mode", "unknown")
                    self.uptime = data.get("uptime_hours", 0)
                    self.last_interaction = data.get("last_interaction_time", "never")
                    self.energy_level = data.get("energy_level", 1.0)
            except Exception:
                pass

    def _load_presence_state(self):
        """Load user presence state"""
        presence_file = self.state_dir / "presence_history.jsonl"
        if presence_file.exists():
            try:
                # Get last line
                with open(presence_file, "r") as f:
                    lines = f.readlines()
                    if lines:
                        data = json.loads(lines[-1])
                        self.user_presence = data.get("presence_state", "unknown")
            except Exception:
                pass

    def _load_personality_state(self):
        """Load recent personality thoughts"""
        personality_file = self.state_dir / "personality_memory.jsonl"
        if personality_file.exists():
            try:
                thoughts = []
                with open(personality_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-5:]:  # Last 5 thoughts
                        data = json.loads(line)
                        thoughts.append(
                            {
                                "content": data.get("content", "")[:80] + "...",
                                "mode": data.get("mode", "unknown"),
                                "timestamp": data.get("generated_at", "")[:19],
                            }
                        )
                self.recent_thoughts = thoughts
            except Exception:
                pass

    def _load_system_metrics(self):
        """Load current system metrics"""
        try:
            self.cpu_usage = psutil.cpu_percent()
            self.memory_usage = psutil.virtual_memory().percent

            # Try to get GPU usage (basic)
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode()
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu_usage[name] = util.gpu
            except:
                self.gpu_usage = {}

        except Exception:
            pass

    def get_health_color(self) -> str:
        """Get color for system health"""
        if not self.is_running:
            return "red"
        elif self.energy_level < 0.3:
            return "yellow"
        elif self.cpu_usage > 80:
            return "orange"
        else:
            return "green"

    def get_status_emoji(self) -> str:
        """Get emoji for current status"""
        if not self.is_running:
            return "💤"
        elif self.mode == "focused":
            return "🎯"
        elif self.mode == "active":
            return "🤖"
        elif self.mode == "silent":
            return "🤫"
        elif self.mode == "paused":
            return "⏸️"
        else:
            return "❓"


class VegaCommands:
    """Command interface for Vega system"""

    @staticmethod
    def start_system():
        """Start the Vega ambient system"""
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--start"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                console.print(
                    "✅ [green]Vega ambient system started successfully![/green]"
                )
                return True
            else:
                console.print(f"❌ [red]Failed to start: {result.stderr}[/red]")
                return False
        except Exception as e:
            console.print(f"❌ [red]Error starting system: {e}[/red]")
            return False

    @staticmethod
    def stop_system():
        """Stop the Vega ambient system"""
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--stop"],
                capture_output=True,
                text=True,
            )

            console.print("⏹️ [yellow]Vega ambient system stopped[/yellow]")
            return True
        except Exception as e:
            console.print(f"❌ [red]Error stopping system: {e}[/red]")
            return False

    @staticmethod
    def force_interaction():
        """Force an immediate interaction"""
        try:
            import subprocess

            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--force-prompt"],
                capture_output=True,
                text=True,
            )

            console.print("💬 [blue]Forced interaction triggered[/blue]")
            return True
        except Exception as e:
            console.print(f"❌ [red]Error forcing interaction: {e}[/red]")
            return False


def create_status_panel(status: VegaStatus) -> Panel:
    """Create main status panel"""

    # System status table
    status_table = Table(box=box.ROUNDED, show_header=False)
    status_table.add_column("Property", style="cyan")
    status_table.add_column("Value", style="white")

    # Status emoji and running state
    emoji = status.get_status_emoji()
    health_color = status.get_health_color()

    status_table.add_row(
        "Status",
        f"{emoji} [{health_color}]{'Running' if status.is_running else 'Stopped'}[/{health_color}]",
    )
    status_table.add_row("Mode", f"[bold]{status.mode.title()}[/bold]")
    status_table.add_row(
        "Uptime",
        (
            f"{status.uptime}h"
            if isinstance(status.uptime, (int, float))
            else status.uptime
        ),
    )
    status_table.add_row(
        "User Presence", f"[yellow]{status.user_presence.title()}[/yellow]"
    )
    status_table.add_row("Last Interaction", status.last_interaction)
    status_table.add_row(
        "Energy Level",
        f"[green]{'█' * int(status.energy_level * 10)}[/green][dim]{'░' * (10 - int(status.energy_level * 10))}[/dim] {status.energy_level:.1%}",
    )

    return Panel(
        status_table, title="🤖 Vega Ambient AI Status", border_style=health_color
    )


def create_system_panel(status: VegaStatus) -> Panel:
    """Create system metrics panel"""

    metrics_table = Table(box=box.ROUNDED, show_header=False)
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Usage", style="white")
    metrics_table.add_column("Bar", style="white")

    # CPU usage
    cpu_color = (
        "red"
        if status.cpu_usage > 80
        else "yellow" if status.cpu_usage > 60 else "green"
    )
    cpu_bar = f"[{cpu_color}]{'█' * int(status.cpu_usage / 10)}[/{cpu_color}][dim]{'░' * (10 - int(status.cpu_usage / 10))}[/dim]"
    metrics_table.add_row("CPU", f"{status.cpu_usage:.1f}%", cpu_bar)

    # Memory usage
    mem_color = (
        "red"
        if status.memory_usage > 80
        else "yellow" if status.memory_usage > 60 else "green"
    )
    mem_bar = f"[{mem_color}]{'█' * int(status.memory_usage / 10)}[/{mem_color}][dim]{'░' * (10 - int(status.memory_usage / 10))}[/dim]"
    metrics_table.add_row("Memory", f"{status.memory_usage:.1f}%", mem_bar)

    # GPU usage
    for gpu_name, gpu_util in status.gpu_usage.items():
        gpu_color = "red" if gpu_util > 80 else "yellow" if gpu_util > 60 else "green"
        gpu_bar = f"[{gpu_color}]{'█' * int(gpu_util / 10)}[/{gpu_color}][dim]{'░' * (10 - int(gpu_util / 10))}[/dim]"
        short_name = gpu_name.split()[-1] if len(gpu_name) > 15 else gpu_name
        metrics_table.add_row(f"GPU ({short_name})", f"{gpu_util}%", gpu_bar)

    return Panel(metrics_table, title="💻 System Resources", border_style="blue")


def create_thoughts_panel(status: VegaStatus) -> Panel:
    """Create recent thoughts panel"""

    if not status.recent_thoughts:
        content = Text("No recent thoughts...", style="dim")
    else:
        thoughts_table = Table(box=box.SIMPLE, show_header=True)
        thoughts_table.add_column("Time", style="dim", width=12)
        thoughts_table.add_column("Mode", style="cyan", width=12)
        thoughts_table.add_column("Thought", style="white")

        for thought in status.recent_thoughts[-3:]:  # Show last 3
            time_str = (
                thought["timestamp"].split("T")[1][:8]
                if "T" in thought["timestamp"]
                else thought["timestamp"][:8]
            )
            thoughts_table.add_row(
                time_str, thought["mode"].title(), thought["content"]
            )

    return Panel(
        thoughts_table if status.recent_thoughts else content,
        title="🧠 Recent Thoughts",
        border_style="magenta",
    )


def create_controls_panel() -> Panel:
    """Create control buttons panel"""

    controls = Text()
    controls.append("Commands:\n", style="bold white")
    controls.append("  [S] Start System    ", style="green")
    controls.append("  [X] Stop System\n", style="red")
    controls.append("  [F] Force Chat      ", style="blue")
    controls.append("  [R] Refresh Status\n", style="yellow")
    controls.append("  [C] Configuration   ", style="cyan")
    controls.append("  [L] View Logs\n", style="magenta")
    controls.append("  [Q] Quit Interface", style="dim")

    return Panel(controls, title="🎮 Controls", border_style="white")


def quick_status():
    """Show quick status and exit"""
    status = VegaStatus()
    status.refresh()

    console.print()
    console.print(create_status_panel(status))
    console.print()


def monitor_mode():
    """Real-time monitoring mode"""
    status = VegaStatus()

    def make_layout():
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        layout["left"].split_column(Layout(name="status"), Layout(name="system"))

        layout["right"].split_column(Layout(name="thoughts"), Layout(name="controls"))

        return layout

    layout = make_layout()

    with Live(layout, refresh_per_second=2, screen=True):
        while True:
            try:
                status.refresh()

                # Update header
                header_text = Text()
                header_text.append("🤖 VEGA AMBIENT AI DASHBOARD", style="bold white")
                header_text.append(
                    f" • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim"
                )
                layout["header"].update(Panel(Align.center(header_text), style="blue"))

                # Update panels
                layout["status"].update(create_status_panel(status))
                layout["system"].update(create_system_panel(status))
                layout["thoughts"].update(create_thoughts_panel(status))
                layout["controls"].update(create_controls_panel())

                # Update footer
                footer_text = Text("Press Ctrl+C to exit monitoring mode", style="dim")
                layout["footer"].update(Panel(Align.center(footer_text), style="dim"))

                time.sleep(2)

            except KeyboardInterrupt:
                console.print("\n👋 [yellow]Exiting monitoring mode...[/yellow]")
                break
            except Exception as e:
                console.print(f"\n❌ [red]Monitoring error: {e}[/red]")
                break


def interactive_mode():
    """Interactive command mode"""
    status = VegaStatus()
    commands = VegaCommands()

    console.print()
    console.print(
        Panel(
            Text(
                "🤖 Vega Interactive Command Center",
                justify="center",
                style="bold white",
            ),
            style="blue",
        )
    )
    console.print()

    while True:
        try:
            status.refresh()

            # Show current status
            console.print(create_status_panel(status))
            console.print()

            # Show menu
            console.print("[bold white]Available Commands:[/bold white]")
            console.print("  [green][S][/green] Start System")
            console.print("  [red][X][/red] Stop System")
            console.print("  [blue][F][/blue] Force Interaction")
            console.print("  [yellow][R][/yellow] Refresh Status")
            console.print("  [cyan][M][/cyan] Monitor Mode")
            console.print("  [magenta][L][/magenta] View Logs")
            console.print("  [dim][Q][/dim] Quit")
            console.print()

            choice = Prompt.ask(
                "Enter command",
                choices=["s", "x", "f", "r", "m", "l", "q"],
                default="r",
            ).lower()

            console.print()

            if choice == "s":
                with Status("Starting Vega system...", spinner="dots"):
                    commands.start_system()

            elif choice == "x":
                if Confirm.ask("Are you sure you want to stop Vega?"):
                    commands.stop_system()

            elif choice == "f":
                with Status("Triggering interaction...", spinner="dots"):
                    commands.force_interaction()

            elif choice == "r":
                with Status("Refreshing status...", spinner="dots"):
                    time.sleep(1)  # Brief pause for effect
                console.print("✅ [green]Status refreshed[/green]")

            elif choice == "m":
                console.print("🔄 [blue]Entering monitor mode...[/blue]")
                time.sleep(1)
                monitor_mode()

            elif choice == "l":
                view_logs()

            elif choice == "q":
                console.print("👋 [yellow]Goodbye![/yellow]")
                break

            console.print()

        except KeyboardInterrupt:
            console.print("\n👋 [yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"\n❌ [red]Error: {e}[/red]")


def view_logs():
    """View recent system logs"""
    console.print("\n📜 [bold]Recent Vega Activity[/bold]\n")

    state_dir = Path.cwd() / "vega_state"

    # Show presence history
    presence_file = state_dir / "presence_history.jsonl"
    if presence_file.exists():
        console.print("👤 [cyan]User Presence (Last 5 entries):[/cyan]")
        try:
            with open(presence_file, "r") as f:
                lines = f.readlines()
                for line in lines[-5:]:
                    data = json.loads(line)
                    timestamp = data.get("timestamp", "")[:19]
                    presence = data.get("presence_state", "unknown")
                    app = data.get("active_application", "unknown")
                    console.print(
                        f"  {timestamp} • [yellow]{presence}[/yellow] ({app})"
                    )
        except Exception as e:
            console.print(f"  [red]Error reading presence log: {e}[/red]")

    console.print()

    # Show recent thoughts
    thoughts_file = state_dir / "personality_memory.jsonl"
    if thoughts_file.exists():
        console.print("🧠 [magenta]Recent Thoughts (Last 3):[/magenta]")
        try:
            with open(thoughts_file, "r") as f:
                lines = f.readlines()
                for line in lines[-3:]:
                    data = json.loads(line)
                    timestamp = data.get("generated_at", "")[:19]
                    mode = data.get("mode", "unknown")
                    content = data.get("content", "")[:80] + (
                        "..." if len(data.get("content", "")) > 80 else ""
                    )
                    console.print(f"  {timestamp} • [cyan]{mode}[/cyan]: {content}")
        except Exception as e:
            console.print(f"  [red]Error reading thoughts log: {e}[/red]")

    console.print()
    input("Press Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Vega Modern CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🤖 Vega Ambient AI - Modern Terminal Interface

Examples:
  python vega_ui.py                 # Interactive dashboard
  python vega_ui.py --monitor       # Real-time monitoring
  python vega_ui.py --quick-status  # Quick status check
        """,
    )

    parser.add_argument(
        "--monitor", action="store_true", help="Real-time monitoring mode"
    )
    parser.add_argument(
        "--quick-status", action="store_true", help="Show quick status and exit"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive command mode (default)"
    )

    args = parser.parse_args()

    # Check if rich/textual are available
    try:
        from rich.console import Console
    except ImportError:
        print("❌ Rich not installed. Install with: pip install rich textual")
        return

    if args.quick_status:
        quick_status()
    elif args.monitor:
        monitor_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
