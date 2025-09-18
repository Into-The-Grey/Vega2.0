#!/usr/bin/env python3
"""
VEGA LAUNCHER
=============

Unified launcher for the Vega Ambient AI system with modern UX/UI.
Provides easy access to all interfaces and tools.

Usage:
    python vega_launcher.py                    # Interactive menu
    python vega_launcher.py --cli              # Modern CLI interface
    python vega_launcher.py --dashboard        # Web dashboard
    python vega_launcher.py --chat             # Web chat interface
    python vega_launcher.py --start            # Start ambient system
    python vega_launcher.py --stop             # Stop ambient system
    python vega_launcher.py --status           # Quick status check
"""

import os
import sys
import time
import subprocess
import webbrowser
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.columns import Columns
    from rich.align import Align
    from rich.table import Table
    from rich import box

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class VegaLauncher:
    """Main launcher for Vega system"""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.base_dir = Path(__file__).parent
        self.venv_dir = self.base_dir / "vega_ui_env"
        self.python_exe = self.venv_dir / "bin" / "python"

    def print(self, text: str, style: Optional[str] = None):
        """Rich print with fallback"""
        if self.console:
            self.console.print(text, style=style)
        else:
            print(text)

    def check_setup(self) -> bool:
        """Check if UI environment is set up"""
        if not self.venv_dir.exists():
            self.print("❌ UI environment not found. Setting up...", style="yellow")
            return self.setup_environment()

        if not self.python_exe.exists():
            self.print("❌ Python executable not found in UI environment", style="red")
            return False

        return True

    def setup_environment(self) -> bool:
        """Set up the UI environment"""
        try:
            self.print("🔧 Creating virtual environment...", style="blue")
            subprocess.run(
                [sys.executable, "-m", "venv", str(self.venv_dir)],
                check=True,
                capture_output=True,
            )

            self.print("📦 Installing UI dependencies...", style="blue")
            subprocess.run(
                [
                    str(self.python_exe),
                    "-m",
                    "pip",
                    "install",
                    "rich",
                    "textual",
                    "psutil",
                    "fastapi",
                    "uvicorn",
                    "websockets",
                    "jinja2",
                    "python-multipart",
                    "pynvml",
                ],
                check=True,
                capture_output=True,
            )

            self.print("✅ UI environment set up successfully!", style="green")
            return True

        except subprocess.CalledProcessError as e:
            self.print(f"❌ Failed to set up environment: {e}", style="red")
            return False
        except Exception as e:
            self.print(f"❌ Unexpected error: {e}", style="red")
            return False

    def show_banner(self):
        """Show the main banner"""
        if self.console:
            banner_text = Text()
            banner_text.append("🤖 VEGA AMBIENT AI\n", style="bold blue")
            banner_text.append("Modern Interface Launcher", style="cyan")

            banner = Panel(Align.center(banner_text), style="blue", box=box.ROUNDED)
            self.console.print(banner)
        else:
            print("🤖 VEGA AMBIENT AI - Modern Interface Launcher")

        print()

    def show_menu(self):
        """Show interactive menu"""
        if self.console:
            menu_table = Table(box=box.ROUNDED, show_header=False)
            menu_table.add_column("Option", style="cyan", width=4)
            menu_table.add_column("Description", style="white")
            menu_table.add_column("Details", style="dim")

            menu_table.add_row(
                "1",
                "🖥️  Modern CLI Interface",
                "Rich terminal with real-time monitoring",
            )
            menu_table.add_row(
                "2", "📊 Web Dashboard", "Beautiful web interface (port 8080)"
            )
            menu_table.add_row(
                "3", "💬 Web Chat Interface", "Mobile-friendly chat (port 8080/chat)"
            )
            menu_table.add_row(
                "4", "▶️  Start Ambient System", "Launch the Vega ambient AI daemon"
            )
            menu_table.add_row(
                "5", "⏹️  Stop Ambient System", "Stop the ambient AI daemon"
            )
            menu_table.add_row(
                "6", "📊 Quick Status Check", "View current system status"
            )
            menu_table.add_row("7", "🔧 Setup Environment", "Reinstall UI dependencies")
            menu_table.add_row("8", "❌ Exit", "Close launcher")

            menu_panel = Panel(menu_table, title="🎮 Available Options", style="white")
            self.console.print(menu_panel)
        else:
            print("Available Options:")
            print("1. 🖥️  Modern CLI Interface")
            print("2. 📊 Web Dashboard")
            print("3. 💬 Web Chat Interface")
            print("4. ▶️  Start Ambient System")
            print("5. ⏹️  Stop Ambient System")
            print("6. 📊 Quick Status Check")
            print("7. 🔧 Setup Environment")
            print("8. ❌ Exit")

        print()

    def launch_cli(self):
        """Launch modern CLI interface"""
        if not self.check_setup():
            return

        self.print("🚀 Launching Modern CLI Interface...", style="blue")
        try:
            subprocess.run(
                [str(self.python_exe), "vega_ui.py", "--interactive"], cwd=self.base_dir
            )
        except KeyboardInterrupt:
            self.print("\n👋 CLI interface closed", style="yellow")
        except Exception as e:
            self.print(f"❌ Error launching CLI: {e}", style="red")

    def launch_dashboard(self):
        """Launch web dashboard"""
        if not self.check_setup():
            return

        self.print("🚀 Starting Web Dashboard on http://127.0.0.1:8080", style="blue")

        try:
            # Start dashboard in background
            proc = subprocess.Popen(
                [str(self.python_exe), "vega_dashboard.py", "--port", "8080"],
                cwd=self.base_dir,
            )

            # Wait a moment for server to start
            time.sleep(3)

            # Open browser
            self.print("🌐 Opening dashboard in browser...", style="green")
            webbrowser.open("http://127.0.0.1:8080")

            if self.console:
                self.print(
                    "\n📝 Dashboard is running! Press Ctrl+C to stop.", style="yellow"
                )
            else:
                print("\nDashboard is running! Press Ctrl+C to stop.")

            # Wait for user to stop
            try:
                proc.wait()
            except KeyboardInterrupt:
                self.print("\n🛑 Stopping dashboard...", style="yellow")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self.print("✅ Dashboard stopped", style="green")

        except Exception as e:
            self.print(f"❌ Error launching dashboard: {e}", style="red")

    def launch_chat(self):
        """Launch web chat interface"""
        if not self.check_setup():
            return

        self.print(
            "🚀 Starting Web Chat Interface on http://127.0.0.1:8080", style="blue"
        )

        try:
            # Start dashboard in background (includes chat)
            proc = subprocess.Popen(
                [str(self.python_exe), "vega_dashboard.py", "--port", "8080"],
                cwd=self.base_dir,
            )

            # Wait a moment for server to start
            time.sleep(3)

            # Open chat interface
            self.print("💬 Opening chat interface in browser...", style="green")
            webbrowser.open("http://127.0.0.1:8080/static/chat.html")

            if self.console:
                self.print(
                    "\n📝 Chat interface is running! Press Ctrl+C to stop.",
                    style="yellow",
                )
            else:
                print("\nChat interface is running! Press Ctrl+C to stop.")

            # Wait for user to stop
            try:
                proc.wait()
            except KeyboardInterrupt:
                self.print("\n🛑 Stopping chat interface...", style="yellow")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                self.print("✅ Chat interface stopped", style="green")

        except Exception as e:
            self.print(f"❌ Error launching chat interface: {e}", style="red")

    def start_ambient_system(self):
        """Start the ambient AI system"""
        self.print("▶️ Starting Vega Ambient System...", style="green")

        try:
            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--start"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                self.print(
                    "✅ Vega ambient system started successfully!", style="green"
                )
            else:
                self.print(f"❌ Failed to start system: {result.stderr}", style="red")

        except subprocess.TimeoutExpired:
            self.print("⏱️ Start command timed out", style="yellow")
        except Exception as e:
            self.print(f"❌ Error starting system: {e}", style="red")

    def stop_ambient_system(self):
        """Stop the ambient AI system"""
        self.print("⏹️ Stopping Vega Ambient System...", style="yellow")

        try:
            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--stop"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )

            self.print("✅ Vega ambient system stopped", style="green")

        except Exception as e:
            self.print(f"❌ Error stopping system: {e}", style="red")

    def quick_status(self):
        """Show quick status"""
        if not self.check_setup():
            return

        self.print("📊 Checking Vega Status...", style="blue")

        try:
            subprocess.run(
                [str(self.python_exe), "vega_ui.py", "--quick-status"],
                cwd=self.base_dir,
            )
        except Exception as e:
            self.print(f"❌ Error checking status: {e}", style="red")

    def interactive_menu(self):
        """Run interactive menu"""
        while True:
            self.show_banner()
            self.show_menu()

            if self.console:
                choice = Prompt.ask(
                    "Select an option",
                    choices=["1", "2", "3", "4", "5", "6", "7", "8"],
                    default="1",
                )
            else:
                choice = input("Select an option (1-8): ").strip()

            print()

            if choice == "1":
                self.launch_cli()
            elif choice == "2":
                self.launch_dashboard()
            elif choice == "3":
                self.launch_chat()
            elif choice == "4":
                self.start_ambient_system()
            elif choice == "5":
                self.stop_ambient_system()
            elif choice == "6":
                self.quick_status()
            elif choice == "7":
                if self.console:
                    if Confirm.ask("Reinstall UI environment?"):
                        self.setup_environment()
                else:
                    confirm = input("Reinstall UI environment? (y/N): ").lower()
                    if confirm in ["y", "yes"]:
                        self.setup_environment()
            elif choice == "8":
                self.print("👋 Goodbye!", style="yellow")
                break
            else:
                self.print("❌ Invalid option. Please try again.", style="red")

            if choice != "8":
                print()
                input("Press Enter to continue...")
                print()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Vega Ambient AI Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🤖 Vega Ambient AI - Modern Interface Launcher

Examples:
  python vega_launcher.py                # Interactive menu
  python vega_launcher.py --cli          # Modern CLI interface
  python vega_launcher.py --dashboard    # Web dashboard
  python vega_launcher.py --chat         # Web chat interface
  python vega_launcher.py --start        # Start ambient system
  python vega_launcher.py --stop         # Stop ambient system
  python vega_launcher.py --status       # Quick status check
        """,
    )

    parser.add_argument(
        "--cli", action="store_true", help="Launch modern CLI interface"
    )
    parser.add_argument("--dashboard", action="store_true", help="Launch web dashboard")
    parser.add_argument("--chat", action="store_true", help="Launch web chat interface")
    parser.add_argument("--start", action="store_true", help="Start ambient AI system")
    parser.add_argument("--stop", action="store_true", help="Stop ambient AI system")
    parser.add_argument("--status", action="store_true", help="Quick status check")
    parser.add_argument("--setup", action="store_true", help="Set up UI environment")

    args = parser.parse_args()

    launcher = VegaLauncher()

    # Handle direct commands
    if args.cli:
        launcher.launch_cli()
    elif args.dashboard:
        launcher.launch_dashboard()
    elif args.chat:
        launcher.launch_chat()
    elif args.start:
        launcher.start_ambient_system()
    elif args.stop:
        launcher.stop_ambient_system()
    elif args.status:
        launcher.quick_status()
    elif args.setup:
        launcher.setup_environment()
    else:
        # Show interactive menu
        launcher.interactive_menu()


if __name__ == "__main__":
    main()
