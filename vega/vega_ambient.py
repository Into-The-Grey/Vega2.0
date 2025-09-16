#!/usr/bin/env python3
"""
VEGA AMBIENT PRESENCE STARTUP SCRIPT
===================================

Quick startup and management script for the complete Vega ambient AI system.
This script initializes all components and provides easy management commands.
"""

import os
import sys
import asyncio
import argparse
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ["psutil", "pynvml", "httpx", "pynput", "schedule"]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("✅ All required packages are installed")
    return True


def check_optional_tools():
    """Check optional calendar integration tools"""
    tools = ["calcurse", "khal", "xprintidle"]
    available = []

    for tool in tools:
        try:
            result = subprocess.run(["which", tool], capture_output=True, text=True)
            if result.returncode == 0:
                available.append(tool)
        except:
            pass

    if available:
        print(f"✅ Optional tools available: {', '.join(available)}")
    else:
        print("ℹ️  Optional tools (for enhanced features):")
        print("   Linux: sudo apt install calcurse khal xprintidle")
        print("   macOS: brew install calcurse khal")


def create_state_directory():
    """Create state directory if it doesn't exist"""
    state_dir = Path.cwd() / "vega_state"
    state_dir.mkdir(exist_ok=True)
    print(f"✅ State directory ready: {state_dir}")
    return state_dir


def start_ambient_system():
    """Start the complete ambient system"""
    print("🚀 Starting Vega Ambient Presence System...")

    # Check dependencies first
    if not check_dependencies():
        return False

    check_optional_tools()
    state_dir = create_state_directory()

    # Start the main ambient loop
    try:
        print("\n🧠 Starting ambient consciousness daemon...")
        result = subprocess.run(
            [sys.executable, "vega_loop.py", "--start"], cwd=Path.cwd()
        )

        if result.returncode == 0:
            print("✅ Vega ambient system started successfully!")
            print("\nSystem is now running with:")
            print("  • 24/7 system monitoring")
            print("  • User presence detection")
            print("  • Contextual personality engine")
            print("  • Smart silence protocols")
            print("  • Interaction history learning")
            print("  • Spontaneous thought generation")
            print("\nTo check status: python vega_ambient.py --status")
            print("To stop: python vega_ambient.py --stop")
            return True
        else:
            print("❌ Failed to start ambient system")
            return False

    except FileNotFoundError:
        print("❌ vega_loop.py not found. Make sure you're in the Vega2.0 directory.")
        return False
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return False


def stop_ambient_system():
    """Stop the ambient system"""
    print("⏹️  Stopping Vega Ambient Presence System...")

    try:
        result = subprocess.run(
            [sys.executable, "vega_loop.py", "--stop"], cwd=Path.cwd()
        )

        if result.returncode == 0:
            print("✅ Vega ambient system stopped successfully!")
        else:
            print("⚠️  System may not have been running")

    except Exception as e:
        print(f"❌ Error stopping system: {e}")


def check_system_status():
    """Check the status of the ambient system"""
    print("📊 Checking Vega Ambient System Status...")

    try:
        result = subprocess.run(
            [sys.executable, "vega_loop.py", "--status"], cwd=Path.cwd()
        )

        if result.returncode == 0:
            print("\n📈 Additional system info:")

            # Check state files
            state_dir = Path.cwd() / "vega_state"
            if state_dir.exists():
                state_files = list(state_dir.glob("*"))
                print(f"  • State files: {len(state_files)} files")

                # Check recent activity
                presence_log = state_dir / "presence_history.jsonl"
                if presence_log.exists():
                    lines = sum(1 for _ in open(presence_log))
                    print(f"  • Presence records: {lines} entries")

                interaction_db = state_dir / "interaction_history.db"
                if interaction_db.exists():
                    print(
                        f"  • Interaction database: {interaction_db.stat().st_size // 1024}KB"
                    )

            print("\n💡 Use --logs to view recent activity")

    except Exception as e:
        print(f"❌ Error checking status: {e}")


def view_logs():
    """View recent system logs"""
    print("📜 Recent Vega Ambient System Activity:")

    state_dir = Path.cwd() / "vega_state"

    # Show recent presence activity
    presence_log = state_dir / "presence_history.jsonl"
    if presence_log.exists():
        print("\n👤 Recent presence detection:")
        try:
            result = subprocess.run(
                ["tail", "-5", str(presence_log)], capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        try:
                            import json

                            data = json.loads(line)
                            timestamp = data.get("timestamp", "unknown")[:19]
                            presence = data.get("presence_state", "unknown")
                            app = data.get("active_application", "unknown")
                            print(f"  {timestamp}: {presence} ({app})")
                        except:
                            print(f"  {line[:100]}...")
        except:
            print("  Unable to read presence log")

    # Show recent personality thoughts
    personality_log = state_dir / "personality_memory.jsonl"
    if personality_log.exists():
        print("\n🧠 Recent thoughts:")
        try:
            result = subprocess.run(
                ["tail", "-3", str(personality_log)], capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        try:
                            import json

                            data = json.loads(line)
                            content = data.get("content", "unknown")[:80]
                            mode = data.get("mode", "unknown")
                            print(f"  {mode}: {content}...")
                        except:
                            print(f"  {line[:80]}...")
        except:
            print("  Unable to read personality log")


def test_interaction():
    """Test the interaction system"""
    print("🧪 Testing Vega interaction system...")

    try:
        result = subprocess.run(
            [sys.executable, "vega_loop.py", "--force-prompt"], cwd=Path.cwd()
        )

        if result.returncode == 0:
            print("✅ Interaction test completed")
        else:
            print("⚠️  Interaction test had issues")

    except Exception as e:
        print(f"❌ Error testing interaction: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Vega Ambient Presence System Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vega_ambient.py --start     # Start the ambient system
  python vega_ambient.py --status    # Check system status
  python vega_ambient.py --logs      # View recent activity
  python vega_ambient.py --test      # Test interaction system
  python vega_ambient.py --stop      # Stop the system

The ambient system provides:
  • 24/7 contextual awareness
  • Smart conversation timing
  • User presence detection
  • Personality learning
  • Respectful interaction protocols
        """,
    )

    parser.add_argument(
        "--start", action="store_true", help="Start the ambient presence system"
    )
    parser.add_argument(
        "--stop", action="store_true", help="Stop the ambient presence system"
    )
    parser.add_argument("--status", action="store_true", help="Check system status")
    parser.add_argument("--logs", action="store_true", help="View recent system logs")
    parser.add_argument(
        "--test", action="store_true", help="Test the interaction system"
    )
    parser.add_argument("--deps", action="store_true", help="Check dependencies only")

    args = parser.parse_args()

    if args.deps:
        check_dependencies()
        check_optional_tools()
    elif args.start:
        start_ambient_system()
    elif args.stop:
        stop_ambient_system()
    elif args.status:
        check_system_status()
    elif args.logs:
        view_logs()
    elif args.test:
        test_interaction()
    else:
        parser.print_help()
        print("\n🤖 Vega Ambient Presence System")
        print("A JARVIS-like AI companion with 24/7 contextual awareness")
        print("\nQuick start: python vega_ambient.py --start")


if __name__ == "__main__":
    main()
