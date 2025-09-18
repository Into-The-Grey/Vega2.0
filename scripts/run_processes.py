#!/usr/bin/env python3
"""
Vega2.0 Background Process Daemon
=================================

Runs background processes for system monitoring, integrations, and voice processing.
Can be run as a standalone daemon or integrated with the main API server.
"""

import asyncio
import argparse
import logging
import os
import sys
import signal
from pathlib import Path

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.vega.core.process_manager import (
    get_process_manager,
    setup_signal_handlers,
    start_background_processes,
    stop_background_processes,
)


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("/tmp/vega_processes.log"),
        ],
    )


async def run_daemon():
    """Run the background process daemon"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Vega2.0 background process daemon...")

    # Setup signal handlers
    setup_signal_handlers()

    try:
        # Start background processes
        await start_background_processes()

        # Keep daemon running
        logger.info("Background processes started. Daemon running...")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Daemon error: {e}")
        raise
    finally:
        logger.info("Shutting down background processes...")
        await stop_background_processes()
        logger.info("Background process daemon stopped")


async def show_status():
    """Show status of all background processes"""
    manager = get_process_manager()

    if not manager.running:
        print("Background process manager is not running")
        return

    processes = manager.list_processes()
    metrics = manager.get_process_metrics()

    print(f"\nVega2.0 Background Processes Status")
    print(f"=" * 50)
    print(f"Total Processes: {metrics['total_processes']}")
    print(f"Running: {metrics['running_processes']}")
    print(f"Failed: {metrics['failed_processes']}")
    print(f"Health: {metrics['health_percentage']:.1f}%")

    print(f"\nProcess Details:")
    print(f"-" * 50)
    for process in processes:
        status_icon = "✅" if process.state.value == "running" else "❌"
        print(f"{status_icon} {process.name} ({process.type.value})")
        print(f"   State: {process.state.value}")
        print(f"   PID: {process.pid or 'N/A'}")
        print(f"   CPU: {process.cpu_usage:.1f}%")
        print(f"   Memory: {process.memory_usage:.1f}%")
        print(f"   Restarts: {process.restart_count}")
        if process.start_time:
            uptime = process.metrics.get("uptime", 0)
            print(f"   Uptime: {uptime:.0f}s")
        print()


async def stop_daemon():
    """Stop the background process daemon"""
    logger = logging.getLogger(__name__)
    logger.info("Stopping background process daemon...")

    try:
        await stop_background_processes()
        print("Background processes stopped successfully")
    except Exception as e:
        print(f"Error stopping processes: {e}")
        sys.exit(1)


async def restart_process(process_name: str):
    """Restart a specific process"""
    manager = get_process_manager()

    # Find process by name
    target_process = None
    for process in manager.processes.values():
        if process.name == process_name:
            target_process = process
            break

    if not target_process:
        print(f"Process '{process_name}' not found")
        return

    try:
        await manager.restart_process(target_process.id)
        print(f"Process '{process_name}' restarted successfully")
    except Exception as e:
        print(f"Error restarting process '{process_name}': {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vega2.0 Background Process Daemon")
    parser.add_argument(
        "command",
        choices=["start", "stop", "status", "restart"],
        help="Command to execute",
    )
    parser.add_argument("--process", help="Process name for restart command")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--foreground", action="store_true", help="Run in foreground (don't daemonize)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Execute command
    try:
        if args.command == "start":
            if args.foreground:
                # Run in foreground
                asyncio.run(run_daemon())
            else:
                # Run as daemon (simplified - use systemd in production)
                print("Starting background process daemon...")
                print("Use --foreground to run in foreground mode")
                print("For production, use systemd service")
                asyncio.run(run_daemon())

        elif args.command == "stop":
            asyncio.run(stop_daemon())

        elif args.command == "status":
            asyncio.run(show_status())

        elif args.command == "restart":
            if not args.process:
                print("--process argument required for restart command")
                sys.exit(1)
            asyncio.run(restart_process(args.process))

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
