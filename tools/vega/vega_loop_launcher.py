#!/usr/bin/env python3
"""
VEGA LOOP LAUNCHER
==================

Lightweight launcher for vega_loop.py that handles environment setup
and launches the main loop with proper error handling.

This avoids import issues by running vega_loop as a subprocess with
the correct Python path and environment variables.
"""

import os
import sys
import subprocess
import signal
from pathlib import Path
from typing import Optional


def get_vega_root() -> Path:
    """Get the Vega project root directory"""
    return Path(__file__).parent.parent.parent


def get_venv_python() -> str:
    """Get the path to the virtual environment Python"""
    vega_root = get_vega_root()
    venv_python = vega_root / ".venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)

    # Fallback to system python
    return sys.executable


def launch_vega_loop(action: str) -> tuple[int, str, str]:
    """
    Launch Vega server with the specified action

    Args:
        action: One of 'start', 'stop', 'status', 'force-prompt'

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    vega_root = get_vega_root()
    python_exe = get_venv_python()

    # PID and log files
    pid_file = vega_root / "vega_server.pid"
    log_file = vega_root / "vega_server.log"

    if action == "start":
        # Check if already running
        if pid_file.exists():
            try:
                with open(pid_file, "r") as f:
                    old_pid = int(f.read().strip())
                try:
                    import psutil

                    if psutil.pid_exists(old_pid):
                        return 1, "", f"Vega server already running (PID: {old_pid})"
                except ImportError:
                    pass
            except (ValueError, IOError):
                pass

        # Start uvicorn directly in background using nohup
        try:
            import time

            cmd = [
                "nohup",
                python_exe,
                "-m",
                "uvicorn",
                "vega.core.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ]

            with open(log_file, "w") as log:
                proc = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(vega_root / "src"),
                    env={**os.environ, "PYTHONPATH": str(vega_root / "src")},
                    start_new_session=True,
                )

                # Save PID
                with open(pid_file, "w") as f:
                    f.write(str(proc.pid))

                # Give it a moment to start
                time.sleep(2)

                # Check if it's still running
                if proc.poll() is None:
                    return (
                        0,
                        f"Vega server started (PID: {proc.pid}). Logs: {log_file}",
                        "",
                    )
                else:
                    with open(log_file, "r") as f:
                        error_log = f.read()[-500:]  # Last 500 chars
                    return (
                        1,
                        "",
                        f"Server failed to start. Check {log_file}\n{error_log}",
                    )

        except Exception as e:
            return 1, "", f"Failed to start server: {e}"

    elif action == "stop":
        # Stop the server
        stopped_any = False

        # Try PID file first
        if pid_file.exists():
            try:
                with open(pid_file, "r") as f:
                    pid = int(f.read().strip())
                try:
                    import psutil

                    proc = psutil.Process(pid)
                    proc.terminate()
                    proc.wait(timeout=5)
                    pid_file.unlink()
                    return 0, f"Stopped Vega server (PID: {pid})", ""
                except (psutil.NoSuchProcess, psutil.TimeoutExpired, ImportError):
                    if pid_file.exists():
                        pid_file.unlink()
            except (ValueError, IOError):
                pass

        # Fallback: find and kill uvicorn processes
        try:
            import psutil

            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline_list = proc.info.get("cmdline")
                    if cmdline_list:
                        cmdline = " ".join(cmdline_list)
                        if "uvicorn" in cmdline and "vega.core.app:app" in cmdline:
                            proc.terminate()
                            proc.wait(timeout=5)
                            stopped_any = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            if stopped_any:
                return 0, "Stopped Vega server", ""
            else:
                return 0, "No Vega server process found", ""
        except ImportError:
            return 1, "", "psutil not available"

    elif action == "status":
        # Check if server is running
        if pid_file.exists():
            try:
                with open(pid_file, "r") as f:
                    pid = int(f.read().strip())
                try:
                    import psutil

                    if psutil.pid_exists(pid):
                        proc = psutil.Process(pid)
                        cpu = proc.cpu_percent()
                        mem = proc.memory_info().rss / 1024 / 1024  # MB
                        return (
                            0,
                            f"Vega server running (PID: {pid}, CPU: {cpu}%, Memory: {mem:.1f}MB)",
                            "",
                        )
                except ImportError:
                    return 0, f"Vega server running (PID: {pid})", ""
            except (ValueError, IOError, Exception):
                if pid_file.exists():
                    pid_file.unlink()

        # Fallback search
        try:
            import psutil

            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline_list = proc.info.get("cmdline")
                    if cmdline_list:
                        cmdline = " ".join(cmdline_list)
                        if "uvicorn" in cmdline and "vega" in cmdline.lower():
                            return (
                                0,
                                f"Vega server running (PID: {proc.info['pid']})",
                                "",
                            )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return 1, "Vega server is not running", ""
        except ImportError:
            return 1, "", "psutil not available"

    elif action == "force-prompt":
        # Trigger interaction via API
        try:
            import urllib.request
            import json

            url = "http://127.0.0.1:8000/chat"
            data = json.dumps({"prompt": "Hello Vega", "stream": False}).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return 0, "Interaction triggered", ""
        except Exception as e:
            return 1, "", f"Failed: {e}. Is server running?"
    else:
        return 1, "", f"Unknown action: {action}"


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Vega Loop Launcher")
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "force-prompt"],
        help="Action to perform",
    )

    # Support legacy flag format
    parser.add_argument("--start", action="store_true", help="Start Vega (legacy)")
    parser.add_argument("--stop", action="store_true", help="Stop Vega (legacy)")
    parser.add_argument("--status", action="store_true", help="Check status (legacy)")
    parser.add_argument(
        "--force-prompt", action="store_true", help="Force interaction (legacy)"
    )

    args = parser.parse_args()

    # Handle legacy flags
    if args.start:
        action = "start"
    elif args.stop:
        action = "stop"
    elif args.status:
        action = "status"
    elif args.force_prompt:
        action = "force-prompt"
    else:
        action = args.action

    # Execute action
    returncode, stdout, stderr = launch_vega_loop(action)

    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)

    sys.exit(returncode)


if __name__ == "__main__":
    main()
