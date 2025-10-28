"""
system_diagnostics.py - System Resource Monitoring for Vega2.0

Provides detailed system diagnostics including:
- Memory usage (RSS, VMS, available)
- CPU usage and thread count
- File descriptor usage
- Network connections
- Process information
"""

from __future__ import annotations

import os
import sys
import psutil
import threading
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def get_memory_stats() -> Dict[str, Any]:
    """Get detailed memory statistics"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()

        # System-wide memory
        virtual = psutil.virtual_memory()

        return {
            "process": {
                "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
                "percent": round(mem_percent, 2),
            },
            "system": {
                "total_mb": round(virtual.total / (1024 * 1024), 2),
                "available_mb": round(virtual.available / (1024 * 1024), 2),
                "used_mb": round(virtual.used / (1024 * 1024), 2),
                "percent": virtual.percent,
            },
            "status": "healthy" if mem_percent < 80 else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        return {"error": str(e), "status": "error"}


def get_cpu_stats() -> Dict[str, Any]:
    """Get CPU usage statistics"""
    try:
        process = psutil.Process()

        # Get CPU percent (non-blocking)
        cpu_percent = process.cpu_percent(interval=0.1)

        # System-wide CPU
        system_cpu = psutil.cpu_percent(interval=0.1, percpu=False)
        cpu_count = psutil.cpu_count()

        return {
            "process": {
                "percent": round(cpu_percent, 2),
                "num_threads": process.num_threads(),
            },
            "system": {
                "percent": round(system_cpu, 2),
                "cpu_count": cpu_count,
                "load_average": (
                    list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None
                ),
            },
            "status": "healthy" if cpu_percent < 80 else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get CPU stats: {e}")
        return {"error": str(e), "status": "error"}


def get_thread_stats() -> Dict[str, Any]:
    """Get detailed thread information"""
    try:
        process = psutil.Process()
        threads = process.threads()

        # Python threading info
        py_threads = threading.enumerate()

        return {
            "total_threads": len(threads),
            "python_threads": len(py_threads),
            "thread_names": [t.name for t in py_threads],
            "status": "healthy" if len(threads) < 100 else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get thread stats: {e}")
        return {"error": str(e), "status": "error"}


def get_file_descriptor_stats() -> Dict[str, Any]:
    """Get file descriptor usage (Unix-like systems only)"""
    try:
        if sys.platform == "win32":
            return {"platform": "windows", "status": "not_applicable"}

        process = psutil.Process()
        num_fds = process.num_fds()

        # Get system limits
        import resource

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)

        return {
            "open_fds": num_fds,
            "soft_limit": soft_limit,
            "hard_limit": hard_limit,
            "percent_used": round((num_fds / soft_limit) * 100, 2),
            "status": "healthy" if num_fds < (soft_limit * 0.8) else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get file descriptor stats: {e}")
        return {"error": str(e), "status": "error"}


def get_network_stats() -> Dict[str, Any]:
    """Get network connection statistics"""
    try:
        process = psutil.Process()
        connections = process.connections()

        # Count by status
        status_counts = {}
        for conn in connections:
            status = conn.status
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_connections": len(connections),
            "by_status": status_counts,
            "status": "healthy" if len(connections) < 1000 else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get network stats: {e}")
        return {"error": str(e), "status": "error"}


def get_process_info() -> Dict[str, Any]:
    """Get basic process information"""
    try:
        process = psutil.Process()

        return {
            "pid": process.pid,
            "name": process.name(),
            "username": process.username(),
            "create_time": datetime.fromtimestamp(process.create_time()).isoformat(),
            "cwd": process.cwd(),
            "exe": process.exe(),
            "python_version": sys.version,
            "platform": sys.platform,
        }
    except Exception as e:
        logger.error(f"Failed to get process info: {e}")
        return {"error": str(e)}


def get_disk_stats() -> Dict[str, Any]:
    """Get disk usage statistics"""
    try:
        # Get current working directory's disk usage
        usage = psutil.disk_usage(os.getcwd())

        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "free_gb": round(usage.free / (1024**3), 2),
            "percent": usage.percent,
            "status": "healthy" if usage.percent < 80 else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get disk stats: {e}")
        return {"error": str(e), "status": "error"}


async def get_async_event_loop_stats() -> Dict[str, Any]:
    """Get asyncio event loop statistics"""
    try:
        loop = asyncio.get_running_loop()

        # Get all tasks
        tasks = asyncio.all_tasks(loop)

        return {
            "total_tasks": len(tasks),
            "running": loop.is_running(),
            "closed": loop.is_closed(),
            "status": "healthy" if len(tasks) < 100 else "warning",
        }
    except Exception as e:
        logger.error(f"Failed to get event loop stats: {e}")
        return {"error": str(e), "status": "error"}


async def get_full_diagnostics() -> Dict[str, Any]:
    """Get comprehensive system diagnostics"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "memory": get_memory_stats(),
        "cpu": get_cpu_stats(),
        "threads": get_thread_stats(),
        "file_descriptors": get_file_descriptor_stats(),
        "network": get_network_stats(),
        "disk": get_disk_stats(),
        "event_loop": await get_async_event_loop_stats(),
        "process": get_process_info(),
    }


def get_health_summary() -> str:
    """Get overall system health status"""
    try:
        mem = get_memory_stats()
        cpu = get_cpu_stats()
        threads = get_thread_stats()
        fds = get_file_descriptor_stats()
        network = get_network_stats()
        disk = get_disk_stats()

        statuses = [
            mem.get("status", "unknown"),
            cpu.get("status", "unknown"),
            threads.get("status", "unknown"),
            (
                fds.get("status", "unknown")
                if fds.get("status") != "not_applicable"
                else "healthy"
            ),
            network.get("status", "unknown"),
            disk.get("status", "unknown"),
        ]

        if any(s == "error" for s in statuses):
            return "error"
        elif any(s == "warning" for s in statuses):
            return "warning"
        elif all(s == "healthy" for s in statuses):
            return "healthy"
        else:
            return "unknown"
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        return "error"
