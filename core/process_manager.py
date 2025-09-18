"""
Background Process Manager for Vega2.0
=====================================

Manages background processes for system functions, integrations, and voice processing.
Provides lifecycle management, monitoring, and graceful shutdown capabilities.

Features:
- Process lifecycle management (start, stop, restart)
- Health monitoring and auto-restart
- Resource usage tracking
- Graceful shutdown with cleanup
- Process communication via queues
- Logging and metrics collection
"""

import asyncio
import logging
import signal
import threading
import time
import queue
import psutil
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Coroutine
import uuid
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """Process state enumeration"""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    RESTARTING = "restarting"


class ProcessType(Enum):
    """Process type enumeration"""

    SYSTEM = "system"
    INTEGRATION = "integration"
    VOICE = "voice"
    LLM = "llm"
    MONITORING = "monitoring"


@dataclass
class ProcessInfo:
    """Process information"""

    id: str
    name: str
    type: ProcessType
    state: ProcessState
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BaseProcess(ABC):
    """Base class for background processes"""

    def __init__(
        self, name: str, process_type: ProcessType, config: Dict[str, Any] = None
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.type = process_type
        self.config = config or {}
        self.state = ProcessState.CREATED
        self.start_time = None
        self.process = None
        self.shutdown_event = threading.Event()
        self.error_queue = queue.Queue()
        self.metrics = {}

    @abstractmethod
    async def run(self):
        """Main process execution method"""
        pass

    async def start(self):
        """Start the process"""
        self.state = ProcessState.STARTING
        self.start_time = datetime.utcnow()
        try:
            await self.run()
            self.state = ProcessState.RUNNING
        except Exception as e:
            self.state = ProcessState.FAILED
            self.error_queue.put(str(e))
            logger.error(f"Process {self.name} failed to start: {e}")
            raise

    async def stop(self):
        """Stop the process"""
        self.state = ProcessState.STOPPING
        self.shutdown_event.set()
        await self.cleanup()
        self.state = ProcessState.STOPPED

    async def cleanup(self):
        """Cleanup resources"""
        pass

    def is_healthy(self) -> bool:
        """Check if process is healthy"""
        return self.state == ProcessState.RUNNING

    def get_metrics(self) -> Dict[str, Any]:
        """Get process metrics"""
        return {
            **self.metrics,
            "uptime": (
                time.time() - self.start_time.timestamp() if self.start_time else 0
            ),
            "restart_count": getattr(self, "restart_count", 0),
        }


class SystemMonitorProcess(BaseProcess):
    """System monitoring background process"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("system_monitor", ProcessType.SYSTEM, config)
        self.monitoring_interval = config.get("interval", 30) if config else 30

    async def run(self):
        """Monitor system resources"""
        logger.info(
            f"Starting system monitor with {self.monitoring_interval}s interval"
        )

        while not self.shutdown_event.is_set():
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                self.metrics.update(
                    {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available": memory.available,
                        "disk_percent": disk.percent,
                        "disk_free": disk.free,
                        "last_update": datetime.utcnow().isoformat(),
                    }
                )

                # Log warnings for high usage
                if cpu_percent > 90:
                    logger.warning(f"High CPU usage: {cpu_percent}%")
                if memory.percent > 90:
                    logger.warning(f"High memory usage: {memory.percent}%")
                if disk.percent > 90:
                    logger.warning(f"Low disk space: {disk.percent}% used")

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"System monitor error: {e}")
                await asyncio.sleep(5)


class IntegrationWorkerProcess(BaseProcess):
    """Background process for external integrations"""

    def __init__(self, integration_name: str, config: Dict[str, Any] = None):
        super().__init__(
            f"integration_{integration_name}", ProcessType.INTEGRATION, config
        )
        self.integration_name = integration_name
        self.work_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()

    async def run(self):
        """Process integration tasks"""
        logger.info(f"Starting integration worker: {self.integration_name}")

        while not self.shutdown_event.is_set():
            try:
                # Wait for tasks with timeout
                try:
                    task = await asyncio.wait_for(self.work_queue.get(), timeout=1.0)
                    result = await self.process_task(task)
                    await self.result_queue.put(result)
                    self.work_queue.task_done()

                    # Update metrics
                    self.metrics["tasks_processed"] = (
                        self.metrics.get("tasks_processed", 0) + 1
                    )

                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                logger.error(f"Integration worker {self.integration_name} error: {e}")
                await asyncio.sleep(1)

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single integration task"""
        task_type = task.get("type")

        if task_type == "web_search":
            return await self.handle_web_search(task)
        elif task_type == "api_call":
            return await self.handle_api_call(task)
        elif task_type == "data_fetch":
            return await self.handle_data_fetch(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}

    async def handle_web_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle web search task"""
        # Placeholder implementation
        query = task.get("query", "")
        return {
            "type": "web_search",
            "query": query,
            "results": [f"Result for: {query}"],
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def handle_api_call(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API call task"""
        # Placeholder implementation
        endpoint = task.get("endpoint", "")
        return {
            "type": "api_call",
            "endpoint": endpoint,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def handle_data_fetch(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data fetch task"""
        # Placeholder implementation
        source = task.get("source", "")
        return {
            "type": "data_fetch",
            "source": source,
            "data": {"fetched": True},
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def add_task(self, task: Dict[str, Any]):
        """Add a task to the work queue"""
        await self.work_queue.put(task)

    async def get_result(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Get a result from the result queue"""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None


class VoiceProcessorProcess(BaseProcess):
    """Background process for voice processing"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("voice_processor", ProcessType.VOICE, config)
        self.audio_queue = asyncio.Queue()
        self.tts_queue = asyncio.Queue()
        self.stt_queue = asyncio.Queue()

    async def run(self):
        """Process voice tasks"""
        logger.info("Starting voice processor")

        # Start sub-processors
        tasks = [
            asyncio.create_task(self.process_tts()),
            asyncio.create_task(self.process_stt()),
            asyncio.create_task(self.process_audio()),
        ]

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Voice processor error: {e}")
            # Cancel remaining tasks
            for task in tasks:
                task.cancel()

    async def process_tts(self):
        """Process text-to-speech requests"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    request = await asyncio.wait_for(self.tts_queue.get(), timeout=1.0)
                    result = await self.synthesize_speech(request)
                    self.metrics["tts_processed"] = (
                        self.metrics.get("tts_processed", 0) + 1
                    )
                    self.tts_queue.task_done()
                except asyncio.TimeoutError:
                    continue
            except Exception as e:
                logger.error(f"TTS processing error: {e}")
                await asyncio.sleep(1)

    async def process_stt(self):
        """Process speech-to-text requests"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    request = await asyncio.wait_for(self.stt_queue.get(), timeout=1.0)
                    result = await self.transcribe_audio(request)
                    self.metrics["stt_processed"] = (
                        self.metrics.get("stt_processed", 0) + 1
                    )
                    self.stt_queue.task_done()
                except asyncio.TimeoutError:
                    continue
            except Exception as e:
                logger.error(f"STT processing error: {e}")
                await asyncio.sleep(1)

    async def process_audio(self):
        """Process general audio tasks"""
        while not self.shutdown_event.is_set():
            try:
                try:
                    request = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=1.0
                    )
                    result = await self.process_audio_request(request)
                    self.metrics["audio_processed"] = (
                        self.metrics.get("audio_processed", 0) + 1
                    )
                    self.audio_queue.task_done()
                except asyncio.TimeoutError:
                    continue
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                await asyncio.sleep(1)

    async def synthesize_speech(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize text to speech"""
        # Placeholder implementation
        text = request.get("text", "")
        return {
            "type": "tts",
            "text": text,
            "audio_path": f"/tmp/speech_{uuid.uuid4()}.wav",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def transcribe_audio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe audio to text"""
        # Placeholder implementation
        audio_path = request.get("audio_path", "")
        return {
            "type": "stt",
            "audio_path": audio_path,
            "text": "Transcribed text",
            "confidence": 0.95,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def process_audio_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process general audio request"""
        # Placeholder implementation
        return {
            "type": "audio",
            "processed": True,
            "timestamp": datetime.utcnow().isoformat(),
        }


class BackgroundProcessManager:
    """Manages all background processes"""

    def __init__(self):
        self.processes: Dict[str, BaseProcess] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        self.monitor_task = None

    async def start(self):
        """Start the process manager"""
        self.running = True

        # Start default processes
        await self.start_default_processes()

        # Start monitoring
        self.monitor_task = asyncio.create_task(self.monitor_processes())

        logger.info("Background process manager started")

    async def stop(self):
        """Stop the process manager"""
        self.running = False

        # Stop monitoring
        if self.monitor_task:
            self.monitor_task.cancel()

        # Stop all processes
        for process in list(self.processes.values()):
            await self.stop_process(process.id)

        # Shutdown executor
        self.executor.shutdown(wait=True)

        logger.info("Background process manager stopped")

    async def start_default_processes(self):
        """Start default background processes"""
        # System monitor
        system_monitor = SystemMonitorProcess()
        await self.start_process(system_monitor)

        # Integration workers
        for integration in ["web_search", "social_media", "calendar"]:
            worker = IntegrationWorkerProcess(integration)
            await self.start_process(worker)

        # Voice processor
        voice_processor = VoiceProcessorProcess()
        await self.start_process(voice_processor)

    async def start_process(self, process: BaseProcess):
        """Start a background process"""
        process_id = process.id
        self.processes[process_id] = process

        try:
            # Start process in executor
            task = asyncio.create_task(process.start())
            logger.info(f"Started process: {process.name} ({process_id})")
            return process_id
        except Exception as e:
            logger.error(f"Failed to start process {process.name}: {e}")
            if process_id in self.processes:
                del self.processes[process_id]
            raise

    async def stop_process(self, process_id: str):
        """Stop a background process"""
        if process_id not in self.processes:
            raise ValueError(f"Process {process_id} not found")

        process = self.processes[process_id]
        try:
            await process.stop()
            del self.processes[process_id]
            logger.info(f"Stopped process: {process.name} ({process_id})")
        except Exception as e:
            logger.error(f"Error stopping process {process.name}: {e}")
            raise

    async def restart_process(self, process_id: str):
        """Restart a background process"""
        if process_id not in self.processes:
            raise ValueError(f"Process {process_id} not found")

        process = self.processes[process_id]
        logger.info(f"Restarting process: {process.name} ({process_id})")

        # Create new instance with same config
        new_process = type(process)(process.name, process.config)

        # Stop old process
        await self.stop_process(process_id)

        # Start new process
        await self.start_process(new_process)

    async def monitor_processes(self):
        """Monitor process health and restart failed processes"""
        while self.running:
            try:
                for process_id, process in list(self.processes.items()):
                    if not process.is_healthy():
                        logger.warning(f"Unhealthy process detected: {process.name}")

                        # Auto-restart if configured
                        if process.config.get("auto_restart", True):
                            restart_count = getattr(process, "restart_count", 0)
                            max_restarts = process.config.get("max_restarts", 5)

                            if restart_count < max_restarts:
                                logger.info(f"Auto-restarting process: {process.name}")
                                await self.restart_process(process_id)
                                process.restart_count = restart_count + 1
                            else:
                                logger.error(
                                    f"Max restarts reached for process: {process.name}"
                                )

                    # Update process metrics
                    if hasattr(process, "process") and process.process:
                        try:
                            proc = psutil.Process(
                                process.process.pid
                                if hasattr(process.process, "pid")
                                else os.getpid()
                            )
                            process.cpu_usage = proc.cpu_percent()
                            process.memory_usage = proc.memory_percent()
                        except (psutil.NoSuchProcess, AttributeError):
                            pass

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                await asyncio.sleep(5)

    def get_process_info(self, process_id: str) -> Optional[ProcessInfo]:
        """Get process information"""
        if process_id not in self.processes:
            return None

        process = self.processes[process_id]
        return ProcessInfo(
            id=process.id,
            name=process.name,
            type=process.type,
            state=process.state,
            pid=(
                getattr(process.process, "pid", None)
                if hasattr(process, "process") and process.process
                else None
            ),
            start_time=process.start_time,
            restart_count=getattr(process, "restart_count", 0),
            cpu_usage=getattr(process, "cpu_usage", 0.0),
            memory_usage=getattr(process, "memory_usage", 0.0),
            config=process.config,
            metrics=process.get_metrics(),
        )

    def list_processes(self) -> List[ProcessInfo]:
        """List all processes"""
        return [self.get_process_info(pid) for pid in self.processes.keys()]

    def get_process_metrics(self) -> Dict[str, Any]:
        """Get overall process metrics"""
        total_processes = len(self.processes)
        running_processes = sum(
            1 for p in self.processes.values() if p.state == ProcessState.RUNNING
        )
        failed_processes = sum(
            1 for p in self.processes.values() if p.state == ProcessState.FAILED
        )

        return {
            "total_processes": total_processes,
            "running_processes": running_processes,
            "failed_processes": failed_processes,
            "health_percentage": (
                (running_processes / total_processes * 100)
                if total_processes > 0
                else 0
            ),
            "process_types": {
                ptype.value: sum(1 for p in self.processes.values() if p.type == ptype)
                for ptype in ProcessType
            },
        }


# Global process manager instance
_process_manager: Optional[BackgroundProcessManager] = None


def get_process_manager() -> BackgroundProcessManager:
    """Get global process manager instance"""
    global _process_manager
    if _process_manager is None:
        _process_manager = BackgroundProcessManager()
    return _process_manager


async def start_background_processes():
    """Start all background processes"""
    manager = get_process_manager()
    await manager.start()


async def stop_background_processes():
    """Stop all background processes"""
    manager = get_process_manager()
    await manager.stop()


# Signal handling for graceful shutdown
def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(stop_background_processes())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
