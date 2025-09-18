"""
Autonomous Self-Improvement Framework for Vega2.0
Phase 2: Telemetry Infrastructure Implementation

Advanced monitoring, metrics collection, and performance tracking system
for autonomous AI improvement and optimization
"""

import time
import psutil
import threading
import functools
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import logging
import traceback
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for function execution"""

    function_name: str
    execution_time: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime
    args_signature: str
    return_size_bytes: Optional[int]
    exception_info: Optional[str]
    call_depth: int
    thread_id: str


@dataclass
class SystemMetrics:
    """System-wide performance metrics"""

    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    active_threads: int
    open_files: int
    network_connections: int


@dataclass
class ConversationMetrics:
    """Conversation-specific performance metrics"""

    session_id: str
    prompt_length: int
    response_length: int
    response_time_seconds: float
    llm_processing_time: float
    database_write_time: float
    memory_operations: int
    quality_score: Optional[float]
    user_satisfaction: Optional[int]
    timestamp: datetime


class TelemetryCollector:
    """High-performance telemetry data collector"""

    def __init__(self, db_path: str = "/home/ncacord/Vega2.0/telemetry.db"):
        self.db_path = db_path
        self.metrics_buffer = deque(maxlen=10000)  # In-memory buffer
        self.system_metrics_buffer = deque(maxlen=1000)
        self.conversation_metrics_buffer = deque(maxlen=5000)
        self._lock = threading.Lock()
        self._shutdown = False
        self._background_thread = None
        self._init_database()
        self._start_background_collection()

    def _init_database(self):
        """Initialize telemetry database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    function_name TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    cpu_percent REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    args_signature TEXT,
                    return_size_bytes INTEGER,
                    exception_info TEXT,
                    call_depth INTEGER,
                    thread_id TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage_percent REAL NOT NULL,
                    memory_usage_mb REAL NOT NULL,
                    memory_available_mb REAL NOT NULL,
                    disk_usage_percent REAL NOT NULL,
                    active_threads INTEGER,
                    open_files INTEGER,
                    network_connections INTEGER
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    prompt_length INTEGER NOT NULL,
                    response_length INTEGER NOT NULL,
                    response_time_seconds REAL NOT NULL,
                    llm_processing_time REAL NOT NULL,
                    database_write_time REAL NOT NULL,
                    memory_operations INTEGER,
                    quality_score REAL,
                    user_satisfaction INTEGER,
                    timestamp TEXT NOT NULL
                )
            """
            )

            # Create indexes for performance
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_function ON performance_metrics(function_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sys_timestamp ON system_metrics(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_metrics(session_id)"
            )

    def _start_background_collection(self):
        """Start background thread for system metrics collection"""
        self._background_thread = threading.Thread(
            target=self._background_collection_loop, daemon=True
        )
        self._background_thread.start()

    def _background_collection_loop(self):
        """Background loop for collecting system metrics and flushing buffers"""
        while not self._shutdown:
            try:
                # Collect system metrics every 30 seconds
                self._collect_system_metrics()

                # Flush buffers to database every 60 seconds
                self._flush_buffers()

                time.sleep(30)

            except Exception as e:
                logger.error(f"Error in background collection: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            process = psutil.Process(os.getpid())

            # Get system-wide metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Get process-specific metrics
            process_memory = process.memory_info().rss / (1024 * 1024)  # MB
            active_threads = threading.active_count()

            try:
                open_files = len(process.open_files())
            except:
                open_files = 0

            try:
                network_connections = len(process.connections())
            except:
                network_connections = 0

            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                active_threads=active_threads,
                open_files=open_files,
                network_connections=network_connections,
            )

            with self._lock:
                self.system_metrics_buffer.append(metrics)

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        with self._lock:
            self.metrics_buffer.append(metrics)

    def record_conversation(self, metrics: ConversationMetrics):
        """Record conversation metrics"""
        with self._lock:
            self.conversation_metrics_buffer.append(metrics)

    def _flush_buffers(self):
        """Flush all metric buffers to database"""
        with self._lock:
            if not (
                self.metrics_buffer
                or self.system_metrics_buffer
                or self.conversation_metrics_buffer
            ):
                return

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Flush performance metrics
                if self.metrics_buffer:
                    perf_data = []
                    with self._lock:
                        while self.metrics_buffer:
                            metrics = self.metrics_buffer.popleft()
                            perf_data.append(
                                (
                                    metrics.function_name,
                                    metrics.execution_time,
                                    metrics.memory_usage_mb,
                                    metrics.cpu_percent,
                                    metrics.timestamp.isoformat(),
                                    metrics.args_signature,
                                    metrics.return_size_bytes,
                                    metrics.exception_info,
                                    metrics.call_depth,
                                    metrics.thread_id,
                                )
                            )

                    conn.executemany(
                        """
                        INSERT INTO performance_metrics 
                        (function_name, execution_time, memory_usage_mb, cpu_percent, timestamp,
                         args_signature, return_size_bytes, exception_info, call_depth, thread_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        perf_data,
                    )

                # Flush system metrics
                if self.system_metrics_buffer:
                    sys_data = []
                    with self._lock:
                        while self.system_metrics_buffer:
                            metrics = self.system_metrics_buffer.popleft()
                            sys_data.append(
                                (
                                    metrics.timestamp.isoformat(),
                                    metrics.cpu_usage_percent,
                                    metrics.memory_usage_mb,
                                    metrics.memory_available_mb,
                                    metrics.disk_usage_percent,
                                    metrics.active_threads,
                                    metrics.open_files,
                                    metrics.network_connections,
                                )
                            )

                    conn.executemany(
                        """
                        INSERT INTO system_metrics 
                        (timestamp, cpu_usage_percent, memory_usage_mb, memory_available_mb,
                         disk_usage_percent, active_threads, open_files, network_connections)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        sys_data,
                    )

                # Flush conversation metrics
                if self.conversation_metrics_buffer:
                    conv_data = []
                    with self._lock:
                        while self.conversation_metrics_buffer:
                            metrics = self.conversation_metrics_buffer.popleft()
                            conv_data.append(
                                (
                                    metrics.session_id,
                                    metrics.prompt_length,
                                    metrics.response_length,
                                    metrics.response_time_seconds,
                                    metrics.llm_processing_time,
                                    metrics.database_write_time,
                                    metrics.memory_operations,
                                    metrics.quality_score,
                                    metrics.user_satisfaction,
                                    metrics.timestamp.isoformat(),
                                )
                            )

                    conn.executemany(
                        """
                        INSERT INTO conversation_metrics 
                        (session_id, prompt_length, response_length, response_time_seconds,
                         llm_processing_time, database_write_time, memory_operations,
                         quality_score, user_satisfaction, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        conv_data,
                    )

                conn.commit()

        except Exception as e:
            logger.error(f"Failed to flush metrics to database: {e}")

    def shutdown(self):
        """Shutdown telemetry collection"""
        self._shutdown = True
        if self._background_thread:
            self._background_thread.join(timeout=5.0)
        self._flush_buffers()  # Final flush


# Global telemetry collector instance
_telemetry_collector = None


def get_telemetry_collector() -> TelemetryCollector:
    """Get global telemetry collector instance"""
    global _telemetry_collector
    if _telemetry_collector is None:
        _telemetry_collector = TelemetryCollector()
    return _telemetry_collector


def monitor_performance(include_args: bool = False, include_return_size: bool = False):
    """
    Decorator for monitoring function performance

    Args:
        include_args: Whether to include function arguments in telemetry
        include_return_size: Whether to measure return value size
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = get_telemetry_collector()

            # Get current frame info
            frame = sys._getframe()
            call_depth = len(traceback.extract_stack())
            thread_id = str(threading.get_ident())

            # Record start state
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            start_cpu = process.cpu_percent()

            exception_info = None
            return_size = None
            result = None

            try:
                result = func(*args, **kwargs)

                # Measure return size if requested
                if include_return_size and result is not None:
                    try:
                        return_size = len(str(result).encode("utf-8"))
                    except:
                        return_size = None

                return result

            except Exception as e:
                exception_info = f"{type(e).__name__}: {str(e)}"
                raise

            finally:
                # Record end state
                end_time = time.time()
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                end_cpu = process.cpu_percent()

                # Create args signature
                args_signature = ""
                if include_args:
                    try:
                        args_repr = f"args={len(args)}, kwargs={list(kwargs.keys())}"
                        args_signature = args_repr[:200]  # Limit size
                    except:
                        args_signature = "serialization_error"

                # Create performance metrics
                metrics = PerformanceMetrics(
                    function_name=f"{func.__module__}.{func.__name__}",
                    execution_time=end_time - start_time,
                    memory_usage_mb=end_memory - start_memory,
                    cpu_percent=end_cpu - start_cpu,
                    timestamp=datetime.now(),
                    args_signature=args_signature,
                    return_size_bytes=return_size,
                    exception_info=exception_info,
                    call_depth=call_depth,
                    thread_id=thread_id,
                )

                collector.record_performance(metrics)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = get_telemetry_collector()

            # Get current frame info
            call_depth = len(traceback.extract_stack())
            thread_id = str(threading.get_ident())

            # Record start state
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / (1024 * 1024)  # MB
            start_cpu = process.cpu_percent()

            exception_info = None
            return_size = None
            result = None

            try:
                result = await func(*args, **kwargs)

                # Measure return size if requested
                if include_return_size and result is not None:
                    try:
                        return_size = len(str(result).encode("utf-8"))
                    except:
                        return_size = None

                return result

            except Exception as e:
                exception_info = f"{type(e).__name__}: {str(e)}"
                raise

            finally:
                # Record end state
                end_time = time.time()
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                end_cpu = process.cpu_percent()

                # Create args signature
                args_signature = ""
                if include_args:
                    try:
                        args_repr = f"args={len(args)}, kwargs={list(kwargs.keys())}"
                        args_signature = args_repr[:200]  # Limit size
                    except:
                        args_signature = "serialization_error"

                # Create performance metrics
                metrics = PerformanceMetrics(
                    function_name=f"{func.__module__}.{func.__name__}",
                    execution_time=end_time - start_time,
                    memory_usage_mb=end_memory - start_memory,
                    cpu_percent=end_cpu - start_cpu,
                    timestamp=datetime.now(),
                    args_signature=args_signature,
                    return_size_bytes=return_size,
                    exception_info=exception_info,
                    call_depth=call_depth,
                    thread_id=thread_id,
                )

                collector.record_performance(metrics)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class ConversationTelemetry:
    """Specialized telemetry for conversation tracking"""

    def __init__(self):
        self.active_sessions = {}
        self._lock = threading.Lock()

    def start_conversation(self, session_id: str, prompt: str) -> dict:
        """Start tracking a conversation"""
        start_time = time.time()

        with self._lock:
            self.active_sessions[session_id] = {
                "start_time": start_time,
                "prompt_length": len(prompt),
                "memory_operations": 0,
                "llm_start_time": None,
                "db_write_times": [],
            }

        return self.active_sessions[session_id]

    def record_llm_start(self, session_id: str):
        """Record when LLM processing starts"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["llm_start_time"] = time.time()

    def record_memory_operation(self, session_id: str):
        """Record a memory operation"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["memory_operations"] += 1

    def record_db_write(self, session_id: str, write_time: float):
        """Record database write time"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["db_write_times"].append(write_time)

    def end_conversation(
        self,
        session_id: str,
        response: str,
        quality_score: Optional[float] = None,
        user_satisfaction: Optional[int] = None,
    ):
        """End conversation tracking and record metrics"""
        end_time = time.time()

        with self._lock:
            if session_id not in self.active_sessions:
                return

            session_data = self.active_sessions.pop(session_id)

        # Calculate metrics
        total_time = end_time - session_data["start_time"]
        llm_time = 0
        if session_data["llm_start_time"]:
            llm_time = end_time - session_data["llm_start_time"]

        db_write_time = sum(session_data["db_write_times"])

        # Create conversation metrics
        metrics = ConversationMetrics(
            session_id=session_id,
            prompt_length=session_data["prompt_length"],
            response_length=len(response),
            response_time_seconds=total_time,
            llm_processing_time=llm_time,
            database_write_time=db_write_time,
            memory_operations=session_data["memory_operations"],
            quality_score=quality_score,
            user_satisfaction=user_satisfaction,
            timestamp=datetime.now(),
        )

        # Record to telemetry
        collector = get_telemetry_collector()
        collector.record_conversation(metrics)


# Global conversation telemetry instance
_conversation_telemetry = None


def get_conversation_telemetry() -> ConversationTelemetry:
    """Get global conversation telemetry instance"""
    global _conversation_telemetry
    if _conversation_telemetry is None:
        _conversation_telemetry = ConversationTelemetry()
    return _conversation_telemetry


class TelemetryAnalyzer:
    """Analyzer for telemetry data to generate insights"""

    def __init__(self, db_path: str = "/home/ncacord/Vega2.0/telemetry.db"):
        self.db_path = db_path

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            # Top slowest functions
            slowest = conn.execute(
                """
                SELECT function_name, AVG(execution_time) as avg_time, COUNT(*) as call_count
                FROM performance_metrics 
                WHERE timestamp > ?
                GROUP BY function_name
                ORDER BY avg_time DESC
                LIMIT 10
            """,
                (since.isoformat(),),
            ).fetchall()

            # Memory hogs
            memory_hogs = conn.execute(
                """
                SELECT function_name, AVG(memory_usage_mb) as avg_memory, COUNT(*) as call_count
                FROM performance_metrics 
                WHERE timestamp > ? AND memory_usage_mb > 0
                GROUP BY function_name
                ORDER BY avg_memory DESC
                LIMIT 10
            """,
                (since.isoformat(),),
            ).fetchall()

            # Error rates
            errors = conn.execute(
                """
                SELECT function_name, 
                       COUNT(*) as total_calls,
                       SUM(CASE WHEN exception_info IS NOT NULL THEN 1 ELSE 0 END) as error_count
                FROM performance_metrics 
                WHERE timestamp > ?
                GROUP BY function_name
                HAVING error_count > 0
                ORDER BY error_count DESC
            """,
                (since.isoformat(),),
            ).fetchall()

        return {
            "period_hours": hours,
            "slowest_functions": [
                {"name": row[0], "avg_time": row[1], "calls": row[2]} for row in slowest
            ],
            "memory_intensive": [
                {"name": row[0], "avg_memory_mb": row[1], "calls": row[2]}
                for row in memory_hogs
            ],
            "error_prone": [
                {"name": row[0], "total_calls": row[1], "errors": row[2]}
                for row in errors
            ],
        }

    def get_conversation_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get conversation analytics for the last N hours"""
        since = datetime.now() - timedelta(hours=hours)

        with sqlite3.connect(self.db_path) as conn:
            # Average response times
            avg_response = conn.execute(
                """
                SELECT AVG(response_time_seconds), AVG(llm_processing_time)
                FROM conversation_metrics 
                WHERE timestamp > ?
            """,
                (since.isoformat(),),
            ).fetchone()

            # Quality distribution
            quality_dist = conn.execute(
                """
                SELECT 
                    AVG(quality_score) as avg_quality,
                    MIN(quality_score) as min_quality,
                    MAX(quality_score) as max_quality
                FROM conversation_metrics 
                WHERE timestamp > ? AND quality_score IS NOT NULL
            """,
                (since.isoformat(),),
            ).fetchone()

            # Conversation volume
            volume = conn.execute(
                """
                SELECT COUNT(*) as total_conversations
                FROM conversation_metrics 
                WHERE timestamp > ?
            """,
                (since.isoformat(),),
            ).fetchone()

        return {
            "period_hours": hours,
            "avg_response_time": avg_response[0] if avg_response[0] else 0,
            "avg_llm_time": avg_response[1] if avg_response[1] else 0,
            "avg_quality_score": (
                quality_dist[0] if quality_dist and quality_dist[0] else None
            ),
            "min_quality_score": (
                quality_dist[1] if quality_dist and quality_dist[1] else None
            ),
            "max_quality_score": (
                quality_dist[2] if quality_dist and quality_dist[2] else None
            ),
            "total_conversations": volume[0] if volume else 0,
        }


def main():
    """Telemetry system demonstration"""
    print("🔬 TELEMETRY INFRASTRUCTURE INITIALIZATION")
    print("=" * 50)

    # Initialize telemetry
    collector = get_telemetry_collector()
    conversation_telemetry = get_conversation_telemetry()
    analyzer = TelemetryAnalyzer()

    print("✅ Telemetry collector initialized")
    print("✅ Background metrics collection started")
    print("✅ Database schema created")

    # Demonstrate performance monitoring
    @monitor_performance(include_args=True, include_return_size=True)
    def sample_function(x: int, y: str = "test") -> str:
        """Sample function for telemetry demonstration"""
        time.sleep(0.1)  # Simulate work
        return f"Processed {x} with {y}"

    @monitor_performance()
    async def sample_async_function(delay: float) -> str:
        """Sample async function for telemetry demonstration"""
        await asyncio.sleep(delay)
        return f"Async work completed after {delay}s"

    # Test performance monitoring
    print("\n📊 Testing performance monitoring...")

    # Test sync function
    result1 = sample_function(42, "hello")
    print(f"Sync function result: {result1}")

    # Test async function
    async def test_async():
        result = await sample_async_function(0.05)
        print(f"Async function result: {result}")

    asyncio.run(test_async())

    # Test conversation telemetry
    print("\n💬 Testing conversation telemetry...")

    session_id = "test_session_123"
    conversation_telemetry.start_conversation(session_id, "Hello, how are you?")
    conversation_telemetry.record_llm_start(session_id)
    time.sleep(0.1)  # Simulate LLM processing
    conversation_telemetry.record_memory_operation(session_id)
    conversation_telemetry.record_db_write(session_id, 0.02)
    conversation_telemetry.end_conversation(
        session_id,
        "I'm doing well, thank you for asking!",
        quality_score=8.5,
        user_satisfaction=9,
    )

    print("✅ Conversation metrics recorded")

    # Wait a moment for background collection
    time.sleep(2)

    # Force flush buffers
    collector._flush_buffers()

    # Get analytics
    print("\n📈 Performance Analytics:")
    perf_summary = analyzer.get_performance_summary(hours=1)
    print(f"Slowest functions: {len(perf_summary['slowest_functions'])}")
    for func in perf_summary["slowest_functions"][:3]:
        print(
            f"  - {func['name']}: {func['avg_time']:.3f}s avg ({func['calls']} calls)"
        )

    print("\n💬 Conversation Analytics:")
    conv_analytics = analyzer.get_conversation_analytics(hours=1)
    print(f"Total conversations: {conv_analytics['total_conversations']}")
    print(f"Average response time: {conv_analytics['avg_response_time']:.3f}s")
    print(f"Average quality score: {conv_analytics['avg_quality_score']}")

    print("\n🎯 TELEMETRY INFRASTRUCTURE READY")
    print("System is now monitoring all operations for autonomous improvement")


if __name__ == "__main__":
    main()
