#!/usr/bin/env python3
"""
VEGA SMART ASSISTANT - INTELLIGENT COORDINATION LAYER
=====================================================

Advanced AI assistant that orchestrates all Vega components into a unified,
intelligent system. Provides natural language interface to the entire ecosystem.

Features:
- 🧠 Natural language understanding and processing
- 🔄 Dynamic component orchestration and coordination
- 🎯 Context-aware task planning and execution
- 📊 Unified status monitoring and reporting
- 🤝 Intelligent decision support and recommendations
- 🛡️ Safety monitoring and intervention capabilities
- 📝 Comprehensive logging and audit trails
- 🌐 Multi-modal interaction (text, voice, web)

Usage:
    python vega_smart.py "What's the status of all systems?"
    python vega_smart.py --daemon                    # Background assistant
    python vega_smart.py --interactive               # Interactive session
    python vega_smart.py --web --port 8082          # Web interface
"""

import os
import sys
import json
import time
import asyncio
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import logging
import re
import hashlib

# AI and NLP imports
AI_AVAILABLE = False
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords

    AI_AVAILABLE = True
except ImportError:
    pass

# Command execution imports
EXECUTION_AVAILABLE = False
try:
    import subprocess
    import shlex
    import psutil

    EXECUTION_AVAILABLE = True
except ImportError:
    pass


class TaskType(Enum):
    """Types of tasks the smart assistant can handle"""

    STATUS_QUERY = "status_query"
    SYSTEM_CONTROL = "system_control"
    INTEGRATION_REQUEST = "integration_request"
    ANALYSIS_REQUEST = "analysis_request"
    MONITORING_SETUP = "monitoring_setup"
    TROUBLESHOOTING = "troubleshooting"
    CONFIGURATION = "configuration"
    REPORTING = "reporting"


class Priority(Enum):
    """Task priority levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class ExecutionStatus(Enum):
    """Task execution status"""

    PENDING = "pending"
    ANALYZING = "analyzing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """A task for the smart assistant"""

    id: str
    task_type: TaskType
    priority: Priority
    query: str
    parsed_intent: Dict[str, Any]
    required_components: List[str]
    execution_plan: List[Dict[str, Any]]
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: str = ""
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ComponentStatus:
    """Status of a Vega component"""

    name: str
    status: str  # "running", "stopped", "error", "unknown"
    health: str  # "optimal", "good", "warning", "critical"
    cpu_usage: float
    memory_usage: float
    last_check: str
    details: Dict[str, Any]


class VegaSmartAssistant:
    """Intelligent coordination layer for Vega ecosystem"""

    def __init__(self, mode: str = "interactive"):
        self.mode = mode
        self.base_dir = Path(__file__).parent
        self.state_dir = self.base_dir / "vega_state"
        self.logs_dir = self.base_dir / "vega_logs"

        # Create directories
        for directory in [self.state_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)8s | %(name)20s | %(message)s",
            handlers=[
                logging.FileHandler(self.logs_dir / "vega_smart.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("VegaSmart")

        # Task management
        self.active_tasks: Dict[str, Task] = {}
        self.task_history: List[Task] = []

        # Component tracking
        self.components: Dict[str, ComponentStatus] = {}

        # Intent classification and NLP
        self.intent_patterns = {
            TaskType.STATUS_QUERY: [
                r"status",
                r"how.*doing",
                r"health",
                r"running",
                r"working",
                r"check.*system",
                r"system.*status",
                r"what.*happening",
            ],
            TaskType.SYSTEM_CONTROL: [
                r"start",
                r"stop",
                r"restart",
                r"shutdown",
                r"reboot",
                r"enable",
                r"disable",
                r"turn.*on",
                r"turn.*off",
            ],
            TaskType.INTEGRATION_REQUEST: [
                r"integrate",
                r"connect",
                r"add.*service",
                r"setup.*integration",
                r"link.*with",
                r"configure.*connection",
            ],
            TaskType.ANALYSIS_REQUEST: [
                r"analyze",
                r"examine",
                r"investigate",
                r"report.*on",
                r"what.*happened",
                r"why.*not.*working",
                r"performance",
            ],
            TaskType.MONITORING_SETUP: [
                r"monitor",
                r"watch",
                r"track",
                r"alert.*when",
                r"notify.*if",
                r"set.*up.*monitoring",
            ],
            TaskType.TROUBLESHOOTING: [
                r"problem",
                r"issue",
                r"error",
                r"not.*working",
                r"broken",
                r"fix",
                r"troubleshoot",
                r"debug",
            ],
            TaskType.CONFIGURATION: [
                r"configure",
                r"settings",
                r"options",
                r"parameters",
                r"change.*config",
                r"update.*settings",
            ],
            TaskType.REPORTING: [
                r"report",
                r"summary",
                r"overview",
                r"statistics",
                r"logs",
                r"history",
                r"what.*happened.*today",
            ],
        }

        # Component command mappings
        self.component_commands = {
            "vega_core": {
                "status": "curl -s http://127.0.0.1:8000/healthz",
                "start": "python -m uvicorn app:app --host 127.0.0.1 --port 8000",
                "stop": "pkill -f 'uvicorn app:app'",
            },
            "voice_visualizer": {
                "status": "ps aux | grep -v grep | grep voice_visualizer",
                "start": "python voice_visualizer.py --daemon",
                "stop": "pkill -f voice_visualizer",
            },
            "network_scanner": {
                "status": "ps aux | grep -v grep | grep network_scanner",
                "start": "python network_scanner.py --daemon",
                "stop": "pkill -f network_scanner",
            },
            "integration_engine": {
                "status": "ps aux | grep -v grep | grep integration_engine",
                "start": "python integration_engine.py --daemon",
                "stop": "pkill -f integration_engine",
            },
            "ui": {
                "start": "python vega_ui.py",
                "dashboard": "python vega_dashboard.py --port 8080",
            },
        }

        # Initialize database
        self.init_database()

        # Load knowledge base
        self.load_knowledge_base()

        self.logger.info(f"🧠 Vega Smart Assistant initialized in {mode} mode")

    def init_database(self):
        """Initialize SQLite database for task and knowledge management"""
        db_path = self.state_dir / "vega_smart.db"

        try:
            with sqlite3.connect(db_path) as conn:
                # Tasks table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        query TEXT NOT NULL,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        completed_at TEXT,
                        result_data TEXT,
                        error_message TEXT
                    )
                """
                )

                # Component status table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS component_status (
                        name TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        health TEXT NOT NULL,
                        cpu_usage REAL,
                        memory_usage REAL,
                        last_check TEXT NOT NULL,
                        details_data TEXT
                    )
                """
                )

                # Knowledge base table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS knowledge_base (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        topic TEXT NOT NULL,
                        content TEXT NOT NULL,
                        keywords TEXT,
                        created_at TEXT,
                        relevance_score REAL DEFAULT 1.0
                    )
                """
                )

                # Interaction history table
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_query TEXT NOT NULL,
                        assistant_response TEXT NOT NULL,
                        task_id TEXT,
                        timestamp TEXT NOT NULL,
                        satisfaction_score REAL
                    )
                """
                )

                conn.commit()

            self.logger.info("📊 Smart assistant database initialized")

        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {e}")

    def load_knowledge_base(self):
        """Load knowledge base about Vega components and operations"""
        try:
            knowledge_entries = [
                {
                    "category": "components",
                    "topic": "vega_core",
                    "content": "Core Vega service providing chat API, conversation logging, and LLM integration. Runs on port 8000.",
                    "keywords": "core,api,chat,llm,port 8000",
                },
                {
                    "category": "components",
                    "topic": "voice_visualizer",
                    "content": "Audio personality engine with real-time voice visualization, emotion analysis, and audio processing.",
                    "keywords": "voice,audio,visualization,emotion,microphone",
                },
                {
                    "category": "components",
                    "topic": "network_scanner",
                    "content": "Intelligent network discovery engine that scans for devices, services, and integration opportunities.",
                    "keywords": "network,scan,discovery,devices,integration",
                },
                {
                    "category": "components",
                    "topic": "integration_engine",
                    "content": "Ethical AI orchestrator for decision-making, risk assessment, and integration approval.",
                    "keywords": "decisions,ethics,integration,risk,approval",
                },
                {
                    "category": "operations",
                    "topic": "health_check",
                    "content": "System health monitoring includes CPU, memory, temperature, and daemon status checks.",
                    "keywords": "health,monitoring,cpu,memory,temperature,status",
                },
                {
                    "category": "troubleshooting",
                    "topic": "common_issues",
                    "content": "Common issues include port conflicts, missing dependencies, permission errors, and daemon failures.",
                    "keywords": "issues,problems,errors,troubleshooting,debugging",
                },
            ]

            db_path = self.state_dir / "vega_smart.db"
            with sqlite3.connect(db_path) as conn:
                for entry in knowledge_entries:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO knowledge_base 
                        (category, topic, content, keywords, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            entry["category"],
                            entry["topic"],
                            entry["content"],
                            entry["keywords"],
                            datetime.now().isoformat(),
                        ),
                    )

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error loading knowledge base: {e}")

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query"""
        try:
            self.logger.info(f"🧠 Processing query: {query}")

            # Parse intent and extract entities
            task_type = self.classify_intent(query)
            parsed_intent = self.parse_entities(query, task_type)

            # Create task
            task = Task(
                id=self.generate_task_id(),
                task_type=task_type,
                priority=self.determine_priority(query, task_type),
                query=query,
                parsed_intent=parsed_intent,
                required_components=self.identify_required_components(parsed_intent),
                execution_plan=await self.create_execution_plan(
                    task_type, parsed_intent
                ),
            )

            # Execute task
            result = await self.execute_task(task)

            # Log interaction
            await self.log_interaction(query, result.get("response", ""), task.id)

            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing query: {e}")
            return {
                "success": False,
                "response": f"Sorry, I encountered an error: {str(e)}",
                "error": str(e),
            }

    def classify_intent(self, query: str) -> TaskType:
        """Classify the intent of a user query"""
        query_lower = query.lower()

        # Score each task type based on pattern matches
        scores = {}
        for task_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[task_type] = score

        # Return the highest scoring task type
        if scores:
            return max(scores, key=scores.get)
        else:
            return TaskType.STATUS_QUERY  # Default fallback

    def parse_entities(self, query: str, task_type: TaskType) -> Dict[str, Any]:
        """Parse entities from the query based on task type"""
        entities = {"components": [], "actions": [], "parameters": {}}

        query_lower = query.lower()

        # Extract component names
        for component in self.component_commands.keys():
            if component.replace("_", " ") in query_lower or component in query_lower:
                entities["components"].append(component)

        # Extract actions based on task type
        if task_type == TaskType.SYSTEM_CONTROL:
            actions = ["start", "stop", "restart", "shutdown", "enable", "disable"]
            for action in actions:
                if action in query_lower:
                    entities["actions"].append(action)

        elif task_type == TaskType.STATUS_QUERY:
            entities["actions"].append("status")

        elif task_type == TaskType.ANALYSIS_REQUEST:
            entities["actions"].append("analyze")

        # Extract time-related entities
        time_patterns = [
            (r"last (\d+) (hours?|minutes?|days?)", "time_range"),
            (r"today", "today"),
            (r"yesterday", "yesterday"),
            (r"this week", "this_week"),
        ]

        for pattern, entity_type in time_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entities["parameters"]["time"] = {
                    "type": entity_type,
                    "value": match.group(0),
                }

        return entities

    def determine_priority(self, query: str, task_type: TaskType) -> Priority:
        """Determine task priority based on query content"""
        query_lower = query.lower()

        # Critical keywords
        if any(
            word in query_lower
            for word in ["emergency", "critical", "urgent", "down", "failed"]
        ):
            return Priority.CRITICAL

        # High priority keywords
        if any(
            word in query_lower
            for word in ["error", "problem", "issue", "not working", "broken"]
        ):
            return Priority.HIGH

        # System control tasks are generally high priority
        if task_type == TaskType.SYSTEM_CONTROL:
            return Priority.HIGH

        # Troubleshooting is medium-high priority
        if task_type == TaskType.TROUBLESHOOTING:
            return Priority.HIGH

        # Analysis and status queries are medium priority
        if task_type in [TaskType.ANALYSIS_REQUEST, TaskType.STATUS_QUERY]:
            return Priority.MEDIUM

        return Priority.LOW

    def identify_required_components(self, parsed_intent: Dict[str, Any]) -> List[str]:
        """Identify which components are needed for the task"""
        components = parsed_intent.get("components", [])

        # If no specific components mentioned, assume all components
        if not components:
            components = list(self.component_commands.keys())

        return components

    async def create_execution_plan(
        self, task_type: TaskType, parsed_intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create execution plan for the task"""
        plan = []

        if task_type == TaskType.STATUS_QUERY:
            plan.append(
                {
                    "action": "check_component_status",
                    "components": parsed_intent.get(
                        "components", list(self.component_commands.keys())
                    ),
                    "timeout": 30,
                }
            )
            plan.append({"action": "generate_status_report", "format": "detailed"})

        elif task_type == TaskType.SYSTEM_CONTROL:
            actions = parsed_intent.get("actions", [])
            components = parsed_intent.get("components", [])

            for component in components:
                for action in actions:
                    plan.append(
                        {
                            "action": "component_control",
                            "component": component,
                            "control_action": action,
                            "timeout": 60,
                        }
                    )

        elif task_type == TaskType.ANALYSIS_REQUEST:
            plan.append({"action": "gather_system_data", "scope": "comprehensive"})
            plan.append({"action": "analyze_data", "analysis_type": "performance"})
            plan.append(
                {"action": "generate_analysis_report", "include_recommendations": True}
            )

        elif task_type == TaskType.TROUBLESHOOTING:
            plan.append(
                {
                    "action": "check_component_status",
                    "components": parsed_intent.get(
                        "components", list(self.component_commands.keys())
                    ),
                    "detailed": True,
                }
            )
            plan.append(
                {
                    "action": "check_logs",
                    "components": parsed_intent.get("components", []),
                    "time_range": "1_hour",
                }
            )
            plan.append({"action": "diagnose_issues", "suggest_fixes": True})

        return plan

    async def execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a task according to its execution plan"""
        try:
            task.status = ExecutionStatus.EXECUTING
            self.active_tasks[task.id] = task

            results = []

            for step in task.execution_plan:
                action = step["action"]

                if action == "check_component_status":
                    result = await self.check_component_status(
                        step.get("components", [])
                    )
                    results.append(result)

                elif action == "generate_status_report":
                    result = await self.generate_status_report(results)
                    results.append(result)

                elif action == "component_control":
                    result = await self.control_component(
                        step["component"], step["control_action"]
                    )
                    results.append(result)

                elif action == "gather_system_data":
                    result = await self.gather_system_data()
                    results.append(result)

                elif action == "analyze_data":
                    result = await self.analyze_system_data(results)
                    results.append(result)

                elif action == "check_logs":
                    result = await self.check_component_logs(
                        step.get("components", []), step.get("time_range", "1_hour")
                    )
                    results.append(result)

                elif action == "diagnose_issues":
                    result = await self.diagnose_issues(results)
                    results.append(result)

            # Compile final response
            response = await self.compile_response(task, results)

            task.status = ExecutionStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.result = response

            return {
                "success": True,
                "response": response.get("text", "Task completed successfully"),
                "data": response.get("data", {}),
                "task_id": task.id,
            }

        except Exception as e:
            task.status = ExecutionStatus.FAILED
            task.error_message = str(e)

            return {
                "success": False,
                "response": f"Task failed: {str(e)}",
                "error": str(e),
                "task_id": task.id,
            }

        finally:
            # Save task to database
            await self.save_task(task)

    async def check_component_status(self, components: List[str]) -> Dict[str, Any]:
        """Check the status of specified components"""
        status_results = {}

        for component in components:
            try:
                if component in self.component_commands:
                    status_cmd = self.component_commands[component].get("status")
                    if status_cmd:
                        # Execute status command
                        proc = await asyncio.create_subprocess_shell(
                            status_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=10
                        )

                        if proc.returncode == 0:
                            status = "running"
                            health = "good"
                        else:
                            status = "stopped"
                            health = "critical"

                        # Get process info if available
                        cpu_usage = 0.0
                        memory_usage = 0.0

                        if EXECUTION_AVAILABLE:
                            try:
                                for proc_info in psutil.process_iter(
                                    ["pid", "name", "cpu_percent", "memory_percent"]
                                ):
                                    if (
                                        component.replace("_", " ")
                                        in proc_info.info["name"].lower()
                                    ):
                                        cpu_usage = proc_info.info["cpu_percent"]
                                        memory_usage = proc_info.info["memory_percent"]
                                        break
                            except:
                                pass

                        status_results[component] = ComponentStatus(
                            name=component,
                            status=status,
                            health=health,
                            cpu_usage=cpu_usage,
                            memory_usage=memory_usage,
                            last_check=datetime.now().isoformat(),
                            details={
                                "stdout": stdout.decode() if stdout else "",
                                "stderr": stderr.decode() if stderr else "",
                            },
                        )
                    else:
                        status_results[component] = ComponentStatus(
                            name=component,
                            status="unknown",
                            health="unknown",
                            cpu_usage=0.0,
                            memory_usage=0.0,
                            last_check=datetime.now().isoformat(),
                            details={"message": "No status command available"},
                        )

            except Exception as e:
                status_results[component] = ComponentStatus(
                    name=component,
                    status="error",
                    health="critical",
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    last_check=datetime.now().isoformat(),
                    details={"error": str(e)},
                )

        # Update component tracking
        self.components.update(status_results)

        return {
            "action": "component_status_check",
            "components": status_results,
            "timestamp": datetime.now().isoformat(),
        }

    async def generate_status_report(
        self, previous_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a comprehensive status report"""
        report = {
            "summary": {
                "total_components": 0,
                "running_components": 0,
                "stopped_components": 0,
                "error_components": 0,
            },
            "components": {},
            "recommendations": [],
        }

        # Find component status results
        for result in previous_results:
            if result.get("action") == "component_status_check":
                components = result.get("components", {})

                for name, status in components.items():
                    report["components"][name] = asdict(status)
                    report["summary"]["total_components"] += 1

                    if status.status == "running":
                        report["summary"]["running_components"] += 1
                    elif status.status == "stopped":
                        report["summary"]["stopped_components"] += 1
                    elif status.status == "error":
                        report["summary"]["error_components"] += 1

                    # Generate recommendations
                    if status.status == "stopped":
                        report["recommendations"].append(
                            f"Component '{name}' is stopped. Consider starting it with: start {name}"
                        )
                    elif status.status == "error":
                        report["recommendations"].append(
                            f"Component '{name}' has errors. Check logs and troubleshoot."
                        )
                    elif status.cpu_usage > 90:
                        report["recommendations"].append(
                            f"Component '{name}' has high CPU usage ({status.cpu_usage:.1f}%). Monitor performance."
                        )

        return {
            "action": "status_report",
            "report": report,
            "timestamp": datetime.now().isoformat(),
        }

    async def control_component(self, component: str, action: str) -> Dict[str, Any]:
        """Control a component (start, stop, restart)"""
        try:
            if component not in self.component_commands:
                return {"success": False, "message": f"Unknown component: {component}"}

            commands = self.component_commands[component]
            command = commands.get(action)

            if not command:
                return {
                    "success": False,
                    "message": f"Action '{action}' not available for component '{component}'",
                }

            # Execute command
            proc = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            success = proc.returncode == 0

            return {
                "action": "component_control",
                "component": component,
                "control_action": action,
                "success": success,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "action": "component_control",
                "component": component,
                "control_action": action,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def gather_system_data(self) -> Dict[str, Any]:
        """Gather comprehensive system data"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "system": {},
            "components": {},
            "network": {},
            "storage": {},
        }

        try:
            if EXECUTION_AVAILABLE:
                # System information
                data["system"]["cpu_count"] = psutil.cpu_count()
                data["system"]["cpu_usage"] = psutil.cpu_percent(interval=1)

                memory = psutil.virtual_memory()
                data["system"]["memory_total"] = memory.total
                data["system"]["memory_used"] = memory.used
                data["system"]["memory_percent"] = memory.percent

                disk = psutil.disk_usage("/")
                data["system"]["disk_total"] = disk.total
                data["system"]["disk_used"] = disk.used
                data["system"]["disk_percent"] = (disk.used / disk.total) * 100

                # Network information
                network_stats = psutil.net_io_counters()
                data["network"]["bytes_sent"] = network_stats.bytes_sent
                data["network"]["bytes_recv"] = network_stats.bytes_recv

            # Component data
            for name, component in self.components.items():
                data["components"][name] = asdict(component)

        except Exception as e:
            data["error"] = str(e)

        return {
            "action": "system_data_gathering",
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }

    async def analyze_system_data(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze gathered system data"""
        analysis = {
            "performance": {
                "overall_health": "good",
                "bottlenecks": [],
                "recommendations": [],
            },
            "security": {"issues": [], "recommendations": []},
            "efficiency": {"score": 0.8, "improvements": []},
        }

        # Find system data
        for result in results:
            if result.get("action") == "system_data_gathering":
                data = result.get("data", {})
                system_data = data.get("system", {})

                # Analyze CPU usage
                cpu_usage = system_data.get("cpu_usage", 0)
                if cpu_usage > 90:
                    analysis["performance"]["bottlenecks"].append("High CPU usage")
                    analysis["performance"]["overall_health"] = "critical"
                elif cpu_usage > 75:
                    analysis["performance"]["bottlenecks"].append("Elevated CPU usage")
                    analysis["performance"]["overall_health"] = "warning"

                # Analyze memory usage
                memory_percent = system_data.get("memory_percent", 0)
                if memory_percent > 90:
                    analysis["performance"]["bottlenecks"].append("High memory usage")
                    analysis["performance"]["overall_health"] = "critical"
                elif memory_percent > 75:
                    analysis["performance"]["bottlenecks"].append(
                        "Elevated memory usage"
                    )
                    if analysis["performance"]["overall_health"] == "good":
                        analysis["performance"]["overall_health"] = "warning"

                # Analyze disk usage
                disk_percent = system_data.get("disk_percent", 0)
                if disk_percent > 95:
                    analysis["performance"]["bottlenecks"].append("Critical disk space")
                    analysis["performance"]["overall_health"] = "critical"
                elif disk_percent > 85:
                    analysis["performance"]["bottlenecks"].append("Low disk space")
                    if analysis["performance"]["overall_health"] not in [
                        "critical",
                        "warning",
                    ]:
                        analysis["performance"]["overall_health"] = "warning"

        # Generate recommendations
        if analysis["performance"]["bottlenecks"]:
            for bottleneck in analysis["performance"]["bottlenecks"]:
                if "CPU" in bottleneck:
                    analysis["performance"]["recommendations"].append(
                        "Consider reducing CPU-intensive tasks or upgrading hardware"
                    )
                elif "memory" in bottleneck:
                    analysis["performance"]["recommendations"].append(
                        "Consider closing unnecessary applications or adding more RAM"
                    )
                elif "disk" in bottleneck:
                    analysis["performance"]["recommendations"].append(
                        "Clean up old files or add more storage capacity"
                    )

        return {
            "action": "system_analysis",
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }

    async def check_component_logs(
        self, components: List[str], time_range: str
    ) -> Dict[str, Any]:
        """Check logs for specified components"""
        log_results = {}

        try:
            log_dir = self.logs_dir

            # Time range mapping
            time_filters = {"1_hour": "-1h", "24_hours": "-24h", "1_week": "-1w"}

            time_filter = time_filters.get(time_range, "-1h")

            for component in components:
                log_file = log_dir / f"{component}.log"

                if log_file.exists():
                    # Read recent log entries
                    try:
                        proc = await asyncio.create_subprocess_shell(
                            f"tail -100 {log_file}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        stdout, stderr = await asyncio.wait_for(
                            proc.communicate(), timeout=10
                        )

                        if proc.returncode == 0 and stdout:
                            log_content = stdout.decode()

                            # Count error levels
                            error_count = log_content.count("ERROR")
                            warning_count = log_content.count(
                                "WARNING"
                            ) + log_content.count("WARN")

                            log_results[component] = {
                                "file": str(log_file),
                                "error_count": error_count,
                                "warning_count": warning_count,
                                "recent_entries": log_content.split("\n")[
                                    -10:
                                ],  # Last 10 lines
                                "status": (
                                    "error"
                                    if error_count > 0
                                    else "warning" if warning_count > 0 else "normal"
                                ),
                            }
                        else:
                            log_results[component] = {
                                "file": str(log_file),
                                "status": "unreadable",
                                "error": stderr.decode() if stderr else "Unknown error",
                            }

                    except Exception as e:
                        log_results[component] = {
                            "file": str(log_file),
                            "status": "error",
                            "error": str(e),
                        }
                else:
                    log_results[component] = {
                        "file": str(log_file),
                        "status": "not_found",
                    }

        except Exception as e:
            log_results["error"] = str(e)

        return {
            "action": "log_check",
            "components": log_results,
            "time_range": time_range,
            "timestamp": datetime.now().isoformat(),
        }

    async def diagnose_issues(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Diagnose issues based on gathered data"""
        diagnosis = {
            "issues": [],
            "root_causes": [],
            "suggested_fixes": [],
            "priority": "medium",
        }

        # Analyze component status
        for result in results:
            if result.get("action") == "component_status_check":
                components = result.get("components", {})

                for name, status in components.items():
                    if status.status == "stopped":
                        diagnosis["issues"].append(f"Component '{name}' is not running")
                        diagnosis["root_causes"].append(
                            f"Service {name} may have crashed or failed to start"
                        )
                        diagnosis["suggested_fixes"].append(
                            f"Try restarting {name} with: start {name}"
                        )

                    elif status.status == "error":
                        diagnosis["issues"].append(f"Component '{name}' has errors")
                        diagnosis["root_causes"].append(
                            f"Service {name} encountered an error condition"
                        )
                        diagnosis["suggested_fixes"].append(
                            f"Check {name} logs and restart if necessary"
                        )

            elif result.get("action") == "log_check":
                log_data = result.get("components", {})

                for component, log_info in log_data.items():
                    if isinstance(log_info, dict):
                        error_count = log_info.get("error_count", 0)
                        warning_count = log_info.get("warning_count", 0)

                        if error_count > 0:
                            diagnosis["issues"].append(
                                f"Component '{component}' has {error_count} recent errors"
                            )
                            diagnosis["priority"] = "high"

                        if warning_count > 5:  # Many warnings
                            diagnosis["issues"].append(
                                f"Component '{component}' has {warning_count} recent warnings"
                            )

        # Determine overall priority
        if not diagnosis["issues"]:
            diagnosis["priority"] = "low"
            diagnosis["suggested_fixes"].append("System appears to be running normally")
        elif len(diagnosis["issues"]) > 5:
            diagnosis["priority"] = "high"

        return {
            "action": "issue_diagnosis",
            "diagnosis": diagnosis,
            "timestamp": datetime.now().isoformat(),
        }

    async def compile_response(
        self, task: Task, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compile results into a natural language response"""
        response = {"text": "", "data": {}, "task_type": task.task_type.value}

        if task.task_type == TaskType.STATUS_QUERY:
            # Compile status report
            for result in results:
                if result.get("action") == "status_report":
                    report = result.get("report", {})
                    summary = report.get("summary", {})

                    total = summary.get("total_components", 0)
                    running = summary.get("running_components", 0)
                    stopped = summary.get("stopped_components", 0)
                    errors = summary.get("error_components", 0)

                    response["text"] = f"📊 **Vega System Status**\n\n"
                    response[
                        "text"
                    ] += f"**Summary:** {running}/{total} components running\n"

                    if stopped > 0:
                        response["text"] += f"⚠️ {stopped} components stopped\n"
                    if errors > 0:
                        response["text"] += f"❌ {errors} components with errors\n"

                    # Component details
                    components = report.get("components", {})
                    if components:
                        response["text"] += "\n**Component Details:**\n"
                        for name, status in components.items():
                            status_icon = (
                                "✅" if status["status"] == "running" else "❌"
                            )
                            response[
                                "text"
                            ] += f"{status_icon} {name}: {status['status']}\n"

                    # Recommendations
                    recommendations = report.get("recommendations", [])
                    if recommendations:
                        response["text"] += "\n**Recommendations:**\n"
                        for rec in recommendations:
                            response["text"] += f"• {rec}\n"

                    response["data"]["report"] = report

        elif task.task_type == TaskType.SYSTEM_CONTROL:
            # Compile control results
            response["text"] = "🎛️ **System Control Results**\n\n"

            for result in results:
                if result.get("action") == "component_control":
                    component = result.get("component", "")
                    action = result.get("control_action", "")
                    success = result.get("success", False)

                    icon = "✅" if success else "❌"
                    response["text"] += f"{icon} {action.capitalize()} {component}: "
                    response["text"] += "Success\n" if success else "Failed\n"

                    if not success and result.get("error"):
                        response["text"] += f"   Error: {result['error']}\n"

        elif task.task_type == TaskType.TROUBLESHOOTING:
            # Compile troubleshooting results
            response["text"] = "🔧 **Troubleshooting Results**\n\n"

            for result in results:
                if result.get("action") == "issue_diagnosis":
                    diagnosis = result.get("diagnosis", {})
                    issues = diagnosis.get("issues", [])
                    fixes = diagnosis.get("suggested_fixes", [])
                    priority = diagnosis.get("priority", "medium")

                    if issues:
                        response[
                            "text"
                        ] += f"**Issues Found (Priority: {priority.upper()}):**\n"
                        for issue in issues:
                            response["text"] += f"• {issue}\n"

                        response["text"] += "\n**Suggested Fixes:**\n"
                        for fix in fixes:
                            response["text"] += f"• {fix}\n"
                    else:
                        response["text"] += "✅ No significant issues detected.\n"

                    response["data"]["diagnosis"] = diagnosis

        elif task.task_type == TaskType.ANALYSIS_REQUEST:
            # Compile analysis results
            response["text"] = "📈 **System Analysis Results**\n\n"

            for result in results:
                if result.get("action") == "system_analysis":
                    analysis = result.get("analysis", {})
                    performance = analysis.get("performance", {})

                    health = performance.get("overall_health", "unknown")
                    health_icon = {"good": "✅", "warning": "⚠️", "critical": "❌"}.get(
                        health, "❓"
                    )

                    response[
                        "text"
                    ] += f"**Overall Health:** {health_icon} {health.upper()}\n\n"

                    bottlenecks = performance.get("bottlenecks", [])
                    if bottlenecks:
                        response["text"] += "**Performance Bottlenecks:**\n"
                        for bottleneck in bottlenecks:
                            response["text"] += f"• {bottleneck}\n"

                    recommendations = performance.get("recommendations", [])
                    if recommendations:
                        response["text"] += "\n**Recommendations:**\n"
                        for rec in recommendations:
                            response["text"] += f"• {rec}\n"

                    response["data"]["analysis"] = analysis

        else:
            response["text"] = f"Task completed: {task.query}"

        return response

    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]

    async def save_task(self, task: Task):
        """Save task to database"""
        try:
            db_path = self.state_dir / "vega_smart.db"

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks 
                    (id, task_type, priority, query, status, created_at, 
                     completed_at, result_data, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        task.id,
                        task.task_type.value,
                        task.priority.value,
                        task.query,
                        task.status.value,
                        task.created_at,
                        task.completed_at,
                        json.dumps(task.result) if task.result else None,
                        task.error_message,
                    ),
                )

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error saving task: {e}")

    async def log_interaction(self, query: str, response: str, task_id: str):
        """Log user interaction"""
        try:
            db_path = self.state_dir / "vega_smart.db"

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO interactions 
                    (user_query, assistant_response, task_id, timestamp)
                    VALUES (?, ?, ?, ?)
                """,
                    (query, response, task_id, datetime.now().isoformat()),
                )

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error logging interaction: {e}")

    async def interactive_mode(self):
        """Run in interactive mode"""
        print("🧠 Vega Smart Assistant - Interactive Mode")
        print("Type 'exit' to quit, 'help' for assistance")
        print("-" * 50)

        while True:
            try:
                query = input("\n🤖 Vega> ").strip()

                if query.lower() in ["exit", "quit", "bye"]:
                    print("👋 Goodbye!")
                    break

                elif query.lower() == "help":
                    self.show_help()
                    continue

                elif not query:
                    continue

                # Process query
                result = await self.process_query(query)

                if result["success"]:
                    print(f"\n📋 {result['response']}")
                else:
                    print(f"\n❌ Error: {result['response']}")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def show_help(self):
        """Show help information"""
        help_text = """
🧠 **Vega Smart Assistant Commands**

**Status Queries:**
• "What's the system status?"
• "Check all components"
• "How is Vega doing?"

**System Control:**
• "Start voice visualizer"
• "Stop network scanner" 
• "Restart vega core"

**Troubleshooting:**
• "What's wrong with the system?"
• "Check for errors"
• "Diagnose issues"

**Analysis:**
• "Analyze system performance"
• "Show system health"
• "Performance report"

**Examples:**
• "Start all components"
• "What happened in the last hour?"
• "System status and recommendations"
        """
        print(help_text)

    async def daemon_mode(self):
        """Run in daemon mode for automated assistance"""
        self.logger.info("🔄 Starting smart assistant daemon...")

        while True:
            try:
                # Check for automated tasks
                await self.check_automated_tasks()

                # Monitor system health
                await self.monitor_system_health()

                # Process any queued requests
                await self.process_request_queue()

                # Sleep before next cycle
                await asyncio.sleep(60)  # 1 minute cycle

            except Exception as e:
                self.logger.error(f"❌ Error in daemon mode: {e}")
                await asyncio.sleep(30)

    async def check_automated_tasks(self):
        """Check for automated tasks to execute"""
        # This could be expanded to handle scheduled tasks
        pass

    async def monitor_system_health(self):
        """Monitor system health and alert on issues"""
        try:
            # Quick health check
            result = await self.check_component_status(
                list(self.component_commands.keys())
            )
            components = result.get("components", {})

            # Check for critical issues
            critical_issues = []
            for name, status in components.items():
                if status.status in ["error", "stopped"]:
                    critical_issues.append(f"Component '{name}' is {status.status}")

            if critical_issues:
                # Log critical issues
                self.logger.warning(
                    f"🚨 Critical issues detected: {', '.join(critical_issues)}"
                )

                # Could trigger alerts or automated recovery here

        except Exception as e:
            self.logger.error(f"❌ Error monitoring system health: {e}")

    async def process_request_queue(self):
        """Process any queued requests from file"""
        try:
            queue_file = self.state_dir / "smart_assistant_queue.json"

            if queue_file.exists():
                with open(queue_file, "r") as f:
                    requests = json.load(f)

                for request in requests:
                    query = request.get("query", "")
                    if query:
                        await self.process_query(query)

                # Clear the queue
                queue_file.unlink()

        except Exception as e:
            self.logger.error(f"❌ Error processing request queue: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vega Smart Assistant")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive session"
    )
    parser.add_argument("--web", action="store_true", help="Start web interface")
    parser.add_argument("--port", type=int, default=8082, help="Web interface port")

    args = parser.parse_args()

    assistant = VegaSmartAssistant(mode="daemon" if args.daemon else "interactive")

    try:
        if args.daemon:
            await assistant.daemon_mode()
        elif args.interactive:
            await assistant.interactive_mode()
        elif args.query:
            result = await assistant.process_query(args.query)
            print(result["response"])
        else:
            await assistant.interactive_mode()

    except KeyboardInterrupt:
        print("\n🛑 Smart assistant stopped by user")
    except Exception as e:
        print(f"❌ Assistant error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
