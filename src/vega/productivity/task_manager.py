"""
Vega 2.0 Smart Task Management System

This module provides AI-powered task management with:
- Intelligent task prioritization using multiple factors
- Deadline prediction based on historical data and task complexity
- Workload optimization and resource allocation
- Task dependencies and scheduling
- Progress tracking and analytics
- Context-aware task recommendations

The system learns from user behavior and task completion patterns
to provide increasingly accurate predictions and recommendations.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
import json
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priority levels"""

    CRITICAL = "critical"  # Urgent and important
    HIGH = "high"  # Important but not urgent
    MEDIUM = "medium"  # Standard priority
    LOW = "low"  # Nice to have
    BACKLOG = "backlog"  # Future consideration


class TaskStatus(Enum):
    """Task completion status"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    WAITING = "waiting"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class TaskCategory(Enum):
    """Task categories for organization"""

    WORK = "work"
    PERSONAL = "personal"
    LEARNING = "learning"
    HEALTH = "health"
    FINANCE = "finance"
    CREATIVE = "creative"
    SOCIAL = "social"
    MAINTENANCE = "maintenance"
    RESEARCH = "research"
    DEVELOPMENT = "development"


@dataclass
class Task:
    """Represents a task in the system"""

    id: str
    title: str
    description: str
    category: TaskCategory
    priority: TaskPriority
    status: TaskStatus
    created_at: datetime
    due_date: Optional[datetime] = None
    estimated_duration: Optional[timedelta] = None
    actual_duration: Optional[timedelta] = None
    completed_at: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0  # 0.0 to 1.0
    complexity_score: Optional[float] = None  # Calculated complexity
    ai_priority_score: Optional[float] = None  # AI-calculated priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "estimated_duration": (
                self.estimated_duration.total_seconds()
                if self.estimated_duration
                else None
            ),
            "actual_duration": (
                self.actual_duration.total_seconds() if self.actual_duration else None
            ),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "dependencies": self.dependencies,
            "tags": self.tags,
            "metadata": self.metadata,
            "progress": self.progress,
            "complexity_score": self.complexity_score,
            "ai_priority_score": self.ai_priority_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create Task from dictionary"""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            category=TaskCategory(data["category"]),
            priority=TaskPriority(data["priority"]),
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            due_date=(
                datetime.fromisoformat(data["due_date"])
                if data.get("due_date")
                else None
            ),
            estimated_duration=(
                timedelta(seconds=data["estimated_duration"])
                if data.get("estimated_duration")
                else None
            ),
            actual_duration=(
                timedelta(seconds=data["actual_duration"])
                if data.get("actual_duration")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            progress=data.get("progress", 0.0),
            complexity_score=data.get("complexity_score"),
            ai_priority_score=data.get("ai_priority_score"),
        )


class DeadlinePredictor:
    """Predicts task completion deadlines based on historical data"""

    def __init__(self):
        self.historical_tasks: List[Task] = []
        self.category_stats: Dict[TaskCategory, Dict[str, float]] = {}

    def train(self, completed_tasks: List[Task]) -> None:
        """Train predictor on historical completed tasks"""
        self.historical_tasks = [
            t for t in completed_tasks if t.status == TaskStatus.COMPLETED
        ]

        # Calculate statistics per category
        for category in TaskCategory:
            category_tasks = [
                t for t in self.historical_tasks if t.category == category
            ]
            if not category_tasks:
                continue

            durations = [
                t.actual_duration.total_seconds()
                for t in category_tasks
                if t.actual_duration
            ]

            if durations:
                self.category_stats[category] = {
                    "mean_duration": np.mean(durations),
                    "std_duration": np.std(durations),
                    "min_duration": np.min(durations),
                    "max_duration": np.max(durations),
                }

        logger.info(f"Trained deadline predictor on {len(self.historical_tasks)} tasks")

    def predict_duration(self, task: Task) -> timedelta:
        """Predict how long a task will take"""
        # Use estimated duration if provided
        if task.estimated_duration:
            return task.estimated_duration

        # Fall back to category statistics
        if task.category in self.category_stats:
            stats = self.category_stats[task.category]
            base_duration = stats["mean_duration"]

            # Adjust for complexity if available
            if task.complexity_score:
                # Scale by complexity (0.5x to 2.0x)
                multiplier = 0.5 + (task.complexity_score * 1.5)
                base_duration *= multiplier

            return timedelta(seconds=base_duration)

        # Default estimate: 2 hours
        return timedelta(hours=2)

    def predict_deadline(
        self, task: Task, start_date: Optional[datetime] = None
    ) -> datetime:
        """Predict when a task will be completed"""
        if not start_date:
            start_date = datetime.now()

        predicted_duration = self.predict_duration(task)

        # Add buffer for dependencies
        if task.dependencies:
            buffer = timedelta(hours=len(task.dependencies) * 4)
            predicted_duration += buffer

        # Add buffer based on priority (lower priority gets more flexible deadline)
        priority_buffers = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 1.2,
            TaskPriority.MEDIUM: 1.5,
            TaskPriority.LOW: 2.0,
            TaskPriority.BACKLOG: 3.0,
        }
        buffer_multiplier = priority_buffers.get(task.priority, 1.5)
        predicted_duration *= buffer_multiplier

        return start_date + predicted_duration

    def calculate_complexity(self, task: Task) -> float:
        """Calculate task complexity score (0.0 to 1.0)"""
        complexity = 0.0

        # Factor 1: Description length (longer = more complex)
        desc_length = len(task.description)
        complexity += min(desc_length / 1000, 0.2)  # Max 0.2 contribution

        # Factor 2: Number of dependencies
        complexity += min(len(task.dependencies) * 0.1, 0.2)  # Max 0.2

        # Factor 3: Tags count (more tags = more aspects to consider)
        complexity += min(len(task.tags) * 0.05, 0.15)  # Max 0.15

        # Factor 4: Metadata richness
        complexity += min(len(task.metadata) * 0.03, 0.15)  # Max 0.15

        # Factor 5: Category-specific baseline
        category_complexity = {
            TaskCategory.RESEARCH: 0.7,
            TaskCategory.DEVELOPMENT: 0.65,
            TaskCategory.CREATIVE: 0.6,
            TaskCategory.LEARNING: 0.55,
            TaskCategory.WORK: 0.5,
            TaskCategory.FINANCE: 0.45,
            TaskCategory.HEALTH: 0.4,
            TaskCategory.PERSONAL: 0.35,
            TaskCategory.SOCIAL: 0.3,
            TaskCategory.MAINTENANCE: 0.25,
        }
        complexity += category_complexity.get(task.category, 0.5) * 0.3  # Max 0.3

        return min(complexity, 1.0)


class WorkloadOptimizer:
    """Optimizes task workload and provides scheduling recommendations"""

    def __init__(self, max_daily_hours: float = 8.0):
        self.max_daily_hours = max_daily_hours
        self.tasks: List[Task] = []

    def add_tasks(self, tasks: List[Task]) -> None:
        """Add tasks to optimizer"""
        self.tasks = tasks

    def calculate_priority_score(self, task: Task) -> float:
        """Calculate AI-driven priority score (0.0 to 1.0)"""
        score = 0.0

        # Factor 1: Manual priority (40% weight)
        priority_scores = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 0.75,
            TaskPriority.MEDIUM: 0.5,
            TaskPriority.LOW: 0.25,
            TaskPriority.BACKLOG: 0.1,
        }
        score += priority_scores.get(task.priority, 0.5) * 0.4

        # Factor 2: Deadline urgency (30% weight)
        if task.due_date:
            time_until_due = (task.due_date - datetime.now()).total_seconds()
            if time_until_due < 0:
                urgency = 1.0  # Overdue
            elif time_until_due < 86400:  # < 1 day
                urgency = 0.95
            elif time_until_due < 259200:  # < 3 days
                urgency = 0.8
            elif time_until_due < 604800:  # < 1 week
                urgency = 0.6
            else:
                urgency = 0.3
            score += urgency * 0.3
        else:
            score += 0.15  # No deadline = medium urgency

        # Factor 3: Dependencies (15% weight)
        # Tasks blocking others are higher priority
        blocking_count = sum(1 for t in self.tasks if task.id in t.dependencies)
        dependency_score = min(blocking_count * 0.3, 1.0)
        score += dependency_score * 0.15

        # Factor 4: Progress (15% weight)
        # Partially completed tasks get slight boost to finish them
        if 0 < task.progress < 1.0:
            score += (0.5 + task.progress * 0.5) * 0.15
        else:
            score += 0.075

        return min(score, 1.0)

    def get_prioritized_tasks(
        self, limit: Optional[int] = None, status: Optional[TaskStatus] = None
    ) -> List[Task]:
        """Get tasks sorted by priority score"""
        # Filter by status if specified
        tasks = self.tasks
        if status:
            tasks = [t for t in tasks if t.status == status]

        # Calculate and assign priority scores
        for task in tasks:
            task.ai_priority_score = self.calculate_priority_score(task)

        # Sort by priority score
        sorted_tasks = sorted(
            tasks, key=lambda t: t.ai_priority_score or 0.0, reverse=True
        )

        if limit:
            return sorted_tasks[:limit]
        return sorted_tasks

    def optimize_schedule(
        self, days: int = 7, predictor: Optional[DeadlinePredictor] = None
    ) -> Dict[str, List[Task]]:
        """Create optimized schedule for next N days"""
        if not predictor:
            predictor = DeadlinePredictor()

        schedule: Dict[str, List[Task]] = {}
        available_tasks = [
            t
            for t in self.tasks
            if t.status in [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS]
        ]

        # Sort by priority
        available_tasks = sorted(
            available_tasks,
            key=lambda t: self.calculate_priority_score(t),
            reverse=True,
        )

        current_date = datetime.now().date()
        daily_hours = {
            (current_date + timedelta(days=i)).isoformat(): 0.0 for i in range(days)
        }

        for task in available_tasks:
            # Predict duration
            duration = predictor.predict_duration(task)
            hours_needed = duration.total_seconds() / 3600

            # Find earliest day with capacity
            for day_offset in range(days):
                day = (current_date + timedelta(days=day_offset)).isoformat()

                if daily_hours[day] + hours_needed <= self.max_daily_hours:
                    if day not in schedule:
                        schedule[day] = []
                    schedule[day].append(task)
                    daily_hours[day] += hours_needed
                    break

        return schedule

    def get_workload_stats(self) -> Dict[str, Any]:
        """Get current workload statistics"""
        active_tasks = [
            t
            for t in self.tasks
            if t.status in [TaskStatus.NOT_STARTED, TaskStatus.IN_PROGRESS]
        ]

        return {
            "total_tasks": len(self.tasks),
            "active_tasks": len(active_tasks),
            "completed_tasks": len(
                [t for t in self.tasks if t.status == TaskStatus.COMPLETED]
            ),
            "blocked_tasks": len(
                [t for t in self.tasks if t.status == TaskStatus.BLOCKED]
            ),
            "by_priority": {
                priority.value: len([t for t in active_tasks if t.priority == priority])
                for priority in TaskPriority
            },
            "by_category": {
                category.value: len([t for t in active_tasks if t.category == category])
                for category in TaskCategory
            },
            "overdue_tasks": len(
                [t for t in active_tasks if t.due_date and t.due_date < datetime.now()]
            ),
        }


class TaskManager:
    """Main task management system coordinating all components"""

    def __init__(self, storage_path: Optional[Path] = None):
        if isinstance(storage_path, str):
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = storage_path or Path.home() / ".vega" / "tasks"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.tasks: Dict[str, Task] = {}
        self.predictor = DeadlinePredictor()
        self.optimizer = WorkloadOptimizer()

        self._load_tasks()

    def _load_tasks(self) -> None:
        """Load tasks from storage"""
        tasks_file = self.storage_path / "tasks.json"
        if tasks_file.exists():
            try:
                with open(tasks_file, "r") as f:
                    data = json.load(f)
                    self.tasks = {
                        task_id: Task.from_dict(task_data)
                        for task_id, task_data in data.items()
                    }
                logger.info(f"Loaded {len(self.tasks)} tasks from storage")
            except Exception as e:
                logger.error(f"Error loading tasks: {e}")

    def _save_tasks(self) -> None:
        """Save tasks to storage"""
        tasks_file = self.storage_path / "tasks.json"
        try:
            with open(tasks_file, "w") as f:
                json.dump(
                    {task_id: task.to_dict() for task_id, task in self.tasks.items()},
                    f,
                    indent=2,
                )
            logger.debug(f"Saved {len(self.tasks)} tasks to storage")
        except Exception as e:
            logger.error(f"Error saving tasks: {e}")

    def create_task(
        self,
        title: str,
        description: str = "",
        category: TaskCategory = TaskCategory.WORK,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: Optional[datetime] = None,
        estimated_duration: Optional[timedelta] = None,
        tags: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> Task:
        """Create a new task"""
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            category=category,
            priority=priority,
            status=TaskStatus.NOT_STARTED,
            created_at=datetime.now(),
            due_date=due_date,
            estimated_duration=estimated_duration,
            tags=tags or [],
            dependencies=dependencies or [],
        )

        # Calculate complexity
        task.complexity_score = self.predictor.calculate_complexity(task)

        # Predict deadline if not provided
        if not task.due_date:
            task.due_date = self.predictor.predict_deadline(task)

        self.tasks[task.id] = task
        self._save_tasks()
        self._update_optimizer()

        logger.info(f"Created task: {task.title} (ID: {task.id})")
        return task

    def update_task(self, task_id: str, updates: Dict[str, Any]) -> Optional[Task]:
        """Update an existing task"""
        if task_id not in self.tasks:
            logger.warning(f"Task not found: {task_id}")
            return None

        task = self.tasks[task_id]

        # Update fields
        for key, value in updates.items():
            if hasattr(task, key):
                # Handle enum conversions
                if key == "status" and isinstance(value, str):
                    value = TaskStatus(value)
                elif key == "priority" and isinstance(value, str):
                    value = TaskPriority(value)
                elif key == "category" and isinstance(value, str):
                    value = TaskCategory(value)

                setattr(task, key, value)

        # Mark as completed if status changed to completed
        if task.status == TaskStatus.COMPLETED and not task.completed_at:
            task.completed_at = datetime.now()
            task.progress = 1.0

        self._save_tasks()
        self._update_optimizer()

        logger.info(f"Updated task: {task.title}")
        return task

    def delete_task(self, task_id: str) -> bool:
        """Delete a task"""
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            self._save_tasks()
            self._update_optimizer()
            logger.info(f"Deleted task: {task.title}")
            return True
        return False

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID"""
        return self.tasks.get(task_id)

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        category: Optional[TaskCategory] = None,
        priority: Optional[TaskPriority] = None,
    ) -> List[Task]:
        """List tasks with optional filters"""
        tasks = list(self.tasks.values())

        if status:
            tasks = [t for t in tasks if t.status == status]
        if category:
            tasks = [t for t in tasks if t.category == category]
        if priority:
            tasks = [t for t in tasks if t.priority == priority]

        return tasks

    def get_prioritized_tasks(self, limit: int = 10) -> List[Task]:
        """Get top priority tasks"""
        return self.optimizer.get_prioritized_tasks(
            limit=limit, status=TaskStatus.NOT_STARTED
        ) + self.optimizer.get_prioritized_tasks(
            limit=limit, status=TaskStatus.IN_PROGRESS
        )

    def get_schedule(self, days: int = 7) -> Dict[str, List[Task]]:
        """Get optimized schedule for next N days"""
        return self.optimizer.optimize_schedule(days=days, predictor=self.predictor)

    def get_stats(self) -> Dict[str, Any]:
        """Get task management statistics"""
        return self.optimizer.get_workload_stats()

    def train_predictor(self) -> None:
        """Train the deadline predictor on completed tasks"""
        completed_tasks = [
            t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED
        ]
        if completed_tasks:
            self.predictor.train(completed_tasks)

    def _update_optimizer(self) -> None:
        """Update optimizer with current tasks"""
        self.optimizer.add_tasks(list(self.tasks.values()))


# Example usage demonstration
async def demo_task_manager():
    """Demonstrate task manager capabilities"""
    manager = TaskManager()

    # Create some tasks
    task1 = manager.create_task(
        title="Implement new feature",
        description="Add user authentication to the app with OAuth2 support",
        category=TaskCategory.DEVELOPMENT,
        priority=TaskPriority.HIGH,
        estimated_duration=timedelta(hours=8),
        tags=["backend", "security"],
    )

    task2 = manager.create_task(
        title="Write documentation",
        description="Document the new authentication system",
        category=TaskCategory.WORK,
        priority=TaskPriority.MEDIUM,
        dependencies=[task1.id],
        tags=["documentation"],
    )

    task3 = manager.create_task(
        title="Research ML algorithms",
        description="Study recent papers on federated learning optimization",
        category=TaskCategory.RESEARCH,
        priority=TaskPriority.LOW,
        estimated_duration=timedelta(hours=4),
        tags=["ml", "research"],
    )

    # Get prioritized tasks
    print("\n=== Top Priority Tasks ===")
    for task in manager.get_prioritized_tasks(limit=5):
        print(
            f"- {task.title} (Priority: {task.priority.value}, Score: {task.ai_priority_score:.2f})"
        )

    # Get schedule
    print("\n=== Optimized Schedule (Next 3 Days) ===")
    schedule = manager.get_schedule(days=3)
    for day, tasks in sorted(schedule.items()):
        print(f"\n{day}:")
        for task in tasks:
            print(f"  - {task.title}")

    # Get statistics
    print("\n=== Workload Statistics ===")
    stats = manager.get_stats()
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"Active tasks: {stats['active_tasks']}")
    print(f"Completed: {stats['completed_tasks']}")


if __name__ == "__main__":
    asyncio.run(demo_task_manager())
