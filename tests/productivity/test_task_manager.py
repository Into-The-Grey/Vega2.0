"""
Tests for Task Manager

Tests the AI-powered task management system including:
- Task CRUD operations
- Priority scoring and prediction
- Deadline prediction
- Task scheduling
- Workload optimization
"""

import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from src.vega.productivity.task_manager import (
    TaskManager,
    Task,
    TaskPriority,
    TaskStatus,
    TaskCategory,
    DeadlinePredictor,
    WorkloadOptimizer,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def task_manager(temp_storage):
    """Create TaskManager instance with temporary storage"""
    return TaskManager(storage_path=temp_storage)


class TestTaskCreation:
    """Test task creation and basic operations"""

    def test_create_simple_task(self, task_manager):
        """Test creating a basic task"""
        task = task_manager.create_task(
            title="Test task", description="Test description"
        )

        assert task.title == "Test task"
        assert task.description == "Test description"
        assert task.status == TaskStatus.NOT_STARTED
        assert task.priority == TaskPriority.MEDIUM
        assert task.category == TaskCategory.WORK
        assert task.id is not None
        assert task.created_at is not None

    def test_create_task_with_all_fields(self, task_manager):
        """Test creating task with all fields specified"""
        due_date = datetime.now() + timedelta(days=7)
        task = task_manager.create_task(
            title="Complex task",
            description="Detailed description",
            category=TaskCategory.PERSONAL,
            priority=TaskPriority.HIGH,
            due_date=due_date,
            tags=["urgent", "important"],
        )

        assert task.category == TaskCategory.PERSONAL
        assert task.priority == TaskPriority.HIGH
        assert task.due_date == due_date
        assert "urgent" in task.tags
        assert "important" in task.tags

    def test_create_multiple_tasks(self, task_manager):
        """Test creating multiple tasks"""
        task1 = task_manager.create_task(title="Task 1")
        task2 = task_manager.create_task(title="Task 2")
        task3 = task_manager.create_task(title="Task 3")

        assert len(task_manager.tasks) == 3
        assert task1.id != task2.id != task3.id


class TestTaskOperations:
    """Test task update and lifecycle operations"""

    def test_update_task_status(self, task_manager):
        """Test updating task status"""
        task = task_manager.create_task(title="Test task")
        initial_status = task.status

        updated = task_manager.update_task(task.id, status=TaskStatus.IN_PROGRESS)

        assert updated.status == TaskStatus.IN_PROGRESS
        assert updated.status != initial_status
        assert updated.updated_at > task.created_at

    def test_update_task_progress(self, task_manager):
        """Test updating task progress"""
        task = task_manager.create_task(title="Test task")

        updated = task_manager.update_task(task.id, progress=0.5)

        assert updated.progress == 0.5

    def test_complete_task(self, task_manager):
        """Test marking task as complete"""
        task = task_manager.create_task(title="Test task")

        task_manager.complete_task(task.id)
        completed = task_manager.get_task(task.id)

        assert completed.status == TaskStatus.COMPLETED
        assert completed.progress == 1.0
        assert completed.completed_at is not None

    def test_delete_task(self, task_manager):
        """Test deleting a task"""
        task = task_manager.create_task(title="Test task")
        task_id = task.id

        task_manager.delete_task(task_id)

        assert task_id not in [t.id for t in task_manager.tasks]
        assert task_manager.get_task(task_id) is None


class TestTaskFiltering:
    """Test task listing and filtering"""

    def test_list_all_tasks(self, task_manager):
        """Test listing all tasks"""
        task_manager.create_task(title="Task 1")
        task_manager.create_task(title="Task 2")
        task_manager.create_task(title="Task 3")

        tasks = task_manager.list_tasks()

        assert len(tasks) == 3

    def test_filter_by_status(self, task_manager):
        """Test filtering tasks by status"""
        task1 = task_manager.create_task(title="Task 1")
        task2 = task_manager.create_task(title="Task 2")
        task_manager.update_task(task2.id, status=TaskStatus.IN_PROGRESS)

        not_started = task_manager.list_tasks(status=TaskStatus.NOT_STARTED)
        in_progress = task_manager.list_tasks(status=TaskStatus.IN_PROGRESS)

        assert len(not_started) == 1
        assert len(in_progress) == 1
        assert not_started[0].id == task1.id
        assert in_progress[0].id == task2.id

    def test_filter_by_category(self, task_manager):
        """Test filtering tasks by category"""
        task_manager.create_task(title="Work task", category=TaskCategory.WORK)
        task_manager.create_task(title="Personal task", category=TaskCategory.PERSONAL)

        work_tasks = task_manager.list_tasks(category=TaskCategory.WORK)
        personal_tasks = task_manager.list_tasks(category=TaskCategory.PERSONAL)

        assert len(work_tasks) == 1
        assert len(personal_tasks) == 1

    def test_filter_by_priority(self, task_manager):
        """Test filtering tasks by priority"""
        task_manager.create_task(title="High priority", priority=TaskPriority.HIGH)
        task_manager.create_task(title="Low priority", priority=TaskPriority.LOW)

        high_tasks = task_manager.list_tasks(priority=TaskPriority.HIGH)
        low_tasks = task_manager.list_tasks(priority=TaskPriority.LOW)

        assert len(high_tasks) == 1
        assert len(low_tasks) == 1


class TestDeadlinePredictor:
    """Test deadline prediction functionality"""

    def test_predictor_initialization(self):
        """Test deadline predictor initialization"""
        predictor = DeadlinePredictor()
        assert predictor is not None

    def test_predict_duration_basic(self):
        """Test basic duration prediction"""
        predictor = DeadlinePredictor()

        duration = predictor.predict_duration(
            "Complete project documentation",
            category=TaskCategory.WORK,
            complexity_score=0.5,
        )

        assert duration > 0
        assert isinstance(duration, int)

    def test_predict_duration_high_complexity(self):
        """Test duration prediction for high complexity tasks"""
        predictor = DeadlinePredictor()

        simple_duration = predictor.predict_duration(
            "Simple task", category=TaskCategory.WORK, complexity_score=0.2
        )

        complex_duration = predictor.predict_duration(
            "Complex task", category=TaskCategory.WORK, complexity_score=0.9
        )

        # Higher complexity should predict longer duration
        assert complex_duration >= simple_duration


class TestWorkloadOptimizer:
    """Test workload optimization and priority scoring"""

    def test_optimizer_initialization(self):
        """Test workload optimizer initialization"""
        optimizer = WorkloadOptimizer()
        assert optimizer is not None

    def test_calculate_priority_score(self):
        """Test priority score calculation"""
        optimizer = WorkloadOptimizer()

        task = Task(
            title="Test task",
            priority=TaskPriority.HIGH,
            due_date=datetime.now() + timedelta(days=1),
            complexity_score=0.5,
        )

        score = optimizer.calculate_priority_score(task)

        assert 0.0 <= score <= 1.0

    def test_urgent_task_higher_score(self):
        """Test that urgent tasks get higher scores"""
        optimizer = WorkloadOptimizer()

        urgent_task = Task(
            title="Urgent",
            priority=TaskPriority.CRITICAL,
            due_date=datetime.now() + timedelta(hours=1),
        )

        normal_task = Task(
            title="Normal",
            priority=TaskPriority.MEDIUM,
            due_date=datetime.now() + timedelta(days=7),
        )

        urgent_score = optimizer.calculate_priority_score(urgent_task)
        normal_score = optimizer.calculate_priority_score(normal_task)

        assert urgent_score > normal_score


class TestTaskPrioritization:
    """Test task prioritization and scheduling"""

    def test_get_prioritized_tasks(self, task_manager):
        """Test getting prioritized task list"""
        # Create tasks with different priorities and due dates
        task_manager.create_task(
            title="Critical task",
            priority=TaskPriority.CRITICAL,
            due_date=datetime.now() + timedelta(hours=2),
        )
        task_manager.create_task(title="Normal task", priority=TaskPriority.MEDIUM)
        task_manager.create_task(title="Low priority", priority=TaskPriority.LOW)

        prioritized = task_manager.get_prioritized_tasks(limit=3)

        assert len(prioritized) <= 3
        # First task should have highest priority
        assert prioritized[0].title == "Critical task"

    def test_get_schedule(self, task_manager):
        """Test schedule generation"""
        # Create tasks with various due dates
        task_manager.create_task(
            title="Today task", due_date=datetime.now() + timedelta(hours=1)
        )
        task_manager.create_task(
            title="Tomorrow task", due_date=datetime.now() + timedelta(days=1)
        )

        schedule = task_manager.get_schedule(days=7)

        assert isinstance(schedule, dict)
        assert len(schedule) > 0


class TestTaskStatistics:
    """Test task statistics and reporting"""

    def test_get_basic_stats(self, task_manager):
        """Test basic statistics"""
        task_manager.create_task(title="Task 1")
        task_manager.create_task(title="Task 2")
        task2 = task_manager.create_task(title="Task 3")
        task_manager.complete_task(task2.id)

        stats = task_manager.get_stats()

        assert stats["total_tasks"] == 3
        assert stats["active_tasks"] == 2
        assert stats["completed_tasks"] == 1

    def test_stats_by_priority(self, task_manager):
        """Test statistics grouped by priority"""
        task_manager.create_task(title="High 1", priority=TaskPriority.HIGH)
        task_manager.create_task(title="High 2", priority=TaskPriority.HIGH)
        task_manager.create_task(title="Low 1", priority=TaskPriority.LOW)

        stats = task_manager.get_stats()

        assert stats["by_priority"]["high"] == 2
        assert stats["by_priority"]["low"] == 1

    def test_stats_by_category(self, task_manager):
        """Test statistics grouped by category"""
        task_manager.create_task(title="Work 1", category=TaskCategory.WORK)
        task_manager.create_task(title="Personal 1", category=TaskCategory.PERSONAL)

        stats = task_manager.get_stats()

        assert stats["by_category"]["work"] == 1
        assert stats["by_category"]["personal"] == 1

    def test_overdue_tasks_count(self, task_manager):
        """Test counting overdue tasks"""
        # Create overdue task
        task_manager.create_task(
            title="Overdue task", due_date=datetime.now() - timedelta(days=1)
        )

        stats = task_manager.get_stats()

        assert stats["overdue_tasks"] == 1


class TestTaskPersistence:
    """Test task persistence and data loading"""

    def test_save_and_load_tasks(self, temp_storage):
        """Test saving and loading tasks"""
        # Create manager and add tasks
        manager1 = TaskManager(storage_path=temp_storage)
        task1 = manager1.create_task(title="Task 1", description="Description 1")
        task2 = manager1.create_task(title="Task 2", priority=TaskPriority.HIGH)

        # Create new manager instance with same storage
        manager2 = TaskManager(storage_path=temp_storage)

        assert len(manager2.tasks) == 2
        loaded_task1 = manager2.get_task(task1.id)
        assert loaded_task1.title == "Task 1"
        assert loaded_task1.description == "Description 1"

    def test_task_updates_persist(self, temp_storage):
        """Test that task updates are persisted"""
        manager1 = TaskManager(storage_path=temp_storage)
        task = manager1.create_task(title="Test task")
        manager1.update_task(task.id, status=TaskStatus.IN_PROGRESS)

        manager2 = TaskManager(storage_path=temp_storage)
        loaded_task = manager2.get_task(task.id)

        assert loaded_task.status == TaskStatus.IN_PROGRESS


class TestComplexityScoring:
    """Test task complexity scoring"""

    def test_complexity_score_calculated(self, task_manager):
        """Test that complexity score is calculated"""
        task = task_manager.create_task(
            title="Complex task with many subtasks and dependencies",
            description="This is a very detailed description with multiple paragraphs",
        )

        assert task.complexity_score is not None
        assert 0.0 <= task.complexity_score <= 1.0

    def test_longer_description_higher_complexity(self, task_manager):
        """Test that longer descriptions result in higher complexity"""
        simple = task_manager.create_task(
            title="Simple", description="Short description"
        )

        complex_task = task_manager.create_task(
            title="Complex task",
            description="Very long detailed description " * 20,
        )

        # Complex task should have higher or equal complexity
        assert complex_task.complexity_score >= simple.complexity_score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
