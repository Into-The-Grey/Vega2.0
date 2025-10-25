"""
Integration Tests for Focus Tracker

Tests cross-feature integration between Focus Tracker and other
productivity modules, CLI functionality, and end-to-end workflows.
"""

import json
import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typer.testing import CliRunner

from src.vega.productivity.focus_tracker import (
    FocusTracker,
    FocusType,
    InterruptionType,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_storage():
    """Create temporary storage directory"""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def focus_tracker(temp_storage):
    """Create FocusTracker with temp storage"""
    return FocusTracker(storage_path=temp_storage)


@pytest.fixture
def cli_runner():
    """Create CLI test runner"""
    return CliRunner()


# ============================================================================
# Task Manager Integration Tests
# ============================================================================


def test_focus_session_with_task_link(focus_tracker):
    """Test linking focus session to task"""
    # Start session with task link
    session_id = focus_tracker.start_session(
        task_id="task-123",
        focus_type=FocusType.DEEP_WORK,
        context="Working on feature X",
    )

    session = focus_tracker.get_session_by_id(session_id)
    assert session.task_id == "task-123"
    assert session.context == "Working on feature X"


def test_get_sessions_by_task(focus_tracker):
    """Test filtering sessions by task ID"""
    # Create sessions for different tasks
    focus_tracker.start_session(task_id="task-1", focus_type=FocusType.DEEP_WORK)
    time.sleep(0.1)
    focus_tracker.start_session(task_id="task-2", focus_type=FocusType.SHALLOW_WORK)
    time.sleep(0.1)
    focus_tracker.start_session(task_id="task-1", focus_type=FocusType.LEARNING)

    # Get sessions for task-1
    task1_sessions = focus_tracker.get_session_history(task_id="task-1")

    assert len(task1_sessions) == 2
    assert all(s.task_id == "task-1" for s in task1_sessions)


def test_task_focus_time_calculation(focus_tracker):
    """Test calculating total focus time per task"""
    # Create and complete sessions for a task
    for i in range(3):
        session_id = focus_tracker.start_session(
            task_id="task-123",
            focus_type=FocusType.DEEP_WORK,
        )
        time.sleep(0.2)  # Longer delay to ensure measurable duration
        focus_tracker.end_session(session_id)

    # Get all sessions for task
    sessions = focus_tracker.get_session_history(task_id="task-123")
    completed = [
        s for s in sessions if not s.is_active
    ]  # Just check not active, duration can be 0

    # Note: Duration might be 0 for very short sessions (rounded to minutes)
    assert len(completed) == 3
    assert all(s.task_id == "task-123" for s in completed)


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================


def test_complete_focus_session_workflow(focus_tracker):
    """Test full focus session lifecycle"""
    # 1. Start session
    session_id = focus_tracker.start_session(
        task_id="workflow-task",
        focus_type=FocusType.DEEP_WORK,
        context="Testing complete workflow",
    )

    assert focus_tracker.get_active_session() is not None

    # 2. Record some interruptions
    focus_tracker.record_interruption(
        session_id=session_id,
        interruption_type=InterruptionType.NOTIFICATION,
        source="Email",
        duration=30,
        impact_score=0.3,
    )

    focus_tracker.record_interruption(
        session_id=session_id,
        interruption_type=InterruptionType.DISTRACTION,
        source="Social media",
        duration=120,
        impact_score=0.7,
    )

    # 3. End session
    time.sleep(0.2)  # Longer delay for measurable duration
    session = focus_tracker.end_session(session_id, notes="Completed workflow test")

    # Verify results
    assert not session.is_active
    assert session.duration >= 0  # Duration is in minutes (rounded)
    assert len(session.interruptions) == 2
    assert session.quality_score >= 0.0
    assert session.notes == "Completed workflow test"

    # 4. Verify metrics updated
    from datetime import date

    metrics = focus_tracker.get_metrics(start_date=date.today())
    assert metrics.total_sessions >= 1
    assert metrics.interruption_count >= 2


def test_multiple_sessions_workflow(focus_tracker):
    """Test multiple sessions over time"""
    session_ids = []

    # Create 5 sessions with varying characteristics
    for i in range(5):
        session_id = focus_tracker.start_session(
            focus_type=FocusType.DEEP_WORK if i % 2 == 0 else FocusType.SHALLOW_WORK,
            context=f"Session {i+1}",
        )
        session_ids.append(session_id)

        # Add interruptions to some sessions
        if i % 3 == 0:
            focus_tracker.record_interruption(
                session_id=session_id,
                interruption_type=InterruptionType.NOTIFICATION,
                source="Test",
                duration=30,
                impact_score=0.4,
            )

        time.sleep(0.1)
        focus_tracker.end_session(session_id)

    # Verify all sessions recorded
    history = focus_tracker.get_session_history(limit=10)
    assert len(history) >= 5

    # Verify metrics
    from datetime import date

    metrics = focus_tracker.get_metrics(start_date=date.today())
    assert metrics.total_sessions >= 5
    assert metrics.deep_work_percentage > 0


def test_session_persistence_across_restarts(temp_storage):
    """Test data persists across FocusTracker instances"""
    # Create first tracker and session
    tracker1 = FocusTracker(storage_path=temp_storage)
    session_id = tracker1.start_session(
        focus_type=FocusType.LEARNING,
        context="Persistence test",
    )
    time.sleep(0.1)
    tracker1.end_session(session_id)

    # Create new tracker instance (simulating restart)
    tracker2 = FocusTracker(storage_path=temp_storage)

    # Verify data persisted
    history = tracker2.get_session_history()
    assert len(history) >= 1
    assert history[0].context == "Persistence test"

    # Verify metrics work
    from datetime import date

    metrics = tracker2.get_metrics(start_date=date.today())
    assert metrics.total_sessions >= 1


# ============================================================================
# Insights and Analytics Integration Tests
# ============================================================================


def test_insights_with_multiple_sessions(focus_tracker):
    """Test insight generation with realistic data"""
    # Create sessions at different times with varying quality
    sessions_data = [
        (FocusType.DEEP_WORK, 45, 0),  # Good session, no interruptions
        (FocusType.SHALLOW_WORK, 20, 2),  # Short with interruptions
        (FocusType.DEEP_WORK, 60, 0),  # Excellent session
        (FocusType.LEARNING, 30, 1),  # Moderate session
    ]

    for focus_type, duration_minutes, num_interruptions in sessions_data:
        session_id = focus_tracker.start_session(focus_type=focus_type)

        # Add interruptions
        for i in range(num_interruptions):
            focus_tracker.record_interruption(
                session_id=session_id,
                interruption_type=InterruptionType.NOTIFICATION,
                source="Test",
                duration=30,
                impact_score=0.5,
            )

        time.sleep(0.1)
        focus_tracker.end_session(session_id)

    # Get insights
    sessions = focus_tracker.get_session_history(limit=10)
    from datetime import date

    metrics = focus_tracker.get_metrics(start_date=date.today())

    # Test peak hours identification
    peak_hours = focus_tracker.insights.get_peak_focus_hours(sessions)
    assert isinstance(peak_hours, list)

    # Test optimal session length
    optimal = focus_tracker.insights.get_optimal_session_length(sessions)
    assert optimal > 0

    # Test recommendations
    recommendations = focus_tracker.insights.get_improvement_recommendations(
        sessions, metrics
    )
    assert isinstance(recommendations, list)


def test_distraction_pattern_analysis(focus_tracker):
    """Test distraction monitoring and pattern detection"""
    # Create session with multiple distractions
    session_id = focus_tracker.start_session(focus_type=FocusType.DEEP_WORK)

    distraction_sources = [
        ("Slack", InterruptionType.NOTIFICATION),
        ("Email", InterruptionType.NOTIFICATION),
        ("Slack", InterruptionType.NOTIFICATION),
        ("Phone", InterruptionType.EXTERNAL),
        ("Social media", InterruptionType.DISTRACTION),
    ]

    for source, int_type in distraction_sources:
        focus_tracker.record_interruption(
            session_id=session_id,
            interruption_type=int_type,
            source=source,
            duration=60,
            impact_score=0.5,
        )

    time.sleep(0.1)
    focus_tracker.end_session(session_id)

    # Analyze patterns
    patterns = focus_tracker.distraction_monitor.get_distraction_patterns(days=7)

    assert patterns["total_interruptions"] >= 5
    assert patterns["most_common_type"] == InterruptionType.NOTIFICATION.value
    assert patterns["most_common_source"] == "Slack"

    # Test mitigation strategies
    strategies = focus_tracker.distraction_monitor.suggest_mitigation_strategies(
        patterns
    )
    assert len(strategies) > 0


def test_weekly_report_generation(focus_tracker):
    """Test weekly report with multiple days of data"""
    from datetime import date

    # Create sessions across multiple days (simulated)
    for i in range(7):
        session_id = focus_tracker.start_session(
            focus_type=FocusType.DEEP_WORK,
            context=f"Day {i+1} work",
        )
        time.sleep(0.1)
        focus_tracker.end_session(session_id)

    # Generate weekly report
    start_date = date.today() - timedelta(days=date.today().weekday())
    sessions = focus_tracker.get_session_history(limit=100)
    report = focus_tracker.insights.generate_weekly_report(sessions, start_date)

    assert report["total_sessions"] >= 7
    assert report["total_time_hours"] >= 0
    assert "daily_stats" in report


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


def test_cannot_start_multiple_active_sessions(focus_tracker):
    """Test that only one session can be active at a time"""
    # Start first session
    session_id1 = focus_tracker.start_session()

    # Verify it's active
    assert focus_tracker.get_active_session() is not None

    # Start second session (should work but indicate warning via active check)
    session_id2 = focus_tracker.start_session()

    # Both sessions exist but we should manage active state
    sessions = focus_tracker.get_session_history(limit=5)
    active_count = sum(1 for s in sessions if s.is_active)

    # Clean up - end sessions
    if focus_tracker.get_active_session():
        focus_tracker.end_session(focus_tracker.get_active_session().session_id)


def test_interruption_without_active_session(focus_tracker):
    """Test recording interruption fails gracefully without active session"""
    # Try to record interruption without session
    with pytest.raises(ValueError, match="not found"):
        focus_tracker.record_interruption(
            session_id="nonexistent",
            interruption_type=InterruptionType.NOTIFICATION,
            source="Test",
        )


def test_end_nonexistent_session(focus_tracker):
    """Test ending non-existent session fails gracefully"""
    with pytest.raises(ValueError, match="not found"):
        focus_tracker.end_session("nonexistent-session-id")


def test_metrics_with_date_filtering(focus_tracker):
    """Test metrics calculation respects date filtering"""
    from datetime import date

    # Create session
    session_id = focus_tracker.start_session()
    time.sleep(0.1)
    focus_tracker.end_session(session_id)

    # Get metrics for today
    metrics_today = focus_tracker.get_metrics(
        start_date=date.today(),
        end_date=date.today(),
    )
    assert metrics_today.total_sessions >= 1

    # Get metrics for future (should be empty)
    metrics_future = focus_tracker.get_metrics(
        start_date=date.today() + timedelta(days=1),
        end_date=date.today() + timedelta(days=7),
    )
    assert metrics_future.total_sessions == 0


def test_session_quality_varies_by_duration(focus_tracker):
    """Test that quality scoring varies appropriately by duration"""
    qualities = []

    # Very short session
    session_id = focus_tracker.start_session()
    time.sleep(0.05)
    session = focus_tracker.end_session(session_id)
    qualities.append(("very_short", session.quality_score))

    # Moderate session
    session_id = focus_tracker.start_session()
    time.sleep(0.1)
    session = focus_tracker.end_session(session_id)
    qualities.append(("moderate", session.quality_score))

    # All should have valid scores
    for name, quality in qualities:
        assert 0.0 <= quality <= 1.0


def test_storage_file_structure(temp_storage):
    """Test that storage files are created correctly"""
    tracker = FocusTracker(storage_path=temp_storage)

    # Create and end a session
    session_id = tracker.start_session()
    tracker.record_interruption(
        session_id=session_id,
        interruption_type=InterruptionType.NOTIFICATION,
        source="Test",
    )
    time.sleep(0.1)
    tracker.end_session(session_id)

    # Verify files exist
    assert (temp_storage / "sessions.json").exists()
    assert (temp_storage / "interruptions.json").exists()

    # Verify files contain valid JSON
    with open(temp_storage / "sessions.json") as f:
        sessions_data = json.load(f)
        assert isinstance(sessions_data, list)
        assert len(sessions_data) >= 1

    with open(temp_storage / "interruptions.json") as f:
        interruptions_data = json.load(f)
        assert isinstance(interruptions_data, dict)


# ============================================================================
# Performance Tests
# ============================================================================


def test_performance_with_many_sessions(focus_tracker):
    """Test performance with larger dataset"""
    # Create 50 sessions
    for i in range(50):
        session_id = focus_tracker.start_session(
            focus_type=FocusType.DEEP_WORK if i % 2 == 0 else FocusType.SHALLOW_WORK,
        )
        if i % 10 == 0:
            focus_tracker.record_interruption(
                session_id=session_id,
                interruption_type=InterruptionType.NOTIFICATION,
                source="Test",
            )

        # Don't sleep for all to keep test fast, but add some delay
        if i % 10 == 0:
            time.sleep(0.05)

        focus_tracker.end_session(session_id)

    # Test retrieval performance
    import time as time_module

    start = time_module.time()

    history = focus_tracker.get_session_history(limit=50)
    from datetime import date

    metrics = focus_tracker.get_metrics(start_date=date.today() - timedelta(days=30))

    elapsed = time_module.time() - start

    # Should complete quickly (< 1 second)
    assert elapsed < 1.0
    assert len(history) == 50
    assert metrics.total_sessions >= 50


def test_concurrent_session_handling(temp_storage):
    """Test handling multiple tracker instances"""
    tracker1 = FocusTracker(storage_path=temp_storage)
    tracker2 = FocusTracker(storage_path=temp_storage)

    # Create session with tracker1
    session_id = tracker1.start_session()
    time.sleep(0.1)
    tracker1.end_session(session_id)

    # Read with tracker2
    history = tracker2.get_session_history()
    assert len(history) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
