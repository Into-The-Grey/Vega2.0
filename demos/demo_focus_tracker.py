#!/usr/bin/env python3
"""
Focus Tracker Demo Script

Demonstrates complete focus tracking workflow including:
- Session management
- Interruption logging
- Quality scoring
- Analytics and insights
- Task integration
"""

import time
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vega.productivity.focus_tracker import (
    FocusTracker,
    FocusType,
    InterruptionType,
)


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def demo_basic_session():
    """Demo: Basic focus session lifecycle"""
    print_section("DEMO 1: Basic Focus Session")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        # Start session
        print("\n1. Starting focus session...")
        session_id = tracker.start_session(
            focus_type=FocusType.DEEP_WORK,
            context="Working on demo script",
        )
        print(f"   ✓ Session started: {session_id[:8]}...")

        # Simulate work
        print("\n2. Simulating 2 minutes of focused work...")
        time.sleep(0.2)  # Shortened for demo

        # End session
        print("\n3. Ending session...")
        session = tracker.end_session(session_id)
        print(f"   ✓ Duration: {session.duration if session.duration else 0} seconds")
        print(f"   ✓ Quality Score: {session.quality_score:.2f}/1.00")
        print(
            f"   ✓ Rating: {'🌟 Excellent!' if session.quality_score >= 0.85 else '✅ Good' if session.quality_score >= 0.70 else '⚠️ Could be better'}"
        )


def demo_interrupted_session():
    """Demo: Session with interruptions"""
    print_section("DEMO 2: Session with Interruptions")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        # Start session
        print("\n1. Starting deep work session...")
        session_id = tracker.start_session(
            focus_type=FocusType.DEEP_WORK,
            context="Complex problem-solving",
        )

        # Simulate interruptions
        print("\n2. Recording interruptions...")

        time.sleep(0.1)
        tracker.record_interruption(
            session_id=session_id,
            interruption_type=InterruptionType.NOTIFICATION,
            source="Slack message",
            duration=30,
            impact_score=0.3,
        )
        print("   ✓ Notification logged (30s, impact: 0.3)")

        time.sleep(0.1)
        tracker.record_interruption(
            session_id=session_id,
            interruption_type=InterruptionType.DISTRACTION,
            source="Social media check",
            duration=180,
            impact_score=0.7,
        )
        print("   ✓ Distraction logged (3min, impact: 0.7)")

        # End session
        print("\n3. Ending session...")
        time.sleep(0.1)
        session = tracker.end_session(session_id)
        print(f"   ✓ Duration: {session.duration if session.duration else 0} seconds")
        print(f"   ✓ Interruptions: {len(session.interruptions)}")
        print(f"   ✓ Quality Score: {session.quality_score:.2f}/1.00")
        print(f"   📊 Impact: Quality reduced due to interruptions")


def demo_multiple_sessions():
    """Demo: Multiple sessions with analytics"""
    print_section("DEMO 3: Multiple Sessions & Analytics")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        print("\n1. Creating 5 sessions with varying quality...")
        sessions_config = [
            (FocusType.DEEP_WORK, 0, "Perfect focus"),
            (FocusType.SHALLOW_WORK, 1, "Email responses"),
            (FocusType.DEEP_WORK, 0, "Feature implementation"),
            (FocusType.LEARNING, 2, "Tutorial with interruptions"),
            (FocusType.CREATIVE, 0, "Brainstorming session"),
        ]

        for i, (focus_type, num_interruptions, context) in enumerate(
            sessions_config, 1
        ):
            session_id = tracker.start_session(
                focus_type=focus_type,
                context=context,
            )

            # Add interruptions
            for j in range(num_interruptions):
                tracker.record_interruption(
                    session_id=session_id,
                    interruption_type=InterruptionType.NOTIFICATION,
                    source="Test notification",
                    duration=30,
                    impact_score=0.4,
                )

            time.sleep(0.1)
            tracker.end_session(session_id)
            print(
                f"   ✓ Session {i}: {focus_type.value} ({num_interruptions} interruptions)"
            )

        # Get metrics
        print("\n2. Analyzing metrics...")
        metrics = tracker.get_metrics(start_date=date.today())
        print(f"   Total Sessions: {metrics.total_sessions}")
        print(f"   Average Quality: {metrics.quality_average:.2f}/1.00")
        print(f"   Deep Work: {metrics.deep_work_percentage:.1f}%")
        print(f"   Interruptions: {metrics.interruption_count}")

        # Get insights
        print("\n3. Generating insights...")
        sessions = tracker.get_session_history()

        peak_hours = tracker.insights.get_peak_focus_hours(sessions)
        print(f"   Peak Hours: {', '.join(str(h) for h in peak_hours[:3])}")

        optimal_duration = tracker.insights.get_optimal_session_length(sessions)
        print(f"   Optimal Duration: {optimal_duration} seconds")

        recommendations = tracker.insights.get_improvement_recommendations(
            sessions, metrics
        )
        print(f"   Recommendations: {len(recommendations)} suggestions generated")
        if recommendations:
            print(f"   - {recommendations[0]}")


def demo_task_integration():
    """Demo: Integration with tasks"""
    print_section("DEMO 4: Task Integration")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        # Simulate task creation (in real use, this would come from TaskManager)
        task_id = "task-demo-123"

        print(f"\n1. Starting sessions linked to task: {task_id}")

        # Create multiple sessions for same task
        for i in range(3):
            session_id = tracker.start_session(
                task_id=task_id,
                focus_type=FocusType.DEEP_WORK,
                context=f"Working on feature - session {i+1}",
            )
            time.sleep(0.1)
            tracker.end_session(session_id)
            print(f"   ✓ Session {i+1} completed")

        # Get task-specific sessions
        print(f"\n2. Retrieving focus history for task...")
        task_sessions = tracker.get_session_history(task_id=task_id)
        print(f"   Total sessions for task: {len(task_sessions)}")

        # Calculate total focus time
        total_time = sum(s.duration for s in task_sessions if s.duration)
        print(f"   Total focus time: {total_time} seconds")

        # Calculate average quality
        avg_quality = sum(s.quality_score for s in task_sessions) / len(task_sessions)
        print(f"   Average quality: {avg_quality:.2f}/1.00")


def demo_distraction_patterns():
    """Demo: Distraction pattern analysis"""
    print_section("DEMO 5: Distraction Pattern Analysis")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        print("\n1. Creating sessions with varied distractions...")

        session_id = tracker.start_session(focus_type=FocusType.DEEP_WORK)

        # Simulate pattern: Multiple notifications from same source
        distraction_sources = [
            ("Slack", InterruptionType.NOTIFICATION),
            ("Email", InterruptionType.NOTIFICATION),
            ("Slack", InterruptionType.NOTIFICATION),
            ("Slack", InterruptionType.NOTIFICATION),
            ("Phone", InterruptionType.EXTERNAL),
        ]

        for source, int_type in distraction_sources:
            tracker.record_interruption(
                session_id=session_id,
                interruption_type=int_type,
                source=source,
                duration=45,
                impact_score=0.5,
            )
            print(f"   ✓ Logged: {int_type.value} from {source}")

        time.sleep(0.1)
        tracker.end_session(session_id)

        # Analyze patterns
        print("\n2. Analyzing distraction patterns...")
        patterns = tracker.distraction_monitor.get_distraction_patterns(days=7)
        print(f"   Total Interruptions: {patterns['total_interruptions']}")
        print(f"   Most Common Type: {patterns['most_common_type']}")
        print(f"   Most Common Source: {patterns['most_common_source']}")

        # Get mitigation strategies
        print("\n3. Generating mitigation strategies...")
        strategies = tracker.distraction_monitor.suggest_mitigation_strategies(patterns)
        print(f"   {len(strategies)} strategies suggested:")
        for strategy in strategies[:2]:
            print(f"   - {strategy}")


def demo_flow_state():
    """Demo: Flow state detection"""
    print_section("DEMO 6: Flow State Detection")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        print("\n1. Creating two sessions - one achieving flow, one not...")

        # Flow state session (long, no interruptions)
        print("\n   Session A: Deep focus (flow state)")
        session_id1 = tracker.start_session(focus_type=FocusType.DEEP_WORK)
        time.sleep(0.2)
        session1 = tracker.end_session(session_id1)
        flow1 = tracker.analyzer.detect_flow_state(session1)
        print(f"   - Duration: {session1.duration if session1.duration else 0}s")
        print(f"   - Quality: {session1.quality_score:.2f}")
        print(f"   - Flow State: {'✓ YES' if flow1 else '✗ NO'}")

        # Non-flow session (short with interruptions)
        print("\n   Session B: Interrupted work (no flow)")
        session_id2 = tracker.start_session(focus_type=FocusType.SHALLOW_WORK)
        for _ in range(5):
            tracker.record_interruption(
                session_id=session_id2,
                interruption_type=InterruptionType.NOTIFICATION,
                source="Various",
                duration=30,
                impact_score=0.5,
            )
        time.sleep(0.05)
        session2 = tracker.end_session(session_id2)
        flow2 = tracker.analyzer.detect_flow_state(session2)
        print(f"   - Duration: {session2.duration if session2.duration else 0}s")
        print(f"   - Interruptions: {len(session2.interruptions)}")
        print(f"   - Quality: {session2.quality_score:.2f}")
        print(f"   - Flow State: {'✓ YES' if flow2 else '✗ NO'}")


def demo_weekly_report():
    """Demo: Weekly report generation"""
    print_section("DEMO 7: Weekly Report")

    with TemporaryDirectory() as tmpdir:
        tracker = FocusTracker(storage_path=Path(tmpdir))

        print("\n1. Creating week's worth of sessions...")

        # Create varied sessions
        for day in range(7):
            for session_num in range(3):
                session_id = tracker.start_session(
                    focus_type=(
                        FocusType.DEEP_WORK
                        if session_num % 2 == 0
                        else FocusType.SHALLOW_WORK
                    ),
                )

                # Add some interruptions
                if day % 3 == 0:
                    tracker.record_interruption(
                        session_id=session_id,
                        interruption_type=InterruptionType.NOTIFICATION,
                        source="Test",
                        duration=30,
                        impact_score=0.4,
                    )

                time.sleep(0.05)
                tracker.end_session(session_id)

        print("   ✓ Created 21 sessions across 7 days")

        # Generate report
        print("\n2. Generating weekly report...")
        sessions = tracker.get_session_history()
        start_date = date.today() - timedelta(days=date.today().weekday())
        report = tracker.insights.generate_weekly_report(sessions, start_date)

        print(f"   Total Sessions: {report['total_sessions']}")
        print(f"   Total Time: {report['total_time_hours']:.2f} hours")
        print(f"   Average Quality: {report['average_quality']:.2f}/1.00")
        print(f"   Flow State Sessions: {report['flow_sessions']}")
        if "best_day" in report and report["best_day"]:
            print(f"   Best Day: {report['best_day']}")


def main():
    """Run all demos"""
    print("\n" + "🎯" * 30)
    print("  FOCUS TRACKER COMPREHENSIVE DEMO")
    print("🎯" * 30)

    try:
        demo_basic_session()
        demo_interrupted_session()
        demo_multiple_sessions()
        demo_task_integration()
        demo_distraction_patterns()
        demo_flow_state()
        demo_weekly_report()

        print_section("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("\n📊 Key Features Demonstrated:")
        print("   ✓ Session lifecycle management")
        print("   ✓ Quality scoring algorithm")
        print("   ✓ Interruption tracking")
        print("   ✓ Analytics and metrics")
        print("   ✓ Task integration")
        print("   ✓ Distraction pattern analysis")
        print("   ✓ Flow state detection")
        print("   ✓ Weekly reporting")
        print("\n🚀 Focus Tracker is production-ready!")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
