"""
Focus & Attention Tracking System

Tracks deep work sessions, monitors distractions, analyzes productivity patterns,
and provides AI-powered insights for optimizing focus and attention.

Core Components:
- FocusSession: Individual work session tracking
- Interruption: Distraction and interruption logging
- FocusAnalyzer: AI-powered session quality analysis
- DistractionMonitor: Pattern detection and mitigation strategies
- ProductivityInsights: Actionable recommendations engine
- FocusTracker: Main orchestrator class

Storage: JSON-based persistence in ~/.vega/focus/
Integration: Task Manager for task-linked sessions
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional dependencies with graceful fallback
try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Enums
# ============================================================================


class FocusType(str, Enum):
    """Types of focus sessions"""

    DEEP_WORK = "deep_work"
    SHALLOW_WORK = "shallow_work"
    BREAK = "break"
    MEETING = "meeting"
    LEARNING = "learning"
    CREATIVE = "creative"


class InterruptionType(str, Enum):
    """Categories of interruptions"""

    NOTIFICATION = "notification"
    DISTRACTION = "distraction"
    BREAK = "break"
    EXTERNAL = "external"
    CONTEXT_SWITCH = "context_switch"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Interruption:
    """Records focus interruptions"""

    interruption_id: str
    timestamp: datetime
    interruption_type: InterruptionType
    duration: int  # seconds
    source: str
    impact_score: float  # 0.0-1.0
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["interruption_type"] = self.interruption_type.value
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Interruption:
        """Create from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["interruption_type"] = InterruptionType(data["interruption_type"])
        return Interruption(**data)


@dataclass
class FocusSession:
    """Represents a focused work session"""

    session_id: str
    task_id: Optional[str]  # Link to TaskManager
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[int]  # seconds
    session_type: FocusType
    quality_score: float  # 0.0-1.0 AI-scored
    interruptions: List[Interruption] = field(default_factory=list)
    context: str = ""
    notes: Optional[str] = None
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat() if self.end_time else None
        data["session_type"] = self.session_type.value
        data["interruptions"] = [i.to_dict() for i in self.interruptions]
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> FocusSession:
        """Create from dictionary"""
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        data["session_type"] = FocusType(data["session_type"])
        data["interruptions"] = [
            Interruption.from_dict(i) for i in data.get("interruptions", [])
        ]
        return FocusSession(**data)


@dataclass
class FocusMetrics:
    """Aggregated focus analytics"""

    total_sessions: int
    total_focus_time: int  # seconds
    average_session_duration: int  # seconds
    quality_average: float  # 0.0-1.0
    interruption_count: int
    deep_work_percentage: float  # 0.0-100.0
    productivity_trend: List[float]  # Last 7 days quality scores
    peak_focus_hours: List[int]  # Hours of day (0-23)
    best_day: Optional[str] = None
    improvement_rate: float = 0.0  # Percentage change


# ============================================================================
# FocusAnalyzer: AI-powered session quality analysis
# ============================================================================


class FocusAnalyzer:
    """AI-powered focus quality analysis"""

    def __init__(self):
        self.flow_state_threshold = 0.8  # Quality score threshold for flow state

    def analyze_session(self, session: FocusSession) -> float:
        """
        Analyze session quality and return score (0.0-1.0)

        Factors:
        - Session duration (optimal: 25-90 minutes)
        - Interruption count and impact
        - Time of day
        - Session type
        """
        if not session.duration or session.duration <= 0:
            return 0.0

        # Duration score (optimal: 25-90 minutes)
        duration_minutes = session.duration / 60
        if duration_minutes < 5:
            duration_score = 0.2
        elif 25 <= duration_minutes <= 90:
            duration_score = 1.0
        elif duration_minutes < 25:
            duration_score = 0.5 + (duration_minutes / 50)
        else:
            duration_score = max(0.3, 1.0 - ((duration_minutes - 90) / 180))

        # Interruption penalty
        interruption_penalty = sum(i.impact_score for i in session.interruptions) * 0.1
        interruption_score = max(0.0, 1.0 - interruption_penalty)

        # Time of day bonus (peak hours: 9-11am, 2-4pm)
        hour = session.start_time.hour
        if hour in [9, 10, 14, 15]:
            time_bonus = 0.1
        elif hour in [8, 11, 13, 16]:
            time_bonus = 0.05
        else:
            time_bonus = 0.0

        # Session type multiplier
        type_multipliers = {
            FocusType.DEEP_WORK: 1.2,
            FocusType.CREATIVE: 1.15,
            FocusType.LEARNING: 1.1,
            FocusType.SHALLOW_WORK: 0.9,
            FocusType.MEETING: 0.7,
            FocusType.BREAK: 0.5,
        }
        type_multiplier = type_multipliers.get(session.session_type, 1.0)

        # Calculate final score
        base_score = (duration_score * 0.5 + interruption_score * 0.5) * type_multiplier
        final_score = min(1.0, base_score + time_bonus)

        return round(final_score, 3)

    def detect_flow_state(self, session: FocusSession) -> bool:
        """Detect if session achieved flow state"""
        quality = self.analyze_session(session)

        # Flow state criteria:
        # - High quality score
        # - Minimal interruptions
        # - Adequate duration
        has_high_quality = quality >= self.flow_state_threshold
        has_few_interruptions = len(session.interruptions) <= 2
        has_good_duration = session.duration and session.duration >= 1200  # 20+ minutes

        return has_high_quality and has_few_interruptions and has_good_duration

    def predict_optimal_duration(
        self, task_type: str, historical_sessions: List[FocusSession]
    ) -> int:
        """Predict optimal session duration for task type (returns seconds)"""
        # Filter sessions by similar context/type
        relevant_sessions = [
            s
            for s in historical_sessions
            if s.duration
            and s.quality_score >= 0.7
            and (task_type.lower() in s.context.lower() if s.context else False)
        ]

        if not relevant_sessions:
            # Default recommendations by session type
            defaults = {
                "deep_work": 90 * 60,
                "creative": 60 * 60,
                "learning": 45 * 60,
                "meeting": 30 * 60,
            }
            return defaults.get(task_type.lower(), 45 * 60)

        # Calculate average duration of high-quality sessions
        durations = [s.duration for s in relevant_sessions]

        if NUMPY_AVAILABLE:
            optimal = int(np.median(durations))
        else:
            sorted_durations = sorted(durations)
            mid = len(sorted_durations) // 2
            optimal = sorted_durations[mid]

        return optimal

    def calculate_interruption_impact(self, interruptions: List[Interruption]) -> float:
        """Calculate total impact of interruptions (0.0-1.0)"""
        if not interruptions:
            return 0.0

        total_impact = sum(i.impact_score for i in interruptions)
        # Normalize to 0-1 range (assuming max 10 major interruptions)
        normalized = min(1.0, total_impact / 5.0)

        return round(normalized, 3)


# ============================================================================
# DistractionMonitor: Pattern detection and mitigation
# ============================================================================


class DistractionMonitor:
    """Tracks and categorizes distractions"""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.interruptions_file = storage_path / "interruptions.json"

    def record_interruption(self, session_id: str, interruption: Interruption) -> None:
        """Record an interruption"""
        interruptions = self._load_interruptions()

        if session_id not in interruptions:
            interruptions[session_id] = []

        interruptions[session_id].append(interruption.to_dict())
        self._save_interruptions(interruptions)

    def get_distraction_patterns(self, days: int = 7) -> Dict[str, Any]:
        """Analyze distraction patterns over time"""
        interruptions = self._load_interruptions()
        cutoff_date = datetime.now() - timedelta(days=days)

        # Collect recent interruptions
        recent = []
        for session_interruptions in interruptions.values():
            for int_data in session_interruptions:
                interruption = Interruption.from_dict(int_data)
                if interruption.timestamp >= cutoff_date:
                    recent.append(interruption)

        if not recent:
            return {
                "total_interruptions": 0,
                "most_common_type": None,
                "most_common_source": None,
                "average_impact": 0.0,
                "peak_distraction_hours": [],
            }

        # Analyze patterns
        type_counts = {}
        source_counts = {}
        hour_counts = {}
        total_impact = 0.0

        for interruption in recent:
            # Type distribution
            int_type = interruption.interruption_type.value
            type_counts[int_type] = type_counts.get(int_type, 0) + 1

            # Source distribution
            source_counts[interruption.source] = (
                source_counts.get(interruption.source, 0) + 1
            )

            # Hour distribution
            hour = interruption.timestamp.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

            # Impact
            total_impact += interruption.impact_score

        most_common_type = (
            max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        )
        most_common_source = (
            max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else None
        )
        peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3]

        return {
            "total_interruptions": len(recent),
            "most_common_type": most_common_type,
            "most_common_source": most_common_source,
            "average_impact": round(total_impact / len(recent), 3),
            "peak_distraction_hours": [h for h, _ in peak_hours],
            "type_distribution": type_counts,
            "source_distribution": source_counts,
        }

    def suggest_mitigation_strategies(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate mitigation strategies based on patterns"""
        strategies = []

        # Type-specific strategies
        most_common_type = patterns.get("most_common_type")
        if most_common_type == InterruptionType.NOTIFICATION.value:
            strategies.append("🔕 Enable Do Not Disturb mode during focus sessions")
            strategies.append("📱 Turn off non-critical notifications")
        elif most_common_type == InterruptionType.CONTEXT_SWITCH.value:
            strategies.append("🎯 Use time-blocking to batch similar tasks")
            strategies.append("📋 Create a 'later list' for unplanned tasks")
        elif most_common_type == InterruptionType.DISTRACTION.value:
            strategies.append("🧘 Practice mindfulness to improve attention control")
            strategies.append("🎧 Use background music or white noise")

        # High interruption count
        if patterns.get("total_interruptions", 0) > 20:
            strategies.append("⏰ Schedule shorter, more frequent focus sessions")
            strategies.append("🚪 Communicate your focus hours to others")

        # Peak hours
        peak_hours = patterns.get("peak_distraction_hours", [])
        if peak_hours:
            strategies.append(
                f"📅 Avoid scheduling deep work during {peak_hours[0]}:00-{peak_hours[0]+1}:00"
            )

        # Source-specific
        most_common_source = patterns.get("most_common_source", "")
        if (
            "slack" in most_common_source.lower()
            or "email" in most_common_source.lower()
        ):
            strategies.append(
                "📧 Check messages in dedicated time blocks, not continuously"
            )

        return strategies[:5]  # Return top 5 strategies

    def _load_interruptions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load interruptions from storage"""
        if not self.interruptions_file.exists():
            return {}

        try:
            with open(self.interruptions_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_interruptions(
        self, interruptions: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Save interruptions to storage"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self.interruptions_file, "w") as f:
            json.dump(interruptions, f, indent=2)


# ============================================================================
# ProductivityInsights: Actionable recommendations
# ============================================================================


class ProductivityInsights:
    """Generates actionable productivity insights"""

    def __init__(self, analyzer: FocusAnalyzer):
        self.analyzer = analyzer

    def get_peak_focus_hours(self, sessions: List[FocusSession]) -> List[int]:
        """Identify hours with highest focus quality"""
        if not sessions:
            return []

        hour_scores = {}
        for session in sessions:
            if session.quality_score > 0:
                hour = session.start_time.hour
                if hour not in hour_scores:
                    hour_scores[hour] = []
                hour_scores[hour].append(session.quality_score)

        # Calculate average quality per hour
        hour_averages = {
            hour: sum(scores) / len(scores) for hour, scores in hour_scores.items()
        }

        # Return top 3 hours
        top_hours = sorted(hour_averages.items(), key=lambda x: x[1], reverse=True)[:3]
        return [hour for hour, _ in top_hours]

    def get_optimal_session_length(self, sessions: List[FocusSession]) -> int:
        """Determine optimal session length in minutes"""
        high_quality_sessions = [
            s for s in sessions if s.duration and s.quality_score >= 0.7
        ]

        if not high_quality_sessions:
            return 45  # Default: 45 minutes

        durations = [s.duration / 60 for s in high_quality_sessions]

        if NUMPY_AVAILABLE:
            optimal = int(np.median(durations))
        else:
            sorted_durations = sorted(durations)
            mid = len(sorted_durations) // 2
            optimal = int(sorted_durations[mid])

        return optimal

    def get_improvement_recommendations(
        self, sessions: List[FocusSession], metrics: FocusMetrics
    ) -> List[str]:
        """Generate personalized improvement recommendations"""
        recommendations = []

        # Low average quality
        if metrics.quality_average < 0.6:
            recommendations.append(
                "📊 Your focus quality is below optimal. Try shorter sessions (25-45 min)."
            )

        # High interruption rate
        if metrics.interruption_count > metrics.total_sessions * 2:
            recommendations.append(
                "🚫 High interruption rate detected. Review distraction patterns and implement mitigation strategies."
            )

        # Low deep work percentage
        if metrics.deep_work_percentage < 30:
            recommendations.append(
                "🧠 Increase deep work sessions. Schedule 2-3 focused blocks daily."
            )

        # Session length analysis
        avg_duration = metrics.average_session_duration / 60
        if avg_duration < 20:
            recommendations.append(
                "⏱️ Sessions are too short. Aim for 25-45 minute blocks."
            )
        elif avg_duration > 120:
            recommendations.append(
                "⏸️ Sessions are too long. Take regular breaks to maintain quality."
            )

        # Trending analysis
        if len(metrics.productivity_trend) >= 3:
            recent_trend = metrics.productivity_trend[-3:]
            if all(
                recent_trend[i] < recent_trend[i - 1]
                for i in range(1, len(recent_trend))
            ):
                recommendations.append(
                    "📉 Declining trend detected. Consider rest, exercise, or schedule adjustment."
                )

        # Peak hours utilization
        if metrics.peak_focus_hours:
            peak_str = ", ".join(f"{h}:00" for h in metrics.peak_focus_hours[:2])
            recommendations.append(
                f"⭐ Your peak focus hours are {peak_str}. Schedule critical work then."
            )

        return recommendations[:5]  # Top 5 recommendations

    def generate_weekly_report(
        self, sessions: List[FocusSession], start_date: date
    ) -> Dict[str, Any]:
        """Generate comprehensive weekly report"""
        week_sessions = [
            s
            for s in sessions
            if start_date <= s.start_time.date() < start_date + timedelta(days=7)
        ]

        if not week_sessions:
            return {
                "week_start": start_date.isoformat(),
                "total_sessions": 0,
                "message": "No focus sessions recorded this week.",
            }

        # Calculate metrics
        total_time = sum(s.duration for s in week_sessions if s.duration)
        avg_quality = sum(s.quality_score for s in week_sessions) / len(week_sessions)
        flow_sessions = sum(
            1 for s in week_sessions if self.analyzer.detect_flow_state(s)
        )

        # Daily breakdown
        daily_stats = {}
        for session in week_sessions:
            day = session.start_time.date().isoformat()
            if day not in daily_stats:
                daily_stats[day] = {"sessions": 0, "time": 0, "quality": []}
            daily_stats[day]["sessions"] += 1
            daily_stats[day]["time"] += session.duration or 0
            daily_stats[day]["quality"].append(session.quality_score)

        # Calculate daily averages
        for day in daily_stats:
            qualities = daily_stats[day]["quality"]
            daily_stats[day]["avg_quality"] = sum(qualities) / len(qualities)
            del daily_stats[day]["quality"]

        return {
            "week_start": start_date.isoformat(),
            "total_sessions": len(week_sessions),
            "total_time_hours": round(total_time / 3600, 1),
            "average_quality": round(avg_quality, 3),
            "flow_sessions": flow_sessions,
            "daily_stats": daily_stats,
            "best_day": (
                max(daily_stats.items(), key=lambda x: x[1]["avg_quality"])[0]
                if daily_stats
                else None
            ),
        }


# ============================================================================
# FocusTracker: Main orchestrator
# ============================================================================


class FocusTracker:
    """Main focus tracking orchestrator"""

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path.home() / ".vega" / "focus"

        self.storage_path = storage_path
        self.sessions_file = storage_path / "sessions.json"
        self.metrics_cache_file = storage_path / "metrics_cache.json"

        self.analyzer = FocusAnalyzer()
        self.distraction_monitor = DistractionMonitor(storage_path)
        self.insights = ProductivityInsights(self.analyzer)

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def start_session(
        self,
        task_id: Optional[str] = None,
        focus_type: FocusType = FocusType.DEEP_WORK,
        context: str = "",
    ) -> str:
        """Start a new focus session"""
        session = FocusSession(
            session_id=str(uuid.uuid4()),
            task_id=task_id,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            session_type=focus_type,
            quality_score=0.0,
            interruptions=[],
            context=context,
            is_active=True,
        )

        sessions = self._load_sessions()
        sessions.append(session)
        self._save_sessions(sessions)

        return session.session_id

    def end_session(self, session_id: str, notes: Optional[str] = None) -> FocusSession:
        """End an active focus session"""
        sessions = self._load_sessions()

        session = None
        for s in sessions:
            if s.session_id == session_id:
                session = s
                break

        if not session:
            raise ValueError(f"Session {session_id} not found")

        if not session.is_active:
            raise ValueError(f"Session {session_id} is already ended")

        # Update session
        session.end_time = datetime.now()
        session.duration = int((session.end_time - session.start_time).total_seconds())
        session.is_active = False
        if notes:
            session.notes = notes

        # Calculate quality score
        session.quality_score = self.analyzer.analyze_session(session)

        self._save_sessions(sessions)
        self._clear_metrics_cache()

        return session

    def record_interruption(
        self,
        session_id: str,
        interruption_type: InterruptionType,
        source: str,
        duration: int = 60,
        impact_score: float = 0.5,
        notes: Optional[str] = None,
    ) -> str:
        """Record an interruption during a session"""
        interruption = Interruption(
            interruption_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            interruption_type=interruption_type,
            duration=duration,
            source=source,
            impact_score=impact_score,
            notes=notes,
        )

        # Add to session
        sessions = self._load_sessions()
        session = None
        for s in sessions:
            if s.session_id == session_id:
                session = s
                break

        if not session:
            raise ValueError(f"Session {session_id} not found")

        session.interruptions.append(interruption)
        self._save_sessions(sessions)

        # Record in distraction monitor
        self.distraction_monitor.record_interruption(session_id, interruption)

        return interruption.interruption_id

    def get_metrics(
        self, start_date: Optional[date] = None, end_date: Optional[date] = None
    ) -> FocusMetrics:
        """Calculate focus metrics for date range"""
        if start_date is None:
            start_date = date.today() - timedelta(days=30)
        if end_date is None:
            end_date = date.today()

        sessions = self._load_sessions()

        # Filter sessions by date range
        filtered = [
            s
            for s in sessions
            if start_date <= s.start_time.date() <= end_date and not s.is_active
        ]

        if not filtered:
            return FocusMetrics(
                total_sessions=0,
                total_focus_time=0,
                average_session_duration=0,
                quality_average=0.0,
                interruption_count=0,
                deep_work_percentage=0.0,
                productivity_trend=[],
                peak_focus_hours=[],
            )

        # Calculate metrics
        total_sessions = len(filtered)
        total_time = sum(s.duration for s in filtered if s.duration)
        avg_duration = total_time // total_sessions if total_sessions > 0 else 0
        avg_quality = sum(s.quality_score for s in filtered) / total_sessions
        interruption_count = sum(len(s.interruptions) for s in filtered)

        deep_work_sessions = sum(
            1 for s in filtered if s.session_type == FocusType.DEEP_WORK
        )
        deep_work_pct = (
            (deep_work_sessions / total_sessions * 100) if total_sessions > 0 else 0.0
        )

        # Calculate 7-day trend
        today = date.today()
        trend = []
        for i in range(6, -1, -1):
            day = today - timedelta(days=i)
            day_sessions = [s for s in filtered if s.start_time.date() == day]
            if day_sessions:
                day_quality = sum(s.quality_score for s in day_sessions) / len(
                    day_sessions
                )
                trend.append(round(day_quality, 3))
            else:
                trend.append(0.0)

        # Peak hours
        peak_hours = self.insights.get_peak_focus_hours(filtered)

        # Find best day
        daily_quality = {}
        for session in filtered:
            day = session.start_time.date().isoformat()
            if day not in daily_quality:
                daily_quality[day] = []
            daily_quality[day].append(session.quality_score)

        best_day = None
        if daily_quality:
            day_averages = {
                day: sum(scores) / len(scores) for day, scores in daily_quality.items()
            }
            best_day = max(day_averages.items(), key=lambda x: x[1])[0]

        # Improvement rate (compare recent vs older sessions)
        if len(filtered) >= 4:
            mid = len(filtered) // 2
            older_quality = sum(s.quality_score for s in filtered[:mid]) / mid
            recent_quality = sum(s.quality_score for s in filtered[mid:]) / (
                len(filtered) - mid
            )
            improvement = (
                ((recent_quality - older_quality) / older_quality * 100)
                if older_quality > 0
                else 0.0
            )
        else:
            improvement = 0.0

        return FocusMetrics(
            total_sessions=total_sessions,
            total_focus_time=total_time,
            average_session_duration=avg_duration,
            quality_average=round(avg_quality, 3),
            interruption_count=interruption_count,
            deep_work_percentage=round(deep_work_pct, 1),
            productivity_trend=trend,
            peak_focus_hours=peak_hours,
            best_day=best_day,
            improvement_rate=round(improvement, 1),
        )

    def get_session_history(
        self, limit: int = 10, task_id: Optional[str] = None
    ) -> List[FocusSession]:
        """Get recent session history"""
        sessions = self._load_sessions()

        # Filter by task_id if provided
        if task_id:
            sessions = [s for s in sessions if s.task_id == task_id]

        # Sort by start time (most recent first)
        sessions.sort(key=lambda s: s.start_time, reverse=True)

        return sessions[:limit]

    def get_active_session(self) -> Optional[FocusSession]:
        """Get currently active session if any"""
        sessions = self._load_sessions()
        active = [s for s in sessions if s.is_active]
        return active[0] if active else None

    def get_session_by_id(self, session_id: str) -> Optional[FocusSession]:
        """Get session by ID"""
        sessions = self._load_sessions()
        for session in sessions:
            if session.session_id == session_id:
                return session
        return None

    def _load_sessions(self) -> List[FocusSession]:
        """Load sessions from storage"""
        if not self.sessions_file.exists():
            return []

        try:
            with open(self.sessions_file, "r") as f:
                data = json.load(f)
                return [FocusSession.from_dict(s) for s in data]
        except (json.JSONDecodeError, IOError):
            return []

    def _save_sessions(self, sessions: List[FocusSession]) -> None:
        """Save sessions to storage"""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        with open(self.sessions_file, "w") as f:
            json.dump([s.to_dict() for s in sessions], f, indent=2)

    def _clear_metrics_cache(self) -> None:
        """Clear cached metrics"""
        if self.metrics_cache_file.exists():
            self.metrics_cache_file.unlink()


# ============================================================================
# Convenience Functions
# ============================================================================


def get_focus_tracker(storage_path: Optional[Path] = None) -> FocusTracker:
    """Get FocusTracker instance (singleton pattern)"""
    return FocusTracker(storage_path)
