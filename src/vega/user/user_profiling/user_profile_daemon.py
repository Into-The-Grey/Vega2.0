"""
User Profile Daemon
===================

24-hour autonomous enrichment cycles, daily briefing generation,
understanding score tracking, and continuous context adaptation.
"""

import os
import json
import asyncio
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import time
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import aiofiles

from database.user_profile_schema import UserProfileDatabase
from collectors.profile_intel_collector import run_intelligence_scan
from collectors.calendar_sync import run_calendar_sync
from collectors.finance_monitor import run_financial_monitoring
from engines.edu_predictor import run_educational_analysis
from engines.persona_engine import run_persona_analysis, PersonaEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("user_profile_daemon.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for user profile daemon"""

    scan_interval_hours: int = 24
    mini_scan_interval_hours: int = 6
    persona_update_interval_minutes: int = 30
    briefing_generation_hour: int = 8  # 8 AM
    understanding_score_update_hours: int = 12
    database_backup_hours: int = 168  # Weekly
    log_retention_days: int = 30
    enable_intelligence_collection: bool = True
    enable_calendar_sync: bool = True
    enable_financial_monitoring: bool = True
    enable_educational_analysis: bool = True
    enable_persona_tracking: bool = True
    max_concurrent_tasks: int = 3


@dataclass
class UnderstandingScore:
    """AI's understanding score of the user"""

    overall_score: float  # 0.0 to 1.0
    categories: Dict[str, float]
    confidence_level: float
    data_completeness: float
    last_updated: datetime
    improvement_areas: List[str]
    strengths: List[str]


class DailyBriefingGenerator:
    """Generate daily briefings with user insights"""

    def __init__(self, db: UserProfileDatabase):
        self.db = db

    async def generate_daily_briefing(self, date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive daily briefing"""
        if date is None:
            date = datetime.now()

        briefing = {
            "date": date.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "sections": {},
        }

        try:
            # Calendar insights
            briefing["sections"]["calendar"] = await self._generate_calendar_section(
                date
            )

            # Educational insights
            briefing["sections"]["education"] = await self._generate_education_section(
                date
            )

            # Financial insights
            briefing["sections"]["financial"] = await self._generate_financial_section(
                date
            )

            # Personal insights
            briefing["sections"]["personal"] = await self._generate_personal_section(
                date
            )

            # AI understanding status
            briefing["sections"][
                "ai_understanding"
            ] = await self._generate_understanding_section()

            # Recommendations
            briefing["sections"]["recommendations"] = (
                await self._generate_recommendations(date)
            )

            # Summary
            briefing["summary"] = await self._generate_executive_summary(briefing)

        except Exception as e:
            logger.error(f"Error generating daily briefing: {e}")
            briefing["error"] = str(e)

        return briefing

    async def _generate_calendar_section(self, date: datetime) -> Dict[str, Any]:
        """Generate calendar section of briefing"""
        session = self.db.get_session()

        try:
            from database.user_profile_schema import Calendar

            # Today's events
            today_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            today_end = today_start + timedelta(days=1)

            today_events = (
                session.query(Calendar)
                .filter(
                    Calendar.start_time >= today_start,
                    Calendar.start_time < today_end,
                    Calendar.is_active == True,
                )
                .order_by(Calendar.start_time)
                .all()
            )

            # Upcoming events (next 7 days)
            week_end = today_start + timedelta(days=7)
            upcoming_events = (
                session.query(Calendar)
                .filter(
                    Calendar.start_time >= today_end,
                    Calendar.start_time < week_end,
                    Calendar.is_active == True,
                )
                .order_by(Calendar.start_time)
                .limit(5)
                .all()
            )

            # Calculate stress level for today
            stress_levels = [e.stress_level for e in today_events if e.stress_level]
            avg_stress = (
                sum(stress_levels) / len(stress_levels) if stress_levels else 0.3
            )

            return {
                "today_events_count": len(today_events),
                "today_events": [
                    {
                        "title": e.title,
                        "time": e.start_time.strftime("%H:%M"),
                        "type": e.event_type,
                        "stress_level": e.stress_level,
                    }
                    for e in today_events
                ],
                "upcoming_events": [
                    {
                        "title": e.title,
                        "date": e.start_time.strftime("%m/%d"),
                        "type": e.event_type,
                    }
                    for e in upcoming_events
                ],
                "predicted_stress_level": avg_stress,
                "busiest_day_ahead": self._find_busiest_day(upcoming_events),
                "recommendations": self._generate_calendar_recommendations(
                    today_events, avg_stress
                ),
            }

        finally:
            session.close()

    def _find_busiest_day(self, events: List) -> Optional[str]:
        """Find the busiest day in upcoming events"""
        from collections import defaultdict

        day_counts = defaultdict(int)
        for event in events:
            day = event.start_time.strftime("%A, %m/%d")
            day_counts[day] += 1

        if day_counts:
            busiest_day = max(day_counts, key=day_counts.get)
            return f"{busiest_day} ({day_counts[busiest_day]} events)"

        return None

    def _generate_calendar_recommendations(
        self, events: List, stress_level: float
    ) -> List[str]:
        """Generate calendar-based recommendations"""
        recommendations = []

        if len(events) > 6:
            recommendations.append(
                "Heavy schedule today - consider prioritizing essential tasks"
            )

        if stress_level > 0.7:
            recommendations.append(
                "High-stress day predicted - schedule breaks between events"
            )

        if len(events) == 0:
            recommendations.append("Free day - good time for personal projects or rest")

        return recommendations

    async def _generate_education_section(self, date: datetime) -> Dict[str, Any]:
        """Generate education section of briefing"""
        session = self.db.get_session()

        try:
            from database.user_profile_schema import EducationProfile

            # Active courses
            courses = (
                session.query(EducationProfile)
                .filter(
                    EducationProfile.education_type == "course",
                    EducationProfile.status == "current",
                    EducationProfile.is_active == True,
                )
                .all()
            )

            education_summary = {
                "active_courses": len(courses),
                "course_names": [c.program_name for c in courses],
                "upcoming_deadlines": [],
                "study_recommendations": [],
                "performance_alerts": [],
            }

            # Analyze upcoming deadlines
            for course in courses:
                course_details = course.course_details or {}
                assignments = course_details.get("assignments", [])
                exams = course_details.get("exams", [])

                for assignment in assignments:
                    if assignment.get("due_date"):
                        try:
                            due_date = datetime.fromisoformat(assignment["due_date"])
                            days_until = (due_date - date).days
                            if 0 <= days_until <= 7:
                                education_summary["upcoming_deadlines"].append(
                                    {
                                        "type": "assignment",
                                        "title": assignment["title"],
                                        "course": course.program_name,
                                        "days_until": days_until,
                                        "urgency": (
                                            "high" if days_until <= 2 else "medium"
                                        ),
                                    }
                                )
                        except ValueError:
                            continue

                for exam in exams:
                    if exam.get("date"):
                        try:
                            exam_date = datetime.fromisoformat(exam["date"])
                            days_until = (exam_date - date).days
                            if 0 <= days_until <= 14:
                                education_summary["upcoming_deadlines"].append(
                                    {
                                        "type": "exam",
                                        "title": exam["title"],
                                        "course": course.program_name,
                                        "days_until": days_until,
                                        "urgency": (
                                            "critical" if days_until <= 3 else "high"
                                        ),
                                    }
                                )
                        except ValueError:
                            continue

            # Generate study recommendations
            if education_summary["upcoming_deadlines"]:
                urgent_deadlines = [
                    d
                    for d in education_summary["upcoming_deadlines"]
                    if d["urgency"] in ["critical", "high"]
                ]
                if urgent_deadlines:
                    education_summary["study_recommendations"].append(
                        f"Focus on {len(urgent_deadlines)} urgent deadline(s) this week"
                    )

            return education_summary

        finally:
            session.close()

    async def _generate_financial_section(self, date: datetime) -> Dict[str, Any]:
        """Generate financial section of briefing"""
        session = self.db.get_session()

        try:
            from database.user_profile_schema import FinancialStatus

            # Recent financial activity (last 30 days)
            recent_date = date - timedelta(days=30)

            records = (
                session.query(FinancialStatus)
                .filter(
                    FinancialStatus.created_at >= recent_date,
                    FinancialStatus.is_active == True,
                )
                .all()
            )

            financial_summary = {
                "recent_activity": len(records),
                "spending_categories": {},
                "stress_indicators": [],
                "budget_status": "stable",
                "recommendations": [],
            }

            # Categorize spending
            for record in records:
                category = record.subcategory or record.category
                financial_summary["spending_categories"][category] = (
                    financial_summary["spending_categories"].get(category, 0) + 1
                )

            # Check stress indicators
            high_stress_records = [
                r
                for r in records
                if r.financial_stress_indicator and r.financial_stress_indicator > 0.7
            ]
            if high_stress_records:
                financial_summary["stress_indicators"].append(
                    f"{len(high_stress_records)} high-stress financial events in the last month"
                )
                financial_summary["budget_status"] = "stressed"

            # Generate recommendations
            if financial_summary["budget_status"] == "stressed":
                financial_summary["recommendations"].append(
                    "Consider reviewing recent expenses for optimization opportunities"
                )

            return financial_summary

        finally:
            session.close()

    async def _generate_personal_section(self, date: datetime) -> Dict[str, Any]:
        """Generate personal insights section"""
        session = self.db.get_session()

        try:
            from database.user_profile_schema import InterestsHobbies, SocialCircle

            # Interests analysis
            interests = (
                session.query(InterestsHobbies)
                .filter(
                    InterestsHobbies.interest_type == "current",
                    InterestsHobbies.is_active == True,
                )
                .order_by(InterestsHobbies.engagement_level.desc())
                .limit(5)
                .all()
            )

            # Social connections
            connections = (
                session.query(SocialCircle).filter(SocialCircle.is_active == True).all()
            )

            personal_summary = {
                "top_interests": [i.title for i in interests],
                "total_social_connections": len(connections),
                "relationship_types": {},
                "personal_development": [],
                "well_being_indicators": {},
            }

            # Analyze relationships
            for connection in connections:
                rel_type = connection.relationship_type
                personal_summary["relationship_types"][rel_type] = (
                    personal_summary["relationship_types"].get(rel_type, 0) + 1
                )

            # Well-being indicators (calculated from available data)
            well_being_score = 0.5  # Base score

            # Social connections factor
            social_factor = min(1.0, len(connections) / 10) if connections else 0.3

            # Interest engagement factor
            interest_factor = (
                sum(i.engagement_level for i in interests) / len(interests)
                if interests and all(i.engagement_level is not None for i in interests)
                else 0.5
            )

            # Academic factor (if applicable)
            education_records = (
                session.query(EducationProfile)
                .filter(EducationProfile.is_active == True)
                .all()
            )
            academic_factor = (
                min(1.0, len(education_records) / 3) if education_records else 0.5
            )

            # Calculate weighted well-being score
            well_being_score = (
                social_factor * 0.4 + interest_factor * 0.3 + academic_factor * 0.3
            )

            personal_summary["well_being_indicators"] = {
                "social_connections": social_factor,
                "interest_engagement": interest_factor,
                "academic_engagement": academic_factor,
                "overall_score": well_being_score,
            }

            return personal_summary

        finally:
            session.close()

    async def _generate_understanding_section(self) -> Dict[str, Any]:
        """Generate AI understanding status section"""
        # Use the actual understanding calculator
        try:
            understanding_score = (
                self.understanding_calculator.calculate_understanding_score()
            )

            understanding_summary = {
                "overall_understanding": understanding_score.overall_score,
                "data_completeness": understanding_score.data_completeness,
                "confidence_level": understanding_score.confidence_level,
                "category_scores": understanding_score.categories,
                "strengths": understanding_score.strengths,
                "improvement_areas": understanding_score.improvement_areas,
                "last_updated": understanding_score.last_updated.isoformat(),
            }

            # Add detailed breakdown for user insight
            understanding_summary["detailed_breakdown"] = {
                "strong_areas": [
                    f"{area}: {understanding_score.categories.get(area, 0):.2f}"
                    for area in understanding_score.strengths
                ],
                "needs_attention": [
                    f"{area}: {understanding_score.categories.get(area, 0):.2f}"
                    for area in understanding_score.improvement_areas
                ],
            }

        except Exception as e:
            logger.error(f"Failed to calculate understanding score: {e}")
            # Fallback to simplified calculation
            understanding_summary = {
                "overall_understanding": 0.6,  # Conservative fallback
                "data_completeness": 0.5,
                "confidence_level": 0.4,
                "error": "Failed to calculate detailed understanding metrics",
                "knowledge_gaps": ["long-term goals", "personal values"],
                "recent_improvements": ["financial pattern recognition"],
            }

        return understanding_summary

    async def _generate_recommendations(self, date: datetime) -> List[str]:
        """Generate daily recommendations"""
        recommendations = []

        # Time-based recommendations
        hour = date.hour
        if 6 <= hour <= 9:
            recommendations.append(
                "Good morning! Review your calendar and prioritize today's tasks"
            )
        elif 12 <= hour <= 14:
            recommendations.append(
                "Midday check-in: How are you progressing on your priorities?"
            )
        elif 18 <= hour <= 21:
            recommendations.append(
                "Evening reflection: What went well today and what can be improved?"
            )

        # Day-of-week recommendations
        weekday = date.weekday()
        if weekday == 0:  # Monday
            recommendations.append(
                "Start the week strong with clear goals and priorities"
            )
        elif weekday == 4:  # Friday
            recommendations.append(
                "End the week by reviewing accomplishments and planning ahead"
            )
        elif weekday >= 5:  # Weekend
            recommendations.append(
                "Weekend time: Balance rest, personal interests, and preparation"
            )

        return recommendations

    async def _generate_executive_summary(self, briefing: Dict[str, Any]) -> str:
        """Generate executive summary of the briefing"""
        sections = briefing.get("sections", {})

        summary_parts = []

        # Calendar summary
        calendar = sections.get("calendar", {})
        event_count = calendar.get("today_events_count", 0)
        if event_count > 0:
            stress = calendar.get("predicted_stress_level", 0)
            stress_desc = (
                "high-stress"
                if stress > 0.7
                else "moderate-stress" if stress > 0.4 else "low-stress"
            )
            summary_parts.append(
                f"{event_count} scheduled events today ({stress_desc})"
            )
        else:
            summary_parts.append("Free day with no scheduled events")

        # Education summary
        education = sections.get("education", {})
        deadlines = education.get("upcoming_deadlines", [])
        if deadlines:
            urgent = len(
                [d for d in deadlines if d.get("urgency") in ["critical", "high"]]
            )
            summary_parts.append(f"{urgent} urgent academic deadlines approaching")

        # Personal summary
        personal = sections.get("personal", {})
        well_being = personal.get("well_being_indicators", {}).get("overall_score", 0.5)
        if well_being > 0.7:
            summary_parts.append("Personal well-being indicators are positive")
        elif well_being < 0.4:
            summary_parts.append("Personal well-being may need attention")

        if summary_parts:
            return "Today's overview: " + "; ".join(summary_parts) + "."
        else:
            return "No significant events or concerns identified for today."


class UnderstandingScoreCalculator:
    """Calculate and track AI's understanding of the user"""

    def __init__(self, db: UserProfileDatabase):
        self.db = db

    def calculate_understanding_score(self) -> UnderstandingScore:
        """Calculate comprehensive understanding score"""
        session = self.db.get_session()

        try:
            category_scores = {}

            # Identity understanding
            category_scores["identity"] = self._calculate_identity_score(session)

            # Calendar understanding
            category_scores["calendar"] = self._calculate_calendar_score(session)

            # Academic understanding
            category_scores["academic"] = self._calculate_academic_score(session)

            # Financial understanding
            category_scores["financial"] = self._calculate_financial_score(session)

            # Social understanding
            category_scores["social"] = self._calculate_social_score(session)

            # Interests understanding
            category_scores["interests"] = self._calculate_interests_score(session)

            # Calculate overall score
            overall_score = sum(category_scores.values()) / len(category_scores)

            # Calculate confidence and completeness
            confidence_level = self._calculate_confidence_level(category_scores)
            data_completeness = self._calculate_data_completeness(session)

            # Identify improvement areas and strengths
            improvement_areas = [
                cat for cat, score in category_scores.items() if score < 0.6
            ]
            strengths = [cat for cat, score in category_scores.items() if score > 0.8]

            return UnderstandingScore(
                overall_score=overall_score,
                categories=category_scores,
                confidence_level=confidence_level,
                data_completeness=data_completeness,
                last_updated=datetime.now(),
                improvement_areas=improvement_areas,
                strengths=strengths,
            )

        finally:
            session.close()

    def _calculate_identity_score(self, session) -> float:
        """Calculate understanding of user identity"""
        from database.user_profile_schema import IdentityCore, ContactInfo

        identity_records = (
            session.query(IdentityCore).filter(IdentityCore.is_active == True).all()
        )
        contact_records = (
            session.query(ContactInfo).filter(ContactInfo.is_active == True).all()
        )

        score = 0.0

        if identity_records:
            identity = identity_records[0]
            if identity.full_name:
                score += 0.3
            if identity.birth_date:
                score += 0.2
            if identity.primary_language:
                score += 0.1

        if contact_records:
            score += min(0.4, len(contact_records) * 0.1)

        return min(1.0, score)

    def _calculate_calendar_score(self, session) -> float:
        """Calculate understanding of user's calendar patterns"""
        from database.user_profile_schema import Calendar

        # Count calendar events
        event_count = session.query(Calendar).filter(Calendar.is_active == True).count()

        # More events = better understanding of patterns
        score = min(1.0, event_count / 50)  # Normalize to 50 events for full score

        return score

    def _calculate_academic_score(self, session) -> float:
        """Calculate understanding of academic life"""
        from database.user_profile_schema import EducationProfile

        education_records = (
            session.query(EducationProfile)
            .filter(EducationProfile.is_active == True)
            .all()
        )

        if not education_records:
            return 0.5  # Neutral if no academic info

        score = 0.0
        for record in education_records:
            if record.program_name:
                score += 0.2
            if record.course_details:
                score += 0.3
            if record.performance_metrics:
                score += 0.2

        return min(1.0, score / len(education_records))

    def _calculate_financial_score(self, session) -> float:
        """Calculate understanding of financial patterns"""
        from database.user_profile_schema import FinancialStatus

        financial_records = (
            session.query(FinancialStatus)
            .filter(FinancialStatus.is_active == True)
            .all()
        )

        if not financial_records:
            return 0.3  # Low if no financial data

        # Score based on data richness and recency
        recent_records = [
            r for r in financial_records if (datetime.now() - r.created_at).days <= 30
        ]

        score = min(1.0, len(recent_records) / 20)  # Normalize to 20 recent records

        return score

    def _calculate_social_score(self, session) -> float:
        """Calculate understanding of social relationships"""
        from database.user_profile_schema import SocialCircle

        social_records = (
            session.query(SocialCircle).filter(SocialCircle.is_active == True).all()
        )

        if not social_records:
            return 0.4  # Low-medium if no social data

        score = 0.0
        for record in social_records:
            if record.name:
                score += 0.2
            if record.relationship_type:
                score += 0.2
            if record.personality_notes:
                score += 0.1

        return min(1.0, score / len(social_records))

    def _calculate_interests_score(self, session) -> float:
        """Calculate understanding of user interests"""
        from database.user_profile_schema import InterestsHobbies

        interest_records = (
            session.query(InterestsHobbies)
            .filter(InterestsHobbies.is_active == True)
            .all()
        )

        if not interest_records:
            return 0.4

        score = min(1.0, len(interest_records) / 10)  # Normalize to 10 interests

        # Bonus for engagement level data
        engagement_data = [
            r for r in interest_records if r.engagement_level is not None
        ]
        if engagement_data:
            score += 0.2

        return min(1.0, score)

    def _calculate_confidence_level(self, category_scores: Dict[str, float]) -> float:
        """Calculate confidence level in understanding"""
        # Confidence is based on consistency across categories
        scores = list(category_scores.values())
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)

        # Lower variance = higher confidence
        confidence = max(0.1, 1.0 - variance)

        return confidence

    def _calculate_data_completeness(self, session) -> float:
        """Calculate overall data completeness"""
        from database.user_profile_schema import (
            IdentityCore,
            ContactInfo,
            Calendar,
            EducationProfile,
            FinancialStatus,
            SocialCircle,
            InterestsHobbies,
        )

        table_counts = {
            "identity": session.query(IdentityCore)
            .filter(IdentityCore.is_active == True)
            .count(),
            "contacts": session.query(ContactInfo)
            .filter(ContactInfo.is_active == True)
            .count(),
            "calendar": session.query(Calendar)
            .filter(Calendar.is_active == True)
            .count(),
            "education": session.query(EducationProfile)
            .filter(EducationProfile.is_active == True)
            .count(),
            "financial": session.query(FinancialStatus)
            .filter(FinancialStatus.is_active == True)
            .count(),
            "social": session.query(SocialCircle)
            .filter(SocialCircle.is_active == True)
            .count(),
            "interests": session.query(InterestsHobbies)
            .filter(InterestsHobbies.is_active == True)
            .count(),
        }

        # Calculate completeness based on presence of data in each category
        completeness_scores = []
        for table, count in table_counts.items():
            if count > 0:
                completeness_scores.append(1.0)
            else:
                completeness_scores.append(0.0)

        return sum(completeness_scores) / len(completeness_scores)


class UserProfileDaemon:
    """Main daemon for autonomous user profiling"""

    def __init__(self, config: DaemonConfig = None, db_path: str = None):
        self.config = config or DaemonConfig()
        self.db = UserProfileDatabase(db_path)
        self.briefing_generator = DailyBriefingGenerator(self.db)
        self.understanding_calculator = UnderstandingScoreCalculator(self.db)
        self.persona_engine = PersonaEngine(self.db)

        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)

        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    async def start_daemon(self):
        """Start the autonomous profiling daemon"""
        logger.info("Starting User Profile Daemon...")
        self.running = True

        # Schedule periodic tasks
        self._schedule_tasks()

        # Run initial scan
        await self._run_initial_scan()

        # Main daemon loop
        while self.running:
            try:
                # Run scheduled tasks
                schedule.run_pending()

                # Update persona state
                await self.persona_engine.update_persona_state()

                # Sleep for a short interval
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

        logger.info("User Profile Daemon stopped")

    def _schedule_tasks(self):
        """Schedule all periodic tasks"""
        # Daily full scan
        schedule.every().day.at("02:00").do(self._schedule_full_scan)

        # Mini scans every 6 hours
        schedule.every(self.config.mini_scan_interval_hours).hours.do(
            self._schedule_mini_scan
        )

        # Daily briefing generation
        schedule.every().day.at(f"{self.config.briefing_generation_hour:02d}:00").do(
            self._schedule_briefing_generation
        )

        # Understanding score update
        schedule.every(self.config.understanding_score_update_hours).hours.do(
            self._schedule_understanding_update
        )

        # Weekly database backup
        schedule.every().week.do(self._schedule_database_backup)

        # Daily log cleanup
        schedule.every().day.at("03:00").do(self._schedule_log_cleanup)

    def _schedule_full_scan(self):
        """Schedule full intelligence scan"""
        if self.running:
            asyncio.create_task(self._run_full_scan())

    def _schedule_mini_scan(self):
        """Schedule mini scan (essential updates only)"""
        if self.running:
            asyncio.create_task(self._run_mini_scan())

    def _schedule_briefing_generation(self):
        """Schedule daily briefing generation"""
        if self.running:
            asyncio.create_task(self._generate_daily_briefing())

    def _schedule_understanding_update(self):
        """Schedule understanding score update"""
        if self.running:
            asyncio.create_task(self._update_understanding_score())

    def _schedule_database_backup(self):
        """Schedule database backup"""
        if self.running:
            asyncio.create_task(self._backup_database())

    def _schedule_log_cleanup(self):
        """Schedule log cleanup"""
        if self.running:
            asyncio.create_task(self._cleanup_logs())

    async def _run_initial_scan(self):
        """Run initial scan on daemon startup"""
        logger.info("Running initial scan...")

        try:
            # Quick persona update
            await self.persona_engine.update_persona_state(force_update=True)

            # Generate understanding score
            await self._update_understanding_score()

            # Generate today's briefing if not already done
            await self._generate_daily_briefing()

            logger.info("Initial scan completed")

        except Exception as e:
            logger.error(f"Initial scan error: {e}")

    async def _run_full_scan(self):
        """Run comprehensive full scan"""
        logger.info("Starting full intelligence scan...")

        scan_results = {
            "scan_type": "full",
            "start_time": datetime.now().isoformat(),
            "results": {},
            "errors": [],
        }

        try:
            tasks = []

            # Intelligence collection
            if self.config.enable_intelligence_collection:
                tasks.append(("intelligence", run_intelligence_scan()))

            # Calendar sync
            if self.config.enable_calendar_sync:
                tasks.append(("calendar", run_calendar_sync()))

            # Financial monitoring
            if self.config.enable_financial_monitoring:
                tasks.append(("financial", run_financial_monitoring()))

            # Educational analysis
            if self.config.enable_educational_analysis:
                tasks.append(("educational", run_educational_analysis()))

            # Persona analysis
            if self.config.enable_persona_tracking:
                tasks.append(("persona", run_persona_analysis()))

            # Run tasks concurrently
            results = await asyncio.gather(
                *[task[1] for task in tasks], return_exceptions=True
            )

            # Process results
            for i, (task_name, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    error_msg = f"{task_name} scan failed: {result}"
                    scan_results["errors"].append(error_msg)
                    logger.error(error_msg)
                else:
                    scan_results["results"][task_name] = result
                    logger.info(f"{task_name} scan completed successfully")

        except Exception as e:
            error_msg = f"Full scan error: {e}"
            scan_results["errors"].append(error_msg)
            logger.error(error_msg)

        scan_results["end_time"] = datetime.now().isoformat()

        # Save scan results
        await self._save_scan_results(scan_results)

        logger.info("Full intelligence scan completed")

    async def _run_mini_scan(self):
        """Run mini scan (essential updates only)"""
        logger.info("Starting mini scan...")

        try:
            # Quick calendar sync
            if self.config.enable_calendar_sync:
                await run_calendar_sync()

            # Persona update
            if self.config.enable_persona_tracking:
                await self.persona_engine.update_persona_state(force_update=True)

            logger.info("Mini scan completed")

        except Exception as e:
            logger.error(f"Mini scan error: {e}")

    async def _generate_daily_briefing(self):
        """Generate and save daily briefing"""
        logger.info("Generating daily briefing...")

        try:
            briefing = await self.briefing_generator.generate_daily_briefing()

            # Save briefing to file
            briefing_dir = Path("user_profiling/briefings")
            briefing_dir.mkdir(exist_ok=True)

            briefing_file = (
                briefing_dir / f"briefing_{datetime.now().strftime('%Y%m%d')}.json"
            )

            async with aiofiles.open(briefing_file, "w") as f:
                await f.write(json.dumps(briefing, indent=2, default=str))

            logger.info(f"Daily briefing saved to {briefing_file}")

        except Exception as e:
            logger.error(f"Briefing generation error: {e}")

    async def _update_understanding_score(self):
        """Update AI understanding score"""
        logger.info("Updating understanding score...")

        try:
            score = self.understanding_calculator.calculate_understanding_score()

            # Save score to file
            score_dir = Path("user_profiling/understanding")
            score_dir.mkdir(exist_ok=True)

            score_file = score_dir / "understanding_score.json"

            async with aiofiles.open(score_file, "w") as f:
                await f.write(json.dumps(asdict(score), indent=2, default=str))

            logger.info(f"Understanding score updated: {score.overall_score:.2f}")

        except Exception as e:
            logger.error(f"Understanding score update error: {e}")

    async def _backup_database(self):
        """Backup user database"""
        logger.info("Creating database backup...")

        try:
            backup_dir = Path("user_profiling/backups")
            backup_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"user_core_backup_{timestamp}.db"

            self.db.backup_database(str(backup_file))

            logger.info(f"Database backup saved to {backup_file}")

        except Exception as e:
            logger.error(f"Database backup error: {e}")

    async def _cleanup_logs(self):
        """Cleanup old log files"""
        logger.info("Cleaning up old logs...")

        try:
            log_files = Path(".").glob("*.log")
            cutoff_date = datetime.now() - timedelta(
                days=self.config.log_retention_days
            )

            for log_file in log_files:
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    logger.info(f"Deleted old log file: {log_file}")

        except Exception as e:
            logger.error(f"Log cleanup error: {e}")

    async def _save_scan_results(self, results: Dict[str, Any]):
        """Save scan results to file"""
        try:
            results_dir = Path("user_profiling/scan_results")
            results_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"scan_results_{timestamp}.json"

            async with aiofiles.open(results_file, "w") as f:
                await f.write(json.dumps(results, indent=2, default=str))

        except Exception as e:
            logger.error(f"Error saving scan results: {e}")

    def get_daemon_status(self) -> Dict[str, Any]:
        """Get current daemon status"""
        return {
            "running": self.running,
            "config": asdict(self.config),
            "persona_state": (
                self.persona_engine.get_persona_summary()
                if hasattr(self, "persona_engine")
                else {}
            ),
            "next_scheduled_tasks": [str(job) for job in schedule.jobs],
            "uptime": time.time() - getattr(self, "start_time", time.time()),
        }


async def run_daemon(config_dict: Dict = None, db_path: str = None):
    """Main function to run the user profile daemon"""
    config = DaemonConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    daemon = UserProfileDaemon(config, db_path)
    await daemon.start_daemon()


if __name__ == "__main__":
    # Run daemon
    try:
        asyncio.run(run_daemon())
    except KeyboardInterrupt:
        logger.info("Daemon stopped by user")
    except Exception as e:
        logger.error(f"Daemon error: {e}")
        sys.exit(1)
