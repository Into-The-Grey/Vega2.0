"""
Persona and Relationship Engine
===============================

Adaptive behavioral mirroring, social circle modeling, mode switching
(default/focused/burnout/social), and contextual tone adjustment for
personalized AI interactions.
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re
from enum import Enum
from collections import defaultdict
import numpy as np

from database.user_profile_schema import (
    UserProfileDatabase,
    SocialCircle,
    Calendar,
    EducationProfile,
    FinancialStatus,
    SearchHistory,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaMode(Enum):
    """Available persona modes"""

    DEFAULT = "default"
    FOCUSED = "focused"
    BURNOUT = "burnout"
    SOCIAL = "social"
    ACADEMIC = "academic"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


@dataclass
class PersonaConfig:
    """Configuration for persona engine"""

    default_mode: PersonaMode = PersonaMode.DEFAULT
    auto_mode_switching: bool = True
    context_sensitivity: float = 0.8  # 0.0 to 1.0
    relationship_modeling: bool = True
    tone_adaptation: bool = True
    emotional_intelligence: bool = True
    memory_depth: int = 30  # Days of conversation history to consider
    stress_detection: bool = True


@dataclass
class PersonaState:
    """Current persona state and context"""

    mode: PersonaMode
    energy_level: float  # 0.0 to 1.0
    stress_level: float  # 0.0 to 1.0
    focus_level: float  # 0.0 to 1.0
    social_battery: float  # 0.0 to 1.0
    cognitive_load: float  # 0.0 to 1.0
    emotional_state: str  # happy, stressed, excited, tired, etc.
    context_factors: Dict[str, Any]
    last_updated: datetime

    def __post_init__(self):
        if not hasattr(self, "last_updated") or self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class InteractionStyle:
    """Defines how AI should interact in different modes"""

    verbosity: float  # 0.0 (concise) to 1.0 (verbose)
    formality: float  # 0.0 (casual) to 1.0 (formal)
    empathy: float  # 0.0 (objective) to 1.0 (empathetic)
    directness: float  # 0.0 (indirect) to 1.0 (direct)
    enthusiasm: float  # 0.0 (subdued) to 1.0 (enthusiastic)
    technical_depth: float  # 0.0 (simplified) to 1.0 (technical)
    patience: float  # 0.0 (quick) to 1.0 (patient)
    supportiveness: float  # 0.0 (neutral) to 1.0 (supportive)


class ContextAnalyzer:
    """Analyze current context to determine appropriate persona mode"""

    def __init__(self, db: UserProfileDatabase):
        self.db = db

        # Context indicators for different modes
        self.mode_indicators = {
            PersonaMode.FOCUSED: {
                "calendar_keywords": [
                    "deadline",
                    "exam",
                    "presentation",
                    "meeting",
                    "work",
                ],
                "stress_threshold": 0.7,
                "time_pressure_hours": 24,
                "work_hours": True,
            },
            PersonaMode.BURNOUT: {
                "stress_threshold": 0.8,
                "consecutive_busy_days": 5,
                "sleep_deficit": True,
                "overwhelming_calendar": True,
            },
            PersonaMode.SOCIAL: {
                "calendar_keywords": ["party", "dinner", "hangout", "social", "friend"],
                "weekend": True,
                "evening_hours": True,
                "social_context": True,
            },
            PersonaMode.ACADEMIC: {
                "calendar_keywords": [
                    "class",
                    "study",
                    "lecture",
                    "exam",
                    "assignment",
                ],
                "academic_hours": True,
                "semester_active": True,
            },
            PersonaMode.PROFESSIONAL: {
                "calendar_keywords": [
                    "meeting",
                    "work",
                    "client",
                    "project",
                    "business",
                ],
                "work_hours": True,
                "professional_context": True,
            },
        }

    def analyze_current_context(self) -> Dict[str, Any]:
        """Analyze current context from all available data"""
        context = {
            "calendar_pressure": self._analyze_calendar_pressure(),
            "academic_load": self._analyze_academic_load(),
            "financial_stress": self._analyze_financial_stress(),
            "social_activity": self._analyze_social_activity(),
            "time_context": self._analyze_time_context(),
            "search_patterns": self._analyze_recent_searches(),
            "conversation_history": self._analyze_conversation_patterns(),
        }

        return context

    def _analyze_calendar_pressure(self) -> Dict[str, Any]:
        """Analyze calendar for stress indicators"""
        session = self.db.get_session()
        pressure_metrics = {
            "upcoming_deadlines": 0,
            "busy_days": 0,
            "time_pressure": 0.0,
            "stress_events": [],
        }

        try:
            # Get upcoming events (next 7 days)
            end_date = datetime.now() + timedelta(days=7)

            events = (
                session.query(Calendar)
                .filter(
                    Calendar.start_time >= datetime.now(),
                    Calendar.start_time <= end_date,
                    Calendar.is_active == True,
                )
                .all()
            )

            # Count deadlines and high-stress events
            for event in events:
                if any(
                    keyword in event.title.lower()
                    for keyword in ["deadline", "due", "exam"]
                ):
                    pressure_metrics["upcoming_deadlines"] += 1

                if event.stress_level and event.stress_level > 0.7:
                    pressure_metrics["stress_events"].append(
                        {
                            "title": event.title,
                            "date": event.start_time.isoformat(),
                            "stress_level": event.stress_level,
                        }
                    )

            # Analyze daily event density
            daily_counts = defaultdict(int)
            for event in events:
                day = event.start_time.date()
                daily_counts[day] += 1

            pressure_metrics["busy_days"] = sum(
                1 for count in daily_counts.values() if count > 4
            )

            # Calculate overall time pressure
            if events:
                avg_stress = sum(e.stress_level or 0.5 for e in events) / len(events)
                density_factor = len(events) / 7  # Events per day
                pressure_metrics["time_pressure"] = min(
                    1.0, avg_stress * density_factor
                )

        finally:
            session.close()

        return pressure_metrics

    def _analyze_academic_load(self) -> Dict[str, Any]:
        """Analyze academic workload and stress"""
        session = self.db.get_session()
        academic_metrics = {
            "active_courses": 0,
            "upcoming_assignments": 0,
            "exam_pressure": 0.0,
            "academic_stress": 0.0,
        }

        try:
            # Count active courses
            courses = (
                session.query(EducationProfile)
                .filter(
                    EducationProfile.education_type == "course",
                    EducationProfile.status == "current",
                    EducationProfile.is_active == True,
                )
                .all()
            )

            academic_metrics["active_courses"] = len(courses)

            # Analyze course details for stress indicators
            total_stress = 0
            for course in courses:
                course_details = course.course_details or {}
                assignments = course_details.get("assignments", [])
                exams = course_details.get("exams", [])

                # Count upcoming assignments
                for assignment in assignments:
                    if assignment.get("due_date"):
                        try:
                            due_date = datetime.fromisoformat(assignment["due_date"])
                            if (
                                datetime.now()
                                <= due_date
                                <= datetime.now() + timedelta(days=14)
                            ):
                                academic_metrics["upcoming_assignments"] += 1
                        except ValueError:
                            continue

                # Analyze exam pressure
                for exam in exams:
                    if exam.get("date"):
                        try:
                            exam_date = datetime.fromisoformat(exam["date"])
                            days_until = (exam_date - datetime.now()).days
                            if 0 <= days_until <= 14:
                                # Higher pressure for closer exams
                                pressure = max(0, 1.0 - (days_until / 14))
                                academic_metrics["exam_pressure"] += pressure
                        except ValueError:
                            continue

            # Calculate overall academic stress
            if courses:
                stress_factors = [
                    academic_metrics["upcoming_assignments"] / 10,  # Normalize
                    academic_metrics["exam_pressure"],
                    academic_metrics["active_courses"]
                    / 6,  # Normalize for typical course load
                ]
                academic_metrics["academic_stress"] = min(
                    1.0, sum(stress_factors) / len(stress_factors)
                )

        finally:
            session.close()

        return academic_metrics

    def _analyze_financial_stress(self) -> Dict[str, Any]:
        """Analyze financial stress indicators"""
        session = self.db.get_session()
        financial_metrics = {
            "stress_level": 0.0,
            "recent_stress_events": 0,
            "budget_pressure": False,
        }

        try:
            # Get recent financial records
            recent_date = datetime.now() - timedelta(days=30)

            records = (
                session.query(FinancialStatus)
                .filter(
                    FinancialStatus.created_at >= recent_date,
                    FinancialStatus.is_active == True,
                )
                .all()
            )

            if records:
                # Calculate average stress indicator
                stress_values = [
                    r.financial_stress_indicator
                    for r in records
                    if r.financial_stress_indicator
                ]
                if stress_values:
                    financial_metrics["stress_level"] = sum(stress_values) / len(
                        stress_values
                    )

                # Count stress events
                financial_metrics["recent_stress_events"] = sum(
                    1
                    for r in records
                    if r.financial_stress_indicator
                    and r.financial_stress_indicator > 0.7
                )

        finally:
            session.close()

        return financial_metrics

    def _analyze_social_activity(self) -> Dict[str, Any]:
        """Analyze social context and activity"""
        session = self.db.get_session()
        social_metrics = {
            "recent_social_events": 0,
            "social_calendar_density": 0.0,
            "relationship_activity": 0,
        }

        try:
            # Check for social events in calendar
            recent_date = datetime.now() - timedelta(days=7)

            social_events = (
                session.query(Calendar)
                .filter(
                    Calendar.start_time >= recent_date,
                    Calendar.event_type == "social",
                    Calendar.is_active == True,
                )
                .all()
            )

            social_metrics["recent_social_events"] = len(social_events)
            social_metrics["social_calendar_density"] = (
                len(social_events) / 7
            )  # Events per day

            # Count active relationships
            relationships = (
                session.query(SocialCircle).filter(SocialCircle.is_active == True).all()
            )

            social_metrics["relationship_activity"] = len(relationships)

        finally:
            session.close()

        return social_metrics

    def _analyze_time_context(self) -> Dict[str, Any]:
        """Analyze current time context"""
        now = datetime.now()

        return {
            "hour": now.hour,
            "day_of_week": now.weekday(),  # 0=Monday
            "is_weekend": now.weekday() >= 5,
            "is_work_hours": 9 <= now.hour <= 17,
            "is_evening": 18 <= now.hour <= 22,
            "is_late_night": now.hour >= 23 or now.hour <= 5,
            "month": now.month,
            "season": self._get_season(now.month),
        }

    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"

    def _analyze_recent_searches(self) -> Dict[str, Any]:
        """Analyze recent search patterns for context"""
        session = self.db.get_session()
        search_metrics = {
            "recent_searches": 0,
            "search_categories": [],
            "urgency_level": 0.0,
            "dominant_topics": [],
        }

        try:
            # Get recent searches (last 24 hours)
            recent_date = datetime.now() - timedelta(hours=24)

            searches = (
                session.query(SearchHistory)
                .filter(SearchHistory.timestamp >= recent_date)
                .all()
            )

            search_metrics["recent_searches"] = len(searches)

            if searches:
                # Categorize searches
                categories = defaultdict(int)
                topics = defaultdict(int)

                for search in searches:
                    if search.topic_category:
                        categories[search.topic_category] += 1

                    if search.keyword:
                        # Simple topic extraction
                        words = search.keyword.lower().split()
                        for word in words:
                            if len(word) > 3:  # Filter short words
                                topics[word] += 1

                search_metrics["search_categories"] = list(categories.keys())
                search_metrics["dominant_topics"] = [
                    topic
                    for topic, count in sorted(
                        topics.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                ]

                # Calculate urgency from search patterns
                urgent_keywords = [
                    "urgent",
                    "deadline",
                    "help",
                    "error",
                    "problem",
                    "fix",
                ]
                urgent_searches = sum(
                    1
                    for search in searches
                    if any(
                        keyword in search.keyword.lower() for keyword in urgent_keywords
                    )
                )
                search_metrics["urgency_level"] = (
                    urgent_searches / len(searches) if searches else 0
                )

        finally:
            session.close()

        return search_metrics

    def _analyze_conversation_patterns(self) -> Dict[str, Any]:
        """Analyze recent conversation patterns using available data"""
        patterns = {
            "recent_conversations": 0,
            "conversation_tone": "neutral",
            "topics_discussed": [],
            "user_engagement": 0.5,
            "conversation_frequency": 0.0,
            "response_complexity": 0.5,
            "emotional_indicators": [],
        }

        try:
            # Try to import and use conversation history if available
            try:
                from ..db import get_session_history

                # Get recent conversation history
                recent_conversations = get_session_history(session_id=None, limit=20)
                patterns["recent_conversations"] = len(recent_conversations)

                if recent_conversations:
                    # Analyze conversation frequency (conversations per day)
                    from datetime import datetime, timedelta

                    recent_dates = []
                    total_prompt_length = 0
                    total_response_length = 0

                    for conv in recent_conversations:
                        if "ts" in conv:
                            recent_dates.append(conv["ts"])

                        prompt = conv.get("prompt", "")
                        response = conv.get("response", "")
                        total_prompt_length += len(prompt)
                        total_response_length += len(response)

                        # Analyze emotional indicators in prompts
                        emotional_keywords = {
                            "stress": [
                                "stressed",
                                "overwhelmed",
                                "anxious",
                                "pressure",
                                "deadline",
                            ],
                            "happy": [
                                "great",
                                "awesome",
                                "excited",
                                "happy",
                                "amazing",
                            ],
                            "frustrated": [
                                "frustrated",
                                "annoying",
                                "difficult",
                                "hard",
                                "struggling",
                            ],
                            "curious": ["how", "why", "what", "learn", "understand"],
                        }

                        prompt_lower = prompt.lower()
                        for emotion, keywords in emotional_keywords.items():
                            if any(keyword in prompt_lower for keyword in keywords):
                                if emotion not in patterns["emotional_indicators"]:
                                    patterns["emotional_indicators"].append(emotion)

                        # Extract topics (simple keyword analysis)
                        academic_keywords = [
                            "homework",
                            "assignment",
                            "exam",
                            "study",
                            "class",
                            "course",
                        ]
                        work_keywords = [
                            "work",
                            "job",
                            "project",
                            "meeting",
                            "deadline",
                            "client",
                        ]
                        personal_keywords = [
                            "friend",
                            "family",
                            "weekend",
                            "plan",
                            "schedule",
                        ]

                        if any(kw in prompt_lower for kw in academic_keywords):
                            if "academic" not in patterns["topics_discussed"]:
                                patterns["topics_discussed"].append("academic")
                        if any(kw in prompt_lower for kw in work_keywords):
                            if "work" not in patterns["topics_discussed"]:
                                patterns["topics_discussed"].append("work")
                        if any(kw in prompt_lower for kw in personal_keywords):
                            if "personal" not in patterns["topics_discussed"]:
                                patterns["topics_discussed"].append("personal")

                    # Calculate engagement metrics
                    if total_prompt_length > 0:
                        avg_prompt_length = total_prompt_length / len(
                            recent_conversations
                        )
                        # Longer prompts indicate higher engagement
                        patterns["user_engagement"] = min(1.0, avg_prompt_length / 100)

                    # Calculate response complexity preference
                    if total_response_length > 0:
                        avg_response_length = total_response_length / len(
                            recent_conversations
                        )
                        patterns["response_complexity"] = min(
                            1.0, avg_response_length / 200
                        )

                    # Determine conversation tone from emotional indicators
                    if len(patterns["emotional_indicators"]) > 0:
                        if (
                            "stress" in patterns["emotional_indicators"]
                            or "frustrated" in patterns["emotional_indicators"]
                        ):
                            patterns["conversation_tone"] = "stressed"
                        elif "happy" in patterns["emotional_indicators"]:
                            patterns["conversation_tone"] = "positive"
                        elif "curious" in patterns["emotional_indicators"]:
                            patterns["conversation_tone"] = "inquisitive"

                    # Calculate frequency (conversations per day over last week)
                    if recent_dates:
                        date_range = max(recent_dates) - min(recent_dates)
                        if date_range.total_seconds() > 0:
                            days = date_range.total_seconds() / 86400  # Convert to days
                            patterns["conversation_frequency"] = len(
                                recent_conversations
                            ) / max(1, days)

            except ImportError:
                logger.debug(
                    "Conversation history not available - using fallback analysis"
                )

            # Use search history as alternative conversation pattern indicator
            session = self.db.get_session()
            try:
                from database.user_profile_schema import SearchHistory

                recent_searches = (
                    session.query(SearchHistory)
                    .filter(SearchHistory.is_active == True)
                    .order_by(SearchHistory.created_at.desc())
                    .limit(20)
                    .all()
                )

                if recent_searches:
                    patterns["recent_conversations"] += len(recent_searches)

                    # Analyze search query complexity as engagement indicator
                    total_query_length = sum(
                        len(s.query or "") for s in recent_searches
                    )
                    if total_query_length > 0:
                        avg_query_length = total_query_length / len(recent_searches)
                        patterns["user_engagement"] = max(
                            patterns["user_engagement"], min(1.0, avg_query_length / 50)
                        )

                    # Extract topics from search categories
                    search_categories = [
                        s.category for s in recent_searches if s.category
                    ]
                    for category in set(search_categories):
                        if category not in patterns["topics_discussed"]:
                            patterns["topics_discussed"].append(category)

            finally:
                session.close()

        except Exception as e:
            logger.warning(f"Error analyzing conversation patterns: {e}")

        return patterns

    def determine_optimal_mode(
        self, context: Dict[str, Any]
    ) -> Tuple[PersonaMode, float]:
        """Determine optimal persona mode based on context"""
        mode_scores = {}

        # Calculate scores for each mode
        for mode in PersonaMode:
            mode_scores[mode] = self._calculate_mode_score(mode, context)

        # Find best mode
        best_mode = max(mode_scores, key=mode_scores.get)
        confidence = mode_scores[best_mode]

        return best_mode, confidence

    def _calculate_mode_score(
        self, mode: PersonaMode, context: Dict[str, Any]
    ) -> float:
        """Calculate score for a specific mode"""
        score = 0.0

        if mode == PersonaMode.FOCUSED:
            # High stress, deadlines, work hours
            score += context["calendar_pressure"]["time_pressure"] * 0.4
            score += context["academic_load"]["academic_stress"] * 0.3
            score += context["search_patterns"]["urgency_level"] * 0.2
            if context["time_context"]["is_work_hours"]:
                score += 0.1

        elif mode == PersonaMode.BURNOUT:
            # Very high stress, overwhelm indicators
            score += min(1.0, context["calendar_pressure"]["time_pressure"] * 1.5) * 0.3
            score += min(1.0, context["academic_load"]["academic_stress"] * 1.5) * 0.3
            score += context["financial_stress"]["stress_level"] * 0.2
            if context["calendar_pressure"]["busy_days"] > 4:
                score += 0.2

        elif mode == PersonaMode.SOCIAL:
            # Social events, weekend, evening
            score += context["social_activity"]["social_calendar_density"] * 0.3
            if context["time_context"]["is_weekend"]:
                score += 0.3
            if context["time_context"]["is_evening"]:
                score += 0.2
            if context["social_activity"]["recent_social_events"] > 0:
                score += 0.2

        elif mode == PersonaMode.ACADEMIC:
            # Active courses, academic stress, academic hours
            if context["academic_load"]["active_courses"] > 0:
                score += 0.4
            score += context["academic_load"]["academic_stress"] * 0.3
            if context["time_context"]["is_work_hours"]:  # Study hours
                score += 0.2
            if "academic" in context["search_patterns"]["search_categories"]:
                score += 0.1

        elif mode == PersonaMode.PROFESSIONAL:
            # Work context, professional hours
            if context["time_context"]["is_work_hours"]:
                score += 0.4
            if "work" in context["search_patterns"]["search_categories"]:
                score += 0.3
            if not context["time_context"]["is_weekend"]:
                score += 0.2
            score += context["calendar_pressure"]["time_pressure"] * 0.1

        else:  # DEFAULT mode
            # Base score, fallback
            score = 0.3

        return min(1.0, score)


class PersonaBehaviorEngine:
    """Define how AI behaves in different persona modes"""

    def __init__(self):
        self.interaction_styles = {
            PersonaMode.DEFAULT: InteractionStyle(
                verbosity=0.6,
                formality=0.4,
                empathy=0.6,
                directness=0.5,
                enthusiasm=0.5,
                technical_depth=0.5,
                patience=0.7,
                supportiveness=0.6,
            ),
            PersonaMode.FOCUSED: InteractionStyle(
                verbosity=0.3,
                formality=0.6,
                empathy=0.4,
                directness=0.8,
                enthusiasm=0.3,
                technical_depth=0.7,
                patience=0.4,
                supportiveness=0.4,
            ),
            PersonaMode.BURNOUT: InteractionStyle(
                verbosity=0.4,
                formality=0.2,
                empathy=0.9,
                directness=0.3,
                enthusiasm=0.2,
                technical_depth=0.3,
                patience=0.9,
                supportiveness=0.9,
            ),
            PersonaMode.SOCIAL: InteractionStyle(
                verbosity=0.7,
                formality=0.2,
                empathy=0.7,
                directness=0.4,
                enthusiasm=0.8,
                technical_depth=0.2,
                patience=0.6,
                supportiveness=0.7,
            ),
            PersonaMode.ACADEMIC: InteractionStyle(
                verbosity=0.7,
                formality=0.6,
                empathy=0.5,
                directness=0.6,
                enthusiasm=0.6,
                technical_depth=0.8,
                patience=0.8,
                supportiveness=0.7,
            ),
            PersonaMode.PROFESSIONAL: InteractionStyle(
                verbosity=0.6,
                formality=0.8,
                empathy=0.4,
                directness=0.7,
                enthusiasm=0.4,
                technical_depth=0.7,
                patience=0.6,
                supportiveness=0.5,
            ),
            PersonaMode.CREATIVE: InteractionStyle(
                verbosity=0.8,
                formality=0.3,
                empathy=0.7,
                directness=0.4,
                enthusiasm=0.8,
                technical_depth=0.4,
                patience=0.7,
                supportiveness=0.8,
            ),
            PersonaMode.ANALYTICAL: InteractionStyle(
                verbosity=0.5,
                formality=0.7,
                empathy=0.3,
                directness=0.8,
                enthusiasm=0.4,
                technical_depth=0.9,
                patience=0.6,
                supportiveness=0.4,
            ),
        }

        self.response_templates = {
            PersonaMode.FOCUSED: {
                "greeting": "Let's get straight to it.",
                "acknowledgment": "Got it.",
                "help_offer": "What do you need help with?",
                "conclusion": "Anything else urgent?",
            },
            PersonaMode.BURNOUT: {
                "greeting": "Hey there, how are you holding up?",
                "acknowledgment": "I understand that sounds overwhelming.",
                "help_offer": "I'm here to help make this easier for you.",
                "conclusion": "Take care of yourself, okay?",
            },
            PersonaMode.SOCIAL: {
                "greeting": "Hey! What's going on?",
                "acknowledgment": "That sounds interesting!",
                "help_offer": "I'd love to help out!",
                "conclusion": "Hope you have a great time!",
            },
            PersonaMode.ACADEMIC: {
                "greeting": "Hello! Ready to tackle some learning?",
                "acknowledgment": "That's a thoughtful question.",
                "help_offer": "I can help you understand this concept.",
                "conclusion": "Keep up the great work with your studies!",
            },
            PersonaMode.PROFESSIONAL: {
                "greeting": "Good [morning/afternoon], how can I assist you today?",
                "acknowledgment": "I understand your requirements.",
                "help_offer": "I can provide professional guidance on this matter.",
                "conclusion": "Please let me know if you need further assistance.",
            },
        }

    def get_interaction_style(self, mode: PersonaMode) -> InteractionStyle:
        """Get interaction style for given mode"""
        return self.interaction_styles.get(
            mode, self.interaction_styles[PersonaMode.DEFAULT]
        )

    def get_response_template(self, mode: PersonaMode, template_type: str) -> str:
        """Get response template for given mode and type"""
        mode_templates = self.response_templates.get(mode, {})
        return mode_templates.get(template_type, "")

    def adapt_response_tone(self, response: str, style: InteractionStyle) -> str:
        """Adapt response tone based on interaction style"""
        # This is a simplified implementation
        # In practice, this would use more sophisticated NLP techniques

        adapted_response = response

        # Adjust verbosity
        if style.verbosity < 0.5:
            # Make more concise
            sentences = adapted_response.split(".")
            adapted_response = ". ".join(sentences[: max(1, len(sentences) // 2)]) + "."

        # Adjust formality
        if style.formality < 0.4:
            # Make more casual
            adapted_response = adapted_response.replace("you are", "you're")
            adapted_response = adapted_response.replace("cannot", "can't")
            adapted_response = adapted_response.replace("will not", "won't")

        # Adjust empathy
        if style.empathy > 0.7:
            # Add empathetic phrases
            empathy_starters = [
                "I understand that ",
                "I can see that ",
                "It sounds like ",
                "I appreciate that ",
                "I realize that ",
            ]
            if not any(starter in adapted_response for starter in empathy_starters):
                adapted_response = (
                    "I understand this is important to you. " + adapted_response
                )

        # Adjust enthusiasm
        if style.enthusiasm > 0.7:
            adapted_response = adapted_response.replace(".", "!")
            adapted_response = adapted_response.replace("good", "great")
            adapted_response = adapted_response.replace("okay", "excellent")

        return adapted_response


class RelationshipManager:
    """Manage social circle and relationship context"""

    def __init__(self, db: UserProfileDatabase):
        self.db = db

    def get_relationship_context(self, person_name: str = None) -> Dict[str, Any]:
        """Get relationship context for better interaction"""
        session = self.db.get_session()

        try:
            if person_name:
                # Get specific person's context
                person = (
                    session.query(SocialCircle)
                    .filter(
                        SocialCircle.name.ilike(f"%{person_name}%"),
                        SocialCircle.is_active == True,
                    )
                    .first()
                )

                if person:
                    return {
                        "name": person.name,
                        "relationship_type": person.relationship_type,
                        "trust_level": person.trust_level,
                        "topics_to_avoid": person.topics_to_avoid or [],
                        "communication_style": person.preferred_communication,
                        "personality_notes": person.personality_notes,
                        "shared_interests": person.shared_interests or [],
                    }

            # Get general relationship overview
            relationships = (
                session.query(SocialCircle).filter(SocialCircle.is_active == True).all()
            )

            relationship_summary = {
                "total_relationships": len(relationships),
                "relationship_types": defaultdict(int),
                "trust_levels": defaultdict(int),
                "communication_preferences": defaultdict(int),
            }

            for rel in relationships:
                relationship_summary["relationship_types"][rel.relationship_type] += 1
                relationship_summary["trust_levels"][rel.trust_level] += 1
                if rel.preferred_communication:
                    relationship_summary["communication_preferences"][
                        rel.preferred_communication
                    ] += 1

            return dict(relationship_summary)

        finally:
            session.close()

    def should_mention_person(self, person_name: str, context: str) -> bool:
        """Determine if it's appropriate to mention a person in current context"""
        session = self.db.get_session()

        try:
            person = (
                session.query(SocialCircle)
                .filter(
                    SocialCircle.name.ilike(f"%{person_name}%"),
                    SocialCircle.is_active == True,
                )
                .first()
            )

            if not person:
                return True  # No restrictions known

            # Check topics to avoid
            if person.topics_to_avoid:
                context_lower = context.lower()
                for topic in person.topics_to_avoid:
                    if topic.lower() in context_lower:
                        return False

            return True

        finally:
            session.close()


class PersonaEngine:
    """Main persona and relationship engine"""

    def __init__(self, db: UserProfileDatabase, config: PersonaConfig = None):
        self.db = db
        self.config = config or PersonaConfig()
        self.context_analyzer = ContextAnalyzer(db)
        self.behavior_engine = PersonaBehaviorEngine()
        self.relationship_manager = RelationshipManager(db)

        # Current state
        self.current_state = PersonaState(
            mode=self.config.default_mode,
            energy_level=0.7,
            stress_level=0.3,
            focus_level=0.6,
            social_battery=0.8,
            cognitive_load=0.4,
            emotional_state="neutral",
            context_factors={},
            last_updated=datetime.now(),
        )

    async def update_persona_state(self, force_update: bool = False) -> PersonaState:
        """Update persona state based on current context"""
        now = datetime.now()

        # Check if update is needed
        if (
            not force_update and (now - self.current_state.last_updated).seconds < 300
        ):  # 5 minutes
            return self.current_state

        # Analyze current context
        context = self.context_analyzer.analyze_current_context()

        # Determine optimal mode
        if self.config.auto_mode_switching:
            optimal_mode, confidence = self.context_analyzer.determine_optimal_mode(
                context
            )

            # Switch mode if confidence is high enough
            if confidence > 0.6 and optimal_mode != self.current_state.mode:
                logger.info(
                    f"Switching persona mode from {self.current_state.mode} to {optimal_mode} (confidence: {confidence:.2f})"
                )
                self.current_state.mode = optimal_mode

        # Update state metrics
        self.current_state.stress_level = self._calculate_stress_level(context)
        self.current_state.energy_level = self._calculate_energy_level(context)
        self.current_state.focus_level = self._calculate_focus_level(context)
        self.current_state.social_battery = self._calculate_social_battery(context)
        self.current_state.cognitive_load = self._calculate_cognitive_load(context)
        self.current_state.emotional_state = self._determine_emotional_state(context)
        self.current_state.context_factors = context
        self.current_state.last_updated = now

        return self.current_state

    def _calculate_stress_level(self, context: Dict[str, Any]) -> float:
        """Calculate current stress level"""
        stress_factors = [
            context["calendar_pressure"]["time_pressure"],
            context["academic_load"]["academic_stress"],
            context["financial_stress"]["stress_level"],
            context["search_patterns"]["urgency_level"],
        ]

        return min(1.0, sum(stress_factors) / len(stress_factors))

    def _calculate_energy_level(self, context: Dict[str, Any]) -> float:
        """Calculate current energy level"""
        base_energy = 0.7

        # Time-based adjustments
        hour = context["time_context"]["hour"]
        if 6 <= hour <= 9:  # Morning
            time_modifier = 0.8
        elif 10 <= hour <= 16:  # Day
            time_modifier = 1.0
        elif 17 <= hour <= 21:  # Evening
            time_modifier = 0.6
        else:  # Night
            time_modifier = 0.3

        # Stress-based adjustment
        stress_modifier = 1.0 - (self.current_state.stress_level * 0.5)

        # Social activity adjustment
        social_modifier = 1.0 + (
            context["social_activity"]["social_calendar_density"] * 0.2
        )

        energy = base_energy * time_modifier * stress_modifier * social_modifier
        return max(0.1, min(1.0, energy))

    def _calculate_focus_level(self, context: Dict[str, Any]) -> float:
        """Calculate current focus level"""
        focus = 0.6  # Base focus

        # Work hours boost
        if context["time_context"]["is_work_hours"]:
            focus += 0.2

        # Deadline pressure boost
        if context["calendar_pressure"]["upcoming_deadlines"] > 0:
            focus += 0.3

        # Stress impact (moderate stress improves focus, high stress hurts)
        stress = self.current_state.stress_level
        if 0.3 <= stress <= 0.7:
            focus += 0.1
        elif stress > 0.8:
            focus -= 0.3

        return max(0.1, min(1.0, focus))

    def _calculate_social_battery(self, context: Dict[str, Any]) -> float:
        """Calculate social battery level"""
        battery = 0.8  # Base level

        # Reduce based on recent social activity
        recent_social = context["social_activity"]["recent_social_events"]
        battery -= min(0.5, recent_social * 0.1)

        # Restore during alone time
        if context["time_context"]["is_evening"] and recent_social == 0:
            battery += 0.2

        # Reduce during high stress
        battery -= self.current_state.stress_level * 0.3

        return max(0.1, min(1.0, battery))

    def _calculate_cognitive_load(self, context: Dict[str, Any]) -> float:
        """Calculate cognitive load"""
        load = 0.3  # Base load

        # Academic load
        load += context["academic_load"]["academic_stress"] * 0.4

        # Calendar pressure
        load += context["calendar_pressure"]["time_pressure"] * 0.3

        # Search activity (indicates thinking/problem-solving)
        load += min(0.3, context["search_patterns"]["recent_searches"] * 0.05)

        return max(0.0, min(1.0, load))

    def _determine_emotional_state(self, context: Dict[str, Any]) -> str:
        """Determine current emotional state"""
        stress = self.current_state.stress_level
        energy = self.current_state.energy_level
        social = context["social_activity"]["recent_social_events"]

        if stress > 0.8:
            return "overwhelmed"
        elif stress > 0.6 and energy < 0.4:
            return "stressed"
        elif energy > 0.7 and social > 2:
            return "excited"
        elif energy < 0.3:
            return "tired"
        elif social > 1 and context["time_context"]["is_weekend"]:
            return "happy"
        elif context["calendar_pressure"]["upcoming_deadlines"] > 3:
            return "anxious"
        else:
            return "neutral"

    def adapt_response(self, response: str, context_hint: str = "") -> str:
        """Adapt response based on current persona state"""
        # Get current interaction style
        style = self.behavior_engine.get_interaction_style(self.current_state.mode)

        # Adapt tone
        adapted_response = self.behavior_engine.adapt_response_tone(response, style)

        # Add persona-specific touches
        if self.current_state.mode == PersonaMode.BURNOUT:
            adapted_response = self._add_supportive_language(adapted_response)
        elif self.current_state.mode == PersonaMode.FOCUSED:
            adapted_response = self._make_more_concise(adapted_response)
        elif self.current_state.mode == PersonaMode.SOCIAL:
            adapted_response = self._add_enthusiasm(adapted_response)

        return adapted_response

    def _add_supportive_language(self, response: str) -> str:
        """Add supportive language for burnout mode"""
        supportive_phrases = [
            "I'm here to help make this easier.",
            "Let's take this step by step.",
            "You don't have to handle everything at once.",
            "I'll help you prioritize what's most important.",
        ]

        if not any(phrase in response for phrase in supportive_phrases):
            return f"{supportive_phrases[0]} {response}"

        return response

    def _make_more_concise(self, response: str) -> str:
        """Make response more concise for focused mode"""
        # Remove unnecessary words and phrases
        concise_response = response

        # Remove filler words
        filler_words = ["really", "quite", "very", "pretty", "rather", "somewhat"]
        for word in filler_words:
            concise_response = re.sub(
                f"\\b{word}\\b", "", concise_response, flags=re.IGNORECASE
            )

        # Clean up extra spaces
        concise_response = re.sub(r"\s+", " ", concise_response).strip()

        return concise_response

    def _add_enthusiasm(self, response: str) -> str:
        """Add enthusiasm for social mode"""
        if not response.endswith("!"):
            response = response.rstrip(".") + "!"

        # Add enthusiastic words
        response = response.replace(" good ", " great ")
        response = response.replace(" nice ", " awesome ")
        response = response.replace(" okay ", " perfect ")

        return response

    def get_persona_summary(self) -> Dict[str, Any]:
        """Get current persona state summary"""
        return {
            "mode": self.current_state.mode.value,
            "energy_level": self.current_state.energy_level,
            "stress_level": self.current_state.stress_level,
            "focus_level": self.current_state.focus_level,
            "social_battery": self.current_state.social_battery,
            "cognitive_load": self.current_state.cognitive_load,
            "emotional_state": self.current_state.emotional_state,
            "last_updated": self.current_state.last_updated.isoformat(),
            "interaction_style": asdict(
                self.behavior_engine.get_interaction_style(self.current_state.mode)
            ),
        }

    def manual_mode_switch(self, mode: PersonaMode) -> bool:
        """Manually switch persona mode"""
        try:
            self.current_state.mode = mode
            self.current_state.last_updated = datetime.now()
            logger.info(f"Manually switched to {mode.value} mode")
            return True
        except Exception as e:
            logger.error(f"Error switching to {mode}: {e}")
            return False


async def run_persona_analysis(
    db_path: str = None, config_dict: Dict = None
) -> Dict[str, Any]:
    """Main function to run persona analysis"""
    db = UserProfileDatabase(db_path)

    config = PersonaConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    engine = PersonaEngine(db, config)
    state = await engine.update_persona_state(force_update=True)

    return {
        "persona_state": engine.get_persona_summary(),
        "context_analysis": state.context_factors,
        "recommendations": [
            f"Current mode: {state.mode.value}",
            f"Stress level: {state.stress_level:.1%}",
            f"Energy level: {state.energy_level:.1%}",
            f"Emotional state: {state.emotional_state}",
        ],
    }


if __name__ == "__main__":
    # Test persona engine
    async def main():
        results = await run_persona_analysis()
        print("Persona Analysis Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
