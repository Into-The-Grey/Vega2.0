#!/usr/bin/env python3
"""
VEGA INTERACTION HISTORY & DIALOGUE CONDITIONING SYSTEM
======================================================

This system tracks all interactions between Vega and the user, analyzing
conversation patterns to automatically adjust personality, tone, and timing.

Key Features:
- Conversation logging with context preservation
- Response quality analysis and pattern recognition
- Automatic tone adjustment based on user feedback
- Dialogue conditioning for improved future interactions
- User preference learning and adaptation
- Context-aware conversation memory

This system learns what the user likes, how they communicate,
and adapts its approach accordingly over time.
"""

import os
import re
import json
import sqlite3
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics

logger = logging.getLogger(__name__)


class ResponseQuality(Enum):
    EXCELLENT = "excellent"  # User engaged deeply, asked follow-ups
    GOOD = "good"  # User responded positively
    NEUTRAL = "neutral"  # User acknowledged briefly
    POOR = "poor"  # User gave minimal response
    NEGATIVE = "negative"  # User expressed annoyance/dismissal
    IGNORED = "ignored"  # User completely ignored


class ConversationContext(Enum):
    WORK_FOCUSED = "work_focused"
    EXPLORATORY = "exploratory"
    PROBLEM_SOLVING = "problem_solving"
    CASUAL = "casual"
    STRESSED = "stressed"
    LEARNING = "learning"


@dataclass
class InteractionRecord:
    """Complete record of a single interaction"""

    id: str
    timestamp: datetime
    vega_initiative: str  # What Vega said to initiate
    user_response: str  # How user responded
    response_quality: ResponseQuality
    conversation_context: ConversationContext
    spontaneity_mode: str  # Which personality mode was used
    interaction_tone: str  # Tone Vega used
    user_engagement_score: float  # 0-1 score of user engagement
    response_time_seconds: float  # How long user took to respond
    follow_up_questions: int  # Number of follow-up questions from user
    user_satisfaction_signals: List[str]  # Positive/negative signals detected
    context_relevance_score: float  # How relevant Vega's comment was
    conversation_thread_id: str  # Links related conversations
    metadata: Dict[str, Any]


@dataclass
class ConversationPattern:
    """Learned pattern about user preferences"""

    pattern_id: str
    pattern_type: str  # "tone_preference", "timing_preference", "topic_interest", etc.
    context_conditions: Dict[str, Any]  # When this pattern applies
    success_rate: float
    confidence: float
    sample_size: int
    last_updated: datetime
    pattern_data: Dict[str, Any]


@dataclass
class UserPreferences:
    """Consolidated user preferences learned over time"""

    preferred_tones: Dict[str, float]  # Tone -> success rate
    optimal_interaction_times: List[int]  # Hours when user is most receptive
    topic_interests: Dict[str, float]  # Topic -> engagement score
    conversation_styles: Dict[str, float]  # Style -> preference score
    interruption_tolerance: float  # How tolerant user is of interruptions
    response_patterns: Dict[str, Any]  # How user typically responds
    last_updated: datetime


class InteractionHistoryDB:
    """Database manager for interaction history"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize interaction history database"""
        conn = sqlite3.connect(self.db_path)

        # Main interactions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                vega_initiative TEXT NOT NULL,
                user_response TEXT,
                response_quality TEXT,
                conversation_context TEXT,
                spontaneity_mode TEXT,
                interaction_tone TEXT,
                user_engagement_score REAL,
                response_time_seconds REAL,
                follow_up_questions INTEGER DEFAULT 0,
                user_satisfaction_signals TEXT,
                context_relevance_score REAL,
                conversation_thread_id TEXT,
                metadata TEXT
            )
        """
        )

        # Conversation patterns table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                context_conditions TEXT,
                success_rate REAL,
                confidence REAL,
                sample_size INTEGER,
                last_updated TEXT,
                pattern_data TEXT
            )
        """
        )

        # User preferences summary table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                preferred_tones TEXT,
                optimal_interaction_times TEXT,
                topic_interests TEXT,
                conversation_styles TEXT,
                interruption_tolerance REAL,
                response_patterns TEXT,
                last_updated TEXT
            )
        """
        )

        # Conversation threads table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_threads (
                thread_id TEXT PRIMARY KEY,
                started_timestamp TEXT,
                ended_timestamp TEXT,
                total_interactions INTEGER,
                average_engagement REAL,
                conversation_summary TEXT,
                outcome_quality TEXT
            )
        """
        )

        # Create indices for performance
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_thread ON interactions(conversation_thread_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_quality ON interactions(response_quality)"
        )

        conn.commit()
        conn.close()

    def log_interaction(self, interaction: InteractionRecord):
        """Log a new interaction to the database"""
        try:
            conn = sqlite3.connect(self.db_path)

            conn.execute(
                """
                INSERT OR REPLACE INTO interactions (
                    id, timestamp, vega_initiative, user_response, response_quality,
                    conversation_context, spontaneity_mode, interaction_tone,
                    user_engagement_score, response_time_seconds, follow_up_questions,
                    user_satisfaction_signals, context_relevance_score,
                    conversation_thread_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    interaction.id,
                    interaction.timestamp.isoformat(),
                    interaction.vega_initiative,
                    interaction.user_response,
                    interaction.response_quality.value,
                    interaction.conversation_context.value,
                    interaction.spontaneity_mode,
                    interaction.interaction_tone,
                    interaction.user_engagement_score,
                    interaction.response_time_seconds,
                    interaction.follow_up_questions,
                    json.dumps(interaction.user_satisfaction_signals),
                    interaction.context_relevance_score,
                    interaction.conversation_thread_id,
                    json.dumps(interaction.metadata),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")

    def get_recent_interactions(self, days: int = 7) -> List[InteractionRecord]:
        """Get recent interactions for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()

            cursor = conn.execute(
                """
                SELECT * FROM interactions 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """,
                (cutoff_time,),
            )

            interactions = []
            for row in cursor.fetchall():
                interaction = self._row_to_interaction(row)
                if interaction:
                    interactions.append(interaction)

            conn.close()
            return interactions

        except Exception as e:
            logger.error(f"Failed to get recent interactions: {e}")
            return []

    def _row_to_interaction(self, row) -> Optional[InteractionRecord]:
        """Convert database row to InteractionRecord"""
        try:
            return InteractionRecord(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                vega_initiative=row[2],
                user_response=row[3] or "",
                response_quality=ResponseQuality(row[4]),
                conversation_context=ConversationContext(row[5]),
                spontaneity_mode=row[6],
                interaction_tone=row[7],
                user_engagement_score=row[8] or 0.0,
                response_time_seconds=row[9] or 0.0,
                follow_up_questions=row[10] or 0,
                user_satisfaction_signals=json.loads(row[11] or "[]"),
                context_relevance_score=row[12] or 0.0,
                conversation_thread_id=row[13],
                metadata=json.loads(row[14] or "{}"),
            )
        except Exception as e:
            logger.error(f"Error converting row to interaction: {e}")
            return None


class DialogueConditioner:
    """Analyzes conversation patterns and conditions future interactions"""

    def __init__(self, db_path: Path):
        self.db = InteractionHistoryDB(db_path)
        self.preferences_cache = None
        self.patterns_cache = {}
        self.cache_expiry = datetime.now()

    async def analyze_user_response(
        self,
        vega_message: str,
        user_response: str,
        conversation_context: ConversationContext,
    ) -> InteractionRecord:
        """Analyze a user response and create interaction record"""

        # Generate unique interaction ID
        interaction_id = hashlib.md5(
            f"{datetime.now().isoformat()}_{vega_message}_{user_response}".encode()
        ).hexdigest()[:12]

        # Analyze response quality
        response_quality = self._analyze_response_quality(user_response)

        # Calculate engagement score
        engagement_score = self._calculate_engagement_score(user_response)

        # Detect satisfaction signals
        satisfaction_signals = self._detect_satisfaction_signals(user_response)

        # Count follow-up questions
        follow_up_questions = self._count_follow_up_questions(user_response)

        # Calculate context relevance (would need more context)
        context_relevance = 0.7  # Placeholder

        # Create interaction record
        interaction = InteractionRecord(
            id=interaction_id,
            timestamp=datetime.now(),
            vega_initiative=vega_message,
            user_response=user_response,
            response_quality=response_quality,
            conversation_context=conversation_context,
            spontaneity_mode="curiosity",  # Would come from personality engine
            interaction_tone="professional",  # Would come from personality engine
            user_engagement_score=engagement_score,
            response_time_seconds=0.0,  # Would need to track timing
            follow_up_questions=follow_up_questions,
            user_satisfaction_signals=satisfaction_signals,
            context_relevance_score=context_relevance,
            conversation_thread_id=self._generate_thread_id(),
            metadata={},
        )

        # Log the interaction
        self.db.log_interaction(interaction)

        # Update patterns and preferences
        await self._update_conversation_patterns(interaction)
        await self._update_user_preferences(interaction)

        return interaction

    def _analyze_response_quality(self, user_response: str) -> ResponseQuality:
        """Analyze the quality of user response"""
        if not user_response or user_response.strip() == "":
            return ResponseQuality.IGNORED

        response_lower = user_response.lower().strip()

        # Negative indicators
        negative_indicators = [
            "not interested",
            "busy",
            "later",
            "not now",
            "stop",
            "annoying",
            "leave me alone",
            "don't care",
            "whatever",
        ]
        if any(indicator in response_lower for indicator in negative_indicators):
            return ResponseQuality.NEGATIVE

        # Positive indicators
        positive_indicators = [
            "interesting",
            "tell me more",
            "yes",
            "please",
            "thanks",
            "helpful",
            "good point",
            "exactly",
            "absolutely",
            "definitely",
        ]
        excellent_indicators = [
            "?",
            "how",
            "why",
            "what about",
            "can you",
            "tell me about",
        ]

        # Count indicators
        positive_count = sum(
            1 for indicator in positive_indicators if indicator in response_lower
        )
        excellent_count = sum(
            1 for indicator in excellent_indicators if indicator in response_lower
        )

        # Length analysis
        word_count = len(response_lower.split())

        if excellent_count > 0 or word_count > 15:
            return ResponseQuality.EXCELLENT
        elif positive_count > 0 or word_count > 8:
            return ResponseQuality.GOOD
        elif word_count > 3:
            return ResponseQuality.NEUTRAL
        else:
            return ResponseQuality.POOR

    def _calculate_engagement_score(self, user_response: str) -> float:
        """Calculate user engagement score (0-1)"""
        if not user_response:
            return 0.0

        score = 0.0

        # Length factor (longer responses usually indicate more engagement)
        word_count = len(user_response.split())
        length_score = min(1.0, word_count / 20.0)  # Max at 20 words
        score += length_score * 0.3

        # Question factor (questions indicate curiosity)
        question_count = user_response.count("?")
        question_score = min(1.0, question_count / 3.0)  # Max at 3 questions
        score += question_score * 0.4

        # Emotional indicators
        positive_emotions = ["interested", "curious", "excited", "thanks", "helpful"]
        emotion_score = sum(
            0.2 for emotion in positive_emotions if emotion in user_response.lower()
        )
        score += min(0.3, emotion_score)

        return min(1.0, score)

    def _detect_satisfaction_signals(self, user_response: str) -> List[str]:
        """Detect positive and negative satisfaction signals"""
        signals = []
        response_lower = user_response.lower()

        # Positive signals
        positive_signals = {
            "gratitude": ["thank", "thanks", "appreciate"],
            "interest": ["interesting", "cool", "neat", "wow"],
            "engagement": ["tell me more", "how", "why", "what about"],
            "approval": ["good", "helpful", "useful", "perfect", "exactly"],
        }

        # Negative signals
        negative_signals = {
            "dismissal": ["whatever", "ok", "sure", "fine"],
            "annoyance": ["annoying", "stop", "busy", "not now"],
            "disinterest": ["don't care", "not interested", "boring"],
        }

        # Check for positive signals
        for signal_type, keywords in positive_signals.items():
            if any(keyword in response_lower for keyword in keywords):
                signals.append(f"positive_{signal_type}")

        # Check for negative signals
        for signal_type, keywords in negative_signals.items():
            if any(keyword in response_lower for keyword in keywords):
                signals.append(f"negative_{signal_type}")

        return signals

    def _count_follow_up_questions(self, user_response: str) -> int:
        """Count follow-up questions in user response"""
        return user_response.count("?")

    def _generate_thread_id(self) -> str:
        """Generate conversation thread ID"""
        # For now, use a simple time-based ID
        # In a real implementation, this would track conversation continuity
        return f"thread_{datetime.now().strftime('%Y%m%d_%H')}"

    async def _update_conversation_patterns(self, interaction: InteractionRecord):
        """Update learned conversation patterns"""
        try:
            # Pattern: Tone effectiveness
            tone_pattern_id = f"tone_effectiveness_{interaction.interaction_tone}"
            await self._update_pattern(
                pattern_id=tone_pattern_id,
                pattern_type="tone_effectiveness",
                context_conditions={"tone": interaction.interaction_tone},
                success_indicator=interaction.response_quality
                not in [ResponseQuality.NEGATIVE, ResponseQuality.IGNORED],
                pattern_data={
                    "tone": interaction.interaction_tone,
                    "engagement": interaction.user_engagement_score,
                },
            )

            # Pattern: Mode effectiveness
            mode_pattern_id = f"mode_effectiveness_{interaction.spontaneity_mode}"
            await self._update_pattern(
                pattern_id=mode_pattern_id,
                pattern_type="mode_effectiveness",
                context_conditions={"mode": interaction.spontaneity_mode},
                success_indicator=interaction.user_engagement_score > 0.5,
                pattern_data={
                    "mode": interaction.spontaneity_mode,
                    "context": interaction.conversation_context.value,
                },
            )

            # Pattern: Time preferences
            hour = interaction.timestamp.hour
            time_pattern_id = f"time_preference_{hour}"
            await self._update_pattern(
                pattern_id=time_pattern_id,
                pattern_type="time_preference",
                context_conditions={"hour": hour},
                success_indicator=interaction.response_quality
                in [ResponseQuality.EXCELLENT, ResponseQuality.GOOD],
                pattern_data={
                    "hour": hour,
                    "day_of_week": interaction.timestamp.weekday(),
                },
            )

        except Exception as e:
            logger.error(f"Error updating conversation patterns: {e}")

    async def _update_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        context_conditions: Dict[str, Any],
        success_indicator: bool,
        pattern_data: Dict[str, Any],
    ):
        """Update a specific conversation pattern"""
        try:
            conn = sqlite3.connect(self.db.db_path)

            # Get existing pattern
            cursor = conn.execute(
                "SELECT success_rate, sample_size FROM conversation_patterns WHERE pattern_id = ?",
                (pattern_id,),
            )
            result = cursor.fetchone()

            if result:
                # Update existing pattern
                old_success_rate, old_sample_size = result
                new_sample_size = old_sample_size + 1

                # Calculate new success rate
                old_successes = old_success_rate * old_sample_size
                new_successes = old_successes + (1 if success_indicator else 0)
                new_success_rate = new_successes / new_sample_size

                # Calculate confidence (higher with more samples)
                confidence = min(
                    0.95, new_sample_size / 20.0
                )  # Max confidence at 20 samples

                conn.execute(
                    """
                    UPDATE conversation_patterns 
                    SET success_rate = ?, confidence = ?, sample_size = ?, 
                        last_updated = ?, pattern_data = ?
                    WHERE pattern_id = ?
                """,
                    (
                        new_success_rate,
                        confidence,
                        new_sample_size,
                        datetime.now().isoformat(),
                        json.dumps(pattern_data),
                        pattern_id,
                    ),
                )
            else:
                # Create new pattern
                success_rate = 1.0 if success_indicator else 0.0
                confidence = 0.1  # Low confidence with only one sample

                conn.execute(
                    """
                    INSERT INTO conversation_patterns (
                        pattern_id, pattern_type, context_conditions, success_rate,
                        confidence, sample_size, last_updated, pattern_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern_id,
                        pattern_type,
                        json.dumps(context_conditions),
                        success_rate,
                        confidence,
                        1,
                        datetime.now().isoformat(),
                        json.dumps(pattern_data),
                    ),
                )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error updating pattern {pattern_id}: {e}")

    async def _update_user_preferences(self, interaction: InteractionRecord):
        """Update consolidated user preferences"""
        try:
            # Get recent interactions for preference calculation
            recent_interactions = self.db.get_recent_interactions(days=30)

            if not recent_interactions:
                return

            # Calculate tone preferences
            tone_scores = {}
            for inter in recent_interactions:
                tone = inter.interaction_tone
                if tone not in tone_scores:
                    tone_scores[tone] = []
                tone_scores[tone].append(inter.user_engagement_score)

            tone_preferences = {
                tone: statistics.mean(scores) for tone, scores in tone_scores.items()
            }

            # Calculate optimal interaction times
            hourly_engagement = {}
            for inter in recent_interactions:
                hour = inter.timestamp.hour
                if hour not in hourly_engagement:
                    hourly_engagement[hour] = []
                hourly_engagement[hour].append(inter.user_engagement_score)

            optimal_times = [
                hour
                for hour, scores in hourly_engagement.items()
                if statistics.mean(scores) > 0.6  # Good engagement threshold
            ]

            # Calculate topic interests (would need more sophisticated analysis)
            topic_interests = {}  # Placeholder

            # Calculate conversation styles
            conversation_styles = {}  # Placeholder

            # Calculate interruption tolerance
            interruption_scores = [
                inter.user_engagement_score
                for inter in recent_interactions
                if inter.response_quality
                not in [ResponseQuality.NEGATIVE, ResponseQuality.IGNORED]
            ]
            interruption_tolerance = (
                statistics.mean(interruption_scores) if interruption_scores else 0.5
            )

            # Save preferences
            preferences = UserPreferences(
                preferred_tones=tone_preferences,
                optimal_interaction_times=optimal_times,
                topic_interests=topic_interests,
                conversation_styles=conversation_styles,
                interruption_tolerance=interruption_tolerance,
                response_patterns={},  # Placeholder
                last_updated=datetime.now(),
            )

            await self._save_user_preferences(preferences)

        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")

    async def _save_user_preferences(self, preferences: UserPreferences):
        """Save user preferences to database"""
        try:
            conn = sqlite3.connect(self.db.db_path)

            conn.execute(
                """
                INSERT OR REPLACE INTO user_preferences (
                    id, preferred_tones, optimal_interaction_times, topic_interests,
                    conversation_styles, interruption_tolerance, response_patterns,
                    last_updated
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    json.dumps(preferences.preferred_tones),
                    json.dumps(preferences.optimal_interaction_times),
                    json.dumps(preferences.topic_interests),
                    json.dumps(preferences.conversation_styles),
                    preferences.interruption_tolerance,
                    json.dumps(preferences.response_patterns),
                    preferences.last_updated.isoformat(),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error saving user preferences: {e}")

    async def get_conditioning_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for conditioning future interactions"""
        try:
            # Refresh cache if needed
            if datetime.now() > self.cache_expiry:
                await self._refresh_caches()

            recommendations = {}

            # Tone recommendations
            if self.preferences_cache and self.preferences_cache.preferred_tones:
                best_tone = max(
                    self.preferences_cache.preferred_tones.items(), key=lambda x: x[1]
                )[0]
                recommendations["preferred_tone"] = best_tone

            # Timing recommendations
            if (
                self.preferences_cache
                and self.preferences_cache.optimal_interaction_times
            ):
                recommendations["optimal_hours"] = (
                    self.preferences_cache.optimal_interaction_times
                )

            # Interruption guidance
            if self.preferences_cache:
                recommendations["interruption_tolerance"] = (
                    self.preferences_cache.interruption_tolerance
                )

            return recommendations

        except Exception as e:
            logger.error(f"Error getting conditioning recommendations: {e}")
            return {}

    async def _refresh_caches(self):
        """Refresh preference and pattern caches"""
        try:
            conn = sqlite3.connect(self.db.db_path)

            # Load user preferences
            cursor = conn.execute("SELECT * FROM user_preferences WHERE id = 1")
            pref_row = cursor.fetchone()

            if pref_row:
                self.preferences_cache = UserPreferences(
                    preferred_tones=json.loads(pref_row[1] or "{}"),
                    optimal_interaction_times=json.loads(pref_row[2] or "[]"),
                    topic_interests=json.loads(pref_row[3] or "{}"),
                    conversation_styles=json.loads(pref_row[4] or "{}"),
                    interruption_tolerance=pref_row[5] or 0.5,
                    response_patterns=json.loads(pref_row[6] or "{}"),
                    last_updated=datetime.fromisoformat(pref_row[7]),
                )

            # Load conversation patterns
            cursor = conn.execute(
                "SELECT * FROM conversation_patterns WHERE confidence > 0.3"
            )
            for row in cursor.fetchall():
                pattern = ConversationPattern(
                    pattern_id=row[0],
                    pattern_type=row[1],
                    context_conditions=json.loads(row[2] or "{}"),
                    success_rate=row[3],
                    confidence=row[4],
                    sample_size=row[5],
                    last_updated=datetime.fromisoformat(row[6]),
                    pattern_data=json.loads(row[7] or "{}"),
                )
                self.patterns_cache[pattern.pattern_id] = pattern

            conn.close()

            # Set cache expiry to 1 hour
            self.cache_expiry = datetime.now() + timedelta(hours=1)

        except Exception as e:
            logger.error(f"Error refreshing caches: {e}")


# Integration functions for the main ambient loop
async def log_interaction(
    state_dir: Path,
    vega_message: str,
    user_response: str,
    context: ConversationContext = ConversationContext.CASUAL,
) -> InteractionRecord:
    """Log an interaction and get analysis"""
    db_path = state_dir / "interaction_history.db"
    conditioner = DialogueConditioner(db_path)
    return await conditioner.analyze_user_response(vega_message, user_response, context)


async def get_dialogue_conditioning(state_dir: Path) -> Dict[str, Any]:
    """Get current dialogue conditioning recommendations"""
    db_path = state_dir / "interaction_history.db"
    conditioner = DialogueConditioner(db_path)
    return await conditioner.get_conditioning_recommendations()
