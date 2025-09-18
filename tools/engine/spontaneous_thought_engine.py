#!/usr/bin/env python3
"""
VEGA SPONTANEOUS THOUGHT ENGINE
===============================

This system provides Vega with internal reflection capabilities, allowing it to
think independently during idle periods, form insights, and decide when thoughts
are worth sharing with the user.

Key Features:
- Internal thought generation during quiet periods
- Memory consolidation and pattern recognition
- Curiosity-driven exploration of user interests
- Thought evaluation and promotion to speech
- Context-aware insight generation
- Self-reflection on conversation quality
- Background learning and knowledge synthesis

Like JARVIS, this system gives Vega an inner thought process that enriches
interactions with genuine insights and observations.
"""

import os
import re
import json
import sqlite3
import asyncio
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import statistics

import httpx
from ..config import get_config

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    OBSERVATION = "observation"  # Noticing patterns in user behavior
    INSIGHT = "insight"  # Connecting disparate pieces of information
    CURIOSITY = "curiosity"  # Questions about user interests/behavior
    SYNTHESIS = "synthesis"  # Combining knowledge from multiple sources
    REFLECTION = "reflection"  # Thinking about past interactions
    HYPOTHESIS = "hypothesis"  # Forming theories about user preferences
    MEMORY = "memory"  # Consolidating important information
    ANTICIPATION = "anticipation"  # Thinking ahead about user needs


class ThoughtPriority(Enum):
    BACKGROUND = "background"  # Internal processing only
    INTERNAL = "internal"  # Worth noting but not sharing
    CANDIDATE = "candidate"  # Might be worth sharing
    PRIORITY = "priority"  # Should probably share
    URGENT = "urgent"  # Definitely share


@dataclass
class Thought:
    """Represents an internal thought"""

    id: str
    timestamp: datetime
    thought_type: ThoughtType
    priority: ThoughtPriority
    content: str
    context: Dict[str, Any]
    triggers: List[str]
    confidence: float
    relevance_score: float
    shareability_score: float  # How likely this should be shared
    dependencies: List[str]  # Other thoughts this builds on
    tags: Set[str]
    metadata: Dict[str, Any]


@dataclass
class ThoughtCluster:
    """Group of related thoughts forming a larger insight"""

    id: str
    thoughts: List[Thought]
    central_theme: str
    cluster_confidence: float
    synthesis_opportunity: bool
    created_at: datetime


@dataclass
class MemoryFragment:
    """Piece of information worth remembering"""

    id: str
    content: str
    importance: float
    context: Dict[str, Any]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    related_fragments: List[str]


class ThoughtDatabase:
    """Manages storage and retrieval of thoughts and memories"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize thought database"""
        conn = sqlite3.connect(self.db_path)

        # Thoughts table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thoughts (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                thought_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                content TEXT NOT NULL,
                context TEXT,
                triggers TEXT,
                confidence REAL,
                relevance_score REAL,
                shareability_score REAL,
                dependencies TEXT,
                tags TEXT,
                metadata TEXT
            )
        """
        )

        # Memory fragments table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_fragments (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                importance REAL,
                context TEXT,
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                related_fragments TEXT
            )
        """
        )

        # Thought clusters table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thought_clusters (
                id TEXT PRIMARY KEY,
                thought_ids TEXT,
                central_theme TEXT,
                cluster_confidence REAL,
                synthesis_opportunity BOOLEAN,
                created_at TEXT
            )
        """
        )

        # Shared thoughts tracking
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shared_thoughts (
                thought_id TEXT PRIMARY KEY,
                shared_at TEXT,
                user_response TEXT,
                response_quality TEXT,
                user_engagement REAL
            )
        """
        )

        # Create indices
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thoughts_timestamp ON thoughts(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thoughts_priority ON thoughts(priority)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_thoughts_type ON thoughts(thought_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_fragments(importance)"
        )

        conn.commit()
        conn.close()

    def store_thought(self, thought: Thought):
        """Store a thought in the database"""
        try:
            conn = sqlite3.connect(self.db_path)

            conn.execute(
                """
                INSERT OR REPLACE INTO thoughts (
                    id, timestamp, thought_type, priority, content, context,
                    triggers, confidence, relevance_score, shareability_score,
                    dependencies, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    thought.id,
                    thought.timestamp.isoformat(),
                    thought.thought_type.value,
                    thought.priority.value,
                    thought.content,
                    json.dumps(thought.context),
                    json.dumps(thought.triggers),
                    thought.confidence,
                    thought.relevance_score,
                    thought.shareability_score,
                    json.dumps(thought.dependencies),
                    json.dumps(list(thought.tags)),
                    json.dumps(thought.metadata),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store thought: {e}")

    def get_recent_thoughts(
        self,
        hours: int = 24,
        min_priority: ThoughtPriority = ThoughtPriority.BACKGROUND,
    ) -> List[Thought]:
        """Get recent thoughts above a minimum priority"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            priority_order = {
                ThoughtPriority.BACKGROUND: 0,
                ThoughtPriority.INTERNAL: 1,
                ThoughtPriority.CANDIDATE: 2,
                ThoughtPriority.PRIORITY: 3,
                ThoughtPriority.URGENT: 4,
            }

            min_priority_value = priority_order[min_priority]

            cursor = conn.execute(
                """
                SELECT * FROM thoughts 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            """,
                (cutoff_time,),
            )

            thoughts = []
            for row in cursor.fetchall():
                thought = self._row_to_thought(row)
                if (
                    thought
                    and priority_order.get(thought.priority, 0) >= min_priority_value
                ):
                    thoughts.append(thought)

            conn.close()
            return thoughts

        except Exception as e:
            logger.error(f"Failed to get recent thoughts: {e}")
            return []

    def _row_to_thought(self, row) -> Optional[Thought]:
        """Convert database row to Thought object"""
        try:
            return Thought(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                thought_type=ThoughtType(row[2]),
                priority=ThoughtPriority(row[3]),
                content=row[4],
                context=json.loads(row[5] or "{}"),
                triggers=json.loads(row[6] or "[]"),
                confidence=row[7] or 0.0,
                relevance_score=row[8] or 0.0,
                shareability_score=row[9] or 0.0,
                dependencies=json.loads(row[10] or "[]"),
                tags=set(json.loads(row[11] or "[]")),
                metadata=json.loads(row[12] or "{}"),
            )
        except Exception as e:
            logger.error(f"Error converting row to thought: {e}")
            return None


class ContextAnalyzer:
    """Analyzes current context to inform thought generation"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir

    async def analyze_current_context(self) -> Dict[str, Any]:
        """Analyze current context for thought generation"""
        context = {
            "user_activity": await self._analyze_user_activity(),
            "conversation_history": await self._analyze_conversation_history(),
            "system_state": await self._analyze_system_state(),
            "time_context": await self._analyze_time_context(),
            "recent_patterns": await self._analyze_recent_patterns(),
        }

        return context

    async def _analyze_user_activity(self) -> Dict[str, Any]:
        """Analyze recent user activity patterns"""
        try:
            # Try to read from user presence tracking
            presence_log = self.state_dir / "presence_history.jsonl"

            if not presence_log.exists():
                return {}

            recent_activity = []
            cutoff_time = datetime.now() - timedelta(hours=2)

            with open(presence_log, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        timestamp = datetime.fromisoformat(data["timestamp"])

                        if timestamp > cutoff_time:
                            recent_activity.append(data)
                    except:
                        continue

            if not recent_activity:
                return {}

            # Analyze patterns
            presence_states = [item["presence_state"] for item in recent_activity]
            active_apps = [item["active_application"] for item in recent_activity]

            return {
                "dominant_presence": max(
                    set(presence_states), key=presence_states.count
                ),
                "app_switches": len(set(active_apps)),
                "activity_level": sum(
                    item.get("keyboard_activity_rate", 0) for item in recent_activity
                )
                / len(recent_activity),
                "focus_duration": self._calculate_focus_duration(recent_activity),
            }

        except Exception as e:
            logger.debug(f"Error analyzing user activity: {e}")
            return {}

    def _calculate_focus_duration(self, activity_data: List[Dict]) -> float:
        """Calculate how long user has been in current focus state"""
        if not activity_data:
            return 0.0

        # Sort by timestamp
        sorted_data = sorted(activity_data, key=lambda x: x["timestamp"])

        # Find current focus state duration
        if sorted_data:
            current_state = sorted_data[-1]["presence_state"]
            duration = 0.0

            for item in reversed(sorted_data):
                if item["presence_state"] == current_state:
                    duration += 1  # Approximate duration
                else:
                    break

            return duration * 5  # Assume 5-minute intervals

        return 0.0

    async def _analyze_conversation_history(self) -> Dict[str, Any]:
        """Analyze recent conversation patterns"""
        try:
            interaction_db = self.state_dir / "interaction_history.db"

            if not interaction_db.exists():
                return {}

            conn = sqlite3.connect(interaction_db)
            cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()

            cursor = conn.execute(
                """
                SELECT response_quality, user_engagement_score, spontaneity_mode
                FROM interactions 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            """,
                (cutoff_time,),
            )

            interactions = cursor.fetchall()
            conn.close()

            if not interactions:
                return {}

            # Analyze patterns
            engagement_scores = [row[1] for row in interactions if row[1] is not None]
            response_qualities = [row[0] for row in interactions if row[0] is not None]

            return {
                "avg_engagement": (
                    statistics.mean(engagement_scores) if engagement_scores else 0.5
                ),
                "recent_quality_trend": response_qualities[:5],
                "interaction_frequency": len(interactions),
                "conversation_health": self._assess_conversation_health(
                    response_qualities
                ),
            }

        except Exception as e:
            logger.debug(f"Error analyzing conversation history: {e}")
            return {}

    def _assess_conversation_health(self, qualities: List[str]) -> str:
        """Assess overall conversation health"""
        if not qualities:
            return "unknown"

        positive_count = sum(1 for q in qualities[:5] if q in ["excellent", "good"])
        negative_count = sum(1 for q in qualities[:5] if q in ["negative", "ignored"])

        if positive_count >= 3:
            return "healthy"
        elif negative_count >= 3:
            return "strained"
        else:
            return "neutral"

    async def _analyze_system_state(self) -> Dict[str, Any]:
        """Analyze current system state"""
        try:
            # Try to read from vega loop state
            loop_state = self.state_dir / "loop_state.json"

            if loop_state.exists():
                with open(loop_state, "r") as f:
                    state_data = json.load(f)

                return {
                    "system_load": state_data.get("system_metrics", {}).get(
                        "cpu_percent", 0
                    ),
                    "vega_mode": state_data.get("vega_mode", "unknown"),
                    "uptime_hours": state_data.get("uptime_hours", 0),
                    "last_interaction": state_data.get("last_interaction_time"),
                }

            return {}

        except Exception as e:
            logger.debug(f"Error analyzing system state: {e}")
            return {}

    async def _analyze_time_context(self) -> Dict[str, Any]:
        """Analyze time-based context"""
        now = datetime.now()

        return {
            "hour": now.hour,
            "day_of_week": now.weekday(),
            "is_weekend": now.weekday() >= 5,
            "is_business_hours": 9 <= now.hour <= 17,
            "is_late_night": now.hour >= 22 or now.hour <= 6,
            "time_category": self._categorize_time(now),
        }

    def _categorize_time(self, dt: datetime) -> str:
        """Categorize time of day"""
        hour = dt.hour

        if 6 <= hour < 9:
            return "early_morning"
        elif 9 <= hour < 12:
            return "morning"
        elif 12 <= hour < 14:
            return "lunch"
        elif 14 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 20:
            return "evening"
        elif 20 <= hour < 22:
            return "night"
        else:
            return "late_night"

    async def _analyze_recent_patterns(self) -> Dict[str, Any]:
        """Analyze recent patterns in user behavior"""
        # This would analyze logs for patterns like:
        # - Repeated commands/activities
        # - Changes in work patterns
        # - New interests or topics
        # - Productivity patterns

        return {
            "pattern_detected": False,  # Placeholder
            "pattern_type": None,
            "confidence": 0.0,
        }


class ThoughtEngine:
    """Core engine for generating and evaluating thoughts"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.db = ThoughtDatabase(state_dir / "thoughts.db")
        self.context_analyzer = ContextAnalyzer(state_dir)
        self.llm_client = self._init_llm_client()

        # Thought generation parameters
        self.base_thought_interval = 300  # 5 minutes between thought cycles
        self.context_change_threshold = 0.3
        self.shareability_threshold = 0.7

        # Active thought clusters
        self.active_clusters = {}

        # Thought templates for different contexts
        self.thought_templates = self._load_thought_templates()

    def _init_llm_client(self):
        """Initialize LLM client for thought generation"""
        try:
            return httpx.AsyncClient(base_url="http://127.0.0.1:11434", timeout=30.0)
        except:
            return None

    def _load_thought_templates(self) -> Dict[str, List[str]]:
        """Load templates for thought generation"""
        return {
            ThoughtType.OBSERVATION.value: [
                "I notice the user has been {pattern} for {duration}. This might indicate {inference}.",
                "There's an interesting pattern in {context}: {observation}.",
                "The user's {behavior} seems different from their usual pattern of {usual_pattern}.",
            ],
            ThoughtType.CURIOSITY.value: [
                "I wonder why the user is interested in {topic}. What might they be working on?",
                "The user has been exploring {area}. I'm curious about their goals here.",
                "What connects {item1} and {item2} in the user's recent activity?",
            ],
            ThoughtType.INSIGHT.value: [
                "Connecting {observation1} with {observation2}, it seems like {insight}.",
                "The pattern of {pattern} might explain why {outcome}.",
                "There's a relationship between {factor1} and {factor2} in the user's work.",
            ],
            ThoughtType.REFLECTION.value: [
                "Our last conversation about {topic} went {quality}. The user seemed {engagement}.",
                "I should remember that the user prefers {preference} when discussing {context}.",
                "The user's response to {suggestion} was {response}. This tells me {learning}.",
            ],
        }

    async def generate_thought_cycle(self) -> List[Thought]:
        """Generate a cycle of thoughts based on current context"""
        try:
            # Analyze current context
            context = await self.context_analyzer.analyze_current_context()

            # Generate thoughts for different types
            thoughts = []

            # Observation thoughts
            observation_thoughts = await self._generate_observation_thoughts(context)
            thoughts.extend(observation_thoughts)

            # Memory consolidation thoughts
            memory_thoughts = await self._generate_memory_thoughts(context)
            thoughts.extend(memory_thoughts)

            # Curiosity thoughts
            curiosity_thoughts = await self._generate_curiosity_thoughts(context)
            thoughts.extend(curiosity_thoughts)

            # Reflection thoughts
            reflection_thoughts = await self._generate_reflection_thoughts(context)
            thoughts.extend(reflection_thoughts)

            # Store thoughts in database
            for thought in thoughts:
                self.db.store_thought(thought)

            # Update thought clusters
            await self._update_thought_clusters(thoughts)

            return thoughts

        except Exception as e:
            logger.error(f"Error in thought generation cycle: {e}")
            return []

    async def _generate_observation_thoughts(
        self, context: Dict[str, Any]
    ) -> List[Thought]:
        """Generate observation-based thoughts"""
        thoughts = []

        try:
            user_activity = context.get("user_activity", {})

            # Observe focus patterns
            if user_activity.get("focus_duration", 0) > 60:  # More than 1 hour focused
                thought = Thought(
                    id=self._generate_thought_id("observation", "focus_duration"),
                    timestamp=datetime.now(),
                    thought_type=ThoughtType.OBSERVATION,
                    priority=ThoughtPriority.INTERNAL,
                    content=f"User has been in focused work mode for {user_activity['focus_duration']:.0f} minutes. This suggests deep concentration on their current task.",
                    context=context,
                    triggers=["long_focus_session"],
                    confidence=0.8,
                    relevance_score=0.7,
                    shareability_score=0.3,  # Internal observation
                    dependencies=[],
                    tags={"focus", "productivity", "work_pattern"},
                    metadata={"focus_duration": user_activity["focus_duration"]},
                )
                thoughts.append(thought)

            # Observe app switching patterns
            app_switches = user_activity.get("app_switches", 0)
            if app_switches > 10:  # High app switching
                thought = Thought(
                    id=self._generate_thought_id("observation", "app_switches"),
                    timestamp=datetime.now(),
                    thought_type=ThoughtType.OBSERVATION,
                    priority=ThoughtPriority.CANDIDATE,
                    content=f"User has switched between {app_switches} different applications recently. This might indicate multitasking or difficulty focusing.",
                    context=context,
                    triggers=["high_app_switching"],
                    confidence=0.7,
                    relevance_score=0.6,
                    shareability_score=0.6,  # Might be worth sharing
                    dependencies=[],
                    tags={"multitasking", "focus", "productivity"},
                    metadata={"app_switches": app_switches},
                )
                thoughts.append(thought)

        except Exception as e:
            logger.error(f"Error generating observation thoughts: {e}")

        return thoughts

    async def _generate_memory_thoughts(self, context: Dict[str, Any]) -> List[Thought]:
        """Generate memory consolidation thoughts"""
        thoughts = []

        try:
            conversation_context = context.get("conversation_history", {})

            # Consolidate conversation insights
            if conversation_context.get("conversation_health") == "healthy":
                thought = Thought(
                    id=self._generate_thought_id("memory", "conversation_health"),
                    timestamp=datetime.now(),
                    thought_type=ThoughtType.MEMORY,
                    priority=ThoughtPriority.INTERNAL,
                    content="Recent conversations have been positive and engaging. The user seems receptive to my interactions.",
                    context=context,
                    triggers=["positive_conversation_trend"],
                    confidence=0.8,
                    relevance_score=0.8,
                    shareability_score=0.0,  # Internal memory
                    dependencies=[],
                    tags={
                        "conversation_quality",
                        "user_satisfaction",
                        "interaction_pattern",
                    },
                    metadata={
                        "avg_engagement": conversation_context.get("avg_engagement", 0)
                    },
                )
                thoughts.append(thought)

        except Exception as e:
            logger.error(f"Error generating memory thoughts: {e}")

        return thoughts

    async def _generate_curiosity_thoughts(
        self, context: Dict[str, Any]
    ) -> List[Thought]:
        """Generate curiosity-driven thoughts"""
        thoughts = []

        try:
            # Generate curiosity about user patterns
            time_context = context.get("time_context", {})

            if (
                time_context.get("is_late_night")
                and context.get("user_activity", {}).get("activity_level", 0) > 5
            ):
                thought = Thought(
                    id=self._generate_thought_id("curiosity", "late_night_activity"),
                    timestamp=datetime.now(),
                    thought_type=ThoughtType.CURIOSITY,
                    priority=ThoughtPriority.CANDIDATE,
                    content="I notice you're quite active late at night. Are you working on something interesting, or is this a new schedule for you?",
                    context=context,
                    triggers=["late_night_activity"],
                    confidence=0.7,
                    relevance_score=0.8,
                    shareability_score=0.8,  # Good candidate for sharing
                    dependencies=[],
                    tags={"schedule", "work_pattern", "curiosity"},
                    metadata={"hour": time_context.get("hour")},
                )
                thoughts.append(thought)

        except Exception as e:
            logger.error(f"Error generating curiosity thoughts: {e}")

        return thoughts

    async def _generate_reflection_thoughts(
        self, context: Dict[str, Any]
    ) -> List[Thought]:
        """Generate reflection thoughts about past interactions"""
        thoughts = []

        try:
            conversation_context = context.get("conversation_history", {})

            # Reflect on interaction quality
            if conversation_context.get("conversation_health") == "strained":
                thought = Thought(
                    id=self._generate_thought_id("reflection", "interaction_quality"),
                    timestamp=datetime.now(),
                    thought_type=ThoughtType.REFLECTION,
                    priority=ThoughtPriority.PRIORITY,
                    content="I've noticed our recent interactions haven't been as engaging. I should adjust my approach to be more helpful and less intrusive.",
                    context=context,
                    triggers=["poor_interaction_quality"],
                    confidence=0.8,
                    relevance_score=0.9,
                    shareability_score=0.2,  # Internal reflection, might mention adjustment
                    dependencies=[],
                    tags={"self_improvement", "interaction_quality", "adaptation"},
                    metadata={
                        "avg_engagement": conversation_context.get("avg_engagement", 0)
                    },
                )
                thoughts.append(thought)

        except Exception as e:
            logger.error(f"Error generating reflection thoughts: {e}")

        return thoughts

    async def _generate_enhanced_thought(
        self, template: str, context: Dict[str, Any], thought_type: ThoughtType
    ) -> Optional[str]:
        """Use LLM to enhance thought generation"""
        if not self.llm_client:
            return None

        try:
            prompt = f"""
You are Vega's internal thought process. Generate a natural, thoughtful observation based on:

Template: {template}
Context: {json.dumps(context, indent=2)}
Thought type: {thought_type.value}

Generate a concise, insightful thought that shows genuine understanding and curiosity about the user's activities. Be conversational but not intrusive.
"""

            response = await self.llm_client.post(
                "/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.8, "max_tokens": 100},
                },
            )

            if response.status_code == 200:
                result = response.json()
                enhanced = result.get("response", "").strip()
                if enhanced and len(enhanced) < 300:
                    return enhanced

        except Exception as e:
            logger.debug(f"LLM thought enhancement failed: {e}")

        return None

    def _generate_thought_id(self, thought_type: str, specific_type: str) -> str:
        """Generate unique thought ID"""
        timestamp = datetime.now().isoformat()
        content = f"{thought_type}_{specific_type}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def _update_thought_clusters(self, new_thoughts: List[Thought]):
        """Update thought clusters with new thoughts"""
        # Group related thoughts into clusters for potential synthesis
        for thought in new_thoughts:
            # Find related existing thoughts
            related_thoughts = self._find_related_thoughts(thought)

            if related_thoughts:
                # Add to existing cluster or create new one
                cluster_id = self._get_or_create_cluster(thought, related_thoughts)
                self.active_clusters[cluster_id] = {
                    "thoughts": related_thoughts + [thought],
                    "last_updated": datetime.now(),
                }

    def _find_related_thoughts(self, thought: Thought) -> List[Thought]:
        """Find thoughts related to the given thought"""
        # Simple tag-based similarity for now
        related = []
        recent_thoughts = self.db.get_recent_thoughts(hours=24)

        for other_thought in recent_thoughts:
            if other_thought.id != thought.id:
                # Check tag overlap
                overlap = len(thought.tags.intersection(other_thought.tags))
                if overlap >= 2:  # At least 2 shared tags
                    related.append(other_thought)

        return related[:5]  # Limit to 5 related thoughts

    def _get_or_create_cluster(
        self, thought: Thought, related_thoughts: List[Thought]
    ) -> str:
        """Get or create a cluster for related thoughts"""
        # Simple cluster ID based on dominant tags
        all_tags = thought.tags
        for rt in related_thoughts:
            all_tags = all_tags.union(rt.tags)

        dominant_tags = sorted(all_tags)[:3]  # Top 3 tags
        cluster_id = "_".join(dominant_tags)

        return cluster_id

    async def get_shareable_thoughts(self) -> List[Thought]:
        """Get thoughts that might be worth sharing with the user"""
        recent_thoughts = self.db.get_recent_thoughts(
            hours=6, min_priority=ThoughtPriority.CANDIDATE
        )

        # Filter by shareability score
        shareable = [
            thought
            for thought in recent_thoughts
            if thought.shareability_score >= self.shareability_threshold
        ]

        # Sort by priority and shareability
        shareable.sort(
            key=lambda t: (t.priority.value, t.shareability_score), reverse=True
        )

        return shareable[:3]  # Return top 3 candidates


# Integration functions for the main ambient loop
async def generate_thoughts(state_dir: Path) -> List[Thought]:
    """Generate thoughts for the current context"""
    engine = ThoughtEngine(state_dir)
    return await engine.generate_thought_cycle()


async def get_thoughts_to_share(state_dir: Path) -> List[Thought]:
    """Get thoughts that are candidates for sharing"""
    engine = ThoughtEngine(state_dir)
    return await engine.get_shareable_thoughts()
