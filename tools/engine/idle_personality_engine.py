#!/usr/bin/env python3
"""
VEGA IDLE PERSONALITY ENGINE
============================

The core of Vega's contextual intelligence and conversational awareness.
This engine evaluates user context, constructs relevant questions/thoughts,
and manages different modes of spontaneous interaction.

Like JARVIS, this system:
- Observes patterns in user behavior
- Constructs meaningful, relevant questions
- Adapts tone and approach based on interaction history
- Never speaks randomly - always with purpose
- Respects user's current context and mood

Spontaneity Modes:
- CURIOSITY: Asks clarifying questions about user interests
- CONCERN: Notices inconsistencies, stress signals, or unusual patterns
- RECOMMENDATION: Offers helpful tools, optimizations, or suggestions
- REFLECTION: Prompts memory completion or seeks user opinions
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

import httpx
from ..config import get_config

logger = logging.getLogger(__name__)


class SpontaneityMode(Enum):
    CURIOSITY = "curiosity"
    CONCERN = "concern"
    RECOMMENDATION = "recommendation"
    REFLECTION = "reflection"
    OBSERVATION = "observation"


class InteractionTone(Enum):
    PLAYFUL = "playful"
    PROFESSIONAL = "professional"
    MINIMAL = "minimal"
    CARING = "caring"
    DIRECT = "direct"


@dataclass
class UserContext:
    """Captures current user context for personality evaluation"""

    recent_commands: List[str]
    recent_files: List[str]
    calendar_events: List[Dict[str, Any]]
    time_patterns: Dict[str, Any]
    stress_indicators: List[str]
    interests: List[str]
    current_projects: List[str]
    interaction_history: List[Dict[str, Any]]


@dataclass
class PersonalityThought:
    """Represents a potential interaction thought"""

    id: str
    mode: SpontaneityMode
    content: str
    relevance_score: float
    confidence: float
    context_triggers: List[str]
    estimated_interruption_cost: float
    generated_at: datetime
    user_context_hash: str


@dataclass
class InteractionTemplate:
    """Template for generating contextual interactions"""

    mode: SpontaneityMode
    pattern: str
    template: str
    min_relevance: float
    triggers: List[str]


class UserProfiler:
    """Analyzes user behavior patterns and builds context"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.profile_db = state_dir / "user_profile.db"
        self._init_profile_db()

    def _init_profile_db(self):
        """Initialize user profile database"""
        conn = sqlite3.connect(self.profile_db)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_activities (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                activity_type TEXT,
                content TEXT,
                context TEXT,
                metadata TEXT
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                last_updated TEXT,
                frequency INTEGER DEFAULT 1
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_context (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                context_type TEXT,
                data TEXT,
                relevance_score REAL
            )
        """
        )

        conn.commit()
        conn.close()

    def log_user_activity(
        self, activity_type: str, content: str, context: Dict[str, Any] = None
    ):
        """Log user activity for pattern analysis"""
        try:
            conn = sqlite3.connect(self.profile_db)

            conn.execute(
                """
                INSERT INTO user_activities (timestamp, activity_type, content, context, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    activity_type,
                    content,
                    json.dumps(context or {}),
                    json.dumps({"source": "ambient_observer"}),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to log user activity: {e}")

    def analyze_recent_activity(self, hours: int = 24) -> UserContext:
        """Analyze recent user activity to build context"""
        try:
            conn = sqlite3.connect(self.profile_db)
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            cursor = conn.execute(
                """
                SELECT activity_type, content, context, timestamp
                FROM user_activities 
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 100
            """,
                (cutoff_time,),
            )

            activities = cursor.fetchall()
            conn.close()

            # Process activities into context
            recent_commands = []
            recent_files = []
            stress_indicators = []
            interests = []
            current_projects = []

            for activity_type, content, context_json, timestamp in activities:
                try:
                    context = json.loads(context_json) if context_json else {}

                    if activity_type == "command":
                        recent_commands.append(content)
                    elif activity_type == "file_access":
                        recent_files.append(content)
                    elif activity_type == "stress_signal":
                        stress_indicators.append(content)
                    elif activity_type == "interest":
                        interests.append(content)
                    elif activity_type == "project":
                        current_projects.append(content)

                except json.JSONDecodeError:
                    pass

            return UserContext(
                recent_commands=recent_commands[:20],
                recent_files=recent_files[:20],
                calendar_events=[],  # TODO: Implement calendar integration
                time_patterns=self._analyze_time_patterns(activities),
                stress_indicators=stress_indicators,
                interests=list(set(interests)),
                current_projects=list(set(current_projects)),
                interaction_history=[],  # TODO: Load from interaction history
            )

        except Exception as e:
            logger.error(f"Failed to analyze user activity: {e}")
            return UserContext([], [], [], {}, [], [], [], [])

    def _analyze_time_patterns(self, activities: List[Tuple]) -> Dict[str, Any]:
        """Analyze temporal patterns in user activity"""
        if not activities:
            return {}

        # Group activities by hour
        hourly_activity = {}
        for _, _, _, timestamp_str in activities:
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                hour = timestamp.hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            except:
                pass

        # Find peak activity hours
        if hourly_activity:
            peak_hour = max(hourly_activity.keys(), key=lambda h: hourly_activity[h])
            return {
                "peak_activity_hour": peak_hour,
                "hourly_distribution": hourly_activity,
                "most_active_period": (
                    "morning"
                    if peak_hour < 12
                    else "afternoon" if peak_hour < 18 else "evening"
                ),
            }

        return {}


class PersonalityEngine:
    """Core personality and conversation engine"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.profiler = UserProfiler(state_dir)
        self.personality_log = state_dir / "personality_memory.jsonl"
        self.current_tone = InteractionTone.PROFESSIONAL
        self.interaction_templates = self._load_interaction_templates()
        self.llm_client = self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client for personality generation"""
        try:
            config = get_config()
            return httpx.AsyncClient(
                base_url="http://127.0.0.1:11434", timeout=30.0  # Ollama default
            )
        except Exception:
            return None

    def _load_interaction_templates(self) -> List[InteractionTemplate]:
        """Load predefined interaction templates"""
        return [
            # CURIOSITY templates
            InteractionTemplate(
                mode=SpontaneityMode.CURIOSITY,
                pattern=r"(?i)(research|paper|study|learn)",
                template="I noticed you've been exploring {topic}. What drew you to this area?",
                min_relevance=0.6,
                triggers=["repeated_research", "new_topic"],
            ),
            InteractionTemplate(
                mode=SpontaneityMode.CURIOSITY,
                pattern=r"(?i)(script|automation|tool)",
                template="You've been working with a lot of scripts lately. Are you building something specific?",
                min_relevance=0.7,
                triggers=["coding_pattern", "automation_focus"],
            ),
            # CONCERN templates
            InteractionTemplate(
                mode=SpontaneityMode.CONCERN,
                pattern=r"(?i)(error|fail|crash|bug)",
                template="I've noticed some error patterns in your recent work. Would you like me to help analyze them?",
                min_relevance=0.8,
                triggers=["error_frequency", "troubleshooting"],
            ),
            InteractionTemplate(
                mode=SpontaneityMode.CONCERN,
                pattern=r"late_night_activity",
                template="You've been working quite late recently. Everything alright?",
                min_relevance=0.6,
                triggers=["unusual_hours", "stress_signals"],
            ),
            # RECOMMENDATION templates
            InteractionTemplate(
                mode=SpontaneityMode.RECOMMENDATION,
                pattern=r"(?i)(inefficient|slow|manual)",
                template="I see you're doing {task} manually. I could help automate that for you.",
                min_relevance=0.7,
                triggers=["inefficiency_detected", "automation_opportunity"],
            ),
            InteractionTemplate(
                mode=SpontaneityMode.RECOMMENDATION,
                pattern=r"(?i)(duplicate|repeat)",
                template="You seem to be repeating similar tasks. Want me to create a shortcut?",
                min_relevance=0.75,
                triggers=["repetitive_behavior", "optimization_chance"],
            ),
            # REFLECTION templates
            InteractionTemplate(
                mode=SpontaneityMode.REFLECTION,
                pattern=r"project_milestone",
                template="How do you feel about the progress on {project} so far?",
                min_relevance=0.6,
                triggers=["project_activity", "milestone_detected"],
            ),
            InteractionTemplate(
                mode=SpontaneityMode.REFLECTION,
                pattern=r"learning_session",
                template="What's the most interesting thing you've learned about {topic} recently?",
                min_relevance=0.65,
                triggers=["educational_content", "skill_development"],
            ),
        ]

    async def evaluate_interaction_opportunity(
        self, user_context: UserContext
    ) -> Optional[PersonalityThought]:
        """Evaluate whether to initiate interaction and generate thought"""

        # Calculate context hash for deduplication
        context_str = json.dumps(
            {
                "commands": user_context.recent_commands[:5],
                "files": user_context.recent_files[:5],
                "interests": user_context.interests,
                "projects": user_context.current_projects,
            },
            sort_keys=True,
        )
        context_hash = hashlib.md5(context_str.encode()).hexdigest()

        # Check recent thoughts to avoid repetition
        if await self._is_recent_context_hash(context_hash):
            return None

        # Evaluate each spontaneity mode
        best_thought = None
        best_score = 0.0

        for mode in SpontaneityMode:
            thought = await self._generate_thought_for_mode(
                mode, user_context, context_hash
            )
            if thought and thought.relevance_score > best_score:
                best_thought = thought
                best_score = thought.relevance_score

        # Only return if relevance exceeds threshold
        if best_thought and best_thought.relevance_score >= 0.6:
            await self._log_personality_thought(best_thought)
            return best_thought

        return None

    async def _generate_thought_for_mode(
        self, mode: SpontaneityMode, user_context: UserContext, context_hash: str
    ) -> Optional[PersonalityThought]:
        """Generate a thought for a specific spontaneity mode"""

        try:
            # Find relevant templates for this mode
            relevant_templates = [
                t for t in self.interaction_templates if t.mode == mode
            ]

            for template in relevant_templates:
                # Check if template triggers apply
                triggers_found = []

                # Analyze user context for triggers
                if mode == SpontaneityMode.CURIOSITY:
                    triggers_found = self._detect_curiosity_triggers(
                        user_context, template
                    )
                elif mode == SpontaneityMode.CONCERN:
                    triggers_found = self._detect_concern_triggers(
                        user_context, template
                    )
                elif mode == SpontaneityMode.RECOMMENDATION:
                    triggers_found = self._detect_recommendation_triggers(
                        user_context, template
                    )
                elif mode == SpontaneityMode.REFLECTION:
                    triggers_found = self._detect_reflection_triggers(
                        user_context, template
                    )

                if triggers_found:
                    # Generate contextual content
                    content = await self._generate_contextual_content(
                        template, user_context, triggers_found
                    )

                    if content:
                        # Calculate relevance score
                        relevance = self._calculate_relevance_score(
                            user_context, triggers_found, template
                        )

                        return PersonalityThought(
                            id=hashlib.md5(
                                f"{mode.value}_{content}_{datetime.now().isoformat()}".encode()
                            ).hexdigest()[:12],
                            mode=mode,
                            content=content,
                            relevance_score=relevance,
                            confidence=0.8,  # Base confidence
                            context_triggers=triggers_found,
                            estimated_interruption_cost=self._estimate_interruption_cost(
                                mode
                            ),
                            generated_at=datetime.now(),
                            user_context_hash=context_hash,
                        )

        except Exception as e:
            logger.error(f"Error generating thought for mode {mode}: {e}")

        return None

    def _detect_curiosity_triggers(
        self, user_context: UserContext, template: InteractionTemplate
    ) -> List[str]:
        """Detect triggers for curiosity-based interactions"""
        triggers = []

        # Check for research patterns
        research_keywords = [
            "paper",
            "study",
            "research",
            "learn",
            "tutorial",
            "documentation",
        ]
        recent_text = " ".join(
            user_context.recent_commands + user_context.recent_files
        ).lower()

        if any(keyword in recent_text for keyword in research_keywords):
            triggers.append("research_activity")

        # Check for new topics/interests
        if len(user_context.interests) > 0:
            triggers.append("new_interests")

        # Check for coding/script activity
        coding_keywords = ["script", "code", "python", "javascript", "automation"]
        if any(keyword in recent_text for keyword in coding_keywords):
            triggers.append("coding_activity")

        return triggers

    def _detect_concern_triggers(
        self, user_context: UserContext, template: InteractionTemplate
    ) -> List[str]:
        """Detect triggers for concern-based interactions"""
        triggers = []

        # Check for stress indicators
        if user_context.stress_indicators:
            triggers.append("stress_detected")

        # Check for unusual time patterns
        if user_context.time_patterns.get("peak_activity_hour", 12) > 22:
            triggers.append("late_night_activity")

        # Check for error patterns
        error_keywords = ["error", "fail", "crash", "bug", "exception"]
        recent_text = " ".join(user_context.recent_commands).lower()
        if any(keyword in recent_text for keyword in error_keywords):
            triggers.append("error_patterns")

        return triggers

    def _detect_recommendation_triggers(
        self, user_context: UserContext, template: InteractionTemplate
    ) -> List[str]:
        """Detect triggers for recommendation-based interactions"""
        triggers = []

        # Check for repetitive commands
        if len(user_context.recent_commands) >= 3:
            command_counts = {}
            for cmd in user_context.recent_commands[:10]:
                # Simplify command for pattern detection
                simple_cmd = " ".join(cmd.split()[:2])  # First two words
                command_counts[simple_cmd] = command_counts.get(simple_cmd, 0) + 1

            if any(count >= 3 for count in command_counts.values()):
                triggers.append("repetitive_commands")

        # Check for manual processes that could be automated
        automation_keywords = ["manual", "copy", "paste", "repeated"]
        recent_text = " ".join(user_context.recent_commands).lower()
        if any(keyword in recent_text for keyword in automation_keywords):
            triggers.append("automation_opportunity")

        return triggers

    def _detect_reflection_triggers(
        self, user_context: UserContext, template: InteractionTemplate
    ) -> List[str]:
        """Detect triggers for reflection-based interactions"""
        triggers = []

        # Check for project activity
        if user_context.current_projects:
            triggers.append("project_activity")

        # Check for learning/educational content
        learning_keywords = ["learn", "tutorial", "course", "study", "practice"]
        recent_text = " ".join(
            user_context.recent_files + user_context.recent_commands
        ).lower()
        if any(keyword in recent_text for keyword in learning_keywords):
            triggers.append("learning_activity")

        return triggers

    async def _generate_contextual_content(
        self,
        template: InteractionTemplate,
        user_context: UserContext,
        triggers: List[str],
    ) -> Optional[str]:
        """Generate contextual content using template and LLM"""

        try:
            # Start with template
            content = template.template

            # Extract key context for substitution
            if "{topic}" in content:
                # Try to identify the main topic from recent activity
                topic = self._extract_main_topic(user_context)
                content = content.replace("{topic}", topic or "this subject")

            if "{task}" in content:
                # Try to identify the main task
                task = self._extract_main_task(user_context)
                content = content.replace("{task}", task or "this task")

            if "{project}" in content:
                # Use current project
                project = (
                    user_context.current_projects[0]
                    if user_context.current_projects
                    else "your current project"
                )
                content = content.replace("{project}", project)

            # Enhance with LLM if available
            if self.llm_client:
                enhanced_content = await self._enhance_with_llm(
                    content, user_context, triggers
                )
                if enhanced_content:
                    content = enhanced_content

            return content

        except Exception as e:
            logger.error(f"Error generating contextual content: {e}")
            return None

    def _extract_main_topic(self, user_context: UserContext) -> Optional[str]:
        """Extract the main topic from user context"""
        # Simple keyword extraction from interests and recent activity
        interests = user_context.interests
        if interests:
            return interests[0]

        # Fallback to analyzing recent files/commands
        recent_text = " ".join(user_context.recent_files + user_context.recent_commands)

        # Look for technical topics
        tech_keywords = {
            "ai": "AI and machine learning",
            "python": "Python development",
            "javascript": "JavaScript development",
            "data": "data analysis",
            "quantum": "quantum computing",
            "security": "cybersecurity",
            "automation": "automation and scripting",
        }

        for keyword, topic in tech_keywords.items():
            if keyword in recent_text.lower():
                return topic

        return None

    def _extract_main_task(self, user_context: UserContext) -> Optional[str]:
        """Extract the main task from user context"""
        if not user_context.recent_commands:
            return None

        # Analyze recent commands for task patterns
        commands_text = " ".join(user_context.recent_commands).lower()

        if "git" in commands_text:
            return "version control"
        elif "python" in commands_text or "script" in commands_text:
            return "scripting"
        elif "file" in commands_text or "copy" in commands_text:
            return "file management"
        elif "install" in commands_text or "pip" in commands_text:
            return "package installation"

        return "your current work"

    async def _enhance_with_llm(
        self, base_content: str, user_context: UserContext, triggers: List[str]
    ) -> Optional[str]:
        """Enhance content using LLM for more natural conversation"""
        if not self.llm_client:
            return None

        try:
            prompt = f"""
You are Vega, an ambient AI assistant. Based on the user's recent activity, enhance this interaction to be more natural and contextually relevant.

Base interaction: {base_content}
User context: Recent commands include {user_context.recent_commands[:3]}, working on {user_context.current_projects}
Triggers: {triggers}
Tone: {self.current_tone.value}

Make it concise (1-2 sentences), respectful, and genuinely helpful. Avoid being pushy or overly familiar.
"""

            response = await self.llm_client.post(
                "/api/generate",
                json={
                    "model": "llama3.2:3b",  # Use small model for quick responses
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "max_tokens": 100},
                },
            )

            if response.status_code == 200:
                result = response.json()
                enhanced = result.get("response", "").strip()
                if enhanced and len(enhanced) < 200:  # Keep it concise
                    return enhanced

        except Exception as e:
            logger.debug(f"LLM enhancement failed: {e}")

        return None

    def _calculate_relevance_score(
        self,
        user_context: UserContext,
        triggers: List[str],
        template: InteractionTemplate,
    ) -> float:
        """Calculate relevance score for a potential interaction"""
        base_score = template.min_relevance

        # Boost score based on triggers
        trigger_boost = len(triggers) * 0.1

        # Boost based on recent activity relevance
        activity_boost = 0.0
        if user_context.recent_commands or user_context.recent_files:
            activity_boost = 0.15

        # Reduce score if user seems busy (many recent commands)
        if len(user_context.recent_commands) > 15:
            activity_boost -= 0.1

        # Boost for current projects
        if user_context.current_projects:
            activity_boost += 0.1

        final_score = min(1.0, base_score + trigger_boost + activity_boost)
        return final_score

    def _estimate_interruption_cost(self, mode: SpontaneityMode) -> float:
        """Estimate the cost of interrupting the user"""
        # Different modes have different interruption costs
        cost_map = {
            SpontaneityMode.CURIOSITY: 0.3,  # Low cost, usually welcome
            SpontaneityMode.REFLECTION: 0.4,  # Medium-low cost
            SpontaneityMode.RECOMMENDATION: 0.2,  # Very low cost, helpful
            SpontaneityMode.CONCERN: 0.6,  # Higher cost, but important
            SpontaneityMode.OBSERVATION: 0.5,  # Medium cost
        }

        return cost_map.get(mode, 0.5)

    async def _is_recent_context_hash(self, context_hash: str) -> bool:
        """Check if we've recently processed this context hash"""
        try:
            if not self.personality_log.exists():
                return False

            cutoff_time = datetime.now() - timedelta(
                hours=2
            )  # 2-hour deduplication window

            with open(self.personality_log, "r") as f:
                for line in f:
                    try:
                        thought_data = json.loads(line)
                        thought_time = datetime.fromisoformat(
                            thought_data.get("generated_at", "")
                        )

                        if (
                            thought_time > cutoff_time
                            and thought_data.get("user_context_hash") == context_hash
                        ):
                            return True

                    except (json.JSONDecodeError, ValueError):
                        continue

            return False

        except Exception as e:
            logger.error(f"Error checking recent context hash: {e}")
            return False

    async def _log_personality_thought(self, thought: PersonalityThought):
        """Log personality thought to persistent storage"""
        try:
            thought_data = asdict(thought)
            thought_data["generated_at"] = thought.generated_at.isoformat()

            with open(self.personality_log, "a") as f:
                f.write(json.dumps(thought_data) + "\n")

        except Exception as e:
            logger.error(f"Error logging personality thought: {e}")

    def adjust_tone(self, interaction_response: str, response_quality: str):
        """Adjust personality tone based on user response"""
        if response_quality == "positive":
            # User engaged positively, can be slightly more conversational
            if self.current_tone == InteractionTone.PROFESSIONAL:
                self.current_tone = InteractionTone.CARING
        elif response_quality == "brief" or response_quality == "ignored":
            # User gave brief response or ignored, be more minimal
            self.current_tone = InteractionTone.MINIMAL
        elif response_quality == "negative":
            # User was negative, be more professional and direct
            self.current_tone = InteractionTone.DIRECT

        logger.info(f"Adjusted tone to: {self.current_tone.value}")


# Integration function for the main ambient loop
async def evaluate_personality_interaction(
    state_dir: Path, user_context: UserContext = None
) -> Optional[str]:
    """Main entry point for personality evaluation from ambient loop"""
    try:
        engine = PersonalityEngine(state_dir)

        if not user_context:
            # Build user context from profiler
            user_context = engine.profiler.analyze_recent_activity()

        # Evaluate interaction opportunity
        thought = await engine.evaluate_interaction_opportunity(user_context)

        if thought:
            logger.info(f"🤖 Vega thought ({thought.mode.value}): {thought.content}")
            return thought.content

        return None

    except Exception as e:
        logger.error(f"Error in personality evaluation: {e}")
        return None
