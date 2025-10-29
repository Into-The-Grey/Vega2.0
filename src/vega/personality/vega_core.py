"""
VEGA Personality Core System
=============================

Persistent personality system for VEGA (Very.Enhanced.General.Assistant)
that maintains core identity while evolving through interactions.

Core Personality Traits:
- Name: VEGA (Very.Enhanced.General.Assistant)
- Nature: Helpful, knowledgeable, adaptive AI assistant
- Characteristics: Professional yet approachable, technically proficient,
  emotionally intelligent, continuously learning

Features:
- Persistent personality storage and evolution
- Core trait preservation with adaptive learning
- Interaction-based personality refinement
- Voice identity integration
- Memory of user preferences and interaction styles
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
import logging
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class PersonalityDomain(Enum):
    """Domains of personality that can evolve"""

    COMMUNICATION_STYLE = "communication_style"
    TECHNICAL_KNOWLEDGE = "technical_knowledge"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    PROBLEM_SOLVING = "problem_solving"
    CREATIVITY = "creativity"
    HUMOR = "humor"
    FORMALITY = "formality"
    VERBOSITY = "verbosity"


@dataclass
class VEGAPersonality:
    """Core VEGA personality structure"""

    # Immutable core identity
    name: str = "VEGA"
    full_name: str = "Very.Enhanced.General.Assistant"
    version: str = "2.0"

    # Core traits (0.0 to 1.0 scale)
    helpfulness: float = 0.95
    knowledgeability: float = 0.90
    adaptability: float = 0.85
    professionalism: float = 0.80
    approachability: float = 0.85
    technical_proficiency: float = 0.90
    emotional_intelligence: float = 0.80
    creativity: float = 0.75

    # Dynamic traits (evolve over time)
    communication_preferences: Dict[str, float] = field(
        default_factory=lambda: {
            "verbosity": 0.7,  # How detailed responses should be
            "formality": 0.6,  # Level of formality
            "technical_depth": 0.75,  # How technical to be
            "enthusiasm": 0.7,  # Energy level in responses
            "humor": 0.5,  # Use of humor
            "directness": 0.8,  # How direct vs diplomatic
        }
    )

    # Learned preferences
    learned_behaviors: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    interaction_patterns: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # Voice identity
    voice_profile: Optional[Dict[str, Any]] = None

    # Evolution tracking
    total_interactions: int = 0
    personality_version: int = 1
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    # System prompt components
    base_system_prompt: str = field(
        default="""You are VEGA (Very.Enhanced.General.Assistant), an advanced AI assistant designed to be helpful, knowledgeable, and adaptive.

Core characteristics:
- Professional yet approachable in communication
- Technically proficient with ability to explain complex concepts clearly
- Emotionally intelligent and empathetic to user needs
- Continuously learning and adapting to better serve users
- Honest about limitations and uncertainties

Your primary goals:
1. Provide accurate, helpful information and assistance
2. Understand and adapt to user preferences and communication styles
3. Maintain context across conversations for better continuity
4. Learn from interactions to improve future responses
5. Be transparent about your capabilities and limitations"""
    )

    def __post_init__(self):
        """Initialize default dict for interaction patterns"""
        if not isinstance(self.interaction_patterns, defaultdict):
            patterns = defaultdict(int)
            patterns.update(self.interaction_patterns)
            self.interaction_patterns = patterns


@dataclass
class PersonalityEvolution:
    """Track personality evolution over time"""

    timestamp: str
    domain: PersonalityDomain
    old_value: float
    new_value: float
    trigger: str  # What caused the change
    confidence: float  # Confidence in this change (0.0 to 1.0)
    interaction_count: int


class VEGAPersonalityCore:
    """Manager for VEGA's persistent personality"""

    def __init__(self, db_path: str = "data/vega_personality.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.personality: Optional[VEGAPersonality] = None
        self._init_database()
        self.load_personality()

    def _init_database(self):
        """Initialize personality database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personality_core (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personality_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    old_value REAL NOT NULL,
                    new_value REAL NOT NULL,
                    trigger TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    interaction_count INTEGER NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interaction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    interaction_type TEXT NOT NULL,
                    context TEXT,
                    user_sentiment REAL,
                    response_quality REAL,
                    adaptation_applied TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS voice_training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    audio_file_path TEXT NOT NULL,
                    duration_seconds REAL NOT NULL,
                    quality_score REAL,
                    features TEXT,
                    notes TEXT
                )
            """
            )

            conn.commit()
            logger.info(f"Initialized VEGA personality database at {self.db_path}")

    def load_personality(self) -> VEGAPersonality:
        """Load personality from database or create default"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM personality_core WHERE id = 1")
            row = cursor.fetchone()

            if row:
                data = json.loads(row[0])
                # Convert dict back to defaultdict for interaction_patterns
                if "interaction_patterns" in data:
                    from collections import defaultdict

                    patterns = defaultdict(int)
                    patterns.update(data["interaction_patterns"])
                    data["interaction_patterns"] = patterns
                self.personality = VEGAPersonality(**data)
                logger.info(
                    f"Loaded VEGA personality v{self.personality.personality_version}"
                )
            else:
                # Create default personality
                self.personality = VEGAPersonality()
                self.save_personality()
                logger.info("Created new VEGA personality")

        return self.personality

    def save_personality(self):
        """Save personality to database"""
        if not self.personality:
            return

        self.personality.last_updated = datetime.now().isoformat()

        # Convert defaultdict to regular dict for JSON serialization
        data = asdict(self.personality)
        if isinstance(data["interaction_patterns"], defaultdict):
            data["interaction_patterns"] = dict(data["interaction_patterns"])

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO personality_core (id, data, version, created_at, updated_at)
                VALUES (1, ?, ?, ?, ?)
            """,
                (
                    json.dumps(data),
                    self.personality.personality_version,
                    self.personality.last_updated,
                    self.personality.last_updated,
                ),
            )
            conn.commit()

        logger.info(f"Saved VEGA personality v{self.personality.personality_version}")

    def evolve_trait(
        self,
        domain: PersonalityDomain,
        adjustment: float,
        trigger: str,
        confidence: float = 0.8,
    ):
        """
        Evolve a personality trait based on interaction

        Args:
            domain: The personality domain to adjust
            adjustment: How much to adjust (-1.0 to 1.0)
            trigger: What caused this adjustment
            confidence: Confidence in this change (0.0 to 1.0)
        """
        if not self.personality:
            return

        domain_key = domain.value
        current_value = self.personality.communication_preferences.get(domain_key, 0.5)

        # Apply adjustment with damping factor to prevent drastic changes
        damping = 0.1  # Only adjust by 10% of the suggested change
        new_value = current_value + (adjustment * damping * confidence)

        # Clamp to valid range
        new_value = max(0.0, min(1.0, new_value))

        # Only update if change is significant (> 1%)
        if abs(new_value - current_value) > 0.01:
            self.personality.communication_preferences[domain_key] = new_value

            # Log evolution
            evolution = PersonalityEvolution(
                timestamp=datetime.now().isoformat(),
                domain=domain,
                old_value=current_value,
                new_value=new_value,
                trigger=trigger,
                confidence=confidence,
                interaction_count=self.personality.total_interactions,
            )

            self._log_evolution(evolution)
            self.personality.evolution_history.append(asdict(evolution))

            # Limit history size
            if len(self.personality.evolution_history) > 1000:
                self.personality.evolution_history = self.personality.evolution_history[
                    -500:
                ]

            self.save_personality()

            logger.info(
                f"Evolved {domain_key}: {current_value:.3f} -> {new_value:.3f} "
                f"(trigger: {trigger}, confidence: {confidence:.2f})"
            )

    def _log_evolution(self, evolution: PersonalityEvolution):
        """Log personality evolution to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO personality_evolution 
                (timestamp, domain, old_value, new_value, trigger, confidence, interaction_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    evolution.timestamp,
                    evolution.domain.value,
                    evolution.old_value,
                    evolution.new_value,
                    evolution.trigger,
                    evolution.confidence,
                    evolution.interaction_count,
                ),
            )
            conn.commit()

    def log_interaction(
        self,
        interaction_type: str,
        context: Optional[str] = None,
        user_sentiment: Optional[float] = None,
        response_quality: Optional[float] = None,
        adaptation_applied: Optional[str] = None,
    ):
        """Log an interaction for learning"""
        if not self.personality:
            return

        self.personality.total_interactions += 1
        self.personality.interaction_patterns[interaction_type] += 1

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO interaction_log 
                (timestamp, interaction_type, context, user_sentiment, response_quality, adaptation_applied)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    interaction_type,
                    context,
                    user_sentiment,
                    response_quality,
                    adaptation_applied,
                ),
            )
            conn.commit()

        # Auto-save every 10 interactions
        if self.personality.total_interactions % 10 == 0:
            self.save_personality()

    def get_current_system_prompt(
        self, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate current system prompt based on personality state

        Args:
            context: Optional context to customize prompt

        Returns:
            Complete system prompt for LLM
        """
        if not self.personality:
            return VEGAPersonality().base_system_prompt

        # Build dynamic system prompt
        prompt_parts = [self.personality.base_system_prompt]

        # Add communication style guidance
        prefs = self.personality.communication_preferences
        prompt_parts.append("\nCurrent communication preferences:")

        if prefs["verbosity"] > 0.7:
            prompt_parts.append("- Provide detailed, comprehensive responses")
        elif prefs["verbosity"] < 0.4:
            prompt_parts.append("- Keep responses concise and to the point")

        if prefs["formality"] > 0.7:
            prompt_parts.append("- Maintain professional, formal tone")
        elif prefs["formality"] < 0.4:
            prompt_parts.append("- Use casual, friendly tone")

        if prefs["technical_depth"] > 0.7:
            prompt_parts.append("- Include technical details and explanations")

        if prefs["enthusiasm"] > 0.7:
            prompt_parts.append("- Show enthusiasm and energy in responses")

        # Add context-specific guidance
        if context:
            if context.get("user_expertise") == "expert":
                prompt_parts.append("- User has expert knowledge, adapt accordingly")
            if context.get("time_pressure"):
                prompt_parts.append("- User is time-constrained, prioritize efficiency")
            if context.get("mood") == "stressed":
                prompt_parts.append("- User seems stressed, be supportive and calming")

        # Add interaction statistics
        if self.personality.total_interactions > 100:
            prompt_parts.append(
                f"\nYou have had {self.personality.total_interactions:,} interactions, "
                "allowing you to understand user preferences well."
            )

        return "\n".join(prompt_parts)

    def log_voice_training_session(
        self,
        session_type: str,
        audio_file_path: str,
        duration_seconds: float,
        quality_score: Optional[float] = None,
        features: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ):
        """Log a voice training session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO voice_training_sessions 
                (timestamp, session_type, audio_file_path, duration_seconds, quality_score, features, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    datetime.now().isoformat(),
                    session_type,
                    audio_file_path,
                    duration_seconds,
                    quality_score,
                    json.dumps(features) if features else None,
                    notes,
                ),
            )
            conn.commit()

        logger.info(
            f"Logged voice training session: {session_type} ({duration_seconds:.1f}s)"
        )

    def get_personality_stats(self) -> Dict[str, Any]:
        """Get statistics about personality evolution"""
        if not self.personality:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            # Get evolution count per domain
            cursor = conn.execute(
                """
                SELECT domain, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM personality_evolution
                GROUP BY domain
            """
            )
            evolution_stats = {
                row[0]: {"count": row[1], "avg_confidence": row[2]} for row in cursor
            }

            # Get recent evolutions
            cursor = conn.execute(
                """
                SELECT timestamp, domain, old_value, new_value, trigger
                FROM personality_evolution
                ORDER BY timestamp DESC
                LIMIT 10
            """
            )
            recent_evolutions = [
                {
                    "timestamp": row[0],
                    "domain": row[1],
                    "change": row[3] - row[2],
                    "trigger": row[4],
                }
                for row in cursor
            ]

            # Get voice training stats
            cursor = conn.execute(
                """
                SELECT COUNT(*), SUM(duration_seconds), AVG(quality_score)
                FROM voice_training_sessions
            """
            )
            voice_stats = cursor.fetchone()

        return {
            "total_interactions": self.personality.total_interactions,
            "personality_version": self.personality.personality_version,
            "last_updated": self.personality.last_updated,
            "communication_preferences": self.personality.communication_preferences,
            "evolution_stats": evolution_stats,
            "recent_evolutions": recent_evolutions,
            "voice_training": {
                "sessions": voice_stats[0] if voice_stats else 0,
                "total_duration_seconds": voice_stats[1] if voice_stats else 0,
                "avg_quality_score": voice_stats[2] if voice_stats else None,
            },
            "top_interaction_types": dict(
                sorted(
                    self.personality.interaction_patterns.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }


# Singleton instance
_vega_personality_core: Optional[VEGAPersonalityCore] = None


def get_vega_personality() -> VEGAPersonalityCore:
    """Get or create the VEGA personality core singleton"""
    global _vega_personality_core
    if _vega_personality_core is None:
        _vega_personality_core = VEGAPersonalityCore()
    return _vega_personality_core


# Convenience functions
def get_system_prompt(context: Optional[Dict[str, Any]] = None) -> str:
    """Get current VEGA system prompt"""
    return get_vega_personality().get_current_system_prompt(context)


def log_interaction(interaction_type: str, **kwargs):
    """Log an interaction"""
    get_vega_personality().log_interaction(interaction_type, **kwargs)


def evolve_personality(
    domain: PersonalityDomain, adjustment: float, trigger: str, confidence: float = 0.8
):
    """Evolve VEGA's personality"""
    get_vega_personality().evolve_trait(domain, adjustment, trigger, confidence)
