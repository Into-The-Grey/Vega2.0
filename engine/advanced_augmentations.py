#!/usr/bin/env python3
"""
VEGA ADVANCED AUGMENTATIONS SYSTEM
==================================

This system provides sophisticated protocols and augmentations that make Vega
truly intelligent about when and how to interact. It includes smart silence
protocols, curiosity buffering, notification suppression, and calendar integration.

Key Features:
- Smart silence protocols that learn when NOT to interrupt
- Curiosity buffer system for collecting interesting observations
- Reactive notification suppression during important work
- Calendar integration for context-aware timing
- Adaptive interruption thresholds based on user state
- Energy-aware interaction scheduling
- Context-sensitive communication protocols

Like JARVIS, this system makes Vega truly respectful of the user's time and
attention while maximizing the value of interactions.
"""

import os
import re
import json
import sqlite3
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


class SilenceProtocol(Enum):
    STANDARD = "standard"  # Normal interaction rules
    FOCUS_AWARE = "focus_aware"  # Heightened focus detection
    DO_NOT_DISTURB = "do_not_disturb"  # Minimal interruptions only
    EMERGENCY_ONLY = "emergency_only"  # Only critical issues
    ADAPTIVE = "adaptive"  # Learns user preferences


class NotificationPriority(Enum):
    LOW = "low"  # Can wait indefinitely
    MEDIUM = "medium"  # Can wait for good moment
    HIGH = "high"  # Should interrupt soon
    CRITICAL = "critical"  # Interrupt immediately
    EMERGENCY = "emergency"  # Override all silence protocols


class ContextWindow(Enum):
    IMMEDIATE = "immediate"  # Current moment
    SHORT_TERM = "short_term"  # Next 15-30 minutes
    MEDIUM_TERM = "medium_term"  # Next 1-2 hours
    LONG_TERM = "long_term"  # Rest of day/week


@dataclass
class CuriosityItem:
    """Item in the curiosity buffer"""

    id: str
    content: str
    priority: NotificationPriority
    context: Dict[str, Any]
    created_at: datetime
    decay_rate: float  # How quickly this becomes less relevant
    relevance_score: float
    tags: Set[str]
    optimal_timing: Optional[datetime]  # Best time to share this


@dataclass
class SilenceRule:
    """Rule for when to maintain silence"""

    rule_id: str
    name: str
    conditions: Dict[str, Any]  # Conditions that trigger this rule
    silence_duration: int  # Minutes to stay quiet
    override_priority: NotificationPriority  # What can override this
    confidence: float  # How sure we are about this rule
    usage_count: int  # How often this rule has been applied
    success_rate: float  # How well this rule has worked


@dataclass
class InterruptionContext:
    """Context for evaluating interruption appropriateness"""

    user_presence: str
    current_app: str
    activity_level: float
    focus_duration: float
    calendar_events: List[Dict[str, Any]]
    silence_protocol: SilenceProtocol
    time_context: Dict[str, Any]
    recent_interactions: List[Dict[str, Any]]


class CalendarIntegration:
    """Integrates with user's calendar for context-aware timing"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.calendar_cache = state_dir / "calendar_cache.json"
        self.cache_expiry = None
        self.cached_events = []

    async def get_current_events(self) -> List[Dict[str, Any]]:
        """Get current and upcoming calendar events"""
        try:
            # Try to refresh cache if needed
            await self._refresh_calendar_cache()

            # Filter for current and near-future events
            now = datetime.now()
            current_events = []

            for event in self.cached_events:
                event_start = datetime.fromisoformat(
                    event.get("start", now.isoformat())
                )
                event_end = datetime.fromisoformat(event.get("end", now.isoformat()))

                # Include events that are happening now or starting within 30 minutes
                if (event_start <= now <= event_end) or (
                    event_start <= now + timedelta(minutes=30)
                ):
                    current_events.append(event)

            return current_events

        except Exception as e:
            logger.debug(f"Error getting calendar events: {e}")
            return []

    async def _refresh_calendar_cache(self):
        """Refresh calendar cache from various sources"""
        if self.cache_expiry and datetime.now() < self.cache_expiry:
            return  # Cache still valid

        events = []

        # Try different calendar sources
        sources = [
            self._get_events_from_calcurse,
            self._get_events_from_khal,
            self._get_events_from_evolution,
            self._get_events_from_gcal,
        ]

        for source in sources:
            try:
                source_events = await source()
                if source_events:
                    events.extend(source_events)
                    break  # Use first successful source
            except Exception as e:
                logger.debug(f"Calendar source failed: {e}")
                continue

        self.cached_events = events
        self.cache_expiry = datetime.now() + timedelta(
            minutes=15
        )  # Cache for 15 minutes

        # Save cache to file
        try:
            with open(self.calendar_cache, "w") as f:
                json.dump(
                    {"events": events, "cached_at": datetime.now().isoformat()}, f
                )
        except Exception as e:
            logger.debug(f"Failed to save calendar cache: {e}")

    async def _get_events_from_calcurse(self) -> List[Dict[str, Any]]:
        """Get events from calcurse"""
        try:
            result = subprocess.run(
                ["calcurse", "-n", "--format-apt", "%S %s"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                events = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        # Parse calcurse output
                        events.append(
                            {
                                "title": line.strip(),
                                "start": datetime.now().isoformat(),  # Simplified
                                "end": (
                                    datetime.now() + timedelta(hours=1)
                                ).isoformat(),
                                "source": "calcurse",
                            }
                        )
                return events
        except:
            pass
        return []

    async def _get_events_from_khal(self) -> List[Dict[str, Any]]:
        """Get events from khal"""
        try:
            result = subprocess.run(
                ["khal", "list", "today"], capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                events = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip() and not line.startswith("No events"):
                        events.append(
                            {
                                "title": line.strip(),
                                "start": datetime.now().isoformat(),
                                "end": (
                                    datetime.now() + timedelta(hours=1)
                                ).isoformat(),
                                "source": "khal",
                            }
                        )
                return events
        except:
            pass
        return []

    async def _get_events_from_evolution(self) -> List[Dict[str, Any]]:
        """Get events from Evolution calendar"""
        # This would require more complex integration
        return []

    async def _get_events_from_gcal(self) -> List[Dict[str, Any]]:
        """Get events from Google Calendar (if credentials available)"""
        # This would require Google Calendar API setup
        return []

    def is_meeting_time(self, events: List[Dict[str, Any]]) -> bool:
        """Check if user is currently in a meeting"""
        now = datetime.now()

        for event in events:
            try:
                start_time = datetime.fromisoformat(event.get("start", ""))
                end_time = datetime.fromisoformat(event.get("end", ""))

                if start_time <= now <= end_time:
                    # Check if this looks like a meeting
                    title = event.get("title", "").lower()
                    meeting_keywords = [
                        "meeting",
                        "call",
                        "conference",
                        "interview",
                        "standup",
                        "sync",
                    ]

                    if any(keyword in title for keyword in meeting_keywords):
                        return True
            except:
                continue

        return False


class SmartSilenceManager:
    """Manages smart silence protocols and interruption decisions"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.silence_rules = self._load_silence_rules()
        self.current_protocol = SilenceProtocol.STANDARD
        self.override_until = None
        self.calendar = CalendarIntegration(state_dir)

        # Adaptive learning
        self.interruption_history = []
        self.learning_rate = 0.1

    def _load_silence_rules(self) -> List[SilenceRule]:
        """Load predefined silence rules"""
        return [
            SilenceRule(
                rule_id="deep_focus",
                name="Deep Focus Session",
                conditions={
                    "presence_state": "focused",
                    "focus_duration_min": 30,
                    "app_categories": ["development", "writing", "design"],
                },
                silence_duration=60,  # 1 hour
                override_priority=NotificationPriority.CRITICAL,
                confidence=0.9,
                usage_count=0,
                success_rate=0.8,
            ),
            SilenceRule(
                rule_id="gaming_session",
                name="Gaming Session",
                conditions={"presence_state": "gaming", "activity_level_min": 10},
                silence_duration=120,  # 2 hours
                override_priority=NotificationPriority.EMERGENCY,
                confidence=0.95,
                usage_count=0,
                success_rate=0.9,
            ),
            SilenceRule(
                rule_id="meeting_time",
                name="Meeting in Progress",
                conditions={"presence_state": "meeting", "calendar_meeting": True},
                silence_duration=90,  # 1.5 hours
                override_priority=NotificationPriority.EMERGENCY,
                confidence=1.0,
                usage_count=0,
                success_rate=0.95,
            ),
            SilenceRule(
                rule_id="late_night_focus",
                name="Late Night Work",
                conditions={
                    "time_hour_min": 22,
                    "presence_state": "focused",
                    "activity_level_min": 5,
                },
                silence_duration=30,
                override_priority=NotificationPriority.HIGH,
                confidence=0.7,
                usage_count=0,
                success_rate=0.6,
            ),
            SilenceRule(
                rule_id="high_cpu_usage",
                name="System Under Load",
                conditions={"cpu_usage_min": 80, "system_state": "heavy_work"},
                silence_duration=15,
                override_priority=NotificationPriority.HIGH,
                confidence=0.8,
                usage_count=0,
                success_rate=0.7,
            ),
        ]

    async def evaluate_interruption_appropriateness(
        self, context: InterruptionContext, priority: NotificationPriority
    ) -> Tuple[bool, str, float]:
        """Evaluate whether it's appropriate to interrupt right now"""

        # Check for manual overrides
        if self.override_until and datetime.now() < self.override_until:
            if priority.value in ["critical", "emergency"]:
                return True, "Override period but critical priority", 0.9
            else:
                return False, "Manual override in effect", 0.9

        # Emergency always goes through
        if priority == NotificationPriority.EMERGENCY:
            return True, "Emergency priority overrides all", 1.0

        # Check calendar context
        calendar_events = await self.calendar.get_current_events()
        if self.calendar.is_meeting_time(calendar_events):
            if priority not in [
                NotificationPriority.CRITICAL,
                NotificationPriority.EMERGENCY,
            ]:
                return False, "User is in a meeting", 0.95

        # Evaluate against silence rules
        best_match = None
        best_confidence = 0.0

        for rule in self.silence_rules:
            match_confidence = self._evaluate_rule_match(rule, context, calendar_events)

            if match_confidence > best_confidence:
                best_match = rule
                best_confidence = match_confidence

        if best_match and best_confidence > 0.6:
            # Check if priority can override this rule
            priority_levels = {
                NotificationPriority.LOW: 0,
                NotificationPriority.MEDIUM: 1,
                NotificationPriority.HIGH: 2,
                NotificationPriority.CRITICAL: 3,
                NotificationPriority.EMERGENCY: 4,
            }

            can_override = priority_levels.get(priority, 0) >= priority_levels.get(
                best_match.override_priority, 4
            )

            if can_override:
                return (
                    True,
                    f"Priority {priority.value} overrides {best_match.name}",
                    best_confidence,
                )
            else:
                return False, f"Blocked by {best_match.name} protocol", best_confidence

        # Default evaluation based on user state
        return self._default_interruption_evaluation(context, priority)

    def _evaluate_rule_match(
        self,
        rule: SilenceRule,
        context: InterruptionContext,
        calendar_events: List[Dict[str, Any]],
    ) -> float:
        """Evaluate how well a rule matches the current context"""

        conditions = rule.conditions
        match_score = 0.0
        total_conditions = len(conditions)

        if total_conditions == 0:
            return 0.0

        # Check presence state
        if "presence_state" in conditions:
            if context.user_presence == conditions["presence_state"]:
                match_score += 1.0
            else:
                return 0.0  # Must match presence state

        # Check focus duration
        if "focus_duration_min" in conditions:
            required_duration = conditions["focus_duration_min"]
            if context.focus_duration >= required_duration:
                match_score += 1.0

        # Check activity level
        if "activity_level_min" in conditions:
            required_activity = conditions["activity_level_min"]
            if context.activity_level >= required_activity:
                match_score += 1.0

        # Check time conditions
        if "time_hour_min" in conditions:
            current_hour = datetime.now().hour
            if current_hour >= conditions["time_hour_min"]:
                match_score += 1.0

        # Check calendar meeting
        if "calendar_meeting" in conditions:
            if (
                self.calendar.is_meeting_time(calendar_events)
                == conditions["calendar_meeting"]
            ):
                match_score += 1.0

        # Check system state
        if "cpu_usage_min" in conditions:
            # This would need to be passed in context
            # For now, assume it's available
            match_score += 0.5  # Partial match

        confidence = (match_score / total_conditions) * rule.confidence
        return confidence

    def _default_interruption_evaluation(
        self, context: InterruptionContext, priority: NotificationPriority
    ) -> Tuple[bool, str, float]:
        """Default evaluation when no specific rules match"""

        # Simple heuristics
        if context.user_presence == "away":
            return False, "User is away", 0.8

        if context.user_presence == "focused" and context.focus_duration > 15:
            if priority in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
                return True, "High priority overrides focus", 0.7
            else:
                return False, "User is focused", 0.7

        if context.activity_level > 20:  # Very high activity
            if priority == NotificationPriority.LOW:
                return False, "User is very busy", 0.6

        # Default to allowing the interruption
        return True, "No blocking conditions detected", 0.5

    def set_manual_override(self, duration_minutes: int, protocol: SilenceProtocol):
        """Set manual override for silence protocol"""
        self.override_until = datetime.now() + timedelta(minutes=duration_minutes)
        self.current_protocol = protocol
        logger.info(
            f"Silence protocol manually set to {protocol.value} for {duration_minutes} minutes"
        )

    def learn_from_interruption(
        self,
        was_appropriate: bool,
        context: InterruptionContext,
        user_response_quality: str,
    ):
        """Learn from user response to interruption"""
        # This would update rule success rates and adapt thresholds
        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "was_appropriate": was_appropriate,
            "context": asdict(context),
            "response_quality": user_response_quality,
        }

        self.interruption_history.append(learning_data)

        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(days=30)
        self.interruption_history = [
            item
            for item in self.interruption_history
            if datetime.fromisoformat(item["timestamp"]) > cutoff_time
        ]


class CuriosityBuffer:
    """Buffers interesting observations until appropriate moment to share"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.buffer_file = state_dir / "curiosity_buffer.jsonl"
        self.max_buffer_size = 20
        self.max_item_age = timedelta(hours=24)

    def add_item(
        self,
        content: str,
        priority: NotificationPriority,
        context: Dict[str, Any],
        tags: Set[str] = None,
    ) -> str:
        """Add item to curiosity buffer"""

        item_id = hashlib.md5(
            f"{content}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        item = CuriosityItem(
            id=item_id,
            content=content,
            priority=priority,
            context=context,
            created_at=datetime.now(),
            decay_rate=self._calculate_decay_rate(priority),
            relevance_score=1.0,  # Starts at full relevance
            tags=tags or set(),
            optimal_timing=self._estimate_optimal_timing(context, priority),
        )

        # Save to buffer
        self._save_item(item)

        # Clean old items
        self._cleanup_buffer()

        return item_id

    def _calculate_decay_rate(self, priority: NotificationPriority) -> float:
        """Calculate how quickly an item loses relevance"""
        decay_rates = {
            NotificationPriority.LOW: 0.1,  # Decays slowly
            NotificationPriority.MEDIUM: 0.05,  # Moderate decay
            NotificationPriority.HIGH: 0.02,  # Slow decay
            NotificationPriority.CRITICAL: 0.0,  # No decay
        }
        return decay_rates.get(priority, 0.05)

    def _estimate_optimal_timing(
        self, context: Dict[str, Any], priority: NotificationPriority
    ) -> Optional[datetime]:
        """Estimate optimal time to share this item"""

        # Simple heuristics for now
        now = datetime.now()

        if priority == NotificationPriority.LOW:
            # Wait for idle time
            return now + timedelta(hours=2)
        elif priority == NotificationPriority.MEDIUM:
            # Wait for break in activity
            return now + timedelta(minutes=30)
        else:
            # Share soon
            return now + timedelta(minutes=5)

    def _save_item(self, item: CuriosityItem):
        """Save item to buffer file"""
        try:
            item_data = asdict(item)
            item_data["created_at"] = item.created_at.isoformat()
            item_data["optimal_timing"] = (
                item.optimal_timing.isoformat() if item.optimal_timing else None
            )
            item_data["tags"] = list(item.tags)

            with open(self.buffer_file, "a") as f:
                f.write(json.dumps(item_data) + "\n")

        except Exception as e:
            logger.error(f"Failed to save curiosity item: {e}")

    def get_ready_items(self, current_context: Dict[str, Any]) -> List[CuriosityItem]:
        """Get items that are ready to be shared"""
        items = self._load_all_items()
        ready_items = []

        now = datetime.now()

        for item in items:
            # Check if item is ready based on optimal timing
            if item.optimal_timing and now >= item.optimal_timing:
                # Update relevance based on decay
                age_hours = (now - item.created_at).total_seconds() / 3600
                current_relevance = max(
                    0.0, item.relevance_score - (age_hours * item.decay_rate)
                )

                if current_relevance > 0.3:  # Still relevant enough
                    item.relevance_score = current_relevance
                    ready_items.append(item)

        # Sort by priority and relevance
        ready_items.sort(
            key=lambda x: (x.priority.value, x.relevance_score), reverse=True
        )

        return ready_items[:3]  # Return top 3

    def _load_all_items(self) -> List[CuriosityItem]:
        """Load all items from buffer"""
        items = []

        if not self.buffer_file.exists():
            return items

        try:
            with open(self.buffer_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        item = CuriosityItem(
                            id=data["id"],
                            content=data["content"],
                            priority=NotificationPriority(data["priority"]),
                            context=data["context"],
                            created_at=datetime.fromisoformat(data["created_at"]),
                            decay_rate=data["decay_rate"],
                            relevance_score=data["relevance_score"],
                            tags=set(data.get("tags", [])),
                            optimal_timing=(
                                datetime.fromisoformat(data["optimal_timing"])
                                if data["optimal_timing"]
                                else None
                            ),
                        )

                        items.append(item)

                    except Exception as e:
                        logger.debug(f"Error loading curiosity item: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error loading curiosity buffer: {e}")

        return items

    def _cleanup_buffer(self):
        """Remove old and irrelevant items from buffer"""
        items = self._load_all_items()

        # Filter out old and irrelevant items
        now = datetime.now()
        cutoff_time = now - self.max_item_age

        valid_items = []
        for item in items:
            # Keep if not too old and still relevant
            if item.created_at > cutoff_time and item.relevance_score > 0.1:
                valid_items.append(item)

        # Keep only the most recent items if buffer is too large
        if len(valid_items) > self.max_buffer_size:
            valid_items.sort(key=lambda x: x.created_at, reverse=True)
            valid_items = valid_items[: self.max_buffer_size]

        # Rewrite buffer file
        try:
            self.buffer_file.unlink(missing_ok=True)
            for item in valid_items:
                self._save_item(item)
        except Exception as e:
            logger.error(f"Error cleaning up curiosity buffer: {e}")

    def remove_item(self, item_id: str):
        """Remove specific item from buffer"""
        items = self._load_all_items()
        remaining_items = [item for item in items if item.id != item_id]

        # Rewrite buffer
        try:
            self.buffer_file.unlink(missing_ok=True)
            for item in remaining_items:
                self._save_item(item)
        except Exception as e:
            logger.error(f"Error removing item from buffer: {e}")


class AdvancedAugmentationManager:
    """Main manager for all advanced augmentation systems"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.silence_manager = SmartSilenceManager(state_dir)
        self.curiosity_buffer = CuriosityBuffer(state_dir)
        self.calendar = CalendarIntegration(state_dir)

        # Energy management
        self.energy_threshold = 0.7  # Minimum energy level for interactions
        self.current_energy = 1.0

        # Notification suppression
        self.notification_suppressed_until = None
        self.suppression_reason = None

    async def evaluate_interaction_request(
        self, content: str, priority: NotificationPriority, context: Dict[str, Any]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Main entry point for evaluating whether to proceed with interaction"""

        # Build interruption context
        interruption_context = InterruptionContext(
            user_presence=context.get("user_presence", "unknown"),
            current_app=context.get("current_app", "unknown"),
            activity_level=context.get("activity_level", 0),
            focus_duration=context.get("focus_duration", 0),
            calendar_events=await self.calendar.get_current_events(),
            silence_protocol=self.silence_manager.current_protocol,
            time_context=context.get("time_context", {}),
            recent_interactions=context.get("recent_interactions", []),
        )

        # Check energy levels
        if (
            self.current_energy < self.energy_threshold
            and priority != NotificationPriority.EMERGENCY
        ):
            return (
                False,
                "Low energy state - conserving interactions",
                {"action": "buffer"},
            )

        # Check notification suppression
        if (
            self.notification_suppressed_until
            and datetime.now() < self.notification_suppressed_until
            and priority
            not in [NotificationPriority.CRITICAL, NotificationPriority.EMERGENCY]
        ):
            return (
                False,
                f"Notifications suppressed: {self.suppression_reason}",
                {"action": "buffer"},
            )

        # Evaluate with silence manager
        can_interrupt, reason, confidence = (
            await self.silence_manager.evaluate_interruption_appropriateness(
                interruption_context, priority
            )
        )

        if not can_interrupt:
            # Add to curiosity buffer for later
            if priority != NotificationPriority.LOW:
                self.curiosity_buffer.add_item(
                    content=content,
                    priority=priority,
                    context=context,
                    tags=set(context.get("tags", [])),
                )

            return False, reason, {"action": "buffered", "confidence": confidence}

        # Interaction approved
        self._consume_energy(priority)

        return True, reason, {"action": "approved", "confidence": confidence}

    def _consume_energy(self, priority: NotificationPriority):
        """Consume energy based on interaction priority"""
        energy_costs = {
            NotificationPriority.LOW: 0.02,
            NotificationPriority.MEDIUM: 0.05,
            NotificationPriority.HIGH: 0.1,
            NotificationPriority.CRITICAL: 0.15,
            NotificationPriority.EMERGENCY: 0.0,  # No energy cost for emergencies
        }

        cost = energy_costs.get(priority, 0.05)
        self.current_energy = max(0.0, self.current_energy - cost)

    def restore_energy(self, amount: float = 0.1):
        """Restore energy during quiet periods"""
        self.current_energy = min(1.0, self.current_energy + amount)

    async def get_buffered_items_ready_to_share(
        self, context: Dict[str, Any]
    ) -> List[CuriosityItem]:
        """Get items from curiosity buffer that are ready to share"""
        return self.curiosity_buffer.get_ready_items(context)

    def suppress_notifications(self, duration_minutes: int, reason: str):
        """Temporarily suppress notifications"""
        self.notification_suppressed_until = datetime.now() + timedelta(
            minutes=duration_minutes
        )
        self.suppression_reason = reason
        logger.info(
            f"Notifications suppressed for {duration_minutes} minutes: {reason}"
        )

    def set_silence_protocol(
        self, protocol: SilenceProtocol, duration_minutes: int = 60
    ):
        """Set silence protocol manually"""
        self.silence_manager.set_manual_override(duration_minutes, protocol)

    async def process_user_feedback(
        self,
        interaction_content: str,
        user_response: str,
        response_quality: str,
        context: Dict[str, Any],
    ):
        """Process user feedback to improve future decisions"""

        # Determine if interruption was appropriate
        was_appropriate = response_quality in ["excellent", "good", "neutral"]

        # Build context for learning
        interruption_context = InterruptionContext(
            user_presence=context.get("user_presence", "unknown"),
            current_app=context.get("current_app", "unknown"),
            activity_level=context.get("activity_level", 0),
            focus_duration=context.get("focus_duration", 0),
            calendar_events=await self.calendar.get_current_events(),
            silence_protocol=self.silence_manager.current_protocol,
            time_context=context.get("time_context", {}),
            recent_interactions=context.get("recent_interactions", []),
        )

        # Learn from the interaction
        self.silence_manager.learn_from_interruption(
            was_appropriate, interruption_context, response_quality
        )

    def get_system_status(self) -> Dict[str, Any]:
        """Get current status of augmentation systems"""
        return {
            "silence_protocol": self.silence_manager.current_protocol.value,
            "energy_level": self.current_energy,
            "notifications_suppressed": self.notification_suppressed_until is not None,
            "suppression_reason": self.suppression_reason,
            "buffer_size": len(self.curiosity_buffer._load_all_items()),
        }


# Integration functions for the main ambient loop
async def check_interaction_approval(
    state_dir: Path, content: str, priority: str, context: Dict[str, Any]
) -> Tuple[bool, str, Dict[str, Any]]:
    """Check if interaction is approved by augmentation systems"""
    manager = AdvancedAugmentationManager(state_dir)
    priority_enum = NotificationPriority(priority)
    return await manager.evaluate_interaction_request(content, priority_enum, context)


async def get_ready_curiosity_items(
    state_dir: Path, context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Get curiosity items ready to share"""
    manager = AdvancedAugmentationManager(state_dir)
    items = await manager.get_buffered_items_ready_to_share(context)

    # Convert to serializable format
    return [
        {
            "id": item.id,
            "content": item.content,
            "priority": item.priority.value,
            "relevance_score": item.relevance_score,
            "tags": list(item.tags),
        }
        for item in items
    ]


def set_silence_mode(state_dir: Path, mode: str, duration_minutes: int = 60):
    """Set silence protocol mode"""
    manager = AdvancedAugmentationManager(state_dir)
    protocol = SilenceProtocol(mode)
    manager.set_silence_protocol(protocol, duration_minutes)


def get_augmentation_status(state_dir: Path) -> Dict[str, Any]:
    """Get status of all augmentation systems"""
    manager = AdvancedAugmentationManager(state_dir)
    return manager.get_system_status()
