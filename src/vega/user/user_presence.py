#!/usr/bin/env python3
"""
VEGA USER PRESENCE TRACKING SYSTEM
==================================

This system passively monitors user presence and activity patterns to inform
optimal interaction timing. It respects privacy while gathering essential
context about user availability and focus states.

Key Features:
- Keyboard/mouse activity detection (no keylogging)
- Application focus and window title monitoring
- Presence classification (active/idle/away/focused)
- Activity pattern analysis and learning
- Non-invasive monitoring with privacy safeguards
- Integration with system idle detection

Like JARVIS, this system knows when the user is present, busy, or available
for interaction without being intrusive.
"""

import os
import re
import time
import json
import sqlite3
import asyncio
import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

try:
    import psutil
    import pynput
    from pynput import mouse, keyboard
except ImportError:
    psutil = None
    pynput = None

logger = logging.getLogger(__name__)


class PresenceState(Enum):
    ACTIVE = "active"  # User actively interacting
    IDLE = "idle"  # User present but not actively working
    AWAY = "away"  # User away from computer
    FOCUSED = "focused"  # User in deep work mode
    GAMING = "gaming"  # User gaming (avoid interruptions)
    MEETING = "meeting"  # User in meeting/call
    UNKNOWN = "unknown"  # Cannot determine state


class ActivityType(Enum):
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    APPLICATION_SWITCH = "app_switch"
    WINDOW_FOCUS = "window_focus"
    SYSTEM_IDLE = "system_idle"


@dataclass
class ActivityEvent:
    """Represents a user activity event"""

    timestamp: datetime
    activity_type: ActivityType
    details: Dict[str, Any]
    privacy_safe: bool = True  # Whether this event is safe to log


@dataclass
class PresenceSnapshot:
    """Snapshot of user presence at a point in time"""

    timestamp: datetime
    presence_state: PresenceState
    idle_time_seconds: float
    active_application: str
    window_title_hash: str  # Hashed for privacy
    keyboard_activity_rate: float  # Events per minute
    mouse_activity_rate: float  # Events per minute
    confidence: float  # Confidence in presence classification
    focus_indicators: List[str]  # Indicators of focused work


@dataclass
class ActivityPattern:
    """Learned pattern about user activity"""

    pattern_id: str
    time_range: Tuple[int, int]  # Hour range (start, end)
    typical_presence: PresenceState
    activity_characteristics: Dict[str, float]
    confidence: float
    sample_count: int
    last_updated: datetime


class PrivacyFilter:
    """Filters sensitive information from activity monitoring"""

    def __init__(self):
        self.sensitive_patterns = [
            r"password",
            r"login",
            r"credential",
            r"secret",
            r"token",
            r"private",
            r"confidential",
            r"bank",
            r"payment",
            r"card",
        ]

    def filter_window_title(self, title: str) -> str:
        """Filter sensitive information from window titles"""
        if not title:
            return ""

        # Check for sensitive patterns
        title_lower = title.lower()
        for pattern in self.sensitive_patterns:
            if re.search(pattern, title_lower):
                return "FILTERED_SENSITIVE"

        # Hash the title for privacy while maintaining consistency
        return hashlib.md5(title.encode()).hexdigest()[:8]

    def is_safe_to_log(self, activity_details: Dict[str, Any]) -> bool:
        """Determine if activity details are safe to log"""
        # Never log actual key content or mouse coordinates
        if "key_content" in activity_details or "mouse_coords" in activity_details:
            return False

        # Check for sensitive app names
        app_name = activity_details.get("app_name", "").lower()
        for pattern in self.sensitive_patterns:
            if pattern in app_name:
                return False

        return True


class SystemActivityMonitor:
    """Monitors system-level activity indicators"""

    def __init__(self):
        self.last_idle_time = 0
        self.privacy_filter = PrivacyFilter()

    def get_system_idle_time(self) -> float:
        """Get system idle time in seconds"""
        try:
            if os.name == "posix":  # Linux/macOS
                # Try multiple methods for idle time detection
                methods = [
                    self._get_idle_time_xprintidle,
                    self._get_idle_time_xscreensaver,
                    self._get_idle_time_loginctl,
                ]

                for method in methods:
                    try:
                        idle_time = method()
                        if idle_time is not None:
                            return idle_time
                    except:
                        continue

                # Fallback: use last activity time tracking
                return time.time() - self.last_idle_time

            else:  # Windows
                return self._get_idle_time_windows()

        except Exception as e:
            logger.debug(f"Could not get system idle time: {e}")
            return 0.0

    def _get_idle_time_xprintidle(self) -> Optional[float]:
        """Get idle time using xprintidle (Linux)"""
        try:
            result = subprocess.run(
                ["xprintidle"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return float(result.stdout.strip()) / 1000.0  # Convert ms to seconds
        except:
            pass
        return None

    def _get_idle_time_xscreensaver(self) -> Optional[float]:
        """Get idle time using xscreensaver (Linux)"""
        try:
            result = subprocess.run(
                ["xscreensaver-command", "-time"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                # Parse output like "screen blanked since 00:02:30"
                output = result.stdout.strip()
                if "screen blanked since" in output:
                    time_str = output.split("since")[-1].strip()
                    # Convert time string to seconds
                    time_parts = time_str.split(":")
                    if len(time_parts) == 3:
                        hours, minutes, seconds = map(int, time_parts)
                        return hours * 3600 + minutes * 60 + seconds
        except:
            pass
        return None

    def _get_idle_time_loginctl(self) -> Optional[float]:
        """Get idle time using loginctl (systemd)"""
        try:
            result = subprocess.run(
                ["loginctl", "show-session", "--property=IdleHint,IdleSinceHint"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                idle_since = None
                for line in lines:
                    if line.startswith("IdleSinceHint="):
                        idle_since = line.split("=", 1)[1]
                        break

                if idle_since and idle_since != "0":
                    # Parse timestamp and calculate difference
                    idle_timestamp = int(idle_since) / 1000000  # Convert microseconds
                    return time.time() - idle_timestamp
        except:
            pass
        return None

    def _get_idle_time_windows(self) -> float:
        """Get idle time on Windows"""
        try:
            import ctypes
            from ctypes import wintypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.UINT),
                    ("dwTime", wintypes.DWORD),
                ]

            lastInputInfo = LASTINPUTINFO()
            lastInputInfo.cbSize = ctypes.sizeof(lastInputInfo)

            if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lastInputInfo)):
                current_time = ctypes.windll.kernel32.GetTickCount()
                idle_time = (current_time - lastInputInfo.dwTime) / 1000.0
                return idle_time
        except:
            pass
        return 0.0

    def get_active_window_info(self) -> Dict[str, str]:
        """Get information about the currently active window"""
        try:
            if os.name == "posix":
                return self._get_active_window_linux()
            else:
                return self._get_active_window_windows()
        except Exception as e:
            logger.debug(f"Could not get active window info: {e}")
            return {"app_name": "unknown", "window_title": "unknown"}

    def _get_active_window_linux(self) -> Dict[str, str]:
        """Get active window info on Linux"""
        try:
            # Try xdotool first
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                window_title = result.stdout.strip()

                # Get process name
                pid_result = subprocess.run(
                    ["xdotool", "getactivewindow", "getwindowpid"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                app_name = "unknown"
                if pid_result.returncode == 0:
                    try:
                        pid = int(pid_result.stdout.strip())
                        if psutil:
                            process = psutil.Process(pid)
                            app_name = process.name()
                    except:
                        pass

                return {
                    "app_name": app_name,
                    "window_title": self.privacy_filter.filter_window_title(
                        window_title
                    ),
                }
        except:
            pass

        # Fallback methods
        try:
            # Try wmctrl
            result = subprocess.run(
                ["wmctrl", "-l"], capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                for line in lines:
                    if line:  # Get first window as approximation
                        parts = line.split(None, 3)
                        if len(parts) >= 4:
                            return {
                                "app_name": "wmctrl_detected",
                                "window_title": self.privacy_filter.filter_window_title(
                                    parts[3]
                                ),
                            }
        except:
            pass

        return {"app_name": "unknown", "window_title": "unknown"}

    def _get_active_window_windows(self) -> Dict[str, str]:
        """Get active window info on Windows"""
        try:
            import ctypes
            from ctypes import wintypes

            # Get foreground window
            hwnd = ctypes.windll.user32.GetForegroundWindow()

            # Get window title
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
            window_title = buff.value

            # Get process name
            pid = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

            app_name = "unknown"
            if psutil:
                try:
                    process = psutil.Process(pid.value)
                    app_name = process.name()
                except:
                    pass

            return {
                "app_name": app_name,
                "window_title": self.privacy_filter.filter_window_title(window_title),
            }

        except:
            pass

        return {"app_name": "unknown", "window_title": "unknown"}


class ActivityTracker:
    """Tracks user activity events with privacy safeguards"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.activity_log = state_dir / "activity_tracking.jsonl"
        self.privacy_filter = PrivacyFilter()

        # Activity rate tracking
        self.keyboard_events = []
        self.mouse_events = []
        self.max_event_age = 300  # 5 minutes

        # Listeners
        self.keyboard_listener = None
        self.mouse_listener = None
        self.monitoring = False

    def start_monitoring(self):
        """Start activity monitoring"""
        if not pynput:
            logger.warning("pynput not available, activity monitoring disabled")
            return

        if self.monitoring:
            return

        self.monitoring = True

        # Start keyboard listener (only for counting, not content)
        def on_key_press(key):
            self._record_activity_event(
                ActivityType.KEYBOARD, {"timestamp": time.time()}
            )

        def on_mouse_move(x, y):
            self._record_activity_event(
                ActivityType.MOUSE, {"event_type": "move", "timestamp": time.time()}
            )

        def on_mouse_click(x, y, button, pressed):
            if pressed:  # Only count press, not release
                self._record_activity_event(
                    ActivityType.MOUSE,
                    {"event_type": "click", "timestamp": time.time()},
                )

        try:
            self.keyboard_listener = keyboard.Listener(on_press=on_key_press)
            self.mouse_listener = mouse.Listener(
                on_move=on_mouse_move, on_click=on_mouse_click
            )

            self.keyboard_listener.start()
            self.mouse_listener.start()

            logger.info("User activity monitoring started")

        except Exception as e:
            logger.error(f"Failed to start activity monitoring: {e}")
            self.monitoring = False

    def stop_monitoring(self):
        """Stop activity monitoring"""
        self.monitoring = False

        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None

        logger.info("User activity monitoring stopped")

    def _record_activity_event(
        self, activity_type: ActivityType, details: Dict[str, Any]
    ):
        """Record an activity event"""
        if not self.monitoring:
            return

        # Privacy check
        if not self.privacy_filter.is_safe_to_log(details):
            return

        # Add to appropriate event list for rate calculation
        current_time = time.time()
        event_data = {"timestamp": current_time}

        if activity_type == ActivityType.KEYBOARD:
            self.keyboard_events.append(event_data)
        elif activity_type == ActivityType.MOUSE:
            self.mouse_events.append(event_data)

        # Clean old events
        self._clean_old_events()

    def _clean_old_events(self):
        """Remove old events for rate calculation"""
        current_time = time.time()
        cutoff_time = current_time - self.max_event_age

        self.keyboard_events = [
            event for event in self.keyboard_events if event["timestamp"] > cutoff_time
        ]
        self.mouse_events = [
            event for event in self.mouse_events if event["timestamp"] > cutoff_time
        ]

    def get_activity_rates(self) -> Tuple[float, float]:
        """Get current keyboard and mouse activity rates (events per minute)"""
        self._clean_old_events()

        time_window = min(self.max_event_age, 60)  # Use 1 minute for rate calculation

        keyboard_rate = len(self.keyboard_events) * (60.0 / time_window)
        mouse_rate = len(self.mouse_events) * (60.0 / time_window)

        return keyboard_rate, mouse_rate


class PresenceDetector:
    """Detects and classifies user presence state"""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.presence_log = state_dir / "presence_history.jsonl"
        self.system_monitor = SystemActivityMonitor()
        self.activity_tracker = ActivityTracker(state_dir)
        self.current_presence = PresenceState.UNKNOWN

        # Thresholds (configurable)
        self.idle_threshold = 300  # 5 minutes
        self.away_threshold = 900  # 15 minutes
        self.focus_threshold = 1800  # 30 minutes of consistent work

        # Focus detection patterns
        self.focus_apps = {
            "code",
            "vim",
            "emacs",
            "intellij",
            "pycharm",
            "vscode",
            "terminal",
            "konsole",
            "gnome-terminal",
            "xterm",
            "writer",
            "word",
            "docs",
            "excel",
            "calc",
            "photoshop",
            "gimp",
            "blender",
            "maya",
        }

        self.gaming_apps = {
            "steam",
            "game",
            "wow",
            "minecraft",
            "unity",
            "unreal",
            "dota",
            "league",
            "valorant",
            "csgo",
            "overwatch",
        }

        self.meeting_apps = {
            "zoom",
            "teams",
            "meet",
            "skype",
            "discord",
            "slack",
            "webex",
            "gotomeeting",
            "hangouts",
        }

    def start_monitoring(self):
        """Start presence monitoring"""
        self.activity_tracker.start_monitoring()
        logger.info("Presence detection started")

    def stop_monitoring(self):
        """Stop presence monitoring"""
        self.activity_tracker.stop_monitoring()
        logger.info("Presence detection stopped")

    async def get_current_presence(self) -> PresenceSnapshot:
        """Get current user presence snapshot"""
        try:
            timestamp = datetime.now()

            # Get system information
            idle_time = self.system_monitor.get_system_idle_time()
            window_info = self.system_monitor.get_active_window_info()
            keyboard_rate, mouse_rate = self.activity_tracker.get_activity_rates()

            # Classify presence state
            presence_state, confidence, focus_indicators = self._classify_presence(
                idle_time, window_info, keyboard_rate, mouse_rate
            )

            # Create snapshot
            snapshot = PresenceSnapshot(
                timestamp=timestamp,
                presence_state=presence_state,
                idle_time_seconds=idle_time,
                active_application=window_info.get("app_name", "unknown"),
                window_title_hash=window_info.get("window_title", "unknown"),
                keyboard_activity_rate=keyboard_rate,
                mouse_activity_rate=mouse_rate,
                confidence=confidence,
                focus_indicators=focus_indicators,
            )

            # Log snapshot
            await self._log_presence_snapshot(snapshot)

            # Update current presence
            self.current_presence = presence_state

            return snapshot

        except Exception as e:
            logger.error(f"Error getting current presence: {e}")
            return PresenceSnapshot(
                timestamp=datetime.now(),
                presence_state=PresenceState.UNKNOWN,
                idle_time_seconds=0,
                active_application="unknown",
                window_title_hash="unknown",
                keyboard_activity_rate=0,
                mouse_activity_rate=0,
                confidence=0,
                focus_indicators=[],
            )

    def _classify_presence(
        self,
        idle_time: float,
        window_info: Dict[str, str],
        keyboard_rate: float,
        mouse_rate: float,
    ) -> Tuple[PresenceState, float, List[str]]:
        """Classify current presence state"""

        app_name = window_info.get("app_name", "").lower()
        focus_indicators = []
        confidence = 0.8

        # Away detection
        if idle_time > self.away_threshold:
            return PresenceState.AWAY, 0.9, ["long_idle_time"]

        # Gaming detection
        if any(game_app in app_name for game_app in self.gaming_apps):
            focus_indicators.append("gaming_app")
            return PresenceState.GAMING, 0.9, focus_indicators

        # Meeting detection
        if any(meeting_app in app_name for meeting_app in self.meeting_apps):
            focus_indicators.append("meeting_app")
            return PresenceState.MEETING, 0.9, focus_indicators

        # Idle detection
        if idle_time > self.idle_threshold:
            return PresenceState.IDLE, 0.8, ["moderate_idle_time"]

        # Focus detection
        if any(focus_app in app_name for focus_app in self.focus_apps):
            focus_indicators.append("focus_app")

            # Check for sustained activity patterns
            if keyboard_rate > 30 or mouse_rate > 10:  # Active typing/interaction
                focus_indicators.append("high_activity")
                return PresenceState.FOCUSED, 0.9, focus_indicators

        # Active detection
        if keyboard_rate > 5 or mouse_rate > 5:
            return PresenceState.ACTIVE, 0.8, ["recent_activity"]

        # Default to active with lower confidence
        return PresenceState.ACTIVE, 0.6, []

    async def _log_presence_snapshot(self, snapshot: PresenceSnapshot):
        """Log presence snapshot to file"""
        try:
            snapshot_data = asdict(snapshot)
            snapshot_data["timestamp"] = snapshot.timestamp.isoformat()

            with open(self.presence_log, "a") as f:
                f.write(json.dumps(snapshot_data) + "\n")

        except Exception as e:
            logger.error(f"Error logging presence snapshot: {e}")

    def is_good_time_for_interaction(self) -> Tuple[bool, str]:
        """Determine if now is a good time for interaction"""

        if self.current_presence == PresenceState.AWAY:
            return False, "User is away"

        if self.current_presence == PresenceState.GAMING:
            return False, "User is gaming"

        if self.current_presence == PresenceState.MEETING:
            return False, "User is in a meeting"

        if self.current_presence == PresenceState.FOCUSED:
            return False, "User is in focused work mode"

        if self.current_presence == PresenceState.ACTIVE:
            return True, "User is active and available"

        if self.current_presence == PresenceState.IDLE:
            return True, "User is idle and potentially available"

        return False, "Presence state unknown"


# Integration functions for the main ambient loop
async def get_user_presence(state_dir: Path) -> PresenceSnapshot:
    """Get current user presence for ambient loop"""
    detector = PresenceDetector(state_dir)
    return await detector.get_current_presence()


async def check_interaction_timing(state_dir: Path) -> Tuple[bool, str]:
    """Check if timing is good for interaction"""
    detector = PresenceDetector(state_dir)
    await detector.get_current_presence()  # Update current state
    return detector.is_good_time_for_interaction()


def start_presence_monitoring(state_dir: Path) -> PresenceDetector:
    """Start presence monitoring and return detector instance"""
    detector = PresenceDetector(state_dir)
    detector.start_monitoring()
    return detector
