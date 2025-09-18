"""
Calendar Synchronization System
===============================

Comprehensive calendar integration for Google Calendar and Apple CalDAV
with OAuth2 authentication, event parsing, and contextual intelligence generation.
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
import pickle
import base64

# Google Calendar API
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logging.warning(
        "Google Calendar API not available. Install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
    )

# CalDAV for Apple/other calendars
try:
    import caldav
    from caldav import DAVClient

    CALDAV_AVAILABLE = True
except ImportError:
    CALDAV_AVAILABLE = False
    logging.warning("CalDAV not available. Install: pip install caldav")

from ..database.user_profile_schema import UserProfileDatabase, Calendar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CalendarConfig:
    """Configuration for calendar synchronization"""

    google_credentials_file: str = ""
    google_token_file: str = "token.pickle"
    google_scopes: List[str] = None
    caldav_url: str = ""
    caldav_username: str = ""
    caldav_password: str = ""
    sync_past_days: int = 30
    sync_future_days: int = 90
    auto_categorize_events: bool = True
    stress_prediction: bool = True

    def __post_init__(self):
        if self.google_scopes is None:
            self.google_scopes = ["https://www.googleapis.com/auth/calendar.readonly"]

        # Load configuration from environment variables if not provided
        if not self.google_credentials_file:
            self.google_credentials_file = os.getenv(
                "GOOGLE_CALENDAR_CREDENTIALS", "credentials.json"
            )

        if not self.caldav_url:
            self.caldav_url = os.getenv("CALDAV_URL", "")

        if not self.caldav_username:
            self.caldav_username = os.getenv("CALDAV_USERNAME", "")

        if not self.caldav_password:
            self.caldav_password = os.getenv("CALDAV_PASSWORD", "")

        # Parse Google Calendar scopes from environment if available
        scopes_env = os.getenv("GOOGLE_CALENDAR_SCOPES")
        if scopes_env:
            try:
                import json

                self.google_scopes = json.loads(scopes_env)
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, keep default scopes
                pass


@dataclass
class CalendarEvent:
    """Standardized calendar event structure"""

    external_id: str
    title: str
    description: str = ""
    start_time: datetime = None
    end_time: datetime = None
    location: str = ""
    source: str = ""
    is_all_day: bool = False
    is_recurring: bool = False
    recurrence_pattern: Dict = None
    attendees: List[str] = None
    status: str = "confirmed"

    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []
        if self.recurrence_pattern is None:
            self.recurrence_pattern = {}


class GoogleCalendarSync:
    """Google Calendar integration"""

    def __init__(self, config: CalendarConfig):
        self.config = config
        self.service = None
        self.credentials = None

    def authenticate(self) -> bool:
        """Authenticate with Google Calendar API"""
        if not GOOGLE_AVAILABLE:
            logger.error("Google Calendar API not available")
            return False

        creds = None
        token_path = os.path.join(
            os.path.dirname(__file__), self.config.google_token_file
        )

        # Load existing token
        if os.path.exists(token_path):
            with open(token_path, "rb") as token:
                creds = pickle.load(token)

        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")
                    creds = None

            if not creds:
                credentials_path = os.path.join(
                    os.path.dirname(__file__), self.config.google_credentials_file
                )
                if not os.path.exists(credentials_path):
                    logger.error(
                        f"Google credentials file not found: {credentials_path}"
                    )
                    logger.info("Download credentials.json from Google Cloud Console")
                    return False

                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, self.config.google_scopes
                )
                creds = flow.run_local_server(port=0)

            # Save credentials
            with open(token_path, "wb") as token:
                pickle.dump(creds, token)

        self.credentials = creds
        self.service = build("calendar", "v3", credentials=creds)
        return True

    def list_calendars(self) -> List[Dict]:
        """List all available calendars"""
        if not self.service:
            return []

        try:
            calendars_result = self.service.calendarList().list().execute()
            calendars = calendars_result.get("items", [])

            calendar_info = []
            for calendar in calendars:
                calendar_info.append(
                    {
                        "id": calendar["id"],
                        "name": calendar["summary"],
                        "description": calendar.get("description", ""),
                        "primary": calendar.get("primary", False),
                        "access_role": calendar.get("accessRole", ""),
                        "color": calendar.get("backgroundColor", ""),
                        "timezone": calendar.get("timeZone", ""),
                    }
                )

            return calendar_info

        except Exception as e:
            logger.error(f"Error listing calendars: {e}")
            return []

    def fetch_events(
        self,
        calendar_id: str = "primary",
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[CalendarEvent]:
        """Fetch events from Google Calendar"""
        if not self.service:
            return []

        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.config.sync_past_days)
        if end_date is None:
            end_date = datetime.now() + timedelta(days=self.config.sync_future_days)

        try:
            events_result = (
                self.service.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=start_date.isoformat() + "Z",
                    timeMax=end_date.isoformat() + "Z",
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )

            events = events_result.get("items", [])
            calendar_events = []

            for event in events:
                # Parse start/end times
                start = event["start"].get("dateTime", event["start"].get("date"))
                end = event["end"].get("dateTime", event["end"].get("date"))

                # Handle all-day events
                is_all_day = "date" in event["start"]

                if is_all_day:
                    start_time = datetime.fromisoformat(start)
                    end_time = datetime.fromisoformat(end)
                else:
                    start_time = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(end.replace("Z", "+00:00"))

                # Extract attendees
                attendees = []
                for attendee in event.get("attendees", []):
                    attendees.append(attendee.get("email", ""))

                # Check for recurrence
                is_recurring = "recurrence" in event
                recurrence_pattern = {}
                if is_recurring:
                    recurrence_pattern = {
                        "rules": event.get("recurrence", []),
                        "recurring_event_id": event.get("recurringEventId", ""),
                    }

                calendar_event = CalendarEvent(
                    external_id=event["id"],
                    title=event.get("summary", "No Title"),
                    description=event.get("description", ""),
                    start_time=start_time,
                    end_time=end_time,
                    location=event.get("location", ""),
                    source="google",
                    is_all_day=is_all_day,
                    is_recurring=is_recurring,
                    recurrence_pattern=recurrence_pattern,
                    attendees=attendees,
                    status=event.get("status", "confirmed"),
                )

                calendar_events.append(calendar_event)

            return calendar_events

        except Exception as e:
            logger.error(f"Error fetching Google Calendar events: {e}")
            return []


class CalDAVSync:
    """CalDAV integration for Apple Calendar and others"""

    def __init__(self, config: CalendarConfig):
        self.config = config
        self.client = None
        self.principal = None

    def authenticate(self) -> bool:
        """Authenticate with CalDAV server"""
        if not CALDAV_AVAILABLE:
            logger.error("CalDAV not available")
            return False

        if not self.config.caldav_url or not self.config.caldav_username:
            logger.error("CalDAV configuration incomplete")
            return False

        try:
            self.client = DAVClient(
                url=self.config.caldav_url,
                username=self.config.caldav_username,
                password=self.config.caldav_password,
            )
            self.principal = self.client.principal()
            return True

        except Exception as e:
            logger.error(f"CalDAV authentication failed: {e}")
            return False

    def list_calendars(self) -> List[Dict]:
        """List available CalDAV calendars"""
        if not self.principal:
            return []

        try:
            calendars = self.principal.calendars()
            calendar_info = []

            for calendar in calendars:
                calendar_info.append(
                    {
                        "id": calendar.url,
                        "name": calendar.get_display_name(),
                        "description": getattr(calendar, "description", ""),
                        "color": getattr(calendar, "calendar_color", ""),
                        "supported_components": getattr(
                            calendar, "supported_calendar_component_set", []
                        ),
                    }
                )

            return calendar_info

        except Exception as e:
            logger.error(f"Error listing CalDAV calendars: {e}")
            return []

    def fetch_events(
        self,
        calendar_url: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[CalendarEvent]:
        """Fetch events from CalDAV calendar"""
        if not self.principal:
            return []

        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.config.sync_past_days)
        if end_date is None:
            end_date = datetime.now() + timedelta(days=self.config.sync_future_days)

        try:
            calendars = self.principal.calendars()
            if calendar_url:
                calendars = [cal for cal in calendars if cal.url == calendar_url]

            calendar_events = []

            for calendar in calendars:
                events = calendar.date_search(start=start_date, end=end_date)

                for event in events:
                    # Parse iCalendar data
                    ical_data = event.data
                    calendar_event = self._parse_ical_event(ical_data)
                    if calendar_event:
                        calendar_events.append(calendar_event)

            return calendar_events

        except Exception as e:
            logger.error(f"Error fetching CalDAV events: {e}")
            return []

    def _parse_ical_event(self, ical_data: str) -> Optional[CalendarEvent]:
        """Parse iCalendar event data"""
        try:
            from icalendar import Calendar as iCalendar

            cal = iCalendar.from_ical(ical_data)

            for component in cal.walk():
                if component.name == "VEVENT":
                    # Extract event properties
                    summary = str(component.get("summary", "No Title"))
                    description = str(component.get("description", ""))
                    location = str(component.get("location", ""))
                    uid = str(component.get("uid", ""))

                    # Parse dates
                    dtstart = component.get("dtstart")
                    dtend = component.get("dtend")

                    start_time = dtstart.dt if dtstart else None
                    end_time = dtend.dt if dtend else None

                    # Check if all-day event
                    is_all_day = hasattr(dtstart.dt, "date") and not hasattr(
                        dtstart.dt, "time"
                    )

                    # Check for recurrence
                    rrule = component.get("rrule")
                    is_recurring = rrule is not None
                    recurrence_pattern = {"rrule": str(rrule)} if rrule else {}

                    return CalendarEvent(
                        external_id=uid,
                        title=summary,
                        description=description,
                        start_time=start_time,
                        end_time=end_time,
                        location=location,
                        source="caldav",
                        is_all_day=is_all_day,
                        is_recurring=is_recurring,
                        recurrence_pattern=recurrence_pattern,
                    )

        except ImportError:
            logger.error(
                "icalendar library not available. Install with: pip install icalendar"
            )
        except Exception as e:
            logger.error(f"Error parsing iCal event: {e}")

        return None


class CalendarIntelligence:
    """Intelligence and context analysis for calendar events"""

    def __init__(self):
        self.event_categories = {
            "work": ["meeting", "standup", "presentation", "deadline", "project"],
            "school": ["class", "lecture", "exam", "assignment", "study", "course"],
            "medical": ["doctor", "appointment", "checkup", "therapy", "dentist"],
            "personal": ["birthday", "anniversary", "vacation", "holiday"],
            "social": ["dinner", "party", "hangout", "date", "wedding"],
        }

        self.stress_indicators = {
            "high": ["exam", "deadline", "presentation", "interview", "surgery"],
            "medium": ["meeting", "appointment", "class", "project"],
            "low": ["vacation", "holiday", "leisure", "break"],
        }

    def categorize_event(self, event: CalendarEvent) -> str:
        """Automatically categorize event type"""
        text = (event.title + " " + event.description + " " + event.location).lower()

        for category, keywords in self.event_categories.items():
            if any(keyword in text for keyword in keywords):
                return category

        return "other"

    def predict_stress_level(self, event: CalendarEvent) -> float:
        """Predict stress level for event (0.0 to 1.0)"""
        text = (event.title + " " + event.description).lower()

        for level, keywords in self.stress_indicators.items():
            if any(keyword in text for keyword in keywords):
                if level == "high":
                    return 0.8
                elif level == "medium":
                    return 0.5
                elif level == "low":
                    return 0.2

        # Default stress based on duration
        if event.start_time and event.end_time:
            duration_hours = (event.end_time - event.start_time).total_seconds() / 3600
            if duration_hours > 4:
                return 0.6
            elif duration_hours > 2:
                return 0.4

        return 0.3  # Default moderate stress

    def extract_context_insights(self, events: List[CalendarEvent]) -> Dict[str, Any]:
        """Extract contextual insights from calendar events"""
        insights = {
            "total_events": len(events),
            "categories": {},
            "busy_days": [],
            "stress_periods": [],
            "patterns": {},
            "upcoming_deadlines": [],
            "free_time_blocks": [],
        }

        # Categorize events
        for event in events:
            category = self.categorize_event(event)
            insights["categories"][category] = (
                insights["categories"].get(category, 0) + 1
            )

        # Find busy days (more than 5 events)
        day_counts = {}
        for event in events:
            if event.start_time:
                day = event.start_time.date()
                day_counts[day] = day_counts.get(day, 0) + 1

        insights["busy_days"] = [
            str(day) for day, count in day_counts.items() if count > 5
        ]

        # Identify high-stress periods
        for event in events:
            if event.start_time:
                stress = self.predict_stress_level(event)
                if stress > 0.7:
                    insights["stress_periods"].append(
                        {
                            "date": event.start_time.isoformat(),
                            "event": event.title,
                            "stress_level": stress,
                        }
                    )

        # Find upcoming deadlines (next 7 days)
        now = datetime.now()
        week_ahead = now + timedelta(days=7)

        for event in events:
            if (
                event.start_time
                and now <= event.start_time <= week_ahead
                and any(
                    keyword in event.title.lower()
                    for keyword in ["deadline", "due", "exam"]
                )
            ):
                insights["upcoming_deadlines"].append(
                    {
                        "title": event.title,
                        "date": event.start_time.isoformat(),
                        "days_until": (event.start_time - now).days,
                    }
                )

        return insights


class CalendarSync:
    """Main calendar synchronization orchestrator"""

    def __init__(self, db: UserProfileDatabase, config: CalendarConfig = None):
        self.db = db
        self.config = config or CalendarConfig()
        self.google_sync = GoogleCalendarSync(self.config)
        self.caldav_sync = CalDAVSync(self.config)
        self.intelligence = CalendarIntelligence()

    async def sync_all_calendars(self) -> Dict[str, Any]:
        """Synchronize all configured calendars"""
        sync_results = {
            "sync_start": datetime.now().isoformat(),
            "google_events": 0,
            "caldav_events": 0,
            "total_events_stored": 0,
            "insights": {},
            "errors": [],
        }

        all_events = []

        # Sync Google Calendar
        if self.google_sync.authenticate():
            try:
                calendars = self.google_sync.list_calendars()
                for calendar in calendars:
                    events = self.google_sync.fetch_events(calendar["id"])
                    all_events.extend(events)
                    sync_results["google_events"] += len(events)
                    logger.info(
                        f"Synced {len(events)} events from Google calendar: {calendar['name']}"
                    )

            except Exception as e:
                error_msg = f"Google Calendar sync error: {e}"
                sync_results["errors"].append(error_msg)
                logger.error(error_msg)

        # Sync CalDAV calendars
        if self.caldav_sync.authenticate():
            try:
                events = self.caldav_sync.fetch_events()
                all_events.extend(events)
                sync_results["caldav_events"] = len(events)
                logger.info(f"Synced {len(events)} events from CalDAV")

            except Exception as e:
                error_msg = f"CalDAV sync error: {e}"
                sync_results["errors"].append(error_msg)
                logger.error(error_msg)

        # Store events in database
        if all_events:
            stored_count = await self._store_events(all_events)
            sync_results["total_events_stored"] = stored_count

        # Generate insights
        if all_events:
            sync_results["insights"] = self.intelligence.extract_context_insights(
                all_events
            )

        sync_results["sync_end"] = datetime.now().isoformat()
        return sync_results

    async def _store_events(self, events: List[CalendarEvent]) -> int:
        """Store calendar events in database"""
        session = self.db.get_session()
        stored_count = 0

        try:
            for event in events:
                # Check if event already exists
                existing = (
                    session.query(Calendar)
                    .filter(
                        Calendar.external_id == event.external_id,
                        Calendar.source == event.source,
                    )
                    .first()
                )

                if existing:
                    # Update existing event
                    existing.title = event.title
                    existing.description = event.description
                    existing.start_time = event.start_time
                    existing.end_time = event.end_time
                    existing.location = event.location
                    existing.updated_at = datetime.now()
                else:
                    # Create new event
                    calendar_record = Calendar(
                        title=event.title,
                        description=event.description,
                        start_time=event.start_time,
                        end_time=event.end_time,
                        location=event.location,
                        event_type=self.intelligence.categorize_event(event),
                        priority="medium",  # Default, could be enhanced
                        stress_level=self.intelligence.predict_stress_level(event),
                        is_recurring=event.is_recurring,
                        recurrence_pattern=event.recurrence_pattern,
                        source=event.source,
                        external_id=event.external_id,
                        status="scheduled",
                    )
                    session.add(calendar_record)
                    stored_count += 1

            session.commit()
            logger.info(f"Stored {stored_count} new calendar events")

        except Exception as e:
            session.rollback()
            logger.error(f"Error storing calendar events: {e}")
        finally:
            session.close()

        return stored_count

    def get_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Get upcoming events for context"""
        session = self.db.get_session()

        try:
            end_date = datetime.now() + timedelta(days=days_ahead)

            events = (
                session.query(Calendar)
                .filter(
                    Calendar.start_time >= datetime.now(),
                    Calendar.start_time <= end_date,
                    Calendar.is_active == True,
                )
                .order_by(Calendar.start_time)
                .all()
            )

            return [
                {
                    "title": event.title,
                    "start_time": event.start_time.isoformat(),
                    "event_type": event.event_type,
                    "stress_level": event.stress_level,
                    "priority": event.priority,
                }
                for event in events
            ]

        finally:
            session.close()


class Microsoft365CalendarSync:
    """Microsoft 365/Outlook Calendar integration using Microsoft Graph API"""

    def __init__(self, config: CalendarConfig):
        self.config = config
        self.access_token = None
        self.app_id = os.getenv("MICROSOFT_365_APP_ID", "")
        self.app_secret = os.getenv("MICROSOFT_365_APP_SECRET", "")
        self.tenant_id = os.getenv("MICROSOFT_365_TENANT_ID", "")
        self.redirect_uri = os.getenv(
            "MICROSOFT_365_REDIRECT_URI", "http://localhost:8000/auth/callback"
        )

    def authenticate(self) -> bool:
        """Authenticate with Microsoft Graph API"""
        if not all([self.app_id, self.app_secret, self.tenant_id]):
            logger.error("Microsoft 365 credentials not configured")
            return False

        try:
            import requests

            # Use client credentials flow for service authentication
            auth_url = (
                f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            )

            auth_data = {
                "client_id": self.app_id,
                "client_secret": self.app_secret,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials",
            }

            response = requests.post(auth_url, data=auth_data)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get("access_token")
                logger.info("Microsoft 365 authentication successful")
                return True
            else:
                logger.error(f"Microsoft 365 authentication failed: {response.text}")
                return False

        except ImportError:
            logger.error("requests library required for Microsoft 365 integration")
            return False
        except Exception as e:
            logger.error(f"Microsoft 365 authentication error: {e}")
            return False

    def fetch_events(
        self,
        user_email: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
    ) -> List[CalendarEvent]:
        """Fetch events from Microsoft 365 Calendar"""
        if not self.access_token:
            logger.error("Not authenticated with Microsoft 365")
            return []

        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.config.sync_past_days)
        if end_date is None:
            end_date = datetime.now() + timedelta(days=self.config.sync_future_days)

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            # Default to the authenticated user's calendar if no email specified
            calendar_endpoint = (
                f"https://graph.microsoft.com/v1.0/me/events"
                if not user_email
                else f"https://graph.microsoft.com/v1.0/users/{user_email}/events"
            )

            params = {
                "$filter": f"start/dateTime ge '{start_date.isoformat()}' and end/dateTime le '{end_date.isoformat()}'",
                "$orderby": "start/dateTime",
                "$select": "id,subject,body,start,end,location,isAllDay,recurrence,attendees,importance",
            }

            response = requests.get(calendar_endpoint, headers=headers, params=params)

            if response.status_code != 200:
                logger.error(f"Failed to fetch Microsoft 365 events: {response.text}")
                return []

            events_data = response.json()
            calendar_events = []

            for event in events_data.get("value", []):
                try:
                    # Parse start/end times
                    start_info = event["start"]
                    end_info = event["end"]

                    start_time = datetime.fromisoformat(
                        start_info["dateTime"].replace("Z", "+00:00")
                    )
                    end_time = datetime.fromisoformat(
                        end_info["dateTime"].replace("Z", "+00:00")
                    )

                    # Extract attendees
                    attendees = []
                    for attendee in event.get("attendees", []):
                        email = attendee.get("emailAddress", {}).get("address", "")
                        if email:
                            attendees.append(email)

                    # Handle recurrence
                    is_recurring = event.get("recurrence") is not None
                    recurrence_pattern = {}
                    if is_recurring:
                        recurrence_pattern = {
                            "pattern": event.get("recurrence", {}),
                            "series_id": event.get("seriesMasterId", ""),
                        }

                    calendar_event = CalendarEvent(
                        external_id=event["id"],
                        title=event.get("subject", "No Title"),
                        description=event.get("body", {}).get("content", ""),
                        start_time=start_time,
                        end_time=end_time,
                        location=event.get("location", {}).get("displayName", ""),
                        source="microsoft365",
                        is_all_day=event.get("isAllDay", False),
                        is_recurring=is_recurring,
                        recurrence_pattern=recurrence_pattern,
                        attendees=attendees,
                        priority=self._map_importance(
                            event.get("importance", "normal")
                        ),
                    )

                    calendar_events.append(calendar_event)

                except Exception as e:
                    logger.warning(f"Error parsing Microsoft 365 event: {e}")
                    continue

            logger.info(f"Fetched {len(calendar_events)} events from Microsoft 365")
            return calendar_events

        except Exception as e:
            logger.error(f"Error fetching Microsoft 365 events: {e}")
            return []

    def _map_importance(self, importance: str) -> str:
        """Map Microsoft 365 importance to priority"""
        mapping = {"low": "low", "normal": "medium", "high": "high"}
        return mapping.get(importance.lower(), "medium")


async def run_calendar_sync(
    db_path: str = None, config_dict: Dict = None
) -> Dict[str, Any]:
    """Main function to run calendar synchronization"""
    db = UserProfileDatabase(db_path)

    config = CalendarConfig()
    if config_dict:
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Initialize sync services
    sync = CalendarSync(db, config)

    # Collect results from all calendar providers
    all_results = {}

    # Google Calendar sync
    if config.google_credentials_file and os.path.exists(
        config.google_credentials_file
    ):
        try:
            google_results = await sync.sync_all_calendars()
            all_results["google"] = google_results
        except Exception as e:
            logger.error(f"Google Calendar sync failed: {e}")
            all_results["google"] = {"error": str(e)}

    # Microsoft 365 sync
    ms365_sync = Microsoft365CalendarSync(config)
    if ms365_sync.authenticate():
        try:
            ms365_events = ms365_sync.fetch_events()
            ms365_stored = sync.store_events(ms365_events)
            all_results["microsoft365"] = {
                "events_fetched": len(ms365_events),
                "events_stored": ms365_stored,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Microsoft 365 sync failed: {e}")
            all_results["microsoft365"] = {"error": str(e)}

    # CalDAV sync
    if CALDAV_AVAILABLE and config.caldav_url:
        try:
            caldav_sync = CalDAVSync(config)
            if caldav_sync.connect():
                caldav_events = caldav_sync.fetch_events()
                caldav_stored = sync.store_events(caldav_events)
                all_results["caldav"] = {
                    "events_fetched": len(caldav_events),
                    "events_stored": caldav_stored,
                    "status": "success",
                }
        except Exception as e:
            logger.error(f"CalDAV sync failed: {e}")
            all_results["caldav"] = {"error": str(e)}

    # Calculate summary
    total_events = sum(
        r.get("events_fetched", 0) for r in all_results.values() if isinstance(r, dict)
    )
    total_stored = sum(
        r.get("events_stored", 0) for r in all_results.values() if isinstance(r, dict)
    )

    all_results["summary"] = {
        "total_events_fetched": total_events,
        "total_events_stored": total_stored,
        "providers_synced": len(
            [
                k
                for k, v in all_results.items()
                if k != "summary" and isinstance(v, dict) and "error" not in v
            ]
        ),
        "sync_timestamp": datetime.now().isoformat(),
    }

    return all_results


if __name__ == "__main__":
    # Test calendar sync
    async def main():
        results = await run_calendar_sync()
        print("Calendar Sync Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())
