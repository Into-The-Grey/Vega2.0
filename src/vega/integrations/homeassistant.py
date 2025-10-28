"""
Home Assistant Integration for Vega 2.0
========================================

Enables voice-activated "Hey Vega" through Home Assistant Assist on ANY device:
- iOS/iPadOS companion apps (iPhone, iPad, Apple Watch)
- macOS companion app
- Windows companion app
- Any browser with HA frontend
- Smart speakers with HA Assist integration

Architecture:
    1. User: "Hey Vega, what's the weather?" → HA Assist (on iPhone/Mac/etc)
    2. HA STT → Text transcription
    3. HA → Vega webhook (POST /hass/webhook with text + context)
    4. Vega → LLM processing with conversation context
    5. Vega → HA TTS service (send response back to originating device)
    6. HA TTS → User's device speakers

Features:
- Device-agnostic voice I/O (works with all HA companion apps)
- Conversation Agent API integration (Vega appears in HA Assist)
- Webhook-based event handling (HA automations trigger Vega)
- Bidirectional communication (Vega can query HA state, control devices)
- Session management (track conversations across devices)
- TTS voice customization per device
- Persistent context across voice interactions

Home Assistant Setup Required:
    1. Create Long-Lived Access Token in HA
    2. Configure HA Assist pipeline with wake word
    3. Install Vega custom component (optional, for conversation agent)
    4. Create automation to route Assist commands to Vega webhook
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


class HAVoiceDevice(Enum):
    """Home Assistant voice-enabled device types"""

    IOS = "ios"  # iPhone, iPad, Apple Watch
    MACOS = "macos"
    WINDOWS = "windows"
    BROWSER = "browser"
    SMART_SPEAKER = "smart_speaker"
    UNKNOWN = "unknown"


@dataclass
class HAVoiceContext:
    """Context for a voice interaction from Home Assistant"""

    text: str  # Transcribed speech text
    conversation_id: Optional[str] = None  # HA conversation session ID
    device_id: Optional[str] = None  # Originating device
    device_type: HAVoiceDevice = HAVoiceDevice.UNKNOWN
    user_id: Optional[str] = None  # HA user ID
    language: str = "en"
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class HAVoiceResponse:
    """Response to send back to Home Assistant"""

    text: str  # Response text
    tts_message: Optional[str] = None  # TTS-formatted message (if different)
    conversation_id: Optional[str] = None
    media_player_entity_id: Optional[str] = None  # Target device for TTS
    language: str = "en"
    voice: Optional[str] = None  # TTS voice name

    def to_ha_service_call(self) -> Dict[str, Any]:
        """Format as Home Assistant TTS service call"""
        return {
            "entity_id": self.media_player_entity_id or "media_player.all",
            "message": self.tts_message or self.text,
            "language": self.language,
            "options": {"voice": self.voice} if self.voice else {},
        }


class HomeAssistantClient:
    """
    Async client for Home Assistant REST API

    Handles:
    - Authentication with long-lived access token
    - State queries and device control
    - TTS service calls
    - Conversation Agent API registration
    - Event subscription via webhooks
    """

    def __init__(self, base_url: str, access_token: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self.timeout = timeout
        self._client: Optional[Any] = None
        self._owns_client = False

    async def __aenter__(self):
        # Try to use shared HTTP client from resource manager
        try:
            from ..core.resource_manager import get_resource_manager

            manager = await get_resource_manager()
            self._client = manager.get_http_client_direct()
            self._owns_client = False
            # Note: Resource manager doesn't support custom headers per-client,
            # so we'll need to add headers per-request
        except (ImportError, Exception):
            # Fallback to local client if resource manager unavailable
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                },
            )
            self._owns_client = True
        return self

    async def __aexit__(self, *args):
        # Only close if we created our own client
        if self._client and self._owns_client:
            await self._client.aclose()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests (needed when using shared client)"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def get_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of an entity"""
        try:
            headers = self._get_headers() if not self._owns_client else {}
            response = await self._client.get(
                f"{self.base_url}/api/states/{entity_id}", headers=headers or None
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get state for {entity_id}: {e}")
            return None

    async def call_service(
        self,
        domain: str,
        service: str,
        service_data: Optional[Dict[str, Any]] = None,
        target: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Call a Home Assistant service"""
        try:
            payload = {}
            if service_data:
                payload.update(service_data)
            if target:
                payload["target"] = target

            response = await self._client.post(
                f"{self.base_url}/api/services/{domain}/{service}", json=payload
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to call service {domain}.{service}: {e}")
            return False

    async def send_tts(
        self,
        message: str,
        media_player: Optional[str] = None,
        language: str = "en",
        voice: Optional[str] = None,
    ) -> bool:
        """
        Send TTS message to Home Assistant

        Args:
            message: Text to speak
            media_player: Target entity_id (defaults to all media players)
            language: TTS language code
            voice: TTS voice name (provider-specific)
        """
        service_data = {"message": message, "language": language}

        if media_player:
            service_data["entity_id"] = media_player

        if voice:
            service_data["options"] = {"voice": voice}

        # Try cloud TTS first, fall back to local
        success = await self.call_service("tts", "cloud_say", service_data)
        if not success:
            success = await self.call_service("tts", "speak", service_data)

        return success

    async def register_conversation_agent(
        self, agent_id: str = "vega", name: str = "Vega AI", webhook_url: str = None
    ) -> bool:
        """
        Register Vega as a conversation agent in Home Assistant

        Note: Requires custom component installation
        """
        try:
            payload = {
                "agent_id": agent_id,
                "name": name,
                "webhook_url": webhook_url,
                "supported_languages": ["en", "es", "fr", "de", "it", "pt"],
            }

            response = await self._client.post(
                f"{self.base_url}/api/conversation/agent/register", json=payload
            )
            response.raise_for_status()
            logger.info(f"Successfully registered Vega as conversation agent")
            return True
        except Exception as e:
            logger.warning(f"Could not register conversation agent: {e}")
            logger.info("Falling back to webhook-only mode")
            return False

    async def get_devices(self) -> List[Dict[str, Any]]:
        """Get all registered devices"""
        try:
            response = await self._client.get(
                f"{self.base_url}/api/config/device_registry/list"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get devices: {e}")
            return []

    async def get_conversation_history(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """Get conversation history from Home Assistant"""
        try:
            response = await self._client.get(
                f"{self.base_url}/api/conversation/{conversation_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []

    async def send_notification(
        self,
        message: str,
        title: Optional[str] = None,
        target: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send notification to Home Assistant devices"""
        service_data = {"message": message}
        if title:
            service_data["title"] = title
        if target:
            service_data["target"] = target
        if data:
            service_data["data"] = data

        return await self.call_service("notify", "notify", service_data)

    async def health_check(self) -> bool:
        """Check if Home Assistant is accessible"""
        try:
            response = await self._client.get(f"{self.base_url}/api/")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Home Assistant health check failed: {e}")
            return False


class VegaHomeAssistantBridge:
    """
    Bridge between Vega's LLM backend and Home Assistant's voice interface

    Handles:
    - Voice command processing from HA Assist
    - TTS response delivery to originating device
    - Conversation session management
    - Device tracking and routing
    """

    def __init__(
        self,
        ha_client: HomeAssistantClient,
        vega_chat_callback: Callable[[str, Optional[str]], Any],
    ):
        self.ha_client = ha_client
        self.vega_chat_callback = (
            vega_chat_callback  # Async function to call Vega's chat endpoint
        )
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

    async def process_voice_command(self, context: HAVoiceContext) -> HAVoiceResponse:
        """
        Process voice command from Home Assistant

        Flow:
            1. Extract context (user, device, conversation ID)
            2. Call Vega's chat endpoint with persistent context
            3. Format response for TTS
            4. Track session for follow-up commands
        """
        try:
            # Get or create session
            session_id = context.conversation_id or context.device_id
            if session_id:
                session = self.active_sessions.get(session_id, {})
                vega_session_id = session.get("vega_session_id")
            else:
                vega_session_id = None

            # Call Vega's chat endpoint with conversation context
            logger.info(
                f"Processing voice command from {context.device_type.value}: {context.text[:50]}..."
            )

            vega_response = await self.vega_chat_callback(context.text, vega_session_id)

            # Update session tracking
            if session_id:
                self.active_sessions[session_id] = {
                    "vega_session_id": vega_response.get("session_id"),
                    "last_interaction": datetime.utcnow(),
                    "device_id": context.device_id,
                    "device_type": context.device_type,
                    "user_id": context.user_id,
                }

            # Format response
            response_text = vega_response.get("response", "")

            # Get appropriate media player for device type
            media_player = self._get_media_player_for_device(
                context.device_id, context.device_type
            )

            return HAVoiceResponse(
                text=response_text,
                tts_message=self._format_for_tts(response_text),
                conversation_id=context.conversation_id,
                media_player_entity_id=media_player,
                language=context.language,
                voice=self._get_voice_for_device(context.device_type),
            )

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            return HAVoiceResponse(
                text="I'm sorry, I encountered an error processing that request.",
                conversation_id=context.conversation_id,
            )

    async def send_response(self, response: HAVoiceResponse) -> bool:
        """Send TTS response back to Home Assistant"""
        return await self.ha_client.send_tts(
            message=response.tts_message or response.text,
            media_player=response.media_player_entity_id,
            language=response.language,
            voice=response.voice,
        )

    def _get_media_player_for_device(
        self, device_id: Optional[str], device_type: HAVoiceDevice
    ) -> Optional[str]:
        """Get appropriate media player entity for device"""
        if not device_id:
            return None

        # Map device types to likely media player entity patterns
        type_patterns = {
            HAVoiceDevice.IOS: ["mobile_app", "iphone", "ipad", "apple_watch"],
            HAVoiceDevice.MACOS: ["mac", "macos"],
            HAVoiceDevice.WINDOWS: ["windows", "pc"],
            HAVoiceDevice.BROWSER: ["browser", "cast"],
            HAVoiceDevice.SMART_SPEAKER: ["speaker", "echo", "google_home"],
        }

        # This would need device registry lookup in production
        # For now, return generic patterns
        patterns = type_patterns.get(device_type, [])
        if patterns:
            # Return first matching pattern as entity_id hint
            return f"media_player.{patterns[0]}_{device_id[:8]}"

        return None

    def _get_voice_for_device(self, device_type: HAVoiceDevice) -> Optional[str]:
        """Get appropriate TTS voice for device type"""
        # Customize TTS voice based on device capabilities
        voice_map = {
            HAVoiceDevice.IOS: "en-US-Neural2-J",  # iOS supports high-quality neural voices
            HAVoiceDevice.MACOS: "en-US-Neural2-J",
            HAVoiceDevice.WINDOWS: "en-US-Standard-H",
            HAVoiceDevice.SMART_SPEAKER: "en-US-Standard-B",
            HAVoiceDevice.BROWSER: "en-US-Standard-C",
        }
        return voice_map.get(device_type)

    def _format_for_tts(self, text: str) -> str:
        """Format text for TTS (add pauses, clean up formatting)"""
        # Remove markdown formatting
        tts_text = text.replace("**", "").replace("*", "")
        tts_text = tts_text.replace("`", "")

        # Add natural pauses
        tts_text = tts_text.replace(". ", '. <break time="300ms"/> ')
        tts_text = tts_text.replace("? ", '? <break time="300ms"/> ')
        tts_text = tts_text.replace("! ", '! <break time="300ms"/> ')

        # Truncate for voice if too long
        if len(tts_text) > 500:
            tts_text = tts_text[:497] + "..."

        return tts_text

    async def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Clean up old conversation sessions"""
        cutoff = datetime.utcnow().timestamp() - (max_age_hours * 3600)
        to_remove = [
            session_id
            for session_id, session in self.active_sessions.items()
            if session["last_interaction"].timestamp() < cutoff
        ]

        for session_id in to_remove:
            del self.active_sessions[session_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old HA voice sessions")


# Utility functions for webhook handlers


def parse_ha_webhook_payload(payload: Dict[str, Any]) -> Optional[HAVoiceContext]:
    """
    Parse Home Assistant webhook payload into HAVoiceContext

    Expected payload format:
    {
        "text": "what's the weather",
        "conversation_id": "abc123",
        "device_id": "iphone_john",
        "device_type": "ios",
        "user_id": "user123",
        "language": "en"
    }
    """
    try:
        device_type_str = payload.get("device_type", "unknown")
        device_type = (
            HAVoiceDevice(device_type_str)
            if device_type_str in HAVoiceDevice.__members__.values()
            else HAVoiceDevice.UNKNOWN
        )

        return HAVoiceContext(
            text=payload["text"],
            conversation_id=payload.get("conversation_id"),
            device_id=payload.get("device_id"),
            device_type=device_type,
            user_id=payload.get("user_id"),
            language=payload.get("language", "en"),
        )
    except KeyError as e:
        logger.error(f"Missing required field in webhook payload: {e}")
        return None


async def test_ha_connection(base_url: str, access_token: str) -> bool:
    """Test Home Assistant connection"""
    async with HomeAssistantClient(base_url, access_token) as client:
        return await client.health_check()


# Example usage functions


async def example_register_vega_in_ha(
    ha_url: str, ha_token: str, vega_webhook_url: str
):
    """
    Example: Register Vega as a conversation agent in Home Assistant
    """
    async with HomeAssistantClient(ha_url, ha_token) as client:
        success = await client.register_conversation_agent(
            agent_id="vega", name="Vega AI Assistant", webhook_url=vega_webhook_url
        )

        if success:
            logger.info("✅ Vega registered as HA conversation agent")
        else:
            logger.warning(
                "⚠️ Using webhook-only mode (install custom component for full integration)"
            )


async def example_process_voice_command(
    ha_client: HomeAssistantClient,
    vega_chat_function,
    text: str,
    device_id: str = "iphone_main",
):
    """
    Example: Process a voice command and send TTS response
    """
    bridge = VegaHomeAssistantBridge(ha_client, vega_chat_function)

    context = HAVoiceContext(
        text=text, device_id=device_id, device_type=HAVoiceDevice.IOS
    )

    response = await bridge.process_voice_command(context)
    await bridge.send_response(response)

    logger.info(f"Voice command processed: {text[:50]}... → {response.text[:50]}...")
