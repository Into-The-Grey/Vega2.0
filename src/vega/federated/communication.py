"""
Network Communication Framework for Federated Learning

REST-based communication layer with connection pooling, retry mechanisms,
and participant registration for personal/family federated learning.

Design Principles:
- HTTP/HTTPS REST communication (sufficient for 2-3 participants)
- Connection pooling and retry logic
- Manual participant registration
- Integration with dynamic encryption
- Trusted environment model (warnings vs blocking)
- Ubuntu rack server optimized
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin
import ssl
from pathlib import Path
from enum import Enum


from .encryption import DynamicEncryption, SecureChannel
from .security import check_api_key, audit_log


logger = logging.getLogger(__name__)
try:
    from ..core.config import get_config

    _CONFIG = get_config()
    _API_KEYS = set([_CONFIG.api_key] + list(_CONFIG.api_keys_extra))
except Exception:
    _API_KEYS = set()


class MessageType(Enum):
    """Enumeration of federated message types."""

    MODEL_UPDATE = "model_update"
    GRADIENT_UPDATE = "gradient_update"
    AGGREGATION_REQUEST = "aggregation_request"
    AGGREGATION_RESPONSE = "aggregation_response"
    PARTICIPANT_JOIN = "participant_join"
    PARTICIPANT_LEAVE = "participant_leave"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CONTROL = "control"


@dataclass
@dataclass
class NetworkMetrics:
    """Network performance and connection metrics."""

    latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0
    packet_loss_percent: float = 0.0
    connection_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    last_updated: float = 0.0
    total_messages_sent: int = 0
    total_messages_received: int = 0
    connection_errors: int = 0
    successful_connections: int = 0


@dataclass
class ConnectionPool:
    """Connection pool for managing HTTP connections."""

    max_connections: int = 10
    active_connections: int = 0
    available_connections: int = 10
    total_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    connections: Dict[str, Any] = field(default_factory=dict)

    async def get_connection(self, url: str) -> "MockConnection":
        """Get a connection from the pool."""
        import aiohttp

        self.active_connections += 1
        self.available_connections -= 1
        connection = MockConnection(url=url)
        self.connections[url] = connection
        return connection

    async def release_connection(self, url: str, connection: "MockConnection") -> None:
        """Release a connection back to the pool."""
        self.active_connections -= 1
        self.available_connections += 1


@dataclass
class MockConnection:
    """Mock connection for testing."""

    url: str
    is_closed: bool = False

    async def close(self):
        """Close the connection."""
        self.is_closed = True


@dataclass
class RetryStrategy:
    """Configuration for retry logic."""

    max_attempts: int
    initial_delay: float
    max_delay: float
    backoff_factor: float
    jitter: bool


@dataclass
class ParticipantInfo:
    """Information about a federated learning participant."""

    participant_id: str
    host: str
    port: int
    name: str
    capabilities: Dict[str, Any]
    last_seen: float
    status: str = "active"  # active, inactive, offline

    @property
    def base_url(self) -> str:
        """Get the base URL for this participant."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_online(self, timeout_seconds: int = 300) -> bool:
        """Check if participant was seen recently."""
        return (time.time() - self.last_seen) < timeout_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParticipantInfo":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FederatedMessage:
    """Structured message for federated communication."""

    message_type: str
    sender_id: str
    recipient_id: str
    session_id: str
    data: Any
    timestamp: float
    message_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedMessage":
        """Create from dictionary."""
        return cls(**data)


class NetworkClient:
    """
    HTTP client for federated learning communication.

    Features connection pooling, retry logic, and encryption integration.
    """

    def __init__(
        self,
        participant_id: str,
        encryption: Optional[DynamicEncryption] = None,
        max_connections: int = 10,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        """
        Initialize network client.
        """
        self.participant_id = participant_id
        self.encryption = encryption or DynamicEncryption()
        self.max_connections = max_connections
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.api_key = api_key or (list(_API_KEYS)[0] if _API_KEYS else None)
        self.connector: Optional[aiohttp.TCPConnector] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # Create connector lazily when needed (and event loop exists)
            if self.connector is None:
                self.connector = aiohttp.TCPConnector(
                    limit=self.max_connections,
                    limit_per_host=5,
                    keepalive_timeout=60,
                    enable_cleanup_closed=True,
                )

            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            headers = {"User-Agent": f"Vega-Federated-{self.participant_id}"}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            self._session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=timeout,
                headers=headers,
            )

        return self._session

    async def close(self):
        """Close the client session and connector."""
        if self._session and not self._session.closed:
            await self._session.close()
        if self.connector:
            await self.connector.close()

    async def send_message(
        self,
        recipient_url: str,
        message: FederatedMessage,
        encrypt: bool = True,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send message to recipient.

        Args:
            recipient_url: URL of recipient
            message: Message to send
            encrypt: Whether to encrypt the message

        Returns:
            Response data
        """
        session = await self._get_session()

        # Prepare payload
        payload = message.to_dict()

        # Encrypt if requested
        if encrypt and self.encryption:
            encrypted_payload = self.encryption.encrypt_json(payload)
            payload = {"encrypted": True, "data": encrypted_payload}
        else:
            payload["encrypted"] = False

        # Prepare headers (include API key if provided)
        headers = {}
        key = api_key or self.api_key
        if key:
            headers["X-API-Key"] = key

        # Send with retry logic
        last_exception = None

        # Audit log - message send attempt
        audit_log(
            "message_send_attempt",
            {
                "recipient_url": recipient_url,
                "message_type": message.message_type,
                "message_id": message.message_id,
                "encrypted": encrypt,
                "has_api_key": bool(key),
            },
            participant_id=message.sender_id,
            session_id=message.session_id,
        )

        for attempt in range(self.max_retries + 1):
            try:
                url = urljoin(recipient_url, "/federated/message")
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        # Decrypt response if needed
                        if response_data.get("encrypted") and self.encryption:
                            decrypted_data = self.encryption.decrypt_json(
                                response_data["data"]
                            )

                            # Audit log - successful send
                            audit_log(
                                "message_send_success",
                                {
                                    "recipient_url": recipient_url,
                                    "message_type": message.message_type,
                                    "message_id": message.message_id,
                                    "attempt": attempt + 1,
                                    "response_encrypted": True,
                                },
                                participant_id=message.sender_id,
                                session_id=message.session_id,
                            )
                            return decrypted_data

                        # Audit log - successful send (unencrypted response)
                        audit_log(
                            "message_send_success",
                            {
                                "recipient_url": recipient_url,
                                "message_type": message.message_type,
                                "message_id": message.message_id,
                                "attempt": attempt + 1,
                                "response_encrypted": False,
                            },
                            participant_id=message.sender_id,
                            session_id=message.session_id,
                        )
                        return response_data
                    else:
                        error_text = await response.text()

                        # Audit log - HTTP error
                        audit_log(
                            "message_send_http_error",
                            {
                                "recipient_url": recipient_url,
                                "message_type": message.message_type,
                                "message_id": message.message_id,
                                "attempt": attempt + 1,
                                "status_code": response.status,
                                "error": error_text[:200],  # Truncate long errors
                            },
                            participant_id=message.sender_id,
                            session_id=message.session_id,
                        )

                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=error_text,
                        )
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff

                    # Audit log - retry attempt
                    audit_log(
                        "message_send_retry",
                        {
                            "recipient_url": recipient_url,
                            "message_type": message.message_type,
                            "message_id": message.message_id,
                            "attempt": attempt + 1,
                            "error": str(e)[:200],
                            "retry_in_seconds": wait_time,
                        },
                        participant_id=message.sender_id,
                        session_id=message.session_id,
                    )

                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")

        # Audit log - complete failure
        audit_log(
            "message_send_failure",
            {
                "recipient_url": recipient_url,
                "message_type": message.message_type,
                "message_id": message.message_id,
                "total_attempts": self.max_retries + 1,
                "final_error": (
                    str(last_exception)[:200] if last_exception else "Unknown error"
                ),
            },
            participant_id=message.sender_id,
            session_id=message.session_id,
        )
        if last_exception is not None:
            raise last_exception
        else:
            raise Exception("Unknown error in send_message: no exception captured.")

        raise last_exception

    async def broadcast_message(
        self, recipients: List[str], message: FederatedMessage
    ) -> Dict[str, Any]:
        """
        Broadcast message to multiple recipients.

        Args:
            recipients: List of recipient URLs
            message: Message to broadcast

        Returns:
            Dictionary of recipient -> response
        """
        tasks = []
        for recipient_url in recipients:
            task = self.send_message(recipient_url, message)
            tasks.append((recipient_url, task))

        results = {}
        for recipient_url, task in tasks:
            try:
                response = await task
                results[recipient_url] = {"success": True, "data": response}
            except Exception as e:
                results[recipient_url] = {"success": False, "error": str(e)}
                logger.error(f"Failed to send to {recipient_url}: {e}")

        return results

    async def ping_participant(self, participant_url: str) -> Dict[str, Any]:
        """
        Ping a participant to check availability.

        Args:
            participant_url: URL of participant to ping

        Returns:
            Ping response with status and latency
        """
        start_time = time.time()
        session = await self._get_session()

        try:
            url = urljoin(participant_url, "/federated/ping")
            async with session.get(url) as response:
                latency = time.time() - start_time

                if response.status == 200:
                    data = await response.json()
                    return {"online": True, "latency_ms": latency * 1000, "data": data}
                else:
                    return {
                        "online": False,
                        "latency_ms": latency * 1000,
                        "error": f"HTTP {response.status}",
                    }

        except Exception as e:
            latency = time.time() - start_time
            return {"online": False, "latency_ms": latency * 1000, "error": str(e)}


class ParticipantRegistry:
    """
    Registry for managing federated learning participants.

    Handles manual registration, participant discovery, and status tracking
    for the trusted family environment.
    """

    def __init__(self):
        """Initialize participant registry."""
        self.participants: Dict[str, ParticipantInfo] = {}
        self.client: Optional[NetworkClient] = None

    def register_participant(
        self,
        participant_id: str,
        host: str,
        port: int,
        name: str,
        api_key: Optional[str] = None,
        capabilities: Optional[Dict[str, Any]] = None,
    ) -> ParticipantInfo:
        """
        Manually register a participant.

        Args:
            participant_id: Unique participant identifier
            host: Participant's host address
            port: Participant's port
            name: Human-readable name
            capabilities: Participant capabilities

                    self.api_key = api_key or (list(_API_KEYS)[0] if _API_KEYS else None)
        Returns:
            Created ParticipantInfo
        """
        participant = ParticipantInfo(
            participant_id=participant_id,
            host=host,
            port=port,
            name=name,
            capabilities=capabilities or {},
            last_seen=time.time(),
            status="active",
        )

        self.participants[participant_id] = participant

        # Audit log - participant registration
        audit_log(
            "participant_registered",
            {
                "participant_id": participant_id,
                "host": host,
                "port": port,
                "name": name,
                "capabilities": capabilities or {},
                "has_api_key": bool(api_key),
            },
        )

        logger.info(
            f"Registered participant: {name} ({participant_id}) at {host}:{port}"
        )

        return participant

    def unregister_participant(self, participant_id: str) -> bool:
        """
        Remove a participant from registry.

        Args:
            participant_id: Participant to remove

        Returns:
            True if participant was removed
        """
        if participant_id in self.participants:
            participant = self.participants.pop(participant_id)

            # Audit log - participant unregistration
            audit_log(
                "participant_unregistered",
                {
                    "participant_id": participant_id,
                    "participant_name": participant.name,
                    "host": participant.host,
                    "port": participant.port,
                    "last_status": participant.status,
                },
            )

            logger.info(f"Unregistered participant: {participant.name}")
            return True
        return False

    def get_participant(self, participant_id: str) -> Optional[ParticipantInfo]:
        """Get participant by ID."""
        return self.participants.get(participant_id)

    def list_participants(
        self, status_filter: Optional[str] = None
    ) -> List[ParticipantInfo]:
        """
        List registered participants.

        Args:
            status_filter: Filter by status (active, inactive, offline)

        Returns:
            List of participants
        """
        participants = list(self.participants.values())

        if status_filter:
            participants = [p for p in participants if p.status == status_filter]

        return participants

    def get_active_participants(self) -> List[ParticipantInfo]:
        """Get list of currently active participants."""
        return [
            p
            for p in self.participants.values()
            if p.status == "active" and p.is_online
        ]

    async def ping_all_participants(
        self, client: NetworkClient
    ) -> Dict[str, Dict[str, Any]]:
        """
        Ping all registered participants to check status.

        Args:
            client: Network client for communication

        Returns:
            Dictionary of participant_id -> ping_result
        """
        results = {}

        for participant_id, participant in self.participants.items():
            try:
                ping_result = await client.ping_participant(participant.base_url)
                results[participant_id] = ping_result

                # Update participant status
                if ping_result["online"]:
                    participant.last_seen = time.time()
                    participant.status = "active"
                else:
                    participant.status = "offline"

            except Exception as e:
                results[participant_id] = {"online": False, "error": str(e)}
                participant.status = "offline"
                logger.warning(f"Failed to ping {participant_id}: {e}")

        return results

    def update_participant_status(self, participant_id: str, status: str):
        """Update participant status."""
        if participant_id in self.participants:
            self.participants[participant_id].status = status
            if status == "active":
                self.participants[participant_id].last_seen = time.time()

    def export_registry(self) -> Dict[str, Dict[str, Any]]:
        """Export registry for backup or sharing."""
        return {pid: p.to_dict() for pid, p in self.participants.items()}

    def import_registry(self, registry_data: Dict[str, Dict[str, Any]]):
        """Import registry from backup or sharing."""
        imported_count = 0
        for pid, data in registry_data.items():
            if pid not in self.participants:
                self.participants[pid] = ParticipantInfo.from_dict(data)
                imported_count += 1

        logger.info(f"Imported {imported_count} participants")


class CommunicationManager:
    """
    High-level communication manager for federated learning.

    Combines NetworkClient, ParticipantRegistry, and encryption for
    seamless federated communication.
    """

    def __init__(
        self,
        participant_id: str,
        participant_name: str,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        """
        Initialize communication manager.

        Args:
            participant_id: This participant's unique ID
            participant_name: Human-readable name
            host: This participant's host
            port: This participant's port
        """
        self.participant_id = participant_id
        self.participant_name = participant_name
        self.host = host
        self.port = port

        # Initialize components
        self.encryption = DynamicEncryption()
        self.client = NetworkClient(participant_id, self.encryption)
        self.registry = ParticipantRegistry()
        self._metrics = NetworkMetrics()

        # Register self
        self.registry.register_participant(
            participant_id=participant_id,
            host=host,
            port=port,
            name=participant_name,
            capabilities={
                "model_types": ["pytorch", "tensorflow"],
                "encryption": True,
                "version": "1.0.0",
            },
        )

        logger.info(f"Communication manager initialized for {participant_name}")

    async def connect_to_participant(
        self, participant_id: str, host: str, port: int, name: str
    ) -> bool:
        """
        Connect to and register a new participant.

        Args:
            participant_id: Target participant ID
            host: Target host
            port: Target port
            name: Participant name

        Returns:
            True if connection successful
        """
        try:
            # Audit log - connection attempt
            audit_log(
                "participant_connection_attempt",
                {
                    "target_participant_id": participant_id,
                    "target_host": host,
                    "target_port": port,
                    "target_name": name,
                },
                participant_id=self.participant_id,
            )

            # Register participant
            participant = self.registry.register_participant(
                participant_id=participant_id, host=host, port=port, name=name
            )

            # Test connection
            ping_result = await self.client.ping_participant(participant.base_url)

            if ping_result["online"]:
                # Audit log - successful connection
                audit_log(
                    "participant_connection_success",
                    {
                        "target_participant_id": participant_id,
                        "target_host": host,
                        "target_port": port,
                        "target_name": name,
                        "latency_ms": ping_result.get("latency_ms"),
                    },
                    participant_id=self.participant_id,
                )
                logger.info(f"Successfully connected to {name} at {host}:{port}")
                return True
            else:
                # Audit log - connection failed (ping failed)
                audit_log(
                    "participant_connection_failure",
                    {
                        "target_participant_id": participant_id,
                        "target_host": host,
                        "target_port": port,
                        "target_name": name,
                        "reason": "ping_failed",
                        "error": ping_result.get("error"),
                    },
                    participant_id=self.participant_id,
                )
                logger.error(f"Failed to connect to {name}: {ping_result.get('error')}")
                self.registry.unregister_participant(participant_id)
                return False

        except Exception as e:
            # Audit log - connection error
            audit_log(
                "participant_connection_error",
                {
                    "target_participant_id": participant_id,
                    "target_host": host,
                    "target_port": port,
                    "target_name": name,
                    "error": str(e)[:200],
                },
                participant_id=self.participant_id,
            )
            logger.error(f"Error connecting to participant {name}: {e}")
            return False

    async def send_to_participant(
        self,
        recipient_id: str,
        message_type: str,
        data: Any,
        session_id: str = "default",
        api_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send message to specific participant.

        Args:
            recipient_id: Target participant ID
            message_type: Type of message
            data: Message data
            session_id: Federated learning session ID

        Returns:
            Response data or None if failed
        """
        # API key check
        key = api_key or (getattr(self.client, "api_key", None))
        if not check_api_key(str(key) if key else "", _API_KEYS):
            # Audit log - authentication failure
            audit_log(
                "auth_failure",
                {
                    "operation": "send_to_participant",
                    "recipient_id": recipient_id,
                    "message_type": message_type,
                    "reason": "invalid_api_key",
                },
                participant_id=self.participant_id,
                session_id=session_id,
            )
            logger.error(
                f"API key authentication failed for send_to_participant (recipient {recipient_id})"
            )
            return None

        participant = self.registry.get_participant(recipient_id)
        if not participant:
            # Audit log - participant not found
            audit_log(
                "participant_not_found",
                {
                    "operation": "send_to_participant",
                    "recipient_id": recipient_id,
                    "message_type": message_type,
                },
                participant_id=self.participant_id,
                session_id=session_id,
            )
            logger.error(f"Participant {recipient_id} not found")
            return None

        # Create message
        message = FederatedMessage(
            message_type=message_type,
            sender_id=self.participant_id,
            recipient_id=recipient_id,
            session_id=session_id,
            data=data,
            timestamp=time.time(),
            message_id=f"{self.participant_id}_{int(time.time() * 1000)}",
        )

        try:
            response = await self.client.send_message(
                participant.base_url, message, api_key=key
            )
            participant.last_seen = time.time()
            participant.status = "active"

            # Audit log - successful participant communication
            audit_log(
                "participant_communication_success",
                {
                    "operation": "send_to_participant",
                    "recipient_id": recipient_id,
                    "message_type": message_type,
                    "message_id": message.message_id,
                    "participant_url": participant.base_url,
                },
                participant_id=self.participant_id,
                session_id=session_id,
            )

            return response

        except Exception as e:
            # Audit log - communication failure
            audit_log(
                "participant_communication_failure",
                {
                    "operation": "send_to_participant",
                    "recipient_id": recipient_id,
                    "message_type": message_type,
                    "message_id": message.message_id,
                    "participant_url": participant.base_url,
                    "error": str(e)[:200],
                },
                participant_id=self.participant_id,
                session_id=session_id,
            )
            logger.error(f"Failed to send message to {recipient_id}: {e}")
            participant.status = "offline"
            return None

    async def send_to_participants(
        self,
        recipient_ids: List[str],
        message_type: str,
        data: Any,
        session_id: str = "default",
        api_key: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Send a message concurrently to a subset of participants by ID.

        Returns dict participant_id -> {success: bool, data|error}
        """
        # API key check
        key = api_key or (getattr(self.client, "api_key", None))
        if not check_api_key(str(key) if key else "", _API_KEYS):
            logger.error(f"API key authentication failed for send_to_participants")
            return {
                pid: {"success": False, "error": "API key authentication failed"}
                for pid in recipient_ids
            }

        tasks = []
        for pid in recipient_ids:
            tasks.append(
                (
                    pid,
                    self.send_to_participant(
                        pid, message_type, data, session_id, api_key=key
                    ),
                )
            )
        results: Dict[str, Dict[str, Any]] = {}
        for pid, task in tasks:
            try:
                resp = await task
                if resp is not None:
                    results[pid] = {"success": True, "data": resp}
                else:
                    results[pid] = {"success": False, "error": "No response"}
            except Exception as e:
                results[pid] = {"success": False, "error": str(e)}
        return results

    async def broadcast_to_all(
        self,
        message_type: str,
        data: Any,
        session_id: str = "default",
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Broadcast message to all active participants.

        Args:
            message_type: Type of message
            data: Message data
            session_id: Federated learning session ID

        Returns:
            Dictionary of participant responses
        """
        # API key check
        key = api_key or (getattr(self.client, "api_key", None))
        if not check_api_key(str(key) if key else "", _API_KEYS):
            # Audit log - broadcast authentication failure
            audit_log(
                "auth_failure",
                {
                    "operation": "broadcast_to_all",
                    "message_type": message_type,
                    "reason": "invalid_api_key",
                },
                participant_id=self.participant_id,
                session_id=session_id,
            )
            logger.error("API key authentication failed for broadcast_to_all")
            return {}

        active_participants = self.registry.get_active_participants()

        if not active_participants:
            # Audit log - no participants for broadcast
            audit_log(
                "broadcast_no_participants",
                {"operation": "broadcast_to_all", "message_type": message_type},
                participant_id=self.participant_id,
                session_id=session_id,
            )
            logger.warning("No active participants for broadcast")
            return {}

        # Create message
        message = FederatedMessage(
            message_type=message_type,
            sender_id=self.participant_id,
            recipient_id="*",  # Broadcast
            session_id=session_id,
            data=data,
            timestamp=time.time(),
            message_id=f"{self.participant_id}_broadcast_{int(time.time() * 1000)}",
        )

        # Audit log - broadcast attempt
        participant_ids = [p.participant_id for p in active_participants]
        audit_log(
            "broadcast_attempt",
            {
                "operation": "broadcast_to_all",
                "message_type": message_type,
                "message_id": message.message_id,
                "participant_count": len(active_participants),
                "participant_ids": participant_ids,
            },
            participant_id=self.participant_id,
            session_id=session_id,
        )

        # Send to all participants
        recipient_urls = [p.base_url for p in active_participants]
        results = await self.client.broadcast_message(recipient_urls, message)

        # Update participant statuses and audit results
        successful_participants = []
        failed_participants = []

        for participant in active_participants:
            result = results.get(participant.base_url, {})
            if result.get("success"):
                participant.last_seen = time.time()
                participant.status = "active"
                successful_participants.append(participant.participant_id)
            else:
                participant.status = "offline"
                failed_participants.append(
                    {
                        "participant_id": participant.participant_id,
                        "error": result.get("error", "Unknown error"),
                    }
                )

        # Audit log - broadcast results
        audit_log(
            "broadcast_complete",
            {
                "operation": "broadcast_to_all",
                "message_type": message_type,
                "message_id": message.message_id,
                "total_participants": len(active_participants),
                "successful_count": len(successful_participants),
                "failed_count": len(failed_participants),
                "successful_participants": successful_participants,
                "failed_participants": failed_participants[
                    :5
                ],  # Limit to avoid huge logs
            },
            participant_id=self.participant_id,
            session_id=session_id,
        )

        return results

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all participants.

        Returns:
            Health status report
        """
        ping_results = await self.registry.ping_all_participants(self.client)

        active_count = len([r for r in ping_results.values() if r.get("online")])
        total_count = len(ping_results)

        return {
            "total_participants": total_count,
            "active_participants": active_count,
            "offline_participants": total_count - active_count,
            "encryption_status": self.encryption.get_security_status(),
            "participant_details": ping_results,
        }

    async def send_message(
        self,
        message: "FederatedMessage",
    ) -> bool:
        """Send message using FederatedMessage object."""
        try:
            response = await self.send_to_participant(
                recipient_id=message.recipient_id,
                message_type=message.message_type,
                data=message.data,
                session_id=message.session_id,
            )
            return response is not None
        except Exception:
            return False

    async def receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive message (mock implementation for testing)."""
        # In a real implementation, this would listen for incoming messages
        # For testing, return None or a mock message
        return None

    async def register_participant(
        self, participant_id: str, api_key: Optional[str] = None
    ) -> bool:
        """Register participant (delegate to registry)."""
        # Validate input
        if not participant_id or not participant_id.strip():
            return False

        try:
            result = self.registry.register_participant(
                participant_id=participant_id,
                host="localhost",
                port=8000,
                name=f"Participant-{participant_id}",
                api_key=api_key or "default-key",
            )
            return result is not None
        except Exception:
            return False

    async def send_heartbeat(self, participant_id: str) -> bool:
        """Send heartbeat message."""
        try:
            response = await self.send_to_participant(
                recipient_id=participant_id,
                message_type="heartbeat",
                data={"timestamp": time.time()},
            )
            return response is not None
        except Exception:
            return False

    @property
    def metrics(self) -> NetworkMetrics:
        """Get communication metrics."""
        # Update dynamic values
        self._metrics.connection_count = len(self.registry.participants)
        self._metrics.last_updated = time.time()
        return self._metrics

    async def close(self):
        """Close connections (alias for cleanup)."""
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        await self.client.close()


# Example usage and testing
if __name__ == "__main__":

    async def test_communication():
        # Test communication manager
        manager1 = CommunicationManager("participant_1", "Alice's Device")
        manager2 = CommunicationManager("participant_2", "Bob's Device", port=8001)

        try:
            # Connect participants
            await manager1.connect_to_participant(
                "participant_2", "127.0.0.1", 8001, "Bob's Device"
            )
            await manager2.connect_to_participant(
                "participant_1", "127.0.0.1", 8000, "Alice's Device"
            )

            # Test health check
            health = await manager1.health_check()
            print(f"Health check: {health}")

            # Test message sending (would need actual federated server running)
            # response = await manager1.send_to_participant("participant_2", "test", {"hello": "world"})
            # print(f"Response: {response}")

        finally:
            await manager1.cleanup()
            await manager2.cleanup()

    # Run test
    asyncio.run(test_communication())
