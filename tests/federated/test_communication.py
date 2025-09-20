"""
Comprehensive unit tests for Vega 2.0 federated learning communication module.
Tests network communication, participant registration, message handling, and security.
"""

import pytest
import asyncio
import json
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Dict, Any, List

# Import the communication module
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.vega.federated.communication import (
    CommunicationManager,
    ParticipantInfo,
    FederatedMessage,
    MessageType,
    NetworkMetrics,
    ConnectionPool,
    RetryStrategy,
)


class TestParticipantInfo:
    """Test ParticipantInfo dataclass."""

    def test_participant_info_validation(self):
        """Test validation of ParticipantInfo fields."""
        import time

        participant = ParticipantInfo(
            participant_id="participant_1",
            host="localhost",
            port=8001,
            name="Test Participant",
            capabilities={"model_types": ["pytorch"], "data_size": 1000},
            last_seen=time.time(),
        )

    def test_participant_info_to_dict(self):
        """Test conversion to dictionary."""
        import time

        participant = ParticipantInfo(
            participant_id="participant_1",
            host="localhost",
            port=8001,
            name="Test Participant",
            capabilities={},
            last_seen=time.time(),
        )

        participant_dict = participant.__dict__

        assert isinstance(participant_dict, dict)
        assert participant_dict["participant_id"] == "participant_1"
        assert participant_dict["host"] == "localhost"


class TestFederatedMessage:
    """Test FederatedMessage dataclass."""

    def test_message_creation(self):
        """Test creation of FederatedMessage."""
        import time
        import uuid

        message = FederatedMessage(
            message_type=MessageType.MODEL_UPDATE.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"weights": [1.0, 2.0, 3.0], "round_number": 5},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        assert message.message_type == MessageType.MODEL_UPDATE.value
        assert message.sender_id == "participant_1"
        assert message.recipient_id == "coordinator"
        assert message.data["weights"] == [1.0, 2.0, 3.0]
        assert message.data["round_number"] == 5

    def test_message_serialization(self):
        """Test message serialization to JSON."""
        import time
        import uuid
        import json

        message = FederatedMessage(
            message_type=MessageType.PARTICIPANT_JOIN.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"api_key": "test_key"},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        serialized = json.dumps(message.to_dict())

        assert isinstance(serialized, str)
        data = json.loads(serialized)
        assert data["message_type"] == MessageType.PARTICIPANT_JOIN.value
        assert data["sender_id"] == "participant_1"
        assert data["data"]["api_key"] == "test_key"


class TestMessageType:
    """Test MessageType enum."""

    def test_message_types(self):
        """Test all message type values."""
        assert MessageType.MODEL_UPDATE.value == "model_update"
        assert MessageType.GRADIENT_UPDATE.value == "gradient_update"
        assert MessageType.PARTICIPANT_JOIN.value == "participant_join"
        assert MessageType.PARTICIPANT_LEAVE.value == "participant_leave"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.ERROR.value == "error"


class TestNetworkMetrics:
    """Test NetworkMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        import time

        metrics = NetworkMetrics(
            latency_ms=50.0,
            bandwidth_mbps=100.0,
            packet_loss_percent=0.1,
            connection_count=5,
            bytes_sent=1024,
            bytes_received=2048,
            last_updated=time.time(),
        )

        assert metrics.latency_ms == 50.0
        assert metrics.bandwidth_mbps == 100.0
        assert metrics.packet_loss_percent == 0.1
        assert metrics.connection_count == 5
        assert metrics.bytes_sent == 1024
        assert metrics.bytes_received == 2048

    def test_metrics_update(self):
        """Test metrics updates."""
        import time

        metrics = NetworkMetrics(
            latency_ms=25.0,
            bandwidth_mbps=200.0,
            packet_loss_percent=0.05,
            connection_count=10,
            bytes_sent=512,
            bytes_received=1024,
            last_updated=time.time(),
        )

        # Verify initial values
        assert metrics.latency_ms == 25.0
        assert metrics.bandwidth_mbps == 200.0
        assert metrics.packet_loss_percent == 0.05
        assert metrics.connection_count == 10


class TestRetryStrategy:
    """Test RetryStrategy functionality."""

    def test_retry_strategy_creation(self):
        """Test retry strategy creation."""
        strategy = RetryStrategy(
            max_attempts=5,
            initial_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=False,
        )

        assert strategy.max_attempts == 5
        assert strategy.initial_delay == 1.0
        assert strategy.max_delay == 30.0
        assert strategy.backoff_factor == 2.0
        assert strategy.jitter == False

    def test_retry_strategy_with_jitter(self):
        """Test retry strategy with jitter enabled."""
        strategy = RetryStrategy(
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
            backoff_factor=2.0,
            jitter=True,
        )

        assert strategy.max_attempts == 3
        assert strategy.jitter == True

    def test_retry_strategy_configuration(self):
        """Test different retry strategy configurations."""
        strategy = RetryStrategy(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=5.0,
            backoff_factor=3.0,
            jitter=True,
        )

        assert strategy.max_attempts == 5
        assert strategy.initial_delay == 0.5
        assert strategy.max_delay == 5.0


class TestConnectionPool:
    """Test ConnectionPool functionality."""

    @pytest.fixture
    def connection_pool(self):
        """Create connection pool instance."""
        return ConnectionPool(
            max_connections=10,
            active_connections=0,
            available_connections=10,
            total_requests=0,
            failed_requests=0,
            average_response_time=0.0,
        )

    def test_connection_pool_creation(self, connection_pool):
        """Test connection pool creation."""
        assert connection_pool.max_connections == 10
        assert connection_pool.active_connections == 0
        assert connection_pool.available_connections == 10
        assert connection_pool.total_requests == 0
        assert connection_pool.failed_requests == 0
        assert connection_pool.average_response_time == 0.0

    @pytest.mark.asyncio
    async def test_get_connection(self, connection_pool):
        """Test getting connection from pool."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value = AsyncMock()

            connection = await connection_pool.get_connection("http://localhost:8000")

            assert connection is not None
            assert "http://localhost:8000" in connection_pool.connections

    @pytest.mark.asyncio
    async def test_release_connection(self, connection_pool):
        """Test releasing connection back to pool."""
        with patch("aiohttp.ClientSession") as mock_session:
            mock_session.return_value = AsyncMock()

            # Get and release connection
            connection = await connection_pool.get_connection("http://localhost:8000")
            await connection_pool.release_connection(
                "http://localhost:8000", connection
            )

            # Connection should still be in pool
            assert "http://localhost:8000" in connection_pool.connections


class TestCommunicationManager:
    """Test CommunicationManager class functionality."""

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = AsyncMock()

        # Create a proper mock for the context manager
        mock_context_manager = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)

        # Important: Set return_value instead of side_effect to avoid coroutine issues
        session.post.return_value = mock_context_manager
        session.get.return_value = mock_context_manager
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def communication_manager(self, mock_session):
        """Create CommunicationManager instance with mocks."""
        # Create a patcher that we can start and stop
        patcher = patch("aiohttp.ClientSession", return_value=mock_session)
        patcher.start()

        manager = CommunicationManager(
            participant_id="participant_1", participant_name="Test Participant"
        )

        # Ensure the manager has the mock session
        manager.client._session = mock_session

        yield manager

        patcher.stop()

    def test_communication_manager_init(self):
        """Test CommunicationManager initialization."""
        with patch("aiohttp.ClientSession"):
            manager = CommunicationManager(
                participant_id="participant_1", participant_name="Test Participant"
            )

        assert manager.participant_id == "participant_1"
        assert manager.participant_name == "Test Participant"
        assert hasattr(manager, "encryption")
        assert hasattr(manager, "client")

    @pytest.mark.asyncio
    async def test_send_message_success(self, communication_manager, mock_session):
        """Test successful message sending."""
        # Register coordinator participant to make communication possible
        communication_manager.registry.register_participant(
            participant_id="coordinator",
            host="localhost",
            port=8000,
            name="Coordinator",
            api_key="test-key",
        )

        import time
        import uuid

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"status": "alive"},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        result = await communication_manager.send_message(message)

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_failure(self, communication_manager, mock_session):
        """Test message sending failure."""
        # Setup mock response with error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_session.post.return_value.__aenter__.return_value = mock_response

        import time
        import uuid

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"status": "alive"},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        result = await communication_manager.send_message(message)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_with_retry(self, communication_manager, mock_session):
        """Test message sending with retry logic."""
        # Setup mock to fail first time, succeed second time
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500

        mock_response_success = AsyncMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"status": "success"})

        mock_session.post.return_value.__aenter__.side_effect = [
            mock_response_fail,
            mock_response_success,
        ]

        import time
        import uuid

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"status": "alive"},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        result = await communication_manager.send_message(message)

        assert result is True
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_receive_message(self, communication_manager, mock_session):
        """Test message receiving."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "message_type": "model_update",
                "sender_id": "coordinator",
                "recipient_id": "participant_1",
                "data": {"weights": [1.0, 2.0, 3.0]},
                "session_id": "session_123",
                "timestamp": 1234567890.0,
                "message_id": "msg_123",
            }
        )
        mock_session.get.return_value.__aenter__.return_value = mock_response

        message = await communication_manager.receive_message()

        assert message is not None
        assert message.message_type == MessageType.MODEL_UPDATE.value
        assert message.sender_id == "coordinator"
        assert message.data["weights"] == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_register_participant(self, communication_manager, mock_session):
        """Test participant registration."""
        result = await communication_manager.register_participant(
            "participant_1", "test_api_key"
        )

        assert result is True
        # Verify the participant was added to the registry
        participant = communication_manager.registry.get_participant("participant_1")
        assert participant is not None
        assert participant.participant_id == "participant_1"

    @pytest.mark.asyncio
    async def test_register_participant_failure(
        self, communication_manager, mock_session
    ):
        """Test participant registration failure."""
        # Test invalid input
        result = await communication_manager.register_participant(
            "", "test_api_key"  # Empty participant_id should fail
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat(self, communication_manager, mock_session):
        """Test heartbeat functionality."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "alive"})
        mock_session.post.return_value.__aenter__.return_value = mock_response

        result = await communication_manager.send_heartbeat("participant_1")

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, communication_manager, mock_session):
        """Test connection error handling."""
        # Setup mock to raise connection error
        mock_session.post.side_effect = aiohttp.ClientConnectionError()

        import time
        import uuid

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"status": "alive"},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        result = await communication_manager.send_message(message)

        assert result is False
        assert communication_manager.metrics.connection_errors > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, communication_manager, mock_session):
        """Test timeout handling."""
        # Setup mock to raise timeout error
        mock_session.post.side_effect = asyncio.TimeoutError()

        import time
        import uuid

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT.value,
            sender_id="participant_1",
            recipient_id="coordinator",
            session_id="session_123",
            data={"status": "alive"},
            timestamp=time.time(),
            message_id=str(uuid.uuid4()),
        )

        result = await communication_manager.send_message(message)

        assert result is False

    def test_get_metrics(self, communication_manager):
        """Test metrics retrieval."""
        # Update some metrics
        communication_manager.metrics.total_messages_sent = 5
        communication_manager.metrics.total_messages_received = 3
        communication_manager.metrics.connection_errors = 1

        metrics = communication_manager.get_metrics()

        assert metrics.total_messages_sent == 5
        assert metrics.total_messages_received == 3
        assert metrics.connection_errors == 1

    @pytest.mark.asyncio
    async def test_close_connections(self, communication_manager, mock_session):
        """Test connection cleanup."""
        await communication_manager.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_security_integration(self, communication_manager, mock_session):
        """Test security integration in communications."""
        import time
        import uuid

        with patch(
            "vega.federated.communication.check_api_key"
        ) as mock_check_api, patch(
            "vega.federated.communication.audit_log"
        ) as mock_audit:

            mock_check_api.return_value = True

            # Setup mock response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"status": "success"})
            mock_session.post.return_value.__aenter__.return_value = mock_response

            message = FederatedMessage(
                message_type=MessageType.PARTICIPANT_JOIN.value,
                sender_id="participant_1",
                recipient_id="coordinator",
                session_id="session_123",
                data={"api_key": "test_api_key"},
                timestamp=time.time(),
                message_id=str(uuid.uuid4()),
            )

            result = await communication_manager.send_message(message)

            assert result is True
            # Verify security functions were called
            mock_audit.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
