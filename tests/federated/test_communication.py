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

    def test_participant_info_creation(self):
        """Test creation of ParticipantInfo."""
        participant = ParticipantInfo(
            participant_id="participant_1",
            host="localhost",
            port=8001,
            name="Test Participant",
            capabilities={"model_types": ["pytorch"], "data_size": 1000},
        )

        assert participant.participant_id == "participant_1"
        assert participant.host == "localhost"
        assert participant.port == 8001
        assert participant.name == "Test Participant"
        assert participant.capabilities["model_types"] == ["pytorch"]
        assert participant.capabilities["data_size"] == 1000

    def test_participant_info_to_dict(self):
        """Test conversion to dictionary."""
        participant = ParticipantInfo(
            participant_id="participant_1",
            host="localhost",
            port=8001,
            name="Test Participant",
            capabilities={},
        )

        participant_dict = participant.__dict__

        assert isinstance(participant_dict, dict)
        assert participant_dict["participant_id"] == "participant_1"
        assert participant_dict["host"] == "localhost"


class TestFederatedMessage:
    """Test FederatedMessage dataclass."""

    def test_message_creation(self):
        """Test creation of FederatedMessage."""
        message = FederatedMessage(
            message_type=MessageType.MODEL_UPDATE,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"weights": [1.0, 2.0, 3.0]},
            session_id="session_123",
            round_number=5,
        )

        assert message.message_type == MessageType.MODEL_UPDATE
        assert message.sender_id == "participant_1"
        assert message.recipient_id == "coordinator"
        assert message.payload["weights"] == [1.0, 2.0, 3.0]
        assert message.session_id == "session_123"
        assert message.round_number == 5
        assert message.timestamp is not None
        assert message.message_id is not None

    def test_message_serialization(self):
        """Test message serialization to JSON."""
        message = FederatedMessage(
            message_type=MessageType.REGISTRATION,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"api_key": "test_key"},
        )

        serialized = message.to_json()

        assert isinstance(serialized, str)
        data = json.loads(serialized)
        assert data["message_type"] == MessageType.REGISTRATION.value
        assert data["sender_id"] == "participant_1"
        assert data["payload"]["api_key"] == "test_key"


class TestMessageType:
    """Test MessageType enum."""

    def test_message_types(self):
        """Test all message type values."""
        assert MessageType.REGISTRATION.value == "registration"
        assert MessageType.MODEL_UPDATE.value == "model_update"
        assert MessageType.GLOBAL_MODEL.value == "global_model"
        assert MessageType.SESSION_JOIN.value == "session_join"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.ERROR.value == "error"


class TestNetworkMetrics:
    """Test NetworkMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = NetworkMetrics()

        assert metrics.total_messages_sent == 0
        assert metrics.total_messages_received == 0
        assert metrics.total_bytes_sent == 0
        assert metrics.total_bytes_received == 0
        assert metrics.connection_errors == 0
        assert metrics.retry_attempts == 0
        assert metrics.average_latency == 0.0
        assert metrics.success_rate == 0.0

    def test_metrics_update(self):
        """Test metrics updates."""
        metrics = NetworkMetrics()

        # Simulate message transmission
        metrics.total_messages_sent = 10
        metrics.total_bytes_sent = 1024
        metrics.average_latency = 50.5
        metrics.success_rate = 0.95

        assert metrics.total_messages_sent == 10
        assert metrics.total_bytes_sent == 1024
        assert metrics.average_latency == 50.5
        assert metrics.success_rate == 0.95


class TestRetryStrategy:
    """Test RetryStrategy functionality."""

    def test_retry_strategy_creation(self):
        """Test retry strategy creation."""
        strategy = RetryStrategy(
            max_attempts=5, base_delay=1.0, max_delay=30.0, backoff_factor=2.0
        )

        assert strategy.max_attempts == 5
        assert strategy.base_delay == 1.0
        assert strategy.max_delay == 30.0
        assert strategy.backoff_factor == 2.0

    def test_retry_delay_calculation(self):
        """Test retry delay calculation."""
        strategy = RetryStrategy(
            max_attempts=3, base_delay=1.0, max_delay=10.0, backoff_factor=2.0
        )

        # Test exponential backoff
        delay1 = strategy.get_delay(1)
        delay2 = strategy.get_delay(2)
        delay3 = strategy.get_delay(3)

        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_retry_max_delay(self):
        """Test maximum delay enforcement."""
        strategy = RetryStrategy(
            max_attempts=5, base_delay=1.0, max_delay=5.0, backoff_factor=3.0
        )

        # Should be capped at max_delay
        delay = strategy.get_delay(4)  # Would be 27.0 without cap
        assert delay == 5.0


class TestConnectionPool:
    """Test ConnectionPool functionality."""

    @pytest.fixture
    def connection_pool(self):
        """Create connection pool instance."""
        return ConnectionPool(max_connections=10, timeout=30.0)

    def test_connection_pool_creation(self, connection_pool):
        """Test connection pool creation."""
        assert connection_pool.max_connections == 10
        assert connection_pool.timeout == 30.0
        assert len(connection_pool.connections) == 0

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
        session.post = AsyncMock()
        session.get = AsyncMock()
        session.close = AsyncMock()
        return session

    @pytest.fixture
    def communication_manager(self, mock_session):
        """Create CommunicationManager instance with mocks."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            manager = CommunicationManager(
                coordinator_endpoint="http://localhost:8000", api_key="test_api_key"
            )
            manager.session = mock_session
            return manager

    def test_communication_manager_init(self):
        """Test CommunicationManager initialization."""
        with patch("aiohttp.ClientSession"):
            manager = CommunicationManager(
                coordinator_endpoint="http://localhost:8000", api_key="test_api_key"
            )

        assert manager.coordinator_endpoint == "http://localhost:8000"
        assert manager.api_key == "test_api_key"
        assert isinstance(manager.metrics, NetworkMetrics)
        assert isinstance(manager.retry_strategy, RetryStrategy)

    @pytest.mark.asyncio
    async def test_send_message_success(self, communication_manager, mock_session):
        """Test successful message sending."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_session.post.return_value.__aenter__.return_value = mock_response

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"status": "alive"},
        )

        result = await communication_manager.send_message(message)

        assert result is True
        mock_session.post.assert_called_once()
        assert communication_manager.metrics.total_messages_sent == 1

    @pytest.mark.asyncio
    async def test_send_message_failure(self, communication_manager, mock_session):
        """Test message sending failure."""
        # Setup mock response with error
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_session.post.return_value.__aenter__.return_value = mock_response

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"status": "alive"},
        )

        result = await communication_manager.send_message(message)

        assert result is False
        assert communication_manager.metrics.connection_errors > 0

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

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"status": "alive"},
        )

        result = await communication_manager.send_message(message)

        assert result is True
        assert mock_session.post.call_count == 2
        assert communication_manager.metrics.retry_attempts > 0

    @pytest.mark.asyncio
    async def test_receive_message(self, communication_manager, mock_session):
        """Test message receiving."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "message_type": "global_model",
                "sender_id": "coordinator",
                "recipient_id": "participant_1",
                "payload": {"weights": [1.0, 2.0, 3.0]},
                "session_id": "session_123",
                "round_number": 1,
                "timestamp": 1234567890.0,
                "message_id": "msg_123",
            }
        )
        mock_session.get.return_value.__aenter__.return_value = mock_response

        message = await communication_manager.receive_message()

        assert message is not None
        assert message.message_type == MessageType.GLOBAL_MODEL
        assert message.sender_id == "coordinator"
        assert message.payload["weights"] == [1.0, 2.0, 3.0]
        assert communication_manager.metrics.total_messages_received == 1

    @pytest.mark.asyncio
    async def test_register_participant(self, communication_manager, mock_session):
        """Test participant registration."""
        # Setup mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "registered"})
        mock_session.post.return_value.__aenter__.return_value = mock_response

        result = await communication_manager.register_participant(
            "participant_1", "test_api_key"
        )

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_participant_failure(
        self, communication_manager, mock_session
    ):
        """Test participant registration failure."""
        # Setup mock response with error
        mock_response = AsyncMock()
        mock_response.status = 401  # Unauthorized
        mock_session.post.return_value.__aenter__.return_value = mock_response

        result = await communication_manager.register_participant(
            "participant_1", "invalid_api_key"
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

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"status": "alive"},
        )

        result = await communication_manager.send_message(message)

        assert result is False
        assert communication_manager.metrics.connection_errors > 0

    @pytest.mark.asyncio
    async def test_timeout_handling(self, communication_manager, mock_session):
        """Test timeout handling."""
        # Setup mock to raise timeout error
        mock_session.post.side_effect = asyncio.TimeoutError()

        message = FederatedMessage(
            message_type=MessageType.HEARTBEAT,
            sender_id="participant_1",
            recipient_id="coordinator",
            payload={"status": "alive"},
        )

        result = await communication_manager.send_message(message)

        assert result is False
        assert communication_manager.metrics.connection_errors > 0

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
                message_type=MessageType.REGISTRATION,
                sender_id="participant_1",
                recipient_id="coordinator",
                payload={"api_key": "test_api_key"},
            )

            result = await communication_manager.send_message(message)

            assert result is True
            # Verify security functions were called
            mock_audit.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
