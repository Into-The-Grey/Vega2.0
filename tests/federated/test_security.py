"""
Comprehensive unit tests for Vega 2.0 federated learning security module.
Tests all security utilities including authentication, audit logging, anomaly detection,
and model verification.
"""

import pytest
import json
import tempfile
import os
import ssl
import hashlib
import hmac
from unittest.mock import patch, mock_open, MagicMock
from typing import Dict, Any

# Import the security module
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.vega.federated.security import (
    check_api_key,
    create_ssl_context,
    audit_log,
    detect_anomalies,
    verify_model_signature,
    create_model_signature,
    validate_participant_data,
    AuditLogger,
)


class TestAPIKeyAuthentication:
    """Test API key authentication functionality."""

    def test_valid_api_key(self):
        """Test validation with valid API key."""
        allowed_keys = {"valid_key_1", "valid_key_2"}
        assert check_api_key("valid_key_1", allowed_keys) is True
        assert check_api_key("valid_key_2", allowed_keys) is True

    def test_invalid_api_key(self):
        """Test validation with invalid API key."""
        allowed_keys = {"valid_key_1", "valid_key_2"}
        assert check_api_key("invalid_key", allowed_keys) is False

    def test_empty_api_key(self):
        """Test validation with empty API key."""
        allowed_keys = {"valid_key_1"}
        assert check_api_key("", allowed_keys) is False
        assert check_api_key(None, allowed_keys) is False

    def test_empty_allowed_keys(self):
        """Test validation with empty allowed keys set."""
        allowed_keys = set()
        assert check_api_key("any_key", allowed_keys) is False


class TestSSLContext:
    """Test SSL context creation functionality."""

    @patch("ssl.create_default_context")
    @patch("ssl.SSLContext.load_cert_chain")
    @patch("ssl.SSLContext.load_verify_locations")
    def test_create_ssl_context_basic(
        self, mock_load_verify, mock_load_cert, mock_create_context
    ):
        """Test basic SSL context creation."""
        mock_ctx = MagicMock()
        mock_create_context.return_value = mock_ctx

        result = create_ssl_context("cert.pem", "key.pem")

        mock_create_context.assert_called_once_with(ssl.Purpose.CLIENT_AUTH)
        mock_load_cert.assert_called_once_with("cert.pem", "key.pem")
        mock_load_verify.assert_not_called()
        assert result == mock_ctx

    @patch("ssl.create_default_context")
    @patch("ssl.SSLContext.load_cert_chain")
    @patch("ssl.SSLContext.load_verify_locations")
    def test_create_ssl_context_with_ca(
        self, mock_load_verify, mock_load_cert, mock_create_context
    ):
        """Test SSL context creation with CA file."""
        mock_ctx = MagicMock()
        mock_create_context.return_value = mock_ctx

        result = create_ssl_context("cert.pem", "key.pem", "ca.pem")

        mock_create_context.assert_called_once_with(ssl.Purpose.CLIENT_AUTH)
        mock_load_cert.assert_called_once_with("cert.pem", "key.pem")
        mock_load_verify.assert_called_once_with("ca.pem")
        assert result == mock_ctx


class TestAuditLogging:
    """Test audit logging functionality."""

    @patch("vega.federated.security.time.time")
    @patch("builtins.open", new_callable=mock_open)
    def test_audit_log_basic(self, mock_file, mock_time):
        """Test basic audit logging."""
        mock_time.return_value = 1234567890.0

        audit_log("test_event", {"key": "value"})

        mock_file.assert_called_once_with("audit.log", "a")
        written_content = mock_file().write.call_args[0][0]
        log_entry = json.loads(written_content)

        assert log_entry["event"] == "test_event"
        assert log_entry["details"] == {"key": "value"}
        assert log_entry["timestamp"] == 1234567890.0
        assert log_entry["participant_id"] is None
        assert log_entry["session_id"] is None

    @patch("vega.federated.security.time.time")
    @patch("builtins.open", new_callable=mock_open)
    def test_audit_log_with_ids(self, mock_file, mock_time):
        """Test audit logging with participant and session IDs."""
        mock_time.return_value = 1234567890.0

        audit_log("test_event", {"key": "value"}, "participant_123", "session_456")

        written_content = mock_file().write.call_args[0][0]
        log_entry = json.loads(written_content)

        assert log_entry["participant_id"] == "participant_123"
        assert log_entry["session_id"] == "session_456"


class TestAnomalyDetection:
    """Test anomaly detection functionality."""

    def test_detect_anomalies_normal_data(self):
        """Test anomaly detection with normal data."""
        data = {
            "weights": {"layer1": [1.0, 2.0, 3.0], "layer2": [0.5, -0.3, 0.8]},
            "metadata": {"epoch": 5},
        }

        result = detect_anomalies(data, "participant_1", "session_1")

        assert result["is_anomalous"] is False
        assert result["anomaly_types"] == []
        assert result["participant_id"] == "participant_1"
        assert result["session_id"] == "session_1"

    def test_detect_anomalies_large_values(self):
        """Test anomaly detection with large values."""
        data = {
            "weights": {
                "layer1": [100.0, 2.0, 3.0],  # 100.0 exceeds threshold
                "layer2": [0.5, -0.3, 0.8],
            },
            "metadata": {"epoch": 5},
        }

        result = detect_anomalies(data, "participant_1", "session_1")

        assert result["is_anomalous"] is True
        assert "large_values" in result["anomaly_types"]
        assert len(result["details"]["large_values"]) == 1
        assert result["details"]["large_values"][0]["value"] == 100.0

    def test_detect_anomalies_nan_values(self):
        """Test anomaly detection with NaN values."""
        data = {
            "weights": {"layer1": [float("nan"), 2.0, 3.0], "layer2": [0.5, -0.3, 0.8]},
            "metadata": {"epoch": 5},
        }

        result = detect_anomalies(data, "participant_1", "session_1")

        assert result["is_anomalous"] is True
        assert "nan_inf_values" in result["anomaly_types"]
        assert len(result["details"]["nan_inf_values"]) == 1

    def test_detect_anomalies_inf_values(self):
        """Test anomaly detection with infinite values."""
        data = {
            "weights": {"layer1": [1.0, float("inf"), 3.0], "layer2": [0.5, -0.3, 0.8]},
            "metadata": {"epoch": 5},
        }

        result = detect_anomalies(data, "participant_1", "session_1")

        assert result["is_anomalous"] is True
        assert "nan_inf_values" in result["anomaly_types"]

    def test_detect_anomalies_unexpected_structure(self):
        """Test anomaly detection with unexpected data structure."""
        data = {"unexpected_key": [1.0, 2.0, 3.0]}

        result = detect_anomalies(data, "participant_1", "session_1")

        assert result["is_anomalous"] is True
        assert "unexpected_structure" in result["anomaly_types"]
        assert "expected_keys" in result["details"]
        assert "available_keys" in result["details"]


class TestModelSignatures:
    """Test model signature and verification functionality."""

    def test_create_model_signature(self):
        """Test model signature creation."""
        model_data = {"weights": [1.0, 2.0, 3.0]}
        secret_key = "test_secret_key"

        signature = create_model_signature(model_data, secret_key)

        # Verify signature format
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA-256 hex digest length

    def test_verify_model_signature_valid(self):
        """Test model signature verification with valid signature."""
        model_data = {"weights": [1.0, 2.0, 3.0]}
        secret_key = "test_secret_key"

        signature = create_model_signature(model_data, secret_key)
        result = verify_model_signature(model_data, signature, secret_key)

        assert result is True

    def test_verify_model_signature_invalid(self):
        """Test model signature verification with invalid signature."""
        model_data = {"weights": [1.0, 2.0, 3.0]}
        secret_key = "test_secret_key"

        invalid_signature = "invalid_signature"
        result = verify_model_signature(model_data, invalid_signature, secret_key)

        assert result is False

    def test_verify_model_signature_tampered_data(self):
        """Test model signature verification with tampered data."""
        original_data = {"weights": [1.0, 2.0, 3.0]}
        tampered_data = {"weights": [1.0, 2.0, 4.0]}  # Changed 3.0 to 4.0
        secret_key = "test_secret_key"

        signature = create_model_signature(original_data, secret_key)
        result = verify_model_signature(tampered_data, signature, secret_key)

        assert result is False


class TestParticipantDataValidation:
    """Test participant data validation functionality."""

    def test_validate_participant_data_valid(self):
        """Test validation with valid participant data."""
        data = {
            "weights": {"layer1": [1.0, 2.0, 3.0], "layer2": [0.5, -0.3, 0.8]},
            "metadata": {"epoch": 5},
        }

        result = validate_participant_data(data, "participant_1", "session_1")

        assert result["is_valid"] is True
        assert result["validation_errors"] == []

    def test_validate_participant_data_anomalous(self):
        """Test validation with anomalous participant data."""
        data = {
            "weights": {
                "layer1": [100.0, 2.0, 3.0],  # Large value
                "layer2": [0.5, float("nan"), 0.8],  # NaN value
            },
            "metadata": {"epoch": 5},
        }

        result = validate_participant_data(data, "participant_1", "session_1")

        assert result["is_valid"] is False
        assert len(result["validation_errors"]) > 0
        assert any("large_values" in error for error in result["validation_errors"])
        assert any("nan_inf_values" in error for error in result["validation_errors"])


class TestAuditLoggerClass:
    """Test AuditLogger class functionality."""

    @patch("builtins.open", new_callable=mock_open)
    def test_audit_logger_init(self, mock_file):
        """Test AuditLogger initialization."""
        logger = AuditLogger("test_audit.log")
        assert logger.log_file == "test_audit.log"

    @patch("vega.federated.security.time.time")
    @patch("builtins.open", new_callable=mock_open)
    def test_audit_logger_log(self, mock_file, mock_time):
        """Test AuditLogger log method."""
        mock_time.return_value = 1234567890.0
        logger = AuditLogger("test_audit.log")

        logger.log("test_event", {"key": "value"}, "participant_1", "session_1")

        mock_file.assert_called_with("test_audit.log", "a")
        written_content = mock_file().write.call_args[0][0]
        log_entry = json.loads(written_content)

        assert log_entry["event"] == "test_event"
        assert log_entry["details"] == {"key": "value"}
        assert log_entry["participant_id"] == "participant_1"
        assert log_entry["session_id"] == "session_1"


if __name__ == "__main__":
    pytest.main([__file__])
