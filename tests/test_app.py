"""
test_app.py - Tests for core FastAPI application

Tests all endpoints including:
- Health checks (/healthz, /livez, /readyz)
- Chat endpoints (/chat)
- History endpoints (/history, /session/{id})
- Admin endpoints (/admin/logs, /admin/config)
- Authentication and authorization
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json

# Import the app under test
from .app import app
from .config_manager import ConfigManager

# Test client
client = TestClient(app)

# Test configuration
TEST_API_KEY = "test-api-key"


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_healthz(self):
        """Test basic health check"""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "timestamp" in data

    def test_livez(self):
        """Test liveness probe"""
        response = client.get("/livez")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True

    @patch("core.app.get_history")
    def test_readyz_healthy(self, mock_get_history):
        """Test readiness probe when healthy"""
        mock_get_history.return_value = []
        response = client.get("/readyz")
        assert response.status_code == 200
        data = response.json()
        assert data["ready"] is True

    @patch("core.app.get_history")
    def test_readyz_unhealthy(self, mock_get_history):
        """Test readiness probe when unhealthy"""
        mock_get_history.side_effect = Exception("Database connection failed")
        response = client.get("/readyz")
        assert response.status_code == 503

    def test_metrics(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "responses_total" in data
        assert "errors_total" in data


class TestChatEndpoints:
    """Test chat-related endpoints"""

    @patch("core.app.cfg")
    @patch("core.app.query_llm")
    @patch("core.app.log_conversation")
    def test_chat_success(self, mock_log, mock_llm, mock_cfg):
        """Test successful chat request"""
        # Mock configuration
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []

        # Mock LLM response
        mock_llm.return_value = "Test response"
        mock_log.return_value = None

        response = client.post(
            "/chat",
            json={"prompt": "Hello", "stream": False},
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Test response"
        assert "session_id" in data

    @patch("core.app.cfg")
    def test_chat_no_api_key(self, mock_cfg):
        """Test chat request without API key"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []

        response = client.post("/chat", json={"prompt": "Hello", "stream": False})

        assert response.status_code == 401

    @patch("core.app.cfg")
    def test_chat_invalid_api_key(self, mock_cfg):
        """Test chat request with invalid API key"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []

        response = client.post(
            "/chat",
            json={"prompt": "Hello", "stream": False},
            headers={"X-API-Key": "invalid-key"},
        )

        assert response.status_code == 401

    @patch("core.app.cfg")
    @patch("core.app.query_llm")
    def test_chat_llm_error(self, mock_llm, mock_cfg):
        """Test chat request when LLM fails"""
        from .llm import LLMBackendError

        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_llm.side_effect = LLMBackendError("LLM service down")

        response = client.post(
            "/chat",
            json={"prompt": "Hello", "stream": False},
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert response.status_code == 503


class TestHistoryEndpoints:
    """Test history-related endpoints"""

    @patch("core.app.cfg")
    @patch("core.app.get_history")
    def test_get_history_success(self, mock_get_history, mock_cfg):
        """Test successful history retrieval"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_get_history.return_value = [
            {"prompt": "Hello", "response": "Hi there", "ts": "2024-01-01T00:00:00"}
        ]

        response = client.get("/history?limit=10", headers={"X-API-Key": TEST_API_KEY})

        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert len(data["history"]) == 1

    @patch("core.app.cfg")
    @patch("core.app.get_session_history")
    def test_get_session_history_success(self, mock_get_session_history, mock_cfg):
        """Test successful session history retrieval"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_get_session_history.return_value = [
            {"prompt": "Hello", "response": "Hi there", "ts": "2024-01-01T00:00:00"}
        ]

        session_id = "test-session-123"
        response = client.get(
            f"/session/{session_id}?limit=10", headers={"X-API-Key": TEST_API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert "history" in data


class TestFeedbackEndpoints:
    """Test feedback endpoints"""

    @patch("core.app.cfg")
    @patch("core.app.set_feedback")
    def test_submit_feedback_success(self, mock_set_feedback, mock_cfg):
        """Test successful feedback submission"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_set_feedback.return_value = None

        response = client.post(
            "/feedback",
            json={"conversation_id": 123, "rating": 5, "comment": "Great response!"},
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestAdminEndpoints:
    """Test admin endpoints"""

    @patch("core.app.cfg")
    @patch("core.app.VegaLogger.list_modules")
    def test_admin_logs_list(self, mock_list_modules, mock_cfg):
        """Test listing log modules"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_list_modules.return_value = ["app", "llm", "db"]

        response = client.get("/admin/logs", headers={"X-API-Key": TEST_API_KEY})

        assert response.status_code == 200
        data = response.json()
        assert data["modules"] == ["app", "llm", "db"]

    @patch("core.app.cfg")
    @patch("core.app.VegaLogger.tail_log")
    def test_admin_logs_tail(self, mock_tail_log, mock_cfg):
        """Test tailing log files"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_tail_log.return_value = ["2024-01-01 00:00:00 - INFO - Test log line"]

        response = client.get(
            "/admin/logs/app?lines=50", headers={"X-API-Key": TEST_API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["module"] == "app"
        assert len(data["lines"]) == 1

    @patch("core.app.cfg")
    @patch("core.app.config_manager.list_modules")
    def test_admin_config_list(self, mock_list_modules, mock_cfg):
        """Test listing config modules"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_list_modules.return_value = ["app", "llm", "ui", "voice"]

        response = client.get("/admin/config", headers={"X-API-Key": TEST_API_KEY})

        assert response.status_code == 200
        data = response.json()
        assert data["modules"] == ["app", "llm", "ui", "voice"]

    @patch("core.app.cfg")
    @patch("core.app.config_manager.get_config")
    def test_admin_config_get(self, mock_get_config, mock_cfg):
        """Test getting module configuration"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_get_config.return_value = {"setting1": "value1", "setting2": "value2"}

        response = client.get("/admin/config/app", headers={"X-API-Key": TEST_API_KEY})

        assert response.status_code == 200
        data = response.json()
        assert data["module"] == "app"
        assert data["config"]["setting1"] == "value1"

    @patch("core.app.cfg")
    @patch("core.app.config_manager.update_config")
    def test_admin_config_update(self, mock_update_config, mock_cfg):
        """Test updating module configuration"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_update_config.return_value = None

        response = client.put(
            "/admin/config/app",
            json={"config": {"setting1": "new_value"}},
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    @patch("core.app.cfg")
    @patch("core.app.config_manager.get_llm_behavior")
    def test_admin_llm_behavior_get(self, mock_get_behavior, mock_cfg):
        """Test getting LLM behavior configuration"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_get_behavior.return_value = {
            "content_moderation": {"censorship_level": "moderate"},
            "response_style": {"personality": "helpful"},
            "model_parameters": {"temperature": 0.7},
        }

        response = client.get(
            "/admin/llm/behavior", headers={"X-API-Key": TEST_API_KEY}
        )

        assert response.status_code == 200
        data = response.json()
        assert "behavior" in data

    @patch("core.app.cfg")
    @patch("core.app.config_manager.update_llm_behavior")
    def test_admin_llm_behavior_update(self, mock_update_behavior, mock_cfg):
        """Test updating LLM behavior configuration"""
        mock_cfg.api_key = TEST_API_KEY
        mock_cfg.api_keys_extra = []
        mock_update_behavior.return_value = None

        response = client.put(
            "/admin/llm/behavior",
            json={
                "censorship_level": "strict",
                "personality": "professional",
                "temperature": 0.5,
            },
            headers={"X-API-Key": TEST_API_KEY},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


class TestRootEndpoint:
    """Test root endpoint"""

    def test_index(self):
        """Test root index page"""
        response = client.get("/")
        assert response.status_code == 200
        assert "Vega2.0" in response.text
        assert "/static/index.html" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
