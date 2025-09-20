#!/usr/bin/env python3
"""
config_manager.py - Configuration management utilities for tests

Provides configuration management classes and utilities used by test modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import os


@dataclass
class LLMBehaviorConfig:
    """Configuration for LLM behavior during testing"""

    model_name: str = "test-model"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_seconds: int = 30
    retry_count: int = 3
    enable_streaming: bool = False
    mock_responses: bool = True


class ConfigManager:
    """Test configuration manager for managing test environment settings"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("test_config.json")
        self._config_data: Dict[str, Any] = {}
        self._llm_behavior = LLMBehaviorConfig()
        self._load_default_config()

    def _load_default_config(self):
        """Load default test configuration"""
        self._config_data = {
            "api_key": "test-api-key",
            "host": "127.0.0.1",
            "port": 8000,
            "database_url": "sqlite:///:memory:",
            "enable_logging": False,
            "log_level": "INFO",
            "mock_external_services": True,
            "test_data_dir": "tests/data",
            "temp_dir": "/tmp/vega_tests",
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        return self._config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config_data[key] = value

    def get_llm_behavior(self) -> LLMBehaviorConfig:
        """Get LLM behavior configuration"""
        return self._llm_behavior

    def set_llm_behavior(self, config: LLMBehaviorConfig) -> None:
        """Set LLM behavior configuration"""
        self._llm_behavior = config

    def get_test_database_url(self) -> str:
        """Get test database URL"""
        return self.get("database_url", "sqlite:///:memory:")

    def get_temp_dir(self) -> Path:
        """Get temporary directory for tests"""
        temp_dir = Path(self.get("temp_dir", "/tmp/vega_tests"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir

    def is_mock_mode(self) -> bool:
        """Check if external services should be mocked"""
        return self.get("mock_external_services", True)

    def get_api_key(self) -> str:
        """Get test API key"""
        return self.get("api_key", "test-api-key")

    def get_host_port(self) -> tuple[str, int]:
        """Get test host and port"""
        host = self.get("host", "127.0.0.1")
        port = self.get("port", 8000)
        return host, port

    def update_from_env(self) -> None:
        """Update configuration from environment variables"""
        env_mappings = {
            "VEGA_TEST_API_KEY": "api_key",
            "VEGA_TEST_HOST": "host",
            "VEGA_TEST_PORT": "port",
            "VEGA_TEST_DB_URL": "database_url",
            "VEGA_TEST_LOG_LEVEL": "log_level",
            "VEGA_TEST_TEMP_DIR": "temp_dir",
        }

        for env_var, config_key in env_mappings.items():
            if env_var in os.environ:
                value = os.environ[env_var]
                # Convert port to int if needed
                if config_key == "port":
                    value = int(value)
                self.set(config_key, value)

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self._load_default_config()
        self._llm_behavior = LLMBehaviorConfig()


# Global test configuration instance
_test_config_manager = None


def get_test_config_manager() -> ConfigManager:
    """Get global test configuration manager instance"""
    global _test_config_manager
    if _test_config_manager is None:
        _test_config_manager = ConfigManager()
        _test_config_manager.update_from_env()
    return _test_config_manager


def reset_test_config() -> None:
    """Reset global test configuration to defaults"""
    global _test_config_manager
    if _test_config_manager is not None:
        _test_config_manager.reset_to_defaults()
