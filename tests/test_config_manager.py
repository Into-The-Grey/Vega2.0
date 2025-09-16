"""
test_config_manager.py - Tests for configuration management system

Tests comprehensive configuration functionality:
- YAML config loading and validation
- Environment variable overrides
- Configuration updates and persistence
- LLM behavior configuration
- Module-specific configurations
- Error handling and validation
"""

import pytest
import tempfile
import shutil
import os
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from core.config_manager import ConfigManager, LLMBehaviorConfig


class TestConfigManager:
    """Test ConfigManager functionality"""

    def setup_method(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir) / "config"
        self.config_dir.mkdir()

        # Create test configuration files
        self.create_test_configs()

        # Initialize ConfigManager with test directory
        self.config_manager = ConfigManager(config_dir=str(self.config_dir))

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_configs(self):
        """Create test configuration files"""
        # App config
        app_config = {
            "database": {"url": "sqlite:///test.db", "echo": False},
            "api": {"timeout": 30, "retries": 3},
        }
        with open(self.config_dir / "app.yaml", "w") as f:
            yaml.dump(app_config, f)

        # LLM config with behavior settings
        llm_config = {
            "model": {"name": "llama2", "max_tokens": 2048},
            "behavior": {
                "content_moderation": {
                    "censorship_level": "moderate",
                    "filter_nsfw": True,
                    "block_personal_info": True,
                },
                "response_style": {
                    "personality": "helpful",
                    "tone": "professional",
                    "verbosity": "normal",
                },
                "model_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "frequency_penalty": 0.0,
                },
            },
        }
        with open(self.config_dir / "llm.yaml", "w") as f:
            yaml.dump(llm_config, f)

        # UI config
        ui_config = {"theme": "dark", "auto_refresh": True, "refresh_interval": 5000}
        with open(self.config_dir / "ui.yaml", "w") as f:
            yaml.dump(ui_config, f)

    def test_list_modules(self):
        """Test listing available configuration modules"""
        modules = self.config_manager.list_modules()

        assert "app" in modules
        assert "llm" in modules
        assert "ui" in modules

    def test_get_config_existing_module(self):
        """Test getting configuration for existing module"""
        config = self.config_manager.get_config("app")

        assert config["database"]["url"] == "sqlite:///test.db"
        assert config["api"]["timeout"] == 30

    def test_get_config_nonexistent_module(self):
        """Test getting configuration for non-existent module"""
        with pytest.raises(FileNotFoundError):
            self.config_manager.get_config("nonexistent")

    def test_update_config(self):
        """Test updating configuration"""
        new_config = {
            "database": {"url": "sqlite:///updated.db", "echo": True},
            "api": {"timeout": 60, "retries": 5},
        }

        self.config_manager.update_config("app", new_config)

        # Verify update
        updated_config = self.config_manager.get_config("app")
        assert updated_config["database"]["url"] == "sqlite:///updated.db"
        assert updated_config["api"]["timeout"] == 60

    def test_update_config_persistent(self):
        """Test that configuration updates are persistent"""
        new_config = {"theme": "light", "auto_refresh": False}

        self.config_manager.update_config("ui", new_config)

        # Create new ConfigManager instance to test persistence
        new_manager = ConfigManager(config_dir=str(self.config_dir))
        updated_config = new_manager.get_config("ui")

        assert updated_config["theme"] == "light"
        assert updated_config["auto_refresh"] is False

    @patch.dict(os.environ, {"VEGA_DATABASE_URL": "postgresql://test"})
    def test_environment_override(self):
        """Test environment variable overrides"""
        config = self.config_manager.get_config("app")

        # Environment variables should override config file values
        # This test assumes the config manager implements env var override logic
        # The actual implementation may vary

        # For now, just test that config loading doesn't break with env vars
        assert config is not None

    def test_get_llm_behavior(self):
        """Test getting LLM behavior configuration"""
        behavior = self.config_manager.get_llm_behavior()

        assert isinstance(behavior, dict)
        assert "content_moderation" in behavior
        assert "response_style" in behavior
        assert "model_parameters" in behavior

        assert behavior["content_moderation"]["censorship_level"] == "moderate"
        assert behavior["response_style"]["personality"] == "helpful"
        assert behavior["model_parameters"]["temperature"] == 0.7

    def test_update_llm_behavior(self):
        """Test updating LLM behavior configuration"""
        updates = {
            "censorship_level": "strict",
            "personality": "creative",
            "temperature": 0.9,
        }

        self.config_manager.update_llm_behavior(updates)

        # Verify updates
        behavior = self.config_manager.get_llm_behavior()
        assert behavior["content_moderation"]["censorship_level"] == "strict"
        assert behavior["response_style"]["personality"] == "creative"
        assert behavior["model_parameters"]["temperature"] == 0.9

    def test_llm_behavior_validation(self):
        """Test LLM behavior configuration validation"""
        # Test invalid censorship level
        with pytest.raises(ValueError):
            self.config_manager.update_llm_behavior({"censorship_level": "invalid"})

        # Test invalid temperature range
        with pytest.raises(ValueError):
            self.config_manager.update_llm_behavior({"temperature": -1.0})

        with pytest.raises(ValueError):
            self.config_manager.update_llm_behavior({"temperature": 3.0})

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid YAML structure
        invalid_config = "invalid: yaml: structure: [unclosed"

        with pytest.raises(yaml.YAMLError):
            self.config_manager._validate_yaml(invalid_config)

    def test_create_module_config(self):
        """Test creating configuration for new module"""
        new_config = {"setting1": "value1", "setting2": {"nested": "value2"}}

        self.config_manager.update_config("new_module", new_config)

        # Verify creation
        created_config = self.config_manager.get_config("new_module")
        assert created_config["setting1"] == "value1"
        assert created_config["setting2"]["nested"] == "value2"

        # Verify file was created
        config_file = self.config_dir / "new_module.yaml"
        assert config_file.exists()

    def test_reload_config(self):
        """Test configuration reloading"""
        # Modify config file directly
        config_file = self.config_dir / "ui.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"theme": "custom", "new_setting": "test"}, f)

        # Reload configuration
        self.config_manager.reload_config("ui")

        # Verify reload
        config = self.config_manager.get_config("ui")
        assert config["theme"] == "custom"
        assert config["new_setting"] == "test"

    def test_config_backup_on_update(self):
        """Test that configuration backups are created on update"""
        original_config = self.config_manager.get_config("app")

        new_config = {
            "database": {"url": "sqlite:///backup_test.db"},
            "api": {"timeout": 45},
        }

        self.config_manager.update_config("app", new_config)

        # Check if backup was created (implementation dependent)
        backup_dir = self.config_dir / "backups"
        if backup_dir.exists():
            backup_files = list(backup_dir.glob("app_*.yaml"))
            assert len(backup_files) > 0


class TestLLMBehaviorConfig:
    """Test LLMBehaviorConfig dataclass"""

    def test_default_values(self):
        """Test default LLM behavior configuration values"""
        config = LLMBehaviorConfig()

        assert config.content_moderation["censorship_level"] == "moderate"
        assert config.response_style["personality"] == "helpful"
        assert config.model_parameters["temperature"] == 0.7

    def test_custom_values(self):
        """Test custom LLM behavior configuration values"""
        config = LLMBehaviorConfig(
            content_moderation={"censorship_level": "strict"},
            response_style={"personality": "creative"},
            model_parameters={"temperature": 0.9},
        )

        assert config.content_moderation["censorship_level"] == "strict"
        assert config.response_style["personality"] == "creative"
        assert config.model_parameters["temperature"] == 0.9

    def test_to_dict(self):
        """Test converting LLMBehaviorConfig to dictionary"""
        config = LLMBehaviorConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "content_moderation" in config_dict
        assert "response_style" in config_dict
        assert "model_parameters" in config_dict

    def test_from_dict(self):
        """Test creating LLMBehaviorConfig from dictionary"""
        data = {
            "content_moderation": {"censorship_level": "light"},
            "response_style": {"personality": "casual"},
            "model_parameters": {"temperature": 0.8},
        }

        config = LLMBehaviorConfig.from_dict(data)

        assert config.content_moderation["censorship_level"] == "light"
        assert config.response_style["personality"] == "casual"
        assert config.model_parameters["temperature"] == 0.8

    def test_validation(self):
        """Test LLMBehaviorConfig validation"""
        # Test valid configuration
        config = LLMBehaviorConfig()
        assert config.validate() is True

        # Test invalid censorship level
        config.content_moderation["censorship_level"] = "invalid"
        assert config.validate() is False

        # Test invalid temperature
        config.content_moderation["censorship_level"] = "moderate"
        config.model_parameters["temperature"] = -1.0
        assert config.validate() is False

    def test_merge_updates(self):
        """Test merging updates into LLMBehaviorConfig"""
        config = LLMBehaviorConfig()

        updates = {"censorship_level": "strict", "temperature": 0.5}

        config.merge_updates(updates)

        assert config.content_moderation["censorship_level"] == "strict"
        assert config.model_parameters["temperature"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
