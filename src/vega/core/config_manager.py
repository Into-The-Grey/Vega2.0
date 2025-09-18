#!/usr/bin/env python3
"""
Vega Configuration Management System
===================================

Provides centralized configuration management with:
- YAML-based per-module configs
- Environment variable overrides
- Runtime configuration updates
- Validation and type safety
- Human-readable LLM behavior settings
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from copy import deepcopy

from .logging_setup import get_core_logger

logger = get_core_logger()


@dataclass
class LLMBehaviorConfig:
    """Lightweight behavior config matching tests expectations.

    Holds three nested dicts with sensible defaults and provides
    validation and update helpers.
    """

    content_moderation: Dict[str, Any] = field(
        default_factory=lambda: {
            "censorship_level": "moderate",  # none, light, moderate, strict
            "filter_nsfw": True,
            "block_personal_info": True,
        }
    )
    response_style: Dict[str, Any] = field(
        default_factory=lambda: {
            "personality": "helpful",
            "tone": "professional",
            "verbosity": "normal",
        }
    )
    model_parameters: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_moderation": dict(self.content_moderation),
            "response_style": dict(self.response_style),
            "model_parameters": dict(self.model_parameters),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMBehaviorConfig":
        cfg = cls()
        if "content_moderation" in data and isinstance(
            data["content_moderation"], dict
        ):
            cfg.content_moderation.update(data["content_moderation"])
        if "response_style" in data and isinstance(data["response_style"], dict):
            cfg.response_style.update(data["response_style"])
        if "model_parameters" in data and isinstance(data["model_parameters"], dict):
            cfg.model_parameters.update(data["model_parameters"])
        return cfg

    def validate(self) -> bool:
        # Validate censorship level
        allowed = {"none", "light", "moderate", "strict"}
        level = self.content_moderation.get("censorship_level", "moderate")
        if level not in allowed:
            return False
        # Validate temperature range (0.0 - 2.0)
        temp = self.model_parameters.get("temperature", 0.7)
        try:
            if not (0.0 <= float(temp) <= 2.0):
                return False
        except Exception:
            return False
        return True

    def merge_updates(self, updates: Dict[str, Any]) -> None:
        """Merge flat updates into nested dicts.

        Supports keys like "censorship_level", "personality", "temperature".
        """
        for key, value in updates.items():
            if key in self.content_moderation:
                self.content_moderation[key] = value
            elif key in self.response_style:
                self.response_style[key] = value
            elif key in self.model_parameters:
                self.model_parameters[key] = value
            else:
                # Unknown key: ignore to keep behavior simple for tests
                pass


class ConfigManager:
    def list_modules(self) -> list:
        """List all available configuration modules"""
        return self.get_all_modules()

    def update_config(self, module: str, new_config: dict):
        """Update and save configuration for a module"""
        self.save_config(module, new_config)
        self._configs[module] = new_config

    def reload_config(self, module: str):
        """Reload configuration for a module from file"""
        config_file = self.config_dir / f"{module}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file for module '{module}' not found.")
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self._configs[module] = config

    def get_llm_behavior(self) -> dict:
        """Get LLM behavior configuration as dict for this manager instance"""
        llm_cfg = self.load_config("llm", create_default=True)
        behavior_dict = llm_cfg.get("behavior", LLMBehaviorConfig().to_dict())
        return LLMBehaviorConfig.from_dict(behavior_dict).to_dict()

    def update_llm_behavior(self, updates: dict):
        """Update LLM behavior configuration with given updates for this manager instance"""
        cfg = self.load_config("llm", create_default=True)
        behavior = LLMBehaviorConfig.from_dict(cfg.get("behavior", {}))
        behavior.merge_updates(updates)
        if not behavior.validate():
            raise ValueError("Invalid LLM behavior configuration")
        cfg["behavior"] = behavior.to_dict()
        self.save_config("llm", cfg)

    @staticmethod
    def _validate_yaml(yaml_str: str):
        """Validate YAML string, raise YAMLError if invalid"""
        return yaml.safe_load(yaml_str)

    def __init__(self, config_dir: Optional[Path] = None):
        # Accept str or Path
        if config_dir is None:
            self.config_dir = Path("config")
        else:
            self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self._configs: Dict[str, Dict[str, Any]] = {}
        self._env_overrides: Dict[str, str] = {}

        # Load environment overrides
        self._load_env_overrides()

        logger.info(
            f"ConfigManager initialized with config directory: {self.config_dir}"
        )

    def _load_env_overrides(self):
        """Load environment variable overrides"""
        for key, value in os.environ.items():
            if key.startswith("VEGA_"):
                config_key = key[5:].lower()  # Remove VEGA_ prefix
                self._env_overrides[config_key] = value
                logger.debug(f"Environment override: {config_key} = {value}")

    def load_config(self, module: str, create_default: bool = True) -> Dict[str, Any]:
        """Load configuration for a module"""

        if module in self._configs:
            return self._configs[module]

        config_file = self.config_dir / f"{module}.yaml"

        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config for {module} from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config for {module}: {e}")
                config = {}
        else:
            config = {}
            if create_default:
                # Create default config based on module
                config = self._create_default_config(module)
                self.save_config(module, config)
                logger.info(f"Created default config for {module}")

        # Apply environment overrides
        config = self._apply_env_overrides(config, module)

        self._configs[module] = config
        return config

    def save_config(self, module: str, config: Dict[str, Any]):
        """Save configuration for a module"""
        config_file = self.config_dir / f"{module}.yaml"

        try:
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)

            self._configs[module] = config
            logger.info(f"Saved config for {module} to {config_file}")

        except Exception as e:
            logger.error(f"Error saving config for {module}: {e}")
            raise

    def get_config(
        self, module: str, key: Optional[str] = None, default: Any = None
    ) -> Any:
        """Get configuration value(s) for a module"""
        # For test expectations, non-existent module should raise
        # FileNotFoundError rather than creating defaults
        config_file = self.config_dir / f"{module}.yaml"
        if not config_file.exists() and module not in self._configs:
            raise FileNotFoundError(f"Config file for module '{module}' not found.")

        config = self.load_config(module, create_default=False)

        if key is None:
            return config

        # Support nested keys with dot notation
        keys = key.split(".")
        value = config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set_config(self, module: str, key: str, value: Any):
        """Set configuration value for a module"""
        config = self.load_config(module)

        # Support nested keys with dot notation
        keys = key.split(".")
        current = config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        self.save_config(module, config)

        logger.info(f"Set config {module}.{key} = {value}")

    def _apply_env_overrides(
        self, config: Dict[str, Any], module: str
    ) -> Dict[str, Any]:
        """Apply environment variable overrides to config"""
        config = deepcopy(config)

        for env_key, env_value in self._env_overrides.items():
            # Check if this override applies to this module
            if env_key.startswith(f"{module}_"):
                config_key = env_key[len(module) + 1 :]

                # Convert string values to appropriate types
                try:
                    if env_value.lower() in ("true", "false"):
                        value = env_value.lower() == "true"
                    elif env_value.isdigit():
                        value = int(env_value)
                    elif "." in env_value and env_value.replace(".", "").isdigit():
                        value = float(env_value)
                    else:
                        value = env_value

                    # Set the value (supporting nested keys)
                    keys = config_key.split("_")
                    current = config
                    for k in keys[:-1]:
                        if k not in current:
                            current[k] = {}
                        current = current[k]
                    current[keys[-1]] = value

                    logger.debug(
                        f"Applied env override {env_key} to {module}.{config_key}"
                    )

                except Exception as e:
                    logger.warning(f"Failed to apply env override {env_key}: {e}")

        return config

    def _create_default_config(self, module: str) -> Dict[str, Any]:
        """Create default configuration for a module"""
        defaults = {
            "app": {
                "host": "127.0.0.1",
                "port": 8000,
                "debug": False,
                "cors_origins": ["http://localhost:3000"],
                "api_keys": ["vega-default-key"],
                "request_timeout": 30,
                "max_request_size": 10485760,  # 10MB
                "log_level": "INFO",
            },
            "llm": {
                "model_name": "llama3",
                "base_url": "http://localhost:11434",
                "timeout": 60,
                "max_retries": 3,
                "behavior": LLMBehaviorConfig().to_dict(),
                "context_window": 4096,
                "streaming": True,
            },
            "ui": {
                "theme": "dark",
                "auto_refresh": True,
                "refresh_interval": 5,
                "show_debug": False,
                "log_tail_lines": 100,
                "enable_syntax_highlighting": True,
            },
            "voice": {
                "tts_engine": "piper",  # piper, pyttsx3, espeak
                "tts_voice": "en_US-lessac-medium",
                "tts_speed": 1.0,
                "stt_engine": "vosk",  # vosk, whisper
                "stt_model": "vosk-model-en-us-0.22",
                "vad_enabled": True,
                "noise_reduction": True,
                "audio_format": "wav",
                "sample_rate": 16000,
            },
            "datasets": {
                "input_dir": "datasets/samples",
                "output_dir": "datasets/processed",
                "formats": ["txt", "md", "json"],
                "chunk_size": 1024,
                "overlap": 128,
                "quality_threshold": 0.8,
            },
            "training": {
                "output_dir": "training/output",
                "batch_size": 4,
                "learning_rate": 5e-5,
                "epochs": 3,
                "gradient_accumulation_steps": 1,
                "warmup_steps": 100,
                "save_steps": 500,
                "eval_steps": 250,
                "use_lora": True,
                "lora_rank": 16,
                "lora_alpha": 32,
            },
            "integrations": {
                "search_engine": "duckduckgo",
                "max_search_results": 10,
                "fetch_timeout": 10,
                "cache_results": True,
                "cache_ttl": 3600,
                "rate_limit": 10,
            },
            "analysis": {
                "enable_sentiment": True,
                "enable_entities": True,
                "enable_topics": True,
                "model_path": "models/analysis",
                "batch_size": 32,
                "confidence_threshold": 0.7,
            },
            "network": {
                "scan_timeout": 5,
                "max_threads": 50,
                "common_ports": [22, 80, 443, 8000, 8080],
                "subnet_scan": True,
                "service_detection": True,
            },
        }

        return defaults.get(module, {})

    def get_all_modules(self) -> List[str]:
        """Get list of all modules with configurations"""
        modules = set()

        # Add modules with config files
        for config_file in self.config_dir.glob("*.yaml"):
            modules.add(config_file.stem)

        # Add loaded modules
        modules.update(self._configs.keys())

        return sorted(list(modules))

    def validate_config(self, module: str, config: Dict[str, Any]) -> Dict[str, str]:
        """Validate configuration and return any errors"""
        errors = {}

        # Basic validation rules
        validation_rules = {
            "app": {
                "port": (int, lambda x: 1 <= x <= 65535),
                "debug": (bool, None),
                "request_timeout": (int, lambda x: x > 0),
            },
            "llm": {
                "timeout": (int, lambda x: x > 0),
                "max_retries": (int, lambda x: x >= 0),
                "context_window": (int, lambda x: x > 0),
            },
            "voice": {
                "tts_speed": (float, lambda x: 0.1 <= x <= 3.0),
                "sample_rate": (int, lambda x: x in [8000, 16000, 22050, 44100]),
            },
        }

        rules = validation_rules.get(module, {})

        for key, (expected_type, validator) in rules.items():
            if key in config:
                value = config[key]

                # Type check
                if not isinstance(value, expected_type):
                    errors[key] = (
                        f"Expected {expected_type.__name__}, got {type(value).__name__}"
                    )
                    continue

                # Custom validation
                if validator and not validator(value):
                    errors[key] = f"Invalid value: {value}"

        return errors


# Global config manager instance
config_manager = ConfigManager()


# Convenience functions
def get_config(module: str, key: Optional[str] = None, default: Any = None) -> Any:
    """Get configuration value for a module"""
    return config_manager.get_config(module, key, default)


def set_config(module: str, key: str, value: Any):
    """Set configuration value for a module"""
    config_manager.set_config(module, key, value)


def save_config(module: str, config: Dict[str, Any]):
    """Save configuration for a module"""
    config_manager.save_config(module, config)


def get_llm_behavior_config() -> LLMBehaviorConfig:
    """Get the current LLM behavior configuration"""
    behavior_dict = get_config("llm", "behavior", {})
    return LLMBehaviorConfig.from_dict(behavior_dict)


def update_llm_behavior_config(updates: Dict[str, Any]):
    """Update LLM behavior configuration"""
    current = get_llm_behavior_config()
    # Merge updates into nested structure
    current.merge_updates(updates)
    # Validate and raise if invalid as tests expect
    if not current.validate():
        raise ValueError("Invalid LLM behavior configuration")
    # Save back to config
    set_config("llm", "behavior", current.to_dict())
    logger.info(f"Updated LLM behavior config: {updates}")
