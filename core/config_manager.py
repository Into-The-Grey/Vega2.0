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
    """Human-readable LLM behavior configuration"""

    # Content Moderation
    censorship_level: str = "moderate"  # none, light, moderate, strict
    ethical_guidelines: str = "balanced"  # permissive, balanced, strict
    vulgarity_filter: str = "moderate"  # none, light, moderate, strict
    profanity_filter: str = "moderate"  # none, light, moderate, strict

    # Response Style
    personality: str = "helpful"  # professional, helpful, casual, creative
    verbosity: str = "balanced"  # concise, balanced, detailed, verbose
    formality: str = "moderate"  # informal, moderate, formal
    humor_level: str = "light"  # none, light, moderate, high

    # Safety & Ethics
    refuse_harmful: bool = True
    refuse_illegal: bool = True
    refuse_violence: bool = True
    refuse_adult_content: bool = True
    refuse_personal_info: bool = True

    # Creativity & Flexibility
    creativity_level: str = "moderate"  # conservative, moderate, creative, experimental
    follow_instructions: str = "strict"  # flexible, moderate, strict
    challenge_assumptions: bool = False
    ask_clarification: bool = True

    # Model Behavior
    temperature: float = 0.7  # 0.0-2.0, higher = more creative/random
    max_tokens: int = 2048
    top_p: float = 0.9  # 0.0-1.0, nucleus sampling
    frequency_penalty: float = 0.0  # -2.0-2.0, reduce repetition
    presence_penalty: float = 0.0  # -2.0-2.0, encourage new topics

    # Advanced Settings
    system_prompt_prefix: str = ""
    custom_instructions: str = ""
    context_awareness: str = "high"  # low, moderate, high
    memory_retention: str = "session"  # none, session, persistent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "content_moderation": {
                "censorship_level": self.censorship_level,
                "ethical_guidelines": self.ethical_guidelines,
                "vulgarity_filter": self.vulgarity_filter,
                "profanity_filter": self.profanity_filter,
            },
            "response_style": {
                "personality": self.personality,
                "verbosity": self.verbosity,
                "formality": self.formality,
                "humor_level": self.humor_level,
            },
            "safety_ethics": {
                "refuse_harmful": self.refuse_harmful,
                "refuse_illegal": self.refuse_illegal,
                "refuse_violence": self.refuse_violence,
                "refuse_adult_content": self.refuse_adult_content,
                "refuse_personal_info": self.refuse_personal_info,
            },
            "creativity_flexibility": {
                "creativity_level": self.creativity_level,
                "follow_instructions": self.follow_instructions,
                "challenge_assumptions": self.challenge_assumptions,
                "ask_clarification": self.ask_clarification,
            },
            "model_parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            },
            "advanced": {
                "system_prompt_prefix": self.system_prompt_prefix,
                "custom_instructions": self.custom_instructions,
                "context_awareness": self.context_awareness,
                "memory_retention": self.memory_retention,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMBehaviorConfig":
        """Create from dictionary"""
        config = cls()

        # Content moderation
        if "content_moderation" in data:
            cm = data["content_moderation"]
            config.censorship_level = cm.get(
                "censorship_level", config.censorship_level
            )
            config.ethical_guidelines = cm.get(
                "ethical_guidelines", config.ethical_guidelines
            )
            config.vulgarity_filter = cm.get(
                "vulgarity_filter", config.vulgarity_filter
            )
            config.profanity_filter = cm.get(
                "profanity_filter", config.profanity_filter
            )

        # Response style
        if "response_style" in data:
            rs = data["response_style"]
            config.personality = rs.get("personality", config.personality)
            config.verbosity = rs.get("verbosity", config.verbosity)
            config.formality = rs.get("formality", config.formality)
            config.humor_level = rs.get("humor_level", config.humor_level)

        # Safety & ethics
        if "safety_ethics" in data:
            se = data["safety_ethics"]
            config.refuse_harmful = se.get("refuse_harmful", config.refuse_harmful)
            config.refuse_illegal = se.get("refuse_illegal", config.refuse_illegal)
            config.refuse_violence = se.get("refuse_violence", config.refuse_violence)
            config.refuse_adult_content = se.get(
                "refuse_adult_content", config.refuse_adult_content
            )
            config.refuse_personal_info = se.get(
                "refuse_personal_info", config.refuse_personal_info
            )

        # Creativity & flexibility
        if "creativity_flexibility" in data:
            cf = data["creativity_flexibility"]
            config.creativity_level = cf.get(
                "creativity_level", config.creativity_level
            )
            config.follow_instructions = cf.get(
                "follow_instructions", config.follow_instructions
            )
            config.challenge_assumptions = cf.get(
                "challenge_assumptions", config.challenge_assumptions
            )
            config.ask_clarification = cf.get(
                "ask_clarification", config.ask_clarification
            )

        # Model parameters
        if "model_parameters" in data:
            mp = data["model_parameters"]
            config.temperature = mp.get("temperature", config.temperature)
            config.max_tokens = mp.get("max_tokens", config.max_tokens)
            config.top_p = mp.get("top_p", config.top_p)
            config.frequency_penalty = mp.get(
                "frequency_penalty", config.frequency_penalty
            )
            config.presence_penalty = mp.get(
                "presence_penalty", config.presence_penalty
            )

        # Advanced
        if "advanced" in data:
            adv = data["advanced"]
            config.system_prompt_prefix = adv.get(
                "system_prompt_prefix", config.system_prompt_prefix
            )
            config.custom_instructions = adv.get(
                "custom_instructions", config.custom_instructions
            )
            config.context_awareness = adv.get(
                "context_awareness", config.context_awareness
            )
            config.memory_retention = adv.get(
                "memory_retention", config.memory_retention
            )

        return config


class ConfigManager:
    """Centralized configuration manager for all Vega modules"""

    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path("config")
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

    def get_config(self, module: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value(s) for a module"""
        config = self.load_config(module)

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
def get_config(module: str, key: str = None, default: Any = None) -> Any:
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

    # Apply updates
    for key, value in updates.items():
        if hasattr(current, key):
            setattr(current, key, value)

    # Save back to config
    set_config("llm", "behavior", current.to_dict())
    logger.info(f"Updated LLM behavior config: {updates}")
