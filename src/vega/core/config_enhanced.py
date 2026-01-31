"""
config_enhanced.py - Enhanced configuration with caching and validation

Improvements over base config.py:
- Singleton caching to avoid repeated environment parsing
- Pydantic-style validation on first load
- Thread-safe access
- Optional hot-reload capability
"""

from __future__ import annotations

import os
import threading
from typing import Optional
from functools import lru_cache

from .config import Config, ConfigError


_config_lock = threading.Lock()
_config_instance: Optional[Config] = None


def get_cached_config() -> Config:
    """
    Get configuration with singleton caching.

    Thread-safe and parses environment only once.
    Call reset_config() to force reload.
    """
    global _config_instance

    if _config_instance is not None:
        return _config_instance

    with _config_lock:
        # Double-check after acquiring lock
        if _config_instance is not None:
            return _config_instance

        from .config import get_config

        _config_instance = get_config()
        return _config_instance


def reset_config() -> None:
    """Reset cached config to force reload on next access."""
    global _config_instance
    with _config_lock:
        _config_instance = None


def validate_config_completeness(config: Config) -> list[str]:
    """
    Validate configuration completeness and return list of warnings.

    Returns:
        List of warning messages for missing optional configs
    """
    warnings = []

    # Check optional integrations
    if not config.slack_webhook_url:
        warnings.append("SLACK_WEBHOOK_URL not set - Slack notifications disabled")

    if not config.hass_enabled:
        warnings.append("HASS_ENABLED not set - Home Assistant integration disabled")
    elif config.hass_enabled:
        if not config.hass_url:
            warnings.append("HASS_URL required when HASS_ENABLED=true")
        if not config.hass_token:
            warnings.append("HASS_TOKEN required when HASS_ENABLED=true")

    # Check security settings
    if config.api_key == "changeme" or len(config.api_key) < 16:
        warnings.append("API_KEY appears weak - use a strong random key in production")

    # Check performance tuning
    if config.llm_timeout_sec < 30:
        warnings.append(f"LLM_TIMEOUT_SEC={config.llm_timeout_sec} may be too low for complex queries")

    if config.cache_ttl_seconds == 0:
        warnings.append("CACHE_TTL_SECONDS=0 disables caching - may impact performance")

    return warnings


# Convenience alias for migration
get_config = get_cached_config
