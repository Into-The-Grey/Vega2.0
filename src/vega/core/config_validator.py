"""
config_validator.py - Configuration Validation for Vega2.0

Validates configuration on startup to fail fast with clear error messages.
Checks required settings, validates formats, and warns about missing optional configs.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation"""

    field: str
    status: str  # "valid", "invalid", "warning", "missing"
    message: str
    severity: str  # "critical", "warning", "info"


class ConfigValidator:
    """Validates Vega configuration for completeness and correctness"""

    def __init__(self):
        self.results: List[ValidationResult] = []

    def validate_required_string(self, field: str, value: Any, name: str) -> bool:
        """Validate a required string field"""
        if not value or not isinstance(value, str) or not value.strip():
            self.results.append(
                ValidationResult(
                    field=field,
                    status="invalid",
                    message=f"{name} is required but not set or empty",
                    severity="critical",
                )
            )
            return False

        self.results.append(
            ValidationResult(
                field=field,
                status="valid",
                message=f"{name} is properly configured",
                severity="info",
            )
        )
        return True

    def validate_port(self, field: str, value: int, name: str) -> bool:
        """Validate port number"""
        if not isinstance(value, int) or not (1 <= value <= 65535):
            self.results.append(
                ValidationResult(
                    field=field,
                    status="invalid",
                    message=f"{name} must be between 1 and 65535, got: {value}",
                    severity="critical",
                )
            )
            return False

        if value < 1024:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="warning",
                    message=f"{name} is {value} (privileged port, may require sudo)",
                    severity="warning",
                )
            )

        self.results.append(
            ValidationResult(
                field=field,
                status="valid",
                message=f"{name} is valid: {value}",
                severity="info",
            )
        )
        return True

    def validate_host(self, field: str, value: str, name: str) -> bool:
        """Validate host binding"""
        if not value:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="invalid",
                    message=f"{name} is required",
                    severity="critical",
                )
            )
            return False

        # Warn about binding to 0.0.0.0 (security)
        if value == "0.0.0.0":
            self.results.append(
                ValidationResult(
                    field=field,
                    status="warning",
                    message=f"{name} is 0.0.0.0 - server will accept connections from any IP (security risk)",
                    severity="warning",
                )
            )

        # Recommend 127.0.0.1 for localhost
        if value == "127.0.0.1":
            self.results.append(
                ValidationResult(
                    field=field,
                    status="valid",
                    message=f"{name} is 127.0.0.1 (localhost-only, secure)",
                    severity="info",
                )
            )

        return True

    def validate_llm_backend(self, field: str, value: str, name: str) -> bool:
        """Validate LLM backend choice"""
        valid_backends = {"ollama", "hf"}

        if value not in valid_backends:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="invalid",
                    message=f"{name} must be one of {valid_backends}, got: {value}",
                    severity="critical",
                )
            )
            return False

        self.results.append(
            ValidationResult(
                field=field,
                status="valid",
                message=f"{name} is valid: {value}",
                severity="info",
            )
        )
        return True

    def validate_timeout(self, field: str, value: float, name: str) -> bool:
        """Validate timeout values"""
        if not isinstance(value, (int, float)) or value <= 0:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="invalid",
                    message=f"{name} must be positive number, got: {value}",
                    severity="critical",
                )
            )
            return False

        if value < 5:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="warning",
                    message=f"{name} is very low ({value}s) - may cause premature timeouts",
                    severity="warning",
                )
            )

        if value > 300:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="warning",
                    message=f"{name} is very high ({value}s) - may cause long waits",
                    severity="warning",
                )
            )

        return True

    def validate_optional_url(self, field: str, value: str | None, name: str) -> bool:
        """Validate optional URL field"""
        if not value:
            self.results.append(
                ValidationResult(
                    field=field,
                    status="missing",
                    message=f"{name} not configured (optional)",
                    severity="info",
                )
            )
            return True

        # Basic URL validation
        if not value.startswith(("http://", "https://")):
            self.results.append(
                ValidationResult(
                    field=field,
                    status="invalid",
                    message=f"{name} must start with http:// or https://, got: {value}",
                    severity="warning",
                )
            )
            return False

        self.results.append(
            ValidationResult(
                field=field,
                status="valid",
                message=f"{name} is configured",
                severity="info",
            )
        )
        return True

    def validate_config(self, config) -> Tuple[bool, Dict[str, Any]]:
        """Validate entire configuration object

        Returns:
            (is_valid, summary_dict)
        """
        self.results = []

        # Required fields
        self.validate_required_string("API_KEY", config.api_key, "API_KEY")
        self.validate_required_string("MODEL_NAME", config.model_name, "MODEL_NAME")
        self.validate_host("HOST", config.host, "HOST")
        self.validate_port("PORT", config.port, "PORT")
        self.validate_llm_backend("LLM_BACKEND", config.llm_backend, "LLM_BACKEND")

        # Timeouts and limits
        self.validate_timeout(
            "LLM_TIMEOUT_SEC", config.llm_timeout_sec, "LLM_TIMEOUT_SEC"
        )

        # Optional integrations
        self.validate_optional_url(
            "SLACK_WEBHOOK_URL", config.slack_webhook_url, "SLACK_WEBHOOK_URL"
        )

        if config.hass_enabled:
            self.validate_optional_url("HASS_URL", config.hass_url, "HASS_URL")
            if not config.hass_token:
                self.results.append(
                    ValidationResult(
                        field="HASS_TOKEN",
                        status="warning",
                        message="Home Assistant is enabled but HASS_TOKEN is not set",
                        severity="warning",
                    )
                )

        # Redis configuration
        if config.redis_mode == "cluster" and not config.redis_cluster_nodes:
            self.results.append(
                ValidationResult(
                    field="REDIS_CLUSTER_NODES",
                    status="warning",
                    message="REDIS_MODE is 'cluster' but REDIS_CLUSTER_NODES is empty",
                    severity="warning",
                )
            )

        # Security checks
        if len(config.api_key) < 16:
            self.results.append(
                ValidationResult(
                    field="API_KEY",
                    status="warning",
                    message=f"API_KEY is short ({len(config.api_key)} chars) - consider using longer keys for security",
                    severity="warning",
                )
            )

        # Log level validation
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.log_level not in valid_log_levels:
            self.results.append(
                ValidationResult(
                    field="LOG_LEVEL",
                    status="warning",
                    message=f"LOG_LEVEL should be one of {valid_log_levels}, got: {config.log_level}",
                    severity="warning",
                )
            )

        # Generation parameters
        if not (0.0 <= config.temperature <= 2.0):
            self.results.append(
                ValidationResult(
                    field="GEN_TEMPERATURE",
                    status="warning",
                    message=f"Temperature should be 0.0-2.0, got: {config.temperature}",
                    severity="warning",
                )
            )

        if not (0.0 <= config.top_p <= 1.0):
            self.results.append(
                ValidationResult(
                    field="GEN_TOP_P",
                    status="warning",
                    message=f"top_p should be 0.0-1.0, got: {config.top_p}",
                    severity="warning",
                )
            )

        # Build summary
        critical_errors = [r for r in self.results if r.severity == "critical"]
        warnings = [r for r in self.results if r.severity == "warning"]

        is_valid = len(critical_errors) == 0

        summary = {
            "valid": is_valid,
            "total_checks": len(self.results),
            "critical_errors": len(critical_errors),
            "warnings": len(warnings),
            "results": [
                {
                    "field": r.field,
                    "status": r.status,
                    "message": r.message,
                    "severity": r.severity,
                }
                for r in self.results
            ],
        }

        return is_valid, summary

    def print_validation_results(self) -> None:
        """Print validation results to console"""
        critical = [r for r in self.results if r.severity == "critical"]
        warnings = [r for r in self.results if r.severity == "warning"]

        if critical:
            print("\n❌ CRITICAL CONFIGURATION ERRORS:")
            for r in critical:
                print(f"  • {r.field}: {r.message}")

        if warnings:
            print("\n⚠️  CONFIGURATION WARNINGS:")
            for r in warnings:
                print(f"  • {r.field}: {r.message}")

        if not critical and not warnings:
            print("\n✅ Configuration validation passed!")


def validate_startup_config() -> bool:
    """Validate configuration on application startup

    Returns:
        True if valid, False if critical errors found
    """
    try:
        from .config import get_config

        config = get_config()
        validator = ConfigValidator()
        is_valid, summary = validator.validate_config(config)

        # Log results
        if not is_valid:
            logger.error("Configuration validation failed!")
            validator.print_validation_results()
        else:
            logger.info("Configuration validation passed")
            if summary["warnings"] > 0:
                validator.print_validation_results()

        return is_valid
    except Exception as e:
        logger.error(f"Failed to validate configuration: {e}")
        return False
