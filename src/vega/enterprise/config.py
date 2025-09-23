"""
Enterprise Configuration and Initialization
==========================================

Configuration management for enterprise features including
multi-tenancy, billing integration, and advanced security.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class BillingProvider(Enum):
    """Supported billing providers"""

    STRIPE = "stripe"
    PADDLE = "paddle"
    CHARGEBEE = "chargebee"
    CUSTOM = "custom"


class SSOProvider(Enum):
    """Supported SSO providers"""

    SAML = "saml"
    OAUTH2 = "oauth2"
    OIDC = "oidc"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"


@dataclass
class RedisConfig:
    """Redis configuration for enterprise features"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    pool_size: int = 10
    retry_attempts: int = 3


@dataclass
class BillingConfig:
    """Billing integration configuration"""

    provider: BillingProvider = BillingProvider.STRIPE
    api_key: Optional[str] = None
    webhook_secret: Optional[str] = None
    currency: str = "USD"

    # Pricing configuration
    free_tier_limits: Dict[str, int] = field(
        default_factory=lambda: {
            "requests_per_month": 1000,
            "multimodal_requests": 10,
            "storage_mb": 100,
        }
    )

    usage_based_pricing: Dict[str, float] = field(
        default_factory=lambda: {
            "request_cost": 0.001,
            "multimodal_cost": 0.01,
            "storage_cost_per_gb": 0.10,
            "compute_cost_per_minute": 0.05,
        }
    )


@dataclass
class SSOConfig:
    """Single Sign-On configuration"""

    enabled: bool = False
    provider: Optional[SSOProvider] = None

    # SAML configuration
    saml_entity_id: Optional[str] = None
    saml_sso_url: Optional[str] = None
    saml_certificate: Optional[str] = None

    # OAuth2/OIDC configuration
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    authorization_url: Optional[str] = None
    token_url: Optional[str] = None
    userinfo_url: Optional[str] = None

    # Azure AD specific
    tenant_id: Optional[str] = None


@dataclass
class SecurityConfig:
    """Enterprise security configuration"""

    jwt_secret: str = "change-this-secret-key"
    jwt_expiry_hours: int = 24

    # API key configuration
    api_key_expiry_days: int = 365
    require_api_key: bool = True

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_redis_prefix: str = "vega:rate_limit"

    # Encryption
    encrypt_at_rest: bool = False
    encryption_key: Optional[str] = None

    # CORS and security headers
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    security_headers: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""

    enable_metrics: bool = True
    enable_logging: bool = True
    enable_tracing: bool = False

    # Prometheus metrics
    prometheus_enabled: bool = False
    prometheus_port: int = 9090

    # Jaeger tracing
    jaeger_endpoint: Optional[str] = None

    # Log configuration
    log_level: str = "INFO"
    log_format: str = "json"
    log_retention_days: int = 30


@dataclass
class MultiTenancyConfig:
    """Multi-tenancy configuration"""

    enabled: bool = False
    isolation_level: str = "namespace"  # namespace, database, schema
    default_org_tier: str = "basic"

    # Organization limits
    max_users_per_org: int = 100
    max_api_keys_per_org: int = 50

    # Feature flags per organization
    default_feature_flags: Dict[str, bool] = field(
        default_factory=lambda: {
            "multimodal_processing": True,
            "collaboration": True,
            "federated_learning": False,
            "advanced_analytics": False,
        }
    )


@dataclass
class EnterpriseConfig:
    """Main enterprise configuration"""

    # Core settings
    environment: str = "development"
    debug: bool = False

    # Component configurations
    redis: RedisConfig = field(default_factory=RedisConfig)
    billing: BillingConfig = field(default_factory=BillingConfig)
    sso: SSOConfig = field(default_factory=SSOConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    multitenancy: MultiTenancyConfig = field(default_factory=MultiTenancyConfig)

    # External service configuration
    external_services: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "EnterpriseConfig":
        """Load configuration from environment variables"""

        # Redis configuration
        redis_config = RedisConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            pool_size=int(os.getenv("REDIS_POOL_SIZE", "10")),
        )

        # Billing configuration
        billing_provider = os.getenv("BILLING_PROVIDER", "stripe")
        billing_config = BillingConfig(
            provider=BillingProvider(billing_provider),
            api_key=os.getenv("BILLING_API_KEY"),
            webhook_secret=os.getenv("BILLING_WEBHOOK_SECRET"),
            currency=os.getenv("BILLING_CURRENCY", "USD"),
        )

        # SSO configuration
        sso_enabled = os.getenv("SSO_ENABLED", "false").lower() == "true"
        sso_provider = os.getenv("SSO_PROVIDER")
        sso_config = SSOConfig(
            enabled=sso_enabled,
            provider=SSOProvider(sso_provider) if sso_provider else None,
            saml_entity_id=os.getenv("SAML_ENTITY_ID"),
            saml_sso_url=os.getenv("SAML_SSO_URL"),
            saml_certificate=os.getenv("SAML_CERTIFICATE"),
            client_id=os.getenv("SSO_CLIENT_ID"),
            client_secret=os.getenv("SSO_CLIENT_SECRET"),
            authorization_url=os.getenv("SSO_AUTHORIZATION_URL"),
            token_url=os.getenv("SSO_TOKEN_URL"),
            userinfo_url=os.getenv("SSO_USERINFO_URL"),
            tenant_id=os.getenv("AZURE_TENANT_ID"),
        )

        # Security configuration
        security_config = SecurityConfig(
            jwt_secret=os.getenv("JWT_SECRET", "change-this-secret-key"),
            jwt_expiry_hours=int(os.getenv("JWT_EXPIRY_HOURS", "24")),
            api_key_expiry_days=int(os.getenv("API_KEY_EXPIRY_DAYS", "365")),
            require_api_key=os.getenv("REQUIRE_API_KEY", "true").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower()
            == "true",
            encrypt_at_rest=os.getenv("ENCRYPT_AT_REST", "false").lower() == "true",
            encryption_key=os.getenv("ENCRYPTION_KEY"),
            cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000").split(","),
            security_headers=os.getenv("SECURITY_HEADERS", "true").lower() == "true",
        )

        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            enable_logging=os.getenv("ENABLE_LOGGING", "true").lower() == "true",
            enable_tracing=os.getenv("ENABLE_TRACING", "false").lower() == "true",
            prometheus_enabled=os.getenv("PROMETHEUS_ENABLED", "false").lower()
            == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            jaeger_endpoint=os.getenv("JAEGER_ENDPOINT"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            log_retention_days=int(os.getenv("LOG_RETENTION_DAYS", "30")),
        )

        # Multi-tenancy configuration
        multitenancy_config = MultiTenancyConfig(
            enabled=os.getenv("MULTITENANCY_ENABLED", "false").lower() == "true",
            isolation_level=os.getenv("ISOLATION_LEVEL", "namespace"),
            default_org_tier=os.getenv("DEFAULT_ORG_TIER", "basic"),
            max_users_per_org=int(os.getenv("MAX_USERS_PER_ORG", "100")),
            max_api_keys_per_org=int(os.getenv("MAX_API_KEYS_PER_ORG", "50")),
        )

        return cls(
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            redis=redis_config,
            billing=billing_config,
            sso=sso_config,
            security=security_config,
            monitoring=monitoring_config,
            multitenancy=multitenancy_config,
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate JWT secret
        if self.security.jwt_secret == "change-this-secret-key":
            errors.append("JWT secret must be changed from default value")

        # Validate billing configuration
        if self.billing.provider != BillingProvider.CUSTOM and not self.billing.api_key:
            errors.append(f"Billing API key required for {self.billing.provider.value}")

        # Validate SSO configuration
        if self.sso.enabled:
            if not self.sso.provider:
                errors.append("SSO provider must be specified when SSO is enabled")

            if self.sso.provider == SSOProvider.SAML:
                if not all([self.sso.saml_entity_id, self.sso.saml_sso_url]):
                    errors.append("SAML configuration incomplete")

            elif self.sso.provider in [SSOProvider.OAUTH2, SSOProvider.OIDC]:
                if not all([self.sso.client_id, self.sso.client_secret]):
                    errors.append("OAuth2/OIDC configuration incomplete")

        # Validate encryption
        if self.security.encrypt_at_rest and not self.security.encryption_key:
            errors.append("Encryption key required when encryption at rest is enabled")

        return errors


# Global enterprise configuration instance
_enterprise_config: Optional[EnterpriseConfig] = None


def get_enterprise_config() -> EnterpriseConfig:
    """Get the global enterprise configuration"""
    global _enterprise_config

    if _enterprise_config is None:
        _enterprise_config = EnterpriseConfig.from_env()

        # Validate configuration
        errors = _enterprise_config.validate()
        if errors:
            raise ValueError(f"Enterprise configuration errors: {errors}")

    return _enterprise_config


def set_enterprise_config(config: EnterpriseConfig):
    """Set the global enterprise configuration"""
    global _enterprise_config
    _enterprise_config = config
