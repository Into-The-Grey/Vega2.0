"""
Vega2.0 Enterprise Module
========================

Comprehensive enterprise feature set including:
- Multi-tenant SaaS architecture
- Advanced authentication and authorization
- Billing and subscription management
- Enterprise SSO integration
- Advanced RBAC with hierarchical permissions
- API management and monetization
- Real-time monitoring and analytics

This module provides production-ready enterprise features suitable for
B2B SaaS deployments with complex organizational structures.
"""

from .config import (
    EnterpriseConfig,
    RedisConfig,
    BillingConfig,
    SSOConfig,
    SecurityConfig,
    MonitoringConfig,
    MultiTenancyConfig,
    BillingProvider,
    SSOProvider,
    get_enterprise_config,
    set_enterprise_config,
)

from .api_management import (
    EnterpriseAuthManager,
    RateLimitManager,
    UsageTracker,
    APITier,
    UserRole,
    APIScope,
    RateLimitConfig,
    APIKey,
    User,
    Organization,
    UsageRecord,
    get_current_api_key,
    require_scopes,
    rate_limit_check,
    UsageTrackingMiddleware,
    require_api_key,
    require_enterprise_auth,
    APIKeyManager,
)

from .tenant_management import (
    TenantManager,
    TenantConfiguration,
    TenantIsolationStrategy,
    SubscriptionPlan,
    TenantMetrics,
    TenantUser,
    TenantInvitation,
)

from .billing_integration import (
    BillingManager,
    BillingCustomer,
    Subscription,
    Invoice,
    PaymentMethod,
    UsageMeter,
    BillingProvider as BillingIntegrationProvider,
    SubscriptionStatus,
    WebhookEvent,
)

from .sso_integration import (
    SSOManager,
    SSOConfiguration,
    SSOUser,
    SSOProvider as SSOIntegrationProvider,
    SAMLConfig,
    OIDCConfig,
    OAuth2Config,
    AttributeMapping,
    ProvisioningConfig,
)

from .advanced_rbac import (
    RBACManager,
    Permission,
    Role,
    UserRoleAssignment,
    AccessPolicy,
    AccessRequest,
    AccessLogEntry,
    PermissionType,
    ResourceType,
    AccessDecision,
    require_permission,
)

from .app import EnterpriseApp, create_enterprise_app, run_enterprise_server

__all__ = [
    # Configuration
    "EnterpriseConfig",
    "RedisConfig",
    "BillingConfig",
    "SSOConfig",
    "SecurityConfig",
    "MonitoringConfig",
    "MultiTenancyConfig",
    "BillingProvider",
    "SSOProvider",
    "get_enterprise_config",
    "set_enterprise_config",
    # API Management
    "EnterpriseAuthManager",
    "RateLimitManager",
    "UsageTracker",
    "APITier",
    "UserRole",
    "APIScope",
    "RateLimitConfig",
    "APIKey",
    "User",
    "Organization",
    "UsageRecord",
    "get_current_api_key",
    "require_scopes",
    "rate_limit_check",
    "UsageTrackingMiddleware",
    # Application
    "EnterpriseApp",
    "create_enterprise_app",
    "run_enterprise_server",
]
