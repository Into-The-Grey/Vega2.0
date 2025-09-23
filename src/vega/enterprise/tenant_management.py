"""
Advanced Multi-Tenant SaaS Architecture
======================================

Comprehensive tenant isolation, subscription management, and enterprise
SSO integration for Vega2.0 SaaS deployment.

Features:
- Advanced tenant isolation strategies (database per tenant, row-level security)
- Subscription lifecycle management with automated billing
- Enterprise SSO integration (SAML 2.0, OIDC, Azure AD)
- Advanced role-based access control with hierarchical permissions
- Tenant analytics and usage optimization
- Compliance and data governance
"""

import asyncio
import logging
import json
import uuid
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base

try:
    import redis.asyncio as redis
except ImportError:
    import redis
from cryptography.fernet import Fernet
import jwt

logger = logging.getLogger(__name__)

# Database Models Base
Base = declarative_base()


class TenantIsolationStrategy(Enum):
    """Tenant isolation strategies"""

    SHARED_DATABASE = "shared_database"
    DATABASE_PER_TENANT = "database_per_tenant"
    SCHEMA_PER_TENANT = "schema_per_tenant"
    ROW_LEVEL_SECURITY = "row_level_security"
    HYBRID = "hybrid"


class SubscriptionStatus(Enum):
    """Subscription status types"""

    TRIAL = "trial"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"
    UNPAID = "unpaid"


class TenantStatus(Enum):
    """Tenant status types"""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"
    MIGRATING = "migrating"


class SSOIntegrationType(Enum):
    """SSO integration types"""

    SAML2 = "saml2"
    OIDC = "oidc"
    OAUTH2 = "oauth2"
    AZURE_AD = "azure_ad"
    GOOGLE_WORKSPACE = "google_workspace"
    OKTA = "okta"
    LDAP = "ldap"


@dataclass
class TenantConfiguration:
    """Comprehensive tenant configuration"""

    tenant_id: str
    name: str
    domain: str
    isolation_strategy: TenantIsolationStrategy
    status: TenantStatus
    created_at: datetime
    updated_at: datetime

    # Database configuration
    database_config: Dict[str, Any] = field(default_factory=dict)
    schema_name: Optional[str] = None
    encryption_key: Optional[str] = None

    # Subscription details
    subscription_id: str = ""
    plan_id: str = ""
    billing_email: str = ""
    subscription_status: SubscriptionStatus = SubscriptionStatus.TRIAL
    trial_ends_at: Optional[datetime] = None
    next_billing_date: Optional[datetime] = None

    # SSO configuration
    sso_enabled: bool = False
    sso_type: Optional[SSOIntegrationType] = None
    sso_config: Dict[str, Any] = field(default_factory=dict)

    # Feature flags and limits
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    usage_limits: Dict[str, int] = field(default_factory=dict)
    storage_quota_gb: int = 100
    user_limit: int = 10

    # Compliance and governance
    data_residency: str = "us-east-1"
    encryption_at_rest: bool = True
    audit_logging: bool = True
    data_retention_days: int = 90
    gdpr_compliant: bool = True

    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubscriptionPlan:
    """Subscription plan configuration"""

    plan_id: str
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    limits: Dict[str, int]
    api_tier: str

    # Billing configuration
    billing_interval: str = "monthly"  # monthly, yearly, usage_based
    trial_days: int = 14
    setup_fee: float = 0.0

    # Feature access
    multimodal_enabled: bool = True
    collaboration_enabled: bool = True
    federated_learning_enabled: bool = False
    advanced_analytics: bool = False
    priority_support: bool = False
    custom_integrations: bool = False


class TenantManager:
    def get_tenant_session(self, tenant_config: TenantConfiguration) -> AsyncSession:
        """
        Return an AsyncSession for the given tenant, routing to the correct engine/schema.
        For schema-per-tenant and hybrid, call set_schema(session, schema_name) after session creation.
        Ensures only AsyncEngine is used.
        """
        from sqlalchemy.ext.asyncio import AsyncEngine

        strategy = tenant_config.isolation_strategy
        if strategy == TenantIsolationStrategy.DATABASE_PER_TENANT:
            engine = self.database_engines.get(tenant_config.tenant_id)
            if not engine:
                raise RuntimeError(f"No engine for tenant {tenant_config.tenant_id}")
            from sqlalchemy.ext.asyncio import AsyncEngine

            if not isinstance(engine, AsyncEngine):
                raise TypeError(
                    f"Engine for tenant {tenant_config.tenant_id} is not AsyncEngine"
                )
            return AsyncSession(engine)
        else:
            engine = self.database_engines.get("main")
            if not engine:
                raise RuntimeError("No main engine for shared DB")
            from sqlalchemy.ext.asyncio import AsyncEngine

            if not isinstance(engine, AsyncEngine):
                raise TypeError("Main engine is not AsyncEngine")
            return AsyncSession(engine)

    async def set_schema(self, session: AsyncSession, schema_name: str):
        """Set the schema for the session (Postgres only). Call after session creation."""
        await session.execute(sa.text(f"SET search_path TO {schema_name}"))

    def enforce_rls(self, query, tenant_id: str):
        """
        Inject tenant_id filter into ORM queries for RLS enforcement.
        Usage: query = tenant_manager.enforce_rls(query, tenant_id)
        """
        if tenant_id is not None:
            # Assume all models have a tenant_id column for RLS
            return query.filter_by(tenant_id=tenant_id)
        return query

    """Advanced tenant management system"""

    def __init__(self, config: Any):
        self.config = config
        self.redis_client = None
        self.database_engines: Dict[str, Any] = {}
        self.encryption_key = getattr(config, "encryption_key", Fernet.generate_key())
        self.fernet = Fernet(self.encryption_key)

        # Tenant cache
        self.tenant_cache: Dict[str, TenantConfiguration] = {}
        self.cache_ttl = 300  # 5 minutes

        # Default subscription plans
        self.subscription_plans = self._initialize_subscription_plans()

    def _initialize_subscription_plans(self) -> Dict[str, SubscriptionPlan]:
        """Initialize default subscription plans"""

        plans = {
            "starter": SubscriptionPlan(
                plan_id="starter",
                name="Starter",
                description="Perfect for individuals and small teams",
                price_monthly=29.0,
                price_yearly=290.0,
                features=[
                    "Basic AI processing",
                    "5 GB storage",
                    "Standard support",
                    "Basic analytics",
                ],
                limits={
                    "api_calls_monthly": 10000,
                    "storage_gb": 5,
                    "users": 3,
                    "projects": 5,
                },
                api_tier="basic",
                trial_days=14,
            ),
            "professional": SubscriptionPlan(
                plan_id="professional",
                name="Professional",
                description="Advanced features for growing businesses",
                price_monthly=99.0,
                price_yearly=990.0,
                features=[
                    "Advanced AI processing",
                    "Multi-modal support",
                    "50 GB storage",
                    "Priority support",
                    "Advanced analytics",
                    "Collaboration tools",
                ],
                limits={
                    "api_calls_monthly": 100000,
                    "storage_gb": 50,
                    "users": 15,
                    "projects": 25,
                },
                api_tier="professional",
                multimodal_enabled=True,
                collaboration_enabled=True,
                advanced_analytics=True,
                trial_days=14,
            ),
            "enterprise": SubscriptionPlan(
                plan_id="enterprise",
                name="Enterprise",
                description="Full-scale solution for large organizations",
                price_monthly=499.0,
                price_yearly=4990.0,
                features=[
                    "All AI capabilities",
                    "Federated learning",
                    "Unlimited storage",
                    "24/7 support",
                    "Custom integrations",
                    "SSO integration",
                    "Advanced security",
                    "Compliance features",
                ],
                limits={
                    "api_calls_monthly": 1000000,
                    "storage_gb": -1,  # Unlimited
                    "users": 100,
                    "projects": 100,
                },
                api_tier="enterprise",
                multimodal_enabled=True,
                collaboration_enabled=True,
                federated_learning_enabled=True,
                advanced_analytics=True,
                priority_support=True,
                custom_integrations=True,
                trial_days=30,
            ),
            "unlimited": SubscriptionPlan(
                plan_id="unlimited",
                name="Unlimited",
                description="Custom enterprise solution with unlimited access",
                price_monthly=1999.0,
                price_yearly=19990.0,
                features=[
                    "Unlimited everything",
                    "Dedicated infrastructure",
                    "Custom development",
                    "White-label options",
                    "On-premise deployment",
                ],
                limits={
                    "api_calls_monthly": -1,  # Unlimited
                    "storage_gb": -1,  # Unlimited
                    "users": -1,  # Unlimited
                    "projects": -1,  # Unlimited
                },
                api_tier="unlimited",
                multimodal_enabled=True,
                collaboration_enabled=True,
                federated_learning_enabled=True,
                advanced_analytics=True,
                priority_support=True,
                custom_integrations=True,
                trial_days=60,
            ),
        }

        return plans

    async def initialize(self):
        """Initialize tenant manager with Redis and database connections"""

        try:
            from vega.enterprise.redis_factory import create_redis_client
        except ImportError:
            from .redis_factory import create_redis_client

        # Use cluster-aware Redis client
        self.redis_client = create_redis_client(self.config)
        # Test Redis connection
        try:
            await self.redis_client.ping()
            logger.info("Redis connection established for tenant management")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        # Initialize database engines for different isolation strategies
        await self._initialize_database_engines()

    async def _initialize_database_engines(self):
        """Initialize database engines for different tenants"""

        # Main application database
        main_db_url = self.config.get(
            "database_url", "sqlite+aiosqlite:///./tenants.db"
        )
        self.database_engines["main"] = create_async_engine(main_db_url)

        # Initialize tenant-specific databases if using database-per-tenant strategy
        # This would be expanded based on actual tenant configurations

    async def create_tenant(
        self,
        name: str,
        domain: str,
        admin_email: str,
        plan_id: str = "starter",
        isolation_strategy: TenantIsolationStrategy = TenantIsolationStrategy.ROW_LEVEL_SECURITY,
        custom_config: Optional[Dict[str, Any]] = None,
    ) -> TenantConfiguration:
        """Create a new tenant with full configuration"""

        tenant_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Get subscription plan
        plan = self.subscription_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Invalid plan ID: {plan_id}")

        # Create tenant configuration
        tenant_config = TenantConfiguration(
            tenant_id=tenant_id,
            name=name,
            domain=domain,
            isolation_strategy=isolation_strategy,
            status=TenantStatus.PENDING,
            created_at=now,
            updated_at=now,
            subscription_id=str(uuid.uuid4()),
            plan_id=plan_id,
            billing_email=admin_email,
            subscription_status=SubscriptionStatus.TRIAL,
            trial_ends_at=now + timedelta(days=plan.trial_days),
            usage_limits=plan.limits.copy(),
            feature_flags={
                "multimodal_enabled": plan.multimodal_enabled,
                "collaboration_enabled": plan.collaboration_enabled,
                "federated_learning_enabled": plan.federated_learning_enabled,
                "advanced_analytics": plan.advanced_analytics,
                "priority_support": plan.priority_support,
                "custom_integrations": plan.custom_integrations,
            },
        )

        # Apply custom configuration if provided
        if custom_config:
            tenant_config.custom_metadata.update(custom_config)

        # Set up tenant isolation
        await self._setup_tenant_isolation(tenant_config)

        # Store tenant configuration
        await self._store_tenant_configuration(tenant_config)

        # Initialize tenant resources
        await self._initialize_tenant_resources(tenant_config)

        # Update status to active
        tenant_config.status = TenantStatus.ACTIVE
        tenant_config.updated_at = datetime.now(timezone.utc)
        await self._store_tenant_configuration(tenant_config)

        # Cache the tenant
        self.tenant_cache[tenant_id] = tenant_config

        logger.info(f"Created tenant: {tenant_id} ({name}) with plan: {plan_id}")

        return tenant_config

    async def _setup_tenant_isolation(self, tenant_config: TenantConfiguration):
        """Set up tenant isolation based on strategy"""

        strategy = tenant_config.isolation_strategy

        if strategy == TenantIsolationStrategy.DATABASE_PER_TENANT:
            await self._setup_database_per_tenant(tenant_config)
        elif strategy == TenantIsolationStrategy.SCHEMA_PER_TENANT:
            await self._setup_schema_per_tenant(tenant_config)
        elif strategy == TenantIsolationStrategy.ROW_LEVEL_SECURITY:
            await self._setup_row_level_security(tenant_config)
        elif strategy == TenantIsolationStrategy.HYBRID:
            await self._setup_hybrid_isolation(tenant_config)

        # Generate encryption key for tenant data
        tenant_config.encryption_key = Fernet.generate_key().decode()

    async def _setup_database_per_tenant(self, tenant_config: TenantConfiguration):
        """Set up dedicated database for tenant, including DB creation and migrations."""
        import sqlalchemy

        tenant_id = tenant_config.tenant_id
        db_name = f"tenant_{tenant_id.replace('-', '_')}"
        base_url = self.config.get("database_url", "sqlite+aiosqlite:///./tenants.db")
        if "sqlite" in base_url:
            import os

            os.makedirs("./tenant_dbs", exist_ok=True)
            tenant_db_url = f"sqlite+aiosqlite:///./tenant_dbs/{db_name}.db"
        else:
            # For PostgreSQL/MySQL, create database if not exists
            from sqlalchemy.engine.url import make_url

            url_obj = make_url(base_url)
            admin_url = str(
                url_obj.set(
                    database=(
                        "postgres"
                        if url_obj.drivername.startswith("postgres")
                        else url_obj.database
                    )
                )
            )
            engine = sqlalchemy.create_engine(admin_url)
            with engine.connect() as conn:
                if url_obj.drivername.startswith("postgres"):
                    conn.execute(
                        sqlalchemy.text(
                            f"CREATE DATABASE {db_name} ENCODING 'UTF8' TEMPLATE template1"
                        )
                    )
                elif url_obj.drivername.startswith("mysql"):
                    conn.execute(
                        sqlalchemy.text(f"CREATE DATABASE IF NOT EXISTS {db_name}")
                    )
            tenant_db_url = str(url_obj.set(database=db_name))
        tenant_config.database_config = {
            "url": tenant_db_url,
            "isolation": "database_per_tenant",
        }
        # Create async engine for this tenant
        self.database_engines[tenant_config.tenant_id] = create_async_engine(
            tenant_db_url
        )
        logger.info(
            f"Provisioned database for tenant {tenant_config.tenant_id}: {tenant_db_url}"
        )

    async def _setup_schema_per_tenant(self, tenant_config: TenantConfiguration):
        """Set up dedicated schema for tenant, including schema creation."""
        import sqlalchemy

        tenant_id = tenant_config.tenant_id
        schema_name = f"tenant_{tenant_id.replace('-', '_')}"
        base_url = self.config.get("database_url", "sqlite+aiosqlite:///./tenants.db")
        # Use a sync engine only for DDL, do not store it in self.database_engines
        engine = sqlalchemy.create_engine(base_url)
        with engine.connect() as conn:
            try:
                conn.execute(
                    sqlalchemy.text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
                )
                logger.info(f"Created schema for tenant {tenant_id}: {schema_name}")
            except Exception as e:
                logger.error(f"Failed to create schema for tenant {tenant_id}: {e}")
                raise
        tenant_config.schema_name = schema_name
        tenant_config.database_config = {
            "schema": schema_name,
            "isolation": "schema_per_tenant",
        }

    async def _setup_row_level_security(self, tenant_config: TenantConfiguration):
        """Set up row-level security for tenant"""

        tenant_config.database_config = {
            "tenant_id_column": "tenant_id",
            "isolation": "row_level_security",
        }

    async def _setup_hybrid_isolation(self, tenant_config: TenantConfiguration):
        """Set up hybrid isolation strategy"""

        # Use schema-per-tenant for data isolation
        await self._setup_schema_per_tenant(tenant_config)

        # Add row-level security as additional layer
        tenant_config.database_config["row_level_security"] = True
        tenant_config.database_config["isolation"] = "hybrid"

    async def _store_tenant_configuration(self, tenant_config: TenantConfiguration):
        """Store tenant configuration in Redis and database"""

        tenant_id = tenant_config.tenant_id

        # Store in Redis for fast access
        config_json = json.dumps(asdict(tenant_config), default=str)
        encrypted_config = self.fernet.encrypt(config_json.encode()).decode()

        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        await self.redis_client.setex(
            f"tenant:config:{tenant_id}", self.cache_ttl, encrypted_config
        )

        # Store in database for persistence
        # This would use the main database to store tenant metadata

    async def _initialize_tenant_resources(self, tenant_config: TenantConfiguration):
        """Initialize tenant-specific resources"""

        tenant_id = tenant_config.tenant_id

        # Create tenant-specific Redis keys
        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        await self.redis_client.setex(
            f"tenant:status:{tenant_id}", 86400, tenant_config.status.value  # 24 hours
        )

        # Initialize usage tracking
        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        await self.redis_client.setex(
            f"tenant:usage:{tenant_id}",
            86400,
            json.dumps(
                {
                    "api_calls": 0,
                    "storage_used": 0,
                    "users_active": 0,
                    "last_activity": datetime.now(timezone.utc).isoformat(),
                }
            ),
        )

        # Initialize feature flags
        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        await self.redis_client.setex(
            f"tenant:features:{tenant_id}",
            86400,
            json.dumps(tenant_config.feature_flags),
        )

    async def get_tenant_configuration(
        self, tenant_id: str
    ) -> Optional[TenantConfiguration]:
        """Get tenant configuration with caching"""

        # Check cache first
        if tenant_id in self.tenant_cache:
            return self.tenant_cache[tenant_id]

        # Try Redis
        try:
            if not self.redis_client:
                raise RuntimeError(
                    "Redis client is not initialized. Call initialize() first."
                )
            encrypted_config = await self.redis_client.get(f"tenant:config:{tenant_id}")
            if encrypted_config:
                config_json = self.fernet.decrypt(encrypted_config.encode()).decode()
                config_dict = json.loads(config_json)

                # Convert string dates back to datetime objects
                for date_field in [
                    "created_at",
                    "updated_at",
                    "trial_ends_at",
                    "next_billing_date",
                ]:
                    if config_dict.get(date_field):
                        config_dict[date_field] = datetime.fromisoformat(
                            config_dict[date_field]
                        )

                # Convert enums
                config_dict["isolation_strategy"] = TenantIsolationStrategy(
                    config_dict["isolation_strategy"]
                )
                config_dict["status"] = TenantStatus(config_dict["status"])
                config_dict["subscription_status"] = SubscriptionStatus(
                    config_dict["subscription_status"]
                )

                if config_dict.get("sso_type"):
                    config_dict["sso_type"] = SSOIntegrationType(
                        config_dict["sso_type"]
                    )

                tenant_config = TenantConfiguration(**config_dict)
                self.tenant_cache[tenant_id] = tenant_config
                return tenant_config

        except Exception as e:
            logger.error(f"Error retrieving tenant config from Redis: {e}")

        # Fall back to database lookup
        # This would query the main database for tenant information

        return None

    async def update_subscription(
        self, tenant_id: str, new_plan_id: str, billing_period: str = "monthly"
    ) -> bool:
        """Update tenant subscription plan"""

        tenant_config = await self.get_tenant_configuration(tenant_id)
        if not tenant_config:
            return False

        new_plan = self.subscription_plans.get(new_plan_id)
        if not new_plan:
            return False

        # Update subscription details
        tenant_config.plan_id = new_plan_id
        tenant_config.usage_limits = new_plan.limits.copy()
        tenant_config.feature_flags.update(
            {
                "multimodal_enabled": new_plan.multimodal_enabled,
                "collaboration_enabled": new_plan.collaboration_enabled,
                "federated_learning_enabled": new_plan.federated_learning_enabled,
                "advanced_analytics": new_plan.advanced_analytics,
                "priority_support": new_plan.priority_support,
                "custom_integrations": new_plan.custom_integrations,
            }
        )

        # Calculate next billing date
        if billing_period == "yearly":
            tenant_config.next_billing_date = datetime.now(timezone.utc) + timedelta(
                days=365
            )
        else:
            tenant_config.next_billing_date = datetime.now(timezone.utc) + timedelta(
                days=30
            )

        tenant_config.updated_at = datetime.now(timezone.utc)

        # Store updated configuration
        await self._store_tenant_configuration(tenant_config)

        # Update cache
        self.tenant_cache[tenant_id] = tenant_config

        logger.info(
            f"Updated subscription for tenant {tenant_id} to plan {new_plan_id}"
        )

        return True

    async def setup_sso_integration(
        self, tenant_id: str, sso_type: SSOIntegrationType, sso_config: Dict[str, Any]
    ) -> bool:
        """Set up SSO integration for tenant"""

        tenant_config = await self.get_tenant_configuration(tenant_id)
        if not tenant_config:
            return False

        # Encrypt SSO configuration
        sso_config_json = json.dumps(sso_config)
        encrypted_sso_config = self.fernet.encrypt(sso_config_json.encode()).decode()

        # Update tenant configuration
        tenant_config.sso_enabled = True
        tenant_config.sso_type = sso_type
        tenant_config.sso_config = {"encrypted": encrypted_sso_config}
        tenant_config.updated_at = datetime.now(timezone.utc)

        # Store updated configuration
        await self._store_tenant_configuration(tenant_config)

        # Update cache
        self.tenant_cache[tenant_id] = tenant_config

        logger.info(f"Set up {sso_type.value} SSO for tenant {tenant_id}")

        return True

    async def get_tenant_usage_analytics(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive usage analytics for tenant"""

        # Get current usage from Redis
        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        usage_data = await self.redis_client.get(f"tenant:usage:{tenant_id}")
        if not usage_data:
            return {}

        usage = json.loads(usage_data)

        # Get tenant configuration for limits
        tenant_config = await self.get_tenant_configuration(tenant_id)
        if not tenant_config:
            return usage

        # Calculate usage percentages
        limits = tenant_config.usage_limits
        analytics = {
            "current_usage": usage,
            "limits": limits,
            "usage_percentages": {},
            "subscription_info": {
                "plan_id": tenant_config.plan_id,
                "status": tenant_config.subscription_status.value,
                "trial_ends_at": (
                    tenant_config.trial_ends_at.isoformat()
                    if tenant_config.trial_ends_at
                    else None
                ),
                "next_billing_date": (
                    tenant_config.next_billing_date.isoformat()
                    if tenant_config.next_billing_date
                    else None
                ),
            },
        }

        # Calculate percentages
        for metric, current_value in usage.items():
            if metric in limits and limits[metric] > 0:
                percentage = (current_value / limits[metric]) * 100
                analytics["usage_percentages"][metric] = min(percentage, 100)

        return analytics

    async def suspend_tenant(
        self, tenant_id: str, reason: str = "billing_issue"
    ) -> bool:
        """Suspend tenant access"""

        tenant_config = await self.get_tenant_configuration(tenant_id)
        if not tenant_config:
            return False

        tenant_config.status = TenantStatus.SUSPENDED
        tenant_config.updated_at = datetime.now(timezone.utc)
        tenant_config.custom_metadata["suspension_reason"] = reason
        tenant_config.custom_metadata["suspended_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        # Store updated configuration
        await self._store_tenant_configuration(tenant_config)

        # Update cache
        self.tenant_cache[tenant_id] = tenant_config

        # Update Redis status
        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        await self.redis_client.setex(
            f"tenant:status:{tenant_id}", 86400, TenantStatus.SUSPENDED.value
        )

        logger.warning(f"Suspended tenant {tenant_id}: {reason}")

        return True

    async def reactivate_tenant(self, tenant_id: str) -> bool:
        """Reactivate suspended tenant"""

        tenant_config = await self.get_tenant_configuration(tenant_id)
        if not tenant_config:
            return False

        tenant_config.status = TenantStatus.ACTIVE
        tenant_config.updated_at = datetime.now(timezone.utc)

        # Remove suspension metadata
        if "suspension_reason" in tenant_config.custom_metadata:
            del tenant_config.custom_metadata["suspension_reason"]
        if "suspended_at" in tenant_config.custom_metadata:
            del tenant_config.custom_metadata["suspended_at"]

        tenant_config.custom_metadata["reactivated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        # Store updated configuration
        await self._store_tenant_configuration(tenant_config)

        # Update cache
        self.tenant_cache[tenant_id] = tenant_config

        # Update Redis status
        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        await self.redis_client.setex(
            f"tenant:status:{tenant_id}", 86400, TenantStatus.ACTIVE.value
        )

        logger.info(f"Reactivated tenant {tenant_id}")

        return True

    async def list_tenants(
        self,
        status: Optional[TenantStatus] = None,
        plan_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[TenantConfiguration]:
        """List tenants with filtering"""

        # This would typically query the database
        # For now, return cached tenants
        tenants = list(self.tenant_cache.values())

        # Apply filters
        if status:
            tenants = [t for t in tenants if t.status == status]

        if plan_id:
            tenants = [t for t in tenants if t.plan_id == plan_id]

        # Apply pagination
        return tenants[offset : offset + limit]

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide tenant metrics"""

        tenants = list(self.tenant_cache.values())

        metrics = {
            "total_tenants": len(tenants),
            "tenants_by_status": {},
            "tenants_by_plan": {},
            "total_revenue_monthly": 0,
            "total_revenue_yearly": 0,
            "active_trials": 0,
            "expiring_trials": 0,  # Trials expiring in next 7 days
        }

        now = datetime.now(timezone.utc)
        week_from_now = now + timedelta(days=7)

        for tenant in tenants:
            # Count by status
            status = tenant.status.value
            metrics["tenants_by_status"][status] = (
                metrics["tenants_by_status"].get(status, 0) + 1
            )

            # Count by plan
            plan = tenant.plan_id
            metrics["tenants_by_plan"][plan] = (
                metrics["tenants_by_plan"].get(plan, 0) + 1
            )

            # Calculate revenue
            if tenant.subscription_status == SubscriptionStatus.ACTIVE:
                plan_config = self.subscription_plans.get(tenant.plan_id)
                if plan_config:
                    metrics["total_revenue_monthly"] += plan_config.price_monthly
                    metrics["total_revenue_yearly"] += plan_config.price_yearly

            # Count trials
            if tenant.subscription_status == SubscriptionStatus.TRIAL:
                metrics["active_trials"] += 1
                if tenant.trial_ends_at and tenant.trial_ends_at <= week_from_now:
                    metrics["expiring_trials"] += 1

        return metrics


class TenantMiddleware:
    """Middleware for tenant context management"""

    def __init__(self, tenant_manager: TenantManager):
        self.tenant_manager = tenant_manager

    async def __call__(self, request, call_next):
        """Process request with tenant context"""

        # Extract tenant ID from request (domain, header, or token)
        tenant_id = await self.extract_tenant_id(request)

        if tenant_id:
            # Get tenant configuration
            tenant_config = await self.tenant_manager.get_tenant_configuration(
                tenant_id
            )

            if tenant_config and tenant_config.status == TenantStatus.ACTIVE:
                # Add tenant context to request
                request.state.tenant_id = tenant_id
                request.state.tenant_config = tenant_config
            else:
                # Tenant not found or not active
                from fastapi import HTTPException

                raise HTTPException(status_code=403, detail="Tenant access denied")

        response = await call_next(request)
        return response

    async def extract_tenant_id(self, request) -> Optional[str]:
        """Extract tenant ID from request"""

        # Method 1: From subdomain
        host = request.headers.get("host", "")
        if "." in host:
            subdomain = host.split(".")[0]
            # Check if subdomain maps to a tenant
            tenant_config = await self.tenant_manager.get_tenant_configuration(
                subdomain
            )
            if tenant_config:
                return subdomain

        # Method 2: From custom header
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            return tenant_id

        # Method 3: From JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                # Decode JWT to extract tenant ID
                payload = jwt.decode(token, options={"verify_signature": False})
                return payload.get("tenant_id")
            except Exception:
                pass

        return None
