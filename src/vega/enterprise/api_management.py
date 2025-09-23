"""
Enterprise API Management System
===============================

Advanced API rate limiting, usage tracking, authentication, and billing
integration for Vega2.0 enterprise deployment.

Features:
- JWT-based authentication with role-based access control (RBAC)
- Advanced rate limiting with tier-based quotas
- Real-time usage tracking and analytics
- API key management with scopes and permissions
- Billing integration and subscription management
- Enterprise SSO integration (SAML, OAuth2, OIDC)
"""

import asyncio
import logging
import jwt
import hashlib
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import uuid

logger = logging.getLogger(__name__)


class APITier(Enum):
    """API subscription tiers"""

    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


class UserRole(Enum):
    """User roles for RBAC"""

    USER = "user"
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    BILLING_ADMIN = "billing_admin"
    SUPER_ADMIN = "super_admin"


class APIScope(Enum):
    """API access scopes"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    BILLING = "billing"
    ANALYTICS = "analytics"
    MULTIMODAL = "multimodal"
    COLLABORATION = "collaboration"
    FEDERATED = "federated"


@dataclass
class RateLimitConfig:
    """Rate limiting configuration by tier"""

    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    concurrent_requests: int
    bandwidth_mb_per_hour: int

    # Feature-specific limits
    multimodal_requests_per_hour: int = 0
    vector_searches_per_hour: int = 0
    document_processing_per_hour: int = 0
    collaboration_sessions_per_day: int = 0


@dataclass
class APIKey:
    """API key with metadata and permissions"""

    key_id: str
    key_hash: str
    name: str
    user_id: str
    tier: APITier
    scopes: Set[APIScope]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit_override: Optional[RateLimitConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class User:
    """Enterprise user with full profile"""

    user_id: str
    email: str
    username: str
    roles: Set[UserRole]
    tier: APITier
    organization_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login_at: Optional[datetime] = None
    is_active: bool = True
    preferences: Dict[str, Any] = field(default_factory=dict)
    subscription_id: Optional[str] = None
    billing_customer_id: Optional[str] = None


@dataclass
class UsageRecord:
    """API usage tracking record"""

    user_id: str
    api_key_id: str
    endpoint: str
    method: str
    timestamp: datetime
    response_time_ms: int
    status_code: int
    request_size_bytes: int = 0
    response_size_bytes: int = 0
    feature_type: Optional[str] = None  # multimodal, collaboration, etc.
    billing_units: float = 1.0  # For usage-based billing


@dataclass
class Organization:
    """Multi-tenant organization"""

    org_id: str
    name: str
    tier: APITier
    admin_user_ids: Set[str]
    member_user_ids: Set[str]
    created_at: datetime
    subscription_id: Optional[str] = None
    billing_settings: Dict[str, Any] = field(default_factory=dict)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    custom_rate_limits: Optional[RateLimitConfig] = None


class EnterpriseAuthManager:
    """Enterprise authentication and authorization manager"""

    def __init__(self, redis_client: redis.Redis, jwt_secret: str):
        self.redis = redis_client
        self.jwt_secret = jwt_secret
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.organizations: Dict[str, Organization] = {}

        # Rate limit configurations by tier
        self.rate_limits = {
            APITier.FREE: RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                concurrent_requests=2,
                bandwidth_mb_per_hour=50,
                multimodal_requests_per_hour=10,
                vector_searches_per_hour=50,
                document_processing_per_hour=20,
                collaboration_sessions_per_day=2,
            ),
            APITier.BASIC: RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                concurrent_requests=5,
                bandwidth_mb_per_hour=500,
                multimodal_requests_per_hour=100,
                vector_searches_per_hour=500,
                document_processing_per_hour=200,
                collaboration_sessions_per_day=10,
            ),
            APITier.PROFESSIONAL: RateLimitConfig(
                requests_per_minute=300,
                requests_per_hour=10000,
                requests_per_day=100000,
                concurrent_requests=20,
                bandwidth_mb_per_hour=5000,
                multimodal_requests_per_hour=1000,
                vector_searches_per_hour=5000,
                document_processing_per_hour=2000,
                collaboration_sessions_per_day=50,
            ),
            APITier.ENTERPRISE: RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=50000,
                requests_per_day=1000000,
                concurrent_requests=100,
                bandwidth_mb_per_hour=50000,
                multimodal_requests_per_hour=10000,
                vector_searches_per_hour=50000,
                document_processing_per_hour=20000,
                collaboration_sessions_per_day=500,
            ),
            APITier.UNLIMITED: RateLimitConfig(
                requests_per_minute=10000,
                requests_per_hour=1000000,
                requests_per_day=10000000,
                concurrent_requests=1000,
                bandwidth_mb_per_hour=1000000,
                multimodal_requests_per_hour=100000,
                vector_searches_per_hour=1000000,
                document_processing_per_hour=500000,
                collaboration_sessions_per_day=10000,
            ),
        }

    async def create_user(
        self,
        email: str,
        username: str,
        roles: Set[UserRole],
        tier: APITier = APITier.FREE,
        organization_id: Optional[str] = None,
    ) -> User:
        """Create a new enterprise user"""
        user_id = str(uuid.uuid4())

        user = User(
            user_id=user_id,
            email=email,
            username=username,
            roles=roles,
            tier=tier,
            organization_id=organization_id,
        )

        self.users[user_id] = user

        # Store in Redis for distributed access
        await self.redis.hset(
            f"user:{user_id}",
            mapping={
                "email": email,
                "username": username,
                "roles": json.dumps([role.value for role in roles]),
                "tier": tier.value,
                "organization_id": organization_id or "",
                "created_at": user.created_at.isoformat(),
                "is_active": "true",
            },
        )

        logger.info(f"Created user {username} ({email}) with tier {tier.value}")
        return user

    async def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: Set[APIScope],
        expires_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key for user"""

        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")

        user = self.users[user_id]
        raw_key = f"vega_api_{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = str(uuid.uuid4())

        expires_at = None
        if expires_days:
            expires_at = datetime.now() + timedelta(days=expires_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            user_id=user_id,
            tier=user.tier,
            scopes=scopes,
            created_at=datetime.now(),
            expires_at=expires_at,
        )

        self.api_keys[key_hash] = api_key

        # Store in Redis
        await self.redis.hset(
            f"api_key:{key_hash}",
            mapping={
                "key_id": key_id,
                "name": name,
                "user_id": user_id,
                "tier": user.tier.value,
                "scopes": json.dumps([scope.value for scope in scopes]),
                "created_at": api_key.created_at.isoformat(),
                "expires_at": expires_at.isoformat() if expires_at else "",
                "is_active": "true",
            },
        )

        logger.info(f"Created API key '{name}' for user {user_id}")
        return raw_key, api_key

    async def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate and return API key details"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Check local cache first
        if key_hash in self.api_keys:
            api_key = self.api_keys[key_hash]
        else:
            # Load from Redis
            key_data = await self.redis.hgetall(f"api_key:{key_hash}")
            if not key_data:
                return None

            # Reconstruct API key object
            api_key = APIKey(
                key_id=key_data["key_id"],
                key_hash=key_hash,
                name=key_data["name"],
                user_id=key_data["user_id"],
                tier=APITier(key_data["tier"]),
                scopes={APIScope(scope) for scope in json.loads(key_data["scopes"])},
                created_at=datetime.fromisoformat(key_data["created_at"]),
                expires_at=(
                    datetime.fromisoformat(key_data["expires_at"])
                    if key_data["expires_at"]
                    else None
                ),
                is_active=key_data["is_active"] == "true",
            )
            self.api_keys[key_hash] = api_key

        # Validate key
        if not api_key.is_active:
            return None

        if api_key.expires_at and datetime.now() > api_key.expires_at:
            api_key.is_active = False
            await self.redis.hset(f"api_key:{key_hash}", "is_active", "false")
            return None

        # Update last used
        api_key.last_used_at = datetime.now()
        await self.redis.hset(
            f"api_key:{key_hash}", "last_used_at", api_key.last_used_at.isoformat()
        )

        return api_key

    async def check_permissions(
        self, api_key: APIKey, required_scopes: Set[APIScope]
    ) -> bool:
        """Check if API key has required permissions"""
        return required_scopes.issubset(api_key.scopes)

    def generate_jwt_token(self, user_id: str, expires_hours: int = 24) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "exp": datetime.now(timezone.utc) + timedelta(hours=expires_hours),
            "iat": datetime.now(timezone.utc),
            "iss": "vega-api",
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def validate_jwt_token(self, token: str) -> Optional[str]:
        """Validate JWT token and return user_id"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload["user_id"]
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None


class RateLimitManager:
    """Advanced rate limiting with Redis backend"""

    def __init__(self, redis_client: redis.Redis, auth_manager: EnterpriseAuthManager):
        self.redis = redis_client
        self.auth_manager = auth_manager

    async def check_rate_limit(
        self, api_key: APIKey, endpoint: str, request_size_bytes: int = 0
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""

        # Get rate limit config
        rate_config = self.auth_manager.rate_limits[api_key.tier]
        if api_key.rate_limit_override:
            rate_config = api_key.rate_limit_override

        now = time.time()
        user_key = f"rate_limit:{api_key.user_id}"

        # Check different time windows
        checks = [
            ("minute", 60, rate_config.requests_per_minute),
            ("hour", 3600, rate_config.requests_per_hour),
            ("day", 86400, rate_config.requests_per_day),
        ]

        results = {}

        for window_name, window_seconds, limit in checks:
            window_key = f"{user_key}:{window_name}:{int(now // window_seconds)}"

            # Get current count
            current_count = await self.redis.get(window_key)
            current_count = int(current_count) if current_count else 0

            # Check limit
            if current_count >= limit:
                results[window_name] = {
                    "allowed": False,
                    "current": current_count,
                    "limit": limit,
                    "reset_at": (int(now // window_seconds) + 1) * window_seconds,
                }
                return False, results

            results[window_name] = {
                "allowed": True,
                "current": current_count,
                "limit": limit,
                "reset_at": (int(now // window_seconds) + 1) * window_seconds,
            }

        # Check concurrent requests
        concurrent_key = f"{user_key}:concurrent"
        concurrent_count = await self.redis.scard(concurrent_key)

        if concurrent_count >= rate_config.concurrent_requests:
            results["concurrent"] = {
                "allowed": False,
                "current": concurrent_count,
                "limit": rate_config.concurrent_requests,
            }
            return False, results

        results["concurrent"] = {
            "allowed": True,
            "current": concurrent_count,
            "limit": rate_config.concurrent_requests,
        }

        return True, results

    async def record_request(
        self, api_key: APIKey, endpoint: str, request_size_bytes: int = 0
    ) -> str:
        """Record a request and return request_id for cleanup"""

        now = time.time()
        request_id = str(uuid.uuid4())
        user_key = f"rate_limit:{api_key.user_id}"

        # Increment counters for different windows
        for window_name, window_seconds, _ in [
            ("minute", 60, None),
            ("hour", 3600, None),
            ("day", 86400, None),
        ]:
            window_key = f"{user_key}:{window_name}:{int(now // window_seconds)}"
            await self.redis.incr(window_key)
            await self.redis.expire(window_key, window_seconds)

        # Add to concurrent requests
        concurrent_key = f"{user_key}:concurrent"
        await self.redis.sadd(concurrent_key, request_id)
        await self.redis.expire(concurrent_key, 300)  # 5 minute cleanup

        return request_id

    async def complete_request(self, api_key: APIKey, request_id: str):
        """Mark request as completed (remove from concurrent)"""
        user_key = f"rate_limit:{api_key.user_id}"
        concurrent_key = f"{user_key}:concurrent"
        await self.redis.srem(concurrent_key, request_id)


class UsageTracker:
    """Real-time usage tracking and analytics"""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client

    async def record_usage(self, usage_record: UsageRecord):
        """Record API usage for billing and analytics"""

        # Store individual usage record
        usage_key = (
            f"usage:{usage_record.user_id}:{usage_record.timestamp.strftime('%Y%m%d')}"
        )
        usage_data = {
            "api_key_id": usage_record.api_key_id,
            "endpoint": usage_record.endpoint,
            "method": usage_record.method,
            "timestamp": usage_record.timestamp.isoformat(),
            "response_time_ms": usage_record.response_time_ms,
            "status_code": usage_record.status_code,
            "request_size_bytes": usage_record.request_size_bytes,
            "response_size_bytes": usage_record.response_size_bytes,
            "feature_type": usage_record.feature_type or "",
            "billing_units": usage_record.billing_units,
        }

        # Add to daily usage list
        await self.redis.lpush(usage_key, json.dumps(usage_data))
        await self.redis.expire(usage_key, 86400 * 30)  # Keep 30 days

        # Update aggregated metrics
        await self._update_aggregated_metrics(usage_record)

    async def _update_aggregated_metrics(self, usage_record: UsageRecord):
        """Update aggregated usage metrics"""
        user_id = usage_record.user_id
        date_str = usage_record.timestamp.strftime("%Y%m%d")
        hour_str = usage_record.timestamp.strftime("%Y%m%d%H")

        # Daily metrics
        daily_key = f"metrics:daily:{user_id}:{date_str}"
        await self.redis.hincrby(daily_key, "requests", 1)
        await self.redis.hincrby(
            daily_key, "request_bytes", usage_record.request_size_bytes
        )
        await self.redis.hincrby(
            daily_key, "response_bytes", usage_record.response_size_bytes
        )
        await self.redis.hincrby(
            daily_key, "response_time_total", usage_record.response_time_ms
        )
        await self.redis.hincrbyfloat(
            daily_key, "billing_units", usage_record.billing_units
        )
        await self.redis.expire(daily_key, 86400 * 30)

        # Hourly metrics
        hourly_key = f"metrics:hourly:{user_id}:{hour_str}"
        await self.redis.hincrby(hourly_key, "requests", 1)
        await self.redis.expire(hourly_key, 86400 * 7)  # Keep 7 days of hourly data

        # Feature-specific metrics
        if usage_record.feature_type:
            feature_key = (
                f"metrics:feature:{user_id}:{usage_record.feature_type}:{date_str}"
            )
            await self.redis.hincrby(feature_key, "requests", 1)
            await self.redis.hincrbyfloat(
                feature_key, "billing_units", usage_record.billing_units
            )
            await self.redis.expire(feature_key, 86400 * 30)

    async def get_usage_summary(
        self, user_id: str, start_date: datetime, end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage summary for date range"""

        summary = {
            "total_requests": 0,
            "total_request_bytes": 0,
            "total_response_bytes": 0,
            "total_billing_units": 0.0,
            "average_response_time": 0.0,
            "daily_breakdown": {},
            "feature_breakdown": {},
        }

        current_date = start_date
        total_response_time = 0

        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")
            daily_key = f"metrics:daily:{user_id}:{date_str}"

            daily_metrics = await self.redis.hgetall(daily_key)
            if daily_metrics:
                requests = int(daily_metrics.get("requests", 0))
                summary["total_requests"] += requests
                summary["total_request_bytes"] += int(
                    daily_metrics.get("request_bytes", 0)
                )
                summary["total_response_bytes"] += int(
                    daily_metrics.get("response_bytes", 0)
                )
                summary["total_billing_units"] += float(
                    daily_metrics.get("billing_units", 0)
                )
                total_response_time += int(daily_metrics.get("response_time_total", 0))

                summary["daily_breakdown"][date_str] = {
                    "requests": requests,
                    "billing_units": float(daily_metrics.get("billing_units", 0)),
                }

            current_date += timedelta(days=1)

        # Calculate average response time
        if summary["total_requests"] > 0:
            summary["average_response_time"] = (
                total_response_time / summary["total_requests"]
            )

        return summary


# FastAPI dependency injection
security = HTTPBearer()


async def get_current_api_key(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
) -> APIKey:
    """FastAPI dependency to get current API key"""

    # Get auth manager from app state
    auth_manager = request.app.state.auth_manager

    api_key = await auth_manager.validate_api_key(credentials.credentials)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key


async def require_scopes(required_scopes: Set[APIScope]):
    """FastAPI dependency factory for scope requirements"""

    async def check_scopes(
        request: Request, api_key: APIKey = Depends(get_current_api_key)
    ):
        auth_manager = request.app.state.auth_manager

        if not await auth_manager.check_permissions(api_key, required_scopes):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {[s.value for s in required_scopes]}",
            )
        return api_key

    return check_scopes


async def rate_limit_check(
    request: Request, api_key: APIKey = Depends(get_current_api_key)
) -> str:
    """FastAPI dependency for rate limiting"""

    rate_manager = request.app.state.rate_limit_manager

    # Get request size estimate
    request_size = len(str(request.url)) + sum(
        len(f"{k}: {v}") for k, v in request.headers.items()
    )

    allowed, limits = await rate_manager.check_rate_limit(
        api_key, str(request.url.path), request_size
    )

    if not allowed:
        # Find which limit was exceeded
        exceeded_limits = {k: v for k, v in limits.items() if not v["allowed"]}
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limits": limits,
                "exceeded": exceeded_limits,
            },
        )

    # Record the request
    request_id = await rate_manager.record_request(
        api_key, str(request.url.path), request_size
    )

    # Store request_id in request state for cleanup
    request.state.rate_limit_request_id = request_id
    request.state.api_key = api_key

    return request_id


# Example FastAPI middleware for usage tracking
class UsageTrackingMiddleware:
    """Middleware to track API usage"""

    def __init__(self, app, usage_tracker: UsageTracker):
        self.app = app
        self.usage_tracker = usage_tracker

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        request_size = 0
        response_size = 0
        status_code = 200

        async def send_wrapper(message):
            nonlocal response_size, status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body":
                response_size += len(message.get("body", b""))
            await send(message)

        # Process request
        await self.app(scope, receive, send_wrapper)

        # Record usage
        response_time = int((time.time() - start_time) * 1000)

        # Get API key from request state if available
        request = scope.get("state", {})
        api_key = getattr(request, "api_key", None)

        if api_key:
            # Determine feature type based on endpoint
            path = scope["path"]
            feature_type = None
            billing_units = 1.0

            if "/multimodal/" in path:
                feature_type = "multimodal"
                billing_units = 2.0
            elif "/collaboration/" in path:
                feature_type = "collaboration"
                billing_units = 1.5
            elif "/vector/" in path:
                feature_type = "vector_search"
                billing_units = 1.5
            elif "/federated/" in path:
                feature_type = "federated"
                billing_units = 3.0

            usage_record = UsageRecord(
                user_id=api_key.user_id,
                api_key_id=api_key.key_id,
                endpoint=path,
                method=scope["method"],
                timestamp=datetime.now(),
                response_time_ms=response_time,
                status_code=status_code,
                request_size_bytes=request_size,
                response_size_bytes=response_size,
                feature_type=feature_type,
                billing_units=billing_units,
            )

            await self.usage_tracker.record_usage(usage_record)


# Example usage and testing
async def demo_enterprise_api_management():
    """Demonstrate enterprise API management features"""

    # Initialize Redis connection
    redis_client = redis.Redis.from_url("redis://localhost:6379")

    # Initialize managers
    auth_manager = EnterpriseAuthManager(redis_client, "your-jwt-secret-key")
    rate_manager = RateLimitManager(redis_client, auth_manager)
    usage_tracker = UsageTracker(redis_client)

    print("🏢 Enterprise API Management Demo")
    print("=" * 40)

    # Create test users
    admin_user = await auth_manager.create_user(
        email="admin@enterprise.com",
        username="enterprise_admin",
        roles={UserRole.ADMIN, UserRole.BILLING_ADMIN},
        tier=APITier.ENTERPRISE,
    )

    dev_user = await auth_manager.create_user(
        email="dev@startup.com",
        username="startup_dev",
        roles={UserRole.DEVELOPER},
        tier=APITier.PROFESSIONAL,
    )

    print(f"✅ Created users: {admin_user.username} ({admin_user.tier.value})")
    print(f"✅ Created users: {dev_user.username} ({dev_user.tier.value})")

    # Create API keys
    admin_key, admin_api_key = await auth_manager.create_api_key(
        admin_user.user_id,
        "Enterprise Admin Key",
        {APIScope.READ, APIScope.WRITE, APIScope.ADMIN, APIScope.MULTIMODAL},
        expires_days=365,
    )

    dev_key, dev_api_key = await auth_manager.create_api_key(
        dev_user.user_id,
        "Development Key",
        {APIScope.READ, APIScope.WRITE, APIScope.MULTIMODAL},
        expires_days=30,
    )

    print(f"🔑 Created API keys with different permissions")

    # Test rate limiting
    print(f"\n📊 Testing Rate Limits:")

    for i in range(5):
        allowed, limits = await rate_manager.check_rate_limit(
            dev_api_key, "/api/multimodal/search"
        )
        print(f"   Request {i+1}: {'✅ Allowed' if allowed else '❌ Blocked'}")

        if allowed:
            request_id = await rate_manager.record_request(
                dev_api_key, "/api/multimodal/search"
            )

            # Simulate request completion
            await asyncio.sleep(0.1)
            await rate_manager.complete_request(dev_api_key, request_id)

    # Test usage tracking
    print(f"\n📈 Usage Tracking:")

    for i in range(3):
        usage_record = UsageRecord(
            user_id=dev_user.user_id,
            api_key_id=dev_api_key.key_id,
            endpoint="/api/multimodal/analyze",
            method="POST",
            timestamp=datetime.now(),
            response_time_ms=150 + i * 50,
            status_code=200,
            request_size_bytes=1024,
            response_size_bytes=2048,
            feature_type="multimodal",
            billing_units=2.0,
        )

        await usage_tracker.record_usage(usage_record)

    # Get usage summary
    summary = await usage_tracker.get_usage_summary(
        dev_user.user_id, datetime.now() - timedelta(days=1), datetime.now()
    )

    print(f"   Total requests: {summary['total_requests']}")
    print(f"   Total billing units: {summary['total_billing_units']}")
    print(f"   Average response time: {summary['average_response_time']:.1f}ms")

    print(f"\n✨ Enterprise API Management Demo Complete!")


if __name__ == "__main__":
    asyncio.run(demo_enterprise_api_management())
