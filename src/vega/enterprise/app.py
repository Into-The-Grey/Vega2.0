"""
Enterprise FastAPI Integration
=============================

FastAPI application extensions for enterprise features including
advanced authentication, rate limiting, usage tracking, and billing.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, Set
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from fastapi.responses import JSONResponse
import uvicorn

from .api_management import (
    EnterpriseAuthManager,
    RateLimitManager,
    UsageTracker,
    APIKey,
    APIScope,
    UsageRecord,
    get_current_api_key,
    require_scopes,
    rate_limit_check,
)
from .config import get_enterprise_config, EnterpriseConfig

logger = logging.getLogger(__name__)


class EnterpriseApp:
    """Enterprise-enhanced FastAPI application"""

    def __init__(self, title: str = "Vega2.0 Enterprise API"):
        self.config = get_enterprise_config()

        # Initialize FastAPI app with enterprise features
        self.app = FastAPI(
            title=title,
            version="2.0.0",
            description="Advanced AI Platform with Enterprise Features",
            docs_url="/docs" if self.config.debug else None,
            redoc_url="/redoc" if self.config.debug else None,
            lifespan=self.lifespan,
        )

        # Initialize enterprise managers
        self.redis_client: Optional[redis.Redis] = None
        self.auth_manager: Optional[EnterpriseAuthManager] = None
        self.rate_manager: Optional[RateLimitManager] = None
        self.usage_tracker: Optional[UsageTracker] = None

        self._setup_middleware()
        self._setup_routes()

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan management"""

        # Startup
        logger.info("Starting Enterprise API...")

        # Initialize Redis connection
        redis_url = f"redis://{self.config.redis.host}:{self.config.redis.port}/{self.config.redis.db}"
        if self.config.redis.password:
            redis_url = f"redis://:{self.config.redis.password}@{self.config.redis.host}:{self.config.redis.port}/{self.config.redis.db}"

        self.redis_client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=self.config.redis.pool_size,
        )

        # Test Redis connection
        try:
            await self.redis_client.ping()
            logger.info("✅ Redis connection established")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise

        # Initialize enterprise managers
        self.auth_manager = EnterpriseAuthManager(
            self.redis_client, self.config.security.jwt_secret
        )
        self.rate_manager = RateLimitManager(self.redis_client, self.auth_manager)
        self.usage_tracker = UsageTracker(self.redis_client)

        # Store in app state for dependency injection
        app.state.redis_client = self.redis_client
        app.state.auth_manager = self.auth_manager
        app.state.rate_manager = self.rate_manager
        app.state.usage_tracker = self.usage_tracker
        app.state.enterprise_config = self.config

        # Create default admin user if none exists
        await self._ensure_admin_user()

        logger.info("🚀 Enterprise API started successfully")

        yield

        # Shutdown
        logger.info("Shutting down Enterprise API...")

        if self.redis_client:
            await self.redis_client.close()

        logger.info("✅ Enterprise API shutdown complete")

    def _setup_middleware(self):
        """Setup enterprise middleware"""

        # CORS middleware
        if self.config.security.cors_origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.security.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Trusted host middleware
        if not self.config.debug:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=["localhost", "127.0.0.1", "*.enterprise.com"],
            )

        # Usage tracking middleware
        @self.app.middleware("http")
        async def usage_tracking_middleware(request: Request, call_next):
            start_time = time.time()

            # Process request
            response = await call_next(request)

            # Track usage if API key is present
            if hasattr(request.state, "api_key"):
                api_key = request.state.api_key
                response_time = int((time.time() - start_time) * 1000)

                # Determine feature type and billing units
                path = str(request.url.path)
                feature_type = self._get_feature_type(path)
                billing_units = self._calculate_billing_units(
                    path, response.status_code
                )

                # Record usage
                usage_record = UsageRecord(
                    user_id=api_key.user_id,
                    api_key_id=api_key.key_id,
                    endpoint=path,
                    method=request.method,
                    timestamp=datetime.now(),
                    response_time_ms=response_time,
                    status_code=response.status_code,
                    request_size_bytes=self._estimate_request_size(request),
                    response_size_bytes=self._estimate_response_size(response),
                    feature_type=feature_type,
                    billing_units=billing_units,
                )

                # Record usage in background
                if self.usage_tracker:
                    asyncio.create_task(self.usage_tracker.record_usage(usage_record))

                # Complete rate limit request
                if hasattr(request.state, "rate_limit_request_id"):
                    asyncio.create_task(
                        self.rate_manager.complete_request(
                            api_key, request.state.rate_limit_request_id
                        )
                    )

            return response

        # Security headers middleware
        if self.config.security.security_headers:

            @self.app.middleware("http")
            async def security_headers_middleware(request: Request, call_next):
                response = await call_next(request)

                response.headers["X-Content-Type-Options"] = "nosniff"
                response.headers["X-Frame-Options"] = "DENY"
                response.headers["X-XSS-Protection"] = "1; mode=block"
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains"
                )
                response.headers["Content-Security-Policy"] = "default-src 'self'"

                return response

    def _setup_routes(self):
        """Setup enterprise API routes"""

        # Health check endpoints
        @self.app.get("/healthz")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/readyz")
        async def readiness_check(request: Request):
            """Readiness check with dependency validation"""
            checks = {}

            # Check Redis connection
            try:
                await request.app.state.redis_client.ping()
                checks["redis"] = "healthy"
            except Exception as e:
                checks["redis"] = f"unhealthy: {e}"

            # Overall status
            status = (
                "ready" if all(v == "healthy" for v in checks.values()) else "not_ready"
            )
            status_code = 200 if status == "ready" else 503

            return JSONResponse(
                content={"status": status, "checks": checks}, status_code=status_code
            )

        # Authentication and user management endpoints
        @self.app.post("/auth/users")
        async def create_user(
            request: Request,
            user_data: dict,
            api_key: APIKey = Depends(require_scopes({APIScope.ADMIN})),
        ):
            """Create a new user (admin only)"""

            auth_manager = request.app.state.auth_manager

            # Create user
            user = await auth_manager.create_user(
                email=user_data["email"],
                username=user_data["username"],
                roles=set(user_data.get("roles", ["user"])),
                tier=user_data.get("tier", "free"),
                organization_id=user_data.get("organization_id"),
            )

            return {
                "user_id": user.user_id,
                "email": user.email,
                "username": user.username,
                "tier": user.tier.value,
                "created_at": user.created_at.isoformat(),
            }

        @self.app.post("/auth/api-keys")
        async def create_api_key(
            request: Request,
            key_data: dict,
            current_api_key: APIKey = Depends(get_current_api_key),
        ):
            """Create a new API key"""

            auth_manager = request.app.state.auth_manager

            # Create API key for current user
            raw_key, api_key = await auth_manager.create_api_key(
                user_id=current_api_key.user_id,
                name=key_data["name"],
                scopes=set(key_data.get("scopes", ["read"])),
                expires_days=key_data.get("expires_days"),
            )

            return {
                "api_key": raw_key,
                "key_id": api_key.key_id,
                "name": api_key.name,
                "scopes": [scope.value for scope in api_key.scopes],
                "expires_at": (
                    api_key.expires_at.isoformat() if api_key.expires_at else None
                ),
            }

        # Usage and billing endpoints
        @self.app.get("/usage/summary")
        async def get_usage_summary(
            request: Request,
            start_date: str,
            end_date: str,
            api_key: APIKey = Depends(get_current_api_key),
        ):
            """Get usage summary for date range"""

            usage_tracker = request.app.state.usage_tracker

            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)

            summary = await usage_tracker.get_usage_summary(
                api_key.user_id, start_dt, end_dt
            )

            return summary

        @self.app.get("/usage/current-limits")
        async def get_current_limits(
            request: Request, api_key: APIKey = Depends(get_current_api_key)
        ):
            """Get current rate limits and usage"""

            rate_manager = request.app.state.rate_manager

            # Check current limits without recording request
            allowed, limits = await rate_manager.check_rate_limit(
                api_key, "/api/check", 0
            )

            return {
                "tier": api_key.tier.value,
                "limits": limits,
                "within_limits": allowed,
            }

        # Enterprise admin endpoints
        @self.app.get("/admin/metrics")
        async def get_admin_metrics(
            request: Request,
            api_key: APIKey = Depends(require_scopes({APIScope.ADMIN})),
        ):
            """Get system-wide metrics (admin only)"""

            redis_client = request.app.state.redis_client

            # Get Redis metrics
            redis_info = await redis_client.info()

            # Get user counts by tier
            # This is a simplified example - in production, you'd have proper metrics collection
            metrics = {
                "redis_info": {
                    "connected_clients": redis_info.get("connected_clients", 0),
                    "used_memory_human": redis_info.get("used_memory_human", "0B"),
                    "total_commands_processed": redis_info.get(
                        "total_commands_processed", 0
                    ),
                },
                "api_metrics": {
                    "total_requests_today": 0,  # Would be calculated from usage data
                    "active_api_keys": 0,  # Would be calculated from Redis
                    "error_rate": 0.0,  # Would be calculated from logs
                },
            }

            return metrics

    def _get_feature_type(self, path: str) -> Optional[str]:
        """Determine feature type from API path"""
        if "/multimodal/" in path:
            return "multimodal"
        elif "/collaboration/" in path:
            return "collaboration"
        elif "/vector/" in path:
            return "vector_search"
        elif "/federated/" in path:
            return "federated"
        elif "/admin/" in path:
            return "admin"
        return None

    def _calculate_billing_units(self, path: str, status_code: int) -> float:
        """Calculate billing units based on endpoint and response"""
        if status_code >= 400:
            return 0.1  # Reduced cost for errors

        feature_type = self._get_feature_type(path)

        if feature_type == "multimodal":
            return 2.0
        elif feature_type == "collaboration":
            return 1.5
        elif feature_type == "vector_search":
            return 1.5
        elif feature_type == "federated":
            return 3.0
        elif feature_type == "admin":
            return 0.5

        return 1.0  # Default

    def _estimate_request_size(self, request: Request) -> int:
        """Estimate request size in bytes"""
        # This is a simplified estimation
        size = len(str(request.url))
        size += sum(len(f"{k}: {v}") for k, v in request.headers.items())
        return size

    def _estimate_response_size(self, response) -> int:
        """Estimate response size in bytes"""
        # This is a simplified estimation
        content_length = response.headers.get("content-length")
        if content_length:
            return int(content_length)
        return 1024  # Default estimate

    async def _ensure_admin_user(self):
        """Ensure at least one admin user exists"""

        # Check if any admin users exist
        admin_exists = False  # In production, check database/Redis

        if not admin_exists and self.config.debug:
            from .api_management import UserRole, APITier, APIScope

            # Create default admin user for development
            admin_user = await self.auth_manager.create_user(
                email="admin@vega.local",
                username="admin",
                roles={UserRole.SUPER_ADMIN, UserRole.BILLING_ADMIN},
                tier=APITier.UNLIMITED,
            )

            # Create admin API key
            admin_key, _ = await self.auth_manager.create_api_key(
                admin_user.user_id,
                "Default Admin Key",
                {APIScope.READ, APIScope.WRITE, APIScope.ADMIN, APIScope.BILLING},
                expires_days=365,
            )

            logger.info(f"Created default admin user: admin@vega.local")
            logger.info(f"Default admin API key: {admin_key}")


# Factory function to create enterprise app
def create_enterprise_app(title: str = "Vega2.0 Enterprise API") -> FastAPI:
    """Create and configure enterprise FastAPI application"""

    enterprise_app = EnterpriseApp(title)
    return enterprise_app.app


# Development server
def run_enterprise_server(
    host: str = "127.0.0.1", port: int = 8000, reload: bool = False
):
    """Run enterprise development server"""

    app = create_enterprise_app()

    uvicorn.run(
        "src.vega.enterprise.app:create_enterprise_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vega2.0 Enterprise API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    run_enterprise_server(args.host, args.port, args.reload)
