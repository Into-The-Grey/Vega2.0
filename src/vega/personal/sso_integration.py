"""
Personal SSO Integration System

Provides seamless authentication with external services for personal accounts
including Azure AD, Google Workspace, GitHub, and other identity providers.
"""

import asyncio
import json
import logging
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import jwt
from urllib.parse import urlencode, parse_qs
import aiofiles

logger = logging.getLogger(__name__)


class SSOProvider(Enum):
    """Supported SSO providers"""

    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    CUSTOM_OIDC = "custom_oidc"


class AuthStatus(Enum):
    """Authentication status"""

    AUTHENTICATED = "authenticated"
    PENDING = "pending"
    EXPIRED = "expired"
    FAILED = "failed"
    REVOKED = "revoked"


@dataclass
class SSOConfig:
    """SSO provider configuration"""

    provider: SSOProvider
    client_id: str
    client_secret: str
    tenant_id: Optional[str] = None  # For Azure AD
    redirect_uri: str = "http://localhost:8000/auth/callback"
    scopes: List[str] = None
    endpoints: Dict[str, str] = None
    enabled: bool = True

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = self._get_default_scopes()
        if self.endpoints is None:
            self.endpoints = self._get_default_endpoints()

    def _get_default_scopes(self) -> List[str]:
        """Get default scopes for provider"""
        defaults = {
            SSOProvider.AZURE_AD: ["openid", "profile", "email", "User.Read"],
            SSOProvider.GOOGLE: ["openid", "profile", "email"],
            SSOProvider.GITHUB: ["user:email", "read:user"],
            SSOProvider.MICROSOFT: ["openid", "profile", "email", "User.Read"],
            SSOProvider.CUSTOM_OIDC: ["openid", "profile", "email"],
        }
        return defaults.get(self.provider, ["openid", "profile", "email"])

    def _get_default_endpoints(self) -> Dict[str, str]:
        """Get default endpoints for provider"""
        if self.provider == SSOProvider.AZURE_AD:
            base_url = f"https://login.microsoftonline.com/{self.tenant_id or 'common'}"
            return {
                "auth": f"{base_url}/oauth2/v2.0/authorize",
                "token": f"{base_url}/oauth2/v2.0/token",
                "userinfo": "https://graph.microsoft.com/v1.0/me",
                "logout": f"{base_url}/oauth2/v2.0/logout",
            }
        elif self.provider == SSOProvider.GOOGLE:
            return {
                "auth": "https://accounts.google.com/o/oauth2/v2/auth",
                "token": "https://oauth2.googleapis.com/token",
                "userinfo": "https://www.googleapis.com/oauth2/v2/userinfo",
                "logout": "https://accounts.google.com/logout",
            }
        elif self.provider == SSOProvider.GITHUB:
            return {
                "auth": "https://github.com/login/oauth/authorize",
                "token": "https://github.com/login/oauth/access_token",
                "userinfo": "https://api.github.com/user",
                "logout": None,
            }
        elif self.provider == SSOProvider.MICROSOFT:
            return {
                "auth": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
                "token": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
                "userinfo": "https://graph.microsoft.com/v1.0/me",
                "logout": "https://login.microsoftonline.com/common/oauth2/v2.0/logout",
            }
        else:
            return {}


@dataclass
class UserProfile:
    """User profile from SSO provider"""

    provider: SSOProvider
    provider_id: str
    email: str
    name: str
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    locale: Optional[str] = None
    raw_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_data is None:
            self.raw_data = {}


@dataclass
class AuthSession:
    """Authentication session data"""

    session_id: str
    provider: SSOProvider
    status: AuthStatus
    user_profile: Optional[UserProfile]
    access_token: str
    refresh_token: Optional[str]
    token_expires_at: datetime
    created_at: datetime
    last_accessed: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.now() >= self.token_expires_at

    @property
    def expires_in_seconds(self) -> int:
        """Get seconds until expiration"""
        if self.is_expired:
            return 0
        return int((self.token_expires_at - datetime.now()).total_seconds())


class PersonalSSOManager:
    """
    Personal SSO integration manager
    Handles authentication with multiple providers for personal use
    """

    def __init__(self, config_file: str = "configs/sso_config.json"):
        self.config_file = config_file
        self.providers: Dict[SSOProvider, SSOConfig] = {}
        self.sessions: Dict[str, AuthSession] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def initialize(self):
        """Initialize SSO manager"""
        await self._load_configuration()
        logger.info(f"SSO Manager initialized with {len(self.providers)} providers")

    async def _load_configuration(self):
        """Load SSO configuration from file"""
        try:
            async with aiofiles.open(self.config_file, "r") as f:
                config_data = json.loads(await f.read())

            for provider_name, config in config_data.get("providers", {}).items():
                try:
                    provider = SSOProvider(provider_name)
                    sso_config = SSOConfig(
                        provider=provider,
                        client_id=config["client_id"],
                        client_secret=config["client_secret"],
                        tenant_id=config.get("tenant_id"),
                        redirect_uri=config.get(
                            "redirect_uri", "http://localhost:8000/auth/callback"
                        ),
                        scopes=config.get("scopes"),
                        endpoints=config.get("endpoints"),
                        enabled=config.get("enabled", True),
                    )
                    self.providers[provider] = sso_config
                    logger.info(f"Loaded SSO config for {provider_name}")

                except Exception as e:
                    logger.error(f"Failed to load config for {provider_name}: {e}")

        except FileNotFoundError:
            logger.warning(f"SSO config file not found: {self.config_file}")
            await self._create_default_config()
        except Exception as e:
            logger.error(f"Failed to load SSO configuration: {e}")

    async def _create_default_config(self):
        """Create default SSO configuration file"""
        default_config = {
            "providers": {
                "azure_ad": {
                    "client_id": "your_azure_client_id",
                    "client_secret": "your_azure_client_secret",
                    "tenant_id": "your_tenant_id",
                    "enabled": False,
                },
                "google": {
                    "client_id": "your_google_client_id.apps.googleusercontent.com",
                    "client_secret": "your_google_client_secret",
                    "enabled": False,
                },
                "github": {
                    "client_id": "your_github_client_id",
                    "client_secret": "your_github_client_secret",
                    "enabled": False,
                },
            }
        }

        # Ensure config directory exists
        import os

        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

        async with aiofiles.open(self.config_file, "w") as f:
            await f.write(json.dumps(default_config, indent=2))

        logger.info(f"Created default SSO config at {self.config_file}")

    def get_auth_url(self, provider: SSOProvider, state: Optional[str] = None) -> str:
        """Generate authentication URL for provider"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not configured")

        config = self.providers[provider]
        if not config.enabled:
            raise ValueError(f"Provider {provider.value} is disabled")

        # Generate state parameter if not provided
        if state is None:
            state = secrets.token_urlsafe(32)

        # Prepare authorization parameters
        auth_params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": " ".join(config.scopes),
            "state": state,
            "response_type": "code",
        }

        # Provider-specific parameters
        if provider == SSOProvider.AZURE_AD:
            auth_params["response_mode"] = "query"
        elif provider == SSOProvider.GOOGLE:
            auth_params["access_type"] = "offline"
            auth_params["prompt"] = "consent"

        auth_url = f"{config.endpoints['auth']}?{urlencode(auth_params)}"
        logger.info(f"Generated auth URL for {provider.value}")
        return auth_url

    async def handle_callback(
        self, provider: SSOProvider, code: str, state: str
    ) -> AuthSession:
        """Handle OAuth callback and create session"""
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not configured")

        config = self.providers[provider]

        # Exchange code for tokens
        token_data = await self._exchange_code_for_tokens(config, code)

        # Get user profile
        user_profile = await self._get_user_profile(config, token_data["access_token"])

        # Create session
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(
            seconds=token_data.get("expires_in", 3600)
        )

        session = AuthSession(
            session_id=session_id,
            provider=provider,
            status=AuthStatus.AUTHENTICATED,
            user_profile=user_profile,
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token"),
            token_expires_at=expires_at,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            metadata={"state": state},
        )

        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} for {user_profile.email}")
        return session

    async def _exchange_code_for_tokens(
        self, config: SSOConfig, code: str
    ) -> Dict[str, Any]:
        """Exchange authorization code for access tokens"""
        token_data = {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_uri,
        }

        headers = {"Accept": "application/json"}

        try:
            response = await self.http_client.post(
                config.endpoints["token"], data=token_data, headers=headers
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise

    async def _get_user_profile(
        self, config: SSOConfig, access_token: str
    ) -> UserProfile:
        """Get user profile from provider"""
        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            response = await self.http_client.get(
                config.endpoints["userinfo"], headers=headers
            )
            response.raise_for_status()
            user_data = response.json()

            # Parse user data based on provider
            return self._parse_user_profile(config.provider, user_data)

        except Exception as e:
            logger.error(f"Failed to get user profile: {e}")
            raise

    def _parse_user_profile(
        self, provider: SSOProvider, user_data: Dict[str, Any]
    ) -> UserProfile:
        """Parse user profile data from provider response"""
        if provider in [SSOProvider.AZURE_AD, SSOProvider.MICROSOFT]:
            return UserProfile(
                provider=provider,
                provider_id=user_data.get("id"),
                email=user_data.get("mail") or user_data.get("userPrincipalName"),
                name=user_data.get("displayName"),
                given_name=user_data.get("givenName"),
                family_name=user_data.get("surname"),
                picture=None,  # Need separate call for photo
                locale=user_data.get("preferredLanguage"),
                raw_data=user_data,
            )
        elif provider == SSOProvider.GOOGLE:
            return UserProfile(
                provider=provider,
                provider_id=user_data.get("id"),
                email=user_data.get("email"),
                name=user_data.get("name"),
                given_name=user_data.get("given_name"),
                family_name=user_data.get("family_name"),
                picture=user_data.get("picture"),
                locale=user_data.get("locale"),
                raw_data=user_data,
            )
        elif provider == SSOProvider.GITHUB:
            return UserProfile(
                provider=provider,
                provider_id=str(user_data.get("id")),
                email=user_data.get("email"),
                name=user_data.get("name") or user_data.get("login"),
                given_name=None,
                family_name=None,
                picture=user_data.get("avatar_url"),
                locale=None,
                raw_data=user_data,
            )
        else:
            # Generic OIDC
            return UserProfile(
                provider=provider,
                provider_id=user_data.get("sub"),
                email=user_data.get("email"),
                name=user_data.get("name"),
                given_name=user_data.get("given_name"),
                family_name=user_data.get("family_name"),
                picture=user_data.get("picture"),
                locale=user_data.get("locale"),
                raw_data=user_data,
            )

    async def refresh_session(self, session_id: str) -> AuthSession:
        """Refresh authentication session"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]

        if not session.refresh_token:
            raise ValueError("No refresh token available")

        config = self.providers[session.provider]

        # Refresh tokens
        refresh_data = {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "refresh_token": session.refresh_token,
            "grant_type": "refresh_token",
        }

        try:
            response = await self.http_client.post(
                config.endpoints["token"],
                data=refresh_data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_data = response.json()

            # Update session
            session.access_token = token_data["access_token"]
            if "refresh_token" in token_data:
                session.refresh_token = token_data["refresh_token"]

            session.token_expires_at = datetime.now() + timedelta(
                seconds=token_data.get("expires_in", 3600)
            )
            session.last_accessed = datetime.now()

            logger.info(f"Refreshed session {session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to refresh session {session_id}: {e}")
            session.status = AuthStatus.EXPIRED
            raise

    async def logout_session(self, session_id: str) -> bool:
        """Logout and invalidate session"""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        config = self.providers[session.provider]

        # Revoke tokens if provider supports it
        if config.endpoints.get("logout"):
            try:
                logout_url = config.endpoints["logout"]
                if session.provider == SSOProvider.AZURE_AD:
                    logout_url += f"?post_logout_redirect_uri={config.redirect_uri}"

                await self.http_client.get(logout_url)
                logger.info(f"Logged out from {session.provider.value}")

            except Exception as e:
                logger.warning(f"Failed to logout from provider: {e}")

        # Remove session
        session.status = AuthStatus.REVOKED
        del self.sessions[session_id]

        logger.info(f"Session {session_id} logged out")
        return True

    def get_session(self, session_id: str) -> Optional[AuthSession]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session:
            session.last_accessed = datetime.now()
        return session

    def get_active_sessions(self) -> List[AuthSession]:
        """Get all active sessions"""
        return [
            s
            for s in self.sessions.values()
            if s.status == AuthStatus.AUTHENTICATED and not s.is_expired
        ]

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        expired_sessions = [
            sid
            for sid, session in self.sessions.items()
            if session.is_expired
            or session.status in [AuthStatus.EXPIRED, AuthStatus.REVOKED]
        ]

        for session_id in expired_sessions:
            del self.sessions[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()


# Integration with FastAPI
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[UserProfile]:
    """FastAPI dependency to get current authenticated user"""

    # Check for session ID in headers or cookies
    session_id = None

    if credentials:
        session_id = credentials.credentials
    else:
        session_id = request.cookies.get("session_id")

    if not session_id:
        return None

    # Get SSO manager from app state
    sso_manager = getattr(request.app.state, "sso_manager", None)
    if not sso_manager:
        return None

    session = sso_manager.get_session(session_id)
    if not session or session.is_expired:
        return None

    return session.user_profile


async def require_authentication(
    user: UserProfile = Depends(get_current_user),
) -> UserProfile:
    """FastAPI dependency that requires authentication"""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# Demo and testing functions
async def demo_personal_sso():
    """Demonstrate personal SSO capabilities"""

    sso_manager = PersonalSSOManager()
    await sso_manager.initialize()

    print("Personal SSO Integration Demo")
    print(f"Configured providers: {[p.value for p in sso_manager.providers.keys()]}")

    # Generate auth URLs (for demo purposes)
    for provider in sso_manager.providers.keys():
        try:
            auth_url = sso_manager.get_auth_url(provider)
            print(f"{provider.value} auth URL: {auth_url[:80]}...")
        except Exception as e:
            print(f"{provider.value}: {e}")

    await sso_manager.close()
    return sso_manager


if __name__ == "__main__":
    asyncio.run(demo_personal_sso())
