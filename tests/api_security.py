#!/usr/bin/env python3
"""
api_security.py - API security utilities for tests

Provides API security management classes and utilities used by test modules.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING
from enum import Enum
import secrets
import hashlib
import time
import jwt

if TYPE_CHECKING:
    from tests.ecc_crypto import SecureAPIKey


class SecurityLevel(Enum):
    """Security levels for API access"""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized"
    ADMIN = "admin"


@dataclass
class ApiKey:
    """API key representation"""

    key_id: str
    key_value: str
    permissions: Set[str]
    created_at: float
    expires_at: Optional[float] = None
    is_active: bool = True

    def has_permission(self, permission: str) -> bool:
        """Check if the API key has a specific permission"""
        return permission in self.permissions


@dataclass
class SecurityPolicy:
    """Security policy configuration"""

    require_api_key: bool = True
    require_jwt: bool = False
    allow_anonymous: bool = False
    rate_limit_per_minute: int = 60
    max_request_size: int = 1024 * 1024  # 1MB
    allowed_origins: Optional[List[str]] = None
    required_headers: Optional[List[str]] = None


class SecurityManager:
    """Mock security manager for testing"""

    def __init__(self):
        self.api_keys: Dict[str, ApiKey] = {}
        self.jwt_secret = "test-jwt-secret"
        self.security_policy = SecurityPolicy()
        self.request_counts: Dict[str, int] = {}
        self.blocked_ips: Set[str] = set()

        # Generate default test API key
        self._create_default_api_key()

    def _create_default_api_key(self) -> None:
        """Create default API key for testing"""
        key = ApiKey(
            key_id="test-key-1",
            key_value="test-api-key-12345",
            permissions={"read", "write", "admin"},
            created_at=time.time(),
        )
        self.api_keys[key.key_value] = key

    def validate_api_key(self, api_key: str) -> Optional[ApiKey]:
        """Validate API key and return key info if valid"""
        key_info = self.api_keys.get(api_key)

        if not key_info:
            return None

        if not key_info.is_active:
            return None

        if key_info.expires_at and time.time() > key_info.expires_at:
            return None

        return key_info

    def create_api_key(
        self,
        key_id: str,
        permissions: Set[str],
        expires_in_seconds: Optional[int] = None,
    ) -> ApiKey:
        """Create new API key"""
        key_value = f"vega-{secrets.token_urlsafe(32)}"
        expires_at = None

        if expires_in_seconds:
            expires_at = time.time() + expires_in_seconds

        key = ApiKey(
            key_id=key_id,
            key_value=key_value,
            permissions=permissions,
            created_at=time.time(),
            expires_at=expires_at,
        )

        self.api_keys[key_value] = key
        return key

    def generate_secure_api_key(
        self,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
    ) -> "SecureAPIKey":
        """Generate a secure API key with ECC integration"""
        from tests.ecc_crypto import get_ecc_manager, SecureAPIKey

        # Generate an ECC key for the API key
        ecc_manager = get_ecc_manager()
        ecc_key = ecc_manager.generate_key_pair()

        # Mock secure API key generation
        perms = permissions or []
        key_id = f"api_key_{secrets.token_hex(8)}"
        api_key = hashlib.sha256(f"{key_id}:{','.join(perms)}".encode()).hexdigest()[
            :32
        ]

        # Store API key for validation
        expires_at = None
        if expires_in_days:
            expires_at = time.time() + (expires_in_days * 24 * 60 * 60)

        api_key_obj = ApiKey(
            key_id=key_id,
            key_value=api_key,
            permissions=set(perms),
            created_at=time.time(),
            expires_at=expires_at,
            is_active=True,
        )
        self.api_keys[api_key] = api_key_obj

        return SecureAPIKey(
            key_id=key_id,
            api_key=api_key,
            ecc_key_id=ecc_key.key_id or "unknown",
            permissions=perms,
            expires_in_days=expires_in_days,
            rate_limit=rate_limit,
        )

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        if api_key in self.api_keys:
            self.api_keys[api_key].is_active = False
            return True
        return False

    def generate_secure_token(
        self, payload: Dict[str, Any], ecc_key_id: str, expires_in_minutes: int = 30
    ) -> str:
        """Generate a secure JWT token"""
        expiry = time.time() + (expires_in_minutes * 60)
        token_payload = {
            **payload,
            "exp": expiry,
            "iat": time.time(),
            "ecc_key_id": ecc_key_id,
        }
        return jwt.encode(token_payload, self.jwt_secret, algorithm="HS256")

    def verify_secure_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a secure JWT token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

    def create_jwt_token(self, user_id: str, permissions: List[str]) -> str:
        """Create JWT token for user"""
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "iat": time.time(),
            "exp": time.time() + 3600,  # 1 hour expiry
        }

        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    def validate_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token and return payload if valid"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def check_permissions(self, api_key: str, required_permission: str) -> bool:
        """Check if API key has required permission"""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return False

        return (
            required_permission in key_info.permissions
            or "admin" in key_info.permissions
        )

    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits"""
        current_count = self.request_counts.get(client_id, 0)
        return current_count < self.security_policy.rate_limit_per_minute

    def record_request(self, client_id: str) -> None:
        """Record request for rate limiting"""
        self.request_counts[client_id] = self.request_counts.get(client_id, 0) + 1

    def reset_rate_limits(self) -> None:
        """Reset rate limit counters"""
        self.request_counts.clear()

    def block_ip(self, ip_address: str) -> None:
        """Block IP address"""
        self.blocked_ips.add(ip_address)

    def unblock_ip(self, ip_address: str) -> None:
        """Unblock IP address"""
        self.blocked_ips.discard(ip_address)

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips

    def validate_request_size(self, content_length: int) -> bool:
        """Validate request size against policy"""
        return content_length <= self.security_policy.max_request_size

    def validate_origin(self, origin: str) -> bool:
        """Validate request origin against policy"""
        if not self.security_policy.allowed_origins:
            return True

        return origin in self.security_policy.allowed_origins

    def hash_password(
        self, password: str, salt: Optional[str] = None
    ) -> tuple[str, str]:
        """Hash password with salt"""
        if not salt:
            salt = secrets.token_hex(16)

        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000
        )

        return password_hash.hex(), salt

    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        computed_hash, _ = self.hash_password(password, salt)
        return computed_hash == password_hash

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for responses"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }


# Global security manager instance
_security_manager = None


def get_security_manager() -> SecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager


def reset_security_manager() -> None:
    """Reset global security manager"""
    global _security_manager
    _security_manager = None


# Mock authentication decorators for testing
def require_api_key(func):
    """Decorator to require API key for function"""

    def wrapper(*args, **kwargs):
        # Mock implementation - always allow in tests
        return func(*args, **kwargs)

    return wrapper


def require_permission(permission: str):
    """Decorator to require specific permission"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Mock implementation - always allow in tests
            return func(*args, **kwargs)

        return wrapper

    return decorator


def require_jwt(func):
    """Decorator to require JWT token"""

    def wrapper(*args, **kwargs):
        # Mock implementation - always allow in tests
        return func(*args, **kwargs)

    return wrapper


# Mock security utilities
class MockSecurityAuditor:
    """Mock security auditor for testing"""

    def __init__(self):
        self.audit_logs: List[Dict[str, Any]] = []

    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log security event"""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": user_id,
            "ip_address": ip_address,
            "details": details or {},
        }
        self.audit_logs.append(event)

    def get_security_events(
        self, event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get security events by type"""
        if event_type is None:
            return self.audit_logs.copy()

        return [event for event in self.audit_logs if event["event_type"] == event_type]

    def clear_logs(self) -> None:
        """Clear audit logs"""
        self.audit_logs.clear()


class MockEncryptionService:
    """Mock encryption service for testing"""

    def __init__(self):
        self.encryption_key = "test-encryption-key-32-bytes!!"

    def encrypt_data(self, data: bytes) -> bytes:
        """Mock encrypt data"""
        # Simple XOR encryption for testing
        key_bytes = self.encryption_key.encode()
        encrypted = bytearray()

        for i, byte in enumerate(data):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        return bytes(encrypted)

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Mock decrypt data"""
        # Reverse of XOR encryption
        return self.encrypt_data(encrypted_data)

    def generate_random_key(self, length: int = 32) -> str:
        """Generate random encryption key"""
        return secrets.token_urlsafe(length)


# Global instances
_security_auditor = None
_encryption_service = None


def get_security_auditor() -> MockSecurityAuditor:
    """Get global security auditor instance"""
    global _security_auditor
    if _security_auditor is None:
        _security_auditor = MockSecurityAuditor()
    return _security_auditor


def get_encryption_service() -> MockEncryptionService:
    """Get global encryption service instance"""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = MockEncryptionService()
    return _encryption_service
