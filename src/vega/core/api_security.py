"""
API Security module for Vega2.0 with ECC-backed authentication.
"""

import os
import time
from typing import Optional, List, Dict, Any
from functools import wraps
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


class ECCAPISecurityManager:
    """Manages ECC-backed API security for Vega2.0."""
    
    def __init__(self):
        self.api_keys = {}
        self.permissions = {}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment."""
        # Load primary API key
        primary_key = os.getenv("API_KEY")
        if primary_key:
            self.api_keys[primary_key] = {
                "permissions": ["read", "write", "admin"],
                "created_at": time.time(),
                "last_used": None
            }
        
        # Load additional API keys
        extra_keys = os.getenv("API_KEYS_EXTRA", "")
        if extra_keys:
            for key in extra_keys.split(","):
                key = key.strip()
                if key:
                    self.api_keys[key] = {
                        "permissions": ["read", "write"],
                        "created_at": time.time(),
                        "last_used": None
                    }
    
    def verify_api_key(self, api_key: str) -> bool:
        """Verify if API key is valid."""
        if api_key in self.api_keys:
            self.api_keys[api_key]["last_used"] = time.time()
            return True
        return False
    
    def get_permissions(self, api_key: str) -> List[str]:
        """Get permissions for an API key."""
        if api_key in self.api_keys:
            return self.api_keys[api_key]["permissions"]
        return []
    
    def has_permission(self, api_key: str, permission: str) -> bool:
        """Check if API key has specific permission."""
        permissions = self.get_permissions(api_key)
        return permission in permissions or "admin" in permissions
    
    def generate_secure_api_key(self, permissions: Optional[List[str]] = None, expires_in_days: int = 30) -> Dict[str, Any]:
        """Generate a secure API key with specified permissions."""
        import secrets
        import string
        
        if permissions is None:
            permissions = ["read", "write"]
        
        # Generate a secure random API key
        alphabet = string.ascii_letters + string.digits
        api_key = "vega_" + "".join(secrets.choice(alphabet) for _ in range(32))
        
        key_data = {
            "permissions": permissions,
            "created_at": time.time(),
            "expires_at": time.time() + (expires_in_days * 24 * 3600),
            "last_used": None
        }
        
        self.api_keys[api_key] = key_data
        
        return {
            "api_key": api_key,
            "key_id": f"api_{secrets.token_hex(8)}",
            "permissions": permissions,
            "expires_at": key_data["expires_at"]
        }
    
    def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys with their metadata."""
        keys_info = []
        for api_key, data in self.api_keys.items():
            # Don't expose the actual key in listings
            key_info = {
                "key_preview": api_key[:12] + "..." if len(api_key) > 12 else api_key,
                "permissions": data["permissions"],
                "created_at": data["created_at"],
                "last_used": data["last_used"],
                "expired": False
            }
            
            # Check if key has expired
            if "expires_at" in data:
                key_info["expires_at"] = data["expires_at"]
                key_info["expired"] = time.time() > data["expires_at"]
            
            keys_info.append(key_info)
        
        return keys_info


# Global security manager instance
_security_manager = None

def get_security_manager() -> ECCAPISecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = ECCAPISecurityManager()
    return _security_manager


# FastAPI security scheme
security = HTTPBearer(auto_error=False)


async def verify_ecc_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """Verify ECC API key from Authorization header or X-API-Key header."""
    api_key = None
    
    # Try Authorization header first
    if credentials:
        api_key = credentials.credentials
    
    # Try X-API-Key header
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via Authorization header or X-API-Key header."
        )
    
    security_manager = get_security_manager()
    if not security_manager.verify_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


def require_permission(permission: str):
    """Decorator to require specific permission for an endpoint."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract API key from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(
                    status_code=500,
                    detail="Internal error: Request object not found"
                )
            
            # Get API key
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                auth_header = request.headers.get("Authorization")
                if auth_header and auth_header.startswith("Bearer "):
                    api_key = auth_header[7:]
            
            if not api_key:
                raise HTTPException(
                    status_code=401,
                    detail="API key required"
                )
            
            # Check permission
            security_manager = get_security_manager()
            if not security_manager.has_permission(api_key, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission '{permission}' required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Dependency for FastAPI endpoints
async def require_api_key(api_key: str = Depends(verify_ecc_api_key)) -> str:
    """FastAPI dependency to require valid API key."""
    return api_key


def require_read_permission(api_key: str = Depends(require_api_key)) -> str:
    """FastAPI dependency to require read permission."""
    security_manager = get_security_manager()
    if not security_manager.has_permission(api_key, "read"):
        raise HTTPException(
            status_code=403,
            detail="Read permission required"
        )
    return api_key


def require_write_permission(api_key: str = Depends(require_api_key)) -> str:
    """FastAPI dependency to require write permission."""
    security_manager = get_security_manager()
    if not security_manager.has_permission(api_key, "write"):
        raise HTTPException(
            status_code=403,
            detail="Write permission required"
        )
    return api_key


def require_admin_permission(api_key: str = Depends(require_api_key)) -> str:
    """FastAPI dependency to require admin permission."""
    security_manager = get_security_manager()
    if not security_manager.has_permission(api_key, "admin"):
        raise HTTPException(
            status_code=403,
            detail="Admin permission required"
        )
    return api_key