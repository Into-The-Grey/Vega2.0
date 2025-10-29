"""
Personal Access Control System

Provides personal permission management for organizing access to different
workspace features and data in a single-user environment.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib
import fnmatch

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for personal data and features"""

    FULL = "full"  # Complete access
    READ_WRITE = "read_write"  # Read and modify
    READ_ONLY = "read_only"  # View only
    LIMITED = "limited"  # Restricted access
    DENIED = "denied"  # No access


class ResourceType(Enum):
    """Types of resources that can be controlled"""

    WORKSPACE = "workspace"
    DOCUMENT = "document"
    MEDIA = "media"
    ANALYTICS = "analytics"
    TRAINING = "training"
    API = "api"
    FEATURE = "feature"
    DATA = "data"
    EXPORT = "export"
    ADMIN = "admin"


class PermissionScope(Enum):
    """Scope of permissions"""

    GLOBAL = "global"  # System-wide permissions
    WORKSPACE = "workspace"  # Workspace-specific
    SESSION = "session"  # Session-specific
    TEMPORARY = "temporary"  # Time-limited


@dataclass
class Permission:
    """Individual permission definition"""

    resource_type: ResourceType
    resource_id: str  # Specific resource ID or "*" for all
    access_level: AccessLevel
    scope: PermissionScope
    granted_at: datetime
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_expired(self) -> bool:
        """Check if permission is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def matches_resource(self, resource_type: ResourceType, resource_id: str) -> bool:
        """Check if permission matches given resource"""
        if self.resource_type != resource_type:
            return False

        if self.resource_id == "*":
            return True

        # Support wildcard patterns
        return fnmatch.fnmatch(resource_id, self.resource_id)


@dataclass
class AccessProfile:
    """Access profile for organizing permissions"""

    profile_id: str
    name: str
    description: str
    permissions: List[Permission]
    is_active: bool = True
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class AccessRequest:
    """Request for resource access"""

    resource_type: ResourceType
    resource_id: str
    requested_access: AccessLevel
    context: Dict[str, Any] = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class AccessDecision:
    """Result of access control decision"""

    granted: bool
    access_level: AccessLevel
    reason: str
    applicable_permissions: List[Permission]
    conditions: Dict[str, Any] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


class PersonalAccessController:
    """
    Personal access control system for managing permissions
    in a single-user environment with multiple access contexts
    """

    def __init__(self, config_file: str = "config/access_control.json"):
        self.config_file = config_file
        self.profiles: Dict[str, AccessProfile] = {}
        self.active_profile: Optional[str] = None
        self.global_permissions: List[Permission] = []
        self.session_permissions: Dict[str, List[Permission]] = {}

        # Built-in access profiles
        self._initialize_default_profiles()

    def _initialize_default_profiles(self):
        """Initialize default access profiles"""

        # Full admin profile
        admin_permissions = [
            Permission(
                resource_type=ResourceType.ADMIN,
                resource_id="*",
                access_level=AccessLevel.FULL,
                scope=PermissionScope.GLOBAL,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.API,
                resource_id="*",
                access_level=AccessLevel.FULL,
                scope=PermissionScope.GLOBAL,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.DATA,
                resource_id="*",
                access_level=AccessLevel.FULL,
                scope=PermissionScope.GLOBAL,
                granted_at=datetime.now(),
            ),
        ]

        admin_profile = AccessProfile(
            profile_id="admin",
            name="Administrator",
            description="Full system access",
            permissions=admin_permissions,
        )

        # Personal workspace profile
        workspace_permissions = [
            Permission(
                resource_type=ResourceType.WORKSPACE,
                resource_id="*",
                access_level=AccessLevel.FULL,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.DOCUMENT,
                resource_id="*",
                access_level=AccessLevel.READ_WRITE,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.MEDIA,
                resource_id="*",
                access_level=AccessLevel.READ_WRITE,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.ANALYTICS,
                resource_id="personal_*",
                access_level=AccessLevel.READ_ONLY,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
        ]

        workspace_profile = AccessProfile(
            profile_id="workspace",
            name="Personal Workspace",
            description="Standard workspace access",
            permissions=workspace_permissions,
        )

        # Read-only profile
        readonly_permissions = [
            Permission(
                resource_type=ResourceType.DOCUMENT,
                resource_id="*",
                access_level=AccessLevel.READ_ONLY,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.MEDIA,
                resource_id="*",
                access_level=AccessLevel.READ_ONLY,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.ANALYTICS,
                resource_id="*",
                access_level=AccessLevel.READ_ONLY,
                scope=PermissionScope.WORKSPACE,
                granted_at=datetime.now(),
            ),
        ]

        readonly_profile = AccessProfile(
            profile_id="readonly",
            name="Read Only",
            description="View-only access to personal data",
            permissions=readonly_permissions,
        )

        # Limited profile for restricted contexts
        limited_permissions = [
            Permission(
                resource_type=ResourceType.DOCUMENT,
                resource_id="public_*",
                access_level=AccessLevel.READ_ONLY,
                scope=PermissionScope.SESSION,
                granted_at=datetime.now(),
            ),
            Permission(
                resource_type=ResourceType.FEATURE,
                resource_id="basic_*",
                access_level=AccessLevel.LIMITED,
                scope=PermissionScope.SESSION,
                granted_at=datetime.now(),
            ),
        ]

        limited_profile = AccessProfile(
            profile_id="limited",
            name="Limited Access",
            description="Restricted access for specific contexts",
            permissions=limited_permissions,
        )

        # Register profiles
        for profile in [
            admin_profile,
            workspace_profile,
            readonly_profile,
            limited_profile,
        ]:
            self.profiles[profile.profile_id] = profile

        # Set default active profile
        self.active_profile = "workspace"

    async def load_configuration(self):
        """Load access control configuration from file"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                async with config_path.open("r") as f:
                    config_data = json.loads(await f.read())

                # Load custom profiles
                for profile_data in config_data.get("profiles", []):
                    profile = self._deserialize_profile(profile_data)
                    self.profiles[profile.profile_id] = profile

                # Load global permissions
                for perm_data in config_data.get("global_permissions", []):
                    permission = self._deserialize_permission(perm_data)
                    self.global_permissions.append(permission)

                # Set active profile
                if "active_profile" in config_data:
                    self.active_profile = config_data["active_profile"]

                logger.info(
                    f"Loaded access control configuration from {self.config_file}"
                )

        except Exception as e:
            logger.error(f"Failed to load access control config: {e}")
            await self._save_configuration()  # Create default config

    async def _save_configuration(self):
        """Save current configuration to file"""
        config_data = {
            "active_profile": self.active_profile,
            "profiles": [self._serialize_profile(p) for p in self.profiles.values()],
            "global_permissions": [
                self._serialize_permission(p) for p in self.global_permissions
            ],
            "last_updated": datetime.now().isoformat(),
        }

        config_path = Path(self.config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        async with config_path.open("w") as f:
            await f.write(json.dumps(config_data, indent=2))

        logger.info(f"Saved access control configuration to {self.config_file}")

    def _serialize_profile(self, profile: AccessProfile) -> Dict[str, Any]:
        """Serialize access profile to dictionary"""
        return {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "description": profile.description,
            "permissions": [self._serialize_permission(p) for p in profile.permissions],
            "is_active": profile.is_active,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }

    def _deserialize_profile(self, data: Dict[str, Any]) -> AccessProfile:
        """Deserialize access profile from dictionary"""
        permissions = [
            self._deserialize_permission(p) for p in data.get("permissions", [])
        ]

        return AccessProfile(
            profile_id=data["profile_id"],
            name=data["name"],
            description=data["description"],
            permissions=permissions,
            is_active=data.get("is_active", True),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def _serialize_permission(self, permission: Permission) -> Dict[str, Any]:
        """Serialize permission to dictionary"""
        return {
            "resource_type": permission.resource_type.value,
            "resource_id": permission.resource_id,
            "access_level": permission.access_level.value,
            "scope": permission.scope.value,
            "granted_at": permission.granted_at.isoformat(),
            "expires_at": (
                permission.expires_at.isoformat() if permission.expires_at else None
            ),
            "conditions": permission.conditions,
            "metadata": permission.metadata,
        }

    def _deserialize_permission(self, data: Dict[str, Any]) -> Permission:
        """Deserialize permission from dictionary"""
        return Permission(
            resource_type=ResourceType(data["resource_type"]),
            resource_id=data["resource_id"],
            access_level=AccessLevel(data["access_level"]),
            scope=PermissionScope(data["scope"]),
            granted_at=datetime.fromisoformat(data["granted_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            conditions=data.get("conditions", {}),
            metadata=data.get("metadata", {}),
        )

    def check_access(
        self, request: AccessRequest, session_id: Optional[str] = None
    ) -> AccessDecision:
        """Check if access should be granted for a request"""

        applicable_permissions = []
        highest_access = AccessLevel.DENIED
        reasons = []

        # Check global permissions
        for permission in self.global_permissions:
            if permission.matches_resource(request.resource_type, request.resource_id):
                if not permission.is_expired:
                    applicable_permissions.append(permission)
                    if self._access_level_rank(
                        permission.access_level
                    ) > self._access_level_rank(highest_access):
                        highest_access = permission.access_level
                        reasons.append(
                            f"Global permission: {permission.access_level.value}"
                        )

        # Check active profile permissions
        if self.active_profile and self.active_profile in self.profiles:
            profile = self.profiles[self.active_profile]
            if profile.is_active:
                for permission in profile.permissions:
                    if permission.matches_resource(
                        request.resource_type, request.resource_id
                    ):
                        if not permission.is_expired:
                            applicable_permissions.append(permission)
                            if self._access_level_rank(
                                permission.access_level
                            ) > self._access_level_rank(highest_access):
                                highest_access = permission.access_level
                                reasons.append(
                                    f"Profile '{profile.name}': {permission.access_level.value}"
                                )

        # Check session-specific permissions
        if session_id and session_id in self.session_permissions:
            for permission in self.session_permissions[session_id]:
                if permission.matches_resource(
                    request.resource_type, request.resource_id
                ):
                    if not permission.is_expired:
                        applicable_permissions.append(permission)
                        if self._access_level_rank(
                            permission.access_level
                        ) > self._access_level_rank(highest_access):
                            highest_access = permission.access_level
                            reasons.append(
                                f"Session permission: {permission.access_level.value}"
                            )

        # Determine if access is granted
        requested_rank = self._access_level_rank(request.requested_access)
        granted_rank = self._access_level_rank(highest_access)
        granted = granted_rank >= requested_rank

        reason = "; ".join(reasons) if reasons else "No applicable permissions"

        return AccessDecision(
            granted=granted,
            access_level=highest_access,
            reason=reason,
            applicable_permissions=applicable_permissions,
        )

    def _access_level_rank(self, access_level: AccessLevel) -> int:
        """Get numeric rank for access level comparison"""
        ranks = {
            AccessLevel.DENIED: 0,
            AccessLevel.LIMITED: 1,
            AccessLevel.READ_ONLY: 2,
            AccessLevel.READ_WRITE: 3,
            AccessLevel.FULL: 4,
        }
        return ranks.get(access_level, 0)

    def grant_permission(
        self, permission: Permission, session_id: Optional[str] = None
    ) -> bool:
        """Grant a specific permission"""
        try:
            if permission.scope == PermissionScope.GLOBAL:
                self.global_permissions.append(permission)
            elif permission.scope == PermissionScope.SESSION and session_id:
                if session_id not in self.session_permissions:
                    self.session_permissions[session_id] = []
                self.session_permissions[session_id].append(permission)
            else:
                # Add to active profile
                if self.active_profile and self.active_profile in self.profiles:
                    self.profiles[self.active_profile].permissions.append(permission)
                    self.profiles[self.active_profile].updated_at = datetime.now()

            logger.info(
                f"Granted permission: {permission.resource_type.value}:{permission.resource_id} -> {permission.access_level.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to grant permission: {e}")
            return False

    def revoke_permission(
        self,
        resource_type: ResourceType,
        resource_id: str,
        session_id: Optional[str] = None,
    ) -> bool:
        """Revoke permissions for a specific resource"""
        revoked = False

        # Remove from global permissions
        self.global_permissions = [
            p
            for p in self.global_permissions
            if not p.matches_resource(resource_type, resource_id)
        ]

        # Remove from active profile
        if self.active_profile and self.active_profile in self.profiles:
            profile = self.profiles[self.active_profile]
            original_count = len(profile.permissions)
            profile.permissions = [
                p
                for p in profile.permissions
                if not p.matches_resource(resource_type, resource_id)
            ]
            if len(profile.permissions) < original_count:
                profile.updated_at = datetime.now()
                revoked = True

        # Remove from session permissions
        if session_id and session_id in self.session_permissions:
            original_count = len(self.session_permissions[session_id])
            self.session_permissions[session_id] = [
                p
                for p in self.session_permissions[session_id]
                if not p.matches_resource(resource_type, resource_id)
            ]
            if len(self.session_permissions[session_id]) < original_count:
                revoked = True

        if revoked:
            logger.info(f"Revoked permissions for {resource_type.value}:{resource_id}")

        return revoked

    def set_active_profile(self, profile_id: str) -> bool:
        """Set the active access profile"""
        if profile_id in self.profiles:
            self.active_profile = profile_id
            logger.info(f"Set active profile to '{profile_id}'")
            return True
        return False

    def create_temporary_permission(
        self,
        resource_type: ResourceType,
        resource_id: str,
        access_level: AccessLevel,
        duration_minutes: int,
        session_id: str,
    ) -> str:
        """Create a temporary permission"""
        expires_at = datetime.now() + timedelta(minutes=duration_minutes)

        permission = Permission(
            resource_type=resource_type,
            resource_id=resource_id,
            access_level=access_level,
            scope=PermissionScope.TEMPORARY,
            granted_at=datetime.now(),
            expires_at=expires_at,
            metadata={"duration_minutes": duration_minutes},
        )

        self.grant_permission(permission, session_id)

        permission_id = hashlib.sha256(
            f"{resource_type.value}:{resource_id}:{session_id}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        logger.info(
            f"Created temporary permission {permission_id} (expires in {duration_minutes} minutes)"
        )
        return permission_id

    def cleanup_expired_permissions(self):
        """Remove expired permissions"""
        now = datetime.now()

        # Clean global permissions
        original_count = len(self.global_permissions)
        self.global_permissions = [
            p for p in self.global_permissions if not p.is_expired
        ]
        cleaned_global = original_count - len(self.global_permissions)

        # Clean profile permissions
        cleaned_profile = 0
        for profile in self.profiles.values():
            original_count = len(profile.permissions)
            profile.permissions = [p for p in profile.permissions if not p.is_expired]
            cleaned_profile += original_count - len(profile.permissions)

        # Clean session permissions
        cleaned_session = 0
        for session_id in list(self.session_permissions.keys()):
            original_count = len(self.session_permissions[session_id])
            self.session_permissions[session_id] = [
                p for p in self.session_permissions[session_id] if not p.is_expired
            ]
            cleaned_session += original_count - len(
                self.session_permissions[session_id]
            )

            # Remove empty session permission lists
            if not self.session_permissions[session_id]:
                del self.session_permissions[session_id]

        total_cleaned = cleaned_global + cleaned_profile + cleaned_session
        if total_cleaned > 0:
            logger.info(f"Cleaned up {total_cleaned} expired permissions")

    def get_permissions_summary(self) -> Dict[str, Any]:
        """Get summary of current permissions"""
        summary = {
            "active_profile": self.active_profile,
            "profiles": {
                pid: {
                    "name": profile.name,
                    "permission_count": len(profile.permissions),
                    "is_active": profile.is_active,
                }
                for pid, profile in self.profiles.items()
            },
            "global_permissions": len(self.global_permissions),
            "session_permissions": {
                sid: len(perms) for sid, perms in self.session_permissions.items()
            },
            "last_updated": datetime.now().isoformat(),
        }
        return summary


# Decorator for access control
def require_access(
    resource_type: ResourceType,
    resource_id: str = "*",
    access_level: AccessLevel = AccessLevel.READ_ONLY,
):
    """Decorator to require specific access for function execution"""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # Get access controller from context (would be injected in real app)
            access_controller = kwargs.get("_access_controller")
            session_id = kwargs.get("_session_id")

            if access_controller:
                request = AccessRequest(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    requested_access=access_level,
                )

                decision = access_controller.check_access(request, session_id)
                if not decision.granted:
                    raise PermissionError(f"Access denied: {decision.reason}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Demo and testing functions
async def demo_personal_access_control():
    """Demonstrate personal access control capabilities"""

    controller = PersonalAccessController()
    await controller.load_configuration()

    print("Personal Access Control Demo")
    print(f"Active profile: {controller.active_profile}")
    print(f"Available profiles: {list(controller.profiles.keys())}")

    # Test access requests
    test_requests = [
        AccessRequest(ResourceType.DOCUMENT, "my_document", AccessLevel.READ_WRITE),
        AccessRequest(ResourceType.ANALYTICS, "personal_stats", AccessLevel.READ_ONLY),
        AccessRequest(ResourceType.ADMIN, "system_config", AccessLevel.FULL),
        AccessRequest(ResourceType.TRAINING, "model_data", AccessLevel.READ_WRITE),
    ]

    print("\nAccess Test Results:")
    for request in test_requests:
        decision = controller.check_access(request)
        status = "✓ GRANTED" if decision.granted else "✗ DENIED"
        print(
            f"{status}: {request.resource_type.value}:{request.resource_id} -> {request.requested_access.value}"
        )
        print(f"   Reason: {decision.reason}")

    # Test temporary permission
    temp_perm_id = controller.create_temporary_permission(
        ResourceType.EXPORT, "data_export", AccessLevel.READ_WRITE, 30, "demo_session"
    )
    print(f"\nCreated temporary permission: {temp_perm_id}")

    # Get summary
    summary = controller.get_permissions_summary()
    print(f"\nPermissions Summary:")
    print(f"- Global permissions: {summary['global_permissions']}")
    print(f"- Active sessions: {len(summary['session_permissions'])}")

    return controller


if __name__ == "__main__":
    asyncio.run(demo_personal_access_control())
