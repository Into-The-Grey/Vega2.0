"""
Advanced Role-Based Access Control (RBAC) System
===============================================

Comprehensive RBAC implementation with hierarchical roles, dynamic permissions,
and fine-grained access control for enterprise multi-tenant environments.

Features:
- Hierarchical role inheritance
- Dynamic permission assignment
- Resource-based access control
- Tenant-level and global permissions
- Permission caching and performance optimization
- Audit logging for access control decisions
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    import redis.asyncio as redis
except ImportError:
    import redis

logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """Permission types"""

    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"
    MANAGE = "manage"
    AUDIT = "audit"


class ResourceType(Enum):
    """Resource types"""

    USER = "user"
    ORGANIZATION = "organization"
    PROJECT = "project"
    API_KEY = "api_key"
    SUBSCRIPTION = "subscription"
    BILLING = "billing"
    ANALYTICS = "analytics"
    SYSTEM = "system"
    CUSTOM = "custom"


class AccessDecision(Enum):
    """Access control decisions"""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


@dataclass
class Permission:
    """Individual permission definition"""

    permission_id: str
    name: str
    description: str
    permission_type: PermissionType
    resource_type: ResourceType

    # Permission details
    actions: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Scope and context
    tenant_specific: bool = True
    global_permission: bool = False

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Role:
    """Role definition with hierarchical support"""

    role_id: str
    name: str
    description: str
    tenant_id: Optional[str] = None  # None for global roles

    # Role hierarchy
    parent_roles: List[str] = field(default_factory=list)
    child_roles: List[str] = field(default_factory=list)

    # Permissions
    permissions: List[str] = field(default_factory=list)  # Permission IDs
    inherited_permissions: List[str] = field(default_factory=list)

    # Role properties
    is_system_role: bool = False
    is_default_role: bool = False
    max_users: int = -1  # -1 for unlimited

    # Constraints
    conditions: Dict[str, Any] = field(default_factory=dict)
    restrictions: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class UserRoleAssignment:
    """User role assignment with context"""

    assignment_id: str
    user_id: str
    role_id: str
    tenant_id: str

    # Assignment context
    assigned_by: str
    assignment_reason: str = ""

    # Temporal constraints
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None

    # Conditional assignment
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: Optional[datetime] = None


@dataclass
class AccessPolicy:
    """Access control policy"""

    policy_id: str
    name: str
    description: str
    tenant_id: Optional[str] = None

    # Policy rules
    rules: List[Dict[str, Any]] = field(default_factory=list)
    default_decision: AccessDecision = AccessDecision.DENY

    # Policy constraints
    priority: int = 0
    enabled: bool = True

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessRequest:
    """Access control request"""

    user_id: str
    tenant_id: str
    resource_type: ResourceType
    resource_id: str
    action: str

    # Request context
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AccessLogEntry:
    """Access control audit log entry"""

    log_id: str
    user_id: str
    tenant_id: str
    resource_type: ResourceType
    resource_id: str
    action: str
    decision: AccessDecision

    # Context and reasoning
    context: Dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""
    policies_evaluated: List[str] = field(default_factory=list)

    # Timing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    response_time_ms: float = 0.0


class RBACManager:
    """Advanced RBAC management system"""

    def __init__(self, config: Any):
        self.config = config
        self.redis_client = None

        # Core RBAC data
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.role_assignments: Dict[str, List[UserRoleAssignment]] = {}
        self.access_policies: Dict[str, AccessPolicy] = {}

        # Caching
        self.permission_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Audit logging
        self.access_logs: List[AccessLogEntry] = []
        self.audit_enabled = config.get("audit_enabled", True)

        # Performance metrics
        self.access_check_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    async def initialize(self):
        """Initialize RBAC manager"""
        logger.info("Initializing RBAC manager")

        try:
            from vega.enterprise.redis_factory import create_redis_client
        except ImportError:
            from .redis_factory import create_redis_client

        # Use cluster-aware Redis client, override db for RBAC if needed
        self.redis_client = create_redis_client(self.config, db_override=1)
        # Create default permissions and roles
        await self._create_default_permissions()
        await self._create_default_roles()
        # Start background tasks
        import asyncio

        asyncio.create_task(self._permission_cache_cleanup_task())
        asyncio.create_task(self._audit_log_rotation_task())

    async def _create_default_permissions(self):
        """Create default system permissions"""

        default_permissions = [
            # User management
            Permission(
                permission_id="user.read",
                name="Read Users",
                description="View user information",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.USER,
                actions=["view", "list"],
            ),
            Permission(
                permission_id="user.write",
                name="Write Users",
                description="Create and update users",
                permission_type=PermissionType.WRITE,
                resource_type=ResourceType.USER,
                actions=["create", "update"],
            ),
            Permission(
                permission_id="user.delete",
                name="Delete Users",
                description="Delete users",
                permission_type=PermissionType.DELETE,
                resource_type=ResourceType.USER,
                actions=["delete"],
            ),
            Permission(
                permission_id="user.admin",
                name="Administer Users",
                description="Full user administration",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.USER,
                actions=["*"],
            ),
            # Organization management
            Permission(
                permission_id="organization.read",
                name="Read Organization",
                description="View organization information",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.ORGANIZATION,
                actions=["view", "list"],
            ),
            Permission(
                permission_id="organization.write",
                name="Write Organization",
                description="Update organization settings",
                permission_type=PermissionType.WRITE,
                resource_type=ResourceType.ORGANIZATION,
                actions=["update", "configure"],
            ),
            Permission(
                permission_id="organization.admin",
                name="Administer Organization",
                description="Full organization administration",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.ORGANIZATION,
                actions=["*"],
            ),
            # Billing management
            Permission(
                permission_id="billing.read",
                name="Read Billing",
                description="View billing information",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.BILLING,
                actions=["view", "download_invoices"],
            ),
            Permission(
                permission_id="billing.write",
                name="Write Billing",
                description="Update billing settings",
                permission_type=PermissionType.WRITE,
                resource_type=ResourceType.BILLING,
                actions=["update_payment_method", "change_plan"],
            ),
            Permission(
                permission_id="billing.admin",
                name="Administer Billing",
                description="Full billing administration",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.BILLING,
                actions=["*"],
            ),
            # Analytics access
            Permission(
                permission_id="analytics.read",
                name="Read Analytics",
                description="View analytics and reports",
                permission_type=PermissionType.READ,
                resource_type=ResourceType.ANALYTICS,
                actions=["view", "export"],
            ),
            Permission(
                permission_id="analytics.admin",
                name="Administer Analytics",
                description="Configure analytics and create reports",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.ANALYTICS,
                actions=["*"],
            ),
            # System administration
            Permission(
                permission_id="system.admin",
                name="System Administration",
                description="Full system administration",
                permission_type=PermissionType.ADMIN,
                resource_type=ResourceType.SYSTEM,
                actions=["*"],
                global_permission=True,
                tenant_specific=False,
            ),
        ]

        for permission in default_permissions:
            self.permissions[permission.permission_id] = permission

        logger.info(f"Created {len(default_permissions)} default permissions")

    async def _create_default_roles(self):
        """Create default system roles"""

        default_roles = [
            # Basic user role
            Role(
                role_id="user",
                name="User",
                description="Basic user with read access",
                permissions=["user.read"],
                is_system_role=True,
                is_default_role=True,
            ),
            # Developer role
            Role(
                role_id="developer",
                name="Developer",
                description="Developer with API access",
                permissions=[
                    "user.read",
                    "user.write",
                    "organization.read",
                    "analytics.read",
                ],
                is_system_role=True,
            ),
            # Admin role
            Role(
                role_id="admin",
                name="Administrator",
                description="Organization administrator",
                permissions=[
                    "user.read",
                    "user.write",
                    "user.delete",
                    "organization.read",
                    "organization.write",
                    "billing.read",
                    "billing.write",
                    "analytics.read",
                ],
                is_system_role=True,
            ),
            # Super admin role
            Role(
                role_id="super_admin",
                name="Super Administrator",
                description="Full access administrator",
                parent_roles=["admin"],
                permissions=[
                    "user.admin",
                    "organization.admin",
                    "billing.admin",
                    "analytics.admin",
                ],
                is_system_role=True,
            ),
            # Billing admin role
            Role(
                role_id="billing_admin",
                name="Billing Administrator",
                description="Billing and subscription management",
                permissions=[
                    "user.read",
                    "organization.read",
                    "billing.admin",
                    "analytics.read",
                ],
                is_system_role=True,
            ),
            # System admin role (global)
            Role(
                role_id="system_admin",
                name="System Administrator",
                description="Global system administration",
                permissions=["system.admin"],
                is_system_role=True,
                tenant_id=None,  # Global role
            ),
        ]

        for role in default_roles:
            self.roles[role.role_id] = role
            await self._resolve_role_inheritance(role)

        logger.info(f"Created {len(default_roles)} default roles")

    async def _resolve_role_inheritance(self, role: Role):
        """Resolve role inheritance and merge permissions"""

        inherited_permissions = set(role.permissions)

        # Recursively collect permissions from parent roles
        for parent_role_id in role.parent_roles:
            parent_role = self.roles.get(parent_role_id)
            if parent_role:
                await self._resolve_role_inheritance(parent_role)
                inherited_permissions.update(parent_role.inherited_permissions)
                inherited_permissions.update(parent_role.permissions)

        role.inherited_permissions = list(inherited_permissions)

    async def create_permission(
        self,
        name: str,
        description: str,
        permission_type: PermissionType,
        resource_type: ResourceType,
        actions: List[str],
        tenant_id: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Permission:
        """Create a new permission"""

        permission_id = (
            f"{resource_type.value}.{permission_type.value}.{uuid.uuid4().hex[:8]}"
        )

        permission = Permission(
            permission_id=permission_id,
            name=name,
            description=description,
            permission_type=permission_type,
            resource_type=resource_type,
            actions=actions,
            conditions=conditions or {},
            tenant_specific=tenant_id is not None,
        )

        self.permissions[permission_id] = permission

        logger.info(f"Created permission: {permission_id}")

        return permission

    async def create_role(
        self,
        name: str,
        description: str,
        permissions: List[str],
        tenant_id: Optional[str] = None,
        parent_roles: Optional[List[str]] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> Role:
        """Create a new role"""

        role_id = f"{name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

        role = Role(
            role_id=role_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            permissions=permissions,
            parent_roles=parent_roles or [],
            conditions=conditions or {},
        )

        # Validate permissions exist
        for perm_id in permissions:
            if perm_id not in self.permissions:
                raise ValueError(f"Permission not found: {perm_id}")

        # Update parent roles' child relationships
        for parent_role_id in role.parent_roles:
            parent_role = self.roles.get(parent_role_id)
            if parent_role:
                if role_id not in parent_role.child_roles:
                    parent_role.child_roles.append(role_id)

        self.roles[role_id] = role

        # Resolve inheritance
        await self._resolve_role_inheritance(role)

        logger.info(f"Created role: {role_id}")

        return role

    async def assign_role_to_user(
        self,
        user_id: str,
        role_id: str,
        tenant_id: str,
        assigned_by: str,
        assignment_reason: str = "",
        valid_until: Optional[datetime] = None,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> UserRoleAssignment:
        """Assign role to user"""

        # Validate role exists
        role = self.roles.get(role_id)
        if not role:
            raise ValueError(f"Role not found: {role_id}")

        # Check if role is tenant-specific
        if role.tenant_id and role.tenant_id != tenant_id:
            raise ValueError(f"Role {role_id} is not available for tenant {tenant_id}")

        assignment_id = str(uuid.uuid4())

        assignment = UserRoleAssignment(
            assignment_id=assignment_id,
            user_id=user_id,
            role_id=role_id,
            tenant_id=tenant_id,
            assigned_by=assigned_by,
            assignment_reason=assignment_reason,
            valid_until=valid_until,
            conditions=conditions or {},
        )

        # Store assignment
        if user_id not in self.role_assignments:
            self.role_assignments[user_id] = []

        self.role_assignments[user_id].append(assignment)

        # Clear user permission cache
        await self._clear_user_permission_cache(user_id, tenant_id)

        logger.info(f"Assigned role {role_id} to user {user_id} in tenant {tenant_id}")

        return assignment

    async def revoke_role_from_user(
        self, user_id: str, role_id: str, tenant_id: str
    ) -> bool:
        """Revoke role from user"""

        user_assignments = self.role_assignments.get(user_id, [])

        for i, assignment in enumerate(user_assignments):
            if assignment.role_id == role_id and assignment.tenant_id == tenant_id:

                user_assignments.pop(i)

                # Clear user permission cache
                await self._clear_user_permission_cache(user_id, tenant_id)

                logger.info(
                    f"Revoked role {role_id} from user {user_id} in tenant {tenant_id}"
                )

                return True

        return False

    async def check_permission(
        self,
        user_id: str,
        tenant_id: str,
        resource_type: ResourceType,
        resource_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Check if user has permission for action"""

        start_time = datetime.now()
        self.access_check_count += 1

        # Create access request
        access_request = AccessRequest(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            context=context or {},
        )

        # Check cache first
        cache_key = f"{user_id}:{tenant_id}:{resource_type.value}:{action}"
        cached_result = await self._get_cached_permission(cache_key)

        if cached_result is not None:
            self.cache_hits += 1
            decision = cached_result["allowed"]
            reason = cached_result["reason"]
        else:
            self.cache_misses += 1
            decision, reason = await self._evaluate_permission(access_request)

            # Cache result
            await self._cache_permission_result(cache_key, decision, reason)

        # Log access decision
        if self.audit_enabled:
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            await self._log_access_decision(
                access_request, decision, reason, response_time
            )

        return decision, reason

    async def _evaluate_permission(
        self, access_request: AccessRequest
    ) -> Tuple[bool, str]:
        """Evaluate permission request"""

        user_id = access_request.user_id
        tenant_id = access_request.tenant_id
        resource_type = access_request.resource_type
        action = access_request.action

        # Get user's roles for this tenant
        user_roles = await self._get_user_roles(user_id, tenant_id)

        if not user_roles:
            return False, "No roles assigned"

        # Collect all permissions from roles
        user_permissions = set()

        for role_assignment in user_roles:
            # Check if assignment is still valid
            if not await self._is_assignment_valid(role_assignment):
                continue

            role = self.roles.get(role_assignment.role_id)
            if not role:
                continue

            # Add role permissions
            user_permissions.update(role.inherited_permissions)

        # Check if user has required permission
        required_permissions = [
            f"{resource_type.value}.{action}",
            f"{resource_type.value}.admin",
            f"{resource_type.value}.manage",
            "system.admin",
        ]

        for perm_id in required_permissions:
            if perm_id in user_permissions:
                permission = self.permissions.get(perm_id)
                if permission:
                    # Check permission conditions
                    if await self._check_permission_conditions(
                        permission, access_request
                    ):
                        return True, f"Granted via permission: {perm_id}"

        # Check wildcard permissions
        wildcard_permissions = [
            p for p in user_permissions if p.endswith(".*") or p.endswith(".admin")
        ]

        for perm_id in wildcard_permissions:
            permission = self.permissions.get(perm_id)
            if permission and permission.resource_type == resource_type:
                if await self._check_permission_conditions(permission, access_request):
                    return True, f"Granted via wildcard permission: {perm_id}"

        return False, "Insufficient permissions"

    async def _get_user_roles(
        self, user_id: str, tenant_id: str
    ) -> List[UserRoleAssignment]:
        """Get user's role assignments for tenant"""

        user_assignments = self.role_assignments.get(user_id, [])

        # Filter by tenant and include global roles
        relevant_assignments = []

        for assignment in user_assignments:
            if assignment.tenant_id == tenant_id:
                relevant_assignments.append(assignment)
            else:
                # Check if it's a global role
                role = self.roles.get(assignment.role_id)
                if role and role.tenant_id is None:
                    relevant_assignments.append(assignment)

        return relevant_assignments

    async def _is_assignment_valid(self, assignment: UserRoleAssignment) -> bool:
        """Check if role assignment is still valid"""

        now = datetime.now(timezone.utc)

        # Check temporal validity
        if assignment.valid_until and assignment.valid_until < now:
            return False

        if assignment.valid_from > now:
            return False

        # Check conditional validity
        if assignment.conditions:
            # Evaluate conditions (simplified for demo)
            # In production, this would evaluate complex conditions
            pass

        return True

    async def _check_permission_conditions(
        self, permission: Permission, access_request: AccessRequest
    ) -> bool:
        """Check permission-specific conditions"""

        if not permission.conditions:
            return True

        # Evaluate conditions based on context
        # This is simplified - in production, you'd have a full condition engine

        context = access_request.context

        for condition_key, condition_value in permission.conditions.items():
            if condition_key == "time_of_day":
                current_hour = datetime.now().hour
                if not (
                    condition_value["start"] <= current_hour <= condition_value["end"]
                ):
                    return False

            elif condition_key == "ip_whitelist":
                client_ip = context.get("client_ip")
                if client_ip not in condition_value:
                    return False

            elif condition_key == "resource_owner":
                if context.get("resource_owner_id") != access_request.user_id:
                    return False

        return True

    async def _get_cached_permission(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached permission result"""

        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        try:
            cached_data = await self.redis_client.get(f"rbac:perm:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Error getting cached permission: {e}")
        return None

    async def _cache_permission_result(
        self, cache_key: str, allowed: bool, reason: str
    ):
        """Cache permission result"""

        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        try:
            cache_data = {
                "allowed": allowed,
                "reason": reason,
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            await self.redis_client.setex(
                f"rbac:perm:{cache_key}", self.cache_ttl, json.dumps(cache_data)
            )
        except Exception as e:
            logger.error(f"Error caching permission result: {e}")

    async def _clear_user_permission_cache(self, user_id: str, tenant_id: str):
        """Clear permission cache for user"""

        if not self.redis_client:
            raise RuntimeError(
                "Redis client is not initialized. Call initialize() first."
            )
        try:
            # Find all cache keys for this user
            pattern = f"rbac:perm:{user_id}:{tenant_id}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        except Exception as e:
            logger.error(f"Error clearing user permission cache: {e}")

    async def _log_access_decision(
        self,
        access_request: AccessRequest,
        decision: bool,
        reason: str,
        response_time_ms: float,
    ):
        """Log access control decision"""

        log_entry = AccessLogEntry(
            log_id=str(uuid.uuid4()),
            user_id=access_request.user_id,
            tenant_id=access_request.tenant_id,
            resource_type=access_request.resource_type,
            resource_id=access_request.resource_id,
            action=access_request.action,
            decision=AccessDecision.ALLOW if decision else AccessDecision.DENY,
            context=access_request.context,
            decision_reason=reason,
            response_time_ms=response_time_ms,
        )

        self.access_logs.append(log_entry)

        # Keep only recent logs in memory
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-5000:]

    async def get_user_permissions(
        self, user_id: str, tenant_id: str
    ) -> Dict[str, List[str]]:
        """Get all permissions for user organized by resource type"""

        user_roles = await self._get_user_roles(user_id, tenant_id)

        permissions_by_resource = {}

        for role_assignment in user_roles:
            if not await self._is_assignment_valid(role_assignment):
                continue

            role = self.roles.get(role_assignment.role_id)
            if not role:
                continue

            for perm_id in role.inherited_permissions:
                permission = self.permissions.get(perm_id)
                if permission:
                    resource_type = permission.resource_type.value

                    if resource_type not in permissions_by_resource:
                        permissions_by_resource[resource_type] = []

                    permissions_by_resource[resource_type].extend(permission.actions)

        # Remove duplicates
        for resource_type in permissions_by_resource:
            permissions_by_resource[resource_type] = list(
                set(permissions_by_resource[resource_type])
            )

        return permissions_by_resource

    async def get_role_hierarchy(
        self, tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get role hierarchy for tenant"""

        # Filter roles by tenant
        tenant_roles = {}

        for role_id, role in self.roles.items():
            if (
                tenant_id is None
                or role.tenant_id is None
                or role.tenant_id == tenant_id
            ):
                tenant_roles[role_id] = role

        # Build hierarchy
        hierarchy = {}

        for role_id, role in tenant_roles.items():
            hierarchy[role_id] = {
                "name": role.name,
                "description": role.description,
                "parent_roles": role.parent_roles,
                "child_roles": role.child_roles,
                "permissions": role.inherited_permissions,
                "is_system_role": role.is_system_role,
            }

        return hierarchy

    async def get_access_logs(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
        limit: int = 100,
    ) -> List[AccessLogEntry]:
        """Get access logs with filtering"""

        filtered_logs = self.access_logs

        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]

        if tenant_id:
            filtered_logs = [log for log in filtered_logs if log.tenant_id == tenant_id]

        if resource_type:
            filtered_logs = [
                log for log in filtered_logs if log.resource_type == resource_type
            ]

        # Sort by timestamp (most recent first)
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)

        return filtered_logs[:limit]

    async def get_rbac_metrics(self) -> Dict[str, Any]:
        """Get RBAC system metrics"""

        return {
            "permissions": {
                "total": len(self.permissions),
                "by_type": {
                    ptype.value: len(
                        [
                            p
                            for p in self.permissions.values()
                            if p.permission_type == ptype
                        ]
                    )
                    for ptype in PermissionType
                },
            },
            "roles": {
                "total": len(self.roles),
                "system_roles": len(
                    [r for r in self.roles.values() if r.is_system_role]
                ),
                "custom_roles": len(
                    [r for r in self.roles.values() if not r.is_system_role]
                ),
            },
            "assignments": {
                "total": sum(
                    len(assignments) for assignments in self.role_assignments.values()
                ),
                "users_with_roles": len(self.role_assignments),
            },
            "performance": {
                "access_checks": self.access_check_count,
                "cache_hit_rate": self.cache_hits
                / max(self.access_check_count, 1)
                * 100,
                "cache_miss_rate": self.cache_misses
                / max(self.access_check_count, 1)
                * 100,
            },
            "audit": {
                "total_logs": len(self.access_logs),
                "allowed_decisions": len(
                    [
                        log
                        for log in self.access_logs
                        if log.decision == AccessDecision.ALLOW
                    ]
                ),
                "denied_decisions": len(
                    [
                        log
                        for log in self.access_logs
                        if log.decision == AccessDecision.DENY
                    ]
                ),
            },
        }

    async def _permission_cache_cleanup_task(self):
        """Background task to clean up expired permission cache"""

        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Redis TTL handles cache expiration automatically
                # This task could be used for additional cleanup if needed

            except Exception as e:
                logger.error(f"Error in permission cache cleanup task: {e}")

    async def _audit_log_rotation_task(self):
        """Background task to rotate audit logs"""

        while True:
            try:
                await asyncio.sleep(86400)  # Run daily

                # Keep only last 30 days of logs
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

                self.access_logs = [
                    log for log in self.access_logs if log.timestamp > cutoff_date
                ]

                logger.info(
                    f"Rotated audit logs, kept {len(self.access_logs)} recent entries"
                )

            except Exception as e:
                logger.error(f"Error in audit log rotation task: {e}")


def require_permission(
    resource_type: ResourceType, action: str, resource_id_param: str = "resource_id"
):
    """Decorator for requiring specific permissions"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request and user context from function arguments
            # This would be integrated with your FastAPI dependencies

            # For demo purposes, assume we have access to the RBAC manager
            # and user context through dependency injection

            # rbac_manager = get_rbac_manager()
            # user_id = get_current_user_id()
            # tenant_id = get_current_tenant_id()
            # resource_id = kwargs.get(resource_id_param, "")

            # allowed, reason = await rbac_manager.check_permission(
            #     user_id, tenant_id, resource_type, resource_id, action
            # )

            # if not allowed:
            #     raise HTTPException(status_code=403, detail=f"Access denied: {reason}")

            return await func(*args, **kwargs)

        return wrapper

    return decorator
