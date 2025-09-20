"""
Advanced workspace management and permissions system
===================================================

This module provides advanced workspace features including
permissions, access control, and role-based management.
"""

from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json


class Permission(Enum):
    """Workspace permissions"""

    READ = "read"
    WRITE = "write"
    EDIT_DOCS = "edit_docs"
    MANAGE_USERS = "manage_users"
    ADMIN = "admin"
    CREATE_DOCS = "create_docs"
    DELETE_DOCS = "delete_docs"
    VOICE_VIDEO = "voice_video"
    MODERATE = "moderate"


class WorkspaceRole(Enum):
    """Predefined workspace roles"""

    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    EDITOR = "editor"
    VIEWER = "viewer"
    GUEST = "guest"


@dataclass
class RolePermissions:
    """Permission set for a role"""

    role: WorkspaceRole
    permissions: Set[Permission]

    @classmethod
    def get_default_permissions(cls, role: WorkspaceRole) -> Set[Permission]:
        """Get default permissions for a role"""
        defaults = {
            WorkspaceRole.OWNER: {
                Permission.READ,
                Permission.WRITE,
                Permission.EDIT_DOCS,
                Permission.MANAGE_USERS,
                Permission.ADMIN,
                Permission.CREATE_DOCS,
                Permission.DELETE_DOCS,
                Permission.VOICE_VIDEO,
                Permission.MODERATE,
            },
            WorkspaceRole.ADMIN: {
                Permission.READ,
                Permission.WRITE,
                Permission.EDIT_DOCS,
                Permission.MANAGE_USERS,
                Permission.CREATE_DOCS,
                Permission.DELETE_DOCS,
                Permission.VOICE_VIDEO,
                Permission.MODERATE,
            },
            WorkspaceRole.MODERATOR: {
                Permission.READ,
                Permission.WRITE,
                Permission.EDIT_DOCS,
                Permission.CREATE_DOCS,
                Permission.VOICE_VIDEO,
                Permission.MODERATE,
            },
            WorkspaceRole.EDITOR: {
                Permission.READ,
                Permission.WRITE,
                Permission.EDIT_DOCS,
                Permission.CREATE_DOCS,
                Permission.VOICE_VIDEO,
            },
            WorkspaceRole.VIEWER: {Permission.READ, Permission.VOICE_VIDEO},
            WorkspaceRole.GUEST: {Permission.READ},
        }
        return defaults.get(role, set())


@dataclass
class UserPermissions:
    """User-specific permissions in a workspace"""

    user_id: str
    workspace_id: str
    role: WorkspaceRole
    custom_permissions: Set[Permission] = field(default_factory=set)
    granted_at: datetime = field(default_factory=datetime.now)
    granted_by: Optional[str] = None
    expires_at: Optional[datetime] = None

    def get_effective_permissions(self) -> Set[Permission]:
        """Get effective permissions (role + custom)"""
        role_perms = RolePermissions.get_default_permissions(self.role)
        return role_perms.union(self.custom_permissions)

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission"""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return permission in self.get_effective_permissions()

    def is_expired(self) -> bool:
        """Check if permissions are expired"""
        return self.expires_at is not None and datetime.now() > self.expires_at


@dataclass
class WorkspaceSettings:
    """Workspace configuration and settings"""

    workspace_id: str
    name: str
    description: str = ""
    is_public: bool = False
    require_approval: bool = True
    max_participants: int = 50
    allow_guest_access: bool = False
    default_role: WorkspaceRole = WorkspaceRole.VIEWER
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""

    # Document settings
    allow_anonymous_editing: bool = False
    require_edit_approval: bool = False
    max_document_size_mb: int = 10

    # Voice/Video settings
    enable_voice: bool = True
    enable_video: bool = True
    max_voice_participants: int = 10
    max_video_participants: int = 5

    # Moderation settings
    enable_content_moderation: bool = True
    auto_moderate_uploads: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "is_public": self.is_public,
            "require_approval": self.require_approval,
            "max_participants": self.max_participants,
            "allow_guest_access": self.allow_guest_access,
            "default_role": self.default_role.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "allow_anonymous_editing": self.allow_anonymous_editing,
            "require_edit_approval": self.require_edit_approval,
            "max_document_size_mb": self.max_document_size_mb,
            "enable_voice": self.enable_voice,
            "enable_video": self.enable_video,
            "max_voice_participants": self.max_voice_participants,
            "max_video_participants": self.max_video_participants,
            "enable_content_moderation": self.enable_content_moderation,
            "auto_moderate_uploads": self.auto_moderate_uploads,
        }


class WorkspaceManager:
    """Advanced workspace management system"""

    def __init__(self):
        self.workspaces: Dict[str, WorkspaceSettings] = {}
        self.user_permissions: Dict[str, List[UserPermissions]] = (
            {}
        )  # user_id -> permissions
        self.workspace_permissions: Dict[str, List[UserPermissions]] = (
            {}
        )  # workspace_id -> permissions
        self.pending_invitations: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # workspace_id -> invitations
        self.audit_log: List[Dict[str, Any]] = []

    def create_workspace(
        self,
        workspace_id: str,
        name: str,
        owner_id: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> WorkspaceSettings:
        """Create a new workspace"""
        workspace_settings = WorkspaceSettings(
            workspace_id=workspace_id,
            name=name,
            description=description,
            created_by=owner_id,
        )

        # Apply custom settings
        if settings:
            for key, value in settings.items():
                if hasattr(workspace_settings, key):
                    setattr(workspace_settings, key, value)

        self.workspaces[workspace_id] = workspace_settings

        # Grant owner permissions
        owner_perms = UserPermissions(
            user_id=owner_id,
            workspace_id=workspace_id,
            role=WorkspaceRole.OWNER,
            granted_by=owner_id,
        )

        if owner_id not in self.user_permissions:
            self.user_permissions[owner_id] = []
        self.user_permissions[owner_id].append(owner_perms)

        if workspace_id not in self.workspace_permissions:
            self.workspace_permissions[workspace_id] = []
        self.workspace_permissions[workspace_id].append(owner_perms)

        self._log_action(
            "workspace_created",
            owner_id,
            workspace_id,
            {"workspace_name": name, "description": description},
        )

        return workspace_settings

    def invite_user(
        self,
        workspace_id: str,
        inviter_id: str,
        invitee_id: str,
        role: WorkspaceRole,
        custom_permissions: Optional[Set[Permission]] = None,
        expires_in_days: Optional[int] = None,
    ) -> bool:
        """Invite a user to workspace"""
        # Check inviter permissions
        if not self.has_permission(inviter_id, workspace_id, Permission.MANAGE_USERS):
            return False

        # Check if user is already in workspace
        if self.get_user_permissions(invitee_id, workspace_id):
            return False

        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return False

        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)

        invitation = {
            "invitee_id": invitee_id,
            "inviter_id": inviter_id,
            "role": role.value,
            "custom_permissions": (
                list(custom_permissions) if custom_permissions else []
            ),
            "expires_at": expires_at.isoformat() if expires_at else None,
            "created_at": datetime.now().isoformat(),
            "status": "pending",
        }

        if workspace_id not in self.pending_invitations:
            self.pending_invitations[workspace_id] = []
        self.pending_invitations[workspace_id].append(invitation)

        self._log_action(
            "user_invited",
            inviter_id,
            workspace_id,
            {"invitee_id": invitee_id, "role": role.value},
        )

        return True

    def accept_invitation(self, user_id: str, workspace_id: str) -> bool:
        """Accept workspace invitation"""
        if workspace_id not in self.pending_invitations:
            return False

        invitation = None
        for inv in self.pending_invitations[workspace_id]:
            if inv["invitee_id"] == user_id and inv["status"] == "pending":
                invitation = inv
                break

        if not invitation:
            return False

        # Check if invitation is expired
        if invitation["expires_at"]:
            expires_at = datetime.fromisoformat(invitation["expires_at"])
            if datetime.now() > expires_at:
                invitation["status"] = "expired"
                return False

        # Grant permissions
        role = WorkspaceRole(invitation["role"])
        custom_perms = set(Permission(p) for p in invitation["custom_permissions"])

        expires_at = None
        if invitation["expires_at"]:
            expires_at = datetime.fromisoformat(invitation["expires_at"])

        user_perms = UserPermissions(
            user_id=user_id,
            workspace_id=workspace_id,
            role=role,
            custom_permissions=custom_perms,
            granted_by=invitation["inviter_id"],
            expires_at=expires_at,
        )

        if user_id not in self.user_permissions:
            self.user_permissions[user_id] = []
        self.user_permissions[user_id].append(user_perms)

        if workspace_id not in self.workspace_permissions:
            self.workspace_permissions[workspace_id] = []
        self.workspace_permissions[workspace_id].append(user_perms)

        # Mark invitation as accepted
        invitation["status"] = "accepted"
        invitation["accepted_at"] = datetime.now().isoformat()

        self._log_action(
            "invitation_accepted", user_id, workspace_id, {"role": role.value}
        )

        return True

    def remove_user(self, workspace_id: str, remover_id: str, user_id: str) -> bool:
        """Remove user from workspace"""
        # Check permissions
        if not self.has_permission(remover_id, workspace_id, Permission.MANAGE_USERS):
            return False

        # Cannot remove workspace owner
        user_perms = self.get_user_permissions(user_id, workspace_id)
        if user_perms and user_perms.role == WorkspaceRole.OWNER:
            return False

        # Remove user permissions
        if user_id in self.user_permissions:
            self.user_permissions[user_id] = [
                p
                for p in self.user_permissions[user_id]
                if p.workspace_id != workspace_id
            ]

        if workspace_id in self.workspace_permissions:
            self.workspace_permissions[workspace_id] = [
                p
                for p in self.workspace_permissions[workspace_id]
                if p.user_id != user_id
            ]

        self._log_action(
            "user_removed", remover_id, workspace_id, {"removed_user_id": user_id}
        )

        return True

    def update_user_role(
        self,
        workspace_id: str,
        updater_id: str,
        user_id: str,
        new_role: WorkspaceRole,
        custom_permissions: Optional[Set[Permission]] = None,
    ) -> bool:
        """Update user role and permissions"""
        # Check permissions
        if not self.has_permission(updater_id, workspace_id, Permission.MANAGE_USERS):
            return False

        user_perms = self.get_user_permissions(user_id, workspace_id)
        if not user_perms:
            return False

        # Cannot change owner role
        if user_perms.role == WorkspaceRole.OWNER and new_role != WorkspaceRole.OWNER:
            return False

        # Update permissions
        user_perms.role = new_role
        if custom_permissions is not None:
            user_perms.custom_permissions = custom_permissions

        self._log_action(
            "user_role_updated",
            updater_id,
            workspace_id,
            {
                "user_id": user_id,
                "new_role": new_role.value,
                "previous_role": user_perms.role.value,
            },
        )

        return True

    def get_user_permissions(
        self, user_id: str, workspace_id: str
    ) -> Optional[UserPermissions]:
        """Get user permissions for workspace"""
        if user_id not in self.user_permissions:
            return None

        for perms in self.user_permissions[user_id]:
            if perms.workspace_id == workspace_id and not perms.is_expired():
                return perms

        return None

    def has_permission(
        self, user_id: str, workspace_id: str, permission: Permission
    ) -> bool:
        """Check if user has specific permission"""
        user_perms = self.get_user_permissions(user_id, workspace_id)
        if not user_perms:
            return False

        return user_perms.has_permission(permission)

    def get_workspace_users(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Get all users in workspace"""
        if workspace_id not in self.workspace_permissions:
            return []

        users = []
        for perms in self.workspace_permissions[workspace_id]:
            if not perms.is_expired():
                users.append(
                    {
                        "user_id": perms.user_id,
                        "role": perms.role.value,
                        "permissions": [
                            p.value for p in perms.get_effective_permissions()
                        ],
                        "granted_at": perms.granted_at.isoformat(),
                        "granted_by": perms.granted_by,
                        "expires_at": (
                            perms.expires_at.isoformat() if perms.expires_at else None
                        ),
                    }
                )

        return users

    def get_user_workspaces(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all workspaces user has access to"""
        if user_id not in self.user_permissions:
            return []

        workspaces = []
        for perms in self.user_permissions[user_id]:
            if not perms.is_expired():
                workspace = self.workspaces.get(perms.workspace_id)
                if workspace:
                    workspaces.append(
                        {
                            "workspace_id": perms.workspace_id,
                            "name": workspace.name,
                            "description": workspace.description,
                            "role": perms.role.value,
                            "permissions": [
                                p.value for p in perms.get_effective_permissions()
                            ],
                        }
                    )

        return workspaces

    def update_workspace_settings(
        self, workspace_id: str, updater_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update workspace settings"""
        if not self.has_permission(updater_id, workspace_id, Permission.ADMIN):
            return False

        workspace = self.workspaces.get(workspace_id)
        if not workspace:
            return False

        for key, value in updates.items():
            if hasattr(workspace, key):
                setattr(workspace, key, value)

        self._log_action("workspace_updated", updater_id, workspace_id, updates)
        return True

    def get_workspace_settings(self, workspace_id: str) -> Optional[WorkspaceSettings]:
        """Get workspace settings"""
        return self.workspaces.get(workspace_id)

    def _log_action(
        self, action: str, user_id: str, workspace_id: str, details: Dict[str, Any]
    ):
        """Log audit action"""
        self.audit_log.append(
            {
                "action": action,
                "user_id": user_id,
                "workspace_id": workspace_id,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_audit_log(
        self,
        workspace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries"""
        logs = self.audit_log

        if workspace_id:
            logs = [log for log in logs if log["workspace_id"] == workspace_id]

        if user_id:
            logs = [log for log in logs if log["user_id"] == user_id]

        return sorted(logs, key=lambda x: x["timestamp"], reverse=True)[:limit]


# Global workspace manager instance
workspace_manager = WorkspaceManager()
