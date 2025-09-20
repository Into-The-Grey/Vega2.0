"""
Advanced communication tools for collaboration
=============================================

This module provides comprehensive communication features including
chat systems, notifications, annotations, and messaging infrastructure.
"""

from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json


class MessageType(Enum):
    """Message types for communication"""

    TEXT = "text"
    FILE = "file"
    IMAGE = "image"
    CODE = "code"
    SYSTEM = "system"
    NOTIFICATION = "notification"
    MENTION = "mention"
    REACTION = "reaction"
    THREAD_REPLY = "thread_reply"


class NotificationLevel(Enum):
    """Notification priority levels"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AnnotationType(Enum):
    """Document annotation types"""

    COMMENT = "comment"
    SUGGESTION = "suggestion"
    HIGHLIGHT = "highlight"
    NOTE = "note"
    REVIEW = "review"
    QUESTION = "question"


@dataclass
class ChatMessage:
    """Represents a chat message"""

    id: str
    channel_id: str
    author_id: str
    content: str
    message_type: MessageType
    timestamp: datetime
    edited_at: Optional[datetime] = None
    parent_message_id: Optional[str] = None  # For threading
    mentions: List[str] = field(default_factory=list)
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji -> user_ids
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "channel_id": self.channel_id,
            "author_id": self.author_id,
            "content": self.content,
            "message_type": self.message_type.value,
            "timestamp": self.timestamp.isoformat(),
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "parent_message_id": self.parent_message_id,
            "mentions": self.mentions,
            "reactions": self.reactions,
            "attachments": self.attachments,
            "metadata": self.metadata,
        }


@dataclass
class ChatChannel:
    """Represents a chat channel"""

    id: str
    workspace_id: str
    name: str
    description: str
    created_by: str
    created_at: datetime
    is_private: bool = False
    is_archived: bool = False
    members: Set[str] = field(default_factory=set)
    admins: Set[str] = field(default_factory=set)
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "is_private": self.is_private,
            "is_archived": self.is_archived,
            "members": list(self.members),
            "admins": list(self.admins),
            "settings": self.settings,
        }


@dataclass
class Notification:
    """Represents a notification"""

    id: str
    user_id: str
    title: str
    content: str
    level: NotificationLevel
    timestamp: datetime
    read: bool = False
    action_url: Optional[str] = None
    source_type: str = "system"  # system, chat, document, etc.
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "level": self.level.value,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read,
            "action_url": self.action_url,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "metadata": self.metadata,
        }


@dataclass
class DocumentAnnotation:
    """Represents a document annotation"""

    id: str
    document_id: str
    author_id: str
    annotation_type: AnnotationType
    content: str
    position: Dict[str, Any]  # Position info (line, char, selection, etc.)
    timestamp: datetime
    resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    replies: List[str] = field(default_factory=list)  # Reply annotation IDs
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "document_id": self.document_id,
            "author_id": self.author_id,
            "annotation_type": self.annotation_type.value,
            "content": self.content,
            "position": self.position,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "replies": self.replies,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class CommunicationManager:
    """Manages communication features across the collaboration system"""

    def __init__(self):
        # Chat system
        self.channels: Dict[str, ChatChannel] = {}
        self.messages: Dict[str, ChatMessage] = {}
        self.channel_messages: Dict[str, List[str]] = {}  # channel_id -> message_ids

        # Notifications
        self.notifications: Dict[str, Notification] = {}
        self.user_notifications: Dict[str, List[str]] = (
            {}
        )  # user_id -> notification_ids

        # Annotations
        self.annotations: Dict[str, DocumentAnnotation] = {}
        self.document_annotations: Dict[str, List[str]] = (
            {}
        )  # document_id -> annotation_ids

        # User status
        self.user_status: Dict[str, Dict[str, Any]] = {}  # user_id -> status info

        # Presence tracking
        self.online_users: Set[str] = set()
        self.user_activities: Dict[str, Dict[str, Any]] = {}  # user_id -> activity info

    # Chat Management
    def create_channel(
        self,
        workspace_id: str,
        name: str,
        description: str,
        creator_id: str,
        is_private: bool = False,
        initial_members: Optional[List[str]] = None,
    ) -> str:
        """Create a new chat channel"""
        channel_id = str(uuid.uuid4())

        members = set(initial_members or [])
        members.add(creator_id)  # Creator is always a member

        channel = ChatChannel(
            id=channel_id,
            workspace_id=workspace_id,
            name=name,
            description=description,
            created_by=creator_id,
            created_at=datetime.now(),
            is_private=is_private,
            members=members,
            admins={creator_id},
        )

        self.channels[channel_id] = channel
        self.channel_messages[channel_id] = []

        # Send welcome message
        self._send_system_message(
            channel_id, f"Channel '{name}' created by {creator_id}", creator_id
        )

        return channel_id

    def send_message(
        self,
        channel_id: str,
        author_id: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        parent_message_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Send a message to a channel"""
        channel = self.channels.get(channel_id)
        if not channel or author_id not in channel.members:
            raise ValueError("Invalid channel or user not a member")

        message_id = str(uuid.uuid4())

        # Extract mentions from content
        mentions = self._extract_mentions(content)

        message = ChatMessage(
            id=message_id,
            channel_id=channel_id,
            author_id=author_id,
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
            parent_message_id=parent_message_id,
            mentions=mentions,
            attachments=attachments or [],
        )

        self.messages[message_id] = message
        self.channel_messages[channel_id].append(message_id)

        # Send notifications for mentions
        for user_id in mentions:
            if user_id != author_id:  # Don't notify self
                self._create_notification(
                    user_id=user_id,
                    title=f"Mentioned in #{channel.name}",
                    content=f"{author_id} mentioned you: {content[:100]}...",
                    level=NotificationLevel.NORMAL,
                    source_type="chat",
                    source_id=message_id,
                    action_url=f"/chat/{channel_id}#{message_id}",
                )

        return message_id

    def edit_message(self, message_id: str, user_id: str, new_content: str) -> bool:
        """Edit a message"""
        message = self.messages.get(message_id)
        if not message or message.author_id != user_id:
            return False

        message.content = new_content
        message.edited_at = datetime.now()
        message.mentions = self._extract_mentions(new_content)

        return True

    def add_reaction(self, message_id: str, user_id: str, emoji: str) -> bool:
        """Add reaction to message"""
        message = self.messages.get(message_id)
        if not message:
            return False

        if emoji not in message.reactions:
            message.reactions[emoji] = []

        if user_id not in message.reactions[emoji]:
            message.reactions[emoji].append(user_id)

        return True

    def remove_reaction(self, message_id: str, user_id: str, emoji: str) -> bool:
        """Remove reaction from message"""
        message = self.messages.get(message_id)
        if not message or emoji not in message.reactions:
            return False

        if user_id in message.reactions[emoji]:
            message.reactions[emoji].remove(user_id)

            # Remove emoji if no reactions left
            if not message.reactions[emoji]:
                del message.reactions[emoji]

        return True

    def get_channel_messages(
        self, channel_id: str, limit: int = 50, before: Optional[str] = None
    ) -> List[ChatMessage]:
        """Get messages from a channel"""
        message_ids = self.channel_messages.get(channel_id, [])

        messages = [self.messages[mid] for mid in message_ids if mid in self.messages]
        messages.sort(key=lambda m: m.timestamp, reverse=True)

        if before:
            # Find messages before the specified message
            before_index = None
            for i, msg in enumerate(messages):
                if msg.id == before:
                    before_index = i
                    break

            if before_index is not None:
                messages = messages[before_index + 1 :]

        return messages[:limit]

    def join_channel(self, channel_id: str, user_id: str) -> bool:
        """Join a channel"""
        channel = self.channels.get(channel_id)
        if not channel or (channel.is_private and user_id not in channel.members):
            return False

        channel.members.add(user_id)

        # Send system message
        self._send_system_message(channel_id, f"{user_id} joined the channel", user_id)

        return True

    def leave_channel(self, channel_id: str, user_id: str) -> bool:
        """Leave a channel"""
        channel = self.channels.get(channel_id)
        if not channel or user_id not in channel.members:
            return False

        channel.members.discard(user_id)
        channel.admins.discard(user_id)

        # Send system message
        self._send_system_message(channel_id, f"{user_id} left the channel", user_id)

        return True

    # Notification Management
    def _create_notification(
        self,
        user_id: str,
        title: str,
        content: str,
        level: NotificationLevel,
        source_type: str = "system",
        source_id: Optional[str] = None,
        action_url: Optional[str] = None,
    ) -> str:
        """Create a notification"""
        notification_id = str(uuid.uuid4())

        notification = Notification(
            id=notification_id,
            user_id=user_id,
            title=title,
            content=content,
            level=level,
            timestamp=datetime.now(),
            source_type=source_type,
            source_id=source_id,
            action_url=action_url,
        )

        self.notifications[notification_id] = notification

        if user_id not in self.user_notifications:
            self.user_notifications[user_id] = []
        self.user_notifications[user_id].append(notification_id)

        return notification_id

    def get_user_notifications(
        self, user_id: str, unread_only: bool = False, limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a user"""
        notification_ids = self.user_notifications.get(user_id, [])
        notifications = [
            self.notifications[nid]
            for nid in notification_ids
            if nid in self.notifications
        ]

        if unread_only:
            notifications = [n for n in notifications if not n.read]

        notifications.sort(key=lambda n: n.timestamp, reverse=True)
        return notifications[:limit]

    def mark_notification_read(self, notification_id: str, user_id: str) -> bool:
        """Mark notification as read"""
        notification = self.notifications.get(notification_id)
        if not notification or notification.user_id != user_id:
            return False

        notification.read = True
        return True

    def mark_all_notifications_read(self, user_id: str) -> int:
        """Mark all notifications as read for user"""
        notification_ids = self.user_notifications.get(user_id, [])
        count = 0

        for nid in notification_ids:
            notification = self.notifications.get(nid)
            if notification and not notification.read:
                notification.read = True
                count += 1

        return count

    # Document Annotations
    def create_annotation(
        self,
        document_id: str,
        author_id: str,
        annotation_type: AnnotationType,
        content: str,
        position: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> str:
        """Create a document annotation"""
        annotation_id = str(uuid.uuid4())

        annotation = DocumentAnnotation(
            id=annotation_id,
            document_id=document_id,
            author_id=author_id,
            annotation_type=annotation_type,
            content=content,
            position=position,
            timestamp=datetime.now(),
            tags=tags or [],
        )

        self.annotations[annotation_id] = annotation

        if document_id not in self.document_annotations:
            self.document_annotations[document_id] = []
        self.document_annotations[document_id].append(annotation_id)

        return annotation_id

    def reply_to_annotation(
        self, annotation_id: str, author_id: str, content: str
    ) -> str:
        """Reply to an annotation"""
        parent_annotation = self.annotations.get(annotation_id)
        if not parent_annotation:
            raise ValueError("Annotation not found")

        reply_id = self.create_annotation(
            document_id=parent_annotation.document_id,
            author_id=author_id,
            annotation_type=AnnotationType.COMMENT,
            content=content,
            position=parent_annotation.position,
        )

        # Link reply to parent
        parent_annotation.replies.append(reply_id)

        return reply_id

    def resolve_annotation(self, annotation_id: str, resolver_id: str) -> bool:
        """Resolve an annotation"""
        annotation = self.annotations.get(annotation_id)
        if not annotation:
            return False

        annotation.resolved = True
        annotation.resolved_by = resolver_id
        annotation.resolved_at = datetime.now()

        return True

    def get_document_annotations(
        self,
        document_id: str,
        annotation_type: Optional[AnnotationType] = None,
        resolved: Optional[bool] = None,
    ) -> List[DocumentAnnotation]:
        """Get annotations for a document"""
        annotation_ids = self.document_annotations.get(document_id, [])
        annotations = [
            self.annotations[aid] for aid in annotation_ids if aid in self.annotations
        ]

        if annotation_type:
            annotations = [
                a for a in annotations if a.annotation_type == annotation_type
            ]

        if resolved is not None:
            annotations = [a for a in annotations if a.resolved == resolved]

        annotations.sort(key=lambda a: a.timestamp, reverse=True)
        return annotations

    # User Presence and Status
    def set_user_online(self, user_id: str, activity: Optional[Dict[str, Any]] = None):
        """Set user as online"""
        self.online_users.add(user_id)
        if activity:
            self.user_activities[user_id] = {
                **activity,
                "last_seen": datetime.now().isoformat(),
            }

    def set_user_offline(self, user_id: str):
        """Set user as offline"""
        self.online_users.discard(user_id)
        if user_id in self.user_activities:
            self.user_activities[user_id]["last_seen"] = datetime.now().isoformat()

    def update_user_activity(self, user_id: str, activity: Dict[str, Any]):
        """Update user activity"""
        if user_id not in self.user_activities:
            self.user_activities[user_id] = {}

        self.user_activities[user_id].update(activity)
        self.user_activities[user_id]["last_activity"] = datetime.now().isoformat()

    def get_online_users(self) -> List[str]:
        """Get list of online users"""
        return list(self.online_users)

    def get_user_activity(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user activity information"""
        return self.user_activities.get(user_id)

    # Utility Methods
    def _extract_mentions(self, content: str) -> List[str]:
        """Extract @mentions from content"""
        import re

        mentions = re.findall(r"@([a-zA-Z0-9_-]+)", content)
        return mentions

    def _send_system_message(self, channel_id: str, content: str, related_user: str):
        """Send a system message"""
        message_id = str(uuid.uuid4())

        message = ChatMessage(
            id=message_id,
            channel_id=channel_id,
            author_id="system",
            content=content,
            message_type=MessageType.SYSTEM,
            timestamp=datetime.now(),
        )

        self.messages[message_id] = message
        self.channel_messages[channel_id].append(message_id)

    def get_user_channels(
        self, user_id: str, workspace_id: Optional[str] = None
    ) -> List[ChatChannel]:
        """Get channels user is a member of"""
        channels = []
        for channel in self.channels.values():
            if user_id in channel.members:
                if workspace_id is None or channel.workspace_id == workspace_id:
                    channels.append(channel)

        return sorted(channels, key=lambda c: c.name)


# Global communication manager instance
communication_manager = CommunicationManager()
