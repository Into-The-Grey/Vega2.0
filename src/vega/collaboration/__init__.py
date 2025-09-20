"""
Vega 2.0 Real-Time Collaboration Core
=====================================

Central coordination system for real-time collaborative features including:
- WebSocket connection management
- Workspace management and synchronization
- Real-time document editing with conflict resolution
- Multi-user chat and communication
- Voice/video integration coordination

This module serves as the backbone for all collaborative features in Vega 2.0.
"""

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    HTTPException,
    status,
)
from fastapi.responses import HTMLResponse
from typing import Dict, List, Set, Optional, Any, Union
import json
import asyncio
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Message types for real-time collaboration"""

    # System messages
    PING = "ping"
    PONG = "pong"
    JOIN_WORKSPACE = "join_workspace"
    LEAVE_WORKSPACE = "leave_workspace"
    USER_STATUS = "user_status"

    # Document editing
    DOCUMENT_EDIT = "document_edit"
    DOCUMENT_SYNC = "document_sync"
    CURSOR_POSITION = "cursor_position"
    SELECTION_UPDATE = "selection_update"

    # Chat and communication
    CHAT_MESSAGE = "chat_message"
    TYPING_INDICATOR = "typing_indicator"
    VOICE_REQUEST = "voice_request"
    VIDEO_REQUEST = "video_request"

    # Collaboration features
    ANNOTATION_ADD = "annotation_add"
    ANNOTATION_UPDATE = "annotation_update"
    ANNOTATION_DELETE = "annotation_delete"
    SHARE_SCREEN = "share_screen"

    # Federated learning collaboration
    FL_MODEL_UPDATE = "fl_model_update"
    FL_TRAINING_STATUS = "fl_training_status"
    FL_PARTICIPANT_JOIN = "fl_participant_join"


@dataclass
class User:
    """User information for collaboration"""

    id: str
    username: str
    email: Optional[str] = None
    role: str = "user"  # admin, user, viewer
    workspace_id: Optional[str] = None
    last_active: datetime = None
    cursor_position: Optional[Dict[str, Any]] = None
    status: str = "online"  # online, away, busy, offline

    def __post_init__(self):
        if self.last_active is None:
            self.last_active = datetime.now()


@dataclass
class Workspace:
    """Collaborative workspace"""

    id: str
    name: str
    owner_id: str
    created_at: datetime
    participants: Set[str]  # user IDs
    documents: Dict[str, Any]  # document_id -> document_data
    settings: Dict[str, Any]
    last_activity: datetime

    def __post_init__(self):
        if self.last_activity is None:
            self.last_activity = datetime.now()


@dataclass
class CollaborationMessage:
    """Standard message format for collaboration"""

    id: str
    type: MessageType
    sender_id: str
    workspace_id: Optional[str]
    data: Dict[str, Any]
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "sender_id": self.sender_id,
            "workspace_id": self.workspace_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
        }


class CollaborationManager:
    """Central manager for real-time collaboration"""

    def __init__(self):
        self.connections: Dict[WebSocket, User] = {}
        self.workspaces: Dict[str, Workspace] = {}
        self.workspace_connections: Dict[str, Set[WebSocket]] = (
            {}
        )  # workspace_id -> connections
        self.user_connections: Dict[str, WebSocket] = {}  # user_id -> connection
        self.message_history: Dict[str, List[CollaborationMessage]] = (
            {}
        )  # workspace_id -> messages

    async def connect_user(self, websocket: WebSocket, user: User) -> bool:
        """Connect a user via WebSocket"""
        try:
            await websocket.accept()

            # Handle existing connection for same user
            if user.id in self.user_connections:
                old_connection = self.user_connections[user.id]
                if old_connection in self.connections:
                    await self.disconnect_user(old_connection)

            self.connections[websocket] = user
            self.user_connections[user.id] = websocket

            logger.info(f"User {user.username} ({user.id}) connected")

            # Send connection confirmation
            await self._send_to_connection(
                websocket,
                CollaborationMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.USER_STATUS,
                    sender_id="system",
                    workspace_id=None,
                    data={"status": "connected", "user": asdict(user)},
                    timestamp=datetime.now(),
                ),
            )

            return True

        except Exception as e:
            logger.error(f"Error connecting user {user.username}: {e}")
            return False

    async def disconnect_user(self, websocket: WebSocket):
        """Disconnect a user"""
        if websocket not in self.connections:
            return

        user = self.connections[websocket]

        # Remove from workspace
        if user.workspace_id:
            await self.leave_workspace(websocket, user.workspace_id)

        # Clean up connections
        del self.connections[websocket]
        if user.id in self.user_connections:
            del self.user_connections[user.id]

        logger.info(f"User {user.username} ({user.id}) disconnected")

    async def join_workspace(self, websocket: WebSocket, workspace_id: str) -> bool:
        """Join a collaborative workspace"""
        if websocket not in self.connections:
            return False

        user = self.connections[websocket]

        # Create workspace if it doesn't exist
        if workspace_id not in self.workspaces:
            self.workspaces[workspace_id] = Workspace(
                id=workspace_id,
                name=f"Workspace {workspace_id[:8]}",
                owner_id=user.id,
                created_at=datetime.now(),
                participants=set(),
                documents={},
                settings={"max_participants": 50, "allow_anonymous": False},
                last_activity=datetime.now(),
            )
            self.message_history[workspace_id] = []
            self.workspace_connections[workspace_id] = set()

        workspace = self.workspaces[workspace_id]

        # Leave current workspace if any
        if user.workspace_id:
            await self.leave_workspace(websocket, user.workspace_id)

        # Join new workspace
        workspace.participants.add(user.id)
        user.workspace_id = workspace_id
        self.workspace_connections[workspace_id].add(websocket)
        workspace.last_activity = datetime.now()

        # Notify other participants
        join_message = CollaborationMessage(
            id=str(uuid.uuid4()),
            type=MessageType.JOIN_WORKSPACE,
            sender_id=user.id,
            workspace_id=workspace_id,
            data={
                "user": {"id": user.id, "username": user.username, "role": user.role},
                "workspace": {
                    "id": workspace.id,
                    "name": workspace.name,
                    "participant_count": len(workspace.participants),
                },
            },
            timestamp=datetime.now(),
        )

        await self._broadcast_to_workspace(
            workspace_id, join_message, exclude=websocket
        )

        # Send workspace state to joining user
        await self._send_workspace_state(websocket, workspace_id)

        logger.info(f"User {user.username} joined workspace {workspace_id}")
        return True

    async def leave_workspace(self, websocket: WebSocket, workspace_id: str):
        """Leave a collaborative workspace"""
        if websocket not in self.connections or workspace_id not in self.workspaces:
            return

        user = self.connections[websocket]
        workspace = self.workspaces[workspace_id]

        # Remove from workspace
        workspace.participants.discard(user.id)
        user.workspace_id = None
        self.workspace_connections[workspace_id].discard(websocket)

        # Notify other participants
        leave_message = CollaborationMessage(
            id=str(uuid.uuid4()),
            type=MessageType.LEAVE_WORKSPACE,
            sender_id=user.id,
            workspace_id=workspace_id,
            data={
                "user": {"id": user.id, "username": user.username},
                "participant_count": len(workspace.participants),
            },
            timestamp=datetime.now(),
        )

        await self._broadcast_to_workspace(workspace_id, leave_message)

        # Clean up empty workspace
        if len(workspace.participants) == 0:
            del self.workspaces[workspace_id]
            del self.message_history[workspace_id]
            del self.workspace_connections[workspace_id]
            logger.info(f"Workspace {workspace_id} cleaned up (no participants)")

        logger.info(f"User {user.username} left workspace {workspace_id}")

    async def handle_message(self, websocket: WebSocket, raw_message: str):
        """Handle incoming WebSocket message"""
        if websocket not in self.connections:
            logger.warning("Received message from unconnected WebSocket")
            return

        try:
            message_data = json.loads(raw_message)
            user = self.connections[websocket]

            message_type = MessageType(message_data.get("type"))

            message = CollaborationMessage(
                id=message_data.get("id", str(uuid.uuid4())),
                type=message_type,
                sender_id=user.id,
                workspace_id=message_data.get("workspace_id", user.workspace_id),
                data=message_data.get("data", {}),
                timestamp=datetime.now(),
                metadata=message_data.get("metadata"),
            )

            # Handle different message types
            if message_type == MessageType.PING:
                await self._handle_ping(websocket, message)
            elif message_type == MessageType.JOIN_WORKSPACE:
                workspace_id = message.data.get("workspace_id")
                if workspace_id:
                    await self.join_workspace(websocket, workspace_id)
            elif message_type == MessageType.LEAVE_WORKSPACE:
                if user.workspace_id:
                    await self.leave_workspace(websocket, user.workspace_id)
            elif message_type == MessageType.CHAT_MESSAGE:
                await self._handle_chat_message(websocket, message)
            elif message_type == MessageType.DOCUMENT_EDIT:
                await self._handle_document_edit(websocket, message)
            elif message_type == MessageType.CURSOR_POSITION:
                await self._handle_cursor_update(websocket, message)
            elif message_type == MessageType.ANNOTATION_ADD:
                await self._handle_annotation(websocket, message)
            else:
                # Generic message broadcasting
                if message.workspace_id:
                    await self._broadcast_to_workspace(
                        message.workspace_id, message, exclude=websocket
                    )

        except Exception as e:
            logger.error(
                f"Error handling message from {self.connections.get(websocket, {}).username}: {e}"
            )

    async def _handle_ping(self, websocket: WebSocket, message: CollaborationMessage):
        """Handle ping message"""
        pong_message = CollaborationMessage(
            id=str(uuid.uuid4()),
            type=MessageType.PONG,
            sender_id="system",
            workspace_id=None,
            data={"original_id": message.id},
            timestamp=datetime.now(),
        )
        await self._send_to_connection(websocket, pong_message)

    async def _handle_chat_message(
        self, websocket: WebSocket, message: CollaborationMessage
    ):
        """Handle chat message"""
        if not message.workspace_id:
            return

        # Store message in history
        if message.workspace_id not in self.message_history:
            self.message_history[message.workspace_id] = []

        self.message_history[message.workspace_id].append(message)

        # Limit message history
        if len(self.message_history[message.workspace_id]) > 1000:
            self.message_history[message.workspace_id] = self.message_history[
                message.workspace_id
            ][-500:]

        # Broadcast to workspace
        await self._broadcast_to_workspace(message.workspace_id, message)

    async def _handle_document_edit(
        self, websocket: WebSocket, message: CollaborationMessage
    ):
        """Handle document editing with conflict resolution"""
        if not message.workspace_id or message.workspace_id not in self.workspaces:
            return

        workspace = self.workspaces[message.workspace_id]
        document_id = message.data.get("document_id")

        if not document_id:
            return

        # Apply operational transformation for conflict resolution
        edit_data = message.data.get("edit", {})

        # Store document changes
        if document_id not in workspace.documents:
            workspace.documents[document_id] = {
                "content": "",
                "version": 0,
                "last_modified": datetime.now().isoformat(),
                "last_modifier": message.sender_id,
            }

        document = workspace.documents[document_id]

        # Simple conflict resolution (in production, use proper OT)
        if edit_data.get("operation") == "insert":
            position = edit_data.get("position", 0)
            text = edit_data.get("text", "")
            content = document["content"]
            document["content"] = content[:position] + text + content[position:]
        elif edit_data.get("operation") == "delete":
            start = edit_data.get("start", 0)
            end = edit_data.get("end", 0)
            content = document["content"]
            document["content"] = content[:start] + content[end:]

        document["version"] += 1
        document["last_modified"] = datetime.now().isoformat()
        document["last_modifier"] = message.sender_id

        # Broadcast document update
        sync_message = CollaborationMessage(
            id=str(uuid.uuid4()),
            type=MessageType.DOCUMENT_SYNC,
            sender_id=message.sender_id,
            workspace_id=message.workspace_id,
            data={"document_id": document_id, "document": document, "edit": edit_data},
            timestamp=datetime.now(),
        )

        await self._broadcast_to_workspace(
            message.workspace_id, sync_message, exclude=websocket
        )

    async def _handle_cursor_update(
        self, websocket: WebSocket, message: CollaborationMessage
    ):
        """Handle cursor position updates"""
        if websocket in self.connections:
            user = self.connections[websocket]
            user.cursor_position = message.data.get("position")
            user.last_active = datetime.now()

        # Broadcast cursor position to workspace
        if message.workspace_id:
            await self._broadcast_to_workspace(
                message.workspace_id, message, exclude=websocket
            )

    async def _handle_annotation(
        self, websocket: WebSocket, message: CollaborationMessage
    ):
        """Handle annotation creation/updates"""
        if not message.workspace_id or message.workspace_id not in self.workspaces:
            return

        # Store annotation in workspace
        workspace = self.workspaces[message.workspace_id]
        annotation_id = message.data.get("annotation_id", str(uuid.uuid4()))

        if "annotations" not in workspace.documents:
            workspace.documents["annotations"] = {}

        workspace.documents["annotations"][annotation_id] = {
            **message.data,
            "created_by": message.sender_id,
            "created_at": datetime.now().isoformat(),
        }

        # Broadcast annotation to workspace
        await self._broadcast_to_workspace(message.workspace_id, message)

    async def _send_workspace_state(self, websocket: WebSocket, workspace_id: str):
        """Send current workspace state to user"""
        if workspace_id not in self.workspaces:
            return

        workspace = self.workspaces[workspace_id]

        # Get participant info
        participants = []
        for participant_id in workspace.participants:
            if participant_id in self.user_connections:
                conn = self.user_connections[participant_id]
                if conn in self.connections:
                    user = self.connections[conn]
                    participants.append(
                        {
                            "id": user.id,
                            "username": user.username,
                            "role": user.role,
                            "status": user.status,
                            "cursor_position": user.cursor_position,
                            "last_active": user.last_active.isoformat(),
                        }
                    )

        state_message = CollaborationMessage(
            id=str(uuid.uuid4()),
            type=MessageType.DOCUMENT_SYNC,
            sender_id="system",
            workspace_id=workspace_id,
            data={
                "workspace": {
                    "id": workspace.id,
                    "name": workspace.name,
                    "owner_id": workspace.owner_id,
                    "participants": participants,
                    "documents": workspace.documents,
                    "settings": workspace.settings,
                },
                "message_history": [
                    msg.to_dict()
                    for msg in self.message_history.get(workspace_id, [])[-50:]
                ],
            },
            timestamp=datetime.now(),
        )

        await self._send_to_connection(websocket, state_message)

    async def _send_to_connection(
        self, websocket: WebSocket, message: CollaborationMessage
    ):
        """Send message to specific connection"""
        try:
            await websocket.send_text(json.dumps(message.to_dict()))
        except Exception as e:
            logger.error(f"Error sending message to connection: {e}")
            # Connection likely closed, clean up
            await self.disconnect_user(websocket)

    async def _broadcast_to_workspace(
        self,
        workspace_id: str,
        message: CollaborationMessage,
        exclude: Optional[WebSocket] = None,
    ):
        """Broadcast message to all connections in workspace"""
        if workspace_id not in self.workspace_connections:
            return

        connections = self.workspace_connections[workspace_id].copy()
        disconnected = []

        for websocket in connections:
            if websocket == exclude:
                continue

            try:
                await websocket.send_text(json.dumps(message.to_dict()))
            except Exception as e:
                logger.error(f"Error broadcasting to workspace {workspace_id}: {e}")
                disconnected.append(websocket)

        # Clean up disconnected connections
        for websocket in disconnected:
            await self.disconnect_user(websocket)

    def get_workspace_stats(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        """Get workspace statistics"""
        if workspace_id not in self.workspaces:
            return None

        workspace = self.workspaces[workspace_id]

        return {
            "id": workspace.id,
            "name": workspace.name,
            "participant_count": len(workspace.participants),
            "document_count": len(workspace.documents),
            "message_count": len(self.message_history.get(workspace_id, [])),
            "created_at": workspace.created_at.isoformat(),
            "last_activity": workspace.last_activity.isoformat(),
        }

    def get_user_status(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user status and activity"""
        if user_id not in self.user_connections:
            return None

        websocket = self.user_connections[user_id]
        if websocket not in self.connections:
            return None

        user = self.connections[websocket]

        return {
            "id": user.id,
            "username": user.username,
            "status": user.status,
            "workspace_id": user.workspace_id,
            "last_active": user.last_active.isoformat(),
            "cursor_position": user.cursor_position,
        }


# Global collaboration manager instance
collaboration_manager = CollaborationManager()
