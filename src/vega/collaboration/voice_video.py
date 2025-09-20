"""
Voice and Video Integration for Vega 2.0 Collaboration
=====================================================

Provides WebRTC-based voice and video communication capabilities
for real-time collaboration workspaces.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set, Any, Optional
import json
import uuid
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MediaSession:
    """Media session for voice/video calls"""

    id: str
    workspace_id: str
    participants: Set[str]
    session_type: str  # 'voice', 'video', 'screen_share'
    created_at: datetime
    active: bool = True


class MediaManager:
    """Manages voice/video sessions and WebRTC signaling"""

    def __init__(self):
        self.sessions: Dict[str, MediaSession] = {}
        self.participant_sessions: Dict[str, str] = {}  # user_id -> session_id
        self.websocket_connections: Dict[str, WebSocket] = {}  # user_id -> websocket

    async def create_session(
        self, workspace_id: str, session_type: str, creator_id: str
    ) -> str:
        """Create a new media session"""
        session_id = str(uuid.uuid4())

        session = MediaSession(
            id=session_id,
            workspace_id=workspace_id,
            participants={creator_id},
            session_type=session_type,
            created_at=datetime.now(),
        )

        self.sessions[session_id] = session
        self.participant_sessions[creator_id] = session_id

        logger.info(
            f"Created {session_type} session {session_id} for workspace {workspace_id}"
        )
        return session_id

    async def join_session(
        self, session_id: str, user_id: str, websocket: WebSocket
    ) -> bool:
        """Join an existing media session"""
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        if not session.active:
            return False

        session.participants.add(user_id)
        self.participant_sessions[user_id] = session_id
        self.websocket_connections[user_id] = websocket

        # Notify other participants about new joiner
        await self._notify_participants(
            session_id,
            {
                "type": "participant_joined",
                "user_id": user_id,
                "session_id": session_id,
            },
            exclude=user_id,
        )

        logger.info(
            f"User {user_id} joined {session.session_type} session {session_id}"
        )
        return True

    async def leave_session(self, user_id: str):
        """Leave current media session"""
        if user_id not in self.participant_sessions:
            return

        session_id = self.participant_sessions[user_id]
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        session.participants.discard(user_id)

        # Clean up
        del self.participant_sessions[user_id]
        if user_id in self.websocket_connections:
            del self.websocket_connections[user_id]

        # Notify other participants
        await self._notify_participants(
            session_id,
            {"type": "participant_left", "user_id": user_id, "session_id": session_id},
        )

        # Clean up empty session
        if len(session.participants) == 0:
            session.active = False
            del self.sessions[session_id]
            logger.info(f"Closed empty session {session_id}")

        logger.info(f"User {user_id} left session {session_id}")

    async def handle_webrtc_signal(self, user_id: str, signal_data: Dict[str, Any]):
        """Handle WebRTC signaling messages"""
        if user_id not in self.participant_sessions:
            return

        session_id = self.participant_sessions[user_id]
        target_user = signal_data.get("target_user")

        if target_user and target_user in self.websocket_connections:
            # Direct peer-to-peer signaling
            target_ws = self.websocket_connections[target_user]
            try:
                await target_ws.send_text(
                    json.dumps(
                        {
                            "type": "webrtc_signal",
                            "from_user": user_id,
                            "signal": signal_data.get("signal"),
                            "session_id": session_id,
                        }
                    )
                )
            except Exception as e:
                logger.error(f"Error forwarding WebRTC signal: {e}")
        else:
            # Broadcast to all participants (for group calls)
            await self._notify_participants(
                session_id,
                {
                    "type": "webrtc_signal",
                    "from_user": user_id,
                    "signal": signal_data.get("signal"),
                },
                exclude=user_id,
            )

    async def _notify_participants(
        self, session_id: str, message: Dict[str, Any], exclude: Optional[str] = None
    ):
        """Notify all participants in a session"""
        if session_id not in self.sessions:
            return

        session = self.sessions[session_id]
        disconnected = []

        for participant_id in session.participants:
            if participant_id == exclude:
                continue

            if participant_id in self.websocket_connections:
                websocket = self.websocket_connections[participant_id]
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error notifying participant {participant_id}: {e}")
                    disconnected.append(participant_id)

        # Clean up disconnected participants
        for participant_id in disconnected:
            await self.leave_session(participant_id)

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]
        return {
            "id": session.id,
            "workspace_id": session.workspace_id,
            "participants": list(session.participants),
            "session_type": session.session_type,
            "created_at": session.created_at.isoformat(),
            "active": session.active,
        }


# Global media manager instance
media_manager = MediaManager()


def create_voice_video_app() -> FastAPI:
    """Create FastAPI app for voice/video functionality"""
    app = FastAPI(title="Vega Voice/Video", version="1.0.0")

    @app.websocket("/voice/{workspace_id}")
    async def voice_websocket(websocket: WebSocket, workspace_id: str, user_id: str):
        """WebSocket endpoint for voice communication"""
        await websocket.accept()

        try:
            # Create or join voice session
            # For simplicity, we'll have one voice session per workspace
            voice_sessions = [
                s
                for s in media_manager.sessions.values()
                if s.workspace_id == workspace_id
                and s.session_type == "voice"
                and s.active
            ]

            if voice_sessions:
                session_id = voice_sessions[0].id
                await media_manager.join_session(session_id, user_id, websocket)
            else:
                session_id = await media_manager.create_session(
                    workspace_id, "voice", user_id
                )
                await media_manager.join_session(session_id, user_id, websocket)

            # Send session info to user
            session_info = media_manager.get_session_info(session_id)
            await websocket.send_text(
                json.dumps({"type": "session_joined", "session": session_info})
            )

            # Handle WebRTC signaling
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "webrtc_signal":
                    await media_manager.handle_webrtc_signal(user_id, message)
                elif message.get("type") == "mute":
                    # Broadcast mute status
                    await media_manager._notify_participants(
                        session_id,
                        {
                            "type": "participant_muted",
                            "user_id": user_id,
                            "muted": message.get("muted", True),
                        },
                        exclude=user_id,
                    )

        except WebSocketDisconnect:
            await media_manager.leave_session(user_id)
        except Exception as e:
            logger.error(f"Voice WebSocket error for user {user_id}: {e}")
            await media_manager.leave_session(user_id)

    @app.websocket("/video/{workspace_id}")
    async def video_websocket(websocket: WebSocket, workspace_id: str, user_id: str):
        """WebSocket endpoint for video communication"""
        await websocket.accept()

        try:
            # Similar to voice but for video sessions
            video_sessions = [
                s
                for s in media_manager.sessions.values()
                if s.workspace_id == workspace_id
                and s.session_type == "video"
                and s.active
            ]

            if video_sessions:
                session_id = video_sessions[0].id
                await media_manager.join_session(session_id, user_id, websocket)
            else:
                session_id = await media_manager.create_session(
                    workspace_id, "video", user_id
                )
                await media_manager.join_session(session_id, user_id, websocket)

            # Send session info
            session_info = media_manager.get_session_info(session_id)
            await websocket.send_text(
                json.dumps({"type": "session_joined", "session": session_info})
            )

            # Handle messages
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)

                if message.get("type") == "webrtc_signal":
                    await media_manager.handle_webrtc_signal(user_id, message)
                elif message.get("type") == "video_toggle":
                    await media_manager._notify_participants(
                        session_id,
                        {
                            "type": "participant_video_toggle",
                            "user_id": user_id,
                            "video_enabled": message.get("enabled", True),
                        },
                        exclude=user_id,
                    )

        except WebSocketDisconnect:
            await media_manager.leave_session(user_id)
        except Exception as e:
            logger.error(f"Video WebSocket error for user {user_id}: {e}")
            await media_manager.leave_session(user_id)

    @app.get("/sessions/{workspace_id}")
    async def get_workspace_sessions(workspace_id: str):
        """Get active media sessions for a workspace"""
        sessions = []
        for session in media_manager.sessions.values():
            if session.workspace_id == workspace_id and session.active:
                sessions.append(media_manager.get_session_info(session.id))
        return {"sessions": sessions}

    @app.get("/health")
    async def health_check():
        """Health check"""
        return {
            "status": "healthy",
            "active_sessions": len(
                [s for s in media_manager.sessions.values() if s.active]
            ),
            "total_participants": len(media_manager.participant_sessions),
        }

    return app


if __name__ == "__main__":
    import uvicorn

    app = create_voice_video_app()
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
