"""
Integration of collaboration features with main Vega application
==============================================================

This module integrates the collaboration system with the existing
Vega 2.0 infrastructure, providing endpoints and WebSocket handlers
for the main application.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, List, Any, Optional
import json
import uuid
from datetime import datetime

from . import collaboration_manager, User, MessageType, CollaborationMessage
from .document_editor import (
    document_collaboration,
    DocumentChange,
    Operation,
    OperationType,
)
from .voice_video import media_manager
from ..core.config import get_config


def create_collaboration_router() -> APIRouter:
    """Create FastAPI router for collaboration endpoints"""
    router = APIRouter(prefix="/collaboration", tags=["collaboration"])

    @router.websocket("/ws")
    async def collaboration_websocket(
        websocket: WebSocket,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Main collaboration WebSocket endpoint"""
        try:
            # Simple authentication
            config = get_config()
            if api_key and api_key not in [config.api_key] + config.api_keys_extra:
                await websocket.close(code=1008, reason="Invalid API key")
                return

            if not user_id:
                user_id = str(uuid.uuid4())
            if not username:
                username = f"User-{user_id[:8]}"

            user = User(id=user_id, username=username, role="user", status="online")

            # Connect to collaboration system
            connected = await collaboration_manager.connect_user(websocket, user)
            if not connected:
                await websocket.close(code=1008, reason="Connection failed")
                return

            # Handle messages
            while True:
                try:
                    data = await websocket.receive_text()
                    await collaboration_manager.handle_message(websocket, data)
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"Error in collaboration WebSocket: {e}")

        except Exception as e:
            print(f"Collaboration WebSocket error: {e}")
        finally:
            await collaboration_manager.disconnect_user(websocket)

    @router.get("/workspaces")
    async def list_workspaces():
        """List all active workspaces"""
        workspaces = []
        for workspace_id in collaboration_manager.workspaces:
            stats = collaboration_manager.get_workspace_stats(workspace_id)
            if stats:
                workspaces.append(stats)
        return {"workspaces": workspaces}

    @router.get("/workspaces/{workspace_id}")
    async def get_workspace_details(workspace_id: str):
        """Get detailed workspace information"""
        stats = collaboration_manager.get_workspace_stats(workspace_id)
        if not stats:
            raise HTTPException(status_code=404, detail="Workspace not found")
        return stats

    @router.post("/workspaces/{workspace_id}/documents")
    async def create_document(
        workspace_id: str, title: str, user_id: str, initial_content: str = ""
    ):
        """Create a new collaborative document"""
        doc_id = document_collaboration.create_document(title, user_id, initial_content)

        # Notify workspace participants
        if workspace_id in collaboration_manager.workspaces:
            message = CollaborationMessage(
                id=str(uuid.uuid4()),
                type=MessageType.DOCUMENT_SYNC,
                sender_id=user_id,
                workspace_id=workspace_id,
                data={
                    "action": "document_created",
                    "document_id": doc_id,
                    "title": title,
                },
                timestamp=datetime.now(),
            )
            await collaboration_manager._broadcast_to_workspace(workspace_id, message)

        return {"document_id": doc_id, "title": title}

    @router.get("/documents")
    async def list_documents(user_id: Optional[str] = None):
        """List all documents"""
        documents = document_collaboration.list_documents(user_id)
        return {"documents": documents}

    @router.get("/documents/{doc_id}")
    async def get_document(doc_id: str):
        """Get document details and state"""
        state = document_collaboration.get_document_state(doc_id)
        if not state:
            raise HTTPException(status_code=404, detail="Document not found")
        return state

    @router.post("/documents/{doc_id}/edit")
    async def edit_document(
        doc_id: str, operations: List[Dict[str, Any]], user_id: str, base_version: int
    ):
        """Apply edit operations to document"""
        # Convert operations from dict to Operation objects
        ops = []
        for op_data in operations:
            ops.append(Operation.from_dict(op_data))

        # Create document change
        change = DocumentChange(
            id=str(uuid.uuid4()),
            document_id=doc_id,
            user_id=user_id,
            operations=ops,
            base_version=base_version,
            timestamp=datetime.now(),
        )

        # Apply change
        success, transformed_change = document_collaboration.apply_change(
            doc_id, change
        )

        if not success:
            raise HTTPException(
                status_code=400, detail="Failed to apply document change"
            )

        # Broadcast change to workspace if document is in a workspace
        document = document_collaboration.get_document(doc_id)
        if document and user_id in collaboration_manager.user_connections:
            user_connection = collaboration_manager.user_connections[user_id]
            if user_connection in collaboration_manager.connections:
                user = collaboration_manager.connections[user_connection]
                if user.workspace_id:
                    message = CollaborationMessage(
                        id=str(uuid.uuid4()),
                        type=MessageType.DOCUMENT_EDIT,
                        sender_id=user_id,
                        workspace_id=user.workspace_id,
                        data={
                            "document_id": doc_id,
                            "change": (
                                transformed_change.to_dict()
                                if transformed_change
                                else change.to_dict()
                            ),
                            "new_version": document.version,
                        },
                        timestamp=datetime.now(),
                    )
                    await collaboration_manager._broadcast_to_workspace(
                        user.workspace_id, message, exclude=user_connection
                    )

        return {
            "success": True,
            "new_version": document.version if document else None,
            "change_id": transformed_change.id if transformed_change else change.id,
        }

    @router.post("/documents/{doc_id}/join")
    async def join_document_editing(
        doc_id: str,
        user_id: str,
        cursor_position: int = 0,
        selection: Optional[Dict[str, Any]] = None,
    ):
        """Join document editing session"""
        cursor_info = {"position": cursor_position, "selection": selection}

        success = document_collaboration.join_document_editing(
            doc_id, user_id, cursor_info
        )
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")

        return {"success": True, "document_id": doc_id}

    @router.post("/documents/{doc_id}/leave")
    async def leave_document_editing(doc_id: str, user_id: str):
        """Leave document editing session"""
        document_collaboration.leave_document_editing(doc_id, user_id)
        return {"success": True}

    @router.post("/documents/{doc_id}/cursor")
    async def update_cursor_position(
        doc_id: str,
        user_id: str,
        position: int,
        selection: Optional[Dict[str, Any]] = None,
    ):
        """Update cursor position"""
        cursor_info = {"position": position, "selection": selection}

        document_collaboration.update_cursor_position(doc_id, user_id, cursor_info)

        # Broadcast cursor update to workspace
        if user_id in collaboration_manager.user_connections:
            user_connection = collaboration_manager.user_connections[user_id]
            if user_connection in collaboration_manager.connections:
                user = collaboration_manager.connections[user_connection]
                if user.workspace_id:
                    message = CollaborationMessage(
                        id=str(uuid.uuid4()),
                        type=MessageType.CURSOR_POSITION,
                        sender_id=user_id,
                        workspace_id=user.workspace_id,
                        data={"document_id": doc_id, "position": cursor_info},
                        timestamp=datetime.now(),
                    )
                    await collaboration_manager._broadcast_to_workspace(
                        user.workspace_id, message, exclude=user_connection
                    )

        return {"success": True}

    @router.get("/voice/{workspace_id}/sessions")
    async def get_voice_sessions(workspace_id: str):
        """Get active voice sessions for workspace"""
        sessions = []
        for session in media_manager.sessions.values():
            if (
                session.workspace_id == workspace_id
                and session.session_type == "voice"
                and session.active
            ):
                sessions.append(media_manager.get_session_info(session.id))
        return {"sessions": sessions}

    @router.get("/video/{workspace_id}/sessions")
    async def get_video_sessions(workspace_id: str):
        """Get active video sessions for workspace"""
        sessions = []
        for session in media_manager.sessions.values():
            if (
                session.workspace_id == workspace_id
                and session.session_type == "video"
                and session.active
            ):
                sessions.append(media_manager.get_session_info(session.id))
        return {"sessions": sessions}

    @router.get("/dashboard", response_class=HTMLResponse)
    async def collaboration_dashboard():
        """Serve collaboration dashboard"""
        return HTMLResponse(
            content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vega 2.0 Collaboration Dashboard</title>
            <meta charset="utf-8">
            <style>
                body { font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; }
                .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; }
                .status { padding: 10px; border-radius: 4px; margin-bottom: 20px; }
                .connected { background: #d4edda; color: #155724; }
                .disconnected { background: #f8d7da; color: #721c24; }
                h1, h2 { color: #333; }
                .workspace-list, .document-list { list-style: none; padding: 0; }
                .workspace-item, .document-item { 
                    background: #f8f9fa; 
                    margin: 5px 0; 
                    padding: 10px; 
                    border-radius: 4px; 
                    border-left: 4px solid #007bff;
                }
            </style>
        </head>
        <body>
            <h1>🤝 Vega 2.0 Collaboration Dashboard</h1>
            
            <div id="status" class="status disconnected">
                Checking connection status...
            </div>
            
            <div class="grid">
                <div class="card">
                    <h2>🏢 Active Workspaces</h2>
                    <ul id="workspaceList" class="workspace-list">
                        <li>Loading...</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>📄 Documents</h2>
                    <ul id="documentList" class="document-list">
                        <li>Loading...</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>📊 Statistics</h2>
                    <div id="stats">
                        <p>Total Workspaces: <span id="totalWorkspaces">-</span></p>
                        <p>Total Documents: <span id="totalDocuments">-</span></p>
                        <p>Active Users: <span id="activeUsers">-</span></p>
                    </div>
                </div>
                
                <div class="card">
                    <h2>🔧 Quick Actions</h2>
                    <p><a href="/collaboration" target="_blank">Open Collaboration Interface</a></p>
                    <p><a href="/docs" target="_blank">API Documentation</a></p>
                    <p><a href="/" target="_blank">Main Vega Dashboard</a></p>
                </div>
            </div>
            
            <script>
                async function loadDashboardData() {
                    try {
                        // Load workspaces
                        const workspacesResponse = await fetch('/collaboration/workspaces');
                        const workspacesData = await workspacesResponse.json();
                        
                        const workspaceList = document.getElementById('workspaceList');
                        if (workspacesData.workspaces.length > 0) {
                            workspaceList.innerHTML = workspacesData.workspaces.map(ws => 
                                `<li class="workspace-item">
                                    <strong>${ws.name}</strong> (${ws.participant_count} participants)
                                    <br><small>Last activity: ${new Date(ws.last_activity).toLocaleString()}</small>
                                </li>`
                            ).join('');
                        } else {
                            workspaceList.innerHTML = '<li>No active workspaces</li>';
                        }
                        
                        // Load documents
                        const documentsResponse = await fetch('/collaboration/documents');
                        const documentsData = await documentsResponse.json();
                        
                        const documentList = document.getElementById('documentList');
                        if (documentsData.documents.length > 0) {
                            documentList.innerHTML = documentsData.documents.map(doc => 
                                `<li class="document-item">
                                    <strong>${doc.title}</strong> (v${doc.version})
                                    <br><small>${doc.active_editors} active editors</small>
                                </li>`
                            ).join('');
                        } else {
                            documentList.innerHTML = '<li>No documents found</li>';
                        }
                        
                        // Update stats
                        document.getElementById('totalWorkspaces').textContent = workspacesData.workspaces.length;
                        document.getElementById('totalDocuments').textContent = documentsData.documents.length;
                        
                        const activeUsers = workspacesData.workspaces.reduce((sum, ws) => sum + ws.participant_count, 0);
                        document.getElementById('activeUsers').textContent = activeUsers;
                        
                        // Update status
                        const status = document.getElementById('status');
                        status.textContent = '🟢 Collaboration system is running';
                        status.className = 'status connected';
                        
                    } catch (error) {
                        console.error('Error loading dashboard data:', error);
                        const status = document.getElementById('status');
                        status.textContent = '🔴 Error loading collaboration data';
                        status.className = 'status disconnected';
                    }
                }
                
                // Load data on page load
                loadDashboardData();
                
                // Refresh every 30 seconds
                setInterval(loadDashboardData, 30000);
            </script>
        </body>
        </html>
        """
        )

    return router


def integrate_with_main_app(main_app):
    """Integrate collaboration features with main Vega application"""
    # Add collaboration router
    collaboration_router = create_collaboration_router()
    main_app.include_router(collaboration_router)

    # Add collaboration WebSocket to main app
    @main_app.websocket("/ws/collaboration")
    async def main_collaboration_websocket(websocket: WebSocket):
        """Collaboration WebSocket endpoint in main app"""
        await collaboration_router.routes[0].endpoint(websocket)

    # Add collaboration dashboard to main app
    @main_app.get("/collaboration", response_class=HTMLResponse)
    async def main_collaboration_dashboard():
        """Collaboration dashboard in main app"""
        from .server import collaboration_dashboard

        return await collaboration_dashboard()

    print("✅ Collaboration features integrated with main Vega application")
