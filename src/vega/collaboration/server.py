"""
Vega 2.0 Collaboration Server
=============================

FastAPI server providing real-time collaboration endpoints and WebSocket connections.
Integrates with the main Vega application for collaborative features.
"""

from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    Query,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import json
import uuid
from datetime import datetime

from . import collaboration_manager, User, MessageType, CollaborationMessage
from ..core.config import get_config

app = FastAPI(
    title="Vega 2.0 Collaboration Server",
    description="Real-time collaboration and communication system",
    version="1.0.0",
)

# Add CORS middleware for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def authenticate_user(
    user_id: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
) -> User:
    """Simple authentication - extend with proper auth in production"""
    if not user_id:
        user_id = str(uuid.uuid4())

    if not username:
        username = f"User-{user_id[:8]}"

    # In production, validate API key and get user info from database
    config = get_config()
    if api_key and api_key not in [config.api_key] + config.api_keys_extra:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return User(id=user_id, username=username, role="user", status="online")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None),
    username: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
):
    """Main WebSocket endpoint for real-time collaboration"""
    try:
        # Authenticate user
        user = await authenticate_user(user_id, username, api_key)

        # Connect user
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
                print(f"Error handling WebSocket message: {e}")
                # Send error message to client
                error_message = CollaborationMessage(
                    id=str(uuid.uuid4()),
                    type=MessageType.USER_STATUS,
                    sender_id="system",
                    workspace_id=None,
                    data={"error": str(e)},
                    timestamp=datetime.now(),
                )
                await collaboration_manager._send_to_connection(
                    websocket, error_message
                )

    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        await collaboration_manager.disconnect_user(websocket)


@app.get("/", response_class=HTMLResponse)
async def collaboration_dashboard():
    """Serve collaboration dashboard HTML"""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <title>Vega 2.0 Collaboration</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f7;
        }
        .header { 
            text-align: center; 
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .workspace-container {
            display: grid;
            grid-template-columns: 300px 1fr;
            gap: 20px;
            height: 70vh;
        }
        .sidebar {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .main-area {
            display: grid;
            grid-template-rows: 1fr 200px;
            gap: 20px;
        }
        .document-area {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .chat-area {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .participants {
            margin-bottom: 20px;
        }
        .participant {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            padding: 8px;
            border-radius: 6px;
            background: #f0f0f0;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .online { background: #34c759; }
        .away { background: #ff9500; }
        .offline { background: #8e8e93; }
        .document-editor {
            width: 100%;
            height: 100%;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px;
            font-family: 'Monaco', 'Consolas', monospace;
            resize: none;
        }
        .chat-messages {
            height: 120px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 10px;
            background: #fafafa;
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
        }
        .chat-input button {
            padding: 8px 16px;
            background: #007aff;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
        }
        .chat-input button:hover {
            background: #0056b3;
        }
        .connection-status {
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 20px;
            text-align: center;
            font-weight: 500;
        }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .message {
            margin-bottom: 8px;
            padding: 6px 10px;
            border-radius: 6px;
            background: white;
        }
        .message-user { color: #007aff; font-weight: 500; }
        .message-time { color: #8e8e93; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤝 Vega 2.0 Collaboration</h1>
        <p>Real-time collaborative workspace for AI development</p>
    </div>
    
    <div id="connectionStatus" class="connection-status disconnected">
        🔴 Disconnected - Click "Connect" to join
    </div>
    
    <div class="workspace-container">
        <div class="sidebar">
            <div class="participants">
                <h3>👥 Participants</h3>
                <div id="participantsList">
                    <div class="participant">
                        <div class="status-dot offline"></div>
                        <span>No participants</span>
                    </div>
                </div>
            </div>
            
            <div class="workspace-controls">
                <h3>🏢 Workspace</h3>
                <input type="text" id="workspaceId" placeholder="Workspace ID" value="default">
                <button onclick="joinWorkspace()" style="width: 100%; margin-top: 10px; padding: 10px;">
                    Join Workspace
                </button>
                <button onclick="leaveWorkspace()" style="width: 100%; margin-top: 5px; padding: 10px; background: #ff3b30;">
                    Leave Workspace
                </button>
            </div>
            
            <div class="connection-controls" style="margin-top: 20px;">
                <input type="text" id="username" placeholder="Your username" value="">
                <button onclick="connect()" style="width: 100%; margin-top: 10px; padding: 10px; background: #34c759;">
                    Connect
                </button>
                <button onclick="disconnect()" style="width: 100%; margin-top: 5px; padding: 10px; background: #ff3b30;">
                    Disconnect
                </button>
            </div>
        </div>
        
        <div class="main-area">
            <div class="document-area">
                <h3>📝 Collaborative Document</h3>
                <textarea id="documentEditor" class="document-editor" 
                         placeholder="Start typing to collaborate in real-time..."></textarea>
            </div>
            
            <div class="chat-area">
                <h3>💬 Team Chat</h3>
                <div id="chatMessages" class="chat-messages"></div>
                <div class="chat-input">
                    <input type="text" id="chatInput" placeholder="Type a message..." 
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let connected = false;
        let currentWorkspace = null;
        let username = localStorage.getItem('vega_username') || 'User-' + Math.random().toString(36).substr(2, 5);
        
        document.getElementById('username').value = username;
        
        function updateConnectionStatus(isConnected) {
            const status = document.getElementById('connectionStatus');
            connected = isConnected;
            if (isConnected) {
                status.textContent = '🟢 Connected to Vega Collaboration';
                status.className = 'connection-status connected';
            } else {
                status.textContent = '🔴 Disconnected - Click "Connect" to join';
                status.className = 'connection-status disconnected';
            }
        }
        
        function connect() {
            if (ws) return;
            
            username = document.getElementById('username').value || username;
            localStorage.setItem('vega_username', username);
            
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws?username=${encodeURIComponent(username)}`;
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                updateConnectionStatus(true);
                addChatMessage('system', 'Connected to collaboration server');
            };
            
            ws.onclose = () => {
                updateConnectionStatus(false);
                addChatMessage('system', 'Disconnected from server');
                ws = null;
            };
            
            ws.onerror = (error) => {
                addChatMessage('system', 'Connection error: ' + error);
            };
            
            ws.onmessage = (event) => {
                handleMessage(JSON.parse(event.data));
            };
        }
        
        function disconnect() {
            if (ws) {
                ws.close();
                ws = null;
            }
            updateConnectionStatus(false);
        }
        
        function joinWorkspace() {
            if (!connected) {
                connect();
                setTimeout(joinWorkspace, 1000);
                return;
            }
            
            const workspaceId = document.getElementById('workspaceId').value || 'default';
            currentWorkspace = workspaceId;
            
            sendMessage_internal({
                type: 'join_workspace',
                data: { workspace_id: workspaceId }
            });
        }
        
        function leaveWorkspace() {
            if (!connected || !currentWorkspace) return;
            
            sendMessage_internal({
                type: 'leave_workspace',
                data: {}
            });
            currentWorkspace = null;
        }
        
        function sendMessage() {
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (!message || !connected || !currentWorkspace) return;
            
            sendMessage_internal({
                type: 'chat_message',
                data: { message: message }
            });
            
            input.value = '';
        }
        
        function sendMessage_internal(messageData) {
            if (!ws || !connected) return;
            
            ws.send(JSON.stringify({
                id: Math.random().toString(36).substr(2, 9),
                ...messageData,
                timestamp: new Date().toISOString()
            }));
        }
        
        function handleMessage(message) {
            switch(message.type) {
                case 'chat_message':
                    addChatMessage(message.data.username || 'Unknown', message.data.message);
                    break;
                case 'join_workspace':
                    addChatMessage('system', `${message.data.user.username} joined the workspace`);
                    updateParticipants(message.data);
                    break;
                case 'leave_workspace':
                    addChatMessage('system', `${message.data.user.username} left the workspace`);
                    break;
                case 'document_sync':
                    if (message.data.workspace) {
                        updateWorkspaceState(message.data.workspace);
                    }
                    break;
                case 'user_status':
                    if (message.data.error) {
                        addChatMessage('error', message.data.error);
                    }
                    break;
            }
        }
        
        function addChatMessage(sender, message) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            
            const time = new Date().toLocaleTimeString();
            messageDiv.innerHTML = `
                <div class="message-user">${sender}</div>
                <div>${message}</div>
                <div class="message-time">${time}</div>
            `;
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function updateParticipants(data) {
            // This would be implemented to update the participants list
            // based on workspace data
        }
        
        function updateWorkspaceState(workspace) {
            const participantsList = document.getElementById('participantsList');
            participantsList.innerHTML = '';
            
            if (workspace.participants && workspace.participants.length > 0) {
                workspace.participants.forEach(participant => {
                    const participantDiv = document.createElement('div');
                    participantDiv.className = 'participant';
                    participantDiv.innerHTML = `
                        <div class="status-dot ${participant.status}"></div>
                        <span>${participant.username}</span>
                    `;
                    participantsList.appendChild(participantDiv);
                });
            } else {
                participantsList.innerHTML = '<div class="participant"><div class="status-dot offline"></div><span>No participants</span></div>';
            }
        }
        
        // Auto-connect on page load
        window.addEventListener('load', () => {
            // Don't auto-connect, let user choose when to connect
        });
        
        // Document editing functionality
        const documentEditor = document.getElementById('documentEditor');
        let lastDocumentContent = '';
        
        documentEditor.addEventListener('input', () => {
            if (!connected || !currentWorkspace) return;
            
            const content = documentEditor.value;
            if (content !== lastDocumentContent) {
                // Simple diff - in production, use proper operational transformation
                sendMessage_internal({
                    type: 'document_edit',
                    data: {
                        document_id: 'main',
                        edit: {
                            operation: 'replace',
                            content: content
                        }
                    }
                });
                lastDocumentContent = content;
            }
        });
    </script>
</body>
</html>
    """
    )


@app.get("/api/workspaces")
async def list_workspaces() -> List[Dict[str, Any]]:
    """List all active workspaces"""
    workspaces = []
    for workspace_id in collaboration_manager.workspaces:
        stats = collaboration_manager.get_workspace_stats(workspace_id)
        if stats:
            workspaces.append(stats)
    return workspaces


@app.get("/api/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str) -> Dict[str, Any]:
    """Get workspace details"""
    stats = collaboration_manager.get_workspace_stats(workspace_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return stats


@app.get("/api/users/{user_id}")
async def get_user_status(user_id: str) -> Dict[str, Any]:
    """Get user status and activity"""
    status = collaboration_manager.get_user_status(user_id)
    if not status:
        raise HTTPException(status_code=404, detail="User not found")
    return status


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_connections": len(collaboration_manager.connections),
        "active_workspaces": len(collaboration_manager.workspaces),
    }


@app.post("/api/workspaces/{workspace_id}/broadcast")
async def broadcast_to_workspace(workspace_id: str, message: Dict[str, Any]):
    """Broadcast a message to all users in a workspace"""
    if workspace_id not in collaboration_manager.workspaces:
        raise HTTPException(status_code=404, detail="Workspace not found")

    broadcast_message = CollaborationMessage(
        id=str(uuid.uuid4()),
        type=MessageType.CHAT_MESSAGE,
        sender_id="system",
        workspace_id=workspace_id,
        data=message,
        timestamp=datetime.now(),
    )

    await collaboration_manager._broadcast_to_workspace(workspace_id, broadcast_message)

    return {"status": "broadcasted", "workspace_id": workspace_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
