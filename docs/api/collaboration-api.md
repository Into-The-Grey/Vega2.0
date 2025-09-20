# Vega 2.0 Collaboration API Documentation

## Overview

The Vega 2.0 Collaboration API enables real-time collaboration features including workspace management, document editing, team communication, and voice/video sessions.

**Base URL**: `http://localhost:8000` (development) | `https://api.vega2.example.com` (production)

**WebSocket URL**: `ws://localhost:8000/ws` (development) | `wss://api.vega2.example.com/ws` (production)

## WebSocket Connections

### Collaboration WebSocket

Connect to workspace collaboration:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/collaboration/{workspace_id}?api_key=your_key');

ws.onopen = function() {
    console.log('Connected to workspace');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    handleCollaborationMessage(message);
};
```

**Message Types**:

- `user_joined` - User joined workspace
- `user_left` - User left workspace
- `document_change` - Document was modified
- `cursor_move` - User cursor position changed
- `chat_message` - Chat message sent
- `voice_session_start` - Voice session initiated

### Example Messages

**User Joined**:

```json
{
  "type": "user_joined",
  "user_id": "user_123",
  "username": "alice",
  "workspace_id": "workspace_abc",
  "timestamp": "2025-09-20T10:00:00Z"
}
```

**Document Change**:

```json
{
  "type": "document_change",
  "document_id": "doc_456",
  "change": {
    "type": "insert",
    "position": 45,
    "content": "Hello World",
    "author": "user_123"
  },
  "version": 123,
  "timestamp": "2025-09-20T10:01:00Z"
}
```

**Chat Message**:

```json
{
  "type": "chat_message",
  "message_id": "msg_789",
  "channel_id": "general",
  "author": "user_123",
  "content": "Let's review this document",
  "timestamp": "2025-09-20T10:02:00Z",
  "mentions": ["user_456"]
}
```

## REST API Endpoints

### Workspace Management

#### POST /collaboration/workspaces

Create new workspace

**Request Body**:

```json
{
  "name": "Project Alpha",
  "description": "Collaborative workspace for Project Alpha",
  "privacy": "private",
  "members": ["user_123", "user_456"],
  "permissions": {
    "default_role": "editor",
    "allow_public_read": false
  }
}
```

**Response**:

```json
{
  "workspace_id": "workspace_abc123",
  "name": "Project Alpha",
  "description": "Collaborative workspace for Project Alpha",
  "created_by": "user_123",
  "created_at": "2025-09-20T10:00:00Z",
  "privacy": "private",
  "member_count": 2,
  "document_count": 0
}
```

#### GET /collaboration/workspaces

List user's workspaces

**Parameters**:

- `limit` (query): Max workspaces to return (default: 20)
- `privacy` (query): Filter by privacy (public, private, all)
- `role` (query): Filter by user role (owner, admin, editor, viewer)

**Response**:

```json
{
  "workspaces": [
    {
      "workspace_id": "workspace_abc123",
      "name": "Project Alpha",
      "role": "owner",
      "last_activity": "2025-09-20T09:45:00Z",
      "member_count": 2,
      "document_count": 5,
      "unread_messages": 3
    }
  ],
  "total": 1,
  "has_more": false
}
```

#### GET /collaboration/workspaces/{workspace_id}

Get workspace details

**Response**:

```json
{
  "workspace_id": "workspace_abc123",
  "name": "Project Alpha",
  "description": "Collaborative workspace for Project Alpha",
  "created_by": "user_123",
  "created_at": "2025-09-20T10:00:00Z",
  "privacy": "private",
  "members": [
    {
      "user_id": "user_123",
      "username": "alice",
      "role": "owner",
      "joined_at": "2025-09-20T10:00:00Z",
      "last_seen": "2025-09-20T10:30:00Z",
      "status": "online"
    }
  ],
  "documents": [
    {
      "document_id": "doc_456",
      "title": "Requirements Document",
      "created_by": "user_123",
      "last_modified": "2025-09-20T10:15:00Z",
      "collaborators": 2
    }
  ],
  "permissions": {
    "can_edit": true,
    "can_invite": true,
    "can_delete": true
  }
}
```

### Document Management

#### POST /collaboration/workspaces/{workspace_id}/documents

Create new document

**Request Body**:

```json
{
  "title": "Project Proposal",
  "content": "# Project Proposal\n\nThis is the initial draft...",
  "type": "markdown",
  "permissions": {
    "default_access": "editor"
  }
}
```

**Response**:

```json
{
  "document_id": "doc_789",
  "title": "Project Proposal",
  "created_by": "user_123",
  "created_at": "2025-09-20T10:20:00Z",
  "version": 1,
  "type": "markdown",
  "word_count": 25,
  "character_count": 145
}
```

#### GET /collaboration/documents/{document_id}

Get document content and metadata

**Response**:

```json
{
  "document_id": "doc_789",
  "title": "Project Proposal", 
  "content": "# Project Proposal\n\nThis is the initial draft...",
  "type": "markdown",
  "created_by": "user_123",
  "created_at": "2025-09-20T10:20:00Z",
  "last_modified": "2025-09-20T10:25:00Z",
  "version": 3,
  "collaborators": [
    {
      "user_id": "user_123",
      "username": "alice",
      "role": "owner",
      "last_edit": "2025-09-20T10:25:00Z"
    }
  ],
  "metrics": {
    "word_count": 250,
    "character_count": 1450,
    "edit_count": 12,
    "comment_count": 3
  }
}
```

#### PUT /collaboration/documents/{document_id}

Update document (operational transformation)

**Request Body**:

```json
{
  "operations": [
    {
      "type": "insert",
      "position": 45,
      "content": "\n\n## New Section",
      "author": "user_123"
    },
    {
      "type": "delete", 
      "position": 120,
      "length": 15,
      "author": "user_123"
    }
  ],
  "base_version": 3,
  "client_id": "client_abc123"
}
```

**Response**:

```json
{
  "document_id": "doc_789",
  "new_version": 4,
  "applied_operations": 2,
  "conflicts_resolved": 0,
  "updated_at": "2025-09-20T10:30:00Z"
}
```

### Team Communication

#### POST /collaboration/workspaces/{workspace_id}/channels

Create chat channel

**Request Body**:

```json
{
  "name": "general",
  "description": "General discussion for the workspace",
  "type": "public",
  "members": ["user_123", "user_456"]
}
```

#### GET /collaboration/channels/{channel_id}/messages

Get chat messages

**Parameters**:

- `limit` (query): Max messages (default: 50)
- `before` (query): Get messages before timestamp
- `after` (query): Get messages after timestamp

**Response**:

```json
{
  "messages": [
    {
      "message_id": "msg_123",
      "author": "user_123",
      "content": "Let's review the latest changes",
      "timestamp": "2025-09-20T10:35:00Z",
      "edited": false,
      "reactions": [
        {
          "emoji": "👍",
          "users": ["user_456"],
          "count": 1
        }
      ],
      "thread_count": 2
    }
  ],
  "has_more": false,
  "channel_id": "channel_abc"
}
```

#### POST /collaboration/channels/{channel_id}/messages

Send chat message

**Request Body**:

```json
{
  "content": "Great work on the proposal! 👏",
  "mentions": ["user_456"],
  "thread_id": null,
  "attachments": [
    {
      "type": "image",
      "url": "https://example.com/image.png",
      "filename": "screenshot.png"
    }
  ]
}
```

### Voice/Video Sessions

#### POST /collaboration/workspaces/{workspace_id}/voice-session

Start voice/video session

**Request Body**:

```json
{
  "type": "video",
  "title": "Project Review Meeting",
  "participants": ["user_123", "user_456"],
  "settings": {
    "record": true,
    "max_participants": 10,
    "require_permission": false
  }
}
```

**Response**:

```json
{
  "session_id": "voice_session_abc123",
  "room_url": "https://meet.vega2.example.com/room/abc123",
  "webrtc_config": {
    "ice_servers": [
      {
        "urls": ["stun:stun.vega2.example.com:3478"]
      }
    ]
  },
  "created_at": "2025-09-20T10:40:00Z",
  "expires_at": "2025-09-20T14:40:00Z"
}
```

#### GET /collaboration/voice-sessions/{session_id}

Get voice session details

**Response**:

```json
{
  "session_id": "voice_session_abc123",
  "title": "Project Review Meeting",
  "type": "video",
  "status": "active",
  "participants": [
    {
      "user_id": "user_123",
      "username": "alice",
      "joined_at": "2025-09-20T10:41:00Z",
      "status": "connected",
      "audio": true,
      "video": true
    }
  ],
  "started_at": "2025-09-20T10:40:00Z",
  "duration": 900,
  "recording": {
    "enabled": true,
    "status": "recording",
    "file_size": 125000000
  }
}
```

### Permissions & Access Control

#### PUT /collaboration/workspaces/{workspace_id}/members/{user_id}/role

Update member role

**Request Body**:

```json
{
  "role": "admin",
  "permissions": {
    "can_invite": true,
    "can_delete_messages": true,
    "can_manage_documents": true
  }
}
```

**Response**:

```json
{
  "user_id": "user_456",
  "old_role": "editor", 
  "new_role": "admin",
  "updated_by": "user_123",
  "updated_at": "2025-09-20T10:45:00Z"
}
```

#### GET /collaboration/workspaces/{workspace_id}/permissions

Get workspace permissions matrix

**Response**:

```json
{
  "roles": {
    "owner": {
      "can_edit": true,
      "can_delete": true,
      "can_invite": true,
      "can_manage_roles": true,
      "can_delete_workspace": true
    },
    "admin": {
      "can_edit": true,
      "can_delete": false,
      "can_invite": true, 
      "can_manage_roles": true,
      "can_delete_workspace": false
    },
    "editor": {
      "can_edit": true,
      "can_delete": false,
      "can_invite": false,
      "can_manage_roles": false,
      "can_delete_workspace": false
    },
    "viewer": {
      "can_edit": false,
      "can_delete": false,
      "can_invite": false,
      "can_manage_roles": false,
      "can_delete_workspace": false
    }
  }
}
```

## WebRTC Configuration

For voice/video features, clients need WebRTC configuration:

```javascript
const peerConnection = new RTCPeerConnection({
  iceServers: [
    { urls: 'stun:stun.vega2.example.com:3478' },
    { 
      urls: 'turn:turn.vega2.example.com:3478',
      username: 'user123',
      credential: 'temporary_token'
    }
  ]
});

// Connect to voice session
const ws = new WebSocket('ws://localhost:8000/ws/voice/' + sessionId);

ws.onmessage = async function(event) {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'offer':
      await peerConnection.setRemoteDescription(message.offer);
      const answer = await peerConnection.createAnswer();
      await peerConnection.setLocalDescription(answer);
      ws.send(JSON.stringify({type: 'answer', answer}));
      break;
      
    case 'ice-candidate':
      await peerConnection.addIceCandidate(message.candidate);
      break;
  }
};
```

## Real-time Updates

The collaboration system uses operational transformation for conflict resolution:

**Example Conflict Resolution**:

```json
{
  "document_id": "doc_789",
  "conflict": {
    "base_version": 5,
    "operation_a": {
      "type": "insert",
      "position": 10,
      "content": "Hello ",
      "author": "user_123"
    },
    "operation_b": {
      "type": "delete",
      "position": 8,
      "length": 5,
      "author": "user_456"
    }
  },
  "resolution": {
    "transformed_operations": [
      {
        "type": "insert", 
        "position": 5,
        "content": "Hello ",
        "author": "user_123"
      },
      {
        "type": "delete",
        "position": 8, 
        "length": 5,
        "author": "user_456"
      }
    ],
    "final_version": 7
  }
}
```

## Error Responses

### 404 Workspace Not Found

```json
{
  "error": "Workspace not found",
  "code": 404,
  "workspace_id": "workspace_invalid"
}
```

### 403 Insufficient Permissions

```json
{
  "error": "Insufficient permissions",
  "code": 403,
  "required_role": "editor",
  "current_role": "viewer"
}
```

### 409 Conflict

```json
{
  "error": "Document version conflict",
  "code": 409,
  "current_version": 8,
  "submitted_version": 6,
  "resolution_required": true
}
```
