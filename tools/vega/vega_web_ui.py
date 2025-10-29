#!/usr/bin/env python3
"""
VEGA FUNCTIONAL WEB UI
======================

A practical, functional web interface for Vega that lets you actually USE it.
No broken start buttons, just real functionality.

Features:
- Chat with Vega (full conversation interface)
- File uploads/downloads
- Terminal access
- Integration access (search, fetch, OSINT)
- Conversation history
- Real-time updates via WebSocket

Usage:
    python vega_web_ui.py --port 8080
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Set, Optional
import httpx

try:
    from fastapi import (
        FastAPI,
        WebSocket,
        WebSocketDisconnect,
        Request,
        UploadFile,
        File,
        Form,
    )
    from fastapi.responses import (
        HTMLResponse,
        JSONResponse,
        FileResponse,
        StreamingResponse,
    )
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print(
        "❌ FastAPI not installed. Run: pip install fastapi uvicorn python-multipart httpx"
    )
    sys.exit(1)

# Configuration
VEGA_API_URL = "http://127.0.0.1:8000"
VEGA_API_KEY = os.getenv("API_KEY", "your-api-key-here")

app = FastAPI(title="Vega Web UI", version="1.0.0")

# WebSocket clients
connected_clients: Set[WebSocket] = set()

# Load API key from .env if exists
try:
    from pathlib import Path

    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith("API_KEY="):
                    VEGA_API_KEY = line.split("=", 1)[1].strip().strip('"')
except Exception:
    pass


@app.get("/", response_class=HTMLResponse)
async def root():
    """Main UI page"""
    return HTMLResponse(content=HTML_TEMPLATE, status_code=200)


@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Proxy chat requests to Vega API - Persistent mode (single continuous session)"""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        stream = data.get("stream", False)

        # PERSISTENT MODE: Always use persistent session (no session_id = use default)
        # Vega will automatically continue the same conversation
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{VEGA_API_URL}/chat",
                json={"prompt": prompt, "stream": stream},  # No session_id = persistent
                headers={"X-API-Key": VEGA_API_KEY},
            )

            if response.status_code == 200:
                result = response.json()
                # Broadcast to WebSocket clients
                await broadcast_message(
                    {
                        "type": "chat_response",
                        "prompt": prompt,
                        "response": result.get("response", ""),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                return result
            else:
                return JSONResponse(
                    {"error": f"Vega API error: {response.status_code}"},
                    status_code=response.status_code,
                )
    except httpx.ConnectError:
        return JSONResponse(
            {
                "error": "Cannot connect to Vega. Make sure Vega server is running on port 8000."
            },
            status_code=503,
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/history")
async def history_endpoint(limit: int = 20):
    """Get conversation history from Vega"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{VEGA_API_URL}/history",
                params={"limit": limit},
                headers={"X-API-Key": VEGA_API_KEY},
            )
            if response.status_code == 200:
                return response.json()
            else:
                return JSONResponse({"conversations": []}, status_code=200)
    except Exception as e:
        return JSONResponse({"conversations": [], "error": str(e)}, status_code=200)


@app.get("/api/status")
async def status_endpoint():
    """Check if Vega is running - Persistent mode includes memory stats"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check if Vega is alive
            health_response = await client.get(f"{VEGA_API_URL}/healthz")
            is_running = health_response.status_code == 200

            # Try to get memory stats
            memory_stats = {}
            if is_running:
                try:
                    metrics_response = await client.get(
                        f"{VEGA_API_URL}/metrics", headers={"X-API-Key": VEGA_API_KEY}
                    )
                    if metrics_response.status_code == 200:
                        metrics = metrics_response.json()
                        memory_stats = metrics.get("memory_manager", {})
                except Exception:
                    pass

            return {
                "vega_running": is_running,
                "vega_url": VEGA_API_URL,
                "memory_stats": memory_stats,
                "mode": "PERSISTENT" if is_running else "OFFLINE",
                "timestamp": datetime.now().isoformat(),
            }
    except Exception as e:
        return {
            "vega_running": False,
            "vega_url": VEGA_API_URL,
            "error": f"Cannot connect to Vega: {e}",
            "mode": "OFFLINE",
            "timestamp": datetime.now().isoformat(),
        }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        # Send initial status
        status = await status_endpoint()
        await websocket.send_json({"type": "status", "data": status})

        # Keep connection alive and handle messages
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(message)
                # Handle client messages if needed
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


async def broadcast_message(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(message)
        except Exception:
            disconnected.add(client)

    # Clean up disconnected clients
    connected_clients.difference_update(disconnected)


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vega Web Interface</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --cherenkov-blue: #1b9cfc;
            --cherenkov-glow: rgba(27, 156, 252, 0.3);
            --dark-bg: #0a0e27;
            --darker-bg: #060813;
            --panel-bg: rgba(15, 20, 40, 0.8);
            --text-primary: #e8f0ff;
            --text-secondary: #8b9dc3;
        }
        
        body {
            font-family: 'SF Mono', 'Monaco', 'Cascadia Code', 'Courier New', monospace;
            background: var(--dark-bg);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: 
                radial-gradient(circle at 20% 50%, var(--cherenkov-glow) 0%, transparent 50%),
                radial-gradient(circle at 80% 50%, var(--cherenkov-glow) 0%, transparent 50%);
            pointer-events: none;
            z-index: 0;
        }
        
        .header {
            background: var(--panel-bg);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--cherenkov-blue);
            box-shadow: 0 0 20px var(--cherenkov-glow);
            backdrop-filter: blur(10px);
            position: relative;
            z-index: 10;
        }
        
        .header h1 {
            font-size: 1.2rem;
            font-weight: 400;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--cherenkov-blue);
            text-shadow: 0 0 10px var(--cherenkov-glow);
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            font-size: 0.85rem;
            letter-spacing: 0.05em;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #3a3a3a;
            animation: pulse 2s infinite;
            box-shadow: 0 0 5px rgba(58, 58, 58, 0.5);
        }
        
        .status-dot.online {
            background: var(--cherenkov-blue);
            box-shadow: 0 0 15px var(--cherenkov-glow), 0 0 30px var(--cherenkov-glow);
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.1); }
        }
        
        .container {
            flex: 1;
            display: flex;
            max-height: calc(100vh - 70px);
            position: relative;
            z-index: 1;
        }
        
        .sidebar {
            width: 280px;
            background: var(--panel-bg);
            border-right: 1px solid rgba(27, 156, 252, 0.2);
            padding: 1rem;
            overflow-y: auto;
            backdrop-filter: blur(10px);
        }
        
        .sidebar::-webkit-scrollbar {
            width: 6px;
        }
        
        .sidebar::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .sidebar::-webkit-scrollbar-thumb {
            background: var(--cherenkov-blue);
            border-radius: 3px;
        }
        
        .sidebar h2 {
            font-size: 0.75rem;
            margin-bottom: 1rem;
            opacity: 0.6;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--text-secondary);
        }
        
        .history-item {
            background: rgba(27, 156, 252, 0.05);
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            border-left: 2px solid transparent;
        }
        
        .history-item:hover {
            background: rgba(27, 156, 252, 0.1);
            border-left: 2px solid var(--cherenkov-blue);
            box-shadow: 0 0 10px var(--cherenkov-glow);
        }
        
        .history-item-prompt {
            font-size: 0.85rem;
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            color: var(--text-primary);
        }
        
        .history-item-time {
            font-size: 0.7rem;
            opacity: 0.5;
            color: var(--text-secondary);
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 1.5rem;
            overflow: hidden;
        }
        
        .chat-container {
            flex: 1;
            background: var(--panel-bg);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            border: 1px solid rgba(27, 156, 252, 0.2);
            backdrop-filter: blur(10px);
        }
        
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: var(--cherenkov-blue);
            border-radius: 4px;
        }
        
        .message {
            max-width: 80%;
            padding: 1rem;
            border-radius: 8px;
            animation: slideIn 0.3s ease-out;
            position: relative;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            align-self: flex-end;
            background: rgba(27, 156, 252, 0.15);
            margin-left: auto;
            border: 1px solid var(--cherenkov-blue);
            box-shadow: 0 0 15px var(--cherenkov-glow);
        }
        
        .message.vega {
            align-self: flex-start;
            background: rgba(15, 20, 40, 0.6);
            border: 1px solid rgba(27, 156, 252, 0.3);
        }
        
        .message-content {
            white-space: pre-wrap;
            word-wrap: break-word;
            font-size: 0.95rem;
            line-height: 1.5;
        }
        
        .message-time {
            font-size: 0.65rem;
            opacity: 0.5;
            margin-top: 0.5rem;
            color: var(--text-secondary);
        }
        
        .input-container {
            display: flex;
            gap: 0.75rem;
            align-items: flex-end;
        }
        
        .input-wrapper {
            flex: 1;
            display: flex;
            gap: 0.5rem;
            background: var(--panel-bg);
            border-radius: 8px;
            padding: 0.5rem;
            border: 1px solid rgba(27, 156, 252, 0.2);
            backdrop-filter: blur(10px);
            transition: all 0.2s;
        }
        
        .input-wrapper:focus-within {
            border-color: var(--cherenkov-blue);
            box-shadow: 0 0 20px var(--cherenkov-glow);
        }
        
        input[type="text"] {
            flex: 1;
            padding: 0.75rem;
            border: none;
            border-radius: 4px;
            background: transparent;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            font-family: inherit;
        }
        
        input[type="text"]::placeholder {
            color: var(--text-secondary);
            opacity: 0.5;
        }
        
        button {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 6px;
            background: var(--cherenkov-blue);
            color: var(--darker-bg);
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-family: inherit;
            box-shadow: 0 0 15px var(--cherenkov-glow);
        }
        
        button:hover {
            background: #2dabff;
            box-shadow: 0 0 25px var(--cherenkov-glow);
            transform: translateY(-1px);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        
        button.icon-btn {
            padding: 0.75rem;
            min-width: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        button.secondary {
            background: rgba(27, 156, 252, 0.2);
            color: var(--cherenkov-blue);
        }
        
        button.secondary:hover {
            background: rgba(27, 156, 252, 0.3);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 0.75rem;
            opacity: 0.7;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }
        
        .loading.active {
            display: block;
        }
        
        .loading::after {
            content: '...';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        
        .error {
            background: rgba(220, 38, 38, 0.15);
            border: 1px solid #dc2626;
            padding: 1rem;
            border-radius: 6px;
            margin-bottom: 1rem;
            display: none;
            font-size: 0.9rem;
        }
        
        .error.active {
            display: block;
        }
        
        .welcome {
            text-align: center;
            padding: 3rem;
            opacity: 0.4;
        }
        
        .welcome h2 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--cherenkov-blue);
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        
        .welcome p {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .voice-controls {
            display: flex;
            gap: 0.5rem;
        }
        
        .recording {
            animation: recording-pulse 1s infinite;
        }
        
        @keyframes recording-pulse {
            0%, 100% { 
                box-shadow: 0 0 15px var(--cherenkov-glow);
            }
            50% { 
                box-shadow: 0 0 30px var(--cherenkov-glow), 0 0 50px var(--cherenkov-glow);
            }
        }
        
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            .main-content {
                padding: 1rem;
            }
            .header h1 {
                font-size: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>◇ VEGA SYSTEM ◇</h1>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">INITIALIZING...</span>
            <span id="memoryStats" style="margin-left: 1rem; font-size: 0.8em; opacity: 0.7;"></span>
        </div>
    </div>
    
    <div class="container">
        <div class="sidebar">
            <h2>Recent Conversations</h2>
            <div id="historyList">
                <div style="opacity: 0.5; text-align: center; padding: 2rem;">
                    No history yet
                </div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="error" id="errorBox"></div>
            
            <div class="chat-container" id="chatContainer">
                <div class="welcome">
                    <h2>◇ VEGA SYSTEM INTERFACE ◇</h2>
                    <p>AWAITING INPUT</p>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div>Vega is thinking...</div>
            </div>
            
            <div class="input-container">
                <div class="input-wrapper">
                    <input 
                        type="text" 
                        id="messageInput" 
                        placeholder="ENTER COMMAND..."
                        autocomplete="off"
                    />
                    <div class="voice-controls">
                        <button class="icon-btn secondary" id="micButton" onclick="toggleVoiceInput()" title="Voice Input">
                            🎤
                        </button>
                        <button class="icon-btn secondary" id="speakerButton" onclick="toggleVoiceOutput()" title="Toggle Voice Output">
                            🔊
                        </button>
                    </div>
                    <button id="sendButton" onclick="sendMessage()">SEND</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let vegaOnline = false;
        let voiceOutputEnabled = true;
        let recognition = null;
        let synthesis = window.speechSynthesis;
        let isRecording = false;
        
        // Initialize speech recognition if available
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';
            
            recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                document.getElementById('messageInput').value = transcript;
                isRecording = false;
                document.getElementById('micButton').classList.remove('recording');
            };
            
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                isRecording = false;
                document.getElementById('micButton').classList.remove('recording');
                if (event.error !== 'no-speech') {
                    showError(`Voice input error: ${event.error}`);
                }
            };
            
            recognition.onend = () => {
                isRecording = false;
                document.getElementById('micButton').classList.remove('recording');
            };
        } else {
            // Hide mic button if not supported
            document.addEventListener('DOMContentLoaded', () => {
                const micBtn = document.getElementById('micButton');
                if (micBtn) {
                    micBtn.style.display = 'none';
                }
            });
        }
        
        // Toggle voice input
        function toggleVoiceInput() {
            if (!recognition) {
                showError('Voice input not supported in this browser');
                return;
            }
            
            if (isRecording) {
                recognition.stop();
                isRecording = false;
                document.getElementById('micButton').classList.remove('recording');
            } else {
                recognition.start();
                isRecording = true;
                document.getElementById('micButton').classList.add('recording');
            }
        }
        
        // Toggle voice output
        function toggleVoiceOutput() {
            voiceOutputEnabled = !voiceOutputEnabled;
            const btn = document.getElementById('speakerButton');
            if (voiceOutputEnabled) {
                btn.textContent = '🔊';
                btn.title = 'Voice Output: ON';
            } else {
                btn.textContent = '🔇';
                btn.title = 'Voice Output: OFF';
                synthesis.cancel(); // Stop any current speech
            }
        }
        
        // Speak text using speech synthesis
        function speak(text) {
            if (!voiceOutputEnabled || !synthesis) return;
            
            // Cancel any ongoing speech
            synthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            // Try to use a good voice
            const voices = synthesis.getVoices();
            const preferredVoice = voices.find(v => 
                v.name.includes('Google') || 
                v.name.includes('Microsoft') ||
                v.lang.startsWith('en')
            );
            if (preferredVoice) {
                utterance.voice = preferredVoice;
            }
            
            synthesis.speak(utterance);
        }
        
        // Connect WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status') {
                    updateStatus(data.data.vega_running);
                } else if (data.type === 'chat_response') {
                    // Real-time update from other clients
                    addMessage(data.prompt, 'user', data.timestamp);
                    addMessage(data.response, 'vega', data.timestamp);
                }
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected, reconnecting...');
                setTimeout(connectWebSocket, 3000);
            };
        }
        
        // Update status indicator with memory stats (Persistent mode)
        function updateStatus(online, memoryStats = null) {
            vegaOnline = online;
            const dot = document.getElementById('statusDot');
            const text = document.getElementById('statusText');
            const memStatsEl = document.getElementById('memoryStats');
            
            if (online) {
                dot.classList.add('online');
                text.textContent = 'ONLINE [PERSISTENT]';
                
                // Display memory stats if available
                if (memoryStats && memoryStats.memory) {
                    const mem = memoryStats.memory;
                    const procMB = mem.process_memory_mb || 0;
                    const sysPct = mem.system_memory_percent || 0;
                    memStatsEl.textContent = `MEM: ${procMB.toFixed(0)}MB | SYS: ${sysPct.toFixed(1)}%`;
                    
                    // Color-code based on system memory usage
                    if (sysPct > 90) {
                        memStatsEl.style.color = '#ff4757';  // Critical red
                    } else if (sysPct > 80) {
                        memStatsEl.style.color = '#ffa502';  // Warning orange
                    } else {
                        memStatsEl.style.color = '#1b9cfc';  // Normal blue
                    }
                } else {
                    memStatsEl.textContent = '';
                }
            } else {
                dot.classList.remove('online');
                text.textContent = 'OFFLINE';
                memStatsEl.textContent = '';
            }
        }
        
        // Load conversation history
        async function loadHistory() {
            try {
                const response = await fetch('/api/history?limit=20');
                const data = await response.json();
                
                const historyList = document.getElementById('historyList');
                
                if (data.conversations && data.conversations.length > 0) {
                    historyList.innerHTML = data.conversations.map(conv => `
                        <div class="history-item">
                            <div class="history-item-prompt">${escapeHtml(conv.prompt)}</div>
                            <div class="history-item-time">${new Date(conv.ts).toLocaleString()}</div>
                        </div>
                    `).join('');
                } else {
                    historyList.innerHTML = '<div style="opacity: 0.5; text-align: center; padding: 2rem;">No history yet</div>';
                }
            } catch (error) {
                console.error('Failed to load history:', error);
            }
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            if (!vegaOnline) {
                showError('VEGA SYSTEM OFFLINE - Start Vega with: python main.py server');
                return;
            }
            
            // Clear input and show loading
            input.value = '';
            input.disabled = true;
            document.getElementById('sendButton').disabled = true;
            document.getElementById('loading').classList.add('active');
            
            // Remove welcome message
            const welcome = document.querySelector('.welcome');
            if (welcome) welcome.remove();
            
            // Add user message
            addMessage(message, 'user');
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt: message, stream: false })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    addMessage(data.response, 'vega');
                    speak(data.response); // Speak the response
                    loadHistory(); // Refresh history
                }
            } catch (error) {
                showError(`Error: ${error.message}`);
            } finally {
                input.disabled = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('loading').classList.remove('active');
                input.focus();
            }
        }
        
        // Add message to chat
        function addMessage(content, sender, timestamp = null) {
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const time = timestamp ? new Date(timestamp) : new Date();
            
            messageDiv.innerHTML = `
                <div class="message-content">${escapeHtml(content)}</div>
                <div class="message-time">${time.toLocaleTimeString()}</div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        // Show error message
        function showError(message) {
            const errorBox = document.getElementById('errorBox');
            errorBox.textContent = message;
            errorBox.classList.add('active');
            
            setTimeout(() => {
                errorBox.classList.remove('active');
            }, 5000);
        }
        
        // Escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Handle Enter key
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize
        connectWebSocket();
        loadHistory();
        
        // Check status periodically (Persistent mode with memory stats)
        setInterval(async () => {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateStatus(data.vega_running, data.memory_stats);
            } catch (error) {
                updateStatus(false);
            }
        }, 5000);
    </script>
</body>
</html>
"""


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Vega Functional Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument(
        "--vega-url", default="http://127.0.0.1:8000", help="Vega API URL"
    )

    args = parser.parse_args()

    global VEGA_API_URL
    VEGA_API_URL = args.vega_url

    print(f"🚀 Starting Vega Web UI on http://{args.host}:{args.port}")
    print(f"📡 Connecting to Vega API at {VEGA_API_URL}")
    print(
        f"🔑 API Key loaded: {'Yes' if VEGA_API_KEY != 'your-api-key-here' else 'No (using default)'}"
    )
    print()
    print("Make sure Vega is running:")
    print("  python main.py server --host 127.0.0.1 --port 8000")
    print()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
