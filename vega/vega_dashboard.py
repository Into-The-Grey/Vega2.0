#!/usr/bin/env python3
"""
VEGA WEB DASHBOARD
==================

Modern, beautiful web dashboard for the Vega Ambient AI system.
Features real-time updates, interactive controls, and gorgeous visuals.

Requirements:
    pip install fastapi uvicorn websockets jinja2 python-multipart

Usage:
    python vega_dashboard.py                # Start web dashboard on port 8080
    python vega_dashboard.py --port 9090    # Custom port
"""

import os
import sys
import json
import asyncio
import psutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import subprocess

try:
    from fastapi import (
        FastAPI,
        WebSocket,
        WebSocketDisconnect,
        Request,
        HTTPException,
        Form,
    )
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
except ImportError:
    print(
        "❌ FastAPI not installed. Run: pip install fastapi uvicorn websockets jinja2 python-multipart"
    )
    sys.exit(1)

# FastAPI app
app = FastAPI(title="Vega Ambient AI Dashboard", version="2.0.0")

# Mount static files
from fastapi.staticfiles import StaticFiles

static_dir = Path.cwd() / "ui" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global state
connected_clients: Set[WebSocket] = set()
vega_status = {}


class VegaDashboardStatus:
    """Enhanced status tracking for dashboard"""

    def __init__(self):
        self.state_dir = Path.cwd() / "vega_state"
        self.state_dir.mkdir(exist_ok=True)

        # Initialize status
        self.status = {
            "is_running": False,
            "mode": "unknown",
            "uptime_seconds": 0,
            "last_interaction": "never",
            "user_presence": "unknown",
            "energy_level": 1.0,
            "silence_protocol": "standard",
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "gpu_usage": {},
            "recent_thoughts": [],
            "interaction_count": 0,
            "presence_history": [],
            "conversation_history": [],
            "system_health": 100.0,
            "total_conversations": 0,
            "avg_response_time": 0.0,
            "errors_today": 0,
            "timestamp": datetime.now().isoformat(),
        }

    def refresh(self):
        """Refresh all status information"""
        try:
            self.status["timestamp"] = datetime.now().isoformat()

            # Check daemon status
            self.status["is_running"] = self._check_daemon_running()

            # Load state files
            self._load_loop_state()
            self._load_presence_state()
            self._load_personality_state()
            self._load_conversation_history()
            self._load_system_metrics()

            # Calculate derived metrics
            self._calculate_health_metrics()

        except Exception as e:
            print(f"Error refreshing status: {e}")
            self.status["errors_today"] += 1

    def _check_daemon_running(self) -> bool:
        """Check if vega_loop.py is running"""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                if "python" in proc.info["name"].lower():
                    cmdline = " ".join(proc.info["cmdline"] or [])
                    if "vega_loop.py" in cmdline:
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    def _load_loop_state(self):
        """Load vega loop state"""
        state_file = self.state_dir / "loop_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    self.status.update(
                        {
                            "mode": data.get("vega_mode", "unknown"),
                            "uptime_seconds": data.get("uptime_seconds", 0),
                            "last_interaction": data.get(
                                "last_interaction_time", "never"
                            ),
                            "energy_level": data.get("energy_level", 1.0),
                            "silence_protocol": data.get(
                                "silence_protocol", "standard"
                            ),
                        }
                    )
            except Exception:
                pass

    def _load_presence_state(self):
        """Load user presence history"""
        presence_file = self.state_dir / "presence_history.jsonl"
        if presence_file.exists():
            try:
                history = []
                with open(presence_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-50:]:  # Last 50 entries
                        data = json.loads(line)
                        history.append(
                            {
                                "timestamp": data.get("timestamp", ""),
                                "presence_state": data.get("presence_state", "unknown"),
                                "active_application": data.get(
                                    "active_application", "unknown"
                                ),
                                "activity_level": data.get("activity_level", 0.0),
                            }
                        )

                self.status["presence_history"] = history
                if history:
                    self.status["user_presence"] = history[-1]["presence_state"]
            except Exception:
                pass

    def _load_personality_state(self):
        """Load personality thoughts"""
        personality_file = self.state_dir / "personality_memory.jsonl"
        if personality_file.exists():
            try:
                thoughts = []
                with open(personality_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-20:]:  # Last 20 thoughts
                        data = json.loads(line)
                        thoughts.append(
                            {
                                "content": data.get("content", ""),
                                "mode": data.get("mode", "unknown"),
                                "timestamp": data.get("generated_at", ""),
                                "trigger": data.get("trigger", "unknown"),
                                "confidence": data.get("confidence", 0.0),
                            }
                        )

                self.status["recent_thoughts"] = thoughts
            except Exception:
                pass

    def _load_conversation_history(self):
        """Load conversation history from database"""
        try:
            import sqlite3

            db_path = Path.cwd() / "vega.db"
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.cursor()

                    # Get total conversations count
                    cursor.execute("SELECT COUNT(*) FROM conversations")
                    self.status["total_conversations"] = cursor.fetchone()[0]

                    # Get recent conversations
                    cursor.execute(
                        """
                        SELECT prompt, response, ts, session_id 
                        FROM conversations 
                        ORDER BY ts DESC 
                        LIMIT 10
                    """
                    )

                    conversations = []
                    for row in cursor.fetchall():
                        conversations.append(
                            {
                                "prompt": row[0][:100]
                                + ("..." if len(row[0]) > 100 else ""),
                                "response": row[1][:200]
                                + ("..." if len(row[1]) > 200 else ""),
                                "timestamp": row[2],
                                "session_id": row[3],
                            }
                        )

                    self.status["conversation_history"] = conversations
        except Exception:
            pass

    def _load_system_metrics(self):
        """Load current system metrics"""
        try:
            self.status["cpu_usage"] = psutil.cpu_percent(interval=1)
            self.status["memory_usage"] = psutil.virtual_memory().percent

            # GPU metrics
            gpu_usage = {}
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode()
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                    gpu_usage[name] = {
                        "utilization": util.gpu,
                        "memory_used": memory_info.used,
                        "memory_total": memory_info.total,
                        "memory_percent": (memory_info.used / memory_info.total) * 100,
                    }
            except:
                pass

            self.status["gpu_usage"] = gpu_usage

        except Exception:
            pass

    def _calculate_health_metrics(self):
        """Calculate system health score"""
        health_score = 100.0

        # Reduce health for high resource usage
        if self.status["cpu_usage"] > 80:
            health_score -= 20
        elif self.status["cpu_usage"] > 60:
            health_score -= 10

        if self.status["memory_usage"] > 80:
            health_score -= 20
        elif self.status["memory_usage"] > 60:
            health_score -= 10

        # Reduce health if not running
        if not self.status["is_running"]:
            health_score -= 30

        # Reduce health for low energy
        if self.status["energy_level"] < 0.3:
            health_score -= 15

        self.status["system_health"] = max(0.0, health_score)


# Global dashboard status
dashboard_status = VegaDashboardStatus()


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Main dashboard page"""
    return get_dashboard_html()


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    dashboard_status.refresh()
    return JSONResponse(dashboard_status.status)


@app.post("/api/command/{action}")
async def execute_command(action: str):
    """Execute system commands"""
    try:
        if action == "start":
            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--start"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return {"success": True, "message": "Vega system started successfully"}
            else:
                return {
                    "success": False,
                    "message": f"Failed to start: {result.stderr}",
                }

        elif action == "stop":
            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--stop"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return {"success": True, "message": "Vega system stopped"}

        elif action == "force_interaction":
            result = subprocess.run(
                [sys.executable, "vega_loop.py", "--force-prompt"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return {"success": True, "message": "Interaction triggered"}

        elif action == "refresh":
            dashboard_status.refresh()
            return {"success": True, "message": "Status refreshed"}

        else:
            return {"success": False, "message": f"Unknown action: {action}"}

    except subprocess.TimeoutExpired:
        return {"success": False, "message": "Command timed out"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    connected_clients.add(websocket)

    try:
        while True:
            # Send status update every 5 seconds
            dashboard_status.refresh()
            await websocket.send_json(
                {"type": "status_update", "data": dashboard_status.status}
            )
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        connected_clients.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        connected_clients.discard(websocket)


async def broadcast_update(message: dict):
    """Broadcast update to all connected clients"""
    disconnected = set()
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.add(client)

    # Remove disconnected clients
    connected_clients -= disconnected


def get_dashboard_html() -> str:
    """Generate the dashboard HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🤖 Vega Ambient AI Dashboard</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --secondary-color: #1e40af;
            --success-color: #059669;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --border-color: #334155;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.2);
        }
        
        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 10px;
        }
        
        .header .subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
        }
        
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
            margin-top: 15px;
        }
        
        .status-running {
            background: rgba(5, 150, 105, 0.2);
            color: var(--success-color);
        }
        
        .status-stopped {
            background: rgba(220, 38, 38, 0.2);
            color: var(--error-color);
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 15px;
            padding: 25px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding: 10px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
        }
        
        .metric-label {
            color: var(--text-secondary);
            font-size: 0.9rem;
        }
        
        .metric-value {
            font-weight: 600;
            font-size: 1rem;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 8px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--success-color), var(--warning-color));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        
        .controls {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }
        
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
            text-decoration: none;
            font-size: 0.9rem;
        }
        
        .btn-primary {
            background: var(--primary-color);
            color: white;
        }
        
        .btn-success {
            background: var(--success-color);
            color: white;
        }
        
        .btn-warning {
            background: var(--warning-color);
            color: white;
        }
        
        .btn-danger {
            background: var(--error-color);
            color: white;
        }
        
        .btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .thoughts-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .thought-item {
            padding: 12px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 3px solid var(--primary-color);
        }
        
        .thought-meta {
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 5px;
        }
        
        .presence-chart {
            display: flex;
            gap: 2px;
            height: 40px;
            align-items: end;
            margin-top: 10px;
        }
        
        .presence-bar {
            flex: 1;
            background: var(--primary-color);
            border-radius: 2px;
            min-height: 4px;
            opacity: 0.7;
            transition: all 0.2s;
        }
        
        .presence-bar:hover {
            opacity: 1;
        }
        
        .timestamp {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 20px;
        }
        
        .health-score {
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
            margin: 10px 0;
        }
        
        .health-good { color: var(--success-color); }
        .health-warning { color: var(--warning-color); }
        .health-critical { color: var(--error-color); }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .grid {
                grid-template-columns: 1fr;
            }
            
            .controls {
                justify-content: center;
            }
            
            .btn {
                flex: 1;
                justify-content: center;
            }
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: var(--primary-color);
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .connection-status {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            z-index: 1000;
        }
        
        .connected {
            background: rgba(5, 150, 105, 0.9);
            color: white;
        }
        
        .disconnected {
            background: rgba(220, 38, 38, 0.9);
            color: white;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        🔄 Connecting...
    </div>
    
    <div class="container">
        <div class="header">
            <h1>🤖 Vega Ambient AI Dashboard</h1>
            <p class="subtitle">Real-time monitoring and control for your AI companion</p>
            <div class="status-indicator" id="systemStatus">
                <div class="loading"></div>
                <span>Loading...</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-success" onclick="executeCommand('start')">
                ▶️ Start System
            </button>
            <button class="btn btn-danger" onclick="executeCommand('stop')">
                ⏹️ Stop System
            </button>
            <button class="btn btn-primary" onclick="executeCommand('force_interaction')">
                💬 Force Chat
            </button>
            <button class="btn btn-warning" onclick="executeCommand('refresh')">
                🔄 Refresh
            </button>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">
                    🎯 System Status
                </div>
                <div class="metric">
                    <span class="metric-label">Mode</span>
                    <span class="metric-value" id="systemMode">Unknown</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value" id="systemUptime">0s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">User Presence</span>
                    <span class="metric-value" id="userPresence">Unknown</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Energy Level</span>
                    <span class="metric-value" id="energyLevel">100%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" id="energyProgress"></div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    🏥 System Health
                </div>
                <div class="health-score" id="healthScore">100</div>
                <div class="metric">
                    <span class="metric-label">CPU Usage</span>
                    <span class="metric-value" id="cpuUsage">0%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" id="cpuProgress"></div>
                    </div>
                </div>
                <div class="metric">
                    <span class="metric-label">Memory Usage</span>
                    <span class="metric-value" id="memoryUsage">0%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" id="memoryProgress"></div>
                    </div>
                </div>
                <div id="gpuMetrics"></div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    👤 User Presence
                </div>
                <div class="presence-chart" id="presenceChart">
                    <!-- Presence history bars will be generated here -->
                </div>
                <div class="metric">
                    <span class="metric-label">Current State</span>
                    <span class="metric-value" id="currentPresence">Unknown</span>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    🧠 Recent Thoughts
                </div>
                <div class="thoughts-list" id="thoughtsList">
                    <p style="color: var(--text-secondary); text-align: center; padding: 20px;">
                        No recent thoughts...
                    </p>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    📊 Statistics
                </div>
                <div class="metric">
                    <span class="metric-label">Total Conversations</span>
                    <span class="metric-value" id="totalConversations">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Interactions Today</span>
                    <span class="metric-value" id="interactionsToday">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Errors Today</span>
                    <span class="metric-value" id="errorsToday">0</span>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">
                    💬 Recent Conversations
                </div>
                <div id="conversationsList">
                    <p style="color: var(--text-secondary); text-align: center; padding: 20px;">
                        No recent conversations...
                    </p>
                </div>
            </div>
        </div>
        
        <div class="timestamp" id="lastUpdate">
            Last updated: Never
        </div>
    </div>
    
    <script>
        let websocket = null;
        let isConnected = false;
        
        // Initialize WebSocket connection
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function(event) {
                isConnected = true;
                updateConnectionStatus();
                console.log('Connected to Vega Dashboard');
            };
            
            websocket.onmessage = function(event) {
                const message = JSON.parse(event.data);
                if (message.type === 'status_update') {
                    updateDashboard(message.data);
                }
            };
            
            websocket.onclose = function(event) {
                isConnected = false;
                updateConnectionStatus();
                console.log('Disconnected from Vega Dashboard');
                
                // Attempt to reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateConnectionStatus() {
            const statusEl = document.getElementById('connectionStatus');
            if (isConnected) {
                statusEl.className = 'connection-status connected';
                statusEl.innerHTML = '🟢 Connected';
            } else {
                statusEl.className = 'connection-status disconnected';
                statusEl.innerHTML = '🔴 Disconnected';
            }
        }
        
        function updateDashboard(status) {
            // System status
            const systemStatus = document.getElementById('systemStatus');
            if (status.is_running) {
                systemStatus.className = 'status-indicator status-running';
                systemStatus.innerHTML = '🟢 <span>System Running</span>';
            } else {
                systemStatus.className = 'status-indicator status-stopped';
                systemStatus.innerHTML = '🔴 <span>System Stopped</span>';
            }
            
            // Update metrics
            document.getElementById('systemMode').textContent = status.mode.charAt(0).toUpperCase() + status.mode.slice(1);
            document.getElementById('systemUptime').textContent = formatUptime(status.uptime_seconds);
            document.getElementById('userPresence').textContent = status.user_presence.charAt(0).toUpperCase() + status.user_presence.slice(1);
            document.getElementById('currentPresence').textContent = status.user_presence.charAt(0).toUpperCase() + status.user_presence.slice(1);
            
            // Energy level
            const energyPercent = Math.round(status.energy_level * 100);
            document.getElementById('energyLevel').textContent = energyPercent + '%';
            document.getElementById('energyProgress').style.width = energyPercent + '%';
            
            // System health
            const healthScore = Math.round(status.system_health);
            const healthEl = document.getElementById('healthScore');
            healthEl.textContent = healthScore;
            healthEl.className = 'health-score ' + 
                (healthScore >= 80 ? 'health-good' : 
                 healthScore >= 60 ? 'health-warning' : 'health-critical');
            
            // Resource usage
            document.getElementById('cpuUsage').textContent = status.cpu_usage.toFixed(1) + '%';
            document.getElementById('cpuProgress').style.width = status.cpu_usage + '%';
            
            document.getElementById('memoryUsage').textContent = status.memory_usage.toFixed(1) + '%';
            document.getElementById('memoryProgress').style.width = status.memory_usage + '%';
            
            // GPU metrics
            const gpuContainer = document.getElementById('gpuMetrics');
            gpuContainer.innerHTML = '';
            for (const [gpuName, gpuData] of Object.entries(status.gpu_usage)) {
                const gpuDiv = document.createElement('div');
                gpuDiv.className = 'metric';
                gpuDiv.innerHTML = `
                    <span class="metric-label">GPU (${gpuName.split(' ').pop()})</span>
                    <span class="metric-value">${gpuData.utilization}%</span>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${gpuData.utilization}%"></div>
                    </div>
                `;
                gpuContainer.appendChild(gpuDiv);
            }
            
            // Recent thoughts
            updateThoughts(status.recent_thoughts);
            
            // Presence chart
            updatePresenceChart(status.presence_history);
            
            // Statistics
            document.getElementById('totalConversations').textContent = status.total_conversations;
            document.getElementById('interactionsToday').textContent = status.interaction_count;
            document.getElementById('errorsToday').textContent = status.errors_today;
            
            // Conversations
            updateConversations(status.conversation_history);
            
            // Last update time
            document.getElementById('lastUpdate').textContent = 
                'Last updated: ' + new Date(status.timestamp).toLocaleTimeString();
        }
        
        function updateThoughts(thoughts) {
            const container = document.getElementById('thoughtsList');
            if (!thoughts || thoughts.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No recent thoughts...</p>';
                return;
            }
            
            container.innerHTML = '';
            thoughts.slice(-5).forEach(thought => {
                const div = document.createElement('div');
                div.className = 'thought-item';
                div.innerHTML = `
                    <div class="thought-meta">
                        ${new Date(thought.timestamp).toLocaleTimeString()} • ${thought.mode}
                    </div>
                    <div>${thought.content}</div>
                `;
                container.appendChild(div);
            });
        }
        
        function updatePresenceChart(history) {
            const container = document.getElementById('presenceChart');
            if (!history || history.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary); font-size: 0.8rem;">No presence data</p>';
                return;
            }
            
            container.innerHTML = '';
            const maxBars = 24; // Show last 24 data points
            const recentHistory = history.slice(-maxBars);
            
            recentHistory.forEach(entry => {
                const bar = document.createElement('div');
                bar.className = 'presence-bar';
                
                // Height based on activity level
                const height = Math.max(10, (entry.activity_level || 0.1) * 100);
                bar.style.height = height + '%';
                
                // Color based on presence state
                const colors = {
                    'active': '#059669',
                    'idle': '#d97706',
                    'away': '#dc2626',
                    'unknown': '#6b7280'
                };
                bar.style.background = colors[entry.presence_state] || colors.unknown;
                
                // Tooltip
                bar.title = `${entry.presence_state} at ${new Date(entry.timestamp).toLocaleTimeString()}`;
                
                container.appendChild(bar);
            });
        }
        
        function updateConversations(conversations) {
            const container = document.getElementById('conversationsList');
            if (!conversations || conversations.length === 0) {
                container.innerHTML = '<p style="color: var(--text-secondary); text-align: center; padding: 20px;">No recent conversations...</p>';
                return;
            }
            
            container.innerHTML = '';
            conversations.slice(0, 3).forEach(conv => {
                const div = document.createElement('div');
                div.style.cssText = 'margin-bottom: 15px; padding: 10px; background: rgba(255,255,255,0.02); border-radius: 8px; font-size: 0.9rem;';
                div.innerHTML = `
                    <div style="color: var(--text-secondary); font-size: 0.8rem; margin-bottom: 5px;">
                        ${new Date(conv.timestamp).toLocaleString()}
                    </div>
                    <div style="margin-bottom: 5px;"><strong>Q:</strong> ${conv.prompt}</div>
                    <div><strong>A:</strong> ${conv.response}</div>
                `;
                container.appendChild(div);
            });
        }
        
        function formatUptime(seconds) {
            if (seconds < 60) return seconds + 's';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
            if (seconds < 86400) return Math.floor(seconds / 3600) + 'h';
            return Math.floor(seconds / 86400) + 'd';
        }
        
        async function executeCommand(action) {
            try {
                const response = await fetch(`/api/command/${action}`, {
                    method: 'POST'
                });
                
                const result = await response.json();
                
                if (result.success) {
                    console.log('Command successful:', result.message);
                    // Show success feedback (you could add a toast notification here)
                } else {
                    console.error('Command failed:', result.message);
                    alert('Command failed: ' + result.message);
                }
            } catch (error) {
                console.error('Error executing command:', error);
                alert('Error executing command: ' + error.message);
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
            
            // Initial status load
            fetch('/api/status')
                .then(response => response.json())
                .then(status => updateDashboard(status))
                .catch(error => console.error('Error loading initial status:', error));
        });
    </script>
</body>
</html>
"""


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Vega Web Dashboard")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run dashboard on"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")

    args = parser.parse_args()

    print(f"🚀 Starting Vega Dashboard on http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
