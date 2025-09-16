#!/usr/bin/env python3
"""
VOICE VISUALIZER - AUDIO PERSONALITY ENGINE
===========================================

Real-time audio visualization system that reacts to Vega's spoken output.
Creates beautiful visual representations of voice, emotion, and context.

Features:
- 🎙️ Real-time audio waveform visualization
- 🌈 Emotion-based color mapping
- 📊 Spectrum analysis with bar/wave animations
- 🎨 Context-aware visual themes
- 💾 Audio transcript and visual logging
- 🌐 Web-based visualization dashboard
- 📱 Mobile-responsive visual interface

Usage:
    python voice_visualizer.py --daemon          # Background daemon mode
    python voice_visualizer.py --gui             # GUI visualization window
    python voice_visualizer.py --web --port 8081 # Web interface
"""

import os
import sys
import json
import time
import asyncio
import sqlite3
import threading
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import argparse
import logging

# Audio processing imports
AUDIO_AVAILABLE = False
try:
    import pyaudio
    import wave
    import librosa
    import soundfile as sf
    from scipy import signal

    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Audio libraries not available: {e}")

# Visualization imports
VISUAL_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    VISUAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Visualization libraries not available: {e}")

# GUI imports (optional)
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import ttk
    import pygame

    pygame.mixer.init()
    GUI_AVAILABLE = True
except ImportError:
    pass

# Web interface imports
WEB_AVAILABLE = False
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn

    WEB_AVAILABLE = True
except ImportError:
    pass


class EmotionType(Enum):
    """Emotion categories for voice analysis"""

    ANALYTICAL = "analytical"
    CHEERFUL = "cheerful"
    SERIOUS = "serious"
    CURIOUS = "curious"
    CONFIDENT = "confident"
    CONCERNED = "concerned"
    EXCITED = "excited"
    CALM = "calm"


class ContextType(Enum):
    """Context categories for voice output"""

    ALERT = "alert"
    SOCIAL = "social"
    REFLECTIVE = "reflective"
    INFORMATIVE = "informative"
    INTERACTIVE = "interactive"
    ERROR = "error"
    SUCCESS = "success"
    THINKING = "thinking"


@dataclass
class AudioFrame:
    """Single audio frame data"""

    timestamp: str
    amplitude: np.ndarray
    frequency_spectrum: np.ndarray
    dominant_frequency: float
    volume_db: float
    emotion: EmotionType
    context: ContextType
    transcript: str


@dataclass
class VisualizationSettings:
    """Visualization configuration"""

    style: str = "wave"  # wave, bars, spectrum, circular
    color_scheme: str = "emotion"  # emotion, context, intensity, rainbow
    smoothing: float = 0.3
    sensitivity: float = 1.0
    fft_size: int = 1024
    update_rate: int = 30  # FPS
    background_color: str = "#0f172a"
    primary_color: str = "#2563eb"


class VoiceVisualizer:
    """Main voice visualization engine"""

    def __init__(self, mode: str = "daemon"):
        self.mode = mode
        self.base_dir = Path(__file__).parent
        self.state_dir = self.base_dir / "vega_state"
        self.logs_dir = self.base_dir / "vega_logs"

        # Create directories
        for directory in [self.state_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)8s | %(name)20s | %(message)s",
            handlers=[
                logging.FileHandler(self.logs_dir / "voice_visualizer.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("VoiceVisualizer")

        # Audio configuration
        self.sample_rate = 44100
        self.chunk_size = 1024
        self.channels = 1
        self.format = pyaudio.paInt16

        # Visualization settings
        self.settings = VisualizationSettings()

        # Audio processing components
        self.audio_stream = None
        self.pyaudio_instance = None

        # Data storage
        self.audio_buffer = []
        self.visualization_history = []
        self.current_frame = None

        # Database for logging
        self.init_database()

        # Emotion and context mappings
        self.emotion_colors = {
            EmotionType.ANALYTICAL: "#3b82f6",  # Blue
            EmotionType.CHEERFUL: "#eab308",  # Yellow
            EmotionType.SERIOUS: "#dc2626",  # Red
            EmotionType.CURIOUS: "#8b5cf6",  # Purple
            EmotionType.CONFIDENT: "#059669",  # Green
            EmotionType.CONCERNED: "#ea580c",  # Orange
            EmotionType.EXCITED: "#ec4899",  # Pink
            EmotionType.CALM: "#06b6d4",  # Cyan
        }

        self.context_themes = {
            ContextType.ALERT: {"intensity": 1.0, "pulse": True},
            ContextType.SOCIAL: {"intensity": 0.7, "pulse": False},
            ContextType.REFLECTIVE: {"intensity": 0.5, "pulse": False},
            ContextType.INFORMATIVE: {"intensity": 0.8, "pulse": False},
            ContextType.INTERACTIVE: {"intensity": 0.9, "pulse": True},
            ContextType.ERROR: {"intensity": 1.0, "pulse": True},
            ContextType.SUCCESS: {"intensity": 0.8, "pulse": False},
            ContextType.THINKING: {"intensity": 0.6, "pulse": True},
        }

        self.logger.info(f"🎙️ Voice Visualizer initialized in {mode} mode")

    def init_database(self):
        """Initialize SQLite database for audio logging"""
        db_path = self.state_dir / "voice_visual_log.db"

        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audio_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        transcript TEXT,
                        emotion TEXT,
                        context TEXT,
                        duration_ms INTEGER,
                        volume_db REAL,
                        dominant_freq REAL,
                        waveform_data BLOB,
                        spectrum_data BLOB
                    )
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_timestamp 
                    ON audio_sessions(timestamp)
                """
                )

                conn.commit()

            self.logger.info("📊 Audio logging database initialized")

        except Exception as e:
            self.logger.error(f"❌ Error initializing database: {e}")

    def analyze_emotion(
        self, audio_data: np.ndarray, transcript: str = ""
    ) -> EmotionType:
        """Analyze emotion from audio characteristics and transcript"""
        try:
            # Basic emotion detection based on audio features

            # Calculate spectral features
            fft = np.fft.fft(audio_data)
            magnitude = np.abs(fft)

            # Energy in different frequency bands
            low_energy = np.sum(magnitude[0:50])  # 0-1kHz (calm/serious)
            mid_energy = np.sum(magnitude[50:200])  # 1-4kHz (normal speech)
            high_energy = np.sum(magnitude[200:400])  # 4-8kHz (excitement/emotion)

            # Analyze pitch variation (emotional intensity)
            pitch_variance = np.var(magnitude)

            # Volume analysis
            volume = np.sqrt(np.mean(audio_data**2))

            # Simple heuristic emotion classification
            if high_energy > mid_energy * 1.5:
                if volume > 0.3:
                    return EmotionType.EXCITED
                else:
                    return EmotionType.CURIOUS
            elif low_energy > mid_energy:
                if pitch_variance < 0.1:
                    return EmotionType.CALM
                else:
                    return EmotionType.SERIOUS
            elif volume > 0.4:
                return EmotionType.CONFIDENT
            elif "?" in transcript:
                return EmotionType.CURIOUS
            elif any(
                word in transcript.lower() for word in ["error", "problem", "issue"]
            ):
                return EmotionType.CONCERNED
            elif any(
                word in transcript.lower() for word in ["great", "excellent", "perfect"]
            ):
                return EmotionType.CHEERFUL
            else:
                return EmotionType.ANALYTICAL

        except Exception as e:
            self.logger.error(f"❌ Error analyzing emotion: {e}")
            return EmotionType.ANALYTICAL

    def analyze_context(
        self, transcript: str = "", system_state: Dict = None
    ) -> ContextType:
        """Determine context from transcript and system state"""
        try:
            if not transcript:
                return ContextType.THINKING

            transcript_lower = transcript.lower()

            # Context keyword analysis
            if any(
                word in transcript_lower
                for word in ["alert", "warning", "urgent", "critical"]
            ):
                return ContextType.ALERT
            elif any(
                word in transcript_lower for word in ["error", "failed", "problem"]
            ):
                return ContextType.ERROR
            elif any(
                word in transcript_lower for word in ["success", "complete", "done"]
            ):
                return ContextType.SUCCESS
            elif any(
                word in transcript_lower for word in ["hello", "hi", "chat", "talk"]
            ):
                return ContextType.SOCIAL
            elif any(
                word in transcript_lower for word in ["think", "consider", "reflect"]
            ):
                return ContextType.REFLECTIVE
            elif "?" in transcript:
                return ContextType.INTERACTIVE
            else:
                return ContextType.INFORMATIVE

        except Exception as e:
            self.logger.error(f"❌ Error analyzing context: {e}")
            return ContextType.INFORMATIVE

    def process_audio_frame(
        self, audio_data: np.ndarray, transcript: str = ""
    ) -> AudioFrame:
        """Process a single audio frame"""
        try:
            # Convert to float and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0

            # Calculate FFT for frequency analysis
            fft = np.fft.fft(audio_data, n=self.settings.fft_size)
            magnitude = np.abs(fft[: self.settings.fft_size // 2])

            # Find dominant frequency
            dominant_freq_idx = np.argmax(magnitude)
            dominant_frequency = (
                dominant_freq_idx * self.sample_rate / self.settings.fft_size
            )

            # Calculate volume in dB
            rms = np.sqrt(np.mean(audio_data**2))
            volume_db = 20 * np.log10(rms + 1e-10)  # Avoid log(0)

            # Analyze emotion and context
            emotion = self.analyze_emotion(audio_data, transcript)
            context = self.analyze_context(transcript)

            # Create audio frame
            frame = AudioFrame(
                timestamp=datetime.now().isoformat(),
                amplitude=audio_data,
                frequency_spectrum=magnitude,
                dominant_frequency=dominant_frequency,
                volume_db=volume_db,
                emotion=emotion,
                context=context,
                transcript=transcript,
            )

            self.current_frame = frame
            return frame

        except Exception as e:
            self.logger.error(f"❌ Error processing audio frame: {e}")
            return None

    def generate_waveform_visualization(self, frame: AudioFrame) -> Dict[str, Any]:
        """Generate waveform visualization data"""
        try:
            # Smooth the amplitude data
            amplitude = frame.amplitude
            if len(self.audio_buffer) > 0:
                # Apply smoothing with previous frames
                smoothing = self.settings.smoothing
                amplitude = smoothing * amplitude + (1 - smoothing) * np.mean(
                    [f.amplitude for f in self.audio_buffer[-3:]], axis=0
                )

            # Get color based on emotion
            color = self.emotion_colors.get(frame.emotion, self.settings.primary_color)

            # Get intensity based on context
            intensity = self.context_themes.get(frame.context, {"intensity": 0.8})[
                "intensity"
            ]

            # Prepare visualization data
            viz_data = {
                "type": "waveform",
                "timestamp": frame.timestamp,
                "amplitude": amplitude.tolist(),
                "color": color,
                "intensity": intensity,
                "emotion": frame.emotion.value,
                "context": frame.context.value,
                "volume_db": frame.volume_db,
                "dominant_freq": frame.dominant_frequency,
            }

            return viz_data

        except Exception as e:
            self.logger.error(f"❌ Error generating waveform visualization: {e}")
            return {}

    def generate_spectrum_visualization(self, frame: AudioFrame) -> Dict[str, Any]:
        """Generate frequency spectrum visualization data"""
        try:
            # Normalize spectrum
            spectrum = frame.frequency_spectrum
            spectrum = spectrum / (np.max(spectrum) + 1e-10)

            # Apply logarithmic scaling for better visualization
            spectrum = np.log10(spectrum + 1e-10)
            spectrum = (spectrum - np.min(spectrum)) / (
                np.max(spectrum) - np.min(spectrum) + 1e-10
            )

            # Get color scheme
            if self.settings.color_scheme == "emotion":
                base_color = self.emotion_colors.get(
                    frame.emotion, self.settings.primary_color
                )
            else:
                base_color = self.settings.primary_color

            # Create frequency bins
            freqs = np.fft.fftfreq(self.settings.fft_size, 1 / self.sample_rate)[
                : self.settings.fft_size // 2
            ]

            viz_data = {
                "type": "spectrum",
                "timestamp": frame.timestamp,
                "spectrum": spectrum.tolist(),
                "frequencies": freqs.tolist(),
                "color": base_color,
                "emotion": frame.emotion.value,
                "context": frame.context.value,
                "volume_db": frame.volume_db,
            }

            return viz_data

        except Exception as e:
            self.logger.error(f"❌ Error generating spectrum visualization: {e}")
            return {}

    def log_audio_session(self, frame: AudioFrame):
        """Log audio session to database"""
        try:
            db_path = self.state_dir / "voice_visual_log.db"

            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO audio_sessions 
                    (timestamp, transcript, emotion, context, duration_ms, volume_db, 
                     dominant_freq, waveform_data, spectrum_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        frame.timestamp,
                        frame.transcript,
                        frame.emotion.value,
                        frame.context.value,
                        len(frame.amplitude)
                        * 1000
                        / self.sample_rate,  # Duration in ms
                        frame.volume_db,
                        frame.dominant_frequency,
                        frame.amplitude.tobytes(),
                        frame.frequency_spectrum.tobytes(),
                    ),
                )

                conn.commit()

        except Exception as e:
            self.logger.error(f"❌ Error logging audio session: {e}")

    def save_visualization_state(self):
        """Save current visualization state"""
        try:
            state_file = self.state_dir / "voice_visualizer_state.json"

            state_data = {
                "timestamp": datetime.now().isoformat(),
                "settings": asdict(self.settings),
                "current_frame": (
                    asdict(self.current_frame) if self.current_frame else None
                ),
                "buffer_size": len(self.audio_buffer),
                "mode": self.mode,
            }

            with open(state_file, "w") as f:
                json.dump(state_data, f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"❌ Error saving visualization state: {e}")

    async def daemon_mode(self):
        """Run in background daemon mode"""
        self.logger.info("🔄 Starting voice visualizer daemon...")

        # Monitor for audio input triggers
        trigger_file = self.state_dir / "voice_trigger.json"

        while True:
            try:
                # Check for voice trigger
                if trigger_file.exists():
                    with open(trigger_file, "r") as f:
                        trigger_data = json.load(f)

                    # Process the triggered audio
                    if "audio_file" in trigger_data:
                        await self.process_audio_file(trigger_data["audio_file"])
                    elif "transcript" in trigger_data:
                        await self.process_transcript(trigger_data["transcript"])

                    # Remove trigger file
                    trigger_file.unlink()

                # Save state periodically
                self.save_visualization_state()

                # Sleep before next check
                await asyncio.sleep(1)

            except Exception as e:
                self.logger.error(f"❌ Error in daemon mode: {e}")
                await asyncio.sleep(5)

    async def process_audio_file(self, audio_file_path: str):
        """Process an audio file for visualization"""
        try:
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                self.logger.warning(f"⚠️ Audio file not found: {audio_file_path}")
                return

            # Load audio file
            audio_data, sample_rate = librosa.load(audio_file_path, sr=self.sample_rate)

            # Process in chunks
            chunk_size = self.chunk_size
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                # Process frame
                frame = self.process_audio_frame(chunk)
                if frame:
                    # Generate visualizations
                    waveform_viz = self.generate_waveform_visualization(frame)
                    spectrum_viz = self.generate_spectrum_visualization(frame)

                    # Store in buffer
                    self.audio_buffer.append(frame)
                    if len(self.audio_buffer) > 100:  # Keep last 100 frames
                        self.audio_buffer.pop(0)

                    # Log to database
                    self.log_audio_session(frame)

            self.logger.info(f"🎵 Processed audio file: {audio_file_path}")

        except Exception as e:
            self.logger.error(f"❌ Error processing audio file: {e}")

    async def process_transcript(self, transcript: str):
        """Process a transcript for context analysis"""
        try:
            # Create a dummy audio frame for transcript analysis
            dummy_audio = np.zeros(self.chunk_size)
            frame = self.process_audio_frame(dummy_audio, transcript)

            if frame:
                # Log transcript with context analysis
                self.log_audio_session(frame)
                self.logger.info(f"📝 Processed transcript: {transcript[:50]}...")

        except Exception as e:
            self.logger.error(f"❌ Error processing transcript: {e}")


def create_web_interface():
    """Create FastAPI web interface for visualization"""
    if not WEB_AVAILABLE:
        return None

    app = FastAPI(title="Vega Voice Visualizer", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return """
<!DOCTYPE html>
<html>
<head>
    <title>🎙️ Vega Voice Visualizer</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .visualizer {
            background: #1e293b;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #334155;
        }
        .canvas {
            width: 100%;
            height: 200px;
            background: #0f172a;
            border-radius: 10px;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 8px 16px;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            background: #374151;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2563eb;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #9ca3af;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎙️ Vega Voice Visualizer</h1>
            <p>Real-time audio visualization and emotion analysis</p>
        </div>
        
        <div class="visualizer">
            <h3>🌊 Waveform Visualization</h3>
            <canvas id="waveformCanvas" class="canvas"></canvas>
            <div class="controls">
                <button class="btn" onclick="toggleVisualization()">▶️ Start/Stop</button>
                <button class="btn" onclick="changeStyle()">🎨 Change Style</button>
                <button class="btn" onclick="adjustSensitivity()">🔧 Sensitivity</button>
            </div>
        </div>
        
        <div class="visualizer">
            <h3>📊 Frequency Spectrum</h3>
            <canvas id="spectrumCanvas" class="canvas"></canvas>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value" id="volumeMetric">-</div>
                <div class="metric-label">Volume (dB)</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="emotionMetric">-</div>
                <div class="metric-label">Current Emotion</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="contextMetric">-</div>
                <div class="metric-label">Context</div>
            </div>
            <div class="metric">
                <div class="metric-value" id="freqMetric">-</div>
                <div class="metric-label">Dominant Freq (Hz)</div>
            </div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let isActive = false;
        
        function connectWebSocket() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = function(event) {
                console.log('Connected to voice visualizer');
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateVisualization(data);
            };
            
            ws.onclose = function(event) {
                console.log('Disconnected from voice visualizer');
                setTimeout(connectWebSocket, 5000);
            };
        }
        
        function updateVisualization(data) {
            if (data.type === 'waveform') {
                drawWaveform(data);
            } else if (data.type === 'spectrum') {
                drawSpectrum(data);
            }
            
            // Update metrics
            document.getElementById('volumeMetric').textContent = 
                data.volume_db ? data.volume_db.toFixed(1) : '-';
            document.getElementById('emotionMetric').textContent = 
                data.emotion || '-';
            document.getElementById('contextMetric').textContent = 
                data.context || '-';
            document.getElementById('freqMetric').textContent = 
                data.dominant_freq ? data.dominant_freq.toFixed(0) : '-';
        }
        
        function drawWaveform(data) {
            const canvas = document.getElementById('waveformCanvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (!data.amplitude) return;
            
            const amplitude = data.amplitude;
            const width = canvas.width;
            const height = canvas.height;
            const centerY = height / 2;
            
            ctx.strokeStyle = data.color || '#2563eb';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let i = 0; i < amplitude.length; i++) {
                const x = (i / amplitude.length) * width;
                const y = centerY + (amplitude[i] * centerY * (data.intensity || 1));
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            
            ctx.stroke();
        }
        
        function drawSpectrum(data) {
            const canvas = document.getElementById('spectrumCanvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (!data.spectrum) return;
            
            const spectrum = data.spectrum;
            const width = canvas.width;
            const height = canvas.height;
            const barWidth = width / spectrum.length;
            
            ctx.fillStyle = data.color || '#2563eb';
            
            for (let i = 0; i < spectrum.length; i++) {
                const barHeight = spectrum[i] * height;
                const x = i * barWidth;
                const y = height - barHeight;
                
                ctx.fillRect(x, y, barWidth - 1, barHeight);
            }
        }
        
        function toggleVisualization() {
            isActive = !isActive;
            // Send control command to server
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    command: 'toggle',
                    active: isActive
                }));
            }
        }
        
        function changeStyle() {
            // Cycle through visualization styles
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    command: 'change_style'
                }));
            }
        }
        
        function adjustSensitivity() {
            // Adjust sensitivity
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    command: 'adjust_sensitivity'
                }));
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            connectWebSocket();
        });
    </script>
</body>
</html>
        """

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                # Send visualization data
                await websocket.send_json(
                    {
                        "type": "waveform",
                        "amplitude": [0] * 1024,  # Placeholder data
                        "color": "#2563eb",
                        "emotion": "analytical",
                        "context": "informative",
                    }
                )
                await asyncio.sleep(1 / 30)  # 30 FPS
        except WebSocketDisconnect:
            pass

    return app


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vega Voice Visualizer")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--gui", action="store_true", help="Show GUI visualization")
    parser.add_argument("--web", action="store_true", help="Start web interface")
    parser.add_argument("--port", type=int, default=8081, help="Web interface port")

    args = parser.parse_args()

    if not AUDIO_AVAILABLE:
        print(
            "❌ Audio libraries not available. Install with: pip install pyaudio librosa soundfile scipy"
        )
        return

    visualizer = VoiceVisualizer(mode="daemon" if args.daemon else "interactive")

    if args.web and WEB_AVAILABLE:
        print(f"🌐 Starting web interface on http://127.0.0.1:{args.port}")
        app = create_web_interface()

        # Run web interface
        import uvicorn

        await uvicorn.run(app, host="127.0.0.1", port=args.port)

    elif args.daemon:
        await visualizer.daemon_mode()

    else:
        print("🎙️ Voice Visualizer - Interactive mode")
        print("This would start GUI or interactive visualization")


if __name__ == "__main__":
    asyncio.run(main())
