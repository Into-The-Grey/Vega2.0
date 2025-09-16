# Voice Module Development Notes

## Overview

The `voice/` module provides local Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities without cloud dependencies:

- **VoiceManager**: Central coordinator for TTS/STT operations
- **Provider Architecture**: Pluggable TTS/STT engines
- **Local Processing**: No cloud API calls, complete privacy
- **Model Management**: Download and manage voice models locally

## Architecture

### Core Components

- **VoiceManager**: Main interface for voice operations
- **TTSProviderBase**: Abstract base class for TTS engines
- **STTProviderBase**: Abstract base class for STT engines
- **Provider Implementations**: Piper (TTS), Vosk (STT), pyttsx3 (TTS)

### Supported Engines

#### TTS (Text-to-Speech)

- **Piper**: High-quality neural TTS (primary)
- **pyttsx3**: System TTS fallback
- **Festival**: Linux TTS option
- **eSpeak**: Cross-platform TTS

#### STT (Speech-to-Text)

- **Vosk**: Offline speech recognition (primary)
- **SpeechRecognition**: Fallback with multiple backends
- **Whisper**: OpenAI Whisper for high accuracy

## Configuration

### Voice Configuration File

```yaml
# config/voice.yaml
tts:
  provider: "piper"
  default_voice: "en_US-lessac-medium"
  settings:
    speed: 1.0
    pitch: 1.0
    volume: 0.8
    
stt:
  provider: "vosk"
  default_model: "vosk-model-en-us-0.22"
  settings:
    sample_rate: 16000
    language: "en"
    
models:
  download_dir: "./voice/models"
  auto_download: true
  
audio:
  input_device: "default"
  output_device: "default"
  sample_rate: 16000
  channels: 1
```

## Usage Patterns

### Basic TTS Usage

```python
from voice import VoiceManager

# Initialize voice manager
voice = VoiceManager()

# Synthesize text to audio
audio_data = voice.synthesize("Hello, world!")

# Save to file
voice.synthesize_to_file("Hello, world!", "output.wav")

# List available voices
voices = voice.list_voices()
```

### Basic STT Usage

```python
from voice import VoiceManager

# Initialize voice manager  
voice = VoiceManager()

# Transcribe audio data
with open("input.wav", "rb") as f:
    audio_data = f.read()
    
transcription = voice.transcribe(audio_data)

# Transcribe from file
transcription = voice.transcribe_file("input.wav")
```

### Configuration Management

```python
# Update voice configuration
config = {
    "tts": {"provider": "piper", "voice": "en_US-amy-medium"},
    "stt": {"provider": "vosk", "model": "vosk-model-en-us-0.22"}
}
voice.update_config(config)

# Get current configuration
current_config = voice.get_config()
```

## Provider Implementation

### Creating TTS Provider

```python
from voice import TTSProviderBase

class CustomTTSProvider(TTSProviderBase):
    def is_available(self) -> bool:
        # Check if provider dependencies are available
        return True
    
    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        # Implement text-to-speech synthesis
        # Return audio data as bytes
        pass
    
    def list_voices(self) -> list:
        # Return list of available voices
        return ["voice1", "voice2"]
```

### Creating STT Provider

```python
from voice import STTProviderBase

class CustomSTTProvider(STTProviderBase):
    def is_available(self) -> bool:
        # Check if provider dependencies are available
        return True
    
    def transcribe(self, audio_data: bytes, **kwargs) -> str:
        # Implement speech-to-text transcription
        # Return transcribed text
        pass
    
    def list_models(self) -> list:
        # Return list of available models
        return ["model1", "model2"]
```

## Model Management

### Automatic Model Download

```python
# Models are downloaded automatically when needed
voice = VoiceManager(auto_download=True)

# Download specific model
model_path = voice.download_model("vosk-model-en-us-0.22")

# List installed models
installed = voice.list_installed_models()
```

### Model Storage Structure

```
voice/models/
├── tts/
│   ├── piper/
│   │   ├── en_US-lessac-medium/
│   │   └── en_US-amy-medium/
│   └── festival/
└── stt/
    ├── vosk/
    │   ├── vosk-model-en-us-0.22/
    │   └── vosk-model-small-en-us-0.15/
    └── whisper/
        ├── base.en/
        └── small.en/
```

## Integration with Chat System

### Voice Chat Workflow

```python
# 1. User speaks -> audio input
audio_input = capture_audio()

# 2. STT: Convert speech to text
user_text = voice.transcribe(audio_input)

# 3. Process with chat system
response_text = await chat_system.process(user_text)

# 4. TTS: Convert response to speech
response_audio = voice.synthesize(response_text)

# 5. Play audio response
play_audio(response_audio)
```

### Streaming Audio Processing

```python
# Real-time audio processing
def process_audio_stream():
    for audio_chunk in audio_stream:
        # Process in chunks for real-time response
        partial_text = voice.transcribe(audio_chunk)
        if is_complete_phrase(partial_text):
            # Process complete phrase
            response = process_chat(partial_text)
            response_audio = voice.synthesize(response)
            yield response_audio
```

## Testing

### Unit Tests

- Provider availability checking
- Audio synthesis and transcription
- Configuration management
- Model downloading and loading
- Error handling and fallbacks

### Integration Tests

- End-to-end voice chat workflow
- Provider switching and fallbacks
- Audio format conversion
- Real-time processing performance

### Test Utilities

```python
# Mock providers for testing
from tests.test_voice import MockTTSProvider, MockSTTProvider

# Test with mock providers
voice = VoiceManager()
voice.tts_providers["mock"] = MockTTSProvider("mock")
voice.set_tts_provider("mock")
```

## Performance Considerations

### Optimization Strategies

- **Model Caching**: Keep models in memory for repeated use
- **Audio Streaming**: Process audio in chunks for real-time response
- **Provider Selection**: Automatic fallback to faster providers when needed
- **Quality vs Speed**: Configurable quality settings for different use cases

### Memory Management

- **Model Loading**: Load models on-demand to save memory
- **Audio Buffer Management**: Efficient handling of large audio files
- **Resource Cleanup**: Proper cleanup of audio resources

## Security and Privacy

### Local Processing Benefits

- **No Cloud Dependencies**: All processing happens locally
- **Privacy Protection**: Audio data never leaves the device
- **Offline Operation**: Works without internet connection
- **Data Sovereignty**: Complete control over voice data

### Security Considerations

- **Model Integrity**: Verify downloaded models with checksums
- **Audio Input Security**: Sanitize audio input to prevent attacks
- **File System Security**: Secure model storage and access

## Troubleshooting

### Common Issues

#### TTS Not Working

1. Check provider availability: `voice.is_tts_available()`
2. Verify model installation: `voice.list_installed_models()`
3. Test with different provider: `voice.set_tts_provider("pyttsx3")`
4. Check audio output device configuration

#### STT Not Working  

1. Check microphone permissions
2. Verify audio input format (16kHz, mono recommended)
3. Test with different STT model
4. Check for background noise interference

#### Model Download Issues

1. Check internet connection
2. Verify disk space for model storage
3. Check model URL accessibility
4. Manual model download and placement

### Debug Mode

```python
# Enable verbose logging for debugging
voice = VoiceManager(debug=True)

# Check provider status
status = voice.get_provider_status()
print(f"TTS Provider: {status['tts']}")
print(f"STT Provider: {status['stt']}")
```

## Future Enhancements

### Planned Features

1. **Voice Cloning**: Personal voice model training
2. **Emotion Recognition**: Detect emotion in speech
3. **Multi-language Support**: Extended language coverage
4. **Real-time Translation**: Speech-to-speech translation
5. **Voice Biometrics**: Speaker identification and authentication

### Research Areas

- **Neural Vocoding**: Improved audio quality
- **Streaming STT**: Continuous speech recognition
- **Voice Conversion**: Real-time voice transformation
- **Noise Reduction**: Advanced audio preprocessing
