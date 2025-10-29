# Home Assistant Voice Integration for Vega 2.0

## Overview

This integration enables **voice-activated "Hey Vega" functionality** through Home Assistant Assist, working with **ALL your connected devices** - iPhone, iPad, Apple Watch, Mac, Windows PC, browser, and more.

### How It Works

```
1. You: "Hey Vega, what's the weather?"
   ↓
2. Home Assistant Assist (on your iPhone/Mac/etc)
   ↓ (Speech-to-Text)
3. Home Assistant → Vega Server (/hass/webhook)
   ↓
4. Vega LLM Processing (with conversation context)
   ↓
5. Vega → Home Assistant (TTS service)
   ↓
6. Your Device Speakers: "The weather is..."
```

**Key Features:**

- 🎤 **Device-Agnostic**: Works with ANY Home Assistant companion app
- 🧠 **Persistent Memory**: Conversations continue across devices
- 🔒 **Private**: Everything stays on your local network
- ⚡ **Fast**: Typical response time < 2 seconds
- 🌐 **Flexible**: Use ANY wake word ("Hey Vega", "Vega", custom phrases)

---

## Quick Start

### Prerequisites

1. ✅ **Home Assistant** running (local or accessible URL)
2. ✅ **Home Assistant Assist** configured with voice pipeline
3. ✅ **Vega server** running on same network (or accessible to HA)
4. ✅ **Home Assistant companion app** on your devices (iOS/Android/etc)

### Step 1: Configure Vega

Edit your `.env` file:

```bash
# Enable Home Assistant integration
HASS_ENABLED=true

# Your Home Assistant URL
HASS_URL=http://homeassistant.local:8123
# or
HASS_URL=http://192.168.1.100:8123

# Long-lived access token from HA (see below how to create)
HASS_TOKEN=your_long_lived_access_token_here

# Optional: Customize TTS settings
HASS_TTS_SERVICE=tts.cloud_say        # or tts.speak, tts.google_translate_say
HASS_VOICE_NAME=en-US-Neural2-J       # Optional: specific voice
HASS_MEDIA_PLAYER=media_player.living_room  # Optional: default output device
```

### Step 2: Create Home Assistant Access Token

1. Open Home Assistant web interface
2. Click your profile (bottom left)
3. Scroll to **"Long-Lived Access Tokens"**
4. Click **"Create Token"**
5. Name it: `Vega Integration`
6. Copy the token and paste into `.env` as `HASS_TOKEN`

### Step 3: Test the Integration

```bash
# Restart Vega to load new config
python main.py server

# Check integration status
curl -H "X-API-Key: YOUR_API_KEY" http://localhost:8000/hass/status
```

Expected response:

```json
{
  "enabled": true,
  "configured": true,
  "connected": true,
  "url": "http://homeassistant.local:8123",
  "webhook_endpoint": "/hass/webhook",
  "message": "Home Assistant integration operational"
}
```

### Step 4: Configure Home Assistant Automation

Create an automation in Home Assistant to route Assist commands to Vega:

**Method A: Via UI (Easiest)**

1. Go to **Settings** → **Automations & Scenes**
2. Click **"Create Automation"** → **"Start with an empty automation"**
3. Add **Trigger**: **"Sentence"** (Assist phrase trigger)
   - Sentence: `{action}` (catches everything after wake word)
4. Add **Action**: **"Call service"**
   - Service: `rest_command.vega_webhook`
5. Configure the service call (see Method B for YAML)

**Method B: Via YAML (Recommended)**

Add to `configuration.yaml`:

```yaml
# REST command to call Vega webhook
rest_command:
  vega_webhook:
    url: "http://YOUR_VEGA_SERVER_IP:8000/hass/webhook"
    method: POST
    headers:
      X-API-Key: "YOUR_VEGA_API_KEY"
      Content-Type: "application/json"
    payload: >
      {
        "text": "{{ text }}",
        "conversation_id": "{{ conversation_id }}",
        "device_id": "{{ device_id }}",
        "device_type": "{{ device_type }}",
        "user_id": "{{ user_id }}",
        "language": "en"
      }

# Automation to route Assist to Vega
automation:
  - alias: "Vega Voice Assistant"
    trigger:
      - platform: conversation
        command:
          - "{action}"  # Catches all voice commands after wake word
    action:
      - service: rest_command.vega_webhook
        data:
          text: "{{ trigger.text }}"
          conversation_id: "{{ trigger.conversation_id }}"
          device_id: "{{ trigger.device_id }}"
          device_type: "{{ trigger.platform }}"
          user_id: "{{ trigger.user_id }}"
```

### Step 5: Configure Home Assistant Assist Wake Word

1. Go to **Settings** → **Voice Assistants**
2. Click **"Add Assistant"** or edit existing
3. Configure:
   - **Conversation Agent**: Home Assistant (we'll route via automation)
   - **Speech-to-Text**: Any STT provider (Whisper, Google, etc.)
   - **Text-to-Speech**: Any TTS provider (Google, Nabu Casa Cloud, etc.)
   - **Wake Word**: Configure custom wake word:
     - Install **"Wyoming Protocol"** or **"openWakeWord"** add-on
     - Or use **"Hey Jarvis"** and customize to **"Hey Vega"**
4. Save and test

---

## Usage

### From iPhone/iPad

1. Open Home Assistant app
2. Tap microphone icon OR
3. Say: **"Hey Vega, what's the weather?"**
4. Vega responds through your phone speakers

### From Apple Watch

1. Raise wrist
2. Say: **"Hey Vega, remind me to call Mom"**
3. Vega processes and responds on watch

### From Mac/Windows

1. Open Home Assistant in browser
2. Click microphone icon
3. Voice command → Vega response

### From Anywhere

- If you have HA exposed externally (via Nabu Casa Cloud or VPN)
- Use companion app from anywhere
- Full Vega functionality with your voice

---

## Configuration Options

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HASS_ENABLED` | Yes | `false` | Enable/disable integration |
| `HASS_URL` | Yes | - | Home Assistant URL |
| `HASS_TOKEN` | Yes | - | Long-lived access token |
| `HASS_WEBHOOK_ID` | No | `vega_webhook` | Webhook identifier |
| `HASS_TTS_SERVICE` | No | `tts.cloud_say` | TTS service to use |
| `HASS_VOICE_NAME` | No | - | TTS voice name (optional) |
| `HASS_MEDIA_PLAYER` | No | - | Default media player entity |

### Supported TTS Services

- `tts.cloud_say` - Nabu Casa Cloud TTS (best quality)
- `tts.speak` - Generic TTS
- `tts.google_translate_say` - Google Translate TTS (free)
- `tts.amazon_polly_say` - AWS Polly
- `tts.microsoft_say` - Azure TTS
- `tts.elevenlabs_say` - ElevenLabs (premium quality)
- `tts.piper` - Local Piper TTS (private)

### Device Types

The integration automatically detects your device type:

- `ios` - iPhone, iPad, Apple Watch
- `macos` - Mac computers
- `windows` - Windows PCs
- `browser` - Web browser access
- `smart_speaker` - Google Home, Alexa (if integrated)

---

## Advanced Configuration

### Per-Device TTS Voices

Different voices for different devices (configured in HA automation):

```yaml
automation:
  - alias: "Vega Voice - iPhone"
    trigger:
      - platform: conversation
        command: ["{action}"]
    condition:
      - condition: template
        value_template: "{{ 'iphone' in trigger.device_id.lower() }}"
    action:
      - service: rest_command.vega_webhook
        data:
          text: "{{ trigger.text }}"
          device_type: "ios"
```

### Custom Wake Words

Install `openWakeWord` or `Wyoming Porcupine` add-ons in HA to create custom wake words:

1. Go to **Settings** → **Add-ons** → **Add-on Store**
2. Search **"openWakeWord"** or **"Porcupine"**
3. Install and configure
4. Create custom wake word: **"Hey Vega"**, **"Vega"**, **"Computer"**, etc.
5. Associate with your Assist pipeline

### Multi-User Support

Track conversations per-user automatically:

```yaml
# Vega automatically uses conversation_id from HA
# Each user gets their own persistent conversation history
# No additional configuration needed
```

### Conversation Context

Vega automatically maintains conversation context:

```
You: "Hey Vega, what's the capital of France?"
Vega: "The capital of France is Paris."

You: "What's the population?"
Vega: "Paris has a population of approximately 2.1 million people."

You: "What's the weather there today?"
Vega: "In Paris today, it's..."
```

Context persists across:

- ✅ Different devices (iPhone → Mac → iPad)
- ✅ Multiple sessions
- ✅ Days/weeks (configurable retention)

---

## Troubleshooting

### "Integration not available" error

**Problem**: Vega can't find HA integration module

**Solution**:

```bash
# Verify integration file exists
ls -la src/vega/integrations/homeassistant.py

# Reinstall dependencies if needed
pip install httpx
```

### "Cannot connect to Home Assistant" error

**Problem**: Network connectivity or wrong URL

**Solution**:

```bash
# Test HA URL manually
curl http://homeassistant.local:8123/api/

# Check firewall rules
# Ensure Vega server can reach HA on port 8123

# Try IP address instead of hostname
HASS_URL=http://192.168.1.100:8123
```

### "Invalid token" error

**Problem**: Access token expired or incorrect

**Solution**:

1. Generate new long-lived token in HA
2. Update `.env` with new token
3. Restart Vega server

### TTS not working

**Problem**: Audio not playing on device

**Solution**:

```yaml
# Check TTS service in HA
# Try simpler service first:
HASS_TTS_SERVICE=tts.google_translate_say

# Verify media player entity exists:
# Go to HA → Developer Tools → States
# Search for "media_player"
# Use correct entity_id in config
```

### Wake word not detected

**Problem**: HA Assist not triggering

**Solution**:

1. Test HA Assist directly (without wake word)
2. Check microphone permissions on device
3. Verify wake word engine is running (openWakeWord add-on)
4. Try different wake word sensitivity settings
5. Use manual trigger (tap microphone) first to verify automation

### Responses too slow

**Problem**: > 5 second delay

**Solution**:

- Use local STT/TTS services (Whisper, Piper) instead of cloud
- Check network latency between HA and Vega
- Reduce `CONTEXT_WINDOW_SIZE` in Vega config
- Optimize LLM model (use faster model)

---

## API Reference

### POST /hass/webhook

Receive voice commands from Home Assistant.

**Headers:**

```
X-API-Key: YOUR_VEGA_API_KEY
Content-Type: application/json
```

**Request Body:**

```json
{
  "text": "what's the weather",
  "conversation_id": "abc123",
  "device_id": "iphone_john",
  "device_type": "ios",
  "user_id": "user123",
  "language": "en"
}
```

**Response:**

```json
{
  "success": true,
  "response": "The weather today is sunny with a high of 75°F.",
  "tts_sent": true,
  "session_id": "session_xyz"
}
```

### GET /hass/status

Check integration health and configuration.

**Headers:**

```
X-API-Key: YOUR_VEGA_API_KEY
```

**Response:**

```json
{
  "enabled": true,
  "configured": true,
  "connected": true,
  "url": "http://homeassistant.local:8123",
  "webhook_endpoint": "/hass/webhook",
  "message": "Home Assistant integration operational"
}
```

---

## Example Automations

### Smart Home Control via Voice

```yaml
# Vega can now control HA devices!
# In Vega's LLM response, include HA service calls
```

Talk to Vega:

```
"Hey Vega, turn off all the lights"
"Hey Vega, set the thermostat to 72 degrees"
"Hey Vega, what devices are currently on?"
```

### Notifications & Reminders

```
"Hey Vega, remind me to take out the trash in 2 hours"
"Hey Vega, send a notification to my phone saying dinner is ready"
```

### Multi-Room Audio

Configure different media players per room:

```yaml
automation:
  - alias: "Vega Voice - Kitchen"
    trigger:
      - platform: conversation
    condition:
      - condition: state
        entity_id: input_select.active_room
        state: "Kitchen"
    action:
      - service: rest_command.vega_webhook
        data:
          text: "{{ trigger.text }}"
          media_player: "media_player.kitchen_speaker"
```

---

## Security Considerations

### Network Security

- ✅ **Local Network Only**: Default config uses local URLs
- ✅ **API Key Required**: All webhook calls require authentication
- ✅ **HTTPS Support**: Can configure TLS for HA URL
- ⚠️ **External Access**: If exposing HA externally, use VPN or Nabu Casa Cloud

### Privacy

- ✅ **No Cloud Required**: Can run 100% local (Whisper + Piper + Ollama)
- ✅ **Conversation History**: Stored locally on Vega server
- ✅ **No Third-Party**: Data never leaves your infrastructure

### Access Control

```bash
# Use different API keys for different devices/users
API_KEYS_EXTRA=key1,key2,key3

# Rotate tokens regularly
# Generate new HA token monthly
```

---

## Performance Tuning

### Optimize for Speed

```bash
# Use local STT/TTS (fastest)
# Home Assistant: Whisper + Piper

# Reduce context window
CONTEXT_WINDOW_SIZE=5  # Default: 10

# Use faster LLM model
MODEL_NAME=llama3:8b  # Instead of 70b
```

### Optimize for Quality

```bash
# Use cloud STT/TTS (best quality)
# Home Assistant: Nabu Casa Cloud or ElevenLabs

# Increase context window
CONTEXT_WINDOW_SIZE=20

# Use better LLM model
MODEL_NAME=llama3:70b
```

---

## Next Steps

1. **✅ Test basic voice commands** - Try simple queries first
2. **🎙️ Configure custom wake word** - Make it personal
3. **📱 Install HA app on all devices** - iPhone, Watch, Mac, etc.
4. **🏠 Add smart home control** - Let Vega control your devices
5. **🤖 Explore advanced features** - Notifications, automations, etc.

---

## Support & Resources

- **Vega Documentation**: `/docs/`
- **Home Assistant Docs**: <https://www.home-assistant.io/voice_control/>
- **Wyoming Protocol**: <https://github.com/rhasspy/wyoming>
- **Community Forums**: (link to your support channels)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR DEVICES                            │
│  📱 iPhone    💻 Mac    ⌚ Watch    🪟 Windows   🌐 Browser  │
│           ↓                                                  │
│       HA Companion Apps (with microphone access)           │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              HOME ASSISTANT (Raspberry Pi)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Wake Word Detection (openWakeWord/Porcupine)     │  │
│  │  2. Speech-to-Text (Whisper/Google/Cloud)           │  │
│  │  3. Conversation Agent (automation routing)          │  │
│  │  4. Text-to-Speech (Piper/Google/Cloud)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ↕ HTTP                                │
│              (POST /hass/webhook)                           │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              VEGA SERVER (Rack Mount)                        │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  1. Receive webhook with transcribed text            │  │
│  │  2. Load conversation context from database          │  │
│  │  3. Process with LLM (Ollama/OpenAI/etc)            │  │
│  │  4. Send TTS response back to HA                     │  │
│  │  5. Log conversation for persistence                 │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  💾 SQLite DB (conversation history, persistent sessions)   │
└─────────────────────────────────────────────────────────────┘
                           ↓
                  (TTS plays on user's device)
```

**End Result**: "Hey Vega" works from ANY device, ANYWHERE, with full conversation memory! 🎉
