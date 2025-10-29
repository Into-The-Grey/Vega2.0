# Home Assistant YAML Configuration for Vega Voice Integration

## Quick Setup for Your Environment

This configuration is tailored for:

- **Vega Server**: Headless rack-mount server (no speakers/mic)
- **Home Assistant**: Raspberry Pi on same network
- **Devices**: iPhone 15 Pro Max, Mac, Windows, Apple Watch, iPad

---

## Step 1: Add to `configuration.yaml`

```yaml
# ============================================================================
# VEGA VOICE ASSISTANT INTEGRATION
# ============================================================================

# REST command to call Vega webhook
rest_command:
  vega_chat:
    url: "http://YOUR_VEGA_SERVER_IP:8000/hass/webhook"
    method: POST
    headers:
      X-API-Key: !secret vega_api_key
      Content-Type: "application/json"
    payload: >
      {
        "text": "{{ text }}",
        "conversation_id": "{{ conversation_id | default('') }}",
        "device_id": "{{ device_id | default('unknown') }}",
        "device_type": "{{ device_type | default('unknown') }}",
        "user_id": "{{ user_id | default('') }}",
        "language": "en"
      }
    timeout: 30

# Input select for tracking active conversation
input_text:
  vega_last_conversation_id:
    name: "Vega Last Conversation ID"
    initial: ""
    max: 255

# Automation to route Assist to Vega
automation:
  - id: vega_voice_assistant
    alias: "Vega Voice Assistant"
    description: "Routes voice commands to Vega AI server"
    mode: queued
    max: 10
    trigger:
      # Trigger on any voice input
      - platform: conversation
        command:
          - "{action}"
    action:
      - service: rest_command.vega_chat
        data:
          text: "{{ trigger.text }}"
          conversation_id: "{{ states('input_text.vega_last_conversation_id') }}"
          device_id: "{{ trigger.context.user_id }}_{{ trigger.platform }}"
          device_type: >
            {% if 'mobile_app' in trigger.platform %}
              {% if 'iphone' in trigger.platform.lower() or 'ipad' in trigger.platform.lower() %}
                ios
              {% elif 'watch' in trigger.platform.lower() %}
                ios
              {% else %}
                unknown
              {% endif %}
            {% else %}
              browser
            {% endif %}
          user_id: "{{ trigger.context.user_id }}"
      
      # Update conversation ID for context continuity
      - service: input_text.set_value
        target:
          entity_id: input_text.vega_last_conversation_id
        data:
          value: "{{ now().timestamp() | int }}_{{ trigger.context.user_id }}"

# Template sensors for Vega status monitoring
template:
  - sensor:
      - name: "Vega Integration Status"
        unique_id: vega_integration_status
        state: >
          {{ 'Connected' if states('sensor.vega_last_response_time') != 'unknown' else 'Disconnected' }}
        icon: mdi:robot

# Optional: Notification when Vega is ready
notify:
  - name: vega_ready
    platform: notify
    message: "Vega AI voice assistant is ready!"
```

---

## Step 2: Add to `secrets.yaml`

```yaml
# VEGA INTEGRATION
vega_api_key: "your_vega_api_key_here"
vega_server_url: "http://192.168.1.XXX:8000"
```

---

## Step 3: Configure Voice Pipeline in HA

### Via UI (Recommended)

1. **Go to Settings → Voice Assistants**
2. **Click "Add Assistant"**
3. **Configure:**
   - **Name**: Vega
   - **Conversation Agent**: Home Assistant (will route via automation)
   - **Speech-to-Text**: Choose provider:
     - **Recommended**: "Faster Whisper" (local, fast)
     - Alternative: "Google Cloud Speech-to-Text"
     - Alternative: "Nabu Casa Cloud STT"
   - **Text-to-Speech**: Choose provider:
     - **Recommended**: "Piper" (local, high quality)
     - Alternative: "Google Translate TTS" (free)
     - Alternative: "Nabu Casa Cloud TTS" (best quality)
   - **Wake Word**: Install openWakeWord add-on (see Step 4)

4. **Save and Test**

---

## Step 4: Install Wake Word Detection (openWakeWord)

### Installation

1. **Go to Settings → Add-ons → Add-on Store**
2. **Search**: "openWakeWord"
3. **Click**: "openWakeWord"
4. **Install**
5. **Configuration**:

```yaml
# openWakeWord configuration
models:
  - hey_jarvis
  - alexa
  - ok_nabu
custom_model_dir: /share/openwakeword
threshold: 0.5
trigger_level: 1
```

6. **Start** the add-on
7. **Enable** "Start on boot"

### Create Custom "Hey Vega" Wake Word (Optional)

#### Option A: Use Existing Model

Rename "Hey Jarvis" mentally to "Hey Vega" and use `hey_jarvis` model.

#### Option B: Train Custom Model (Advanced)

1. Record 50+ samples of "Hey Vega"
2. Use openWakeWord training tools
3. Upload model to `/share/openwakeword/`
4. Add to config: `- hey_vega`

#### Option C: Use Porcupine (Easier Custom Words)

1. **Install**: Wyoming-Porcupine add-on instead
2. **Go to**: <https://console.picovoice.ai/>
3. **Create**: Custom wake word "Hey Vega"
4. **Download**: `.ppn` file
5. **Upload**: To HA and configure in add-on

---

## Step 5: Device-Specific Configuration

### For iPhone 15 Pro Max

1. **Install**: Home Assistant iOS app from App Store
2. **Login**: To your HA instance
3. **Settings → Voice Assistant**:
   - Enable "Allow Voice Assistant"
   - Select "Vega" as default assistant
4. **Shortcuts (Optional)**:
   - Create Siri shortcut: "Hey Siri, ask Vega..."

### For Mac

1. **Option A**: Install HA macOS app
2. **Option B**: Use browser (Safari/Chrome)
3. **Add to Dock** for quick access

### For Apple Watch

1. **Install**: HA Watch app (comes with iOS app)
2. **Complications**: Add HA widget to watch face
3. **Raise wrist → tap HA complication → microphone**

### For Windows PC

1. **Option A**: Install HA Windows app
2. **Option B**: Use browser (Edge/Chrome)
3. **Pin to taskbar** for quick access

---

## Step 6: Test the Integration

### Test 1: Check Vega Status

```bash
# From your terminal
curl -H "X-API-Key: YOUR_API_KEY" http://VEGA_IP:8000/hass/status

# Expected response:
{
  "enabled": true,
  "configured": true,
  "connected": true,
  "url": "http://homeassistant.local:8123",
  "message": "Home Assistant integration operational"
}
```

### Test 2: Manual Webhook Test

```bash
curl -X POST http://VEGA_IP:8000/hass/webhook \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello Vega, are you there?",
    "device_id": "test_device",
    "device_type": "ios"
  }'
```

### Test 3: Voice Command from iPhone

1. Open HA app
2. Tap microphone icon
3. Say: **"What's the weather?"**
4. Vega should respond through iPhone speakers

### Test 4: Wake Word (if configured)

1. Say: **"Hey Vega"** (or your custom wake word)
2. Wait for beep/confirmation
3. Say: **"What time is it?"**
4. Vega responds

---

## Step 7: Advanced Configuration

### Per-Device TTS Voices

```yaml
automation:
  - alias: "Vega Voice - High Quality for iPhone"
    trigger:
      - platform: conversation
        command: ["{action}"]
    condition:
      - condition: template
        value_template: "{{ 'iphone' in trigger.platform.lower() }}"
    action:
      - service: rest_command.vega_chat
        data:
          text: "{{ trigger.text }}"
      # Then override TTS for iPhone
      - service: tts.cloud_say
        data:
          entity_id: media_player.iphone_john
          message: "{{ states('sensor.vega_last_response') }}"
          options:
            voice: "en-US-Neural2-J"  # High-quality neural voice
```

### Smart Home Control via Vega

```yaml
# Allow Vega to control HA devices
script:
  vega_execute_command:
    alias: "Vega Execute HA Command"
    mode: queued
    fields:
      command:
        description: "Command from Vega"
        example: "turn_off_lights"
    sequence:
      - choose:
          - conditions: "{{ command == 'turn_off_lights' }}"
            sequence:
              - service: light.turn_off
                target:
                  entity_id: all
          - conditions: "{{ command == 'set_thermostat' }}"
            sequence:
              - service: climate.set_temperature
                data:
                  temperature: "{{ temperature }}"
```

### Multi-Room Audio

```yaml
# Route Vega responses to different rooms based on where you spoke
automation:
  - alias: "Vega Voice - Multi-Room"
    trigger:
      - platform: conversation
    action:
      - service: rest_command.vega_chat
      - delay: 1  # Wait for Vega response
      - service: tts.speak
        data:
          entity_id: >
            {% if 'kitchen' in trigger.area_id %}
              media_player.kitchen_speaker
            {% elif 'bedroom' in trigger.area_id %}
              media_player.bedroom_speaker
            {% else %}
              media_player.{{ trigger.platform | regex_replace('mobile_app_', '') }}
            {% endif %}
          message: "{{ states('sensor.vega_last_response') }}"
```

---

## Troubleshooting

### Issue: "Could not reach Vega server"

**Fix**:

```yaml
# Add to configuration.yaml
rest_command:
  vega_chat:
    # ... existing config ...
    verify_ssl: false  # If using self-signed cert
    timeout: 60        # Increase timeout if slow network
```

### Issue: "No TTS response"

**Fix**: Check HA media player states

```bash
# In HA Developer Tools → States
# Search for: media_player
# Verify entity IDs are correct
```

### Issue: "Wake word not detected"

**Fix**: Adjust sensitivity in openWakeWord config

```yaml
threshold: 0.3  # Lower = more sensitive (default 0.5)
trigger_level: 1
```

### Issue: "Conversation context lost"

**Fix**: Verify conversation ID persistence

```yaml
# Check input_text.vega_last_conversation_id
# Should persist across multiple commands
```

---

## Performance Optimization

### For Fastest Response

```yaml
# Use local STT/TTS
voice_assistant:
  stt: faster_whisper  # Local, fast
  tts: piper           # Local, high quality
  
# Reduce Vega context window
# In Vega .env:
CONTEXT_WINDOW_SIZE=5  # Default: 10
```

### For Best Quality

```yaml
# Use cloud STT/TTS
voice_assistant:
  stt: cloud_stt       # Nabu Casa
  tts: cloud_say       # Nabu Casa
  
# Increase Vega context
CONTEXT_WINDOW_SIZE=20
```

---

## Next Steps

1. ✅ Copy this YAML to your `configuration.yaml`
2. ✅ Update IP addresses and API keys
3. ✅ Restart Home Assistant
4. ✅ Test from your iPhone: "Hey Vega, hello!"
5. ✅ Enjoy ambient AI from any device!

---

## Support

For issues or questions:

- Check main docs: `docs/HOME_ASSISTANT_VOICE_INTEGRATION.md`
- Test endpoints: `curl http://VEGA_IP:8000/hass/status`
- Check HA logs: Settings → System → Logs
- Check Vega logs: `journalctl -u vega -f`
