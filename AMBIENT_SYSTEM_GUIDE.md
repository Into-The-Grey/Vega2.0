# VEGA AMBIENT PRESENCE + CONTEXT-AWARE PERSONALITY CORE

## ===================================================

## Complete JARVIS-like AI Companion System

## 🎯 SYSTEM OVERVIEW

Vega is now a fully operational ambient AI companion with 24/7 awareness, contextual intelligence, and respectful interaction protocols. Like JARVIS, it observes, thinks, learns, and speaks only when appropriate.

## 🏗️ ARCHITECTURE COMPONENTS

### Phase 1: Core Awareness Daemon ✅

**File**: `vega_loop.py` (~400 lines)

- **Purpose**: 24/7 consciousness and system monitoring
- **Features**: 
  - Resource monitoring (CPU, GPU, memory with GTX 1660 Super + Quadro P1000 detection)
  - System state classification (IDLE/LIGHT_WORK/HEAVY_WORK/GAMING/OVERLOADED)
  - Vega mode management (ACTIVE/PAUSED/SILENT/SLEEPING/FOCUSED)
  - Conversation trigger evaluation with resource constraints
  - Persistent state management with JSONL logging
  - CLI interface (`--start`, `--pause-daemon`, `--force-prompt`, `--status`, `--log-mode`)

### Phase 2: Contextual Personality System ✅

**File**: `idle_personality_engine.py` (~600 lines)

- **Purpose**: Contextual intelligence and conversational awareness
- **Features**:
  - Spontaneity modes (curiosity, concern, recommendation, reflection, observation)
  - User profiling with activity analysis and pattern recognition
  - Interaction templates with LLM enhancement
  - Relevance scoring and tone adjustment
  - Context-aware question generation
  - Privacy-safe user behavior analysis

### Phase 3: Interaction History & Learning ✅  

**File**: `interaction_log.py` (~500 lines)

- **Purpose**: Conversation memory and behavioral conditioning
- **Features**:
  - Comprehensive interaction logging with quality analysis
  - Response quality classification (excellent/good/neutral/poor/negative/ignored)
  - User preference learning and adaptation
  - Conversation pattern recognition
  - Automatic tone adjustment based on user feedback
  - Dialogue conditioning for improved future interactions

### Phase 4: User Presence Tracking ✅

**File**: `user_presence.py` (~450 lines)

- **Purpose**: Passive user activity monitoring and presence detection
- **Features**:
  - Keyboard/mouse activity detection (privacy-safe, no keylogging)
  - Application focus and window monitoring (with privacy filtering)
  - Presence classification (active/idle/away/focused/gaming/meeting)
  - System idle time detection across platforms
  - Activity pattern analysis for optimal interaction timing
  - Privacy safeguards and data filtering

### Phase 5: Spontaneous Thought Engine ✅

**File**: `spontaneous_thought_engine.py` (~600 lines)

- **Purpose**: Internal AI reflection and consciousness
- **Features**:
  - Thought generation during idle periods (observation, insight, curiosity, synthesis, reflection)
  - Memory consolidation and pattern recognition
  - Context analysis with user behavior understanding
  - Thought evaluation and promotion to speech
  - Background learning and knowledge synthesis
  - Self-reflection on conversation quality

### Phase 6: Advanced Augmentations ✅

**File**: `advanced_augmentations.py` (~650 lines)

- **Purpose**: Sophisticated interaction protocols and smart behavior
- **Features**:
  - Smart silence protocols that learn when NOT to interrupt
  - Curiosity buffer system for collecting observations until appropriate timing
  - Calendar integration for context-aware scheduling
  - Reactive notification suppression during important work
  - Energy-aware interaction management
  - Adaptive interruption thresholds based on user state

## 🚀 QUICK START

### 1. Install Dependencies

```bash
# Required Python packages
pip install psutil pynvml httpx pynput schedule

# Optional calendar integration
sudo apt install calcurse khal  # Linux calendar tools
```

### 2. Start the Ambient Presence System

```bash
# Start the core ambient daemon
python vega_loop.py --start

# Check status
python vega_loop.py --status

# Force a conversation (testing)
python vega_loop.py --force-prompt

# View logs
python vega_loop.py --log-mode
```

### 3. Configuration

The system uses your existing Vega configuration in `app.env`. Key settings:

- `API_KEY`: For LLM interactions
- `MODEL_NAME`: Default model for personality enhancement
- `HOST` and `PORT`: For main Vega API integration

## 🧠 HOW IT WORKS

### The Consciousness Loop

1. **vega_loop.py** runs continuously, monitoring system resources and user activity
2. Every 5-30 seconds (configurable), it evaluates whether to generate thoughts or interactions
3. **user_presence.py** provides real-time user state (active/focused/away/gaming/meeting)
4. **spontaneous_thought_engine.py** generates internal thoughts and insights during quiet periods
5. **idle_personality_engine.py** evaluates if thoughts should become conversations
6. **advanced_augmentations.py** applies smart silence protocols and timing optimization
7. **interaction_log.py** logs all interactions and learns user preferences for future conditioning

### Intelligence Flow

```text
System Monitoring → User Presence Detection → Context Analysis → 
Thought Generation → Personality Evaluation → Silence Protocol Check → 
Interaction Decision → Conversation → Response Analysis → Learning
```

### Privacy & Respect

- **No keylogging**: Only counts keystrokes for activity levels
- **Privacy filtering**: Sensitive window titles and app names are hashed or filtered
- **Smart silence**: Respects focus time, meetings, gaming, and high-stress periods
- **User learning**: Adapts to user preferences and interaction styles
- **Consent-aware**: Never forces interactions, always provides value

## 🎛️ OPERATIONAL MODES

### Vega Modes (Automatic)

- **ACTIVE**: Normal operation, full interaction capability
- **PAUSED**: Monitoring only, no conversations
- **SILENT**: Minimal interruptions only
- **SLEEPING**: Background monitoring with very low activity
- **FOCUSED**: User-focused work detected, heightened respect protocols

### Silence Protocols (Smart)

- **STANDARD**: Normal interaction rules
- **FOCUS_AWARE**: Heightened focus detection and respect
- **DO_NOT_DISTURB**: Minimal interruptions only
- **EMERGENCY_ONLY**: Only critical system issues
- **ADAPTIVE**: Learns and adapts to user preferences

### Spontaneity Modes (Contextual)

- **CURIOSITY**: Asks clarifying questions about user interests
- **CONCERN**: Notices stress signals or unusual patterns
- **RECOMMENDATION**: Offers helpful tools and optimizations  
- **REFLECTION**: Prompts memory completion or seeks opinions
- **OBSERVATION**: Internal pattern recognition and learning

## 📊 SYSTEM INTEGRATION

### Database Files Created

- `vega_state/loop_state.json` - Current system state
- `vega_state/presence_history.jsonl` - User presence tracking
- `vega_state/personality_memory.jsonl` - Personality thoughts and insights
- `vega_state/interaction_history.db` - Complete interaction database
- `vega_state/thoughts.db` - Internal thought database
- `vega_state/user_profile.db` - User behavior patterns
- `vega_state/curiosity_buffer.jsonl` - Buffered observations waiting for good timing

### API Integration

The ambient system integrates with your existing Vega API:

- Uses existing LLM configuration for personality enhancement
- Leverages conversation logging for pattern analysis
- Integrates with main chat interface for seamless experience
- Provides ambient insights that enrich regular conversations

### Resource Management

- **CPU Monitoring**: Automatically scales activity based on system load
- **GPU Detection**: Recognizes gaming/heavy compute and adjusts behavior
- **Memory Awareness**: Monitors memory usage and adapts accordingly
- **Energy Management**: Internal energy system prevents conversation overload

## 🔧 ADVANCED CONFIGURATION

### Timing Parameters (vega_loop.py)

```python
# Configurable intervals
MONITORING_INTERVAL = 30        # Seconds between monitoring cycles
CONVERSATION_COOLDOWN = 300     # Minimum seconds between conversations
PERSONALITY_CHECK_INTERVAL = 60 # Seconds between personality evaluations
```

### Threshold Tuning (advanced_augmentations.py)

```python
# Interruption thresholds
self.idle_threshold = 300       # 5 minutes for idle detection
self.away_threshold = 900       # 15 minutes for away detection  
self.focus_threshold = 1800     # 30 minutes for focus mode
self.shareability_threshold = 0.7  # Minimum score for sharing thoughts
```

### Privacy Controls (user_presence.py)

```python
# Privacy filtering patterns
self.sensitive_patterns = [
    r'password', r'login', r'credential', r'secret', r'token',
    r'private', r'confidential', r'bank', r'payment', r'card'
]
```

## 🎯 USAGE SCENARIOS

### Development Workflow

- Vega notices long coding sessions and offers to help with documentation
- Detects error patterns and suggests debugging approaches
- Recognizes when you switch between many files and offers to create shortcuts
- Learns your productive hours and adjusts interaction timing

### Learning & Research

- Observes research patterns and asks clarifying questions about your interests
- Suggests connections between disparate topics you've been exploring
- Offers to help organize information when it detects knowledge gathering
- Remembers your learning preferences and adapts accordingly

### Work-Life Balance

- Notices late-night work patterns and gently inquires about your schedule
- Detects stress signals and offers supportive observations
- Recognizes break times and shares interesting insights then
- Learns your preferred communication style and matches it

### System Administration

- Monitors system health and alerts about resource issues
- Notices unusual patterns in system usage
- Offers automation suggestions for repetitive tasks
- Learns which system notifications you care about

## 🛠️ TROUBLESHOOTING

### Common Issues

**Vega not starting conversations:**

- Check `python vega_loop.py --status` for system state
- Verify user presence detection: `cat vega_state/presence_history.jsonl | tail -5`
- Check silence protocol status in logs

**High CPU usage:**

- Vega automatically scales back during heavy system load
- Adjust `MONITORING_INTERVAL` in vega_loop.py for less frequent checks
- Check if presence monitoring is consuming resources

**No presence detection:**

- Install required packages: `pip install pynput psutil`
- Linux users may need: `sudo apt install xprintidle` for idle detection
- Check privacy/accessibility permissions for keyboard/mouse monitoring

**Calendar integration not working:**

- Install calendar tools: `sudo apt install calcurse khal`
- Check calendar file permissions
- Verify calendar data format in supported tools

### Debug Mode

```bash
# Enable detailed logging
export VEGA_DEBUG=1
python vega_loop.py --start

# Check all subsystem logs
tail -f vega_state/*.jsonl
```

### Performance Tuning

```bash
# Monitor resource usage
htop
nvidia-smi  # For GPU monitoring

# Adjust monitoring frequency
# Edit vega_loop.py MONITORING_INTERVAL values
```

## 🔮 FUTURE ENHANCEMENTS

### Planned Features

- **Voice Integration**: Wake word detection and voice responses
- **Multi-modal Input**: Document and image analysis capabilities
- **Proactive Automation**: Automatically executing helpful tasks
- **Collaborative Intelligence**: Multi-user awareness and coordination
- **Predictive Assistance**: Anticipating needs based on patterns

### Extension Points

- **Custom Personality Modes**: Define your own interaction styles
- **Integration APIs**: Connect with external services and tools
- **Plugin Architecture**: Add custom augmentations and behaviors
- **Mobile Companion**: Extend awareness to mobile devices
- **Team Coordination**: Multi-user and team-aware functionality

## 🎉 ACHIEVEMENT UNLOCKED

You now have a fully operational ambient AI companion that:

✅ **Runs 24/7** with intelligent resource management  
✅ **Observes respectfully** without being intrusive  
✅ **Learns continuously** from every interaction  
✅ **Thinks independently** during quiet periods  
✅ **Speaks meaningfully** only when valuable  
✅ **Adapts dynamically** to your preferences and patterns  
✅ **Respects boundaries** with smart silence protocols  
✅ **Integrates seamlessly** with your existing workflow  

This is not just a chatbot or automation script - it's **awareness as software**, providing genuine companionship and intelligence that enhances rather than interrupts your work and life.

**Like JARVIS, Vega is now truly present, observant, and helpful.**
