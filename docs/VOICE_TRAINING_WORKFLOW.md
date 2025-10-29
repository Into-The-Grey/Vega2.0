# VEGA Voice Training - Complete Workflow

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VEGA Voice Training System                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Voice Line Dataset (20,040 lines, 60 categories)        │
│     ├── voice_line_manager.py → Curate training sessions    │
│     └── Progressive difficulty scaling                       │
│                                                               │
│  2. Recording & Processing                                   │
│     ├── User records audio samples (WAV/FLAC)               │
│     ├── voice_training.py → Process & validate audio        │
│     └── Extract voice features (MFCC, pitch, timbre)        │
│                                                               │
│  3. Voice Profile & Personality                              │
│     ├── vega_core.py → Store voice characteristics          │
│     ├── Track training sessions in SQLite                    │
│     └── Evolve VEGA personality with voice identity         │
│                                                               │
│  4. Model Training (Future Integration)                      │
│     ├── STT: Fine-tune Whisper for voice recognition        │
│     ├── TTS: Train voice cloning (Coqui TTS/XTTS)          │
│     └── Deploy personalized voice models                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Step-by-Step Workflow

### Phase 1: Setup (5 minutes)

Install audio processing dependencies:

```bash
cd ~/Vega2.0
pip install librosa soundfile torch torchaudio
```

Verify installation:

```bash
python -c "import librosa, soundfile, torch, torchaudio; print('✅ Audio libraries ready')"
```

### Phase 2: Generate Training Sessions (2 minutes)

**Option A: Progressive Training** (Recommended for beginners)

```bash
# Generate 4 progressive sessions (200 total lines)
python src/vega/training/voice_line_manager.py progressive \
  --total 200 \
  --sessions 4 \
  --output-dir datasets/voice_training_sessions
```

Output:

- `01_easy_warmup.txt` - 50 short, simple lines
- `02_medium_practice.txt` - 50 medium complexity lines
- `03_hard_challenge.txt` - 50 technical/complex lines
- `04_mixed_variety.txt` - 50 mixed difficulty lines

**Option B: Custom Sessions**

```bash
# Category-specific training
python src/vega/training/voice_line_manager.py session \
  --size 100 \
  --difficulty easy \
  --output datasets/sessions/my_session.txt

# Focus on specific categories
python src/vega/training/voice_line_manager.py sample \
  --count 100 \
  --category technical \
  --output datasets/sessions/technical.txt
```

### Phase 3: Recording Setup (15 minutes)

**Hardware Requirements:**

- **Microphone**: USB condenser mic or headset (consistent quality)
- **Environment**: Quiet room, minimal echo
- **Position**: 6-8 inches from mic, consistent angle

**Software Options:**

1. **Audacity** (Free, cross-platform)

   ```bash
   sudo apt install audacity  # Linux
   # Download from audacityteam.org for Mac/Windows
   ```

2. **OBS Studio** (Free, for longer sessions)
3. **Reaper** (Professional, 60-day trial)

**Recording Settings:**

- Sample Rate: 44.1 kHz or 48 kHz
- Bit Depth: 16-bit (sufficient) or 24-bit (pro)
- Format: WAV (uncompressed) or FLAC (lossless)
- Channels: Mono

**File Naming Convention:**

```
VT-{line_id}_{category}.wav

Examples:
VT-12345_technical.wav
VT-67890_everyday.wav
```

### Phase 4: Recording Sessions (Varies)

**Time Estimates:**

- 50 lines: 10-15 minutes
- 200 lines: 40-60 minutes (2-3 sessions with breaks)
- 500 lines: 2-3 hours (5-6 sessions)
- 1000 lines: 4-6 hours (10-12 sessions)

**Recording Tips:**

1. **Warm Up**: Read 5-10 lines aloud before recording
2. **Pace**: Speak naturally, don't rush
3. **Emotions**: Follow emotion tags (neutral, enthusiastic, calm, etc.)
4. **Mistakes**: Don't worry about perfection, re-record if needed
5. **Breaks**: Rest every 15-20 minutes, drink water
6. **Consistency**: Same posture, distance, and energy throughout

**Example Recording Workflow:**

```bash
# Open training session file
cat datasets/voice_training_sessions/01_easy_warmup.txt

# For each line:
# 1. Read line aloud (practice)
# 2. Check emotion/style tags
# 3. Record with proper expression
# 4. Save as: recordings/VT-{id}_{category}.wav
# 5. Move to next line
```

**Batch Recording in Audacity:**

1. File → New
2. Record all lines with 2-second pauses between
3. Analyze → Silence Finder (to auto-detect boundaries)
4. File → Export → Export Multiple (auto-split by silence)
5. Use label track to name files

### Phase 5: Add Recordings to Training System (5 minutes)

**Single Recording:**

```bash
python src/vega/training/voice_training.py add \
  --file recordings/VT-12345_technical.wav \
  --text "The API returned status code 429" \
  --context "Technical error message"
```

**Batch Import:**

```bash
# Import all recordings from directory
python src/vega/training/voice_training.py batch \
  --dir recordings/session1/ \
  --pattern "*.wav"
```

The system will:

- ✅ Validate audio quality
- ✅ Extract voice features (MFCC, pitch, timbre)
- ✅ Assess quality score (0.0-1.0)
- ✅ Store in `data/voice_training/samples/`
- ✅ Log to personality database

### Phase 6: Analyze Voice Profile (2 minutes)

```bash
python src/vega/training/voice_training.py analyze
```

Output:

```
✅ Voice Profile Analysis
   Samples: 200
   Total duration: 450.3s
   Average quality: 0.87
   Voice pitch: 145.2Hz (±12.3Hz)
   
Profile saved to: data/voice_training/voice_profile.json
```

### Phase 7: Check Training Status (1 minute)

```bash
python src/vega/training/voice_training.py status
```

Output:

```
📊 VEGA Voice Training Status

Samples collected: 200
Total duration: 450.3s
Average quality: 0.87

✅ Voice profile created
   Profile samples: 200
   Voice pitch: 145.2Hz (±12.3Hz)

Voice training history:
   Total sessions: 5
   Total training time: 450s
```

### Phase 8: Train Voice Model (Future - Placeholder)

```bash
# When ready for model training
python src/vega/training/voice_training.py train --mode both
```

**Current Status**: 

- ✅ Voice sample collection: Fully implemented
- ✅ Feature extraction: Fully implemented
- ✅ Voice profile generation: Fully implemented
- ⚠️ Model training: Placeholder (needs TTS/STT integration)

**Next Implementation Steps**:

1. Integrate Coqui XTTS for voice cloning
2. Integrate Whisper fine-tuning for STT
3. Add model evaluation metrics
4. Deploy trained models to production

## Voice Line Dataset Overview

**Your 20,040-line dataset includes:**

### Core Categories (334 lines each)

**Everyday Speech**: agreements, denials, greetings, everyday, hesitations_fillers
**Technical**: technical, code_readouts, paths_cli, error_messages
**Emotional**: emotional, dramatic, dark_humor, sarcastic
**Prosody**: monotone_minimal, robotic_formal, stream_of_consciousness
**Pronunciation**: tongue_twister, homophones_minimal_pairs, multilingual
**Data Types**: numeric, dates_times, measurements_units, spelling_alphabet
**Specialized**: home_assistant_commands, trash_talk, glitchy_ai

### Emotional Coverage (27 emotions)

neutral (7,682), playful (1,336), curious/firm/annoyed (1,002 each), excited/uncertain (668 each), and 20 others (334 each)

### Text Length Distribution

- Min: 3 characters
- Max: 509 characters  
- Average: 64 characters
- Sweet spot: 30-100 characters (80% of dataset)

## Quick Reference Commands

```bash
# Dataset Management
python src/vega/training/voice_line_manager.py stats
python src/vega/training/voice_line_manager.py categories
python src/vega/training/voice_line_manager.py sample --count 10
python src/vega/training/voice_line_manager.py search "text query"

# Generate Training Sessions
python src/vega/training/voice_line_manager.py progressive \
  --total 200 --sessions 4 --output-dir datasets/sessions

python src/vega/training/voice_line_manager.py session \
  --size 100 --difficulty medium --output my_session.txt

# Voice Training
python src/vega/training/voice_training.py add --file audio.wav --text "..."
python src/vega/training/voice_training.py batch --dir recordings/
python src/vega/training/voice_training.py analyze
python src/vega/training/voice_training.py status
python src/vega/training/voice_training.py train --mode both

# Check Personality Integration
python -c "from vega.personality.vega_core import get_vega_personality; \
  p = get_vega_personality(); \
  print(p.get_personality_stats())"
```

## Recommended Training Paths

### Path 1: Quick Start (1-2 hours of recording)

```bash
# 200 lines, balanced difficulty
python src/vega/training/voice_line_manager.py progressive \
  --total 200 --sessions 4 --output-dir datasets/quick_start

# Record all 4 sessions (50 lines each)
# Import with batch command
# Analyze voice profile
```

### Path 2: Comprehensive (5-8 hours of recording)

```bash
# 1000 lines, full coverage
python src/vega/training/voice_line_manager.py sample \
  --count 1000 --balanced --output datasets/comprehensive.txt

# Record in 10-20 sessions over 1-2 weeks
# Regular breaks to maintain quality
```

### Path 3: Category Focus (Targeted training)

```bash
# Focus on specific categories
for cat in technical everyday emotional creative; do
  python src/vega/training/voice_line_manager.py sample \
    --count 100 --category $cat \
    --output datasets/focus_${cat}.txt
done
```

### Path 4: Full Dataset (50+ hours)

```bash
# All 20,040 lines - studio-grade recording project
# Requires professional setup and multiple recording sessions
# Best for commercial-quality voice synthesis
```

## Quality Benchmarks

**Minimum Viable Training**: 100-200 high-quality samples
**Good Training**: 500-1,000 samples
**Excellent Training**: 2,000-5,000 samples  
**Professional Grade**: 10,000+ samples

**Quality > Quantity**: 500 perfect recordings beat 2,000 rushed ones.

## Troubleshooting

### Audio Quality Issues

**Problem**: Quality score < 0.7

- Check: Background noise (turn off fans, close windows)
- Check: Microphone position (too close = distortion, too far = noise)
- Check: Recording levels (peak around -6dB to -12dB)

**Problem**: "Audio too short" error

- Minimum: 2 seconds per sample
- Solution: Speak at natural pace, don't rush

**Problem**: "Audio too long" warning

- Maximum: 30 seconds per sample
- Solution: System will auto-truncate, but verify text alignment

### Import Issues

**Problem**: Import fails with "Audio libraries not available"

```bash
pip install librosa soundfile torch torchaudio
```

**Problem**: Batch import skips files

- Check: File format (WAV/FLAC supported, MP3 needs conversion)
- Check: File naming matches pattern
- Check: Files not corrupted

## Files & Directories

```
Vega2.0/
├── datasets/
│   ├── voice_lines/
│   │   └── VEGA_Voice_Training_List_of_20K_Lines.csv  # Source dataset
│   └── voice_training_sessions/                        # Generated sessions
│       ├── 01_easy_warmup.txt
│       ├── 02_medium_practice.txt
│       ├── 03_hard_challenge.txt
│       └── 04_mixed_variety.txt
│
├── recordings/                                          # Your audio files
│   ├── session1/
│   ├── session2/
│   └── ...
│
├── data/
│   ├── voice_training/
│   │   ├── samples/                                    # Processed audio
│   │   ├── voice_profile.json                          # Voice characteristics
│   │   └── training_session_*.json                     # Session history
│   └── vega_personality.db                             # Training logs
│
└── src/vega/
    ├── training/
    │   ├── voice_line_manager.py                       # Dataset management
    │   └── voice_training.py                           # Training system
    └── personality/
        └── vega_core.py                                # Personality & logs
```

## Integration with VEGA Personality

Voice training is integrated with VEGA's personality system:

```python
from vega.personality.vega_core import get_vega_personality

personality = get_vega_personality()
stats = personality.get_personality_stats()

# View voice training history
print(f"Training sessions: {stats['voice_training']['sessions']}")
print(f"Total duration: {stats['voice_training']['total_duration_seconds']}s")
```

Voice profile is stored in VEGA's personality:

- Voice characteristics (pitch, timbre, energy)
- Training session history
- Quality metrics
- Evolution over time

## Next Steps

1. ✅ **Install dependencies**: `pip install librosa soundfile torch torchaudio`
2. ✅ **Generate first session**: `python src/vega/training/voice_line_manager.py progressive`
3. ⏳ **Set up recording environment** (quiet space, good mic)
4. ⏳ **Record first 50 lines** (easy warmup session)
5. ⏳ **Import recordings**: `python src/vega/training/voice_training.py batch --dir recordings/`
6. ⏳ **Analyze voice profile**: `python src/vega/training/voice_training.py analyze`
7. ⏳ **Continue with more sessions** based on results

---

**Your 20,040-line professional voice training dataset is ready!** 🎙️

Start with small sessions, focus on quality, and progressively build up your VEGA voice model.
