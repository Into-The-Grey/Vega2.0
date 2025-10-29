# 🎙️ VEGA Voice Training - READY TO RECORD

## ✅ Setup Complete

**Dependencies installed**: librosa, soundfile, torch, torchaudio  
**Dataset loaded**: 20,040 professional voice lines  
**Training plan generated**: 402 sessions across 5 priority levels

## 📊 Your Training Plan

### Priority Breakdown

| Priority | Lines | Sessions | Time | Status |
|----------|-------|----------|------|--------|
| ⚠️ **CRITICAL** | 2,249 | 45 | ~9.4h | **START HERE** |
| 🔥 **HIGH** | 2,761 | 56 | ~11.5h | After CRITICAL |
| 📊 **MEDIUM** | 2,347 | 47 | ~9.8h | Good coverage |
| 📝 **LOW** | 9,839 | 197 | ~41.0h | Complete |
| ⭐ **OPTIONAL** | 2,844 | 57 | ~11.8h | Maximum |
| **TOTAL** | **20,040** | **402** | **~83.5h** | |

### What Each Priority Includes

**⚠️ CRITICAL** (Record first!)

- Everyday phrases: "Okay", "Copy that", "Got it"
- Agreements/denials: Yes/no responses
- Questions: Basic interrogatives
- Commands: Imperative forms
- Error messages: System responses

**🔥 HIGH** (Record second)

- Conversational fillers: "Um", "Uh", "Like"
- Interjections: "Hey", "Listen", "FYI"
- Conditionals: If/then statements
- Numbers & dates: Temporal data
- Measurements: Quantities & units

**📊 MEDIUM** (For expressiveness)

- Emotional range: Full spectrum
- Technical terms: Specialized vocabulary
- Creative content: Narratives
- Structured speech: Paragraphs
- Sarcasm & tone variations

**📝 LOW** (Nice to have)

- Dramatic delivery
- Philosophical content
- Sales pitch style
- Weather descriptions
- Meta-commentary

**⭐ OPTIONAL** (Advanced)

- Tongue twisters (pronunciation)
- Multilingual phrases
- Dialectal variations
- Specialized humor
- Sound effects

## 🚀 Quick Start Commands

### 1. View Your First Session

```bash
cat datasets/voice_training_prioritized/critical/critical_session_001.txt
```

Sample output:

```
# VEGA Voice Training - CRITICAL Priority
# Session 1 of 45
# Lines: 50
# Estimated recording time: 10-15 minutes

0001 | VT-72314 | [agreements] Okay, I'm in.—got it.
0002 | VT-36182 | [agreements] Reminder: Copy that.—copy.
...
```

### 2. Browse All Sessions

```bash
# View master index
cat datasets/voice_training_prioritized/INDEX.txt

# List CRITICAL sessions
ls datasets/voice_training_prioritized/critical/

# View README
cat datasets/voice_training_prioritized/README.md
```

### 3. After Recording (import your audio)

```bash
# Single file
python src/vega/training/voice_training.py add \
  --file recordings/session_001_line_001.wav \
  --text "Okay, I'm in."

# Batch import entire session
python src/vega/training/voice_training.py batch \
  --dir recordings/critical_session_001/

# Check status
python src/vega/training/voice_training.py status
```

## 📁 File Structure

```
datasets/voice_training_prioritized/
├── INDEX.txt                    # Master index (all 402 sessions)
├── README.md                    # Quick reference
│
├── critical/                    # ⚠️ 45 sessions (2,249 lines)
│   ├── critical_session_001.txt
│   ├── critical_session_002.txt
│   └── ...
│
├── high/                        # 🔥 56 sessions (2,761 lines)
│   ├── high_session_001.txt
│   ├── high_session_002.txt
│   └── ...
│
├── medium/                      # 📊 47 sessions (2,347 lines)
│   ├── medium_session_001.txt
│   └── ...
│
├── low/                         # 📝 197 sessions (9,839 lines)
│   ├── low_session_001.txt
│   └── ...
│
└── optional/                    # ⭐ 57 sessions (2,844 lines)
    ├── optional_session_001.txt
    └── ...
```

## 🎯 Recording Recommendations

### Minimum Viable Voice Model

**Record:** CRITICAL only (45 sessions, ~9.4 hours)  
**Result:** Basic functional VEGA voice

### Good Quality Voice Model

**Record:** CRITICAL + HIGH (101 sessions, ~20.9 hours)  
**Result:** Natural-sounding conversational VEGA

### Professional Voice Model

**Record:** CRITICAL + HIGH + MEDIUM (148 sessions, ~30.7 hours)  
**Result:** Expressive, versatile VEGA voice

### Complete Voice Model

**Record:** All 402 sessions (~83.5 hours)  
**Result:** Maximum coverage and flexibility

## 📝 Recording Workflow

### Session-by-Session Approach (Recommended)

1. **Choose session** (start with `critical_session_001.txt`)
2. **Set up recording** (quiet room, consistent mic position)
3. **Record 50 lines** (10-15 minutes)
4. **Save audio files** (WAV format recommended)
5. **Import to system** (batch command)
6. **Move to next session**

### Example: First Session

```bash
# 1. View session
cat datasets/voice_training_prioritized/critical/critical_session_001.txt

# 2. Create recording directory
mkdir -p recordings/critical_001

# 3. Record 50 lines using Audacity/OBS
#    Save each as: VT-{id}.wav (e.g., VT-72314.wav)

# 4. Import recordings
python src/vega/training/voice_training.py batch \
  --dir recordings/critical_001/

# 5. Check quality
python src/vega/training/voice_training.py status
```

### Recording Tips

- **Environment**: Quiet room, no echo
- **Microphone**: 6-8 inches away, consistent position
- **Format**: WAV 44.1kHz, mono, 16-bit
- **Session length**: 15-20 minutes max (take breaks!)
- **Quality over speed**: Clear articulation beats rushing
- **Emotion tags**: Follow the emotion/style in brackets

## 📊 Progress Tracking Template

Create a file `recording_progress.txt`:

```
VEGA Voice Training Progress
============================

Date Started: [DATE]

⚠️ CRITICAL (Priority 1)
[  ] Session 001 (50 lines) - Date: ___
[  ] Session 002 (50 lines) - Date: ___
[  ] Session 003 (50 lines) - Date: ___
... (45 total)

Total CRITICAL: 0/45 sessions (0/2,249 lines)

🔥 HIGH (Priority 2)
[  ] Session 001 (50 lines) - Date: ___
... (56 total)

📊 MEDIUM (Priority 3)
... (47 total)

📝 LOW (Priority 4)
... (197 total)

⭐ OPTIONAL (Priority 5)
... (57 total)

GRAND TOTAL: 0/402 sessions (0/20,040 lines)
```

## 🎯 Milestones

- [ ] **First 50 lines** - You've started!
- [ ] **500 lines** - Minimum viable training data
- [ ] **CRITICAL complete** (2,249 lines) - Basic functionality
- [ ] **CRITICAL + HIGH** (5,010 lines) - Good conversation quality
- [ ] **First 10,000 lines** - Professional-grade voice
- [ ] **All 20,040 lines** - Maximum VEGA capability! 🎉

## 🔧 Voice Training System

Your recordings integrate with the VEGA personality system:

```bash
# Analyze voice profile
python src/vega/training/voice_training.py analyze

# View personality stats (includes voice training)
python -c "from vega.personality.vega_core import get_vega_personality; \
  p = get_vega_personality(); print(p.get_personality_stats())"
```

## 🚀 Ready to Start

**Your first session is waiting:**

```bash
cat datasets/voice_training_prioritized/critical/critical_session_001.txt
```

**50 lines, 10-15 minutes, and you're on your way to training VEGA with your voice!** 🎙️✨

---

**Questions?** Check the comprehensive guides:

- `docs/VOICE_TRAINING_GUIDE.md` - Complete documentation
- `docs/VOICE_TRAINING_WORKFLOW.md` - Step-by-step workflow
- `datasets/voice_training_prioritized/README.md` - Quick reference
