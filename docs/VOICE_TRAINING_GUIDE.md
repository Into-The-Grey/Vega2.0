# VEGA Voice Training Guide

## Overview

Your voice training dataset contains **20,040 professionally-crafted lines** across **60 diverse categories** designed to train VEGA to recognize and synthesize your voice with maximum expressiveness and accuracy.

## Dataset Statistics

- **Total Lines**: 20,040
- **Categories**: 60 (from everyday speech to technical jargon)
- **Emotions**: 27 different emotional states
- **Text Length**: 3-509 characters (avg: 64 chars)
- **Balance**: 334 lines per category (uniform distribution)

## Categories Include

### Everyday Communication

- `everyday` - Common phrases and responses
- `agreements` / `denials_corrections` - Yes/no responses
- `greetings` - Hello, goodbye, etc.
- `hesitations_fillers` - Um, uh, like, you know

### Technical & Professional

- `technical` - Technical terminology
- `code_readouts` - Code snippets and syntax
- `paths_cli` - File paths and terminal commands
- `error_messages` - Error handling responses
- `finance` / `legalese` / `medical` - Domain-specific vocabulary

### Creative & Expressive

- `creative` / `narrative` / `paragraph` - Longer passages
- `emotional` - Full range of emotions
- `dramatic` / `dystopian` - Theatrical delivery
- `dark_humor` / `sarcastic` - Humor styles
- `flirty` / `passive_aggressive` - Subtle tones

### Voice Quality Training

- `tongue_twister` - Articulation challenges
- `homophones_minimal_pairs` - Pronunciation accuracy
- `multilingual` - Multiple languages
- `dialect_southern_us` - Accent variation
- `sound_effects` / `laughter_breaths` - Non-verbal sounds

### Prosody & Style

- `monotone_minimal` - Flat delivery
- `robotic_formal` - Precise articulation
- `stream_of_consciousness` - Natural flow
- `sales_pitch` / `sports_cast` - Dynamic styles
- `philosophical` / `meta` - Abstract concepts

### Data Types

- `numeric` / `numbers_large` - Number reading
- `dates_times` - Temporal expressions
- `measurements_units` - Quantities
- `math_readouts` - Mathematical expressions
- `spelling_alphabet` - Letter-by-letter
- `addresses_generic` - Addresses and locations
- `urls_emails_codes` - Web addresses and codes

### Specialized

- `home_assistant_commands` - Smart home control
- `trash_talk` / `expletive-heavy` - Casual/profane
- `glitchy_ai` - Distorted/broken speech
- `gibberish` - Nonsense sounds
- `emoji` - Emoji descriptions

## Quick Start Commands

### View Statistics

```bash
python src/vega/training/voice_line_manager.py stats
```

### List All Categories

```bash
python src/vega/training/voice_line_manager.py categories
```

### Get Random Sample (10 lines)

```bash
python src/vega/training/voice_line_manager.py sample --count 10
```

### Get Balanced Sample (100 lines across all categories)

```bash
python src/vega/training/voice_line_manager.py sample --count 100 --balanced
```

### Filter by Category

```bash
python src/vega/training/voice_line_manager.py sample --count 20 --category technical
```

### Search for Specific Text

```bash
python src/vega/training/voice_line_manager.py search "example text" --max 20
```

## Progressive Training Sessions

Generate difficulty-scaled training sessions:

```bash
# Create 4 sessions with 100 total lines (25 each)
python src/vega/training/voice_line_manager.py progressive \
  --total 100 \
  --sessions 4 \
  --output-dir datasets/voice_training_sessions
```

This creates:

1. **01_easy_warmup.txt** - Short, simple everyday phrases
2. **02_medium_practice.txt** - Standard complexity (50-150 chars)
3. **03_hard_challenge.txt** - Technical, long, or complex lines
4. **04_mixed_variety.txt** - Balanced mix across all categories

### Custom Training Session

```bash
# 50 lines, hard difficulty, export to file
python src/vega/training/voice_line_manager.py session \
  --size 50 \
  --difficulty hard \
  --output datasets/my_training_session.txt
```

Difficulty levels:

- **easy**: Short everyday phrases, greetings, simple agreements
- **medium**: Standard length (50-150 chars)
- **hard**: Technical terms, tongue twisters, paragraphs, multilingual
- **mixed**: Random selection across all difficulties

## Recording Workflow

### Recommended Approach: Progressive Training

Start with smaller, easier sessions and build up:

```bash
# Week 1: Easy warmup (200 lines)
python src/vega/training/voice_line_manager.py session \
  --size 200 --difficulty easy \
  --output datasets/sessions/week1_easy.txt

# Week 2: Medium practice (300 lines)
python src/vega/training/voice_line_manager.py session \
  --size 300 --difficulty medium \
  --output datasets/sessions/week2_medium.txt

# Week 3: Hard challenge (300 lines)
python src/vega/training/voice_line_manager.py session \
  --size 300 --difficulty hard \
  --output datasets/sessions/week3_hard.txt

# Week 4: Mixed variety (500 lines)
python src/vega/training/voice_line_manager.py session \
  --size 500 --difficulty mixed \
  --output datasets/sessions/week4_mixed.txt
```

### Recording Tips

1. **Environment**: Quiet room, minimal echo
2. **Microphone**: Position consistently (6-8 inches away)
3. **Pace**: Natural speaking speed, clear articulation
4. **Sessions**: 30-45 minutes max to avoid fatigue
5. **Breaks**: Take breaks every 10-15 minutes
6. **Consistency**: Same time of day, same setup

### Audio Format Recommendations

- **Format**: WAV (uncompressed) or FLAC (lossless)
- **Sample Rate**: 44.1kHz or 48kHz
- **Bit Depth**: 16-bit or 24-bit
- **Channels**: Mono (single microphone)
- **File naming**: `VT-{id}_{category}.wav` for easy tracking

## Voice Training Integration

Once you have recordings, use the voice training system:

```bash
# Add single recording with transcription
python src/vega/training/voice_training.py add \
  --file recordings/VT-12345_technical.wav \
  --text "The API endpoint returned status code 429"

# Batch add all recordings from a directory
python src/vega/training/voice_training.py batch \
  --dir recordings/ \
  --pattern "*.wav"

# Analyze voice profile
python src/vega/training/voice_training.py analyze

# Train voice model
python src/vega/training/voice_training.py train --mode both

# Check status
python src/vega/training/voice_training.py status
```

## Emotional Range

Your dataset includes 27 emotional states for natural prosody:

- **Neutral**: calm, neutral, deadpan, monotone
- **Positive**: friendly, enthusiastic, confident, excited, amused
- **Negative**: annoyed, angry, sad, anxious, grim
- **Curious**: curious, analytical, thoughtful, reflective
- **Dynamic**: playful, intense, competitive, sarcastic, wry
- **Uncertain**: uncertain, surprised

## Category-Specific Training

Focus on specific categories if needed:

```bash
# Train on technical vocabulary
python src/vega/training/voice_line_manager.py sample \
  --count 100 --category technical \
  --output datasets/sessions/technical_focus.txt

# Train on emotional expressiveness
python src/vega/training/voice_line_manager.py sample \
  --count 100 --category emotional \
  --output datasets/sessions/emotional_focus.txt

# Train on pronunciation challenges
python src/vega/training/voice_line_manager.py sample \
  --count 100 --category tongue_twister \
  --output datasets/sessions/pronunciation_focus.txt
```

## Estimated Recording Time

Based on average line length (64 chars) and speaking rate:

- **100 lines**: ~15-20 minutes
- **500 lines**: ~75-100 minutes (2 sessions)
- **1,000 lines**: ~150-200 minutes (4-5 sessions)
- **5,000 lines**: ~12-16 hours (25-30 sessions)
- **20,040 lines** (full dataset): ~50-65 hours (100+ sessions)

**Recommendation**: Start with 500-1,000 high-quality recordings for initial training, then expand based on results.

## Quality Over Quantity

Focus on:

- ✅ Clear articulation
- ✅ Consistent microphone distance
- ✅ Natural emotional expression
- ✅ Minimal background noise
- ✅ Proper pacing and prosody

Rather than rushing through all 20K lines.

## Next Steps

1. **Generate first training session**:

   ```bash
   python src/vega/training/voice_line_manager.py progressive \
     --total 200 --sessions 4 \
     --output-dir datasets/voice_training_sessions
   ```

2. **Set up recording environment** (quiet room, good microphone)

3. **Record first session** (01_easy_warmup.txt - 50 lines)

4. **Add recordings to training system**:

   ```bash
   python src/vega/training/voice_training.py batch \
     --dir recordings/session1/
   ```

5. **Analyze voice profile and iterate**

---

**Your 20K line dataset is ready to train VEGA to speak with your voice!** 🎙️
