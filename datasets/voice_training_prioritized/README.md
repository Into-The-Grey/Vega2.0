# VEGA Voice Training - Prioritized Plan

Complete 20,040-line dataset organized by priority.

## Priority Levels

### ⚠️ CRITICAL

- Lines: 2,249
- Sessions: 45
- Est. time: 9.4 hours

### 🔥 HIGH

- Lines: 2,761
- Sessions: 56
- Est. time: 11.5 hours

### 📊 MEDIUM

- Lines: 2,347
- Sessions: 47
- Est. time: 9.8 hours

### 📝 LOW

- Lines: 9,839
- Sessions: 197
- Est. time: 41.0 hours

### ⭐ OPTIONAL

- Lines: 2,844
- Sessions: 57
- Est. time: 11.8 hours

## Recording Order

```
1. CRITICAL  ← Start here for basic functionality
2. HIGH      ← Continue for natural conversation
3. MEDIUM    ← Add expressiveness
4. LOW       ← Complete coverage
5. OPTIONAL  ← Maximum flexibility
```

## Quick Start

```bash
# View first session
cat critical/critical_session_001.txt

# After recording, import
python src/vega/training/voice_training.py batch --dir recordings/
```
