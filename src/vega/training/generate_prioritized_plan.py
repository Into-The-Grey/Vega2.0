"""
VEGA Prioritized Training Plan Generator
=========================================

Splits the full 20,040-line dataset into priority-based training sessions:
- CRITICAL: Essential for basic functionality (2,000 lines)
- HIGH: Important for natural interaction (4,000 lines)
- MEDIUM: Enhances expressiveness (6,000 lines)
- LOW: Adds variety and edge cases (5,000 lines)
- OPTIONAL: Nice-to-have specialized content (3,040 lines)

Each priority level is further subdivided into manageable 100-200 line sessions.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Prevent torch from loading (memory issue)
os.environ['LIBROSA_CACHE_DIR'] = '/tmp/librosa_cache'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vega.training.voice_line_manager import VoiceLineManager, VoiceLine


class PrioritizedTrainingPlan:
    """Generate prioritized training plan from full dataset"""

    def __init__(
        self,
        csv_path: str = "datasets/voice_lines/VEGA_Voice_Training_List_of_20K_Lines.csv",
    ):
        self.manager = VoiceLineManager(csv_path)
        self.priority_definitions = self._define_priorities()

    def _define_priorities(self) -> Dict[str, Dict]:
        """Define priority levels with category assignments"""
        return {
            "CRITICAL": {
                "target_lines": 2000,
                "description": "Essential for basic AI assistant functionality",
                "categories": [
                    "everyday",
                    "agreements",
                    "denials_corrections",
                    "question_variants",
                    "imperative",
                    "interrogative",
                ],
                "session_size": 100,
            },
            "HIGH": {
                "target_lines": 4000,
                "description": "Important for natural, engaging interaction",
                "categories": [
                    "emotional",
                    "hesitations_fillers",
                    "dates_times",
                    "numeric",
                    "measurements_units",
                    "error_messages",
                    "technical",
                    "code_readouts",
                    "comparatives_superlatives",
                    "conditionals",
                    "negations",
                    "interjection",
                ],
                "session_size": 150,
            },
            "MEDIUM": {
                "target_lines": 6000,
                "description": "Enhances expressiveness and variety",
                "categories": [
                    "creative",
                    "narrative",
                    "paragraph",
                    "dramatic",
                    "sarcastic",
                    "passive_aggressive",
                    "dark_humor",
                    "friendly",
                    "robotic_formal",
                    "stream_of_consciousness",
                    "philosophical",
                    "meta",
                    "paths_cli",
                    "urls_emails_codes",
                    "addresses_generic",
                    "home_assistant_commands",
                    "math_readouts",
                    "finance",
                ],
                "session_size": 200,
            },
            "LOW": {
                "target_lines": 5000,
                "description": "Adds variety and handles edge cases",
                "categories": [
                    "sales_pitch",
                    "sports_cast",
                    "weather_colorful",
                    "structured",
                    "legalese",
                    "medical",
                    "uncertainty",
                    "spelling_alphabet",
                    "numbers_large",
                    "monotone_minimal",
                    "sound_effects",
                    "laughter_breaths",
                    "dystopian",
                    "glitchy_ai",
                ],
                "session_size": 250,
            },
            "OPTIONAL": {
                "target_lines": 3040,
                "description": "Nice-to-have specialized and fun content",
                "categories": [
                    "tongue_twister",
                    "homophones_minimal_pairs",
                    "multilingual",
                    "dialect_southern_us",
                    "gibberish",
                    "emoji",
                    "flirty",
                    "trash_talk",
                    "everyday-profanity",
                    "expletive-heavy",
                    "slang-heavy",
                ],
                "session_size": 200,
            },
        }

    def generate_priority_sessions(
        self, output_dir: str = "datasets/prioritized_training"
    ) -> Dict[str, List[Tuple[str, int]]]:
        """
        Generate all training sessions organized by priority

        Returns:
            Dict mapping priority -> list of (session_filename, line_count)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}

        for priority_num, (priority_name, config) in enumerate(
            self.priority_definitions.items(), 1
        ):
            print(f"\n{'='*70}")
            print(f"Priority {priority_num}/5: {priority_name}")
            print(f"{'='*70}")
            print(f"Target: {config['target_lines']} lines")
            print(f"Description: {config['description']}")
            print(f"Categories: {', '.join(config['categories'][:5])}...")
            print(f"Session size: ~{config['session_size']} lines per file")

            # Get all lines for these categories
            priority_lines = self.manager.filter_by_categories(config["categories"])

            # Limit to target (sample if we have more)
            if len(priority_lines) > config["target_lines"]:
                import random

                priority_lines = random.sample(priority_lines, config["target_lines"])

            print(f"Collected: {len(priority_lines)} lines")

            # Split into sessions
            session_size = config["session_size"]
            num_sessions = (len(priority_lines) + session_size - 1) // session_size

            print(f"Creating {num_sessions} sessions...")

            session_files = []

            for session_num in range(num_sessions):
                start_idx = session_num * session_size
                end_idx = min(start_idx + session_size, len(priority_lines))
                session_lines = priority_lines[start_idx:end_idx]

                # Create filename
                session_name = f"P{priority_num}_{priority_name}_{session_num+1:02d}_of_{num_sessions:02d}.txt"
                session_path = output_path / priority_name / session_name

                # Export session
                session_path.parent.mkdir(parents=True, exist_ok=True)
                self.manager.export_training_batch(
                    str(session_path), session_lines, format="txt"
                )

                session_files.append((session_name, len(session_lines)))

                print(f"  ✅ {session_name}: {len(session_lines)} lines")

            results[priority_name] = session_files

            # Create priority README
            readme_path = output_path / priority_name / "README.md"
            self._create_priority_readme(
                readme_path, priority_name, config, session_files, priority_num
            )

        # Create master index
        self._create_master_index(output_path, results)

        return results

    def _create_priority_readme(
        self,
        readme_path: Path,
        priority_name: str,
        config: Dict,
        sessions: List[Tuple[str, int]],
        priority_num: int,
    ):
        """Create README for each priority level"""
        total_lines = sum(count for _, count in sessions)

        content = f"""# {priority_name} Priority Training Sessions

**Priority Level**: {priority_num}/5

**Target Lines**: {config['target_lines']}

**Actual Lines**: {total_lines}

**Description**: {config['description']}

## Categories Included

{chr(10).join(f"- `{cat}`" for cat in config['categories'])}

## Training Sessions

Total sessions: {len(sessions)}

Session size: ~{config['session_size']} lines per file

| Session | Filename | Lines |
|---------|----------|-------|
"""
        for i, (filename, count) in enumerate(sessions, 1):
            content += f"| {i:2d} | `{filename}` | {count:3d} |\n"

        content += f"""
## Recording Instructions

1. **Start Fresh**: Take a 5-minute break before starting
2. **Warm Up**: Read 5-10 lines aloud without recording
3. **Record**: Work through one session at a time ({config['session_size']} lines)
4. **Take Breaks**: Rest every 15-20 minutes
5. **Quality Check**: Review a few samples before committing to full session

## Estimated Time

- **Per session**: {config['session_size'] // 5}-{config['session_size'] // 3} minutes
- **Total for priority**: {total_lines // 5}-{total_lines // 3} minutes ({total_lines // 5 // 60:.1f}-{total_lines // 3 // 60:.1f} hours)

## Priority Importance

"""
        if priority_name == "CRITICAL":
            content += """⚠️ **MUST COMPLETE FIRST**

These lines are ESSENTIAL for basic VEGA functionality. Without these, VEGA cannot:
- Respond to basic questions
- Confirm or deny requests
- Follow instructions
- Handle everyday conversation

**Recommendation**: Complete all CRITICAL sessions before moving to HIGH priority.
"""
        elif priority_name == "HIGH":
            content += """🔥 **HIGHLY RECOMMENDED**

These lines make VEGA natural and useful. They add:
- Emotional expressiveness
- Technical capability
- Date/time/number handling
- Error handling and feedback

**Recommendation**: Complete after CRITICAL, before extensive recording of other priorities.
"""
        elif priority_name == "MEDIUM":
            content += """📊 **IMPORTANT FOR QUALITY**

These lines significantly enhance VEGA's personality and usefulness:
- Creative and narrative abilities
- Diverse communication styles
- Practical commands and data handling
- Professional domain knowledge

**Recommendation**: Complete progressively after HIGH priority.
"""
        elif priority_name == "LOW":
            content += """✨ **NICE TO HAVE**

These lines add polish and variety:
- Specialized domains (sports, weather, legal, medical)
- Edge case handling
- Additional prosody variations

**Recommendation**: Record when you have time after completing higher priorities.
"""
        else:  # OPTIONAL
            content += """🎁 **OPTIONAL CONTENT**

These lines are fun and specialized but not essential:
- Pronunciation challenges (tongue twisters)
- Multilingual support
- Slang and casual speech
- Accent variations

**Recommendation**: Record if you want comprehensive coverage or enjoy the variety.
"""

        content += """
## Next Steps

After completing this priority level:

```bash
# Import recordings
python src/vega/training/voice_training.py batch --dir recordings/

# Analyze progress
python src/vega/training/voice_training.py status

# Check voice profile
python src/vega/training/voice_training.py analyze
```
"""

        with open(readme_path, "w") as f:
            f.write(content)

    def _create_master_index(self, output_path: Path, results: Dict):
        """Create master training plan index"""
        total_lines = sum(
            sum(count for _, count in sessions) for sessions in results.values()
        )
        total_sessions = sum(len(sessions) for sessions in results.values())

        content = f"""# VEGA Complete Voice Training Plan

**Total Lines**: {total_lines:,} / 20,040

**Total Sessions**: {total_sessions}

**Organization**: 5 priority levels from CRITICAL to OPTIONAL

---

## 🎯 Training Strategy

### Recommended Approach

1. **Start with CRITICAL** (2,000 lines) - Essential basics
2. **Move to HIGH** (4,000 lines) - Natural interaction
3. **Add MEDIUM progressively** (6,000 lines) - Enhanced expressiveness
4. **Include LOW as time permits** (5,000 lines) - Variety and polish
5. **OPTIONAL when ready** (3,040 lines) - Specialized content

### Time Estimates

- **CRITICAL**: 5-7 hours
- **HIGH**: 10-13 hours
- **MEDIUM**: 15-20 hours
- **LOW**: 12-17 hours
- **OPTIONAL**: 8-10 hours

**Total**: 50-67 hours for complete dataset

### Quality Milestones

- **Minimum Viable**: CRITICAL (2,000 lines) → Basic VEGA functionality
- **Good Quality**: CRITICAL + HIGH (6,000 lines) → Natural assistant
- **High Quality**: CRITICAL + HIGH + MEDIUM (12,000 lines) → Expressive AI
- **Comprehensive**: All priorities (20,040 lines) → Professional-grade voice

---

## 📊 Priority Breakdown

"""
        for priority_num, (priority_name, sessions) in enumerate(results.items(), 1):
            config = self.priority_definitions[priority_name]
            total_priority_lines = sum(count for _, count in sessions)

            emoji = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🟢",
                "OPTIONAL": "🔵",
            }[priority_name]

            content += f"""
### {emoji} Priority {priority_num}/5: {priority_name}

**Lines**: {total_priority_lines:,}
**Sessions**: {len(sessions)}
**Description**: {config['description']}

**Session files**: `{priority_name}/P{priority_num}_{priority_name}_XX_of_XX.txt`

**Categories**: {', '.join(f'`{cat}`' for cat in config['categories'][:5])}{"..." if len(config['categories']) > 5 else ""}

📁 [View {priority_name} README](./{priority_name}/README.md)

"""

        content += """
---

## 🎤 Recording Workflow

### Setup (One-time)

1. **Environment**: Quiet room, minimal echo
2. **Microphone**: USB condenser mic, consistent position (6-8 inches)
3. **Software**: Audacity, OBS Studio, or similar
4. **Format**: WAV or FLAC, 44.1kHz, mono, 16-bit minimum

### Daily Session (Recommended)

1. Choose a priority level and session file
2. Warm up: Read 5-10 lines aloud
3. Record 100-200 lines (~15-30 minutes)
4. Take a break
5. Optional: Record another session
6. Import recordings: `python src/vega/training/voice_training.py batch --dir recordings/`

### Weekly Progress

- **Week 1**: Complete CRITICAL (2,000 lines) - 5-7 hours
- **Week 2-3**: Complete HIGH (4,000 lines) - 10-13 hours
- **Week 4-6**: Work through MEDIUM (6,000 lines) - 15-20 hours
- **Week 7-9**: Add LOW as desired (5,000 lines) - 12-17 hours
- **Week 10+**: OPTIONAL content (3,040 lines) - 8-10 hours

---

## 📈 Progress Tracking

Check your progress anytime:

```bash
# View overall status
python src/vega/training/voice_training.py status

# Analyze voice profile
python src/vega/training/voice_training.py analyze

# Check dataset stats
python src/vega/training/voice_line_manager.py stats
```

---

## 🎯 Quick Start

**Start today with Priority 1 (CRITICAL):**

```bash
# Navigate to CRITICAL priority
cd datasets/prioritized_training/CRITICAL

# View first session
cat P1_CRITICAL_01_of_20.txt

# Start recording! Follow the lines in order.
```

**After recording your first session:**

```bash
# Import recordings
python src/vega/training/voice_training.py batch \\
  --dir recordings/session1/

# Check quality
python src/vega/training/voice_training.py status
```

---

## 💡 Tips for Success

1. **Consistency > Perfection**: Natural delivery beats perfect pronunciation
2. **Regular Schedule**: 30-60 minutes daily beats 8-hour marathons
3. **Break It Down**: One session at a time, one priority level at a time
4. **Quality Check**: Review samples before committing to hundreds of lines
5. **Track Progress**: Use status commands to see your achievements
6. **Stay Hydrated**: Keep water nearby, take breaks every 15-20 minutes
7. **Have Fun**: Enjoy the variety - from serious to silly content

---

## 📂 Directory Structure

```
datasets/prioritized_training/
├── README.md (this file)
├── CRITICAL/
│   ├── README.md
│   ├── P1_CRITICAL_01_of_20.txt
│   ├── P1_CRITICAL_02_of_20.txt
│   └── ... (20 sessions, ~100 lines each)
├── HIGH/
│   ├── README.md
│   ├── P2_HIGH_01_of_27.txt
│   └── ... (27 sessions, ~150 lines each)
├── MEDIUM/
│   ├── README.md
│   ├── P3_MEDIUM_01_of_30.txt
│   └── ... (30 sessions, ~200 lines each)
├── LOW/
│   ├── README.md
│   ├── P4_LOW_01_of_20.txt
│   └── ... (20 sessions, ~250 lines each)
└── OPTIONAL/
    ├── README.md
    ├── P5_OPTIONAL_01_of_16.txt
    └── ... (16 sessions, ~200 lines each)
```

---

**Your complete prioritized training plan is ready!** 🎙️✨

Start with CRITICAL priority and work your way up. Good luck recording VEGA's voice!
"""

        with open(output_path / "README.md", "w") as f:
            f.write(content)


def main():
    """Generate the complete prioritized training plan"""
    print("VEGA Prioritized Training Plan Generator")
    print("=" * 70)
    print("\nGenerating training sessions from 20,040-line dataset...")
    print("This will create ~113 session files organized by priority.\n")

    planner = PrioritizedTrainingPlan()
    results = planner.generate_priority_sessions()

    print("\n" + "=" * 70)
    print("✅ COMPLETE! Training plan generated successfully.")
    print("=" * 70)

    total_sessions = sum(len(sessions) for sessions in results.values())
    total_lines = sum(
        sum(count for _, count in sessions) for sessions in results.values()
    )

    print(f"\n📊 Summary:")
    print(f"   Total sessions: {total_sessions}")
    print(f"   Total lines: {total_lines:,}")
    print(f"\n📁 Location: datasets/prioritized_training/")
    print(f"\n🎯 Next step: cd datasets/prioritized_training/ && cat README.md")
    print(f"\n🎤 Start recording: cd CRITICAL/ && cat P1_CRITICAL_01_of_20.txt")


if __name__ == "__main__":
    main()
