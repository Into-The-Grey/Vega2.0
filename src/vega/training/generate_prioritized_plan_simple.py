"""
VEGA Prioritized Training Plan Generator
=========================================

Splits the full 20,040-line dataset into priority-based training sessions:
- CRITICAL: Essential for basic functionality (~3,000 lines)
- HIGH: Important for natural interaction (~5,000 lines)
- MEDIUM: Enhances expressiveness (~6,000 lines)
- LOW: Adds variety and edge cases (~4,000 lines)
- OPTIONAL: Specialized/advanced content (~2,040 lines)

Each priority level is split into manageable 50-line recording sessions.
"""

import csv
from pathlib import Path
from typing import Dict, List
from collections import Counter, defaultdict


class VoiceLine:
    """Voice training line with metadata"""

    def __init__(self, row: Dict[str, str]):
        self.id = row["id"]
        self.category = row["category"]
        self.text = row["text"]
        self.emotion = row["emotion"]
        self.style = row["style"]


class PrioritizedPlanGenerator:
    """Generate prioritized training plan"""

    # Category priority mapping (higher = more critical)
    CATEGORY_PRIORITY = {
        # CRITICAL (10) - Essential everyday communication
        "everyday": 10,
        "agreements": 10,
        "denials_corrections": 10,
        "question_variants": 10,
        "imperative": 10,
        "error_messages": 10,
        # HIGH (8) - Important conversational elements
        "hesitations_fillers": 8,
        "interjection": 8,
        "interrogative": 8,
        "conditionals": 8,
        "comparatives_superlatives": 8,
        "dates_times": 8,
        "numeric": 8,
        "numbers_large": 8,
        "measurements_units": 8,
        # MEDIUM (6) - Expressiveness & variety
        "emotional": 6,
        "technical": 6,
        "creative": 6,
        "narrative": 6,
        "structured": 6,
        "paragraph": 6,
        "sarcastic": 6,
        "friendly": 6,
        # LOW (4) - Nice to have
        "dramatic": 4,
        "philosophical": 4,
        "sales_pitch": 4,
        "sports_cast": 4,
        "weather_colorful": 4,
        "stream_of_consciousness": 4,
        "meta": 4,
        "finance": 4,
        # OPTIONAL (2) - Specialized/advanced
        "tongue_twister": 2,
        "multilingual": 2,
        "dialect_southern_us": 2,
        "expletive-heavy": 2,
        "dark_humor": 2,
        "dystopian": 2,
        "gibberish": 2,
        "glitchy_ai": 2,
        "laughter_breaths": 2,
    }

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.lines: List[VoiceLine] = []
        self.priority_buckets: Dict[str, List[VoiceLine]] = {
            "CRITICAL": [],
            "HIGH": [],
            "MEDIUM": [],
            "LOW": [],
            "OPTIONAL": [],
        }

    def load_dataset(self):
        """Load voice lines from CSV"""
        print(f"📂 Loading dataset: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    self.lines.append(VoiceLine(row))
                except Exception as e:
                    print(f"⚠️  Skipped invalid line: {e}")

        print(f"✅ Loaded {len(self.lines):,} voice lines")

    def prioritize_lines(self):
        """Assign lines to priority buckets"""
        print("\n🔄 Prioritizing lines...")

        for line in self.lines:
            # Get base priority from category
            priority_score = self.CATEGORY_PRIORITY.get(line.category, 3)

            # Adjust based on characteristics
            text_length = len(line.text)

            # Short lines are easier for initial training
            if text_length < 30:
                priority_score += 1
            elif text_length > 200:
                priority_score -= 1

            # Neutral emotion easier for initial training
            if line.emotion == "neutral":
                priority_score += 0.5

            # Assign to bucket
            if priority_score >= 9:
                self.priority_buckets["CRITICAL"].append(line)
            elif priority_score >= 7:
                self.priority_buckets["HIGH"].append(line)
            elif priority_score >= 5:
                self.priority_buckets["MEDIUM"].append(line)
            elif priority_score >= 3:
                self.priority_buckets["LOW"].append(line)
            else:
                self.priority_buckets["OPTIONAL"].append(line)

        # Show distribution
        print("\n📊 Priority Distribution:")
        total = 0
        for priority, lines in self.priority_buckets.items():
            count = len(lines)
            total += count
            percentage = (count / len(self.lines)) * 100
            sessions = (count + 49) // 50
            hours = (count * 0.25) / 60
            print(
                f"   {priority:10s}: {count:5,d} lines ({percentage:5.1f}%) → {sessions:3d} sessions (~{hours:.1f}h)"
            )
        print(f"   {'TOTAL':10s}: {total:5,d} lines")

    def generate_sessions(
        self, output_dir: str = "datasets/voice_training_prioritized"
    ):
        """Generate training session files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Generating sessions: {output_path}")

        session_info = []

        for priority, lines in self.priority_buckets.items():
            if not lines:
                continue

            # Create priority directory
            priority_dir = output_path / priority.lower()
            priority_dir.mkdir(exist_ok=True)

            # Split into 50-line sessions
            session_size = 50
            num_sessions = (len(lines) + session_size - 1) // session_size

            print(f"\n   {priority}: {len(lines):,} lines → {num_sessions} sessions")

            for session_idx in range(num_sessions):
                start_idx = session_idx * session_size
                end_idx = min(start_idx + session_size, len(lines))
                session_lines = lines[start_idx:end_idx]

                # Generate filename
                session_num = session_idx + 1
                filename = f"{priority.lower()}_session_{session_num:03d}.txt"
                filepath = priority_dir / filename

                # Write session file
                self._write_session_file(
                    filepath,
                    priority,
                    session_num,
                    num_sessions,
                    session_lines,
                    start_idx,
                )

                session_info.append(
                    {
                        "priority": priority,
                        "session": session_num,
                        "file": str(filepath.relative_to(output_path)),
                        "lines": len(session_lines),
                    }
                )

                if session_num <= 3 or session_num == num_sessions:
                    print(f"      ✅ {filename}: {len(session_lines)} lines")
                elif session_num == 4:
                    print(f"      ... ({num_sessions - 6} more sessions) ...")

        # Generate master files
        self._generate_index(output_path, session_info)
        self._generate_readme(output_path)

        print(f"\n✅ Complete! Generated {len(session_info)} training sessions")

    def _write_session_file(
        self,
        filepath: Path,
        priority: str,
        session_num: int,
        total_sessions: int,
        lines: List[VoiceLine],
        start_idx: int,
    ):
        """Write individual session file"""
        with open(filepath, "w", encoding="utf-8") as f:
            # Header
            f.write(f"# VEGA Voice Training - {priority} Priority\n")
            f.write(f"# Session {session_num} of {total_sessions}\n")
            f.write(f"# Lines: {len(lines)}\n")
            f.write(f"#\n")

            # Priority description
            descriptions = {
                "CRITICAL": (
                    "⚠️  CRITICAL: Essential for basic VEGA functionality",
                    "Record these FIRST for minimum viable voice model",
                ),
                "HIGH": (
                    "🔥 HIGH: Important for natural conversation",
                    "Record after CRITICAL for good voice quality",
                ),
                "MEDIUM": (
                    "📊 MEDIUM: Adds expressiveness and variety",
                    "Record for well-rounded voice model",
                ),
                "LOW": (
                    "📝 LOW: Nice-to-have for completeness",
                    "Record when you have extra time",
                ),
                "OPTIONAL": (
                    "⭐ OPTIONAL: Specialized/advanced content",
                    "Record for maximum coverage and flexibility",
                ),
            }

            desc1, desc2 = descriptions.get(priority, ("", ""))
            f.write(f"# {desc1}\n")
            f.write(f"# {desc2}\n")
            f.write(f"#\n")

            # Timing estimate
            min_time = len(lines) * 0.2
            max_time = len(lines) * 0.3
            f.write(
                f"# Estimated recording time: {min_time:.0f}-{max_time:.0f} minutes\n"
            )
            f.write(f"#{'='*70}\n\n")

            # Lines
            for idx, line in enumerate(lines, 1):
                line_num = start_idx + idx
                f.write(
                    f"{line_num:04d} | {line.id} | [{line.category:25s}] {line.text}\n"
                )

    def _generate_index(self, output_dir: Path, sessions: List[Dict]):
        """Generate master index"""
        index_file = output_dir / "INDEX.txt"

        with open(index_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("VEGA VOICE TRAINING - MASTER INDEX\n")
            f.write("=" * 80 + "\n\n")
            f.write("Complete prioritized training plan for all 20,040 voice lines.\n")
            f.write(
                "Organized by priority level, split into 50-line recording sessions.\n\n"
            )

            # Group by priority
            current_priority = None
            priority_totals = defaultdict(int)

            for session in sessions:
                priority = session["priority"]

                if priority != current_priority:
                    if current_priority:
                        f.write(
                            f"   Subtotal: {priority_totals[current_priority]:,} lines\n\n"
                        )

                    current_priority = priority
                    f.write(f"\n{'='*80}\n")
                    f.write(f"{priority} PRIORITY\n")
                    f.write(f"{'='*80}\n\n")

                f.write(f"  {session['file']:60s} ({session['lines']:2d} lines)\n")
                priority_totals[priority] += session["lines"]

            # Final subtotal
            if current_priority:
                f.write(f"   Subtotal: {priority_totals[current_priority]:,} lines\n\n")

            # Total
            total_lines = sum(priority_totals.values())
            f.write(f"\n{'='*80}\n")
            f.write(f"TOTAL: {total_lines:,} lines across {len(sessions)} sessions\n")
            f.write(f"{'='*80}\n")

        print(f"\n   📋 Index: {index_file.name}")

    def _generate_readme(self, output_dir: Path):
        """Generate README"""
        readme_file = output_dir / "README.md"

        with open(readme_file, "w", encoding="utf-8") as f:
            f.write("# VEGA Voice Training - Prioritized Plan\n\n")
            f.write("Complete 20,040-line dataset organized by priority.\n\n")

            f.write("## Priority Levels\n\n")

            for priority, lines in self.priority_buckets.items():
                count = len(lines)
                if count == 0:
                    continue

                sessions = (count + 49) // 50
                hours = (count * 0.25) / 60

                emoji_map = {
                    "CRITICAL": "⚠️",
                    "HIGH": "🔥",
                    "MEDIUM": "📊",
                    "LOW": "📝",
                    "OPTIONAL": "⭐",
                }

                f.write(f"### {emoji_map.get(priority, '')} {priority}\n\n")
                f.write(f"- Lines: {count:,}\n")
                f.write(f"- Sessions: {sessions}\n")
                f.write(f"- Est. time: {hours:.1f} hours\n\n")

            f.write("## Recording Order\n\n")
            f.write("```\n")
            f.write("1. CRITICAL  ← Start here for basic functionality\n")
            f.write("2. HIGH      ← Continue for natural conversation\n")
            f.write("3. MEDIUM    ← Add expressiveness\n")
            f.write("4. LOW       ← Complete coverage\n")
            f.write("5. OPTIONAL  ← Maximum flexibility\n")
            f.write("```\n\n")

            f.write("## Quick Start\n\n")
            f.write("```bash\n")
            f.write("# View first session\n")
            f.write("cat critical/critical_session_001.txt\n\n")
            f.write("# After recording, import\n")
            f.write(
                "python src/vega/training/voice_training.py batch --dir recordings/\n"
            )
            f.write("```\n")

        print(f"   📖 README: {readme_file.name}")


def main():
    """Main entry point"""
    print("=" * 80)
    print("VEGA Voice Training - Prioritized Plan Generator")
    print("=" * 80 + "\n")

    # Initialize
    csv_path = "datasets/voice_lines/VEGA_Voice_Training_List_of_20K_Lines.csv"
    generator = PrioritizedPlanGenerator(csv_path)

    # Process
    generator.load_dataset()
    generator.prioritize_lines()
    generator.generate_sessions()

    # Done
    print("\n" + "=" * 80)
    print("✅ COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review: datasets/voice_training_prioritized/INDEX.txt")
    print(
        "2. Start: datasets/voice_training_prioritized/critical/critical_session_001.txt"
    )
    print("3. Record with consistent audio quality")
    print(
        "4. Import: python src/vega/training/voice_training.py batch --dir recordings/"
    )
    print("\nHappy recording! 🎙️\n")


if __name__ == "__main__":
    main()
