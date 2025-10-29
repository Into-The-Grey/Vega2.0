"""
VEGA Voice Line Manager
=======================

Manages the 20K+ voice training lines dataset for TTS model training.
Provides filtering, sampling, and batch processing capabilities for
progressive voice training sessions.

Dataset Structure:
- 20,040 unique voice lines
- 60+ categories (technical, emotional, creative, everyday, etc.)
- Metadata: emotion, style, pace, volume, prosody, tone, formality
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class VoiceLine:
    """Single voice training line with metadata"""

    id: str
    category: str
    text: str
    emotion: str
    style: str
    pace: str
    volume: str
    prosody: str
    tone: str
    formality: str
    medium: str
    pitch: str
    intensity: str
    energy_scale: int
    scenario: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.text}"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category,
            "text": self.text,
            "emotion": self.emotion,
            "style": self.style,
            "pace": self.pace,
            "volume": self.volume,
            "prosody": self.prosody,
            "tone": self.tone,
            "formality": self.formality,
            "medium": self.medium,
            "pitch": self.pitch,
            "intensity": self.intensity,
            "energy_scale": self.energy_scale,
            "scenario": self.scenario,
        }


class VoiceLineManager:
    """Manages voice training line dataset"""

    def __init__(
        self,
        csv_path: str = "datasets/voice_lines/VEGA_Voice_Training_List_of_20K_Lines.csv",
    ):
        self.csv_path = Path(csv_path)
        self.lines: List[VoiceLine] = []
        self.categories: Set[str] = set()
        self.load_dataset()

    def load_dataset(self):
        """Load voice lines from CSV"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Voice lines CSV not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    line = VoiceLine(
                        id=row["id"],
                        category=row["category"],
                        text=row["text"],
                        emotion=row["emotion"],
                        style=row["style"],
                        pace=row["pace"],
                        volume=row["volume"],
                        prosody=row["prosody"],
                        tone=row["tone"],
                        formality=row["formality"],
                        medium=row["medium"],
                        pitch=row["pitch"],
                        intensity=row["intensity"],
                        energy_scale=int(row["energy_scale"]),
                        scenario=row["scenario"],
                    )
                    self.lines.append(line)
                    self.categories.add(line.category)
                except Exception as e:
                    print(f"Warning: Skipped invalid line: {e}")
                    continue

        print(
            f"✅ Loaded {len(self.lines):,} voice lines across {len(self.categories)} categories"
        )

    def get_categories(self) -> List[str]:
        """Get sorted list of all categories"""
        return sorted(self.categories)

    def get_category_stats(self) -> Dict[str, int]:
        """Get count of lines per category"""
        return dict(Counter(line.category for line in self.lines))

    def filter_by_category(self, category: str) -> List[VoiceLine]:
        """Get all lines in a specific category"""
        return [line for line in self.lines if line.category == category]

    def filter_by_categories(self, categories: List[str]) -> List[VoiceLine]:
        """Get all lines matching any of the specified categories"""
        category_set = set(categories)
        return [line for line in self.lines if line.category in category_set]

    def filter_by_emotion(self, emotion: str) -> List[VoiceLine]:
        """Get all lines with specific emotion"""
        return [line for line in self.lines if line.emotion == emotion]

    def filter_by_energy(
        self, min_energy: int = 0, max_energy: int = 10
    ) -> List[VoiceLine]:
        """Get lines within energy scale range"""
        return [
            line for line in self.lines if min_energy <= line.energy_scale <= max_energy
        ]

    def sample_random(
        self, count: int = 10, seed: Optional[int] = None
    ) -> List[VoiceLine]:
        """Get random sample of lines"""
        if seed is not None:
            random.seed(seed)
        return random.sample(self.lines, min(count, len(self.lines)))

    def sample_balanced(self, count: int = 100) -> List[VoiceLine]:
        """Get balanced sample across all categories"""
        per_category = max(1, count // len(self.categories))
        samples = []

        for category in self.categories:
            category_lines = self.filter_by_category(category)
            if category_lines:
                sample_size = min(per_category, len(category_lines))
                samples.extend(random.sample(category_lines, sample_size))

        # Shuffle the result
        random.shuffle(samples)
        return samples[:count]

    def get_training_session(
        self,
        session_size: int = 50,
        categories: Optional[List[str]] = None,
        difficulty: str = "mixed",  # easy, medium, hard, mixed
    ) -> List[VoiceLine]:
        """
        Get a curated training session

        Args:
            session_size: Number of lines to return
            categories: Specific categories to focus on (None = all)
            difficulty: Session difficulty level
        """
        # Filter by categories if specified
        if categories:
            pool = self.filter_by_categories(categories)
        else:
            pool = self.lines

        # Filter by difficulty
        if difficulty == "easy":
            # Shorter, simpler lines (everyday, agreements, denials)
            easy_cats = ["everyday", "agreements", "denials_corrections", "greetings"]
            pool = [
                line
                for line in pool
                if line.category in easy_cats or len(line.text) < 50
            ]
        elif difficulty == "medium":
            # Standard complexity
            pool = [line for line in pool if 50 <= len(line.text) <= 150]
        elif difficulty == "hard":
            # Complex lines (technical, tongue twisters, long passages)
            hard_cats = [
                "technical",
                "tongue_twister",
                "paragraph",
                "stream_of_consciousness",
                "multilingual",
            ]
            pool = [
                line
                for line in pool
                if line.category in hard_cats or len(line.text) > 150
            ]
        # mixed = use full pool

        # Sample from pool
        if not pool:
            raise ValueError(f"No lines available for difficulty: {difficulty}")

        sample_size = min(session_size, len(pool))
        return random.sample(pool, sample_size)

    def export_training_batch(
        self,
        output_file: str,
        lines: List[VoiceLine],
        format: str = "txt",
    ):
        """
        Export lines to file for recording

        Args:
            output_file: Output file path
            lines: Lines to export
            format: Output format (txt, csv, json)
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, line in enumerate(lines, 1):
                    f.write(f"{i:04d} | {line.id} | [{line.category}] {line.text}\n")

        elif format == "csv":
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(lines[0].to_dict().keys()))
                writer.writeheader()
                for line in lines:
                    writer.writerow(line.to_dict())

        elif format == "json":
            import json

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([line.to_dict() for line in lines], f, indent=2)

        print(f"✅ Exported {len(lines)} lines to: {output_path}")

    def get_progressive_sessions(
        self, total_size: int = 200, num_sessions: int = 4
    ) -> List[Tuple[str, List[VoiceLine]]]:
        """
        Get progressive training sessions that increase in difficulty

        Returns:
            List of (session_name, lines) tuples
        """
        session_size = total_size // num_sessions
        sessions = [
            (
                "01_easy_warmup",
                self.get_training_session(session_size, difficulty="easy"),
            ),
            (
                "02_medium_practice",
                self.get_training_session(session_size, difficulty="medium"),
            ),
            (
                "03_hard_challenge",
                self.get_training_session(session_size, difficulty="hard"),
            ),
            (
                "04_mixed_variety",
                self.get_training_session(session_size, difficulty="mixed"),
            ),
        ]
        return sessions

    def search_text(self, query: str, case_sensitive: bool = False) -> List[VoiceLine]:
        """Search for lines containing specific text"""
        if not case_sensitive:
            query = query.lower()
            return [line for line in self.lines if query in line.text.lower()]
        return [line for line in self.lines if query in line.text]

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        category_counts = self.get_category_stats()
        emotion_counts = Counter(line.emotion for line in self.lines)
        text_lengths = [len(line.text) for line in self.lines]

        return {
            "total_lines": len(self.lines),
            "total_categories": len(self.categories),
            "categories": category_counts,
            "top_categories": dict(Counter(category_counts).most_common(10)),
            "emotions": dict(emotion_counts),
            "text_length": {
                "min": min(text_lengths),
                "max": max(text_lengths),
                "mean": sum(text_lengths) / len(text_lengths),
            },
        }


def main():
    """CLI for voice line management"""
    import argparse

    parser = argparse.ArgumentParser(
        description="VEGA Voice Line Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")

    # Categories command
    cat_parser = subparsers.add_parser("categories", help="List all categories")

    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Get random sample")
    sample_parser.add_argument(
        "--count", "-n", type=int, default=10, help="Sample size"
    )
    sample_parser.add_argument("--category", "-c", help="Filter by category")
    sample_parser.add_argument(
        "--balanced", "-b", action="store_true", help="Balanced sample"
    )

    # Session command
    session_parser = subparsers.add_parser("session", help="Create training session")
    session_parser.add_argument(
        "--size", "-s", type=int, default=50, help="Session size"
    )
    session_parser.add_argument(
        "--difficulty",
        "-d",
        choices=["easy", "medium", "hard", "mixed"],
        default="mixed",
    )
    session_parser.add_argument("--output", "-o", help="Export to file")

    # Progressive command
    prog_parser = subparsers.add_parser(
        "progressive", help="Create progressive sessions"
    )
    prog_parser.add_argument("--total", "-t", type=int, default=200, help="Total lines")
    prog_parser.add_argument(
        "--sessions", "-s", type=int, default=4, help="Number of sessions"
    )
    prog_parser.add_argument("--output-dir", "-o", default="data/training_sessions")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for text")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max", "-m", type=int, default=20, help="Max results")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize manager
    manager = VoiceLineManager()

    if args.command == "stats":
        stats = manager.get_stats()
        print(f"\n📊 Dataset Statistics")
        print(f"   Total lines: {stats['total_lines']:,}")
        print(f"   Categories: {stats['total_categories']}")
        print(
            f"   Text length: {stats['text_length']['min']}-{stats['text_length']['max']} chars (avg: {stats['text_length']['mean']:.0f})"
        )
        print(f"\n   Top 10 categories:")
        for cat, count in stats["top_categories"].items():
            print(f"     • {cat}: {count}")
        print(f"\n   Emotions:")
        for emotion, count in stats["emotions"].items():
            print(f"     • {emotion}: {count}")

    elif args.command == "categories":
        categories = manager.get_categories()
        stats = manager.get_category_stats()
        print(f"\n📁 Categories ({len(categories)}):\n")
        for cat in categories:
            print(f"  • {cat:30s} ({stats[cat]:3d} lines)")

    elif args.command == "sample":
        if args.balanced:
            lines = manager.sample_balanced(args.count)
        elif args.category:
            all_lines = manager.filter_by_category(args.category)
            lines = random.sample(all_lines, min(args.count, len(all_lines)))
        else:
            lines = manager.sample_random(args.count)

        print(f"\n🎲 Sample ({len(lines)} lines):\n")
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}. [{line.category:20s}] {line.text}")

    elif args.command == "session":
        lines = manager.get_training_session(args.size, difficulty=args.difficulty)
        print(f"\n🎯 Training Session ({args.difficulty}, {len(lines)} lines):\n")

        if args.output:
            manager.export_training_batch(args.output, lines, format="txt")
        else:
            for i, line in enumerate(lines, 1):
                print(f"{i:3d}. [{line.category:20s}] {line.text[:80]}")

    elif args.command == "progressive":
        sessions = manager.get_progressive_sessions(args.total, args.sessions)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"\n📚 Progressive Training Sessions ({args.sessions} sessions, {args.total} total lines):\n"
        )

        for session_name, lines in sessions:
            output_file = output_dir / f"{session_name}.txt"
            manager.export_training_batch(str(output_file), lines, format="txt")
            print(f"   ✅ {session_name}: {len(lines)} lines → {output_file}")

    elif args.command == "search":
        lines = manager.search_text(args.query)
        print(f"\n🔍 Search results for '{args.query}' ({len(lines)} matches):\n")
        for i, line in enumerate(lines[: args.max], 1):
            print(f"{i:3d}. [{line.category:20s}] {line.text}")
        if len(lines) > args.max:
            print(f"\n   ... and {len(lines) - args.max} more")


if __name__ == "__main__":
    main()
