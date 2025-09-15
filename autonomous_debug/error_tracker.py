#!/usr/bin/env python3
"""
Autonomous Error Tracker
========================

Scans all log files recursively, detects exceptions and tracebacks,
parses error details, and indexes them into SQLite database with
deduplication and frequency tracking.

Features:
- Multi-line traceback parsing
- Error fingerprinting and deduplication
- Frequency tracking and flapping detection
- Contextual code snippet extraction
- Real-time log watching capabilities
"""

import os
import re
import sys
import sqlite3
import hashlib
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ErrorRecord:
    """Data structure for parsed error information"""

    id: str
    timestamp: datetime
    file_path: str
    line_number: int
    error_type: str
    message: str
    traceback_hash: str
    frequency: int
    snippet: str
    first_seen: datetime
    last_seen: datetime
    severity: str = "medium"
    resolved: bool = False
    resolution_attempts: int = 0
    full_traceback: str = ""
    context_data: Dict[str, Any] = None

    def __post_init__(self):
        if self.context_data is None:
            self.context_data = {}


class ErrorDatabase:
    """SQLite database for error tracking and management"""

    def __init__(self, db_path: str = "autonomous_debug/error_index.db"):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize the error tracking database"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # Enable dict-like access

            # Create tables
            self._create_tables()
            logger.info(f"Error database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize error database: {e}")
            raise

    def _create_tables(self):
        """Create database tables for error tracking"""
        cursor = self.conn.cursor()

        # Main errors table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS errors (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                line_number INTEGER,
                error_type TEXT NOT NULL,
                message TEXT NOT NULL,
                traceback_hash TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                snippet TEXT,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                severity TEXT DEFAULT 'medium',
                resolved BOOLEAN DEFAULT FALSE,
                resolution_attempts INTEGER DEFAULT 0,
                full_traceback TEXT,
                context_data TEXT
            )
        """
        )

        # Error patterns table for learning
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE,
                pattern_description TEXT,
                common_causes TEXT,
                resolution_strategies TEXT,
                success_rate REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Fix history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fix_history (
                id TEXT PRIMARY KEY,
                error_id TEXT,
                fix_type TEXT,
                fix_description TEXT,
                patch_path TEXT,
                success BOOLEAN,
                confidence_score REAL,
                applied_at TEXT,
                rollback_at TEXT,
                FOREIGN KEY (error_id) REFERENCES errors (id)
            )
        """
        )

        # Error flapping detection
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS error_flapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                traceback_hash TEXT,
                occurrence_count INTEGER,
                time_window_start TEXT,
                time_window_end TEXT,
                flagged_for_priority BOOLEAN DEFAULT FALSE
            )
        """
        )

        # Create indices for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_traceback_hash ON errors (traceback_hash)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON errors (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_path ON errors (file_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resolved ON errors (resolved)")

        self.conn.commit()

    def insert_error(self, error: ErrorRecord) -> bool:
        """Insert or update an error record"""
        try:
            cursor = self.conn.cursor()

            # Check if error already exists
            cursor.execute(
                "SELECT frequency, first_seen FROM errors WHERE traceback_hash = ?",
                (error.traceback_hash,),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing error
                new_frequency = existing["frequency"] + 1
                cursor.execute(
                    """
                    UPDATE errors 
                    SET frequency = ?, last_seen = ?, timestamp = ?
                    WHERE traceback_hash = ?
                """,
                    (
                        new_frequency,
                        error.last_seen.isoformat(),
                        error.timestamp.isoformat(),
                        error.traceback_hash,
                    ),
                )

                # Check for flapping
                self._check_error_flapping(error.traceback_hash, new_frequency)

            else:
                # Insert new error
                cursor.execute(
                    """
                    INSERT INTO errors (
                        id, timestamp, file_path, line_number, error_type, message,
                        traceback_hash, frequency, snippet, first_seen, last_seen,
                        severity, resolved, resolution_attempts, full_traceback, context_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        error.id,
                        error.timestamp.isoformat(),
                        error.file_path,
                        error.line_number,
                        error.error_type,
                        error.message,
                        error.traceback_hash,
                        error.frequency,
                        error.snippet,
                        error.first_seen.isoformat(),
                        error.last_seen.isoformat(),
                        error.severity,
                        error.resolved,
                        error.resolution_attempts,
                        error.full_traceback,
                        json.dumps(error.context_data),
                    ),
                )

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to insert error: {e}")
            return False

    def _check_error_flapping(self, traceback_hash: str, frequency: int):
        """Check if error is flapping (occurring too frequently)"""
        try:
            cursor = self.conn.cursor()

            # Get recent occurrences in last 10 minutes
            ten_minutes_ago = (datetime.now() - timedelta(minutes=10)).isoformat()
            cursor.execute(
                """
                SELECT COUNT(*) as recent_count 
                FROM errors 
                WHERE traceback_hash = ? AND last_seen > ?
            """,
                (traceback_hash, ten_minutes_ago),
            )

            recent_count = cursor.fetchone()["recent_count"]

            if recent_count >= 3:  # Flapping threshold
                # Flag for high priority
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO error_flapping 
                    (traceback_hash, occurrence_count, time_window_start, time_window_end, flagged_for_priority)
                    VALUES (?, ?, ?, ?, TRUE)
                """,
                    (
                        traceback_hash,
                        recent_count,
                        ten_minutes_ago,
                        datetime.now().isoformat(),
                    ),
                )

                self.conn.commit()
                logger.warning(
                    f"Error flapping detected: {traceback_hash} ({recent_count} times in 10 minutes)"
                )

        except Exception as e:
            logger.error(f"Failed to check error flapping: {e}")

    def get_unresolved_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get unresolved errors ordered by priority"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT e.*, f.flagged_for_priority
                FROM errors e
                LEFT JOIN error_flapping f ON e.traceback_hash = f.traceback_hash
                WHERE e.resolved = FALSE
                ORDER BY 
                    f.flagged_for_priority DESC,
                    e.frequency DESC,
                    e.last_seen DESC
                LIMIT ?
            """,
                (limit,),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get unresolved errors: {e}")
            return []

    def mark_error_resolved(self, error_id: str, fix_id: str = None):
        """Mark an error as resolved"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE errors SET resolved = TRUE WHERE id = ?", (error_id,)
            )
            self.conn.commit()
            logger.info(f"Error {error_id} marked as resolved")

        except Exception as e:
            logger.error(f"Failed to mark error as resolved: {e}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for reporting"""
        try:
            cursor = self.conn.cursor()

            # Total errors
            cursor.execute("SELECT COUNT(*) as total FROM errors")
            total = cursor.fetchone()["total"]

            # Unresolved errors
            cursor.execute(
                "SELECT COUNT(*) as unresolved FROM errors WHERE resolved = FALSE"
            )
            unresolved = cursor.fetchone()["unresolved"]

            # Flapping errors
            cursor.execute(
                "SELECT COUNT(*) as flapping FROM error_flapping WHERE flagged_for_priority = TRUE"
            )
            flapping = cursor.fetchone()["flapping"]

            # Most common error types
            cursor.execute(
                """
                SELECT error_type, COUNT(*) as count 
                FROM errors 
                GROUP BY error_type 
                ORDER BY count DESC 
                LIMIT 5
            """
            )
            common_types = [dict(row) for row in cursor.fetchall()]

            return {
                "total_errors": total,
                "unresolved_errors": unresolved,
                "flapping_errors": flapping,
                "most_common_types": common_types,
                "resolution_rate": (
                    (total - unresolved) / total * 100 if total > 0 else 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get error statistics: {e}")
            return {}

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


class ErrorParser:
    """Parses log files and extracts error information"""

    # Regex patterns for different error formats
    ERROR_PATTERNS = {
        "python_traceback": re.compile(
            r"Traceback \(most recent call last\):(.*?)(?=\n\w|\n$|\nTraceback)",
            re.DOTALL | re.MULTILINE,
        ),
        "exception_line": re.compile(r"^(.+?Exception|.+?Error): (.+)$", re.MULTILINE),
        "error_line": re.compile(r"(ERROR|CRITICAL|FATAL).*?:(.+)", re.IGNORECASE),
        "warning_line": re.compile(r"(WARNING|WARN).*?:(.+)", re.IGNORECASE),
    }

    def __init__(self):
        self.file_cache = {}  # Cache for file contents

    def parse_log_file(self, file_path: str) -> List[ErrorRecord]:
        """Parse a single log file and extract errors"""
        errors = []

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Parse Python tracebacks
            errors.extend(self._parse_python_tracebacks(content, file_path))

            # Parse error lines
            errors.extend(self._parse_error_lines(content, file_path))

            logger.debug(f"Parsed {len(errors)} errors from {file_path}")

        except Exception as e:
            logger.error(f"Failed to parse log file {file_path}: {e}")

        return errors

    def _parse_python_tracebacks(
        self, content: str, log_path: str
    ) -> List[ErrorRecord]:
        """Parse Python traceback errors"""
        errors = []

        for match in self.ERROR_PATTERNS["python_traceback"].finditer(content):
            try:
                traceback_text = match.group(0)

                # Extract error type and message from last line
                lines = traceback_text.strip().split("\n")
                last_line = lines[-1] if lines else ""

                error_match = self.ERROR_PATTERNS["exception_line"].search(last_line)
                if error_match:
                    error_type = error_match.group(1)
                    message = error_match.group(2).strip()
                else:
                    error_type = "UnknownError"
                    message = last_line.strip()

                # Extract file and line number from traceback
                file_path, line_number = self._extract_file_line_from_traceback(
                    traceback_text
                )

                # Generate unique ID and hash
                error_id = self._generate_error_id(
                    traceback_text, file_path, line_number
                )
                traceback_hash = self._generate_traceback_hash(traceback_text)

                # Extract code snippet
                snippet = self._extract_code_snippet(file_path, line_number)

                # Create error record
                now = datetime.now()
                error = ErrorRecord(
                    id=error_id,
                    timestamp=now,
                    file_path=file_path or log_path,
                    line_number=line_number or 0,
                    error_type=error_type,
                    message=message,
                    traceback_hash=traceback_hash,
                    frequency=1,
                    snippet=snippet,
                    first_seen=now,
                    last_seen=now,
                    severity=self._assess_severity(error_type, message),
                    full_traceback=traceback_text,
                    context_data={
                        "log_source": log_path,
                        "traceback_lines": len(lines),
                    },
                )

                errors.append(error)

            except Exception as e:
                logger.error(f"Failed to parse traceback: {e}")

        return errors

    def _parse_error_lines(self, content: str, log_path: str) -> List[ErrorRecord]:
        """Parse individual error lines"""
        errors = []

        for pattern_name, pattern in [
            ("error_line", self.ERROR_PATTERNS["error_line"])
        ]:
            for match in pattern.finditer(content):
                try:
                    error_type = match.group(1).upper()
                    message = match.group(2).strip()

                    # Generate unique ID and hash
                    error_text = match.group(0)
                    error_id = self._generate_error_id(error_text, log_path, 0)
                    traceback_hash = self._generate_traceback_hash(error_text)

                    # Create error record
                    now = datetime.now()
                    error = ErrorRecord(
                        id=error_id,
                        timestamp=now,
                        file_path=log_path,
                        line_number=0,
                        error_type=error_type,
                        message=message,
                        traceback_hash=traceback_hash,
                        frequency=1,
                        snippet=error_text,
                        first_seen=now,
                        last_seen=now,
                        severity=self._assess_severity(error_type, message),
                        full_traceback=error_text,
                        context_data={
                            "log_source": log_path,
                            "pattern_type": pattern_name,
                        },
                    )

                    errors.append(error)

                except Exception as e:
                    logger.error(f"Failed to parse error line: {e}")

        return errors

    def _extract_file_line_from_traceback(
        self, traceback_text: str
    ) -> Tuple[Optional[str], Optional[int]]:
        """Extract file path and line number from traceback"""
        try:
            # Look for file paths in traceback
            file_pattern = re.compile(r'File "(.+?)", line (\d+)')
            matches = file_pattern.findall(traceback_text)

            if matches:
                # Get the last (most relevant) file and line
                file_path, line_str = matches[-1]
                return file_path, int(line_str)

        except Exception as e:
            logger.error(f"Failed to extract file/line from traceback: {e}")

        return None, None

    def _extract_code_snippet(
        self, file_path: str, line_number: int, context_lines: int = 5
    ) -> str:
        """Extract code snippet around the error line"""
        if not file_path or not line_number or not os.path.exists(file_path):
            return ""

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()

            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)

            snippet_lines = []
            for i in range(start, end):
                line_num = i + 1
                prefix = ">>> " if line_num == line_number else "    "
                snippet_lines.append(f"{prefix}{line_num:4d}: {lines[i].rstrip()}")

            return "\n".join(snippet_lines)

        except Exception as e:
            logger.error(f"Failed to extract code snippet: {e}")
            return ""

    def _generate_error_id(
        self, error_text: str, file_path: str, line_number: int
    ) -> str:
        """Generate unique error ID"""
        content = f"{error_text}{file_path}{line_number}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_traceback_hash(self, traceback_text: str) -> str:
        """Generate hash for traceback deduplication"""
        # Normalize traceback for consistent hashing
        normalized = re.sub(r"line \d+", "line XXX", traceback_text)
        normalized = re.sub(r'File ".*?"', 'File "XXX"', normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _assess_severity(self, error_type: str, message: str) -> str:
        """Assess error severity based on type and message"""
        critical_patterns = [
            "segmentation fault",
            "memory error",
            "system error",
            "critical",
            "fatal",
            "abort",
            "crash",
        ]

        high_patterns = [
            "exception",
            "error",
            "failed",
            "cannot",
            "unable",
            "timeout",
            "connection",
            "permission",
        ]

        text = f"{error_type} {message}".lower()

        for pattern in critical_patterns:
            if pattern in text:
                return "critical"

        for pattern in high_patterns:
            if pattern in text:
                return "high"

        return "medium"


class LogScanner:
    """Scans directories for log files and parses errors"""

    LOG_EXTENSIONS = {".log", ".txt", ".out", ".err"}
    LOG_PATTERNS = ["*.log", "*.txt", "*.out", "*.err"]

    def __init__(self, db: ErrorDatabase):
        self.db = db
        self.parser = ErrorParser()

    def scan_directory(self, directory: str, recursive: bool = True) -> int:
        """Scan directory for log files and parse errors"""
        error_count = 0

        try:
            directory_path = Path(directory)

            if not directory_path.exists():
                logger.warning(f"Directory does not exist: {directory}")
                return 0

            # Find log files
            if recursive:
                log_files = []
                for ext in self.LOG_EXTENSIONS:
                    log_files.extend(directory_path.rglob(f"*{ext}"))
            else:
                log_files = []
                for ext in self.LOG_EXTENSIONS:
                    log_files.extend(directory_path.glob(f"*{ext}"))

            logger.info(f"Found {len(log_files)} log files in {directory}")

            # Parse each log file
            for log_file in log_files:
                try:
                    errors = self.parser.parse_log_file(str(log_file))

                    # Ensure we got a list of ErrorRecord objects
                    if not isinstance(errors, list):
                        logger.error(
                            f"parse_log_file returned {type(errors)} instead of list for {log_file}"
                        )
                        continue

                    for error in errors:
                        # Validate that error is an ErrorRecord object
                        if not hasattr(error, "insert_error") and hasattr(
                            error, "traceback_hash"
                        ):
                            try:
                                if self.db.insert_error(error):
                                    error_count += 1
                            except Exception as insert_err:
                                logger.error(
                                    f"Failed to insert error {getattr(error, 'id', 'unknown')}: {insert_err}"
                                )
                        else:
                            logger.error(
                                f"Invalid error object type: {type(error)} from {log_file}"
                            )

                    logger.debug(f"Processed {len(errors)} errors from {log_file}")

                except Exception as e:
                    logger.error(f"Failed to process log file {log_file}: {e}")

            logger.info(f"Total errors processed: {error_count}")

        except Exception as e:
            logger.error(f"Failed to scan directory {directory}: {e}")

        return error_count


class ErrorLogbook:
    """Maintains rolling error logbook in JSONL format"""

    def __init__(
        self,
        logbook_path: str = "autonomous_debug/error_logbook.jsonl",
        max_entries: int = 1000,
    ):
        self.logbook_path = logbook_path
        self.max_entries = max_entries
        os.makedirs(os.path.dirname(self.logbook_path), exist_ok=True)

    def append_error(self, error: ErrorRecord):
        """Append error to rolling logbook"""
        try:
            # Append new error
            with open(self.logbook_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(error), default=str) + "\n")

            # Maintain rolling buffer
            self._maintain_rolling_buffer()

        except Exception as e:
            logger.error(f"Failed to append to error logbook: {e}")

    def _maintain_rolling_buffer(self):
        """Keep only the most recent max_entries in logbook"""
        try:
            if not os.path.exists(self.logbook_path):
                return

            with open(self.logbook_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) > self.max_entries:
                # Keep only the most recent entries
                recent_lines = lines[-self.max_entries :]

                with open(self.logbook_path, "w", encoding="utf-8") as f:
                    f.writelines(recent_lines)

                logger.debug(f"Trimmed logbook to {len(recent_lines)} entries")

        except Exception as e:
            logger.error(f"Failed to maintain rolling buffer: {e}")


def main():
    """Main function for error tracking"""
    import argparse

    parser = argparse.ArgumentParser(description="Autonomous Error Tracker")
    parser.add_argument(
        "--log-dir",
        default="/home/ncacord/Vega2.0",
        help="Directory to scan for log files",
    )
    parser.add_argument(
        "--recursive", action="store_true", default=True, help="Scan recursively"
    )
    parser.add_argument("--stats", action="store_true", help="Show error statistics")
    parser.add_argument(
        "--unresolved", action="store_true", help="Show unresolved errors"
    )

    args = parser.parse_args()

    # Initialize components
    db = ErrorDatabase()
    scanner = LogScanner(db)
    logbook = ErrorLogbook()

    try:
        if args.stats:
            # Show statistics
            stats = db.get_error_statistics()
            print("\n🔍 Error Statistics:")
            print(f"  Total Errors: {stats.get('total_errors', 0)}")
            print(f"  Unresolved: {stats.get('unresolved_errors', 0)}")
            print(f"  Flapping: {stats.get('flapping_errors', 0)}")
            print(f"  Resolution Rate: {stats.get('resolution_rate', 0):.1f}%")

            print("\n📊 Most Common Error Types:")
            for error_type in stats.get("most_common_types", []):
                print(f"  {error_type['error_type']}: {error_type['count']}")

        elif args.unresolved:
            # Show unresolved errors
            errors = db.get_unresolved_errors()
            print(f"\n🚨 Unresolved Errors ({len(errors)}):")

            for error in errors:
                print(f"\n  ID: {error['id'][:8]}...")
                print(f"  Type: {error['error_type']}")
                print(f"  File: {error['file_path']}:{error['line_number']}")
                print(f"  Message: {error['message'][:100]}...")
                print(f"  Frequency: {error['frequency']}")
                print(f"  Last Seen: {error['last_seen']}")

        else:
            # Scan for errors
            print(f"🔍 Scanning {args.log_dir} for errors...")
            error_count = scanner.scan_directory(args.log_dir, args.recursive)
            print(f"✅ Processed {error_count} errors")

            # Show summary
            stats = db.get_error_statistics()
            print(
                f"📊 Current Status: {stats.get('unresolved_errors', 0)} unresolved errors"
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()
