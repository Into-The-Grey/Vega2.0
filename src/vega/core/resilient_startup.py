#!/usr/bin/env python3
"""
Resilient Startup System for Vega2.0

This module implements a robust startup system that:
1. Separates core (critical) functions from non-critical features
2. Allows non-critical features to fail without compromising the core
3. Autonomously attempts to resolve failures in the background
4. Prioritizes healing based on feature usage frequency
5. Logs and notifies on persistent failures
6. Learns from successful resolutions for future use

Architecture:
- Core functions: Must succeed or system won't start (e.g., database, LLM backend)
- Non-critical functions: Can fail gracefully (e.g., profiler, analytics, integrations)
- Background healer: Low-priority async process that attempts to fix failures
- Knowledge base: Stores successful resolution strategies

Key Classes:
- ResolutionKnowledgeBase: Persistent storage for successful error resolutions
- FeatureUsageTracker: Tracks feature usage to prioritize healing
- BackgroundHealer: Async process that attempts to heal failed features
- ResilientStartupManager: Main manager orchestrating the startup sequence
- SimplifiedStartupManager: Simplified API for easier integration
"""

from __future__ import annotations

import asyncio
import functools
import logging
import traceback
import json
import hashlib
import shutil
import re
import difflib
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
from collections import defaultdict
import threading
import time

logger = logging.getLogger(__name__)


class FeaturePriority(Enum):
    """Priority levels for features - affects healing order"""

    CRITICAL = 1  # Core system - must work (DB, LLM, config)
    HIGH = 2  # Frequently used features
    MEDIUM = 3  # Regularly used features
    LOW = 4  # Rarely used features
    BACKGROUND = 5  # Nice-to-have features


class FeatureStatus(Enum):
    """Current status of a feature"""

    UNKNOWN = "unknown"
    LOADING = "loading"
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Partially working
    FAILED = "failed"  # Failed, awaiting healing
    HEALING = "healing"  # Currently attempting recovery
    PERMANENTLY_FAILED = "permanently_failed"  # Gave up after max attempts


class ResolutionStrategy(Enum):
    """Types of resolution strategies"""

    REIMPORT = "reimport"  # Try importing again
    REINSTALL_DEPS = "reinstall_deps"  # Reinstall dependencies
    RESET_STATE = "reset_state"  # Reset internal state
    RECREATE_RESOURCE = "recreate_resource"  # Recreate the resource
    FALLBACK_CONFIG = "fallback_config"  # Use fallback configuration
    RESTART_SERVICE = "restart_service"  # Restart dependent service
    CUSTOM = "custom"  # Custom resolution function


@dataclass
class FeatureDefinition:
    """Defines a feature that can be loaded during startup"""

    name: str
    loader: Callable[[], Coroutine[Any, Any, Any]]
    priority: FeaturePriority
    is_critical: bool = False
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    fallback: Optional[Callable[[], Coroutine[Any, Any, Any]]] = None
    resolution_strategies: List[ResolutionStrategy] = field(
        default_factory=lambda: [
            ResolutionStrategy.REIMPORT,
            ResolutionStrategy.RESET_STATE,
        ]
    )
    max_heal_attempts: int = 5
    heal_backoff_seconds: float = 30.0  # Time between heal attempts


@dataclass
class FeatureState:
    """Runtime state of a feature"""

    definition: FeatureDefinition
    status: FeatureStatus = FeatureStatus.UNKNOWN
    error: Optional[Exception] = None
    error_traceback: Optional[str] = None
    load_time: Optional[datetime] = None
    heal_attempts: int = 0
    last_heal_attempt: Optional[datetime] = None
    usage_count: int = 0  # How often this feature is used
    last_used: Optional[datetime] = None
    resolution_applied: Optional[str] = None

    # Enhanced tracking
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    average_load_time_ms: float = 0.0
    health_check_failures: int = 0


@dataclass
class ResolutionRecord:
    """
    Record of a successful resolution for the knowledge base.

    Stores detailed information about how an error was resolved,
    including the error pattern, strategy used, and success metrics.
    """

    # Required fields (no defaults) - must come first
    feature_name: str
    error_signature: str  # Hash of error type + message for exact matching
    error_type: str  # The exception type name
    error_message: str  # The error message (truncated)
    strategy_used: str

    # Optional fields with defaults
    error_keywords: List[str] = field(default_factory=list)  # Key terms for fuzzy matching
    resolution_details: Dict[str, Any] = field(default_factory=dict)

    # Success metrics
    success_count: int = 1
    failure_count: int = 0  # Times this resolution failed

    # Timing information
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    average_resolution_time_ms: float = 0.0

    # Context information
    python_version: str = ""
    platform: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of this resolution"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def is_stale(self) -> bool:
        """Check if this record is stale (not used in 30 days)"""
        age = datetime.now(timezone.utc) - self.last_used
        return age.days > 30

    @property
    def confidence_score(self) -> float:
        """
        Calculate a confidence score for this resolution.
        Higher scores indicate more reliable resolutions.
        """
        # Base score from success rate
        score = self.success_rate * 50

        # Bonus for number of successes (logarithmic)
        import math

        score += math.log1p(self.success_count) * 10

        # Penalty for staleness
        age_days = (datetime.now(timezone.utc) - self.last_used).days
        if age_days > 7:
            score *= max(0.5, 1.0 - (age_days - 7) * 0.02)

        # Bonus for recent use
        if age_days < 1:
            score *= 1.2

        return min(100.0, score)


class ResolutionKnowledgeBase:
    """
    Advanced knowledge base for storing and retrieving successful error resolutions.

    Features:
    - Exact and fuzzy error matching
    - Confidence scoring for resolutions
    - Automatic cleanup of stale/ineffective entries
    - Statistics and analytics
    - Backup and restore capabilities
    - Thread-safe operations

    The knowledge base persists to disk and learns from successful resolutions
    to improve future error recovery.
    """

    # Version for migration support
    SCHEMA_VERSION = 2

    # Configuration constants
    MAX_RECORDS_PER_FEATURE = 50  # Limit records to prevent bloat
    STALE_THRESHOLD_DAYS = 60  # Records older than this are candidates for cleanup
    MIN_SUCCESS_RATE = 0.3  # Records below this success rate get pruned
    FUZZY_MATCH_THRESHOLD = 0.6  # Minimum similarity for fuzzy matching

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        backup_count: int = 5,
        auto_cleanup: bool = True,
    ):
        """
        Initialize the resolution knowledge base.

        Args:
            storage_path: Path to store the knowledge base JSON file
            backup_count: Number of backups to maintain
            auto_cleanup: Whether to automatically cleanup stale entries
        """
        self.storage_path = storage_path or Path("data/resolution_knowledge.json")
        self.backup_count = backup_count
        self.auto_cleanup = auto_cleanup

        self.resolutions: Dict[str, List[ResolutionRecord]] = defaultdict(list)
        self._lock = threading.RLock()  # Reentrant lock for nested calls
        self._dirty = False  # Track if changes need saving
        self._save_timer: Optional[threading.Timer] = None

        # Statistics
        self._stats = {
            "total_lookups": 0,
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "cache_hits": 0,
            "resolutions_recorded": 0,
            "failures_recorded": 0,
        }

        # Simple LRU cache for recent lookups
        self._lookup_cache: Dict[str, Tuple[Optional[ResolutionRecord], float]] = {}
        self._cache_max_size = 100
        self._cache_ttl_seconds = 300

        # Load from disk
        self._load()

        # Run initial cleanup if enabled
        if self.auto_cleanup:
            self._cleanup_stale_records()

    def _load(self):
        """Load knowledge base from disk with migration support"""
        try:
            if not self.storage_path.exists():
                logger.info("No existing knowledge base found, starting fresh")
                return

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Check schema version and migrate if needed
            schema_version = data.get("_schema_version", 1)
            if schema_version < self.SCHEMA_VERSION:
                data = self._migrate_schema(data, schema_version)

            # Load records
            records_data = data.get("resolutions", data)  # Support old format
            if isinstance(records_data, dict) and "_schema_version" not in records_data:
                # Old format: direct feature -> records mapping
                records_data = {k: v for k, v in records_data.items() if not k.startswith("_")}
            elif isinstance(records_data, dict):
                records_data = records_data.get("resolutions", {})

            for feature_name, records in records_data.items():
                if feature_name.startswith("_"):
                    continue

                self.resolutions[feature_name] = []
                for r in records:
                    try:
                        record = self._parse_record(r)
                        self.resolutions[feature_name].append(record)
                    except Exception as e:
                        logger.warning(f"Could not parse record for {feature_name}: {e}")

            total_records = sum(len(v) for v in self.resolutions.values())
            logger.info(f"📚 Loaded {total_records} resolution records from knowledge base")

            # Load statistics if present
            if "_stats" in data:
                self._stats.update(data["_stats"])

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base file: {e}")
            self._restore_from_backup()
        except Exception as e:
            logger.warning(f"Could not load resolution knowledge base: {e}")

    def _parse_record(self, r: Dict[str, Any]) -> ResolutionRecord:
        """Parse a record dictionary into a ResolutionRecord object"""

        # Parse datetime fields with fallback
        def parse_datetime(value, default=None):
            if value is None:
                return default or datetime.now(timezone.utc)
            if isinstance(value, datetime):
                return value
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                return default or datetime.now(timezone.utc)

        return ResolutionRecord(
            feature_name=r.get("feature_name", "unknown"),
            error_signature=r.get("error_signature", ""),
            error_type=r.get("error_type", "Exception"),
            error_message=r.get("error_message", ""),
            error_keywords=r.get("error_keywords", []),
            strategy_used=r.get("strategy_used", "unknown"),
            resolution_details=r.get("resolution_details", {}),
            success_count=r.get("success_count", 1),
            failure_count=r.get("failure_count", 0),
            last_used=parse_datetime(r.get("last_used")),
            created_at=parse_datetime(r.get("created_at")),
            average_resolution_time_ms=r.get("average_resolution_time_ms", 0.0),
            python_version=r.get("python_version", ""),
            platform=r.get("platform", ""),
        )

    def _migrate_schema(self, data: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """Migrate data from older schema versions"""
        logger.info(f"Migrating knowledge base from v{from_version} to v{self.SCHEMA_VERSION}")

        if from_version == 1:
            # Migration from v1 to v2: Add new fields
            new_data = {
                "_schema_version": self.SCHEMA_VERSION,
                "_stats": {},
                "resolutions": {},
            }

            for feature_name, records in data.items():
                if feature_name.startswith("_"):
                    continue

                new_data["resolutions"][feature_name] = []
                for r in records:
                    # Add new fields with defaults
                    r["error_type"] = r.get("error_type", "Exception")
                    r["error_message"] = r.get("error_message", "")
                    r["error_keywords"] = r.get("error_keywords", [])
                    r["failure_count"] = r.get("failure_count", 0)
                    r["average_resolution_time_ms"] = r.get("average_resolution_time_ms", 0.0)
                    r["python_version"] = r.get("python_version", "")
                    r["platform"] = r.get("platform", "")
                    new_data["resolutions"][feature_name].append(r)

            return new_data

        return data

    def _save(self, force: bool = False):
        """
        Save knowledge base to disk with atomic write.

        Uses debounced saving to avoid excessive disk I/O.
        """
        if not force and not self._dirty:
            return

        # Cancel any pending save timer
        if self._save_timer:
            self._save_timer.cancel()
            self._save_timer = None

        if not force:
            # Debounce: schedule save for 5 seconds from now
            self._dirty = True
            self._save_timer = threading.Timer(5.0, lambda: self._save(force=True))
            self._save_timer.daemon = True
            self._save_timer.start()
            return

        with self._lock:
            try:
                # Create backup before saving
                self._create_backup()

                # Prepare data
                data = {
                    "_schema_version": self.SCHEMA_VERSION,
                    "_stats": self._stats,
                    "_last_saved": datetime.now(timezone.utc).isoformat(),
                    "resolutions": {},
                }

                for feature_name, records in self.resolutions.items():
                    data["resolutions"][feature_name] = [
                        {
                            "feature_name": r.feature_name,
                            "error_signature": r.error_signature,
                            "error_type": r.error_type,
                            "error_message": r.error_message,
                            "error_keywords": r.error_keywords,
                            "strategy_used": r.strategy_used,
                            "resolution_details": r.resolution_details,
                            "success_count": r.success_count,
                            "failure_count": r.failure_count,
                            "last_used": r.last_used.isoformat(),
                            "created_at": r.created_at.isoformat(),
                            "average_resolution_time_ms": r.average_resolution_time_ms,
                            "python_version": r.python_version,
                            "platform": r.platform,
                        }
                        for r in records
                    ]

                # Atomic write: write to temp file then rename
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.storage_path.with_suffix(".json.tmp")

                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)

                # Atomic rename
                temp_path.replace(self.storage_path)

                self._dirty = False
                logger.debug(f"Saved knowledge base with {sum(len(v) for v in self.resolutions.values())} records")

            except Exception as e:
                logger.error(f"Could not save resolution knowledge base: {e}")

    def _create_backup(self):
        """Create a backup of the current knowledge base file"""
        if not self.storage_path.exists():
            return

        try:
            backup_dir = self.storage_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"resolution_knowledge_{timestamp}.json"

            shutil.copy2(self.storage_path, backup_path)

            # Cleanup old backups
            backups = sorted(backup_dir.glob("resolution_knowledge_*.json"))
            while len(backups) > self.backup_count:
                oldest = backups.pop(0)
                oldest.unlink()

        except Exception as e:
            logger.warning(f"Could not create backup: {e}")

    def _restore_from_backup(self):
        """Restore from the most recent backup"""
        backup_dir = self.storage_path.parent / "backups"
        if not backup_dir.exists():
            return False

        backups = sorted(backup_dir.glob("resolution_knowledge_*.json"), reverse=True)

        for backup in backups:
            try:
                with open(backup, "r") as f:
                    json.load(f)  # Validate JSON
                shutil.copy2(backup, self.storage_path)
                logger.info(f"Restored knowledge base from backup: {backup.name}")
                self._load()
                return True
            except Exception:
                continue

        return False

    def _cleanup_stale_records(self):
        """Remove stale and ineffective records"""
        with self._lock:
            cleaned = 0

            for feature_name in list(self.resolutions.keys()):
                records = self.resolutions[feature_name]

                # Filter out stale records with low success rate
                filtered = [r for r in records if not (r.is_stale and r.success_rate < self.MIN_SUCCESS_RATE)]

                # Keep only top records if over limit
                if len(filtered) > self.MAX_RECORDS_PER_FEATURE:
                    filtered.sort(key=lambda r: r.confidence_score, reverse=True)
                    filtered = filtered[: self.MAX_RECORDS_PER_FEATURE]

                cleaned += len(records) - len(filtered)
                self.resolutions[feature_name] = filtered

                # Remove empty feature entries
                if not self.resolutions[feature_name]:
                    del self.resolutions[feature_name]

            if cleaned > 0:
                logger.info(f"🧹 Cleaned up {cleaned} stale resolution records")
                self._dirty = True
                self._save()

    @staticmethod
    def _error_signature(error: Exception) -> str:
        """Create a unique signature for exact error matching"""
        error_str = f"{type(error).__name__}:{str(error)[:200]}"
        return hashlib.sha256(error_str.encode()).hexdigest()[:32]

    @staticmethod
    def _extract_error_keywords(error: Exception) -> List[str]:
        """Extract keywords from an error for fuzzy matching"""
        error_str = str(error).lower()

        # Common error patterns to extract
        keywords = set()

        # Extract module/package names
        module_pattern = r"module['\"]?\s*([a-zA-Z_][a-zA-Z0-9_.]*)"
        for match in re.findall(module_pattern, error_str):
            keywords.add(match)

        # Extract file paths
        path_pattern = r"['\"]?([/\\][a-zA-Z0-9_./\\-]+\.(py|json|yaml|yml))['\"]?"
        for match in re.findall(path_pattern, error_str):
            keywords.add(match[0].split("/")[-1])  # Just filename

        # Extract class/function names
        name_pattern = r"['\"]([A-Z][a-zA-Z0-9_]+)['\"]"
        for match in re.findall(name_pattern, str(error)):
            keywords.add(match.lower())

        # Add error type
        keywords.add(type(error).__name__.lower())

        # Add common error words
        error_words = [
            "timeout",
            "connection",
            "permission",
            "not found",
            "missing",
            "invalid",
            "failed",
            "error",
            "exception",
            "refused",
            "denied",
        ]
        for word in error_words:
            if word in error_str:
                keywords.add(word)

        return list(keywords)[:20]  # Limit keywords

    def _calculate_similarity(self, error: Exception, record: ResolutionRecord) -> float:
        """Calculate similarity between an error and a stored record"""
        # Exact signature match is 100%
        if self._error_signature(error) == record.error_signature:
            return 1.0

        # Type match bonus
        score = 0.0
        if type(error).__name__ == record.error_type:
            score += 0.3

        # Keyword overlap
        error_keywords = set(self._extract_error_keywords(error))
        record_keywords = set(record.error_keywords)

        if error_keywords and record_keywords:
            overlap = len(error_keywords & record_keywords)
            total = len(error_keywords | record_keywords)
            score += 0.4 * (overlap / total if total > 0 else 0)

        # Message similarity (fuzzy string matching)
        error_msg = str(error)[:200].lower()
        record_msg = record.error_message.lower()

        if error_msg and record_msg:
            ratio = difflib.SequenceMatcher(None, error_msg, record_msg).ratio()
            score += 0.3 * ratio

        return score

    def find_resolution(
        self,
        feature_name: str,
        error: Exception,
        allow_fuzzy: bool = True,
    ) -> Optional[ResolutionRecord]:
        """
        Find a known resolution for this error.

        Args:
            feature_name: Name of the feature that failed
            error: The exception that occurred
            allow_fuzzy: Whether to use fuzzy matching if exact match not found

        Returns:
            ResolutionRecord if found, None otherwise
        """
        self._stats["total_lookups"] += 1

        # Check cache first
        cache_key = f"{feature_name}:{self._error_signature(error)}"
        if cache_key in self._lookup_cache:
            record, timestamp = self._lookup_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl_seconds:
                self._stats["cache_hits"] += 1
                return record

        signature = self._error_signature(error)

        with self._lock:
            records = self.resolutions.get(feature_name, [])

            # First, try exact match
            for record in sorted(records, key=lambda r: -r.confidence_score):
                if record.error_signature == signature:
                    self._stats["exact_matches"] += 1
                    self._lookup_cache[cache_key] = (record, time.time())
                    self._trim_cache()
                    return record

            # If no exact match, try fuzzy matching
            if allow_fuzzy:
                best_match: Optional[ResolutionRecord] = None
                best_similarity = self.FUZZY_MATCH_THRESHOLD

                for record in records:
                    similarity = self._calculate_similarity(error, record)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = record

                if best_match:
                    self._stats["fuzzy_matches"] += 1
                    self._lookup_cache[cache_key] = (best_match, time.time())
                    self._trim_cache()
                    logger.debug(f"Found fuzzy match for {feature_name} " f"(similarity: {best_similarity:.2f})")
                    return best_match

        # Cache negative result too
        self._lookup_cache[cache_key] = (None, time.time())
        self._trim_cache()
        return None

    def _trim_cache(self):
        """Trim the lookup cache if it's too large"""
        if len(self._lookup_cache) > self._cache_max_size:
            # Remove oldest entries
            sorted_items = sorted(self._lookup_cache.items(), key=lambda x: x[1][1])  # Sort by timestamp
            for key, _ in sorted_items[: len(sorted_items) // 2]:
                del self._lookup_cache[key]

    def record_success(
        self,
        feature_name: str,
        error: Exception,
        strategy: str,
        details: Dict[str, Any],
        resolution_time_ms: float = 0.0,
    ):
        """
        Record a successful resolution.

        Args:
            feature_name: Name of the feature that was healed
            error: The exception that was resolved
            strategy: The strategy that worked
            details: Additional details about the resolution
            resolution_time_ms: How long the resolution took
        """
        import platform as platform_module
        import sys

        signature = self._error_signature(error)
        keywords = self._extract_error_keywords(error)

        with self._lock:
            self._stats["resolutions_recorded"] += 1

            # Check if we already have this resolution
            for record in self.resolutions[feature_name]:
                if record.error_signature == signature and record.strategy_used == strategy:
                    # Update existing record
                    record.success_count += 1
                    record.last_used = datetime.now(timezone.utc)

                    # Update running average of resolution time
                    if resolution_time_ms > 0:
                        total = record.success_count + record.failure_count
                        record.average_resolution_time_ms = (
                            record.average_resolution_time_ms * (total - 1) + resolution_time_ms
                        ) / total

                    # Invalidate cache for this feature
                    self._invalidate_cache(feature_name)

                    self._dirty = True
                    self._save()
                    return

            # Add new resolution
            new_record = ResolutionRecord(
                feature_name=feature_name,
                error_signature=signature,
                error_type=type(error).__name__,
                error_message=str(error)[:500],
                error_keywords=keywords,
                strategy_used=strategy,
                resolution_details=details,
                success_count=1,
                failure_count=0,
                average_resolution_time_ms=resolution_time_ms,
                python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                platform=platform_module.system(),
            )

            self.resolutions[feature_name].append(new_record)

            # Invalidate cache
            self._invalidate_cache(feature_name)

            self._dirty = True
            self._save()

            logger.info(f"📚 Added new resolution to knowledge base: " f"{feature_name} -> {strategy}")

    def record_failure(
        self,
        feature_name: str,
        error: Exception,
        strategy: str,
    ):
        """Record that a resolution attempt failed"""
        signature = self._error_signature(error)

        with self._lock:
            self._stats["failures_recorded"] += 1

            for record in self.resolutions[feature_name]:
                if record.error_signature == signature and record.strategy_used == strategy:
                    record.failure_count += 1
                    self._dirty = True
                    self._save()
                    return

    def _invalidate_cache(self, feature_name: str):
        """Invalidate cache entries for a feature"""
        keys_to_remove = [k for k in self._lookup_cache if k.startswith(f"{feature_name}:")]
        for key in keys_to_remove:
            del self._lookup_cache[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        with self._lock:
            total_records = sum(len(v) for v in self.resolutions.values())

            return {
                "total_records": total_records,
                "features_covered": len(self.resolutions),
                "lookups": self._stats["total_lookups"],
                "exact_matches": self._stats["exact_matches"],
                "fuzzy_matches": self._stats["fuzzy_matches"],
                "cache_hits": self._stats["cache_hits"],
                "cache_hit_rate": (self._stats["cache_hits"] / max(1, self._stats["total_lookups"])),
                "resolutions_recorded": self._stats["resolutions_recorded"],
                "failures_recorded": self._stats["failures_recorded"],
                "average_confidence": (
                    sum(r.confidence_score for records in self.resolutions.values() for r in records)
                    / max(1, total_records)
                ),
            }

    def get_best_strategies(self, feature_name: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most effective strategies, optionally for a specific feature"""
        with self._lock:
            if feature_name:
                records = self.resolutions.get(feature_name, [])
            else:
                records = [r for records in self.resolutions.values() for r in records]

            # Sort by confidence score
            sorted_records = sorted(records, key=lambda r: r.confidence_score, reverse=True)[:limit]

            return [
                {
                    "feature": r.feature_name,
                    "strategy": r.strategy_used,
                    "error_type": r.error_type,
                    "success_count": r.success_count,
                    "success_rate": r.success_rate,
                    "confidence": r.confidence_score,
                }
                for r in sorted_records
            ]

    def export_knowledge(self, path: Optional[Path] = None) -> Path:
        """Export the knowledge base to a file for sharing/backup"""
        export_path = path or Path(f"resolution_kb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        with self._lock:
            self._save(force=True)
            shutil.copy2(self.storage_path, export_path)

        logger.info(f"Exported knowledge base to {export_path}")
        return export_path

    def import_knowledge(self, path: Path, merge: bool = True):
        """
        Import knowledge from an export file.

        Args:
            path: Path to the export file
            merge: If True, merge with existing knowledge. If False, replace.
        """
        with open(path, "r") as f:
            import_data = json.load(f)

        with self._lock:
            if not merge:
                self.resolutions.clear()

            records_data = import_data.get("resolutions", import_data)

            for feature_name, records in records_data.items():
                if feature_name.startswith("_"):
                    continue

                for r in records:
                    try:
                        record = self._parse_record(r)

                        # Check for duplicates if merging
                        if merge:
                            existing = None
                            for existing_r in self.resolutions[feature_name]:
                                if (
                                    existing_r.error_signature == record.error_signature
                                    and existing_r.strategy_used == record.strategy_used
                                ):
                                    existing = existing_r
                                    break

                            if existing:
                                # Merge counts
                                existing.success_count += record.success_count
                                existing.failure_count += record.failure_count
                                if record.last_used > existing.last_used:
                                    existing.last_used = record.last_used
                                continue

                        self.resolutions[feature_name].append(record)

                    except Exception as e:
                        logger.warning(f"Could not import record: {e}")

            self._dirty = True
            self._save(force=True)

        logger.info(f"Imported knowledge base from {path}")

    def close(self):
        """Close the knowledge base, ensuring all data is saved"""
        if self._save_timer:
            self._save_timer.cancel()
        self._save(force=True)

    def save(self):
        """Public method to force save knowledge base data"""
        self._save(force=True)


class FeatureUsageTracker:
    """
    Advanced feature usage tracking with analytics and trend detection.

    Features:
    - Time-decay weighted usage scoring
    - Peak usage time tracking
    - Session-based analytics
    - Usage trend detection (increasing/decreasing/stable)
    - Performance correlation tracking
    - Automatic data aggregation and cleanup

    Usage data is used to prioritize which failed features should be
    healed first - more frequently used features get higher priority.
    """

    # Configuration constants
    SCHEMA_VERSION = 2
    TIME_DECAY_HALF_LIFE_HOURS = 168  # 1 week - usage older than this counts for less
    HOURLY_BUCKET_RETENTION_DAYS = 7  # Keep hourly granularity for 7 days
    DAILY_BUCKET_RETENTION_DAYS = 90  # Keep daily aggregates for 90 days
    SAVE_INTERVAL_USES = 10  # Save after every N uses

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        session_id: Optional[str] = None,
    ):
        """
        Initialize the feature usage tracker.

        Args:
            storage_path: Path to store usage data
            session_id: Unique identifier for this session
        """
        self.storage_path = storage_path or Path("data/feature_usage.json")
        self.session_id = session_id or self._generate_session_id()

        # Basic usage tracking
        self.usage_counts: Dict[str, int] = defaultdict(int)
        self.last_used: Dict[str, datetime] = {}

        # Time-series data for trends
        self.hourly_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.daily_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Session tracking
        self.session_usage: Dict[str, int] = defaultdict(int)
        self.session_start = datetime.now(timezone.utc)

        # Performance tracking (feature_name -> list of response times)
        self.response_times: Dict[str, List[float]] = defaultdict(list)

        # Error tracking per feature
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_error: Dict[str, datetime] = {}

        # Peak usage tracking
        self.peak_usage_hour: Dict[str, int] = defaultdict(int)  # Hour of day (0-23)
        self.peak_usage_day: Dict[str, int] = defaultdict(int)  # Day of week (0-6)

        # Lock for thread safety
        self._lock = threading.RLock()
        self._dirty = False
        self._total_uses_since_save = 0

        # Load existing data
        self._load()

    @staticmethod
    def _generate_session_id() -> str:
        """Generate a unique session ID"""
        import uuid

        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"

    def _load(self):
        """Load usage data from disk with migration support"""
        try:
            if not self.storage_path.exists():
                logger.debug("No existing usage data found")
                return

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            # Check schema version
            schema_version = data.get("_schema_version", 1)
            if schema_version < self.SCHEMA_VERSION:
                data = self._migrate_schema(data, schema_version)

            # Load basic counts
            self.usage_counts = defaultdict(int, data.get("counts", {}))

            # Load last_used timestamps
            self.last_used = {}
            for k, v in data.get("last_used", {}).items():
                try:
                    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    self.last_used[k] = dt
                except Exception:
                    pass

            # Load time-series data
            self.hourly_usage = defaultdict(
                lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in data.get("hourly_usage", {}).items()}
            )
            self.daily_usage = defaultdict(
                lambda: defaultdict(int), {k: defaultdict(int, v) for k, v in data.get("daily_usage", {}).items()}
            )

            # Load performance data
            self.response_times = defaultdict(list, {k: list(v) for k, v in data.get("response_times", {}).items()})

            # Load error data
            self.error_counts = defaultdict(int, data.get("error_counts", {}))
            self.last_error = {}
            for k, v in data.get("last_error", {}).items():
                try:
                    dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    self.last_error[k] = dt
                except Exception:
                    pass

            # Load peak usage patterns
            self.peak_usage_hour = defaultdict(int, data.get("peak_usage_hour", {}))
            self.peak_usage_day = defaultdict(int, data.get("peak_usage_day", {}))

            total_uses = sum(self.usage_counts.values())
            logger.debug(f"Loaded usage data: {total_uses} total uses across {len(self.usage_counts)} features")

            # Run cleanup on load
            self._cleanup_old_data()

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in usage data file: {e}")
        except Exception as e:
            logger.warning(f"Could not load feature usage data: {e}")

    def _migrate_schema(self, data: Dict[str, Any], from_version: int) -> Dict[str, Any]:
        """Migrate data from older schema versions"""
        logger.info(f"Migrating usage data from v{from_version} to v{self.SCHEMA_VERSION}")

        if from_version == 1:
            # v1 -> v2: Add time-series and performance tracking
            data["_schema_version"] = self.SCHEMA_VERSION
            data["hourly_usage"] = {}
            data["daily_usage"] = {}
            data["response_times"] = {}
            data["error_counts"] = {}
            data["last_error"] = {}
            data["peak_usage_hour"] = {}
            data["peak_usage_day"] = {}

        return data

    def _save(self, force: bool = False):
        """Save usage data to disk"""
        if not force and not self._dirty:
            return

        with self._lock:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)

                data = {
                    "_schema_version": self.SCHEMA_VERSION,
                    "_last_saved": datetime.now(timezone.utc).isoformat(),
                    "_session_id": self.session_id,
                    "counts": dict(self.usage_counts),
                    "last_used": {k: v.isoformat() for k, v in self.last_used.items()},
                    "hourly_usage": {k: dict(v) for k, v in self.hourly_usage.items()},
                    "daily_usage": {k: dict(v) for k, v in self.daily_usage.items()},
                    "response_times": {
                        k: list(v[-100:])  # Keep last 100 response times per feature
                        for k, v in self.response_times.items()
                    },
                    "error_counts": dict(self.error_counts),
                    "last_error": {k: v.isoformat() for k, v in self.last_error.items()},
                    "peak_usage_hour": dict(self.peak_usage_hour),
                    "peak_usage_day": dict(self.peak_usage_day),
                }

                # Atomic write
                temp_path = self.storage_path.with_suffix(".json.tmp")
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)
                temp_path.replace(self.storage_path)

                self._dirty = False
                self._total_uses_since_save = 0

            except Exception as e:
                logger.warning(f"Could not save feature usage data: {e}")

    def _cleanup_old_data(self):
        """Remove old time-series data to save space"""
        now = datetime.now(timezone.utc)
        hourly_cutoff = now - timedelta(days=self.HOURLY_BUCKET_RETENTION_DAYS)
        daily_cutoff = now - timedelta(days=self.DAILY_BUCKET_RETENTION_DAYS)

        with self._lock:
            # Cleanup hourly data
            for feature_name in list(self.hourly_usage.keys()):
                hourly_data = self.hourly_usage[feature_name]
                keys_to_remove = [
                    k for k in hourly_data if datetime.fromisoformat(k.replace("Z", "+00:00")) < hourly_cutoff
                ]
                for k in keys_to_remove:
                    del hourly_data[k]

            # Cleanup daily data
            for feature_name in list(self.daily_usage.keys()):
                daily_data = self.daily_usage[feature_name]
                keys_to_remove = [k for k in daily_data if datetime.fromisoformat(k) < daily_cutoff]
                for k in keys_to_remove:
                    del daily_data[k]

    def record_usage(
        self,
        feature_name: str,
        response_time_ms: Optional[float] = None,
    ):
        """
        Record that a feature was used.

        Args:
            feature_name: Name of the feature
            response_time_ms: Optional response time in milliseconds
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            # Basic counting
            self.usage_counts[feature_name] += 1
            self.last_used[feature_name] = now
            self.session_usage[feature_name] += 1

            # Time-series tracking
            hour_key = now.strftime("%Y-%m-%dT%H:00:00")
            day_key = now.strftime("%Y-%m-%d")

            self.hourly_usage[feature_name][hour_key] += 1
            self.daily_usage[feature_name][day_key] += 1

            # Peak usage tracking
            hour_of_day = now.hour
            day_of_week = now.weekday()

            if self.hourly_usage[feature_name][hour_key] > self.usage_counts.get(feature_name, 0) / max(
                1, len(self.hourly_usage[feature_name])
            ):
                self.peak_usage_hour[feature_name] = hour_of_day
                self.peak_usage_day[feature_name] = day_of_week

            # Response time tracking
            if response_time_ms is not None:
                self.response_times[feature_name].append(response_time_ms)
                # Keep only last 1000 response times
                if len(self.response_times[feature_name]) > 1000:
                    self.response_times[feature_name] = self.response_times[feature_name][-500:]

            self._dirty = True
            self._total_uses_since_save += 1

            # Save periodically
            if self._total_uses_since_save >= self.SAVE_INTERVAL_USES:
                self._save()

    def record_error(self, feature_name: str):
        """Record that a feature encountered an error"""
        with self._lock:
            self.error_counts[feature_name] += 1
            self.last_error[feature_name] = datetime.now(timezone.utc)
            self._dirty = True

    def get_priority_score(self, feature_name: str) -> float:
        """
        Calculate a priority score for a feature.

        Higher score = higher priority for healing.
        Uses time-decay weighting so recent usage counts more.

        Args:
            feature_name: Name of the feature

        Returns:
            Priority score (0-100)
        """
        import math

        with self._lock:
            now = datetime.now(timezone.utc)

            # Base score from total usage (logarithmic scaling)
            total_count = self.usage_counts.get(feature_name, 0)
            base_score = math.log1p(total_count) * 5  # 0-35 points typically

            # Time-decay weighted recent usage
            weighted_recent = 0.0
            for day_key, count in self.daily_usage.get(feature_name, {}).items():
                try:
                    day = datetime.fromisoformat(day_key).replace(tzinfo=timezone.utc)
                    hours_ago = (now - day).total_seconds() / 3600
                    # Exponential decay: half weight after TIME_DECAY_HALF_LIFE_HOURS
                    decay = 0.5 ** (hours_ago / self.TIME_DECAY_HALF_LIFE_HOURS)
                    weighted_recent += count * decay
                except Exception:
                    continue

            recent_score = math.log1p(weighted_recent) * 8  # 0-40 points

            # Recency bonus - features used recently get priority
            last = self.last_used.get(feature_name)
            recency_bonus = 0.0
            if last:
                hours_ago = (now - last).total_seconds() / 3600
                if hours_ago < 1:
                    recency_bonus = 20  # Used in last hour
                elif hours_ago < 24:
                    recency_bonus = 15 * (24 - hours_ago) / 24
                elif hours_ago < 168:  # 1 week
                    recency_bonus = 5 * (168 - hours_ago) / 168

            # Session usage bonus
            session_count = self.session_usage.get(feature_name, 0)
            session_bonus = min(10, math.log1p(session_count) * 3)

            # Error rate penalty (features with high error rates might be unstable)
            error_count = self.error_counts.get(feature_name, 0)
            error_penalty = 0.0
            if total_count > 0 and error_count > 0:
                error_rate = error_count / total_count
                if error_rate > 0.5:
                    error_penalty = 10
                elif error_rate > 0.2:
                    error_penalty = 5

            score = base_score + recent_score + recency_bonus + session_bonus - error_penalty

            return max(0.0, min(100.0, score))

    def get_usage_trend(self, feature_name: str, days: int = 7) -> str:
        """
        Detect usage trend for a feature.

        Returns:
            One of: "increasing", "decreasing", "stable", "unknown"
        """
        with self._lock:
            daily_data = self.daily_usage.get(feature_name, {})

            if len(daily_data) < 3:
                return "unknown"

            # Get last N days of data
            now = datetime.now(timezone.utc)
            recent_days = []

            for i in range(days):
                day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
                recent_days.append(daily_data.get(day, 0))

            if len(recent_days) < 3:
                return "unknown"

            # Calculate trend using simple linear regression
            n = len(recent_days)
            x_mean = (n - 1) / 2
            y_mean = sum(recent_days) / n

            numerator = sum((i - x_mean) * (recent_days[i] - y_mean) for i in range(n))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                return "stable"

            slope = numerator / denominator

            # Normalize slope by average usage
            if y_mean > 0:
                normalized_slope = slope / y_mean
            else:
                normalized_slope = 0

            if normalized_slope > 0.1:
                return "increasing"
            elif normalized_slope < -0.1:
                return "decreasing"
            else:
                return "stable"

    def get_peak_usage_times(self, feature_name: str) -> Dict[str, Any]:
        """Get peak usage times for a feature"""
        with self._lock:
            days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            return {
                "peak_hour": self.peak_usage_hour.get(feature_name, 12),
                "peak_day": days_of_week[self.peak_usage_day.get(feature_name, 0)],
                "peak_day_index": self.peak_usage_day.get(feature_name, 0),
            }

    def get_performance_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get performance statistics for a feature"""
        with self._lock:
            response_times = self.response_times.get(feature_name, [])

            if not response_times:
                return {
                    "sample_count": 0,
                    "average_ms": 0.0,
                    "median_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                }

            sorted_times = sorted(response_times)
            n = len(sorted_times)

            return {
                "sample_count": n,
                "average_ms": sum(sorted_times) / n,
                "median_ms": sorted_times[n // 2],
                "p95_ms": sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
                "p99_ms": sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
                "min_ms": sorted_times[0],
                "max_ms": sorted_times[-1],
            }

    def get_feature_analytics(self, feature_name: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a feature"""
        with self._lock:
            return {
                "feature_name": feature_name,
                "total_usage": self.usage_counts.get(feature_name, 0),
                "session_usage": self.session_usage.get(feature_name, 0),
                "last_used": (
                    self.last_used.get(feature_name, datetime.min).isoformat()
                    if feature_name in self.last_used
                    else None
                ),
                "error_count": self.error_counts.get(feature_name, 0),
                "last_error": (
                    self.last_error.get(feature_name, datetime.min).isoformat()
                    if feature_name in self.last_error
                    else None
                ),
                "priority_score": self.get_priority_score(feature_name),
                "usage_trend": self.get_usage_trend(feature_name),
                "peak_times": self.get_peak_usage_times(feature_name),
                "performance": self.get_performance_stats(feature_name),
            }

    def get_all_analytics(self) -> Dict[str, Any]:
        """Get analytics summary for all features"""
        with self._lock:
            features = list(self.usage_counts.keys())

            return {
                "session_id": self.session_id,
                "session_start": self.session_start.isoformat(),
                "total_features": len(features),
                "total_uses": sum(self.usage_counts.values()),
                "session_uses": sum(self.session_usage.values()),
                "features_by_priority": sorted(
                    [(f, self.get_priority_score(f)) for f in features], key=lambda x: x[1], reverse=True
                ),
                "most_used": sorted([(f, self.usage_counts[f]) for f in features], key=lambda x: x[1], reverse=True)[
                    :10
                ],
                "error_prone": sorted(
                    [(f, self.error_counts[f]) for f in features if self.error_counts.get(f, 0) > 0],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10],
            }

    def close(self):
        """Close the tracker and save all data"""
        self._save(force=True)

    def save(self):
        """Public method to force save usage data"""
        self._save(force=True)

    def get_feature_stats(self, feature_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Dictionary with feature statistics
        """
        with self._lock:
            return {
                "total_usage": self.usage_counts.get(feature_name, 0),
                "session_usage": self.session_usage.get(feature_name, 0),
                "last_used": (self.last_used[feature_name].isoformat() if feature_name in self.last_used else None),
                "error_count": self.error_counts.get(feature_name, 0),
                "priority_score": self.get_priority_score(feature_name),
                "usage_trend": self.get_usage_trend(feature_name),
            }

    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall usage statistics.

        Returns:
            Dictionary with overall statistics
        """
        with self._lock:
            return {
                "total_features": len(self.usage_counts),
                "total_uses": sum(self.usage_counts.values()),
                "session_uses": sum(self.session_usage.values()),
                "session_id": self.session_id,
                "session_start": self.session_start.isoformat(),
                "error_count": sum(self.error_counts.values()),
            }


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern"""

    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 3  # Successes to close circuit (half-open)
    timeout_seconds: float = 60.0  # Time before retry when open
    half_open_max_calls: int = 1  # Max concurrent calls in half-open


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerState:
    """State tracking for a circuit breaker"""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    half_open_calls: int = 0


@dataclass
class HealingMetrics:
    """Comprehensive healing metrics"""

    total_heal_attempts: int = 0
    successful_heals: int = 0
    failed_heals: int = 0
    total_healing_time_ms: float = 0.0
    features_healed: Set[str] = field(default_factory=set)
    features_abandoned: Set[str] = field(default_factory=set)
    healing_by_strategy: Dict[str, Dict[str, int]] = field(default_factory=dict)
    healing_history: List[Dict[str, Any]] = field(default_factory=list)
    cycle_count: int = 0
    last_cycle_time: Optional[datetime] = None
    last_cycle_duration_ms: float = 0.0
    average_cycle_duration_ms: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate overall healing success rate"""
        if self.total_heal_attempts == 0:
            return 0.0
        return self.successful_heals / self.total_heal_attempts

    @property
    def average_healing_time_ms(self) -> float:
        """Calculate average healing time"""
        if self.successful_heals == 0:
            return 0.0
        return self.total_healing_time_ms / self.successful_heals


@dataclass
class HealingEvent:
    """Event emitted during healing operations"""

    event_type: str  # started, succeeded, failed, abandoned, paused, resumed
    feature_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_used: Optional[str] = None
    attempt_number: int = 0
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackgroundHealer:
    """
    Advanced background process that attempts to heal failed features.

    Features:
    - Circuit breaker pattern prevents cascading failures
    - Adaptive scheduling based on system load and success rates
    - Comprehensive metrics collection and reporting
    - Resource monitoring to avoid system overload
    - Event-driven architecture for observability
    - Priority queue with dynamic re-prioritization
    - Health check integration
    - Graceful degradation under pressure
    """

    def __init__(
        self,
        resilience_manager: "ResilientStartupManager",
        check_interval: float = 60.0,  # Check every minute
        min_priority: int = 19,  # Nice level for background process (Unix)
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        max_concurrent_heals: int = 2,
        max_healing_time_per_cycle: float = 30.0,  # Max seconds per healing cycle
        cpu_threshold: float = 80.0,  # Pause healing if CPU > this %
        memory_threshold: float = 85.0,  # Pause healing if memory > this %
        enable_adaptive_scheduling: bool = True,
        metrics_history_limit: int = 1000,
    ):
        self.manager = resilience_manager
        self.check_interval = check_interval
        self.min_priority = min_priority
        self.max_concurrent_heals = max_concurrent_heals
        self.max_healing_time_per_cycle = max_healing_time_per_cycle
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.enable_adaptive_scheduling = enable_adaptive_scheduling
        self.metrics_history_limit = metrics_history_limit

        # Circuit breaker configuration and state
        self.circuit_config = circuit_breaker_config or CircuitBreakerConfig()
        self._circuit_breakers: Dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)
        self._global_circuit = CircuitBreakerState()

        # State management
        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._lock = threading.Lock()

        # Metrics
        self.metrics = HealingMetrics()

        # Event handlers
        self._event_handlers: List[Callable[[HealingEvent], Any]] = []

        # Adaptive scheduling state
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._adaptive_interval = check_interval
        self._last_resource_check: Optional[datetime] = None
        self._resource_check_interval = 10.0  # Check resources every 10s

        # Healing queue with priorities
        self._healing_queue: List[Tuple[float, str, datetime]] = []  # (priority, name, queued_at)
        self._currently_healing: Set[str] = set()

        # Health check callbacks
        self._health_checks: Dict[str, Callable[[], bool]] = {}

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the background healer.

        Returns:
            True if started successfully, False if already running
        """
        if self._running:
            logger.warning("Background healer is already running")
            return False

        self._running = True
        self._paused = False
        self._stop_event.clear()
        self._pause_event.set()

        # Reset adaptive scheduling
        self._adaptive_interval = self.check_interval
        self._consecutive_failures = 0
        self._consecutive_successes = 0

        self._task = asyncio.create_task(self._healing_loop())

        await self._emit_event(
            HealingEvent(
                event_type="healer_started",
                feature_name="__system__",
                metadata={
                    "check_interval": self.check_interval,
                    "max_concurrent_heals": self.max_concurrent_heals,
                    "adaptive_scheduling": self.enable_adaptive_scheduling,
                },
            )
        )

        logger.info(
            f"🏥 Background healer started (interval={self.check_interval}s, "
            f"max_concurrent={self.max_concurrent_heals})"
        )
        return True

    async def stop(self, timeout: float = 10.0) -> bool:
        """
        Stop the background healer gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            True if stopped gracefully, False if had to force stop
        """
        if not self._running:
            return True

        self._running = False
        self._stop_event.set()
        self._pause_event.set()  # Unblock if paused

        graceful = True
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Background healer did not stop gracefully, cancelling")
                self._task.cancel()
                graceful = False
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass

        await self._emit_event(
            HealingEvent(
                event_type="healer_stopped",
                feature_name="__system__",
                metadata={
                    "graceful": graceful,
                    "final_metrics": self.get_metrics_summary(),
                },
            )
        )

        logger.info(f"🏥 Background healer stopped (graceful={graceful})")
        return graceful

    async def pause(self, reason: str = "manual") -> bool:
        """
        Pause healing operations temporarily.

        Args:
            reason: Why healing is being paused

        Returns:
            True if paused, False if already paused
        """
        if self._paused:
            return False

        self._paused = True
        self._pause_event.clear()

        await self._emit_event(
            HealingEvent(
                event_type="healer_paused",
                feature_name="__system__",
                metadata={"reason": reason},
            )
        )

        logger.info(f"🏥 Background healer paused: {reason}")
        return True

    async def resume(self) -> bool:
        """
        Resume healing operations.

        Returns:
            True if resumed, False if wasn't paused
        """
        if not self._paused:
            return False

        self._paused = False
        self._pause_event.set()

        await self._emit_event(
            HealingEvent(
                event_type="healer_resumed",
                feature_name="__system__",
            )
        )

        logger.info("🏥 Background healer resumed")
        return True

    # =========================================================================
    # Circuit Breaker Pattern
    # =========================================================================

    def _check_circuit(self, feature_name: str) -> bool:
        """
        Check if circuit breaker allows healing attempt.

        Args:
            feature_name: Name of feature to check

        Returns:
            True if healing is allowed, False if circuit is open
        """
        # Check global circuit first
        if not self._check_circuit_state(self._global_circuit):
            return False

        # Check feature-specific circuit
        circuit = self._circuit_breakers[feature_name]
        return self._check_circuit_state(circuit)

    def _check_circuit_state(self, circuit: CircuitBreakerState) -> bool:
        """Check a specific circuit breaker state"""
        now = datetime.now(timezone.utc)

        if circuit.state == CircuitState.CLOSED:
            return True

        if circuit.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if circuit.last_failure_time:
                elapsed = (now - circuit.last_failure_time).total_seconds()
                if elapsed >= self.circuit_config.timeout_seconds:
                    # Transition to half-open
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.last_state_change = now
                    circuit.half_open_calls = 0
                    logger.info(f"Circuit breaker transitioning to HALF_OPEN after {elapsed:.1f}s")
                    return True
            return False

        if circuit.state == CircuitState.HALF_OPEN:
            # Allow limited calls
            if circuit.half_open_calls < self.circuit_config.half_open_max_calls:
                circuit.half_open_calls += 1
                return True
            return False

        return True

    def _record_circuit_success(self, feature_name: str):
        """Record a successful healing for circuit breaker"""
        # Update feature circuit
        circuit = self._circuit_breakers[feature_name]
        self._record_success_on_circuit(circuit)

        # Update global circuit
        self._record_success_on_circuit(self._global_circuit)

    def _record_success_on_circuit(self, circuit: CircuitBreakerState):
        """Record success on a specific circuit"""
        now = datetime.now(timezone.utc)
        circuit.success_count += 1

        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.success_count >= self.circuit_config.success_threshold:
                # Close the circuit
                circuit.state = CircuitState.CLOSED
                circuit.failure_count = 0
                circuit.success_count = 0
                circuit.last_state_change = now
                logger.info("Circuit breaker CLOSED - healing recovered")

    def _record_circuit_failure(self, feature_name: str):
        """Record a failed healing for circuit breaker"""
        now = datetime.now(timezone.utc)

        # Update feature circuit
        circuit = self._circuit_breakers[feature_name]
        self._record_failure_on_circuit(circuit, now)

        # Update global circuit
        self._record_failure_on_circuit(self._global_circuit, now)

    def _record_failure_on_circuit(self, circuit: CircuitBreakerState, now: datetime):
        """Record failure on a specific circuit"""
        circuit.failure_count += 1
        circuit.last_failure_time = now
        circuit.success_count = 0

        if circuit.state == CircuitState.HALF_OPEN:
            # Failed during test, open circuit again
            circuit.state = CircuitState.OPEN
            circuit.last_state_change = now
            logger.warning("Circuit breaker OPENED - half-open test failed")

        elif circuit.state == CircuitState.CLOSED:
            if circuit.failure_count >= self.circuit_config.failure_threshold:
                circuit.state = CircuitState.OPEN
                circuit.last_state_change = now
                logger.warning(f"Circuit breaker OPENED after {circuit.failure_count} failures")

    def get_circuit_status(self, feature_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get circuit breaker status.

        Args:
            feature_name: Specific feature, or None for global

        Returns:
            Circuit breaker status information
        """
        if feature_name:
            circuit = self._circuit_breakers.get(feature_name, CircuitBreakerState())
        else:
            circuit = self._global_circuit

        return {
            "state": circuit.state.value,
            "failure_count": circuit.failure_count,
            "success_count": circuit.success_count,
            "last_failure": (circuit.last_failure_time.isoformat() if circuit.last_failure_time else None),
            "last_state_change": circuit.last_state_change.isoformat(),
        }

    # =========================================================================
    # Resource Monitoring
    # =========================================================================

    def _check_system_resources(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if system resources allow healing operations.

        Returns:
            Tuple of (resources_ok, resource_info)
        """
        resources = {"cpu_percent": 0.0, "memory_percent": 0.0}

        try:
            import psutil

            resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            resources["memory_percent"] = psutil.virtual_memory().percent

            ok = resources["cpu_percent"] < self.cpu_threshold and resources["memory_percent"] < self.memory_threshold
            return ok, resources

        except ImportError:
            # psutil not available, assume resources are fine
            return True, resources
        except Exception as e:
            logger.debug(f"Resource check failed: {e}")
            return True, resources

    def _should_check_resources(self) -> bool:
        """Determine if we should check system resources"""
        if self._last_resource_check is None:
            return True

        elapsed = (datetime.now(timezone.utc) - self._last_resource_check).total_seconds()
        return elapsed >= self._resource_check_interval

    # =========================================================================
    # Adaptive Scheduling
    # =========================================================================

    def _update_adaptive_interval(self, success: bool):
        """
        Update the check interval based on healing success/failure patterns.

        Args:
            success: Whether the last healing attempt succeeded
        """
        if not self.enable_adaptive_scheduling:
            return

        if success:
            self._consecutive_successes += 1
            self._consecutive_failures = 0

            # Speed up if consistently successful
            if self._consecutive_successes >= 3:
                self._adaptive_interval = max(
                    self.check_interval * 0.5,  # Min 50% of original
                    self._adaptive_interval * 0.8,
                )
        else:
            self._consecutive_failures += 1
            self._consecutive_successes = 0

            # Slow down if consistently failing
            if self._consecutive_failures >= 3:
                self._adaptive_interval = min(
                    self.check_interval * 3.0,  # Max 3x original
                    self._adaptive_interval * 1.5,
                )

        logger.debug(f"Adaptive interval updated to {self._adaptive_interval:.1f}s")

    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive scheduling statistics"""
        return {
            "base_interval": self.check_interval,
            "current_interval": self._adaptive_interval,
            "consecutive_successes": self._consecutive_successes,
            "consecutive_failures": self._consecutive_failures,
            "adaptive_enabled": self.enable_adaptive_scheduling,
        }

    # =========================================================================
    # Event System
    # =========================================================================

    def add_event_handler(self, handler: Callable[[HealingEvent], Any]) -> Callable[[], None]:
        """
        Add an event handler for healing events.

        Args:
            handler: Callback function that receives HealingEvent

        Returns:
            Function to remove the handler
        """
        self._event_handlers.append(handler)

        def remove():
            if handler in self._event_handlers:
                self._event_handlers.remove(handler)

        return remove

    async def _emit_event(self, event: HealingEvent):
        """Emit a healing event to all handlers"""
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")

        # Also log to metrics history
        with self._lock:
            self.metrics.healing_history.append(
                {
                    "event_type": event.event_type,
                    "feature_name": event.feature_name,
                    "timestamp": event.timestamp.isoformat(),
                    "strategy_used": event.strategy_used,
                    "attempt_number": event.attempt_number,
                    "duration_ms": event.duration_ms,
                    "error_message": event.error_message,
                }
            )

            # Trim history if needed
            if len(self.metrics.healing_history) > self.metrics_history_limit:
                self.metrics.healing_history = self.metrics.healing_history[-self.metrics_history_limit :]

    # =========================================================================
    # Health Check Integration
    # =========================================================================

    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register a health check function.

        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
        """
        self._health_checks[name] = check_func

    def unregister_health_check(self, name: str) -> bool:
        """
        Unregister a health check.

        Returns:
            True if removed, False if not found
        """
        if name in self._health_checks:
            del self._health_checks[name]
            return True
        return False

    def run_health_checks(self) -> Dict[str, bool]:
        """
        Run all registered health checks.

        Returns:
            Dict mapping check name to result (True = healthy)
        """
        results = {}
        for name, check_func in self._health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.warning(f"Health check '{name}' failed: {e}")
                results[name] = False
        return results

    def is_healthy(self) -> bool:
        """
        Check if healer is healthy overall.

        Returns:
            True if running and all health checks pass
        """
        if not self._running:
            return False

        if self._paused:
            return True  # Paused is a valid healthy state

        # Check circuit breaker
        if self._global_circuit.state == CircuitState.OPEN:
            return False

        # Run health checks
        results = self.run_health_checks()
        return all(results.values()) if results else True

    # =========================================================================
    # Metrics and Statistics
    # =========================================================================

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        with self._lock:
            return {
                "total_attempts": self.metrics.total_heal_attempts,
                "successful_heals": self.metrics.successful_heals,
                "failed_heals": self.metrics.failed_heals,
                "success_rate": round(self.metrics.success_rate * 100, 2),
                "average_healing_time_ms": round(self.metrics.average_healing_time_ms, 2),
                "features_healed": list(self.metrics.features_healed),
                "features_abandoned": list(self.metrics.features_abandoned),
                "healing_by_strategy": dict(self.metrics.healing_by_strategy),
                "cycle_count": self.metrics.cycle_count,
                "last_cycle_time": (self.metrics.last_cycle_time.isoformat() if self.metrics.last_cycle_time else None),
                "last_cycle_duration_ms": round(self.metrics.last_cycle_duration_ms, 2),
                "average_cycle_duration_ms": round(self.metrics.average_cycle_duration_ms, 2),
                "peak_queue_size": self.metrics.peak_queue_size,
                "current_queue_size": len(self._healing_queue),
                "currently_healing": list(self._currently_healing),
                "circuit_breaker": self.get_circuit_status(),
                "adaptive_scheduling": self.get_adaptive_stats(),
                "is_running": self._running,
                "is_paused": self._paused,
            }

    def reset_metrics(self):
        """Reset all metrics to initial state"""
        with self._lock:
            self.metrics = HealingMetrics()
            logger.info("Healing metrics reset")

    # =========================================================================
    # Main Healing Loop
    # =========================================================================

    async def _healing_loop(self):
        """Main healing loop - runs in background with advanced features"""
        # Set low priority for this task (Unix only)
        try:
            import os

            os.nice(self.min_priority)
        except (OSError, AttributeError):
            pass  # Not supported on this platform

        while self._running:
            try:
                # Wait for pause to clear
                await self._pause_event.wait()

                # Wait with adaptive timeout
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._adaptive_interval)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    pass  # Normal timeout, continue healing

                cycle_start = time.time()
                self.metrics.cycle_count += 1
                self.metrics.last_cycle_time = datetime.now(timezone.utc)

                # Check system resources
                if self._should_check_resources():
                    self._last_resource_check = datetime.now(timezone.utc)
                    resources_ok, resource_info = self._check_system_resources()

                    if not resources_ok:
                        logger.info(
                            f"Pausing healing due to resource pressure: "
                            f"CPU={resource_info['cpu_percent']:.1f}%, "
                            f"Memory={resource_info['memory_percent']:.1f}%"
                        )
                        await asyncio.sleep(self._adaptive_interval)
                        continue

                # Check global circuit breaker
                if not self._check_circuit_state(self._global_circuit):
                    logger.debug("Global circuit breaker is OPEN, skipping cycle")
                    await asyncio.sleep(self._adaptive_interval)
                    continue

                # Get features that need healing, sorted by priority
                failed_features = self.manager.get_failed_features()

                if failed_features:
                    # Update queue metrics
                    self.metrics.current_queue_size = len(failed_features)
                    self.metrics.peak_queue_size = max(self.metrics.peak_queue_size, len(failed_features))

                    # Sort by priority score (higher = heal first)
                    failed_features.sort(
                        key=lambda f: self.manager.usage_tracker.get_priority_score(f.definition.name),
                        reverse=True,
                    )

                    # Process features with concurrency limit
                    healed_this_cycle = 0
                    cycle_time_limit = time.time() + self.max_healing_time_per_cycle

                    for feature_state in failed_features:
                        if not self._running:
                            break

                        # Check time limit
                        if time.time() > cycle_time_limit:
                            logger.debug("Cycle time limit reached, deferring remaining")
                            break

                        # Check circuit breaker for this feature
                        if not self._check_circuit(feature_state.definition.name):
                            continue

                        # Check if enough time has passed since last attempt
                        if feature_state.last_heal_attempt:
                            backoff = feature_state.definition.heal_backoff_seconds * (
                                2 ** min(feature_state.heal_attempts, 5)
                            )
                            last_attempt = feature_state.last_heal_attempt
                            if last_attempt.tzinfo is None:
                                last_attempt = last_attempt.replace(tzinfo=timezone.utc)
                            elapsed = (datetime.now(timezone.utc) - last_attempt).total_seconds()
                            if elapsed < backoff:
                                continue

                        # Check if we've exceeded max attempts
                        if feature_state.heal_attempts >= feature_state.definition.max_heal_attempts:
                            if feature_state.status != FeatureStatus.PERMANENTLY_FAILED:
                                feature_state.status = FeatureStatus.PERMANENTLY_FAILED
                                self.metrics.features_abandoned.add(feature_state.definition.name)
                                await self._emit_event(
                                    HealingEvent(
                                        event_type="abandoned",
                                        feature_name=feature_state.definition.name,
                                        attempt_number=feature_state.heal_attempts,
                                        error_message=str(feature_state.error),
                                    )
                                )
                                await self.manager.notify_permanent_failure(feature_state)
                            continue

                        # Mark as currently healing
                        self._currently_healing.add(feature_state.definition.name)

                        # Emit start event
                        heal_start = time.time()
                        await self._emit_event(
                            HealingEvent(
                                event_type="started",
                                feature_name=feature_state.definition.name,
                                attempt_number=feature_state.heal_attempts + 1,
                            )
                        )

                        # Attempt healing
                        self.metrics.total_heal_attempts += 1
                        success = await self.manager.attempt_heal(feature_state)
                        heal_duration_ms = (time.time() - heal_start) * 1000

                        # Update metrics and circuit breaker
                        if success:
                            self.metrics.successful_heals += 1
                            self.metrics.total_healing_time_ms += heal_duration_ms
                            self.metrics.features_healed.add(feature_state.definition.name)
                            self._record_circuit_success(feature_state.definition.name)
                            self._update_adaptive_interval(True)
                            healed_this_cycle += 1

                            await self._emit_event(
                                HealingEvent(
                                    event_type="succeeded",
                                    feature_name=feature_state.definition.name,
                                    attempt_number=feature_state.heal_attempts,
                                    duration_ms=heal_duration_ms,
                                )
                            )
                        else:
                            self.metrics.failed_heals += 1
                            self._record_circuit_failure(feature_state.definition.name)
                            self._update_adaptive_interval(False)

                            await self._emit_event(
                                HealingEvent(
                                    event_type="failed",
                                    feature_name=feature_state.definition.name,
                                    attempt_number=feature_state.heal_attempts,
                                    duration_ms=heal_duration_ms,
                                    error_message=str(feature_state.error),
                                )
                            )

                        # Remove from currently healing
                        self._currently_healing.discard(feature_state.definition.name)

                        # Small delay between healing attempts
                        await asyncio.sleep(1.0)

                # Update cycle metrics
                cycle_duration_ms = (time.time() - cycle_start) * 1000
                self.metrics.last_cycle_duration_ms = cycle_duration_ms

                # Update rolling average
                if self.metrics.cycle_count == 1:
                    self.metrics.average_cycle_duration_ms = cycle_duration_ms
                else:
                    self.metrics.average_cycle_duration_ms = (
                        self.metrics.average_cycle_duration_ms * 0.9 + cycle_duration_ms * 0.1
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in healing loop: {e}", exc_info=True)
                await asyncio.sleep(self._adaptive_interval)

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if the healer is currently running"""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if the healer is currently paused"""
        return self._paused


@dataclass
class StartupMetrics:
    """Comprehensive startup metrics tracking"""

    startup_start_time: Optional[datetime] = None
    startup_end_time: Optional[datetime] = None
    total_features: int = 0
    critical_features: int = 0
    non_critical_features: int = 0
    successful_loads: int = 0
    failed_loads: int = 0
    degraded_loads: int = 0
    parallel_batches: int = 0
    rollbacks_performed: int = 0
    feature_load_times: Dict[str, float] = field(default_factory=dict)
    dependency_resolution_time_ms: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)

    @property
    def total_startup_time_ms(self) -> float:
        """Total startup time in milliseconds"""
        if not self.startup_start_time or not self.startup_end_time:
            return 0.0
        return (self.startup_end_time - self.startup_start_time).total_seconds() * 1000

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.successful_loads + self.failed_loads + self.degraded_loads
        return (self.successful_loads + self.degraded_loads) / total if total > 0 else 0.0

    @property
    def average_load_time_ms(self) -> float:
        """Average feature load time"""
        if not self.feature_load_times:
            return 0.0
        return sum(self.feature_load_times.values()) / len(self.feature_load_times)


@dataclass
class StartupEvent:
    """Event emitted during startup operations"""

    event_type: (
        str  # phase_start, phase_end, feature_loading, feature_loaded, feature_failed, rollback, startup_complete
    )
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    feature_name: Optional[str] = None
    phase: Optional[str] = None
    success: bool = True
    duration_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyNode:
    """Node in the dependency graph for topological sorting"""

    name: str
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    depth: int = 0  # Distance from root (no dependencies)
    in_degree: int = 0


class ResilientStartupManager:
    """
    Advanced manager for resilient startup system with enhanced capabilities.

    Features:
    - Dependency graph validation with cycle detection
    - Parallel loading of independent features
    - Rollback support for failed critical features
    - Comprehensive startup metrics and event system
    - Health monitoring integration
    - Graceful degradation with partial system operation
    - Configuration hot-reload support
    - Feature grouping and batch loading

    Usage:
        manager = ResilientStartupManager()

        # Register features with dependencies
        manager.register_feature(FeatureDefinition(
            name="database",
            loader=init_database,
            priority=FeaturePriority.CRITICAL,
            is_critical=True,
        ))

        manager.register_feature(FeatureDefinition(
            name="cache",
            loader=init_cache,
            priority=FeaturePriority.HIGH,
            dependencies=["database"],
        ))

        # Start with resilience and parallel loading
        success, status = await manager.startup(parallel=True)
    """

    def __init__(
        self,
        knowledge_base_path: Optional[Path] = None,
        usage_tracker_path: Optional[Path] = None,
        enable_parallel_loading: bool = True,
        max_parallel_loads: int = 5,
        enable_rollback: bool = True,
        startup_timeout: float = 300.0,  # 5 minute total startup timeout
        feature_timeout: float = 30.0,  # 30 second per-feature timeout
    ):
        self.features: Dict[str, FeatureState] = {}
        self.knowledge_base = ResolutionKnowledgeBase(knowledge_base_path)
        self.usage_tracker = FeatureUsageTracker(usage_tracker_path)
        self.healer = BackgroundHealer(self)

        # Configuration
        self.enable_parallel_loading = enable_parallel_loading
        self.max_parallel_loads = max_parallel_loads
        self.enable_rollback = enable_rollback
        self.startup_timeout = startup_timeout
        self.feature_timeout = feature_timeout

        # State management
        self._startup_complete = False
        self._startup_in_progress = False
        self._shutdown_in_progress = False
        self._lock = asyncio.Lock()

        # Dependency graph
        self._dependency_graph: Dict[str, DependencyNode] = {}
        self._load_order: List[List[str]] = []  # Batches for parallel loading

        # Event and notification handlers
        self._notification_handlers: List[Callable] = []
        self._event_handlers: List[Callable[[StartupEvent], Any]] = []

        # Metrics tracking
        self.metrics = StartupMetrics()

        # Rollback tracking
        self._loaded_features: List[str] = []  # Order of successful loads
        self._rollback_handlers: Dict[str, Callable] = {}

        # Health monitoring
        self._health_checks: Dict[str, Callable[[], bool]] = {}
        self._last_health_check: Optional[datetime] = None

    # =========================================================================
    # Feature Registration
    # =========================================================================

    def register_feature(
        self,
        definition: FeatureDefinition,
        rollback_handler: Optional[Callable] = None,
    ) -> bool:
        """
        Register a feature for managed loading.

        Args:
            definition: The feature definition
            rollback_handler: Optional async function to call on rollback

        Returns:
            True if registered successfully
        """
        if self._startup_in_progress:
            logger.warning(f"Cannot register feature '{definition.name}' during startup")
            return False

        self.features[definition.name] = FeatureState(definition=definition)

        if rollback_handler:
            self._rollback_handlers[definition.name] = rollback_handler

        # Update dependency graph
        self._update_dependency_graph(definition.name, definition.dependencies)

        logger.debug(
            f"Registered feature: {definition.name} "
            f"(priority={definition.priority.name}, "
            f"critical={definition.is_critical}, "
            f"deps={definition.dependencies})"
        )
        return True

    def unregister_feature(self, name: str) -> bool:
        """
        Unregister a feature.

        Args:
            name: Feature name to unregister

        Returns:
            True if unregistered successfully
        """
        if self._startup_in_progress:
            logger.warning(f"Cannot unregister feature '{name}' during startup")
            return False

        if name not in self.features:
            return False

        # Check if other features depend on this one
        dependents = self._get_dependents(name)
        if dependents:
            logger.warning(f"Cannot unregister '{name}': features {dependents} depend on it")
            return False

        del self.features[name]
        self._rollback_handlers.pop(name, None)
        self._dependency_graph.pop(name, None)

        # Recalculate load order
        self._calculate_load_order()

        logger.debug(f"Unregistered feature: {name}")
        return True

    def _update_dependency_graph(self, name: str, dependencies: List[str]):
        """Update the dependency graph when a feature is registered"""
        node = DependencyNode(name=name, dependencies=set(dependencies))
        self._dependency_graph[name] = node

        # Update dependents for all dependencies
        for dep in dependencies:
            if dep in self._dependency_graph:
                self._dependency_graph[dep].dependents.add(name)
            else:
                # Create placeholder node for unknown dependency
                self._dependency_graph[dep] = DependencyNode(name=dep, dependents={name})

        # Recalculate load order
        self._calculate_load_order()

    def _get_dependents(self, name: str) -> Set[str]:
        """Get all features that depend on the given feature"""
        node = self._dependency_graph.get(name)
        return node.dependents if node else set()

    # =========================================================================
    # Dependency Graph Validation
    # =========================================================================

    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Validate the dependency graph for issues.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check for missing dependencies
        for name, state in self.features.items():
            for dep in state.definition.dependencies:
                if dep not in self.features:
                    errors.append(f"Feature '{name}' depends on unregistered feature '{dep}'")

        # Check for cycles
        cycles = self._detect_cycles()
        for cycle in cycles:
            errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")

        # Check critical dependency chain
        for name, state in self.features.items():
            if state.definition.is_critical:
                for dep in state.definition.dependencies:
                    dep_state = self.features.get(dep)
                    if dep_state and not dep_state.definition.is_critical:
                        errors.append(f"Critical feature '{name}' depends on non-critical '{dep}'")

        return len(errors) == 0, errors

    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the dependency graph using DFS"""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node_name: str) -> bool:
            visited.add(node_name)
            rec_stack.add(node_name)
            path.append(node_name)

            node = self._dependency_graph.get(node_name)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        # Found cycle
                        cycle_start = path.index(dep)
                        cycles.append(path[cycle_start:] + [dep])
                        return True

            path.pop()
            rec_stack.remove(node_name)
            return False

        for name in self._dependency_graph:
            if name not in visited:
                dfs(name)

        return cycles

    def _calculate_load_order(self) -> List[List[str]]:
        """
        Calculate the optimal load order using topological sort.
        Groups features by depth for parallel loading.
        """
        if not self.features:
            self._load_order = []
            return self._load_order

        # Calculate in-degrees and depths
        in_degree = {name: 0 for name in self.features}
        depth = {name: 0 for name in self.features}

        for name, state in self.features.items():
            for dep in state.definition.dependencies:
                if dep in in_degree:
                    in_degree[name] += 1

        # Kahn's algorithm with depth tracking
        queue = []
        for name in self.features:
            if in_degree[name] == 0:
                queue.append(name)
                depth[name] = 0

        while queue:
            current = queue.pop(0)
            current_depth = depth[current]

            for name, state in self.features.items():
                if current in state.definition.dependencies:
                    in_degree[name] -= 1
                    depth[name] = max(depth[name], current_depth + 1)
                    if in_degree[name] == 0:
                        queue.append(name)

        # Group by depth for parallel loading
        max_depth = max(depth.values()) if depth else 0
        batches = [[] for _ in range(max_depth + 1)]

        for name in self.features:
            # Further sort by priority within each batch
            batches[depth[name]].append(name)

        # Sort each batch: critical first, then by priority
        for batch in batches:
            batch.sort(
                key=lambda n: (
                    0 if self.features[n].definition.is_critical else 1,
                    self.features[n].definition.priority.value,
                )
            )

        self._load_order = [b for b in batches if b]  # Remove empty batches
        return self._load_order

    def get_dependency_graph(self) -> Dict[str, Any]:
        """Get a representation of the dependency graph for visualization"""
        return {
            name: {
                "dependencies": list(node.dependencies),
                "dependents": list(node.dependents),
                "depth": node.depth,
            }
            for name, node in self._dependency_graph.items()
        }

    # =========================================================================
    # Event System
    # =========================================================================

    def add_notification_handler(self, handler: Callable) -> Callable[[], None]:
        """
        Add a handler for failure notifications.

        Returns:
            Function to remove the handler
        """
        self._notification_handlers.append(handler)

        def remove():
            if handler in self._notification_handlers:
                self._notification_handlers.remove(handler)

        return remove

    def add_event_handler(self, handler: Callable[[StartupEvent], Any]) -> Callable[[], None]:
        """
        Add an event handler for startup events.

        Args:
            handler: Callback function that receives StartupEvent

        Returns:
            Function to remove the handler
        """
        self._event_handlers.append(handler)

        def remove():
            if handler in self._event_handlers:
                self._event_handlers.remove(handler)

        return remove

    async def _emit_event(self, event: StartupEvent):
        """Emit a startup event to all handlers"""
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Event handler failed: {e}")

    # =========================================================================
    # Startup Sequence
    # =========================================================================

    async def startup(
        self,
        parallel: Optional[bool] = None,
        validate_first: bool = True,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Execute startup sequence with enhanced features.

        Args:
            parallel: Override parallel loading setting
            validate_first: Validate dependencies before starting

        Returns:
            Tuple of (success, status_dict) where:
            - success: True if all critical features loaded
            - status_dict: Feature name -> status message mapping
        """
        if self._startup_in_progress:
            logger.warning("Startup already in progress")
            return False, {"error": "Startup already in progress"}

        self._startup_in_progress = True
        use_parallel = parallel if parallel is not None else self.enable_parallel_loading

        # Initialize metrics
        self.metrics = StartupMetrics()
        self.metrics.startup_start_time = datetime.now(timezone.utc)
        self.metrics.total_features = len(self.features)
        self.metrics.critical_features = sum(1 for f in self.features.values() if f.definition.is_critical)
        self.metrics.non_critical_features = self.metrics.total_features - self.metrics.critical_features

        status: Dict[str, str] = {}
        critical_failures: List[str] = []
        non_critical_failures: List[str] = []

        try:
            # Emit startup start event
            await self._emit_event(
                StartupEvent(
                    event_type="startup_start",
                    metadata={"parallel": use_parallel, "total_features": len(self.features)},
                )
            )

            print("\n" + "=" * 80)
            print("🚀 Vega2.0 Resilient Startup Sequence")
            print("=" * 80 + "\n")

            # Phase 0: Validate dependencies
            if validate_first:
                phase_start = time.time()
                await self._emit_event(
                    StartupEvent(
                        event_type="phase_start",
                        phase="validation",
                    )
                )

                print("📋 Phase 0: Validating Dependencies")
                print("-" * 40)

                is_valid, errors = self.validate_dependencies()
                self.metrics.dependency_resolution_time_ms = (time.time() - phase_start) * 1000

                if not is_valid:
                    print("   ❌ Dependency validation failed:")
                    for error in errors:
                        print(f"      - {error}")
                    self._startup_in_progress = False
                    return False, {"error": "; ".join(errors)}

                print(f"   ✅ All dependencies valid ({len(self.features)} features)")
                self.metrics.phase_times["validation"] = (time.time() - phase_start) * 1000

            # Calculate load order
            self._calculate_load_order()
            self.metrics.parallel_batches = len(self._load_order)

            # Phase 1: Load critical features
            phase_start = time.time()
            await self._emit_event(
                StartupEvent(
                    event_type="phase_start",
                    phase="critical",
                )
            )

            print("\n📌 Phase 1: Loading CRITICAL Features")
            print("-" * 40)

            critical_names = [name for name, state in self.features.items() if state.definition.is_critical]

            if use_parallel:
                results = await self._load_features_parallel(critical_names, status, is_critical=True)
            else:
                results = await self._load_features_sequential(critical_names, status, is_critical=True)

            critical_failures = [name for name, success in results.items() if not success]
            self.metrics.phase_times["critical"] = (time.time() - phase_start) * 1000

            if critical_failures:
                print(f"\n🚨 CRITICAL FAILURE: {len(critical_failures)} core feature(s) failed")

                # Attempt rollback if enabled
                if self.enable_rollback and self._loaded_features:
                    print("🔄 Attempting rollback...")
                    await self._rollback()

                self._startup_in_progress = False
                self.metrics.startup_end_time = datetime.now(timezone.utc)
                return False, status

            # Phase 2: Load non-critical features
            phase_start = time.time()
            await self._emit_event(
                StartupEvent(
                    event_type="phase_start",
                    phase="non_critical",
                )
            )

            print("\n📌 Phase 2: Loading NON-CRITICAL Features")
            print("-" * 40)

            non_critical_names = [name for name, state in self.features.items() if not state.definition.is_critical]

            if use_parallel:
                results = await self._load_features_parallel(non_critical_names, status, is_critical=False)
            else:
                results = await self._load_features_sequential(non_critical_names, status, is_critical=False)

            non_critical_failures = [name for name, success in results.items() if not success]
            self.metrics.phase_times["non_critical"] = (time.time() - phase_start) * 1000

            # Phase 3: Start background healer if needed
            if non_critical_failures:
                print(f"\n🏥 Starting background healer for " f"{len(non_critical_failures)} failed feature(s)")
                await self.healer.start()

            self._startup_complete = True
            self.metrics.startup_end_time = datetime.now(timezone.utc)

            # Emit completion event
            await self._emit_event(
                StartupEvent(
                    event_type="startup_complete",
                    success=True,
                    duration_ms=self.metrics.total_startup_time_ms,
                    metadata={
                        "healthy": self.metrics.successful_loads,
                        "failed": self.metrics.failed_loads,
                        "degraded": self.metrics.degraded_loads,
                    },
                )
            )

            print("\n" + "=" * 80)
            if non_critical_failures:
                print(f"🌌 VEGA SYSTEM ONLINE (Degraded - " f"{len(non_critical_failures)} features healing)")
            else:
                print("🌌 VEGA SYSTEM ONLINE - All Features Operational")
            print(f"   Startup completed in {self.metrics.total_startup_time_ms:.0f}ms")
            print("=" * 80 + "\n")

            return True, status

        except asyncio.TimeoutError:
            logger.error("Startup timed out")
            if self.enable_rollback:
                await self._rollback()
            return False, {"error": "Startup timed out"}

        except Exception as e:
            logger.error(f"Startup failed with error: {e}", exc_info=True)
            if self.enable_rollback:
                await self._rollback()
            return False, {"error": str(e)}

        finally:
            self._startup_in_progress = False

    async def _load_features_sequential(
        self,
        feature_names: List[str],
        status: Dict[str, str],
        is_critical: bool,
    ) -> Dict[str, bool]:
        """Load features sequentially in dependency order"""
        results = {}

        # Sort by dependency order
        ordered = []
        for batch in self._load_order:
            for name in batch:
                if name in feature_names:
                    ordered.append(name)

        # Add any not in order (should not happen with valid graph)
        for name in feature_names:
            if name not in ordered:
                ordered.append(name)

        for name in ordered:
            feature_state = self.features[name]

            await self._emit_event(
                StartupEvent(
                    event_type="feature_loading",
                    feature_name=name,
                )
            )

            start_time = time.time()
            success, message = await self._load_feature(feature_state)
            load_time_ms = (time.time() - start_time) * 1000

            self.metrics.feature_load_times[name] = load_time_ms
            status[name] = message
            results[name] = success

            if success:
                self._loaded_features.append(name)
                self.metrics.successful_loads += 1
                print(f"   ✅ {name}: OK ({load_time_ms:.0f}ms)")

                await self._emit_event(
                    StartupEvent(
                        event_type="feature_loaded",
                        feature_name=name,
                        success=True,
                        duration_ms=load_time_ms,
                    )
                )
            else:
                if feature_state.status == FeatureStatus.DEGRADED:
                    self.metrics.degraded_loads += 1
                    print(f"   ⚠️  {name}: DEGRADED ({load_time_ms:.0f}ms)")
                else:
                    self.metrics.failed_loads += 1
                    marker = "CRITICAL" if is_critical else "healing"
                    print(f"   ❌ {name}: FAILED ({marker})")

                await self._emit_event(
                    StartupEvent(
                        event_type="feature_failed",
                        feature_name=name,
                        success=False,
                        duration_ms=load_time_ms,
                        error_message=message,
                    )
                )

        return results

    async def _load_features_parallel(
        self,
        feature_names: List[str],
        status: Dict[str, str],
        is_critical: bool,
    ) -> Dict[str, bool]:
        """Load features in parallel batches respecting dependencies"""
        results = {}
        feature_set = set(feature_names)

        for batch in self._load_order:
            # Filter to only features in our list
            batch_features = [name for name in batch if name in feature_set]
            if not batch_features:
                continue

            # Limit concurrent loads - create semaphore once per batch
            batch_semaphore = asyncio.Semaphore(self.max_parallel_loads)

            async def load_with_semaphore(name: str, sem: asyncio.Semaphore) -> Tuple[str, bool, str, float]:
                async with sem:
                    feature_state = self.features[name]

                    await self._emit_event(
                        StartupEvent(
                            event_type="feature_loading",
                            feature_name=name,
                        )
                    )

                    start_time = time.time()
                    try:
                        success, message = await asyncio.wait_for(
                            self._load_feature(feature_state),
                            timeout=self.feature_timeout,
                        )
                    except asyncio.TimeoutError:
                        feature_state.status = FeatureStatus.FAILED
                        feature_state.error = TimeoutError(f"Feature load timed out after {self.feature_timeout}s")
                        success = False
                        message = f"Timeout after {self.feature_timeout}s"

                    load_time_ms = (time.time() - start_time) * 1000
                    return name, success, message, load_time_ms

            # Load batch in parallel
            tasks = [load_with_semaphore(name, batch_semaphore) for name in batch_features]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, BaseException):
                    logger.error(f"Unexpected error in parallel load: {result}")
                    continue

                # Now we know result is a tuple
                if not isinstance(result, tuple) or len(result) != 4:
                    logger.error(f"Unexpected result type: {type(result)}")
                    continue

                name, success, message, load_time_ms = result
                feature_state = self.features[name]

                self.metrics.feature_load_times[name] = load_time_ms
                status[name] = message
                results[name] = success

                if success:
                    self._loaded_features.append(name)
                    self.metrics.successful_loads += 1
                    print(f"   ✅ {name}: OK ({load_time_ms:.0f}ms)")

                    await self._emit_event(
                        StartupEvent(
                            event_type="feature_loaded",
                            feature_name=name,
                            success=True,
                            duration_ms=load_time_ms,
                        )
                    )
                else:
                    if feature_state.status == FeatureStatus.DEGRADED:
                        self.metrics.degraded_loads += 1
                        print(f"   ⚠️  {name}: DEGRADED ({load_time_ms:.0f}ms)")
                    else:
                        self.metrics.failed_loads += 1
                        marker = "CRITICAL" if is_critical else "healing"
                        print(f"   ❌ {name}: FAILED ({marker})")

                    await self._emit_event(
                        StartupEvent(
                            event_type="feature_failed",
                            feature_name=name,
                            success=False,
                            duration_ms=load_time_ms,
                            error_message=message,
                        )
                    )

        return results

    # =========================================================================
    # Rollback Support
    # =========================================================================

    async def _rollback(self) -> bool:
        """
        Rollback loaded features in reverse order.

        Returns:
            True if rollback completed successfully
        """
        if not self._loaded_features:
            return True

        print("\n🔄 Rolling back loaded features...")
        await self._emit_event(
            StartupEvent(
                event_type="rollback",
                metadata={"features": list(self._loaded_features)},
            )
        )

        rollback_errors = []

        for name in reversed(self._loaded_features):
            try:
                handler = self._rollback_handlers.get(name)
                if handler:
                    if asyncio.iscoroutinefunction(handler):
                        await handler()
                    else:
                        handler()
                    print(f"   🔄 Rolled back: {name}")

                # Reset feature state
                state = self.features.get(name)
                if state:
                    state.status = FeatureStatus.UNKNOWN
                    state.load_time = None
                    state.error = None
                    state.error_traceback = None

            except Exception as e:
                logger.error(f"Rollback failed for '{name}': {e}")
                rollback_errors.append((name, str(e)))

        self.metrics.rollbacks_performed += 1
        self._loaded_features.clear()

        if rollback_errors:
            print(f"   ⚠️  {len(rollback_errors)} rollback error(s)")
            return False

        print("   ✅ Rollback complete")
        return True

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def shutdown(self, timeout: float = 30.0) -> bool:
        """
        Shutdown the manager and all components gracefully.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            True if shutdown completed gracefully
        """
        if self._shutdown_in_progress:
            logger.warning("Shutdown already in progress")
            return False

        self._shutdown_in_progress = True
        graceful = True

        try:
            print("\n🛑 Initiating graceful shutdown...")

            # Stop the healer first
            healer_stopped = await self.healer.stop(timeout=timeout / 2)
            if not healer_stopped:
                graceful = False
                logger.warning("Healer did not stop gracefully")

            # Save usage tracker data
            self.usage_tracker.save()

            # Save knowledge base
            self.knowledge_base.save()

            # Optionally rollback features if requested
            # (Usually not needed on normal shutdown)

            logger.info("Resilient startup manager shutdown complete")
            print("✅ Shutdown complete\n")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            graceful = False

        finally:
            self._shutdown_in_progress = False
            self._startup_complete = False

        return graceful

    # =========================================================================
    # Feature Loading
    # =========================================================================

    async def _load_feature(self, feature_state: FeatureState) -> Tuple[bool, str]:
        """Attempt to load a single feature with timeout and error handling"""
        feature_state.status = FeatureStatus.LOADING

        try:
            # Check dependencies first
            for dep_name in feature_state.definition.dependencies:
                dep_state = self.features.get(dep_name)
                if not dep_state or dep_state.status not in (
                    FeatureStatus.HEALTHY,
                    FeatureStatus.DEGRADED,
                ):
                    raise RuntimeError(f"Dependency '{dep_name}' not available")

            # Execute the loader
            await feature_state.definition.loader()

            feature_state.status = FeatureStatus.HEALTHY
            feature_state.load_time = datetime.now(timezone.utc)
            feature_state.error = None
            feature_state.error_traceback = None

            return True, "Loaded successfully"

        except Exception as e:
            feature_state.status = FeatureStatus.FAILED
            feature_state.error = e
            feature_state.error_traceback = traceback.format_exc()

            logger.warning(
                f"Feature '{feature_state.definition.name}' failed to load: {e}",
                exc_info=True,
            )

            # Try fallback if available
            if feature_state.definition.fallback:
                try:
                    await feature_state.definition.fallback()
                    feature_state.status = FeatureStatus.DEGRADED
                    return True, f"Loaded with fallback: {e}"
                except Exception as fallback_error:
                    logger.warning(f"Fallback also failed: {fallback_error}")

            return False, str(e)

    # =========================================================================
    # Healing Support
    # =========================================================================

    def get_failed_features(self) -> List[FeatureState]:
        """Get list of features that need healing"""
        return [fs for fs in self.features.values() if fs.status in (FeatureStatus.FAILED, FeatureStatus.DEGRADED)]

    async def attempt_heal(self, feature_state: FeatureState) -> bool:
        """
        Attempt to heal a failed feature.

        This method:
        1. Checks knowledge base for known resolutions
        2. Tries known resolutions first
        3. Falls back to generic strategies
        4. Records successful resolutions
        """
        async with self._lock:
            if feature_state.status not in (
                FeatureStatus.FAILED,
                FeatureStatus.DEGRADED,
            ):
                return True

            feature_state.status = FeatureStatus.HEALING
            feature_state.heal_attempts += 1
            feature_state.last_heal_attempt = datetime.now(timezone.utc)

            logger.info(
                f"🔧 Attempting to heal '{feature_state.definition.name}' "
                f"(attempt {feature_state.heal_attempts}/{feature_state.definition.max_heal_attempts})"
            )

            # Check knowledge base for known resolution
            if feature_state.error:
                known_resolution = self.knowledge_base.find_resolution(
                    feature_state.definition.name, feature_state.error
                )

                if known_resolution:
                    logger.info(f"📚 Found known resolution: {known_resolution.strategy_used}")
                    success = await self._apply_resolution(
                        feature_state,
                        known_resolution.strategy_used,
                        known_resolution.resolution_details,
                    )
                    if success:
                        self.knowledge_base.record_success(
                            feature_state.definition.name,
                            feature_state.error,
                            known_resolution.strategy_used,
                            known_resolution.resolution_details,
                        )
                        return True

            # Try each resolution strategy
            for strategy in feature_state.definition.resolution_strategies:
                success = await self._try_strategy(feature_state, strategy)
                if success:
                    return True

            # Failed to heal
            feature_state.status = FeatureStatus.FAILED
            logger.warning(
                f"⚠️  Could not heal '{feature_state.definition.name}' " f"(attempt {feature_state.heal_attempts})"
            )
            return False

    async def _try_strategy(self, feature_state: FeatureState, strategy: ResolutionStrategy) -> bool:
        """Try a specific resolution strategy"""
        try:
            details: Dict[str, Any] = {}

            if strategy == ResolutionStrategy.REIMPORT:
                # Clear any cached imports and try again
                import sys

                module_name = f"src.vega.core.{feature_state.definition.name}"
                modules_to_remove = [
                    m for m in sys.modules if m.startswith(module_name) or feature_state.definition.name in m
                ]
                for m in modules_to_remove:
                    del sys.modules[m]
                details = {"cleared_modules": modules_to_remove}

            elif strategy == ResolutionStrategy.RESET_STATE:
                # Wait a bit and try again (transient error)
                await asyncio.sleep(2.0)
                details = {"action": "waited_and_retry"}

            elif strategy == ResolutionStrategy.RECREATE_RESOURCE:
                # Try to recreate any resources
                details = {"action": "recreate_attempted"}

            elif strategy == ResolutionStrategy.FALLBACK_CONFIG:
                # Use fallback/default configuration
                details = {"action": "fallback_config"}

            # Try loading the feature again
            success, _ = await self._load_feature(feature_state)

            if success:
                # Record the successful resolution
                if feature_state.error:
                    self.knowledge_base.record_success(
                        feature_state.definition.name,
                        feature_state.error,
                        strategy.value,
                        details,
                    )
                logger.info(f"✅ Successfully healed '{feature_state.definition.name}' using {strategy.value}")
                await self._notify_healed(feature_state, strategy.value)
                return True

        except Exception as e:
            logger.debug(f"Strategy {strategy.value} failed: {e}")

        return False

    async def _apply_resolution(self, feature_state: FeatureState, strategy: str, details: Dict[str, Any]) -> bool:
        """Apply a known resolution from the knowledge base"""
        try:
            # Apply the resolution based on stored details
            if strategy == ResolutionStrategy.REIMPORT.value:
                import sys

                for module in details.get("cleared_modules", []):
                    if module in sys.modules:
                        del sys.modules[module]

            # Try loading
            success, _ = await self._load_feature(feature_state)
            if success:
                logger.info(f"✅ Known resolution worked for '{feature_state.definition.name}'")
                await self._notify_healed(feature_state, strategy)
            return success

        except Exception as e:
            logger.debug(f"Known resolution failed: {e}")
            return False

    # =========================================================================
    # Notifications
    # =========================================================================

    async def notify_permanent_failure(self, feature_state: FeatureState):
        """Notify about a permanent failure with enhanced detail"""
        message = (
            f"🚨 PERMANENT FAILURE: Feature '{feature_state.definition.name}' "
            f"could not be healed after {feature_state.heal_attempts} attempts.\n"
            f"Error: {feature_state.error}\n"
            f"Manual intervention required."
        )

        logger.error(message)
        print(f"\n{message}\n")

        # Emit event
        await self._emit_event(
            StartupEvent(
                event_type="permanent_failure",
                feature_name=feature_state.definition.name,
                success=False,
                error_message=str(feature_state.error),
                metadata={
                    "heal_attempts": feature_state.heal_attempts,
                    "traceback": feature_state.error_traceback,
                },
            )
        )

        # Call notification handlers
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler("permanent_failure", feature_state)
                else:
                    handler("permanent_failure", feature_state)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    async def _notify_healed(self, feature_state: FeatureState, strategy: str):
        """Notify about successful healing with enhanced detail"""
        message = (
            f"🏥 HEALED: Feature '{feature_state.definition.name}' "
            f"was successfully recovered using strategy '{strategy}'"
        )

        logger.info(message)
        print(f"\n{message}\n")

        # Emit event
        await self._emit_event(
            StartupEvent(
                event_type="healed",
                feature_name=feature_state.definition.name,
                success=True,
                metadata={
                    "strategy": strategy,
                    "heal_attempts": feature_state.heal_attempts,
                },
            )
        )

        # Call notification handlers
        for handler in self._notification_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler("healed", feature_state, strategy)
                else:
                    handler("healed", feature_state, strategy)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def record_feature_usage(self, feature_name: str):
        """Record that a feature was used (call this from feature code)"""
        self.usage_tracker.record_usage(feature_name)

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    def register_health_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function"""
        self._health_checks[name] = check_func

    def unregister_health_check(self, name: str) -> bool:
        """Unregister a health check"""
        if name in self._health_checks:
            del self._health_checks[name]
            return True
        return False

    def run_health_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        for name, check_func in self._health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.warning(f"Health check '{name}' failed: {e}")
                results[name] = False
        return results

    def is_healthy(self) -> bool:
        """Check if the system is healthy overall"""
        if not self._startup_complete:
            return False

        # Check feature health
        failed_critical = any(
            state.status in (FeatureStatus.FAILED, FeatureStatus.PERMANENTLY_FAILED)
            for state in self.features.values()
            if state.definition.is_critical
        )
        if failed_critical:
            return False

        # Run health checks
        health_results = self.run_health_checks()
        return all(health_results.values()) if health_results else True

    # =========================================================================
    # Status and Metrics
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all features and the system"""
        now = datetime.now(timezone.utc)

        return {
            "startup_complete": self._startup_complete,
            "startup_in_progress": self._startup_in_progress,
            "is_healthy": self.is_healthy(),
            "healer": {
                "running": self.healer.is_running,
                "paused": self.healer.is_paused,
                "metrics": self.healer.get_metrics_summary(),
            },
            "features": {
                name: {
                    "status": state.status.value,
                    "priority": state.definition.priority.name,
                    "is_critical": state.definition.is_critical,
                    "heal_attempts": state.heal_attempts,
                    "usage_count": self.usage_tracker.get_feature_stats(name).get("total_usage", 0),
                    "error": str(state.error) if state.error else None,
                    "load_time": (state.load_time.isoformat() if state.load_time else None),
                    "dependencies": state.definition.dependencies,
                }
                for name, state in self.features.items()
            },
            "metrics": {
                "total_startup_time_ms": self.metrics.total_startup_time_ms,
                "success_rate": round(self.metrics.success_rate * 100, 2),
                "successful_loads": self.metrics.successful_loads,
                "failed_loads": self.metrics.failed_loads,
                "degraded_loads": self.metrics.degraded_loads,
                "parallel_batches": self.metrics.parallel_batches,
                "rollbacks_performed": self.metrics.rollbacks_performed,
                "average_load_time_ms": round(self.metrics.average_load_time_ms, 2),
                "phase_times": self.metrics.phase_times,
            },
            "dependency_graph": self.get_dependency_graph(),
            "knowledge_base": {
                "total_entries": self.knowledge_base.get_statistics().get("total_resolutions", 0),
                "features_with_resolutions": self.knowledge_base.get_statistics().get("features_with_resolutions", 0),
            },
            "timestamp": now.isoformat(),
        }

    def get_feature_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific feature"""
        state = self.features.get(name)
        if not state:
            return None

        return {
            "name": name,
            "status": state.status.value,
            "priority": state.definition.priority.name,
            "is_critical": state.definition.is_critical,
            "dependencies": state.definition.dependencies,
            "heal_attempts": state.heal_attempts,
            "max_heal_attempts": state.definition.max_heal_attempts,
            "last_heal_attempt": (state.last_heal_attempt.isoformat() if state.last_heal_attempt else None),
            "load_time": state.load_time.isoformat() if state.load_time else None,
            "error": str(state.error) if state.error else None,
            "error_traceback": state.error_traceback,
            "usage_stats": self.usage_tracker.get_feature_stats(name),
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            "startup": {
                "total_time_ms": self.metrics.total_startup_time_ms,
                "success_rate": self.metrics.success_rate,
                "feature_load_times": self.metrics.feature_load_times,
                "phase_times": self.metrics.phase_times,
                "parallel_batches": self.metrics.parallel_batches,
            },
            "healing": self.healer.get_metrics_summary(),
            "usage": self.usage_tracker.get_overall_stats(),
            "knowledge_base": self.knowledge_base.get_statistics(),
        }


# Global instance (created lazily)
_resilient_manager: Optional[ResilientStartupManager] = None


def get_resilient_manager() -> ResilientStartupManager:
    """Get or create the global resilient startup manager"""
    global _resilient_manager
    if _resilient_manager is None:
        _resilient_manager = ResilientStartupManager(
            knowledge_base_path=Path("data/resolution_knowledge.json"),
            usage_tracker_path=Path("data/feature_usage.json"),
        )
    return _resilient_manager


# Decorator for tracking feature usage
def track_usage(feature_name: str):
    """Decorator to track usage of a feature"""

    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            get_resilient_manager().record_feature_usage(feature_name)
            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            get_resilient_manager().record_feature_usage(feature_name)
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ==============================================================================
# Compatibility Layer for API and Tests
# ==============================================================================
# The following classes provide a simplified interface that matches what the
# app.py startup system and tests expect. This allows flexible usage patterns.


class FeatureCategory(Enum):
    """Simplified category enum for API compatibility"""

    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"
    OPTIONAL = "optional"

    @classmethod
    def from_priority(cls, priority: FeaturePriority, is_critical: bool) -> "FeatureCategory":
        """Convert from internal FeaturePriority to FeatureCategory"""
        if is_critical:
            return cls.CRITICAL
        mapping = {
            FeaturePriority.CRITICAL: cls.CRITICAL,
            FeaturePriority.HIGH: cls.HIGH,
            FeaturePriority.MEDIUM: cls.STANDARD,
            FeaturePriority.LOW: cls.LOW,
            FeaturePriority.BACKGROUND: cls.OPTIONAL,
        }
        return mapping.get(priority, cls.STANDARD)


@dataclass
class RepairStrategy:
    """Simplified repair strategy for API compatibility"""

    name: str
    repair_func: Callable[[], Coroutine[Any, Any, bool]]
    description: str = ""
    success_rate: float = 0.0


@dataclass
class StartupFeature:
    """Simplified feature definition for API compatibility"""

    name: str
    category: FeatureCategory
    init_func: Callable[[], Coroutine[Any, Any, Any]]
    health_check_func: Optional[Callable[[], Coroutine[Any, Any, bool]]] = None
    repair_strategies: List[RepairStrategy] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_repair_attempts: int = 3
    description: str = ""


class SimplifiedStartupManager:
    """
    Enhanced simplified startup manager that provides the interface expected by app.py.

    Features:
    - Async health checks with configurable intervals
    - Graceful degradation support
    - Comprehensive metric collection
    - API-ready status endpoints
    - Event callbacks for monitoring
    - Integration with the full ResilientStartupManager

    This provides a simpler API while delegating complex functionality to the
    underlying ResilientStartupManager.
    """

    def __init__(
        self,
        health_check_interval: float = 60.0,
        enable_health_checks: bool = True,
        auto_repair: bool = True,
    ):
        """
        Initialize the simplified startup manager.

        Args:
            health_check_interval: Interval between health checks in seconds
            enable_health_checks: Whether to run periodic health checks
            auto_repair: Whether to automatically queue failed features for repair
        """
        self.features: Dict[str, StartupFeature] = {}
        self.feature_status: Dict[str, Dict[str, Any]] = {}
        self.feature_usage: Dict[str, int] = defaultdict(int)
        self.repair_queue: asyncio.Queue = asyncio.Queue()
        self.pending_repairs: Set[str] = set()
        self.repair_knowledge_base: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.health_check_interval = health_check_interval
        self.enable_health_checks = enable_health_checks
        self.auto_repair = auto_repair

        # State
        self.is_running = False
        self._startup_time: Optional[datetime] = None
        self._last_health_check: Optional[datetime] = None

        # Background tasks
        self._healer_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Internal manager integration
        self._internal_manager = get_resilient_manager()

        # Event callbacks
        self._on_feature_healthy: List[Callable[[str], Any]] = []
        self._on_feature_failed: List[Callable[[str, str], Any]] = []
        self._on_feature_repaired: List[Callable[[str], Any]] = []

        # Metrics
        self._metrics = {
            "startup_duration_ms": 0.0,
            "total_health_checks": 0,
            "failed_health_checks": 0,
            "repairs_attempted": 0,
            "repairs_successful": 0,
            "uptime_seconds": 0.0,
        }

    # =========================================================================
    # Event Registration
    # =========================================================================

    def on_feature_healthy(self, callback: Callable[[str], Any]) -> Callable[[], None]:
        """Register callback for when a feature becomes healthy"""
        self._on_feature_healthy.append(callback)
        return lambda: self._on_feature_healthy.remove(callback)

    def on_feature_failed(self, callback: Callable[[str, str], Any]) -> Callable[[], None]:
        """Register callback for when a feature fails (name, error)"""
        self._on_feature_failed.append(callback)
        return lambda: self._on_feature_failed.remove(callback)

    def on_feature_repaired(self, callback: Callable[[str], Any]) -> Callable[[], None]:
        """Register callback for when a feature is repaired"""
        self._on_feature_repaired.append(callback)
        return lambda: self._on_feature_repaired.remove(callback)

    async def _emit_feature_healthy(self, name: str):
        """Emit feature healthy event"""
        for callback in self._on_feature_healthy:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(name)
                else:
                    callback(name)
            except Exception as e:
                logger.error(f"Feature healthy callback failed: {e}")

    async def _emit_feature_failed(self, name: str, error: str):
        """Emit feature failed event"""
        for callback in self._on_feature_failed:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(name, error)
                else:
                    callback(name, error)
            except Exception as e:
                logger.error(f"Feature failed callback failed: {e}")

    async def _emit_feature_repaired(self, name: str):
        """Emit feature repaired event"""
        for callback in self._on_feature_repaired:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(name)
                else:
                    callback(name)
            except Exception as e:
                logger.error(f"Feature repaired callback failed: {e}")

    # =========================================================================
    # Feature Registration
    # =========================================================================

    def register_feature(self, feature: StartupFeature):
        """Register a feature with the simplified manager"""
        self.features[feature.name] = feature
        self.feature_status[feature.name] = {
            "status": "unknown",
            "healthy": False,
            "error": None,
            "repair_attempts": 0,
            "last_health_check": None,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.debug(f"Registered feature: {feature.name}")

    def unregister_feature(self, name: str) -> bool:
        """Unregister a feature"""
        if name not in self.features:
            return False

        del self.features[name]
        self.feature_status.pop(name, None)
        self.pending_repairs.discard(name)
        return True

    # =========================================================================
    # Startup Sequence
    # =========================================================================

    async def startup_sequence(self) -> bool:
        """
        Execute the startup sequence.
        Returns True if all critical features succeeded.
        """
        self.is_running = True
        self._startup_time = datetime.now(timezone.utc)
        start_time = time.time()
        all_critical_ok = True

        # Sort features: critical first, then by dependencies
        sorted_features = self._sort_by_dependencies()

        for feature in sorted_features:
            success = await self._init_feature(feature)

            if success:
                await self._emit_feature_healthy(feature.name)
            else:
                await self._emit_feature_failed(
                    feature.name,
                    self.feature_status[feature.name].get("error", "Unknown error"),
                )

            if not success and feature.category == FeatureCategory.CRITICAL:
                all_critical_ok = False
                logger.error(f"Critical feature '{feature.name}' failed - startup aborted")
                break
            elif not success and self.auto_repair:
                # Queue non-critical failures for background repair
                await self.queue_repair(feature.name)

        # Record startup duration
        self._metrics["startup_duration_ms"] = (time.time() - start_time) * 1000

        # Start background tasks if we have features
        if all_critical_ok:
            if self.pending_repairs:
                self._healer_task = asyncio.create_task(self._background_healer())

            if self.enable_health_checks:
                self._health_check_task = asyncio.create_task(self._health_check_loop())

        return all_critical_ok

    async def _init_feature(self, feature: StartupFeature) -> bool:
        """Initialize a single feature"""
        self.feature_status[feature.name]["status"] = "loading"

        try:
            # Check dependencies
            for dep in feature.dependencies:
                dep_status = self.feature_status.get(dep, {})
                if dep_status.get("status") != "healthy":
                    raise RuntimeError(f"Dependency '{dep}' not healthy")

            # Run init with timeout
            await asyncio.wait_for(feature.init_func(), timeout=feature.timeout)

            self.feature_status[feature.name] = {
                "status": "healthy",
                "healthy": True,
                "error": None,
                "repair_attempts": 0,
                "last_health_check": datetime.now(timezone.utc).isoformat(),
                "registered_at": self.feature_status[feature.name].get("registered_at"),
            }
            return True

        except asyncio.TimeoutError:
            error_msg = f"Timeout after {feature.timeout}s"
            self.feature_status[feature.name] = {
                "status": "failed",
                "healthy": False,
                "error": error_msg,
                "repair_attempts": 0,
                "registered_at": self.feature_status[feature.name].get("registered_at"),
            }
            logger.warning(f"Feature '{feature.name}' timed out")
            return False

        except Exception as e:
            self.feature_status[feature.name] = {
                "status": "failed",
                "healthy": False,
                "error": str(e),
                "repair_attempts": 0,
                "registered_at": self.feature_status[feature.name].get("registered_at"),
            }
            logger.warning(f"Feature '{feature.name}' failed: {e}")
            return False

    def _sort_by_dependencies(self) -> List[StartupFeature]:
        """Sort features so dependencies come first"""
        sorted_list: List[StartupFeature] = []
        visited: Set[str] = set()

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            feature = self.features.get(name)
            if feature:
                for dep in feature.dependencies:
                    visit(dep)
                sorted_list.append(feature)

        # Visit critical features first
        for name, feature in self.features.items():
            if feature.category == FeatureCategory.CRITICAL:
                visit(name)

        # Then high priority
        for name, feature in self.features.items():
            if feature.category == FeatureCategory.HIGH:
                visit(name)

        # Then the rest
        for name in self.features:
            visit(name)

        return sorted_list

    # =========================================================================
    # Health Checks
    # =========================================================================

    async def _health_check_loop(self):
        """Background loop that runs periodic health checks"""
        logger.info("Health check loop started")

        while self.is_running:
            try:
                await asyncio.sleep(self.health_check_interval)

                if not self.is_running:
                    break

                await self.run_health_checks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

        logger.info("Health check loop stopped")

    async def run_health_checks(self) -> Dict[str, bool]:
        """
        Run health checks on all features that have health check functions.

        Returns:
            Dict mapping feature name to health status
        """
        results = {}
        self._last_health_check = datetime.now(timezone.utc)
        self._metrics["total_health_checks"] += 1

        for name, feature in self.features.items():
            if not feature.health_check_func:
                # No health check, assume healthy if status is healthy
                results[name] = self.feature_status[name].get("healthy", False)
                continue

            try:
                is_healthy = await asyncio.wait_for(
                    feature.health_check_func(),
                    timeout=min(feature.timeout, 10.0),
                )

                results[name] = is_healthy
                self.feature_status[name]["last_health_check"] = datetime.now(timezone.utc).isoformat()

                if is_healthy:
                    if self.feature_status[name]["status"] != "healthy":
                        # Feature recovered
                        self.feature_status[name]["status"] = "healthy"
                        self.feature_status[name]["healthy"] = True
                        self.feature_status[name]["error"] = None
                        await self._emit_feature_healthy(name)
                else:
                    if self.feature_status[name]["status"] == "healthy":
                        # Feature became unhealthy
                        self.feature_status[name]["status"] = "degraded"
                        self.feature_status[name]["healthy"] = False
                        self._metrics["failed_health_checks"] += 1

                        if self.auto_repair:
                            await self.queue_repair(name)

                        await self._emit_feature_failed(name, "Health check failed")

            except asyncio.TimeoutError:
                results[name] = False
                self.feature_status[name]["healthy"] = False
                self._metrics["failed_health_checks"] += 1
                logger.warning(f"Health check timeout for '{name}'")

            except Exception as e:
                results[name] = False
                self.feature_status[name]["healthy"] = False
                self._metrics["failed_health_checks"] += 1
                logger.warning(f"Health check error for '{name}': {e}")

        return results

    async def check_feature_health(self, name: str) -> bool:
        """Check health of a specific feature"""
        feature = self.features.get(name)
        if not feature:
            return False

        if not feature.health_check_func:
            return self.feature_status[name].get("healthy", False)

        try:
            is_healthy = await asyncio.wait_for(
                feature.health_check_func(),
                timeout=min(feature.timeout, 10.0),
            )
            self.feature_status[name]["last_health_check"] = datetime.now(timezone.utc).isoformat()
            return is_healthy
        except Exception as e:
            logger.warning(f"Health check error for '{name}': {e}")
            return False

    # =========================================================================
    # Repair Queue
    # =========================================================================

    async def queue_repair(self, feature_name: str):
        """Queue a feature for background repair"""
        if feature_name not in self.pending_repairs:
            self.pending_repairs.add(feature_name)
            priority = self.calculate_repair_priority(feature_name)
            await self.repair_queue.put((-priority, feature_name))  # Negative for max-heap behavior
            logger.info(f"Queued '{feature_name}' for repair (priority: {priority})")

    def calculate_repair_priority(self, feature_name: str) -> float:
        """Calculate repair priority based on usage and category"""
        feature = self.features.get(feature_name)
        if not feature:
            return 0.0

        # Base priority from category
        category_scores = {
            FeatureCategory.CRITICAL: 100,
            FeatureCategory.HIGH: 80,
            FeatureCategory.STANDARD: 50,
            FeatureCategory.LOW: 20,
            FeatureCategory.OPTIONAL: 10,
        }
        score = category_scores.get(feature.category, 50)

        # Add usage bonus
        usage = self.feature_usage.get(feature_name, 0)
        import math

        score += math.log1p(usage) * 5

        return score

    async def _background_healer(self):
        """Background task that attempts to repair failed features"""
        logger.info("Background healer started")

        while self.is_running and self.pending_repairs:
            try:
                # Get highest priority item
                _, feature_name = await asyncio.wait_for(self.repair_queue.get(), timeout=60.0)

                if feature_name not in self.pending_repairs:
                    continue

                feature = self.features.get(feature_name)
                if not feature:
                    self.pending_repairs.discard(feature_name)
                    continue

                status = self.feature_status.get(feature_name, {})
                attempts = status.get("repair_attempts", 0)

                if attempts >= feature.max_repair_attempts:
                    # Give up and notify
                    self.pending_repairs.discard(feature_name)
                    logger.error(f"🚨 Giving up on '{feature_name}' after {attempts} repair attempts")
                    continue

                # Update attempt count
                self.feature_status[feature_name]["repair_attempts"] = attempts + 1
                self._metrics["repairs_attempted"] += 1

                # Try repair strategies
                repaired = False
                error_pattern = status.get("error", "")

                # Check knowledge base first
                best_strategy = self.get_best_repair_strategy(feature_name, error_pattern)
                if best_strategy:
                    try:
                        repaired = await best_strategy.repair_func()
                        if repaired:
                            # Verify with init
                            repaired = await self._init_feature(feature)
                    except Exception as e:
                        logger.debug(f"Knowledge base strategy failed: {e}")

                # Try other strategies
                if not repaired:
                    for strategy in feature.repair_strategies:
                        try:
                            if await strategy.repair_func():
                                repaired = await self._init_feature(feature)
                                if repaired:
                                    # Add to knowledge base
                                    self.add_to_knowledge_base(feature_name, strategy.name, error_pattern)
                                    break
                        except Exception as e:
                            logger.debug(f"Repair strategy '{strategy.name}' failed: {e}")

                if repaired:
                    self.pending_repairs.discard(feature_name)
                    self._metrics["repairs_successful"] += 1
                    logger.info(f"✅ Successfully repaired '{feature_name}'")
                    await self._emit_feature_repaired(feature_name)
                else:
                    # Re-queue with backoff
                    await asyncio.sleep(min(30 * (2**attempts), 300))  # Exponential backoff
                    if feature_name in self.pending_repairs:
                        await self.repair_queue.put(
                            (
                                -self.calculate_repair_priority(feature_name),
                                feature_name,
                            )
                        )

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background healer error: {e}")
                await asyncio.sleep(10)

        logger.info("Background healer stopped")

    # =========================================================================
    # Knowledge Base
    # =========================================================================

    def record_feature_usage(self, feature_name: str):
        """Record that a feature was used"""
        self.feature_usage[feature_name] = self.feature_usage.get(feature_name, 0) + 1
        self._internal_manager.record_feature_usage(feature_name)

    def add_to_knowledge_base(self, feature_name: str, strategy_name: str, error_pattern: str):
        """Add a successful repair strategy to the knowledge base"""
        self.repair_knowledge_base[feature_name] = {
            "strategy": strategy_name,
            "error_pattern": error_pattern,
            "success_count": self.repair_knowledge_base.get(feature_name, {}).get("success_count", 0) + 1,
            "last_used": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(f"📚 Added to knowledge base: {feature_name} -> {strategy_name}")

    def get_best_repair_strategy(self, feature_name: str, error_pattern: str) -> Optional[RepairStrategy]:
        """Get the best repair strategy for a feature based on knowledge base"""
        kb_entry = self.repair_knowledge_base.get(feature_name)
        if not kb_entry:
            return None

        # Check if error pattern matches
        if kb_entry.get("error_pattern", "") in error_pattern:
            feature = self.features.get(feature_name)
            if feature:
                for strategy in feature.repair_strategies:
                    if strategy.name == kb_entry["strategy"]:
                        return strategy

        return None

    # =========================================================================
    # Status and Metrics
    # =========================================================================

    def get_status_summary(self) -> Dict[str, Any]:
        """Get a summary of all feature statuses"""
        healthy = []
        degraded = []
        failed = []

        for name, status in self.feature_status.items():
            if status.get("healthy"):
                healthy.append(name)
            elif status.get("status") == "failed":
                failed.append(name)
            else:
                degraded.append(name)

        # Calculate uptime
        uptime = 0.0
        if self._startup_time:
            uptime = (datetime.now(timezone.utc) - self._startup_time).total_seconds()
        self._metrics["uptime_seconds"] = uptime

        return {
            "healthy_features": healthy,
            "degraded_features": degraded,
            "failed_features": failed,
            "total_features": len(self.features),
            "is_running": self.is_running,
            "pending_repairs": len(self.pending_repairs),
            "last_health_check": (self._last_health_check.isoformat() if self._last_health_check else None),
        }

    def get_feature_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific feature"""
        if name not in self.features:
            return None

        feature = self.features[name]
        status = self.feature_status[name].copy()
        status["name"] = name
        status["category"] = feature.category.value
        status["dependencies"] = feature.dependencies
        status["usage_count"] = self.feature_usage.get(name, 0)
        return status

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        if self._startup_time:
            self._metrics["uptime_seconds"] = (datetime.now(timezone.utc) - self._startup_time).total_seconds()

        return {
            **self._metrics,
            "repair_success_rate": (self._metrics["repairs_successful"] / max(1, self._metrics["repairs_attempted"])),
            "health_check_failure_rate": (
                self._metrics["failed_health_checks"] / max(1, self._metrics["total_health_checks"])
            ),
        }

    def is_healthy(self) -> bool:
        """Check if the system is healthy overall (all critical features healthy)"""
        for name, feature in self.features.items():
            if feature.category == FeatureCategory.CRITICAL:
                if not self.feature_status[name].get("healthy", False):
                    return False
        return True

    # =========================================================================
    # Shutdown
    # =========================================================================

    async def shutdown(self, timeout: float = 10.0):
        """Shutdown the manager gracefully"""
        self.is_running = False

        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await asyncio.wait_for(self._health_check_task, timeout=timeout / 2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Cancel healer task
        if self._healer_task:
            self._healer_task.cancel()
            try:
                await asyncio.wait_for(self._healer_task, timeout=timeout / 2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("Simplified startup manager shutdown complete")


# =============================================================================
# UTILITY FUNCTIONS AND FACTORY METHODS
# =============================================================================


class StartupDiagnostics:
    """
    Comprehensive diagnostics and health analysis for startup systems.

    Provides deep analysis of startup performance, identifies bottlenecks,
    generates recommendations, and exports diagnostic reports.
    """

    def __init__(
        self,
        manager: Optional[ResilientStartupManager] = None,
        simplified_manager: Optional[SimplifiedStartupManager] = None,
    ):
        self._manager = manager
        self._simplified_manager = simplified_manager
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_ttl = 60.0  # Cache analysis for 60 seconds
        self._last_analysis: Optional[datetime] = None

    def analyze_startup_performance(self) -> Dict[str, Any]:
        """
        Perform comprehensive startup performance analysis.

        Returns:
            Detailed analysis including bottlenecks, recommendations,
            and performance metrics.
        """
        now = datetime.now(timezone.utc)

        # Check cache validity
        if (
            self._last_analysis
            and (now - self._last_analysis).total_seconds() < self._cache_ttl
            and self._analysis_cache
        ):
            return self._analysis_cache

        analysis: Dict[str, Any] = {
            "timestamp": now.isoformat(),
            "overall_health": "unknown",
            "bottlenecks": [],
            "recommendations": [],
            "performance_grade": "N/A",
            "metrics_summary": {},
            "dependency_analysis": {},
            "resource_analysis": {},
        }

        try:
            if self._manager:
                analysis.update(self._analyze_resilient_manager())
            elif self._simplified_manager:
                analysis.update(self._analyze_simplified_manager())
            else:
                analysis["error"] = "No manager configured"
                return analysis

            # Calculate overall grade
            analysis["performance_grade"] = self._calculate_grade(analysis)
            analysis["overall_health"] = self._determine_health(analysis)

            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)

            # Cache results
            self._analysis_cache = analysis
            self._last_analysis = now

        except Exception as e:
            logger.exception("Error during startup analysis")
            analysis["error"] = str(e)

        return analysis

    def _analyze_resilient_manager(self) -> Dict[str, Any]:
        """Analyze ResilientStartupManager"""
        if not self._manager:
            return {}

        metrics = self._manager.get_metrics()

        # Identify bottlenecks
        bottlenecks = []
        feature_times = metrics.get("feature_load_times", {})
        if feature_times:
            avg_time = sum(feature_times.values()) / len(feature_times)
            for name, load_time in feature_times.items():
                if load_time > avg_time * 2:  # 2x average is a bottleneck
                    bottlenecks.append(
                        {
                            "type": "slow_feature",
                            "feature": name,
                            "load_time": load_time,
                            "threshold": avg_time * 2,
                            "severity": "high" if load_time > avg_time * 3 else "medium",
                        }
                    )

        # Analyze retry patterns
        retry_features = [name for name, count in metrics.get("retry_counts", {}).items() if count > 0]
        if retry_features:
            bottlenecks.append(
                {
                    "type": "retry_prone_features",
                    "features": retry_features,
                    "severity": "medium",
                }
            )

        # Analyze rollbacks
        rollback_count = metrics.get("rollback_count", 0)
        if rollback_count > 0:
            bottlenecks.append(
                {
                    "type": "rollbacks_occurred",
                    "count": rollback_count,
                    "severity": "high",
                }
            )

        # Dependency analysis
        dep_analysis = self._analyze_dependencies()

        return {
            "bottlenecks": bottlenecks,
            "metrics_summary": {
                "total_startup_time": metrics.get("total_startup_time", 0),
                "features_loaded": metrics.get("features_loaded", 0),
                "features_failed": metrics.get("features_failed", 0),
                "total_retries": sum(metrics.get("retry_counts", {}).values()),
                "rollbacks": rollback_count,
                "parallel_loads": metrics.get("parallel_loads", 0),
            },
            "dependency_analysis": dep_analysis,
        }

    def _analyze_simplified_manager(self) -> Dict[str, Any]:
        """Analyze SimplifiedStartupManager"""
        if not self._simplified_manager:
            return {}

        metrics = self._simplified_manager.get_metrics()
        status = self._simplified_manager.get_status_summary()

        bottlenecks = []

        # Check for failed features
        if status.get("failed_features"):
            bottlenecks.append(
                {
                    "type": "failed_features",
                    "features": status["failed_features"],
                    "severity": "critical",
                }
            )

        # Check repair queue
        if status.get("pending_repairs", 0) > 3:
            bottlenecks.append(
                {
                    "type": "repair_queue_backup",
                    "count": status["pending_repairs"],
                    "severity": "high",
                }
            )

        # Check health check failures
        failure_rate = metrics.get("health_check_failure_rate", 0)
        if failure_rate > 0.1:  # More than 10% failure rate
            bottlenecks.append(
                {
                    "type": "health_check_failures",
                    "failure_rate": failure_rate,
                    "severity": "high" if failure_rate > 0.3 else "medium",
                }
            )

        return {
            "bottlenecks": bottlenecks,
            "metrics_summary": {
                "total_features": status.get("total_features", 0),
                "healthy_features": len(status.get("healthy_features", [])),
                "degraded_features": len(status.get("degraded_features", [])),
                "failed_features": len(status.get("failed_features", [])),
                "repairs_attempted": metrics.get("repairs_attempted", 0),
                "repairs_successful": metrics.get("repairs_successful", 0),
                "uptime_seconds": metrics.get("uptime_seconds", 0),
            },
        }

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze dependency graph for potential issues"""
        if not self._manager:
            return {}

        dep_graph = {}
        reverse_deps: Dict[str, List[str]] = {}

        # Access through public features attribute
        for name, feature_state in self._manager.features.items():
            definition = feature_state.definition
            dep_graph[name] = definition.dependencies
            for dep in definition.dependencies:
                if dep not in reverse_deps:
                    reverse_deps[dep] = []
                reverse_deps[dep].append(name)

        # Find critical path (most dependencies)
        critical_features = sorted(
            reverse_deps.keys(),
            key=lambda x: len(reverse_deps[x]),
            reverse=True,
        )[:5]

        # Find orphan features (no dependents)
        orphans = [name for name in dep_graph if name not in reverse_deps]

        # Find deep dependency chains
        max_depth = 0

        def get_depth(name: str, visited: Optional[set] = None) -> int:
            if visited is None:
                visited = set()
            if name in visited:
                return 0  # Cycle detected
            visited.add(name)

            deps = dep_graph.get(name, [])
            if not deps:
                return 1
            return 1 + max(get_depth(d, visited.copy()) for d in deps)

        for name in dep_graph:
            depth = get_depth(name)
            if depth > max_depth:
                max_depth = depth

        return {
            "critical_features": critical_features,
            "orphan_features": orphans[:10],  # Limit output
            "max_dependency_depth": max_depth,
            "total_dependencies": sum(len(deps) for deps in dep_graph.values()),
        }

    def _calculate_grade(self, analysis: Dict[str, Any]) -> str:
        """Calculate performance grade A-F"""
        score = 100

        bottlenecks = analysis.get("bottlenecks", [])
        for bottleneck in bottlenecks:
            severity = bottleneck.get("severity", "low")
            if severity == "critical":
                score -= 25
            elif severity == "high":
                score -= 15
            elif severity == "medium":
                score -= 8
            else:
                score -= 3

        # Factor in metrics
        metrics = analysis.get("metrics_summary", {})
        if metrics.get("features_failed", 0) > 0:
            score -= 10 * metrics["features_failed"]

        # Grade mapping
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _determine_health(self, analysis: Dict[str, Any]) -> str:
        """Determine overall health status"""
        bottlenecks = analysis.get("bottlenecks", [])

        critical_count = sum(1 for b in bottlenecks if b.get("severity") == "critical")
        high_count = sum(1 for b in bottlenecks if b.get("severity") == "high")

        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "degraded"
        elif high_count > 0 or bottlenecks:
            return "warning"
        else:
            return "healthy"

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        bottlenecks = analysis.get("bottlenecks", [])

        for bottleneck in bottlenecks:
            btype = bottleneck.get("type", "")

            if btype == "slow_feature":
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "performance",
                        "title": f"Optimize {bottleneck['feature']} loading",
                        "description": (
                            f"Feature '{bottleneck['feature']}' takes "
                            f"{bottleneck['load_time']:.2f}s to load, which is "
                            f"significantly above average. Consider lazy loading, "
                            f"caching, or optimizing initialization logic."
                        ),
                        "action": "optimize_feature_load",
                    }
                )

            elif btype == "retry_prone_features":
                recommendations.append(
                    {
                        "priority": "medium",
                        "category": "reliability",
                        "title": "Investigate retry-prone features",
                        "description": (
                            f"Features {bottleneck['features']} frequently require "
                            f"retries. Check for transient dependencies, network "
                            f"issues, or initialization race conditions."
                        ),
                        "action": "investigate_retries",
                    }
                )

            elif btype == "rollbacks_occurred":
                recommendations.append(
                    {
                        "priority": "critical",
                        "category": "stability",
                        "title": "Address rollback causes",
                        "description": (
                            f"{bottleneck['count']} rollbacks occurred during startup. "
                            f"Review feature dependencies and initialization order. "
                            f"Consider adding pre-flight checks."
                        ),
                        "action": "fix_rollbacks",
                    }
                )

            elif btype == "failed_features":
                recommendations.append(
                    {
                        "priority": "critical",
                        "category": "functionality",
                        "title": "Fix failed features",
                        "description": (
                            f"Features {bottleneck['features']} failed to initialize. "
                            f"Review error logs and ensure all dependencies are met."
                        ),
                        "action": "fix_failed_features",
                    }
                )

            elif btype == "repair_queue_backup":
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "maintenance",
                        "title": "Clear repair queue backlog",
                        "description": (
                            f"{bottleneck['count']} items pending repair. "
                            f"Consider increasing repair frequency or investigating "
                            f"root causes of repeated failures."
                        ),
                        "action": "clear_repair_queue",
                    }
                )

            elif btype == "health_check_failures":
                recommendations.append(
                    {
                        "priority": "high",
                        "category": "monitoring",
                        "title": "Improve health check reliability",
                        "description": (
                            f"Health check failure rate is {bottleneck['failure_rate']:.1%}. "
                            f"Review health check implementations and thresholds."
                        ),
                        "action": "fix_health_checks",
                    }
                )

        # Add general recommendations based on dependency analysis
        dep_analysis = analysis.get("dependency_analysis", {})
        if dep_analysis.get("max_dependency_depth", 0) > 5:
            recommendations.append(
                {
                    "priority": "medium",
                    "category": "architecture",
                    "title": "Reduce dependency depth",
                    "description": (
                        f"Maximum dependency depth is {dep_analysis['max_dependency_depth']}. "
                        f"Deep chains can slow startup and complicate debugging."
                    ),
                    "action": "reduce_dependency_depth",
                }
            )

        return recommendations

    def export_report(
        self,
        filepath: Optional[str] = None,
        output_format: str = "json",
    ) -> str:
        """
        Export diagnostic report to file.

        Args:
            filepath: Output file path (auto-generated if not provided)
            output_format: Output format ('json', 'markdown', 'html')

        Returns:
            Path to exported file
        """
        analysis = self.analyze_startup_performance()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        if not filepath:
            filepath = f"startup_diagnostics_{timestamp}.{output_format}"

        if output_format == "json":
            content = json.dumps(analysis, indent=2, default=str)
        elif output_format == "markdown":
            content = self._format_markdown_report(analysis)
        elif output_format == "html":
            content = self._format_html_report(analysis)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        logger.info(f"Exported diagnostic report to {filepath}")
        return str(path)

    def _format_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as Markdown"""
        lines = [
            "# Startup Diagnostics Report",
            f"\n**Generated:** {analysis.get('timestamp', 'Unknown')}",
            f"\n**Overall Health:** {analysis.get('overall_health', 'Unknown').upper()}",
            f"\n**Performance Grade:** {analysis.get('performance_grade', 'N/A')}",
            "\n## Metrics Summary\n",
        ]

        metrics = analysis.get("metrics_summary", {})
        for key, value in metrics.items():
            formatted_key = key.replace("_", " ").title()
            lines.append(f"- **{formatted_key}:** {value}")

        bottlenecks = analysis.get("bottlenecks", [])
        if bottlenecks:
            lines.append("\n## Bottlenecks Identified\n")
            for i, bottleneck in enumerate(bottlenecks, 1):
                severity = bottleneck.get("severity", "unknown").upper()
                btype = bottleneck.get("type", "Unknown").replace("_", " ").title()
                lines.append(f"{i}. **[{severity}]** {btype}")
                for key, value in bottleneck.items():
                    if key not in ("type", "severity"):
                        lines.append(f"   - {key}: {value}")

        recommendations = analysis.get("recommendations", [])
        if recommendations:
            lines.append("\n## Recommendations\n")
            for rec in recommendations:
                priority = rec.get("priority", "unknown").upper()
                lines.append(f"### [{priority}] {rec.get('title', 'Unknown')}")
                lines.append(f"\n{rec.get('description', '')}\n")

        return "\n".join(lines)

    def _format_html_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as HTML"""
        health = analysis.get("overall_health", "unknown")
        health_colors = {
            "healthy": "#28a745",
            "warning": "#ffc107",
            "degraded": "#fd7e14",
            "critical": "#dc3545",
            "unknown": "#6c757d",
        }

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Startup Diagnostics Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        .header {{ border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }}
        .health-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; background-color: {health_colors.get(health, '#6c757d')}; }}
        .grade {{ font-size: 48px; font-weight: bold; color: #333; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .bottleneck {{ margin: 10px 0; padding: 10px; border-left: 4px solid #dc3545; background: white; }}
        .bottleneck.medium {{ border-color: #ffc107; }}
        .bottleneck.low {{ border-color: #28a745; }}
        .recommendation {{ margin: 10px 0; padding: 15px; background: white; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .priority-critical {{ border-left: 4px solid #dc3545; }}
        .priority-high {{ border-left: 4px solid #fd7e14; }}
        .priority-medium {{ border-left: 4px solid #ffc107; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #e9ecef; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Startup Diagnostics Report</h1>
        <p>Generated: {analysis.get('timestamp', 'Unknown')}</p>
        <span class="health-badge">{health.upper()}</span>
        <span class="grade">{analysis.get('performance_grade', 'N/A')}</span>
    </div>
    
    <div class="section">
        <h2>Metrics Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
"""

        metrics = analysis.get("metrics_summary", {})
        for key, value in metrics.items():
            formatted_key = key.replace("_", " ").title()
            html += f"            <tr><td>{formatted_key}</td><td>{value}</td></tr>\n"

        html += """        </table>
    </div>
"""

        bottlenecks = analysis.get("bottlenecks", [])
        if bottlenecks:
            html += """    <div class="section">
        <h2>Bottlenecks Identified</h2>
"""
            for bottleneck in bottlenecks:
                severity = bottleneck.get("severity", "unknown")
                btype = bottleneck.get("type", "Unknown").replace("_", " ").title()
                html += f"""        <div class="bottleneck {severity}">
            <strong>[{severity.upper()}] {btype}</strong><br>
"""
                for key, value in bottleneck.items():
                    if key not in ("type", "severity"):
                        html += f"            <em>{key}:</em> {value}<br>\n"
                html += "        </div>\n"
            html += "    </div>\n"

        recommendations = analysis.get("recommendations", [])
        if recommendations:
            html += """    <div class="section">
        <h2>Recommendations</h2>
"""
            for rec in recommendations:
                priority = rec.get("priority", "unknown")
                html += f"""        <div class="recommendation priority-{priority}">
            <strong>{rec.get('title', 'Unknown')}</strong>
            <p>{rec.get('description', '')}</p>
        </div>
"""
            html += "    </div>\n"

        html += """</body>
</html>"""
        return html


class StartupConfigBuilder:
    """
    Builder pattern for constructing startup configurations.

    Provides fluent interface for building complex startup configurations
    with validation and sensible defaults.
    """

    def __init__(self):
        self._config: Dict[str, Any] = {
            "max_retries": 3,
            "retry_delay": 1.0,
            "parallel_limit": 4,
            "timeout": 30.0,
            "enable_healing": True,
            "healing_interval": 60.0,
            "health_check_interval": 30.0,
            "enable_webhooks": False,
            "webhook_urls": [],
            "persistence_path": None,
            "log_level": "INFO",
            "circuit_breaker": {
                "failure_threshold": 5,
                "recovery_timeout": 60.0,
                "half_open_requests": 3,
            },
        }

    def with_retries(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_multiplier: float = 2.0,
    ) -> "StartupConfigBuilder":
        """Configure retry behavior"""
        self._config["max_retries"] = max_retries
        self._config["retry_delay"] = delay
        self._config["backoff_multiplier"] = backoff_multiplier
        return self

    def with_parallelism(self, limit: int = 4) -> "StartupConfigBuilder":
        """Set parallel loading limit"""
        self._config["parallel_limit"] = max(1, limit)
        return self

    def with_timeout(self, seconds: float = 30.0) -> "StartupConfigBuilder":
        """Set default timeout for feature loading"""
        self._config["timeout"] = seconds
        return self

    def with_healing(
        self,
        enabled: bool = True,
        interval: float = 60.0,
    ) -> "StartupConfigBuilder":
        """Configure background healing"""
        self._config["enable_healing"] = enabled
        self._config["healing_interval"] = interval
        return self

    def with_health_checks(
        self,
        interval: float = 30.0,
    ) -> "StartupConfigBuilder":
        """Configure health check interval"""
        self._config["health_check_interval"] = interval
        return self

    def with_webhooks(
        self,
        *urls: str,
    ) -> "StartupConfigBuilder":
        """Add webhook URLs for notifications"""
        self._config["enable_webhooks"] = True
        self._config["webhook_urls"].extend(urls)
        return self

    def with_persistence(
        self,
        path: str,
    ) -> "StartupConfigBuilder":
        """Enable state persistence"""
        self._config["persistence_path"] = path
        return self

    def with_circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_requests: int = 3,
    ) -> "StartupConfigBuilder":
        """Configure circuit breaker settings"""
        self._config["circuit_breaker"] = {
            "failure_threshold": failure_threshold,
            "recovery_timeout": recovery_timeout,
            "half_open_requests": half_open_requests,
        }
        return self

    def with_log_level(self, level: str = "INFO") -> "StartupConfigBuilder":
        """Set logging level"""
        self._config["log_level"] = level.upper()
        return self

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings"""
        warnings = []

        if self._config["max_retries"] < 0:
            warnings.append("max_retries should be >= 0")

        if self._config["retry_delay"] < 0:
            warnings.append("retry_delay should be >= 0")

        if self._config["parallel_limit"] < 1:
            warnings.append("parallel_limit should be >= 1")

        if self._config["timeout"] < 1:
            warnings.append("timeout should be >= 1 second")

        if self._config["enable_webhooks"] and not self._config["webhook_urls"]:
            warnings.append("webhooks enabled but no URLs configured")

        cb = self._config["circuit_breaker"]
        if cb["failure_threshold"] < 1:
            warnings.append("circuit_breaker.failure_threshold should be >= 1")

        return warnings

    def build(self) -> Dict[str, Any]:
        """Build and return the configuration"""
        warnings = self.validate()
        if warnings:
            for warning in warnings:
                logger.warning(f"Config warning: {warning}")

        return self._config.copy()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_startup_manager(
    config: Optional[Dict[str, Any]] = None,
    persistence_path: Optional[str] = None,
) -> ResilientStartupManager:
    """
    Factory function to create a configured ResilientStartupManager.

    Args:
        config: Optional configuration dictionary
        persistence_path: Optional path for state persistence

    Returns:
        Configured ResilientStartupManager instance
    """
    if config is None:
        config = StartupConfigBuilder().build()

    manager = ResilientStartupManager(
        enable_parallel_loading=True,
        max_parallel_loads=config.get("parallel_limit", 4),
        startup_timeout=config.get("timeout", 300.0),
        feature_timeout=config.get("timeout", 30.0),
    )

    if persistence_path or config.get("persistence_path"):
        path = persistence_path or config.get("persistence_path")
        # Set up persistence hooks
        logger.info(f"Persistence enabled at: {path}")

    return manager


def create_simplified_manager(
    config: Optional[Dict[str, Any]] = None,
) -> SimplifiedStartupManager:
    """
    Factory function to create a configured SimplifiedStartupManager.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured SimplifiedStartupManager instance
    """
    if config is None:
        config = StartupConfigBuilder().build()

    manager = SimplifiedStartupManager()

    # Webhooks can be configured after creation if needed
    # The manager handles webhooks internally

    return manager


def create_diagnostics(
    manager: Optional[ResilientStartupManager] = None,
    simplified_manager: Optional[SimplifiedStartupManager] = None,
) -> StartupDiagnostics:
    """
    Factory function to create diagnostics instance.

    Args:
        manager: Optional ResilientStartupManager
        simplified_manager: Optional SimplifiedStartupManager

    Returns:
        Configured StartupDiagnostics instance
    """
    return StartupDiagnostics(
        manager=manager,
        simplified_manager=simplified_manager,
    )


# =============================================================================
# CONVENIENCE DECORATORS
# =============================================================================


def feature(
    name: Optional[str] = None,
    category: FeatureCategory = FeatureCategory.STANDARD,
    dependencies: Optional[List[str]] = None,
    timeout: float = 30.0,
    retries: int = 3,
    critical: bool = False,
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator for registering feature initialization functions.

    Usage:
        @feature(name="database", category=FeatureCategory.CRITICAL, dependencies=["config"])
        async def init_database():
            # Initialize database
            return db_connection

    Args:
        name: Feature name (defaults to function name)
        category: Feature category
        dependencies: List of feature dependencies
        timeout: Maximum time for initialization
        retries: Number of retry attempts
        critical: Shorthand for FeatureCategory.CRITICAL
        description: Feature description
        tags: Optional tags for categorization

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        # Store metadata on the function (public attribute)
        feature_name = name or func.__name__
        feature_category = FeatureCategory.CRITICAL if critical else category

        metadata = {
            "name": feature_name,
            "category": feature_category,
            "dependencies": dependencies or [],
            "timeout": timeout,
            "retries": retries,
            "description": description or func.__doc__ or "",
            "tags": tags or [],
            "is_async": asyncio.iscoroutinefunction(func),
        }

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Use public attribute name
        async_wrapper.feature_metadata = metadata  # type: ignore[attr-defined]
        return async_wrapper

    return decorator


def health_check(
    name: Optional[str] = None,
    interval: float = 30.0,
    timeout: float = 10.0,
    critical: bool = False,
) -> Callable:
    """
    Decorator for registering health check functions.

    Usage:
        @health_check(name="database", interval=60.0, critical=True)
        async def check_database():
            # Check database health
            return True  # or raise exception

    Args:
        name: Health check name (defaults to function name)
        interval: Check interval in seconds
        timeout: Maximum time for check
        critical: Whether this is a critical health check

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        check_name = name or func.__name__

        metadata = {
            "name": check_name,
            "interval": interval,
            "timeout": timeout,
            "critical": critical,
            "is_async": asyncio.iscoroutinefunction(func),
        }

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Use public attribute name
        async_wrapper.health_check_metadata = metadata  # type: ignore[attr-defined]
        return async_wrapper

    return decorator


def repair_handler(
    feature_name: str,
    priority: int = 5,
    max_attempts: int = 3,
) -> Callable:
    """
    Decorator for registering repair handlers.

    Usage:
        @repair_handler("database", priority=1, max_attempts=5)
        async def repair_database(error: Exception):
            # Attempt to repair database connection
            pass

    Args:
        feature_name: Name of feature this handler repairs
        priority: Repair priority (lower = higher priority)
        max_attempts: Maximum repair attempts

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        metadata = {
            "feature_name": feature_name,
            "priority": priority,
            "max_attempts": max_attempts,
            "is_async": asyncio.iscoroutinefunction(func),
        }

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        # Use public attribute name
        async_wrapper.repair_handler_metadata = metadata  # type: ignore[attr-defined]
        return async_wrapper

    return decorator


# =============================================================================
# AUTO-DISCOVERY
# =============================================================================


class FeatureRegistry:
    """
    Registry for auto-discovered features and health checks.

    Scans modules for decorated functions and registers them
    with startup managers automatically.
    """

    _instance: Optional["FeatureRegistry"] = None

    def __init__(self):
        # Initialize instance attributes if not already done
        if not hasattr(self, "_initialized"):
            self._features: Dict[str, Callable] = {}
            self._health_checks: Dict[str, Callable] = {}
            self._repair_handlers: Dict[str, List[Callable]] = {}
            self._initialized = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "FeatureRegistry":
        """Get singleton instance"""
        return cls()

    def register_feature(self, func: Callable) -> None:
        """Register a feature function"""
        metadata = getattr(func, "feature_metadata", None)
        if metadata:
            name = metadata["name"]
            self._features[name] = func
            logger.debug(f"Registered feature: {name}")

    def register_health_check(self, func: Callable) -> None:
        """Register a health check function"""
        metadata = getattr(func, "health_check_metadata", None)
        if metadata:
            name = metadata["name"]
            self._health_checks[name] = func
            logger.debug(f"Registered health check: {name}")

    def register_repair_handler(self, func: Callable) -> None:
        """Register a repair handler function"""
        metadata = getattr(func, "repair_handler_metadata", None)
        if metadata:
            feature_name = metadata["feature_name"]
            if feature_name not in self._repair_handlers:
                self._repair_handlers[feature_name] = []
            self._repair_handlers[feature_name].append(func)
            logger.debug(f"Registered repair handler for: {feature_name}")

    def discover_in_module(self, module) -> int:
        """
        Discover and register decorated functions in a module.

        Args:
            module: Module to scan

        Returns:
            Number of items discovered
        """
        count = 0

        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj):
                # Use public attribute names
                if hasattr(obj, "feature_metadata"):
                    self.register_feature(obj)
                    count += 1
                if hasattr(obj, "health_check_metadata"):
                    self.register_health_check(obj)
                    count += 1
                if hasattr(obj, "repair_handler_metadata"):
                    self.register_repair_handler(obj)
                    count += 1

        return count

    def apply_to_manager(
        self,
        manager: ResilientStartupManager,
    ) -> int:
        """
        Apply all registered features to a manager.

        Args:
            manager: ResilientStartupManager to configure

        Returns:
            Number of features applied
        """
        count = 0

        for name, func in self._features.items():
            meta = getattr(func, "feature_metadata", {})
            # Create FeatureDefinition for ResilientStartupManager
            definition = FeatureDefinition(
                name=name,
                loader=func,
                priority=FeaturePriority.MEDIUM,
                is_critical=(meta.get("category") == FeatureCategory.CRITICAL),
                dependencies=meta.get("dependencies", []),
            )
            manager.register_feature(definition)
            count += 1

        return count

    def apply_to_simplified_manager(
        self,
        manager: SimplifiedStartupManager,
    ) -> int:
        """
        Apply all registered features to a simplified manager.

        Args:
            manager: SimplifiedStartupManager to configure

        Returns:
            Number of features applied
        """
        count = 0

        for name, func in self._features.items():
            meta = getattr(func, "feature_metadata", {})
            # Create StartupFeature for SimplifiedStartupManager
            feature = StartupFeature(
                name=name,
                init_func=func,
                category=meta.get("category", FeatureCategory.STANDARD),
                dependencies=meta.get("dependencies", []),
            )
            manager.register_feature(feature)
            count += 1

        return count

    def get_features(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered features with metadata"""
        return {name: getattr(func, "feature_metadata", {}) for name, func in self._features.items()}

    def get_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered health checks with metadata"""
        return {name: getattr(func, "health_check_metadata", {}) for name, func in self._health_checks.items()}

    def clear(self) -> None:
        """Clear all registrations"""
        self._features.clear()
        self._health_checks.clear()
        self._repair_handlers.clear()


# =============================================================================
# QUICK START UTILITIES
# =============================================================================


async def quick_start(
    features: Dict[str, Callable],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[SimplifiedStartupManager, Dict[str, Any]]:
    """
    Quick start utility for simple use cases.

    Example:
        manager, results = await quick_start({
            "database": init_database,
            "cache": init_cache,
            "api": init_api,
        })

    Args:
        features: Dictionary of feature name -> initializer function
        config: Optional configuration

    Returns:
        Tuple of (manager, initialization results)
    """
    manager = create_simplified_manager(config)

    for name, initializer in features.items():
        # Create StartupFeature for each
        feature = StartupFeature(
            name=name,
            init_func=initializer,
            category=FeatureCategory.STANDARD,
        )
        manager.register_feature(feature)

    # Run startup sequence
    success = await manager.startup_sequence()

    # Gather results
    results = {
        "success": success,
        "features": manager.get_status_summary(),
    }

    return manager, results


async def quick_start_with_deps(
    features: List[Tuple[str, Callable, List[str]]],
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[ResilientStartupManager, Dict[str, Any]]:
    """
    Quick start with dependency support.

    Example:
        manager, results = await quick_start_with_deps([
            ("config", init_config, []),
            ("database", init_database, ["config"]),
            ("cache", init_cache, ["config"]),
            ("api", init_api, ["database", "cache"]),
        ])

    Args:
        features: List of (name, initializer, dependencies) tuples
        config: Optional configuration

    Returns:
        Tuple of (manager, initialization results)
    """
    manager = create_startup_manager(config)

    for name, initializer, deps in features:
        # Create FeatureDefinition for ResilientStartupManager
        definition = FeatureDefinition(
            name=name,
            loader=initializer,
            priority=FeaturePriority.MEDIUM,
            dependencies=deps,
        )
        manager.register_feature(definition)

    # Run startup
    success, status = await manager.startup(parallel=True)

    # Gather results
    results = {
        "success": success,
        "status": status,
        "metrics": manager.get_metrics(),
    }

    return manager, results


def get_startup_template() -> str:
    """
    Get a template for setting up resilient startup.

    Returns:
        Template code string
    """
    return '''"""
Resilient Startup Template
===========================

This template demonstrates how to set up resilient startup
for your application using the Vega startup system.
"""

import asyncio
from vega.core.resilient_startup import (
    feature,
    health_check,
    repair_handler,
    FeatureCategory,
    FeatureRegistry,
    create_startup_manager,
    create_diagnostics,
    StartupConfigBuilder,
)


# =============================================================================
# Define Features
# =============================================================================


@feature(
    name="config",
    category=FeatureCategory.CRITICAL,
    description="Load application configuration",
)
async def init_config():
    """Initialize configuration from environment and files"""
    # Your config loading logic here
    return {"debug": True, "environment": "development"}


@feature(
    name="database",
    category=FeatureCategory.CRITICAL,
    dependencies=["config"],
    timeout=30.0,
    retries=3,
)
async def init_database():
    """Initialize database connection pool"""
    # Your database initialization here
    await asyncio.sleep(0.1)  # Simulated connection
    return "database_connection"


@feature(
    name="cache",
    category=FeatureCategory.STANDARD,
    dependencies=["config"],
)
async def init_cache():
    """Initialize caching layer"""
    # Your cache initialization here
    return "cache_instance"


@feature(
    name="api",
    category=FeatureCategory.STANDARD,
    dependencies=["database", "cache"],
)
async def init_api():
    """Initialize API endpoints"""
    # Your API setup here
    return "api_ready"


# =============================================================================
# Define Health Checks
# =============================================================================


@health_check(name="database", interval=30.0, critical=True)
async def check_database():
    """Check database connectivity"""
    # Your health check logic here
    return True


@health_check(name="cache", interval=60.0)
async def check_cache():
    """Check cache availability"""
    # Your health check logic here
    return True


# =============================================================================
# Define Repair Handlers
# =============================================================================


@repair_handler("database", priority=1, max_attempts=5)
async def repair_database(error: Exception):
    """Attempt to repair database connection"""
    # Your repair logic here
    await asyncio.sleep(1)
    return True


# =============================================================================
# Main Startup
# =============================================================================


async def main():
    # Build configuration
    config = (
        StartupConfigBuilder()
        .with_retries(max_retries=3, delay=1.0)
        .with_parallelism(limit=4)
        .with_healing(enabled=True, interval=60.0)
        .with_health_checks(interval=30.0)
        .build()
    )
    
    # Create manager
    manager = create_startup_manager(config)
    
    # Auto-discover features in this module
    import sys
    registry = FeatureRegistry.get_instance()
    registry.discover_in_module(sys.modules[__name__])
    registry.apply_to_manager(manager)
    
    # Initialize
    print("Starting application...")
    results = await manager.initialize_all()
    
    # Check results
    for name, result in results.items():
        status = "✓" if result.get("success") else "✗"
        print(f"  {status} {name}")
    
    # Create diagnostics
    diagnostics = create_diagnostics(manager=manager)
    analysis = diagnostics.analyze_startup_performance()
    
    print(f"\\nStartup Grade: {analysis['performance_grade']}")
    print(f"Health Status: {analysis['overall_health']}")
    
    # Export report
    diagnostics.export_report("startup_report.md", format="markdown")
    
    # Run application...
    print("\\nApplication running. Press Ctrl+C to stop.")
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    
    # Cleanup
    await manager.shutdown()
    print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
'''


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Core classes
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureState",
    "FeaturePriority",
    "FeatureStatus",
    "StartupFeature",
    "ResolutionRecord",
    "ResolutionKnowledgeBase",
    "FeatureUsageTracker",
    "BackgroundHealer",
    "ResilientStartupManager",
    "SimplifiedStartupManager",
    # Utility classes
    "StartupDiagnostics",
    "StartupConfigBuilder",
    "FeatureRegistry",
    "DependencyNode",
    # Dataclasses
    "CircuitBreakerConfig",
    "CircuitBreakerState",
    "HealingMetrics",
    "HealingEvent",
    "StartupMetrics",
    "StartupEvent",
    # Factory functions
    "create_startup_manager",
    "create_simplified_manager",
    "create_diagnostics",
    # Decorators
    "feature",
    "health_check",
    "repair_handler",
    # Quick start utilities
    "quick_start",
    "quick_start_with_deps",
    "get_startup_template",
]
