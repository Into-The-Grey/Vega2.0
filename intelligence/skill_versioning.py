#!/usr/bin/env python3
"""
🔄 PHASE 5: SKILL VERSIONING & MANAGEMENT SYSTEM
==================================================
Advanced skill version control system for tracking, managing, and optimizing
AI capabilities over time. Implements sophisticated version management with
automatic rollback, A/B testing, and continuous skill evolution.

This system implements:
- Semantic versioning for AI skills (Major.Minor.Patch)
- Git-like branching for skill development
- Automated rollback on performance degradation
- Skill dependency management
- Performance-based version promotion
- Skill composition and inheritance
- Zero-downtime skill updates
"""

import sqlite3
import logging
import json
import time
import asyncio
import hashlib
import pickle
import importlib
import inspect
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from pathlib import Path
import re
import statistics
from enum import Enum
import semver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """Skill version status"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


class VersionType(Enum):
    """Version increment types"""

    MAJOR = "major"  # Breaking changes
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, backward compatible


@dataclass
class SkillVersion:
    """Represents a versioned AI skill"""

    skill_id: str
    version: str  # Semantic version (e.g., "1.2.3")
    name: str
    description: str
    status: SkillStatus

    # Code and Implementation
    implementation: str  # Serialized skill implementation
    dependencies: List[str]  # Required skill dependencies
    interface_signature: str  # Function signature hash

    # Performance Metrics
    performance_score: float  # 0-1 overall performance
    quality_metrics: Dict[str, float]  # Detailed quality metrics
    test_results: Dict[str, Any]  # Test execution results

    # Versioning Metadata
    parent_version: Optional[str]  # Previous version
    branch_name: str  # Development branch
    commit_message: str  # Version description

    # Lifecycle
    created_at: datetime
    activated_at: Optional[datetime]
    deprecated_at: Optional[datetime]
    created_by: str  # Source (human, ai_optimizer, experiment)

    # Deployment
    deployment_config: Dict[str, Any]  # Deployment configuration
    rollback_version: Optional[str]  # Version to rollback to

    # Usage Statistics
    usage_count: int = 0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    last_used: Optional[datetime] = None


@dataclass
class SkillBranch:
    """Represents a skill development branch"""

    branch_id: str
    skill_id: str
    branch_name: str
    base_version: str  # Version this branch was created from
    current_version: str  # Latest version in this branch
    status: str  # active, merged, abandoned
    description: str
    created_at: datetime
    last_commit: datetime

    # Branch metrics
    version_count: int = 0
    merge_conflicts: List[str] = field(default_factory=list)


@dataclass
class SkillDependency:
    """Skill dependency specification"""

    skill_id: str
    required_version: str  # Minimum version requirement
    constraint: str  # Version constraint (>=, ==, ~=, etc.)
    optional: bool = False  # Whether dependency is optional


class SkillRegistry:
    """Central registry for all skill versions"""

    def __init__(self):
        self.registry_db = "skill_registry.db"
        self._init_database()
        self.active_skills: Dict[str, SkillVersion] = {}
        self.skill_cache: Dict[str, Any] = {}

        logger.info("🔄 Skill Registry initialized")

    def _init_database(self):
        """Initialize skill registry database"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS skill_versions (
            skill_id TEXT,
            version TEXT,
            name TEXT,
            description TEXT,
            status TEXT,
            implementation BLOB,
            dependencies TEXT,
            interface_signature TEXT,
            performance_score REAL,
            quality_metrics TEXT,
            test_results TEXT,
            parent_version TEXT,
            branch_name TEXT,
            commit_message TEXT,
            created_at TEXT,
            activated_at TEXT,
            deprecated_at TEXT,
            created_by TEXT,
            deployment_config TEXT,
            rollback_version TEXT,
            usage_count INTEGER DEFAULT 0,
            success_rate REAL DEFAULT 0.0,
            avg_execution_time REAL DEFAULT 0.0,
            last_used TEXT,
            PRIMARY KEY (skill_id, version)
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS skill_branches (
            branch_id TEXT PRIMARY KEY,
            skill_id TEXT,
            branch_name TEXT,
            base_version TEXT,
            current_version TEXT,
            status TEXT,
            description TEXT,
            created_at TEXT,
            last_commit TEXT,
            version_count INTEGER DEFAULT 0,
            merge_conflicts TEXT
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS skill_deployments (
            deployment_id TEXT PRIMARY KEY,
            skill_id TEXT,
            version TEXT,
            environment TEXT,
            deployed_at TEXT,
            deployment_status TEXT,
            rollback_count INTEGER DEFAULT 0,
            last_rollback TEXT
        )
        """
        )

        # Create indexes for performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_skill_status ON skill_versions(skill_id, status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_performance ON skill_versions(performance_score DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON skill_versions(created_at DESC)"
        )

        conn.commit()
        conn.close()

    def register_skill_version(self, skill_version: SkillVersion) -> bool:
        """Register a new skill version"""
        try:
            # Validate version format
            if not self._validate_version(skill_version.version):
                raise ValueError(f"Invalid version format: {skill_version.version}")

            # Check for version conflicts
            if self._version_exists(skill_version.skill_id, skill_version.version):
                logger.warning(
                    f"Version {skill_version.version} already exists for skill {skill_version.skill_id}"
                )
                return False

            # Store in database
            self._store_skill_version(skill_version)

            # Update active skill if this is production
            if skill_version.status == SkillStatus.PRODUCTION:
                self.active_skills[skill_version.skill_id] = skill_version

            logger.info(
                f"✅ Registered skill version {skill_version.skill_id}:{skill_version.version}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to register skill version: {e}")
            return False

    def create_branch(
        self, skill_id: str, branch_name: str, base_version: str, description: str = ""
    ) -> Optional[SkillBranch]:
        """Create a new development branch"""
        try:
            branch_id = f"{skill_id}_{branch_name}_{int(time.time())}"

            branch = SkillBranch(
                branch_id=branch_id,
                skill_id=skill_id,
                branch_name=branch_name,
                base_version=base_version,
                current_version=base_version,
                status="active",
                description=description,
                created_at=datetime.now(),
                last_commit=datetime.now(),
            )

            self._store_branch(branch)
            logger.info(f"🌿 Created branch {branch_name} for skill {skill_id}")
            return branch

        except Exception as e:
            logger.error(f"Failed to create branch: {e}")
            return None

    def increment_version(
        self, skill_id: str, version_type: VersionType, current_version: str
    ) -> str:
        """Generate next version number"""
        try:
            current = semver.Version.parse(current_version)

            if version_type == VersionType.MAJOR:
                next_version = current.bump_major()
            elif version_type == VersionType.MINOR:
                next_version = current.bump_minor()
            else:  # PATCH
                next_version = current.bump_patch()

            return str(next_version)

        except Exception as e:
            logger.error(f"Version increment failed: {e}")
            # Fallback to simple increment
            parts = current_version.split(".")
            if len(parts) >= 3:
                if version_type == VersionType.PATCH:
                    parts[2] = str(int(parts[2]) + 1)
                elif version_type == VersionType.MINOR:
                    parts[1] = str(int(parts[1]) + 1)
                    parts[2] = "0"
                else:  # MAJOR
                    parts[0] = str(int(parts[0]) + 1)
                    parts[1] = "0"
                    parts[2] = "0"
                return ".".join(parts)
            return "1.0.0"

    def promote_version(
        self, skill_id: str, version: str, target_status: SkillStatus
    ) -> bool:
        """Promote a skill version to a higher status"""
        try:
            skill_version = self.get_skill_version(skill_id, version)
            if not skill_version:
                logger.error(f"Skill version {skill_id}:{version} not found")
                return False

            # Validate promotion path
            if not self._validate_promotion(skill_version.status, target_status):
                logger.error(
                    f"Invalid promotion from {skill_version.status} to {target_status}"
                )
                return False

            # Update status
            skill_version.status = target_status

            # If promoting to production, activate it
            if target_status == SkillStatus.PRODUCTION:
                skill_version.activated_at = datetime.now()
                self.active_skills[skill_id] = skill_version

            # Update in database
            self._update_skill_version(skill_version)

            logger.info(f"📈 Promoted {skill_id}:{version} to {target_status.value}")
            return True

        except Exception as e:
            logger.error(f"Version promotion failed: {e}")
            return False

    def rollback_skill(
        self, skill_id: str, target_version: Optional[str] = None
    ) -> bool:
        """Rollback skill to previous stable version"""
        try:
            current_skill = self.active_skills.get(skill_id)
            if not current_skill:
                logger.error(f"No active skill found for {skill_id}")
                return False

            # Determine rollback target
            if target_version:
                rollback_skill = self.get_skill_version(skill_id, target_version)
            else:
                # Find previous production version
                rollback_skill = self._find_previous_production_version(
                    skill_id, current_skill.version
                )

            if not rollback_skill:
                logger.error(f"No suitable rollback version found for {skill_id}")
                return False

            # Perform rollback
            self.active_skills[skill_id] = rollback_skill
            rollback_skill.activated_at = datetime.now()

            # Record rollback
            self._record_rollback(
                skill_id, current_skill.version, rollback_skill.version
            )

            logger.warning(
                f"🔄 Rolled back {skill_id} from {current_skill.version} to {rollback_skill.version}"
            )
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_skill_version(self, skill_id: str, version: str) -> Optional[SkillVersion]:
        """Retrieve specific skill version"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT * FROM skill_versions 
        WHERE skill_id = ? AND version = ?
        """,
            (skill_id, version),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return self._row_to_skill_version(result)
        return None

    def get_latest_version(
        self, skill_id: str, status: Optional[SkillStatus] = None
    ) -> Optional[SkillVersion]:
        """Get the latest version of a skill"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        if status:
            cursor.execute(
                """
            SELECT * FROM skill_versions 
            WHERE skill_id = ? AND status = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
                (skill_id, status.value),
            )
        else:
            cursor.execute(
                """
            SELECT * FROM skill_versions 
            WHERE skill_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
                (skill_id,),
            )

        result = cursor.fetchone()
        conn.close()

        if result:
            return self._row_to_skill_version(result)
        return None

    def get_skill_history(self, skill_id: str, limit: int = 50) -> List[SkillVersion]:
        """Get version history for a skill"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT * FROM skill_versions 
        WHERE skill_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
            (skill_id, limit),
        )

        results = cursor.fetchall()
        conn.close()

        return [self._row_to_skill_version(row) for row in results]

    def check_dependencies(self, skill_version: SkillVersion) -> Dict[str, bool]:
        """Check if all dependencies are satisfied"""
        dependency_status = {}

        for dep_string in skill_version.dependencies:
            try:
                # Parse dependency (format: "skill_id>=1.2.0")
                match = re.match(r"([^>=<~!]+)([>=<~!]+)(.+)", dep_string)
                if not match:
                    dependency_status[dep_string] = False
                    continue

                dep_skill_id, constraint, req_version = match.groups()

                # Get latest production version of dependency
                dep_skill = self.get_latest_version(
                    dep_skill_id, SkillStatus.PRODUCTION
                )
                if not dep_skill:
                    dependency_status[dep_string] = False
                    continue

                # Check version constraint
                satisfied = self._check_version_constraint(
                    dep_skill.version, constraint, req_version
                )
                dependency_status[dep_string] = satisfied

            except Exception as e:
                logger.error(f"Dependency check failed for {dep_string}: {e}")
                dependency_status[dep_string] = False

        return dependency_status

    def update_usage_statistics(
        self, skill_id: str, version: str, execution_time: float, success: bool
    ):
        """Update usage statistics for a skill version"""
        try:
            skill_version = self.get_skill_version(skill_id, version)
            if not skill_version:
                return

            # Update statistics
            skill_version.usage_count += 1
            skill_version.last_used = datetime.now()

            # Update success rate (running average)
            if skill_version.usage_count == 1:
                skill_version.success_rate = 1.0 if success else 0.0
            else:
                current_successes = skill_version.success_rate * (
                    skill_version.usage_count - 1
                )
                new_successes = current_successes + (1 if success else 0)
                skill_version.success_rate = new_successes / skill_version.usage_count

            # Update execution time (running average)
            if skill_version.usage_count == 1:
                skill_version.avg_execution_time = execution_time
            else:
                total_time = skill_version.avg_execution_time * (
                    skill_version.usage_count - 1
                )
                skill_version.avg_execution_time = (
                    total_time + execution_time
                ) / skill_version.usage_count

            # Update in database
            self._update_skill_statistics(skill_version)

        except Exception as e:
            logger.error(f"Failed to update usage statistics: {e}")

    def _validate_version(self, version: str) -> bool:
        """Validate semantic version format"""
        try:
            semver.Version.parse(version)
            return True
        except ValueError:
            # Fallback to simple regex check
            pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?$"
            return bool(re.match(pattern, version))

    def _version_exists(self, skill_id: str, version: str) -> bool:
        """Check if version already exists"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT COUNT(*) FROM skill_versions 
        WHERE skill_id = ? AND version = ?
        """,
            (skill_id, version),
        )

        count = cursor.fetchone()[0]
        conn.close()

        return count > 0

    def _validate_promotion(
        self, current_status: SkillStatus, target_status: SkillStatus
    ) -> bool:
        """Validate promotion path"""
        promotion_paths = {
            SkillStatus.DEVELOPMENT: [SkillStatus.TESTING, SkillStatus.DEPRECATED],
            SkillStatus.TESTING: [
                SkillStatus.STAGING,
                SkillStatus.DEVELOPMENT,
                SkillStatus.DEPRECATED,
            ],
            SkillStatus.STAGING: [
                SkillStatus.PRODUCTION,
                SkillStatus.TESTING,
                SkillStatus.DEPRECATED,
            ],
            SkillStatus.PRODUCTION: [SkillStatus.DEPRECATED],
            SkillStatus.DEPRECATED: [SkillStatus.RETIRED],
            SkillStatus.RETIRED: [],
        }

        return target_status in promotion_paths.get(current_status, [])

    def _find_previous_production_version(
        self, skill_id: str, current_version: str
    ) -> Optional[SkillVersion]:
        """Find previous production version for rollback"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        SELECT * FROM skill_versions 
        WHERE skill_id = ? AND status = ? AND version != ?
        ORDER BY activated_at DESC
        LIMIT 1
        """,
            (skill_id, SkillStatus.PRODUCTION.value, current_version),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            return self._row_to_skill_version(result)
        return None

    def _check_version_constraint(
        self, version: str, constraint: str, required: str
    ) -> bool:
        """Check if version satisfies constraint"""
        try:
            v = semver.Version.parse(version)
            r = semver.Version.parse(required)

            if constraint == ">=":
                return v >= r
            elif constraint == "==":
                return v == r
            elif constraint == "~=":
                return v.major == r.major and v.minor >= r.minor
            elif constraint == ">":
                return v > r
            elif constraint == "<=":
                return v <= r
            elif constraint == "<":
                return v < r

        except ValueError:
            # Fallback to string comparison
            if constraint == ">=":
                return version >= required
            elif constraint == "==":
                return version == required

        return False

    def _store_skill_version(self, skill_version: SkillVersion):
        """Store skill version in database"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        # Serialize implementation
        implementation_blob = pickle.dumps(skill_version.implementation)

        cursor.execute(
            """
        INSERT OR REPLACE INTO skill_versions VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                skill_version.skill_id,
                skill_version.version,
                skill_version.name,
                skill_version.description,
                skill_version.status.value,
                implementation_blob,
                json.dumps(skill_version.dependencies),
                skill_version.interface_signature,
                skill_version.performance_score,
                json.dumps(skill_version.quality_metrics),
                json.dumps(skill_version.test_results),
                skill_version.parent_version,
                skill_version.branch_name,
                skill_version.commit_message,
                skill_version.created_at.isoformat(),
                (
                    skill_version.activated_at.isoformat()
                    if skill_version.activated_at
                    else None
                ),
                (
                    skill_version.deprecated_at.isoformat()
                    if skill_version.deprecated_at
                    else None
                ),
                skill_version.created_by,
                json.dumps(skill_version.deployment_config),
                skill_version.rollback_version,
                skill_version.usage_count,
                skill_version.success_rate,
                skill_version.avg_execution_time,
                (
                    skill_version.last_used.isoformat()
                    if skill_version.last_used
                    else None
                ),
            ),
        )

        conn.commit()
        conn.close()

    def _store_branch(self, branch: SkillBranch):
        """Store branch in database"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO skill_branches VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                branch.branch_id,
                branch.skill_id,
                branch.branch_name,
                branch.base_version,
                branch.current_version,
                branch.status,
                branch.description,
                branch.created_at.isoformat(),
                branch.last_commit.isoformat(),
                branch.version_count,
                json.dumps(branch.merge_conflicts),
            ),
        )

        conn.commit()
        conn.close()

    def _update_skill_version(self, skill_version: SkillVersion):
        """Update existing skill version"""
        self._store_skill_version(skill_version)

    def _update_skill_statistics(self, skill_version: SkillVersion):
        """Update usage statistics in database"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        UPDATE skill_versions 
        SET usage_count = ?, success_rate = ?, avg_execution_time = ?, last_used = ?
        WHERE skill_id = ? AND version = ?
        """,
            (
                skill_version.usage_count,
                skill_version.success_rate,
                skill_version.avg_execution_time,
                (
                    skill_version.last_used.isoformat()
                    if skill_version.last_used
                    else None
                ),
                skill_version.skill_id,
                skill_version.version,
            ),
        )

        conn.commit()
        conn.close()

    def _record_rollback(self, skill_id: str, from_version: str, to_version: str):
        """Record rollback event"""
        conn = sqlite3.connect(self.registry_db)
        cursor = conn.cursor()

        deployment_id = f"rollback_{skill_id}_{int(time.time())}"

        cursor.execute(
            """
        INSERT INTO skill_deployments VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                deployment_id,
                skill_id,
                to_version,
                "production",
                datetime.now().isoformat(),
                f"rollback_from_{from_version}",
                1,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def _row_to_skill_version(self, row) -> SkillVersion:
        """Convert database row to SkillVersion object"""
        return SkillVersion(
            skill_id=row[0],
            version=row[1],
            name=row[2],
            description=row[3],
            status=SkillStatus(row[4]),
            implementation=pickle.loads(row[5]) if row[5] else "",
            dependencies=json.loads(row[6]) if row[6] else [],
            interface_signature=row[7],
            performance_score=row[8],
            quality_metrics=json.loads(row[9]) if row[9] else {},
            test_results=json.loads(row[10]) if row[10] else {},
            parent_version=row[11],
            branch_name=row[12],
            commit_message=row[13],
            created_at=datetime.fromisoformat(row[14]),
            activated_at=datetime.fromisoformat(row[15]) if row[15] else None,
            deprecated_at=datetime.fromisoformat(row[16]) if row[16] else None,
            created_by=row[17],
            deployment_config=json.loads(row[18]) if row[18] else {},
            rollback_version=row[19],
            usage_count=row[20] or 0,
            success_rate=row[21] or 0.0,
            avg_execution_time=row[22] or 0.0,
            last_used=datetime.fromisoformat(row[23]) if row[23] else None,
        )


class SkillManager:
    """High-level skill management interface"""

    def __init__(self):
        self.registry = SkillRegistry()
        self.auto_rollback_threshold = 0.7  # Rollback if success rate drops below 70%

        logger.info("🔄 Skill Manager initialized")

    async def deploy_skill_version(
        self, skill_id: str, version: str, environment: str = "production"
    ) -> bool:
        """Deploy a skill version to specified environment"""
        try:
            skill_version = self.registry.get_skill_version(skill_id, version)
            if not skill_version:
                logger.error(f"Skill version {skill_id}:{version} not found")
                return False

            # Check dependencies
            dep_status = self.registry.check_dependencies(skill_version)
            if not all(dep_status.values()):
                failed_deps = [
                    dep for dep, satisfied in dep_status.items() if not satisfied
                ]
                logger.error(f"Dependency check failed: {failed_deps}")
                return False

            # Promote to production if deploying to production
            if environment == "production":
                success = self.registry.promote_version(
                    skill_id, version, SkillStatus.PRODUCTION
                )
                if not success:
                    return False

            logger.info(f"🚀 Deployed {skill_id}:{version} to {environment}")
            return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False

    async def create_skill_version(
        self,
        skill_id: str,
        name: str,
        description: str,
        implementation: str,
        dependencies: List[str] = None,
        branch_name: str = "main",
        commit_message: str = "",
        version_type: VersionType = VersionType.PATCH,
    ) -> Optional[SkillVersion]:
        """Create a new skill version"""
        try:
            dependencies = dependencies or []

            # Get latest version to increment from
            latest = self.registry.get_latest_version(skill_id)
            if latest:
                new_version = self.registry.increment_version(
                    skill_id, version_type, latest.version
                )
                parent_version = latest.version
            else:
                new_version = "1.0.0"
                parent_version = None

            # Create interface signature
            interface_signature = hashlib.md5(implementation.encode()).hexdigest()

            skill_version = SkillVersion(
                skill_id=skill_id,
                version=new_version,
                name=name,
                description=description,
                status=SkillStatus.DEVELOPMENT,
                implementation=implementation,
                dependencies=dependencies,
                interface_signature=interface_signature,
                performance_score=0.0,
                quality_metrics={},
                test_results={},
                parent_version=parent_version,
                branch_name=branch_name,
                commit_message=commit_message or f"Created version {new_version}",
                created_at=datetime.now(),
                activated_at=None,
                deprecated_at=None,
                created_by="manual",
                deployment_config={},
                rollback_version=parent_version,
            )

            # Register the version
            success = self.registry.register_skill_version(skill_version)
            if success:
                logger.info(f"✅ Created skill version {skill_id}:{new_version}")
                return skill_version

            return None

        except Exception as e:
            logger.error(f"Failed to create skill version: {e}")
            return None

    async def monitor_skill_performance(self, skill_id: str, version: str):
        """Monitor skill performance and trigger automatic rollback if needed"""
        try:
            skill_version = self.registry.get_skill_version(skill_id, version)
            if not skill_version:
                return

            # Check if success rate has dropped below threshold
            if (
                skill_version.usage_count > 10
                and skill_version.success_rate < self.auto_rollback_threshold
            ):

                logger.warning(
                    f"⚠️ Skill {skill_id}:{version} success rate dropped to {skill_version.success_rate:.2f}"
                )

                # Trigger automatic rollback
                rollback_success = self.registry.rollback_skill(skill_id)
                if rollback_success:
                    logger.warning(
                        f"🔄 Automatically rolled back {skill_id} due to performance degradation"
                    )

        except Exception as e:
            logger.error(f"Performance monitoring failed: {e}")

    async def get_skill_analytics(
        self, skill_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for a skill"""
        try:
            history = self.registry.get_skill_history(skill_id, limit=100)
            if not history:
                return {"error": "No skill history found"}

            # Calculate analytics
            versions = len(history)
            latest = history[0]

            # Performance trends
            recent_versions = [
                v
                for v in history
                if v.created_at > datetime.now() - timedelta(days=days)
            ]
            performance_trend = "stable"

            if len(recent_versions) > 1:
                scores = [
                    v.performance_score
                    for v in recent_versions
                    if v.performance_score > 0
                ]
                if len(scores) > 1:
                    if scores[0] > scores[-1]:
                        performance_trend = "improving"
                    elif scores[0] < scores[-1]:
                        performance_trend = "declining"

            # Usage statistics
            total_usage = sum(v.usage_count for v in history)
            avg_success_rate = (
                statistics.mean([v.success_rate for v in history if v.success_rate > 0])
                if history
                else 0
            )

            analytics = {
                "skill_id": skill_id,
                "total_versions": versions,
                "latest_version": latest.version,
                "latest_status": latest.status.value,
                "performance_trend": performance_trend,
                "total_usage": total_usage,
                "avg_success_rate": avg_success_rate,
                "latest_performance": latest.performance_score,
                "version_history": [
                    {
                        "version": v.version,
                        "status": v.status.value,
                        "performance": v.performance_score,
                        "usage": v.usage_count,
                        "success_rate": v.success_rate,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in recent_versions[:10]
                ],
            }

            return analytics

        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            return {"error": str(e)}


# Test and demonstration functions
async def demonstrate_skill_versioning():
    """Demonstrate the skill versioning system"""
    print("🔄 SKILL VERSIONING & MANAGEMENT SYSTEM")
    print("=" * 55)

    manager = SkillManager()

    # Create a sample skill
    print("📝 Creating sample skill versions...")

    skill_id = "conversation_improver"

    # Version 1.0.0 - Initial implementation
    v1 = await manager.create_skill_version(
        skill_id=skill_id,
        name="Conversation Improver",
        description="Improves conversation quality through response optimization",
        implementation="def improve_response(prompt, response): return response.upper()",
        dependencies=[],
        commit_message="Initial conversation improver implementation",
        version_type=VersionType.MAJOR,
    )

    if v1:
        print(f"✅ Created v{v1.version}: {v1.description}")

        # Simulate testing and promotion
        manager.registry.promote_version(skill_id, v1.version, SkillStatus.TESTING)
        manager.registry.promote_version(skill_id, v1.version, SkillStatus.STAGING)
        manager.registry.promote_version(skill_id, v1.version, SkillStatus.PRODUCTION)

        # Simulate usage
        for i in range(50):
            manager.registry.update_usage_statistics(skill_id, v1.version, 0.1, True)

        print(f"� v{v1.version} promoted to production with {v1.usage_count} uses")

    # Version 1.1.0 - Minor improvement
    v2 = await manager.create_skill_version(
        skill_id=skill_id,
        name="Conversation Improver Enhanced",
        description="Enhanced conversation improver with better context handling",
        implementation="def improve_response(prompt, response, context=None): return response.title()",
        dependencies=["context_analyzer>=1.0.0"],
        commit_message="Added context handling capabilities",
        version_type=VersionType.MINOR,
    )

    if v2:
        print(f"✅ Created v{v2.version}: Enhanced with context handling")

        # Test the new version
        manager.registry.promote_version(skill_id, v2.version, SkillStatus.TESTING)

        # Simulate some failures to trigger rollback
        for i in range(20):
            success = i < 10  # 50% success rate
            manager.registry.update_usage_statistics(
                skill_id, v2.version, 0.12, success
            )

        print(
            f"⚠️ v{v2.version} showing poor performance: {v2.success_rate:.1%} success rate"
        )

    # Version 2.0.0 - Major rewrite
    v3 = await manager.create_skill_version(
        skill_id=skill_id,
        name="Advanced Conversation Improver",
        description="Complete rewrite with ML-based improvements",
        implementation="def improve_response(prompt, response, context=None, model=None): return ml_improve(response)",
        dependencies=["ml_processor>=2.0.0", "context_analyzer>=1.2.0"],
        commit_message="Major rewrite with ML capabilities",
        version_type=VersionType.MAJOR,
    )

    if v3:
        print(f"✅ Created v{v3.version}: Major rewrite with ML")

    # Create development branch
    branch = manager.registry.create_branch(
        skill_id=skill_id,
        branch_name="experimental",
        base_version=v3.version if v3 else "1.0.0",
        description="Experimental features branch",
    )

    if branch:
        print(f"🌿 Created experimental branch: {branch.branch_name}")

    # Deploy a version
    print("\n🚀 Deploying version to production...")
    if v1:
        deployed = await manager.deploy_skill_version(skill_id, v1.version)
        if deployed:
            print(f"✅ Successfully deployed v{v1.version} to production")

    # Get analytics
    print("\n📊 Generating skill analytics...")
    analytics = await manager.get_skill_analytics(skill_id)

    print(f"📈 Skill Analytics for {skill_id}:")
    print(f"  Total versions: {analytics.get('total_versions', 0)}")
    print(f"  Latest version: {analytics.get('latest_version', 'N/A')}")
    print(f"  Performance trend: {analytics.get('performance_trend', 'unknown')}")
    print(f"  Total usage: {analytics.get('total_usage', 0)}")
    print(f"  Average success rate: {analytics.get('avg_success_rate', 0):.1%}")

    # Test rollback functionality
    print("\n🔄 Testing rollback functionality...")
    if v2:
        # Simulate performance monitoring
        await manager.monitor_skill_performance(skill_id, v2.version)

    print("\n🎯 SKILL VERSIONING SYSTEM OPERATIONAL")
    print("System is now tracking skill versions and managing deployments")


if __name__ == "__main__":
    asyncio.run(demonstrate_skill_versioning())
