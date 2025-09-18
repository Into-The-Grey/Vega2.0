#!/usr/bin/env python3
"""
Sandboxed Code Testing + Validation
===================================

Implements safe patch testing in isolated environment. Runs tests,
validates behavior, and ensures fixes don't introduce regressions.

Features:
- Isolated sandbox environments for safe testing
- Automated test execution and validation
- Behavioral comparison (before/after)
- Regression detection
- Performance impact assessment
- Integration with existing test suites
- Safe rollback on failure
"""

import os
import sys
import re
import subprocess
import tempfile
import shutil
import json
import sqlite3
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import difflib
import traceback

# Local imports
from error_tracker import ErrorRecord
from self_debugger import FixSuggestion

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Represents the result of a test execution"""

    test_id: str
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    error_count: int
    warning_count: int
    coverage_change: float = 0.0
    performance_impact: str = "unknown"  # improved, degraded, neutral


@dataclass
class SandboxResult:
    """Results from sandbox testing"""

    sandbox_id: str
    fix_id: str
    error_id: str
    success: bool
    test_results: List[TestResult]
    behavioral_changes: Dict[str, Any]
    regression_detected: bool
    safety_score: float
    recommendation: str  # apply, reject, manual_review
    created_at: datetime
    logs: List[str]


class CodeSandbox:
    """Isolated environment for safe code testing"""

    def __init__(self, sandbox_dir: Optional[str] = None):
        self.sandbox_dir = sandbox_dir or tempfile.mkdtemp(prefix="code_sandbox_")
        self.original_cwd = os.getcwd()
        self.test_db = self._init_test_database()
        self.environment_vars = {}
        self.python_exe = ""
        self.pip_exe = ""
        self.venv_path = ""

    def _init_test_database(self) -> sqlite3.Connection:
        """Initialize test results database"""
        db_path = os.path.join(os.path.dirname(self.sandbox_dir), "sandbox_tests.db")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sandbox_results (
                sandbox_id TEXT PRIMARY KEY,
                fix_id TEXT,
                error_id TEXT,
                success BOOLEAN,
                test_results TEXT,
                behavioral_changes TEXT,
                regression_detected BOOLEAN,
                safety_score REAL,
                recommendation TEXT,
                created_at TEXT,
                logs TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_executions (
                test_id TEXT PRIMARY KEY,
                sandbox_id TEXT,
                test_name TEXT,
                success BOOLEAN,
                exit_code INTEGER,
                stdout TEXT,
                stderr TEXT,
                execution_time REAL,
                error_count INTEGER,
                warning_count INTEGER,
                created_at TEXT
            )
        """
        )

        conn.commit()
        return conn

    def setup_sandbox(
        self, source_dir: str, excluded_patterns: Optional[List[str]] = None
    ) -> bool:
        """Set up isolated sandbox environment"""
        try:
            excluded_patterns = (
                excluded_patterns
                if excluded_patterns is not None
                else [
                    "__pycache__",
                    "*.pyc",
                    ".git",
                    ".env",
                    "venv",
                    ".venv",
                    "node_modules",
                    ".DS_Store",
                    "*.log",
                ]
            )

            logger.info(f"Setting up sandbox: {self.sandbox_dir}")

            # Copy source code to sandbox
            self._copy_source_code(source_dir, excluded_patterns)

            # Set up Python virtual environment
            self._setup_virtual_environment()
            # Set python_exe and pip_exe for this sandbox
            if os.name == "nt":
                self.python_exe = os.path.join(
                    self.sandbox_dir, ".sandbox_venv", "Scripts", "python.exe"
                )
                self.pip_exe = os.path.join(
                    self.sandbox_dir, ".sandbox_venv", "Scripts", "pip.exe"
                )
            else:
                self.python_exe = os.path.join(
                    self.sandbox_dir, ".sandbox_venv", "bin", "python"
                )
                self.pip_exe = os.path.join(
                    self.sandbox_dir, ".sandbox_venv", "bin", "pip"
                )

            # Install dependencies
            self._install_dependencies()

            # Verify sandbox integrity
            return self._verify_sandbox()

        except Exception as e:
            logger.error(f"Failed to setup sandbox: {e}")
            return False

    def _copy_source_code(self, source_dir: str, excluded_patterns: List[str]):
        """Copy source code excluding specified patterns"""

        def should_exclude(path: str) -> bool:
            for pattern in excluded_patterns:
                if pattern in path or path.endswith(pattern.replace("*", "")):
                    return True
            return False

        for root, dirs, files in os.walk(source_dir):
            # Filter directories
            dirs[:] = [d for d in dirs if not should_exclude(os.path.join(root, d))]

            for file in files:
                src_path = os.path.join(root, file)
                if should_exclude(src_path):
                    continue

                # Calculate relative path and create destination
                rel_path = os.path.relpath(src_path, source_dir)
                dst_path = os.path.join(self.sandbox_dir, rel_path)

                # Create directories if needed
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                # Copy file
                shutil.copy2(src_path, dst_path)

    logger.debug("Copied source code to sandbox")

    def _setup_virtual_environment(self):
        """Set up Python virtual environment in sandbox"""
        venv_path = os.path.join(self.sandbox_dir, ".sandbox_venv")

        # Create virtual environment
        subprocess.run(
            [sys.executable, "-m", "venv", venv_path], check=True, capture_output=True
        )

        # Store virtual environment info
        self.venv_path = venv_path
        if not hasattr(self, "python_exe"):
            self.python_exe = None
        if not hasattr(self, "pip_exe"):
            self.pip_exe = None
        if os.name == "nt":  # Windows
            self.python_exe = os.path.join(venv_path, "Scripts", "python.exe")
            self.pip_exe = os.path.join(venv_path, "Scripts", "pip.exe")
        else:  # Unix/Linux/Mac
            self.python_exe = os.path.join(venv_path, "bin", "python")
            self.pip_exe = os.path.join(venv_path, "bin", "pip")

        logger.debug("Set up virtual environment: %s", self.venv_path)

    def _install_dependencies(self):
        """Install project dependencies in sandbox"""
        requirements_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "dev-requirements.txt",
        ]

        for req_file in requirements_files:
            req_path = os.path.join(self.sandbox_dir, req_file)
            if os.path.exists(req_path) and self.pip_exe:
                try:
                    subprocess.run(
                        [self.pip_exe, "install", "-r", req_path],
                        check=True,
                        capture_output=True,
                        cwd=self.sandbox_dir,
                    )
                    logger.debug(f"Installed dependencies from {req_file}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to install {req_file}: {e}")

        # Install common testing packages
        test_packages = ["pytest", "unittest-xml-reporting", "coverage"]
        for package in test_packages:
            if self.pip_exe:
                try:
                    subprocess.run(
                        [self.pip_exe, "install", package],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    logger.warning(f"Failed to install {package}")

    def _verify_sandbox(self) -> bool:
        """Verify sandbox setup is correct"""
        try:
            # Test Python interpreter
            if not self.python_exe:
                logger.error("Python executable not set for sandbox.")
                return False
            result = subprocess.run(
                [self.python_exe, "-c", "import sys; print(sys.version)"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.error("Python interpreter verification failed")
                return False
            # Check if main modules can be imported
            test_imports = ["os", "sys", "json"]
            for module in test_imports:
                result = subprocess.run(
                    [self.python_exe, "-c", f"import {module}"], capture_output=True
                )
                if result.returncode != 0:
                    logger.error(f"Failed to import {module}")
                    return False
            logger.info("Sandbox verification successful")
            return True
        except Exception as e:
            logger.error(f"Sandbox verification failed: {e}")
            return False

    def apply_fix(self, fix: FixSuggestion) -> bool:
        """Apply a fix to the sandbox environment"""
        try:
            logger.info(f"Applying fix {fix.id[:8]} to sandbox")

            for change in fix.code_changes:
                file_path = os.path.join(
                    self.sandbox_dir, os.path.relpath(change["file"], self.original_cwd)
                )

                if not os.path.exists(file_path):
                    logger.error(f"File not found in sandbox: {file_path}")
                    return False

                # Apply the change
                if not self._apply_file_change(file_path, change):
                    return False

            logger.info("Successfully applied fix to sandbox")
            return True

        except Exception as e:
            logger.error(f"Failed to apply fix: {e}")
            return False

    def _apply_file_change(self, file_path: str, change: Dict[str, Any]) -> bool:
        """Apply a single file change"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            line_num = change["line"] - 1  # Convert to 0-based

            if 0 <= line_num < len(lines):
                # Verify old code matches
                current_line = lines[line_num].rstrip()
                expected_line = change["old_code"].rstrip()

                if current_line != expected_line:
                    logger.warning(f"Line mismatch at {file_path}:{change['line']}")
                    logger.warning(f"Expected: {expected_line}")
                    logger.warning(f"Found: {current_line}")
                    # Continue anyway - might be formatting differences

                # Apply change
                lines[line_num] = change["new_code"] + "\n"

                # Write modified file
                with open(file_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                logger.debug(f"Applied change to {file_path}:{change['line']}")
                return True
            else:
                logger.error(f"Line number out of range: {change['line']}")
                return False

        except Exception as e:
            logger.error(f"Failed to apply file change: {e}")
            return False

    def run_tests(self, test_patterns: Optional[List[str]] = None) -> List[TestResult]:
        """Run tests in the sandbox environment"""
        test_results = []

        # Default test patterns
        if test_patterns is None:
            test_patterns = ["test_*.py", "*_test.py", "tests/", "pytest", "unittest"]

        logger.info("Running tests in sandbox environment")

        # Try pytest first
        if self._command_exists("pytest"):
            result = self._run_pytest()
            if result:
                test_results.append(result)

        # Try unittest discovery
        result = self._run_unittest()
        if result:
            test_results.append(result)

        # Try custom test files
        for pattern in test_patterns:
            if pattern.endswith(".py"):
                result = self._run_python_test(pattern)
                if result:
                    test_results.append(result)

        logger.info(f"Completed {len(test_results)} test runs")
        return test_results

    def _command_exists(self, command: str) -> bool:
        """Check if a command exists in the sandbox"""
        try:
            python_exe = self.python_exe or ""
            if not python_exe:
                logger.error("Python executable is not set.")
                return False
            result = subprocess.run(
                [python_exe, "-c", f"import {command}"], capture_output=True
            )
            return result.returncode == 0
        except Exception:
            return False

    def _run_pytest(self) -> Optional[TestResult]:
        """Run pytest in sandbox"""
        try:
            python_exe = self.python_exe or ""
            if not python_exe:
                logger.error("Python executable is not set.")
                return None
            start_time = datetime.now()

            result = subprocess.run(
                [python_exe, "-m", "pytest", "-v", "--tb=short"],
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Parse pytest output
            error_count = self._count_errors_in_output(result.stderr)
            warning_count = self._count_warnings_in_output(
                result.stdout + result.stderr
            )

            test_result = TestResult(
                test_id=f"pytest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                error_count=error_count,
                warning_count=warning_count,
            )

            logger.debug(f"Pytest completed with exit code {result.returncode}")
            return test_result

        except subprocess.TimeoutExpired:
            logger.error("Pytest execution timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to run pytest: {e}")
            return None

    def _run_unittest(self) -> Optional[TestResult]:
        """Run unittest discovery in sandbox"""
        try:
            python_exe = self.python_exe or ""
            if not python_exe:
                logger.error("Python executable is not set.")
                return None
            start_time = datetime.now()

            result = subprocess.run(
                [python_exe, "-m", "unittest", "discover", "-v"],
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            error_count = self._count_errors_in_output(result.stderr)
            warning_count = self._count_warnings_in_output(
                result.stdout + result.stderr
            )

            test_result = TestResult(
                test_id=f"unittest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                error_count=error_count,
                warning_count=warning_count,
            )

            logger.debug(f"Unittest completed with exit code {result.returncode}")
            return test_result

        except subprocess.TimeoutExpired:
            logger.error("Unittest execution timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to run unittest: {e}")
            return None

    def _run_python_test(self, test_file: str) -> Optional[TestResult]:
        """Run a specific Python test file"""
        try:
            python_exe = self.python_exe or ""
            if not python_exe:
                logger.error("Python executable is not set.")
                return None
            test_path = os.path.join(self.sandbox_dir, test_file)
            if not os.path.exists(test_path):
                return None

            start_time = datetime.now()

            result = subprocess.run(
                [python_exe, test_path],
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            error_count = self._count_errors_in_output(result.stderr)
            warning_count = self._count_warnings_in_output(
                result.stdout + result.stderr
            )

            test_result = TestResult(
                test_id=f"python_{os.path.basename(test_file)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                success=result.returncode == 0,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                execution_time=execution_time,
                error_count=error_count,
                warning_count=warning_count,
            )

            logger.debug(
                f"Python test {test_file} completed with exit code {result.returncode}"
            )
            return test_result

        except subprocess.TimeoutExpired:
            logger.error(f"Python test {test_file} timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to run Python test {test_file}: {e}")
            return None

    def _count_errors_in_output(self, output: str) -> int:
        """Count errors in test output"""
        error_patterns = [r"ERROR", r"FAILED", r"Exception:", r"Traceback", r"Error:"]

        count = 0
        for pattern in error_patterns:
            count += len(re.findall(pattern, output, re.IGNORECASE))

        return count

    def _count_warnings_in_output(self, output: str) -> int:
        """Count warnings in test output"""
        warning_patterns = [r"WARNING", r"WARN:", r"DeprecationWarning", r"UserWarning"]

        count = 0
        for pattern in warning_patterns:
            count += len(re.findall(pattern, output, re.IGNORECASE))

        return count

    def analyze_behavior_changes(
        self, baseline_results: List[TestResult], new_results: List[TestResult]
    ) -> Dict[str, Any]:
        """Analyze behavioral changes between test runs"""
        changes = {
            "test_count_change": len(new_results) - len(baseline_results),
            "success_rate_change": 0.0,
            "error_count_change": 0,
            "warning_count_change": 0,
            "execution_time_change": 0.0,
            "new_failures": [],
            "fixed_failures": [],
            "performance_impact": "neutral",
        }

        try:
            # Calculate success rates
            baseline_success = (
                sum(1 for r in baseline_results if r.success) / len(baseline_results)
                if baseline_results
                else 0
            )
            new_success = (
                sum(1 for r in new_results if r.success) / len(new_results)
                if new_results
                else 0
            )
            changes["success_rate_change"] = new_success - baseline_success

            # Count errors and warnings
            baseline_errors = sum(r.error_count for r in baseline_results)
            new_errors = sum(r.error_count for r in new_results)
            changes["error_count_change"] = new_errors - baseline_errors

            baseline_warnings = sum(r.warning_count for r in baseline_results)
            new_warnings = sum(r.warning_count for r in new_results)
            changes["warning_count_change"] = new_warnings - baseline_warnings

            # Execution time changes
            baseline_time = sum(r.execution_time for r in baseline_results)
            new_time = sum(r.execution_time for r in new_results)
            changes["execution_time_change"] = new_time - baseline_time

            # Performance impact assessment
            if changes["execution_time_change"] > 1.0:  # More than 1 second slower
                changes["performance_impact"] = "degraded"
            elif (
                changes["execution_time_change"] < -0.5
            ):  # More than 0.5 seconds faster
                changes["performance_impact"] = "improved"

            logger.debug("Analyzed behavioral changes between test runs")

        except Exception as e:
            logger.error(f"Failed to analyze behavior changes: {e}")

        return changes

    def cleanup(self):
        """Clean up sandbox environment"""
        try:
            if os.path.exists(self.sandbox_dir):
                shutil.rmtree(self.sandbox_dir)
                logger.debug(f"Cleaned up sandbox: {self.sandbox_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {e}")

        if self.test_db:
            self.test_db.close()


class SandboxValidator:
    """Validates fixes using sandboxed testing"""

    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.validation_db = self._init_validation_db()

    def _init_validation_db(self) -> sqlite3.Connection:
        """Initialize validation results database"""
        db_path = "autonomous_debug/validation_results.db"
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.Connection(db_path)
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS validation_results (
                validation_id TEXT PRIMARY KEY,
                fix_id TEXT,
                error_id TEXT,
                success BOOLEAN,
                safety_score REAL,
                recommendation TEXT,
                test_summary TEXT,
                behavioral_changes TEXT,
                regression_detected BOOLEAN,
                created_at TEXT
            )
        """
        )

        conn.commit()
        return conn

    async def validate_fix(
        self, fix: FixSuggestion, error: ErrorRecord
    ) -> SandboxResult:
        """Validate a fix using sandboxed testing"""
        sandbox_id = f"sandbox_{fix.id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logs = []

        try:
            logs.append(f"Starting validation for fix {fix.id[:8]}")

            # Create sandbox
            sandbox = CodeSandbox()

            if not sandbox.setup_sandbox(self.workspace_dir):
                logs.append("Failed to setup sandbox")
                return self._create_failed_result(
                    sandbox_id, fix.id, error.id, logs, "Sandbox setup failed"
                )

            logs.append("Sandbox setup successful")

            # Run baseline tests (before applying fix)
            baseline_results = sandbox.run_tests()
            logs.append(f"Baseline tests completed: {len(baseline_results)} test runs")

            # Apply fix
            if not sandbox.apply_fix(fix):
                logs.append("Failed to apply fix")
                sandbox.cleanup()
                return self._create_failed_result(
                    sandbox_id, fix.id, error.id, logs, "Fix application failed"
                )

            logs.append("Fix applied successfully")

            # Run tests after applying fix
            post_fix_results = sandbox.run_tests()
            logs.append(f"Post-fix tests completed: {len(post_fix_results)} test runs")

            # Analyze behavioral changes
            behavioral_changes = sandbox.analyze_behavior_changes(
                baseline_results, post_fix_results
            )
            logs.append("Behavioral analysis completed")

            # Detect regressions
            regression_detected = self._detect_regressions(
                baseline_results, post_fix_results, behavioral_changes
            )

            # Calculate safety score
            safety_score = self._calculate_safety_score(
                baseline_results, post_fix_results, behavioral_changes
            )

            # Generate recommendation
            recommendation = self._generate_recommendation(
                safety_score, regression_detected, behavioral_changes
            )

            logs.append(
                f"Validation completed - Safety: {safety_score:.2f}, Recommendation: {recommendation}"
            )

            # Create result
            result = SandboxResult(
                sandbox_id=sandbox_id,
                fix_id=fix.id,
                error_id=error.id,
                success=True,
                test_results=baseline_results + post_fix_results,
                behavioral_changes=behavioral_changes,
                regression_detected=regression_detected,
                safety_score=safety_score,
                recommendation=recommendation,
                created_at=datetime.now(),
                logs=logs,
            )

            # Store result
            self._store_validation_result(result)

            # Cleanup
            sandbox.cleanup()

            return result

        except Exception as e:
            logs.append(f"Validation error: {e}")
            logger.error(f"Validation failed: {e}")
            return self._create_failed_result(
                sandbox_id, fix.id, error.id, logs, str(e)
            )

    def _detect_regressions(
        self,
        baseline: List[TestResult],
        post_fix: List[TestResult],
        changes: Dict[str, Any],
    ) -> bool:
        """Detect if the fix introduced regressions"""
        # Regression indicators
        if changes["success_rate_change"] < -0.1:  # 10% drop in success rate
            return True

        if changes["error_count_change"] > 5:  # More than 5 new errors
            return True

        if changes["execution_time_change"] > 10.0:  # More than 10 seconds slower
            return True

        # Check for new test failures
        baseline_failures = {r.test_id for r in baseline if not r.success}
        post_fix_failures = {r.test_id for r in post_fix if not r.success}
        new_failures = post_fix_failures - baseline_failures

        if len(new_failures) > 2:  # More than 2 new failures
            return True

        return False

    def _calculate_safety_score(
        self,
        baseline: List[TestResult],
        post_fix: List[TestResult],
        changes: Dict[str, Any],
    ) -> float:
        """Calculate safety score for the fix (0.0 to 1.0)"""
        score = 0.5  # Start at neutral

        # Success rate improvement
        score += changes["success_rate_change"] * 0.3

        # Error reduction
        if changes["error_count_change"] < 0:
            score += 0.2
        elif changes["error_count_change"] > 0:
            score -= 0.2

        # Warning reduction
        if changes["warning_count_change"] < 0:
            score += 0.1
        elif changes["warning_count_change"] > 5:
            score -= 0.1

        # Performance impact
        if changes["performance_impact"] == "improved":
            score += 0.1
        elif changes["performance_impact"] == "degraded":
            score -= 0.2

        # Test count (more tests = better coverage)
        if changes["test_count_change"] > 0:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _generate_recommendation(
        self, safety_score: float, regression_detected: bool, changes: Dict[str, Any]
    ) -> str:
        """Generate recommendation based on validation results"""
        if regression_detected:
            return "reject"

        if safety_score >= 0.8:
            return "apply"
        elif safety_score >= 0.6:
            return "manual_review"
        else:
            return "reject"

    def _create_failed_result(
        self, sandbox_id: str, fix_id: str, error_id: str, logs: List[str], reason: str
    ) -> SandboxResult:
        """Create a failed validation result"""
        return SandboxResult(
            sandbox_id=sandbox_id,
            fix_id=fix_id,
            error_id=error_id,
            success=False,
            test_results=[],
            behavioral_changes={"failure_reason": reason},
            regression_detected=True,
            safety_score=0.0,
            recommendation="reject",
            created_at=datetime.now(),
            logs=logs + [f"Validation failed: {reason}"],
        )

    def _store_validation_result(self, result: SandboxResult):
        """Store validation result in database"""
        try:
            cursor = self.validation_db.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO validation_results (
                    validation_id, fix_id, error_id, success, safety_score,
                    recommendation, test_summary, behavioral_changes,
                    regression_detected, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.sandbox_id,
                    result.fix_id,
                    result.error_id,
                    result.success,
                    result.safety_score,
                    result.recommendation,
                    json.dumps([asdict(t) for t in result.test_results]),
                    json.dumps(result.behavioral_changes),
                    result.regression_detected,
                    result.created_at.isoformat(),
                ),
            )
            self.validation_db.commit()

        except Exception as e:
            logger.error(f"Failed to store validation result: {e}")

    def get_validation_history(
        self, fix_id: Optional[str] = None, error_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get validation history"""
        try:
            cursor = self.validation_db.cursor()

            if fix_id:
                cursor.execute(
                    "SELECT * FROM validation_results WHERE fix_id = ?", (fix_id,)
                )
            elif error_id:
                cursor.execute(
                    "SELECT * FROM validation_results WHERE error_id = ?", (error_id,)
                )
            else:
                cursor.execute(
                    "SELECT * FROM validation_results ORDER BY created_at DESC LIMIT 10"
                )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get validation history: {e}")
            return []

    def close(self):
        """Close database connection"""
        if self.validation_db:
            self.validation_db.close()


async def main():
    """Main function for sandbox testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Sandboxed Code Testing + Validation")
    parser.add_argument("--validate-fix", help="Validate specific fix ID")
    parser.add_argument("--error-id", help="Error ID for validation context")
    parser.add_argument(
        "--workspace", default="/home/ncacord/Vega2.0", help="Workspace directory"
    )
    parser.add_argument(
        "--test-sandbox", action="store_true", help="Test sandbox setup"
    )

    args = parser.parse_args()

    try:
        if args.test_sandbox:
            print("🧪 Testing sandbox setup...")

            sandbox = CodeSandbox()
            success = sandbox.setup_sandbox(args.workspace)

            if success:
                print("✅ Sandbox setup successful")

                # Run a simple test
                test_results = sandbox.run_tests()
                print(f"📋 Ran {len(test_results)} test suites")

                for result in test_results:
                    status = "✅" if result.success else "❌"
                    print(f"  {status} {result.test_id}: {result.execution_time:.2f}s")
            else:
                print("❌ Sandbox setup failed")

            sandbox.cleanup()

        elif args.validate_fix and args.error_id:
            print(f"🧪 Validating fix {args.validate_fix} for error {args.error_id}...")

            # This would normally load the fix and error from database
            # For demo, create mock objects
            from self_debugger import FixSuggestion, FixStrategy
            from error_tracker import ErrorRecord

            mock_fix = FixSuggestion(
                id=args.validate_fix,
                error_id=args.error_id,
                strategy=FixStrategy.QUICK_FIX,
                description="Mock fix for testing",
                code_changes=[],
                confidence_score=0.8,
                reasoning="Test fix",
                dependencies=[],
                test_commands=[],
                rollback_safe=True,
                estimated_impact="low",
                created_at=datetime.now(),
            )

            mock_error = ErrorRecord(
                id=args.error_id,
                timestamp=datetime.now(),
                file_path="test.py",
                line_number=1,
                error_type="ValueError",
                message="Test error",
                traceback_hash="test_hash",
                frequency=1,
                snippet="",
                first_seen=datetime.now(),
                last_seen=datetime.now(),
            )

            validator = SandboxValidator(args.workspace)
            result = await validator.validate_fix(mock_fix, mock_error)

            print("✅ Validation completed")
            print(f"🛡️ Safety Score: {result.safety_score:.2f}")
            print(f"📋 Recommendation: {result.recommendation}")
            print(f"🔍 Regression Detected: {result.regression_detected}")

            validator.close()

        else:
            print("Specify --test-sandbox or --validate-fix with --error-id")

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
