#!/usr/bin/env python3
"""
LLM-Powered Self-Debugging Engine
================================

Analyzes errors using LLM, understands code context, generates targeted
fixes, and validates solutions before applying. Includes confidence
scoring and multiple fix strategies.

Features:
- Context-aware error analysis with LLM integration
- Multi-strategy fix generation (quick fix, refactor, architectural)
- Code understanding and dependency analysis
- Confidence scoring for fix reliability
- Validation pipeline before applying changes
- Learning from previous successful fixes
"""

import os
import re
import ast
import sys
import json
import sqlite3
import inspect
import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import logging

# External dependencies
import httpx
from error_tracker import ErrorDatabase, ErrorRecord

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Types of fix strategies"""

    QUICK_FIX = "quick_fix"  # Simple one-line fixes
    REFACTOR = "refactor"  # Code restructuring
    DEPENDENCY = "dependency"  # Library/import fixes
    ARCHITECTURAL = "architectural"  # Design pattern changes
    CONFIG = "config"  # Configuration adjustments


@dataclass
class FixSuggestion:
    """Generated fix suggestion with metadata"""

    id: str
    error_id: str
    strategy: FixStrategy
    description: str
    code_changes: List[Dict[str, Any]]  # File, line, old_code, new_code
    confidence_score: float
    reasoning: str
    dependencies: List[str]
    test_commands: List[str]
    rollback_safe: bool
    estimated_impact: str  # low, medium, high
    created_at: datetime


class CodeAnalyzer:
    """Analyzes code structure and context around errors"""

    def __init__(self):
        self.ast_cache = {}  # Cache parsed ASTs

    def analyze_error_context(self, error: ErrorRecord) -> Dict[str, Any]:
        """Analyze the code context around an error"""
        context = {
            "file_info": {},
            "function_context": {},
            "class_context": {},
            "imports": [],
            "dependencies": [],
            "code_structure": {},
            "related_files": [],
        }

        try:
            if not error.file_path or not os.path.exists(error.file_path):
                return context

            # Analyze file structure
            context["file_info"] = self._analyze_file_structure(error.file_path)

            # Get function/class context around error line
            if error.line_number:
                context["function_context"] = self._get_function_context(
                    error.file_path, error.line_number
                )
                context["class_context"] = self._get_class_context(
                    error.file_path, error.line_number
                )

            # Extract imports and dependencies
            context["imports"] = self._extract_imports(error.file_path)
            context["dependencies"] = self._identify_dependencies(error.file_path)

            # Find related files
            context["related_files"] = self._find_related_files(error.file_path)

            logger.debug(f"Analyzed context for {error.file_path}:{error.line_number}")

        except Exception as e:
            logger.error(f"Failed to analyze error context: {e}")

        return context

    def _analyze_file_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze the overall structure of a Python file"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST if Python file
            if file_path.endswith(".py"):
                tree = ast.parse(content)

                return {
                    "line_count": len(content.split("\n")),
                    "functions": [
                        node.name
                        for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef)
                    ],
                    "classes": [
                        node.name
                        for node in ast.walk(tree)
                        if isinstance(node, ast.ClassDef)
                    ],
                    "imports": [
                        self._get_import_name(node)
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.Import, ast.ImportFrom))
                    ],
                    "complexity_score": self._calculate_complexity(tree),
                }
            else:
                # Non-Python file analysis
                lines = content.split("\n")
                return {
                    "line_count": len(lines),
                    "file_type": Path(file_path).suffix,
                    "size_kb": len(content) / 1024,
                }

        except Exception as e:
            logger.error(f"Failed to analyze file structure: {e}")
            return {}

    def _get_function_context(self, file_path: str, line_number: int) -> Dict[str, Any]:
        """Get the function containing the error line"""
        try:
            if not file_path.endswith(".py"):
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        if (
                            node.lineno
                            <= line_number
                            <= (node.end_lineno or node.lineno)
                        ):
                            return {
                                "name": node.name,
                                "args": [arg.arg for arg in node.args.args],
                                "decorators": [
                                    d.id if hasattr(d, "id") else str(d)
                                    for d in node.decorator_list
                                ],
                                "docstring": ast.get_docstring(node),
                                "start_line": node.lineno,
                                "end_line": node.end_lineno,
                            }

        except Exception as e:
            logger.error(f"Failed to get function context: {e}")

        return {}

    def _get_class_context(self, file_path: str, line_number: int) -> Dict[str, Any]:
        """Get the class containing the error line"""
        try:
            if not file_path.endswith(".py"):
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                        if (
                            node.lineno
                            <= line_number
                            <= (node.end_lineno or node.lineno)
                        ):
                            return {
                                "name": node.name,
                                "bases": [
                                    b.id if hasattr(b, "id") else str(b)
                                    for b in node.bases
                                ],
                                "methods": [
                                    n.name
                                    for n in node.body
                                    if isinstance(n, ast.FunctionDef)
                                ],
                                "docstring": ast.get_docstring(node),
                                "start_line": node.lineno,
                                "end_line": node.end_lineno,
                            }

        except Exception as e:
            logger.error(f"Failed to get class context: {e}")

        return {}

    def _extract_imports(self, file_path: str) -> List[Dict[str, str]]:
        """Extract all imports from a Python file"""
        imports = []

        try:
            if not file_path.endswith(".py"):
                return imports

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(
                            {
                                "type": "import",
                                "module": alias.name,
                                "alias": alias.asname or alias.name,
                            }
                        )
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(
                            {
                                "type": "from_import",
                                "module": module,
                                "name": alias.name,
                                "alias": alias.asname or alias.name,
                            }
                        )

        except Exception as e:
            logger.error(f"Failed to extract imports: {e}")

        return imports

    def _identify_dependencies(self, file_path: str) -> List[str]:
        """Identify external dependencies used in the file"""
        dependencies = set()

        try:
            imports = self._extract_imports(file_path)

            # Standard library modules to exclude
            stdlib_modules = {
                "os",
                "sys",
                "json",
                "sqlite3",
                "datetime",
                "logging",
                "re",
                "ast",
                "inspect",
                "asyncio",
                "traceback",
                "pathlib",
                "typing",
                "dataclasses",
                "enum",
                "collections",
                "itertools",
            }

            for imp in imports:
                module = imp["module"].split(".")[0]  # Get top-level module
                if module not in stdlib_modules and not module.startswith("_"):
                    dependencies.add(module)

        except Exception as e:
            logger.error(f"Failed to identify dependencies: {e}")

        return list(dependencies)

    def _find_related_files(self, file_path: str, max_files: int = 10) -> List[str]:
        """Find files related to the error file"""
        related_files = []

        try:
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            file_stem = Path(file_path).stem

            # Look for related files in same directory
            for item in os.listdir(file_dir):
                item_path = os.path.join(file_dir, item)

                if os.path.isfile(item_path) and item != file_name:
                    # Check for naming patterns
                    if (
                        item.startswith(file_stem)
                        or file_stem in item
                        or item.endswith("_test.py")
                        or item.endswith("test_.py")
                    ):
                        related_files.append(item_path)

                if len(related_files) >= max_files:
                    break

        except Exception as e:
            logger.error(f"Failed to find related files: {e}")

        return related_files

    def _get_import_name(self, node) -> str:
        """Extract import name from AST node"""
        if isinstance(node, ast.Import):
            return node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            return (
                f"{node.module}.{node.names[0].name}"
                if node.module
                else node.names[0].name
            )
        return ""

    def _calculate_complexity(self, tree) -> int:
        """Calculate rough complexity score for the file"""
        complexity = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity += 1
            elif isinstance(node, ast.ClassDef):
                complexity += 2
            elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1

        return complexity


class LLMDebugger:
    """LLM-powered debugging engine"""

    def __init__(
        self, api_base_url: str = "http://127.0.0.1:11434", model: str = "llama3.1:8b"
    ):
        self.api_base_url = api_base_url
        self.model = model
        self.analyzer = CodeAnalyzer()
        self.fix_history = {}  # Cache successful fixes

    async def analyze_error(self, error: ErrorRecord) -> Dict[str, Any]:
        """Analyze error using LLM and return insights"""
        try:
            # Get code context
            context = self.analyzer.analyze_error_context(error)

            # Prepare prompt for LLM
            prompt = self._build_analysis_prompt(error, context)

            # Query LLM
            response = await self._query_llm(prompt)

            # Parse LLM response
            analysis = self._parse_analysis_response(response)

            # Add context information
            analysis["code_context"] = context
            analysis["error_metadata"] = {
                "frequency": error.frequency,
                "severity": error.severity,
                "first_seen": error.first_seen.isoformat(),
                "last_seen": error.last_seen.isoformat(),
            }

            logger.info(f"Analyzed error {error.id[:8]} with LLM")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze error with LLM: {e}")
            return {}

    async def generate_fixes(
        self, error: ErrorRecord, analysis: Dict[str, Any]
    ) -> List[FixSuggestion]:
        """Generate multiple fix suggestions using different strategies"""
        fixes = []

        try:
            # Generate fixes for each strategy
            strategies = [
                FixStrategy.QUICK_FIX,
                FixStrategy.REFACTOR,
                FixStrategy.DEPENDENCY,
            ]

            for strategy in strategies:
                fix = await self._generate_strategy_fix(error, analysis, strategy)
                if fix:
                    fixes.append(fix)

            # Sort by confidence score
            fixes.sort(key=lambda x: x.confidence_score, reverse=True)

            logger.info(
                f"Generated {len(fixes)} fix suggestions for error {error.id[:8]}"
            )

        except Exception as e:
            logger.error(f"Failed to generate fixes: {e}")

        return fixes

    async def _generate_strategy_fix(
        self, error: ErrorRecord, analysis: Dict[str, Any], strategy: FixStrategy
    ) -> Optional[FixSuggestion]:
        """Generate a fix for a specific strategy"""
        try:
            # Build strategy-specific prompt
            prompt = self._build_fix_prompt(error, analysis, strategy)

            # Query LLM for fix
            response = await self._query_llm(prompt)

            # Parse fix response
            fix_data = self._parse_fix_response(response, strategy)

            if not fix_data:
                return None

            # Create fix suggestion
            fix = FixSuggestion(
                id=self._generate_fix_id(error.id, strategy),
                error_id=error.id,
                strategy=strategy,
                description=fix_data.get("description", ""),
                code_changes=fix_data.get("code_changes", []),
                confidence_score=fix_data.get("confidence_score", 0.5),
                reasoning=fix_data.get("reasoning", ""),
                dependencies=fix_data.get("dependencies", []),
                test_commands=fix_data.get("test_commands", []),
                rollback_safe=fix_data.get("rollback_safe", True),
                estimated_impact=fix_data.get("estimated_impact", "medium"),
                created_at=datetime.now(),
            )

            return fix

        except Exception as e:
            logger.error(f"Failed to generate {strategy.value} fix: {e}")
            return None

    def _build_analysis_prompt(
        self, error: ErrorRecord, context: Dict[str, Any]
    ) -> str:
        """Build LLM prompt for error analysis"""
        prompt = f"""Analyze this error and provide insights:

ERROR DETAILS:
Type: {error.error_type}
Message: {error.message}
File: {error.file_path}:{error.line_number}
Frequency: {error.frequency}
Severity: {error.severity}

TRACEBACK:
{error.full_traceback}

CODE SNIPPET:
{error.snippet}

CONTEXT:
- Function: {context.get('function_context', {}).get('name', 'Unknown')}
- Class: {context.get('class_context', {}).get('name', 'Unknown')}
- Imports: {context.get('imports', [])}
- Dependencies: {context.get('dependencies', [])}

Please provide:
1. Root cause analysis
2. Contributing factors
3. Error category (syntax, logic, runtime, dependency, etc.)
4. Complexity assessment (simple, moderate, complex)
5. Recommended fix approach
6. Potential side effects
7. Testing requirements

Respond in JSON format."""

        return prompt

    def _build_fix_prompt(
        self, error: ErrorRecord, analysis: Dict[str, Any], strategy: FixStrategy
    ) -> str:
        """Build LLM prompt for fix generation"""
        strategy_guidance = {
            FixStrategy.QUICK_FIX: "Provide a minimal, single-line fix that directly addresses the error",
            FixStrategy.REFACTOR: "Restructure the code to eliminate the error and improve maintainability",
            FixStrategy.DEPENDENCY: "Fix import/dependency issues or version conflicts",
            FixStrategy.ARCHITECTURAL: "Redesign the component using better patterns",
            FixStrategy.CONFIG: "Adjust configuration or environment settings",
        }

        prompt = f"""Generate a {strategy.value} for this error:

ERROR: {error.error_type}: {error.message}
FILE: {error.file_path}:{error.line_number}

STRATEGY: {strategy_guidance.get(strategy, '')}

CODE CONTEXT:
{error.snippet}

ANALYSIS:
{json.dumps(analysis, indent=2)}

Provide a specific fix with:
1. Description of the fix
2. Exact code changes (file, line, old_code, new_code)
3. Confidence score (0.0-1.0)
4. Reasoning for the fix
5. Required dependencies
6. Test commands to validate
7. Rollback safety (true/false)
8. Estimated impact (low/medium/high)

Respond in JSON format with precise code changes."""

        return prompt

    async def _query_llm(self, prompt: str) -> str:
        """Query the LLM with a prompt"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent fixes
                            "top_p": 0.9,
                            "num_ctx": 4096,
                        },
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"LLM query failed: {response.status_code}")
                    # Return fallback response when LLM is unavailable
                    return self._generate_fallback_response(prompt)

        except httpx.ConnectError:
            logger.warning("LLM service unavailable, using fallback analysis")
            return self._generate_fallback_response(prompt)
        except Exception as e:
            logger.error(f"Failed to query LLM: {e}")
            return self._generate_fallback_response(prompt)

    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate fallback response when LLM is unavailable"""
        if "analyze error" in prompt.lower():
            return json.dumps(
                {
                    "root_cause": "Error analysis unavailable - LLM service not running",
                    "category": "unknown",
                    "complexity": "medium",
                    "confidence": 0.3,
                    "fallback": True,
                }
            )
        elif "generate fix" in prompt.lower():
            return json.dumps(
                {
                    "description": "Fallback fix suggestion - please review manually",
                    "code": "# TODO: Manual fix required - LLM unavailable",
                    "explanation": "LLM service is not available for intelligent fix generation",
                    "confidence": 0.1,
                    "fallback": True,
                }
            )
        else:
            return json.dumps(
                {
                    "message": "LLM service unavailable",
                    "fallback": True,
                    "confidence": 0.0,
                }
            )

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM analysis response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")

        # Fallback to structured parsing
        return {
            "root_cause": self._extract_field(response, "root cause"),
            "category": self._extract_field(response, "category"),
            "complexity": self._extract_field(response, "complexity"),
            "raw_response": response,
        }

    def _parse_fix_response(
        self, response: str, strategy: FixStrategy
    ) -> Optional[Dict[str, Any]]:
        """Parse LLM fix response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                fix_data = json.loads(json_match.group())

                # Validate required fields
                if "description" in fix_data and "code_changes" in fix_data:
                    return fix_data

        except Exception as e:
            logger.error(f"Failed to parse fix response: {e}")

        # Fallback parsing
        return {
            "description": self._extract_field(response, "description"),
            "confidence_score": 0.5,
            "reasoning": self._extract_field(response, "reasoning"),
            "code_changes": [],
            "dependencies": [],
            "test_commands": [],
            "rollback_safe": True,
            "estimated_impact": "medium",
        }

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field value from unstructured text"""
        pattern = rf"{field_name}[:\s]*([^\n]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _generate_fix_id(self, error_id: str, strategy: FixStrategy) -> str:
        """Generate unique fix ID"""
        import hashlib

        content = f"{error_id}{strategy.value}{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()


class FixValidator:
    """Validates fix suggestions before application"""

    def __init__(self):
        self.validation_rules = self._load_validation_rules()

    def validate_fix(self, fix: FixSuggestion, error: ErrorRecord) -> Dict[str, Any]:
        """Validate a fix suggestion"""
        validation_result = {
            "is_valid": True,
            "confidence_adjustment": 0.0,
            "warnings": [],
            "blocking_issues": [],
            "recommendations": [],
        }

        try:
            # Validate code changes
            for change in fix.code_changes:
                change_validation = self._validate_code_change(change, error)

                if not change_validation["is_valid"]:
                    validation_result["blocking_issues"].extend(
                        change_validation["issues"]
                    )
                    validation_result["is_valid"] = False

                validation_result["warnings"].extend(change_validation["warnings"])

            # Check for risky patterns
            risk_assessment = self._assess_risk(fix, error)
            validation_result["confidence_adjustment"] = risk_assessment[
                "confidence_adjustment"
            ]
            validation_result["warnings"].extend(risk_assessment["warnings"])

            # Validate dependencies
            dependency_check = self._validate_dependencies(fix.dependencies)
            if not dependency_check["all_available"]:
                validation_result["warnings"].append(
                    f"Missing dependencies: {dependency_check['missing']}"
                )

            logger.info(
                f"Validated fix {fix.id[:8]} - Valid: {validation_result['is_valid']}"
            )

        except Exception as e:
            logger.error(f"Failed to validate fix: {e}")
            validation_result["is_valid"] = False
            validation_result["blocking_issues"].append(f"Validation error: {e}")

        return validation_result

    def _validate_code_change(
        self, change: Dict[str, Any], error: ErrorRecord
    ) -> Dict[str, Any]:
        """Validate an individual code change"""
        result = {"is_valid": True, "warnings": [], "issues": []}

        try:
            # Check required fields
            required_fields = ["file", "line", "old_code", "new_code"]
            missing_fields = [field for field in required_fields if field not in change]

            if missing_fields:
                result["is_valid"] = False
                result["issues"].append(f"Missing fields: {missing_fields}")
                return result

            # Validate file exists
            if not os.path.exists(change["file"]):
                result["is_valid"] = False
                result["issues"].append(f"File does not exist: {change['file']}")
                return result

            # Check if old code matches current content
            if not self._verify_old_code_match(change):
                result["warnings"].append("Old code doesn't match current file content")

            # Validate Python syntax for .py files
            if change["file"].endswith(".py"):
                syntax_check = self._validate_python_syntax(change["new_code"])
                if not syntax_check["is_valid"]:
                    result["is_valid"] = False
                    result["issues"].append(f"Syntax error: {syntax_check['error']}")

        except Exception as e:
            result["is_valid"] = False
            result["issues"].append(f"Change validation error: {e}")

        return result

    def _verify_old_code_match(self, change: Dict[str, Any]) -> bool:
        """Verify that old_code matches the current file content"""
        try:
            with open(change["file"], "r", encoding="utf-8") as f:
                lines = f.readlines()

            line_num = change["line"] - 1  # Convert to 0-based
            if 0 <= line_num < len(lines):
                current_line = lines[line_num].strip()
                expected_line = change["old_code"].strip()
                return current_line == expected_line

        except Exception as e:
            logger.error(f"Failed to verify old code match: {e}")

        return False

    def _validate_python_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax"""
        try:
            ast.parse(code)
            return {"is_valid": True, "error": None}
        except SyntaxError as e:
            return {"is_valid": False, "error": str(e)}

    def _assess_risk(self, fix: FixSuggestion, error: ErrorRecord) -> Dict[str, Any]:
        """Assess risk of applying the fix"""
        risk_assessment = {"confidence_adjustment": 0.0, "warnings": []}

        # Low confidence fixes are riskier
        if fix.confidence_score < 0.6:
            risk_assessment["confidence_adjustment"] -= 0.1
            risk_assessment["warnings"].append("Low confidence fix")

        # High impact changes are riskier
        if fix.estimated_impact == "high":
            risk_assessment["confidence_adjustment"] -= 0.15
            risk_assessment["warnings"].append("High impact change")

        # Multiple file changes are riskier
        files_affected = len(set(change["file"] for change in fix.code_changes))
        if files_affected > 1:
            risk_assessment["confidence_adjustment"] -= 0.05
            risk_assessment["warnings"].append(f"Affects {files_affected} files")

        return risk_assessment

    def _validate_dependencies(self, dependencies: List[str]) -> Dict[str, Any]:
        """Check if dependencies are available"""
        missing = []

        for dep in dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)

        return {"all_available": len(missing) == 0, "missing": missing}

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules from configuration"""
        # TODO: Load from config file
        return {
            "max_confidence_threshold": 0.8,
            "min_confidence_threshold": 0.3,
            "risk_patterns": ["rm -rf", "delete", "drop table", "truncate"],
        }


class SelfDebugger:
    """Main self-debugging engine orchestrator"""

    def __init__(self):
        self.error_db = ErrorDatabase()
        self.llm_debugger = LLMDebugger()
        self.validator = FixValidator()
        self.fix_db = self._init_fix_database()

    def _init_fix_database(self) -> sqlite3.Connection:
        """Initialize fix suggestions database"""
        conn = sqlite3.connect("autonomous_debug/fix_suggestions.db")
        conn.row_factory = sqlite3.Row

        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS fix_suggestions (
                id TEXT PRIMARY KEY,
                error_id TEXT,
                strategy TEXT,
                description TEXT,
                code_changes TEXT,
                confidence_score REAL,
                reasoning TEXT,
                dependencies TEXT,
                test_commands TEXT,
                rollback_safe BOOLEAN,
                estimated_impact TEXT,
                validation_result TEXT,
                created_at TEXT,
                applied_at TEXT,
                success BOOLEAN
            )
        """
        )
        conn.commit()

        return conn

    async def debug_error(self, error_id: str) -> Dict[str, Any]:
        """Debug a specific error and generate fixes"""
        try:
            # Get error from database
            cursor = self.error_db.conn.cursor()
            cursor.execute("SELECT * FROM errors WHERE id = ?", (error_id,))
            error_row = cursor.fetchone()

            if not error_row:
                return {"success": False, "error": "Error not found"}

            # Convert to ErrorRecord
            error = self._row_to_error_record(error_row)

            # Analyze error with LLM
            analysis = await self.llm_debugger.analyze_error(error)

            # Generate fix suggestions
            fixes = await self.llm_debugger.generate_fixes(error, analysis)

            # Validate fixes
            validated_fixes = []
            for fix in fixes:
                validation = self.validator.validate_fix(fix, error)

                # Adjust confidence based on validation
                fix.confidence_score += validation["confidence_adjustment"]
                fix.confidence_score = max(0.0, min(1.0, fix.confidence_score))

                # Store fix in database
                self._store_fix_suggestion(fix, validation)

                validated_fixes.append({"fix": fix, "validation": validation})

            result = {
                "success": True,
                "error_id": error_id,
                "analysis": analysis,
                "fixes": validated_fixes,
                "best_fix": validated_fixes[0] if validated_fixes else None,
            }

            logger.info(
                f"Debugged error {error_id[:8]} - Generated {len(validated_fixes)} fixes"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to debug error {error_id}: {e}")
            return {"success": False, "error": str(e)}

    async def debug_all_unresolved(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Debug all unresolved errors"""
        results = []

        try:
            # Get unresolved errors
            unresolved_errors = self.error_db.get_unresolved_errors(limit)

            for error_row in unresolved_errors:
                error_id = error_row["id"]
                result = await self.debug_error(error_id)
                results.append(result)

                # Small delay between debugging sessions
                await asyncio.sleep(1)

            logger.info(f"Debugged {len(results)} unresolved errors")

        except Exception as e:
            logger.error(f"Failed to debug all unresolved errors: {e}")

        return results

    def get_fix_suggestions(self, error_id: str) -> List[Dict[str, Any]]:
        """Get fix suggestions for an error"""
        try:
            cursor = self.fix_db.cursor()
            cursor.execute(
                """
                SELECT * FROM fix_suggestions 
                WHERE error_id = ? 
                ORDER BY confidence_score DESC
            """,
                (error_id,),
            )

            return [dict(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get fix suggestions: {e}")
            return []

    def _row_to_error_record(self, row) -> ErrorRecord:
        """Convert database row to ErrorRecord"""
        context_data = json.loads(row["context_data"]) if row["context_data"] else {}

        return ErrorRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            file_path=row["file_path"],
            line_number=row["line_number"],
            error_type=row["error_type"],
            message=row["message"],
            traceback_hash=row["traceback_hash"],
            frequency=row["frequency"],
            snippet=row["snippet"] or "",
            first_seen=datetime.fromisoformat(row["first_seen"]),
            last_seen=datetime.fromisoformat(row["last_seen"]),
            severity=row["severity"],
            resolved=bool(row["resolved"]),
            resolution_attempts=row["resolution_attempts"],
            full_traceback=row["full_traceback"] or "",
            context_data=context_data,
        )

    def _store_fix_suggestion(self, fix: FixSuggestion, validation: Dict[str, Any]):
        """Store fix suggestion in database"""
        try:
            cursor = self.fix_db.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO fix_suggestions (
                    id, error_id, strategy, description, code_changes,
                    confidence_score, reasoning, dependencies, test_commands,
                    rollback_safe, estimated_impact, validation_result,
                    created_at, applied_at, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    fix.id,
                    fix.error_id,
                    fix.strategy.value,
                    fix.description,
                    json.dumps(fix.code_changes),
                    fix.confidence_score,
                    fix.reasoning,
                    json.dumps(fix.dependencies),
                    json.dumps(fix.test_commands),
                    fix.rollback_safe,
                    fix.estimated_impact,
                    json.dumps(validation),
                    fix.created_at.isoformat(),
                    None,
                    None,
                ),
            )
            self.fix_db.commit()

        except Exception as e:
            logger.error(f"Failed to store fix suggestion: {e}")

    def close(self):
        """Close database connections"""
        if self.error_db:
            self.error_db.close()
        if self.fix_db:
            self.fix_db.close()


async def main():
    """Main function for self-debugging"""
    import argparse

    parser = argparse.ArgumentParser(description="LLM-Powered Self-Debugger")
    parser.add_argument("--error-id", help="Debug specific error ID")
    parser.add_argument(
        "--debug-all", action="store_true", help="Debug all unresolved errors"
    )
    parser.add_argument(
        "--limit", type=int, default=5, help="Limit for batch debugging"
    )
    parser.add_argument("--show-fixes", help="Show fix suggestions for error ID")

    args = parser.parse_args()

    debugger = SelfDebugger()

    try:
        if args.error_id:
            print(f"🔍 Debugging error {args.error_id}...")
            result = await debugger.debug_error(args.error_id)

            if result["success"]:
                print(f"✅ Analysis complete")
                print(f"📋 Generated {len(result['fixes'])} fix suggestions")

                if result["best_fix"]:
                    best_fix = result["best_fix"]["fix"]
                    print(f"🏆 Best fix: {best_fix.description}")
                    print(f"   Confidence: {best_fix.confidence_score:.2f}")
                    print(f"   Strategy: {best_fix.strategy.value}")
            else:
                print(f"❌ Failed: {result['error']}")

        elif args.debug_all:
            print(f"🔍 Debugging top {args.limit} unresolved errors...")
            results = await debugger.debug_all_unresolved(args.limit)

            successful = sum(1 for r in results if r["success"])
            print(f"✅ Successfully debugged {successful}/{len(results)} errors")

        elif args.show_fixes:
            fixes = debugger.get_fix_suggestions(args.show_fixes)
            print(f"🔧 Fix suggestions for {args.show_fixes}:")

            for i, fix in enumerate(fixes, 1):
                print(f"\n{i}. {fix['description']}")
                print(f"   Strategy: {fix['strategy']}")
                print(f"   Confidence: {fix['confidence_score']:.2f}")
                print(f"   Impact: {fix['estimated_impact']}")

        else:
            print("Specify --error-id, --debug-all, or --show-fixes")

    finally:
        debugger.close()


if __name__ == "__main__":
    asyncio.run(main())
