"""
Autonomous Self-Improvement Framework for Vega2.0
Phase 1: Project Analysis & Intelligence Gathering
"""

import os
import json
import ast
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Comprehensive file analysis metadata"""

    path: str
    size_bytes: int
    lines_count: int
    language: str
    file_type: str
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    dependencies: List[str]
    docstring_coverage: float
    last_modified: float
    content_hash: str
    potential_issues: List[str]
    improvement_opportunities: List[str]


@dataclass
class ProjectIntelligence:
    """Project-wide intelligence metadata"""

    scan_timestamp: str
    total_files: int
    total_lines: int
    language_distribution: Dict[str, int]
    dependency_graph: Dict[str, List[str]]
    function_registry: Dict[str, str]
    class_registry: Dict[str, str]
    performance_hotspots: List[str]
    technical_debt_score: float
    maintainability_index: float
    test_coverage_estimate: float


class AdvancedProjectAnalyzer:
    """
    Autonomous project analysis system for self-improvement
    """

    def __init__(self, project_root: str = "/home/ncacord/Vega2.0"):
        self.project_root = Path(project_root)
        self.analysis_cache = {}
        self.ignore_patterns = {
            ".venv",
            "__pycache__",
            ".git",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache",
            "vega.db-wal",
            "vega.db-shm",
        }
        self.file_extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".md": "markdown",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".cfg": "config",
            ".txt": "text",
        }

    def should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed"""
        # Skip ignored directories
        for part in file_path.parts:
            if part in self.ignore_patterns:
                return False

        # Skip binary files and databases
        if file_path.suffix in [".db", ".pyc", ".so", ".dll"]:
            return False

        # Only analyze known file types
        return file_path.suffix in self.file_extensions

    def extract_python_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract detailed metadata from Python files"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            functions = []
            classes = []
            imports = []
            docstrings = 0
            total_defs = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                    total_defs += 1
                    if ast.get_docstring(node):
                        docstrings += 1

                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    total_defs += 1
                    if ast.get_docstring(node):
                        docstrings += 1

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports.append(node.module)

            # Calculate complexity score (simplified cyclomatic complexity)
            complexity = self.calculate_complexity(tree)

            # Calculate docstring coverage
            docstring_coverage = (docstrings / max(total_defs, 1)) * 100

            # Identify potential issues
            issues = self.identify_python_issues(content, tree)

            # Identify improvement opportunities
            improvements = self.identify_improvements(content, tree)

            return {
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "complexity_score": complexity,
                "docstring_coverage": docstring_coverage,
                "potential_issues": issues,
                "improvement_opportunities": improvements,
            }

        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 0,
                "docstring_coverage": 0,
                "potential_issues": [f"Parse error: {str(e)}"],
                "improvement_opportunities": [],
            }

    def calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return complexity

    def identify_python_issues(self, content: str, tree: ast.AST) -> List[str]:
        """Identify potential code issues"""
        issues = []

        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.end_lineno and node.lineno:
                    if (node.end_lineno - node.lineno) > 50:
                        issues.append(
                            f"Long function: {node.name} ({node.end_lineno - node.lineno} lines)"
                        )

        # Check for missing type hints
        if (
            "from typing import" not in content
            and "from __future__ import annotations" not in content
        ):
            issues.append("Missing type hints")

        # Check for hardcoded values
        if re.search(r'["\'][^"\']*localhost[^"\']*["\']', content):
            issues.append("Hardcoded localhost values detected")

        # Check for TODO/FIXME comments
        todo_count = len(re.findall(r"#.*(?:TODO|FIXME|XXX)", content, re.IGNORECASE))
        if todo_count > 0:
            issues.append(f"{todo_count} TODO/FIXME comments")

        return issues

    def identify_improvements(self, content: str, tree: ast.AST) -> List[str]:
        """Identify improvement opportunities"""
        improvements = []

        # Check for async opportunities
        if "def " in content and "async def" not in content and "requests." in content:
            improvements.append("Consider using async/await for HTTP calls")

        # Check for caching opportunities
        if "def get_" in content and "@cache" not in content:
            improvements.append("Consider adding caching to getter functions")

        # Check for logging opportunities
        if "print(" in content and "logging" not in content:
            improvements.append("Replace print statements with proper logging")

        # Check for error handling
        try_count = content.count("try:")
        except_count = content.count("except")
        if try_count > 0 and except_count < try_count:
            improvements.append("Add more specific exception handling")

        return improvements

    def analyze_file(self, file_path: Path) -> Optional[FileAnalysis]:
        """Perform comprehensive file analysis"""
        try:
            stat = file_path.stat()

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines_count = content.count("\n") + 1
            content_hash = hashlib.md5(content.encode()).hexdigest()

            # Get file type
            file_type = self.file_extensions.get(file_path.suffix, "unknown")

            # Initialize metadata
            metadata = {
                "functions": [],
                "classes": [],
                "imports": [],
                "complexity_score": 1,
                "docstring_coverage": 0,
                "potential_issues": [],
                "improvement_opportunities": [],
            }

            # Extract language-specific metadata
            if file_type == "python":
                metadata.update(self.extract_python_metadata(file_path))

            # Extract dependencies (simplified)
            dependencies = self.extract_dependencies(content, file_type)

            return FileAnalysis(
                path=str(file_path),
                size_bytes=stat.st_size,
                lines_count=lines_count,
                language=file_type,
                file_type=file_type,
                functions=metadata["functions"],
                classes=metadata["classes"],
                imports=metadata["imports"],
                complexity_score=metadata["complexity_score"],
                dependencies=dependencies,
                docstring_coverage=metadata["docstring_coverage"],
                last_modified=stat.st_mtime,
                content_hash=content_hash,
                potential_issues=metadata["potential_issues"],
                improvement_opportunities=metadata["improvement_opportunities"],
            )

        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return None

    def extract_dependencies(self, content: str, file_type: str) -> List[str]:
        """Extract file dependencies"""
        dependencies = []

        if file_type == "python":
            # Extract imports
            import_patterns = [r"from\s+(\w+(?:\.\w+)*)", r"import\s+(\w+(?:\.\w+)*)"]
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                dependencies.extend(matches)

        elif file_type == "json":
            # For package.json or similar
            try:
                data = json.loads(content)
                if "dependencies" in data:
                    dependencies.extend(data["dependencies"].keys())
                if "devDependencies" in data:
                    dependencies.extend(data["devDependencies"].keys())
            except:
                pass

        return list(set(dependencies))  # Remove duplicates

    def build_dependency_graph(
        self, file_analyses: List[FileAnalysis]
    ) -> Dict[str, List[str]]:
        """Build project dependency graph"""
        graph = {}

        for analysis in file_analyses:
            file_name = Path(analysis.path).stem
            graph[file_name] = analysis.dependencies

        return graph

    def calculate_technical_debt(self, file_analyses: List[FileAnalysis]) -> float:
        """Calculate overall technical debt score"""
        total_issues = sum(len(fa.potential_issues) for fa in file_analyses)
        total_files = len(file_analyses)

        if total_files == 0:
            return 0.0

        # Normalize to 0-100 scale
        debt_score = min((total_issues / total_files) * 10, 100)
        return debt_score

    def calculate_maintainability_index(
        self, file_analyses: List[FileAnalysis]
    ) -> float:
        """Calculate maintainability index"""
        if not file_analyses:
            return 0.0

        total_complexity = sum(fa.complexity_score for fa in file_analyses)
        avg_complexity = total_complexity / len(file_analyses)

        total_doc_coverage = sum(fa.docstring_coverage for fa in file_analyses)
        avg_doc_coverage = total_doc_coverage / len(file_analyses)

        # Simple maintainability formula
        maintainability = (avg_doc_coverage / 2) + (50 - min(avg_complexity, 50))
        return maintainability

    def perform_full_analysis(self) -> ProjectIntelligence:
        """Perform comprehensive project analysis"""
        logger.info("Starting comprehensive project analysis...")

        file_analyses = []
        language_stats = {}
        total_lines = 0

        # Scan all files
        for root, dirs, files in os.walk(self.project_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]

            for file in files:
                file_path = Path(root) / file

                if self.should_analyze_file(file_path):
                    analysis = self.analyze_file(file_path)
                    if analysis:
                        file_analyses.append(analysis)

                        # Update language statistics
                        lang = analysis.language
                        language_stats[lang] = language_stats.get(lang, 0) + 1
                        total_lines += analysis.lines_count

        logger.info(f"Analyzed {len(file_analyses)} files")

        # Build comprehensive intelligence
        dependency_graph = self.build_dependency_graph(file_analyses)
        technical_debt = self.calculate_technical_debt(file_analyses)
        maintainability = self.calculate_maintainability_index(file_analyses)

        # Build function and class registries
        function_registry = {}
        class_registry = {}

        for analysis in file_analyses:
            file_name = Path(analysis.path).name
            for func in analysis.functions:
                function_registry[func] = file_name
            for cls in analysis.classes:
                class_registry[cls] = file_name

        # Identify performance hotspots
        performance_hotspots = [
            analysis.path
            for analysis in file_analyses
            if analysis.complexity_score > 20 or analysis.lines_count > 500
        ]

        return ProjectIntelligence(
            scan_timestamp=datetime.now().isoformat(),
            total_files=len(file_analyses),
            total_lines=total_lines,
            language_distribution=language_stats,
            dependency_graph=dependency_graph,
            function_registry=function_registry,
            class_registry=class_registry,
            performance_hotspots=performance_hotspots,
            technical_debt_score=technical_debt,
            maintainability_index=maintainability,
            test_coverage_estimate=0.0,  # To be implemented with actual test discovery
        )

    def save_analysis_results(
        self, intelligence: ProjectIntelligence, file_analyses: List[FileAnalysis]
    ) -> None:
        """Save analysis results to files"""

        # Save project intelligence
        with open(self.project_root / "project_intelligence.json", "w") as f:
            json.dump(asdict(intelligence), f, indent=2)

        # Save detailed file analyses
        analyses_dict = {fa.path: asdict(fa) for fa in file_analyses}
        with open(self.project_root / "file_analyses.json", "w") as f:
            json.dump(analyses_dict, f, indent=2)

        logger.info(
            "Analysis results saved to project_intelligence.json and file_analyses.json"
        )


def main():
    """Main analysis execution"""
    analyzer = AdvancedProjectAnalyzer()

    # Perform full project analysis
    intelligence = analyzer.perform_full_analysis()

    # Get file analyses for saving
    file_analyses = []
    for root, dirs, files in os.walk(analyzer.project_root):
        dirs[:] = [d for d in dirs if d not in analyzer.ignore_patterns]
        for file in files:
            file_path = Path(root) / file
            if analyzer.should_analyze_file(file_path):
                analysis = analyzer.analyze_file(file_path)
                if analysis:
                    file_analyses.append(analysis)

    # Save results
    analyzer.save_analysis_results(intelligence, file_analyses)

    # Print summary
    print(f"\n🧠 AUTONOMOUS PROJECT ANALYSIS COMPLETE")
    print(f"=" * 50)
    print(f"📁 Total Files Analyzed: {intelligence.total_files}")
    print(f"📄 Total Lines of Code: {intelligence.total_lines:,}")
    print(f"🏗️  Technical Debt Score: {intelligence.technical_debt_score:.1f}/100")
    print(f"🔧 Maintainability Index: {intelligence.maintainability_index:.1f}/100")
    print(f"⚡ Performance Hotspots: {len(intelligence.performance_hotspots)}")

    print(f"\n📊 Language Distribution:")
    for lang, count in intelligence.language_distribution.items():
        print(f"  {lang}: {count} files")

    if intelligence.performance_hotspots:
        print(f"\n🔥 Performance Hotspots:")
        for hotspot in intelligence.performance_hotspots[:5]:  # Top 5
            print(f"  - {hotspot}")

    print(f"\n💾 Results saved to:")
    print(f"  - project_intelligence.json")
    print(f"  - file_analyses.json")


if __name__ == "__main__":
    main()
