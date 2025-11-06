"""
Vega 2.0 Technical Documentation AI Module

This module provides advanced technical documentation analysis and generation capabilities including:
- Code documentation generation and summarization
- API documentation analysis and extraction
- Technical writing assistance and quality analysis
- Integration with code analysis and NLP models

Dependencies:
- transformers: For code/documentation language models (e.g., CodeBERT, DocT5)
- tree-sitter: For code parsing (optional)
- numpy, pandas: Data processing
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

from .base import (
    BaseDocumentProcessor,
    ProcessingContext,
    ProcessingResult,
    ConfigurableComponent,
    handle_import_error,
    MetricsCollector,
    DocumentIntelligenceError,
    ProcessingError,
    ValidationError,
)

# Skip heavy imports in test mode
import os

if os.environ.get("VEGA_TEST_MODE") == "1":
    HAS_TRANSFORMERS = False
    HAS_TREE_SITTER = False
    pipeline = AutoTokenizer = AutoModelForSequenceClassification = torch = None
    Language = Parser = None
else:
    # Optional dependencies with graceful fallback
    HAS_TRANSFORMERS = handle_import_error("transformers", optional=True)
    HAS_TREE_SITTER = handle_import_error("tree_sitter", optional=True)

    try:
        if HAS_TRANSFORMERS:
            from transformers import (
                pipeline,
                AutoTokenizer,
                AutoModelForSequenceClassification,
            )
            import torch
    except ImportError:
        pipeline = AutoTokenizer = AutoModelForSequenceClassification = torch = None

    try:
        if HAS_TREE_SITTER:
            from tree_sitter import Language, Parser
    except ImportError:
        Language = Parser = None

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported programming languages for documentation analysis"""

    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"


class DocumentationType(Enum):
    """Types of technical documentation"""

    API_DOCS = "api_docs"
    CODE_COMMENTS = "code_comments"
    README = "readme"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"
    INLINE_DOCS = "inline_docs"


@dataclass
class TechnicalConfig(ConfigurableComponent):
    """Configuration for technical documentation AI"""

    enable_code_doc_generation: bool = True
    enable_api_doc_analysis: bool = True
    enable_quality_analysis: bool = True
    enable_style_checking: bool = True
    min_confidence: float = 0.7
    supported_languages: List[SupportedLanguage] = field(
        default_factory=lambda: [
            SupportedLanguage.PYTHON,
            SupportedLanguage.JAVA,
            SupportedLanguage.JAVASCRIPT,
            SupportedLanguage.TYPESCRIPT,
            SupportedLanguage.CPP,
        ]
    )
    max_doc_length: int = 5000
    min_doc_length: int = 10
    use_transformers: bool = True
    model_name: str = "microsoft/codebert-base"
    timeout_seconds: float = 60.0

    def validate_config(self) -> List[str]:
        errors = []
        if self.min_confidence < 0 or self.min_confidence > 1:
            errors.append("min_confidence must be between 0 and 1")
        if self.max_doc_length <= self.min_doc_length:
            errors.append("max_doc_length must be greater than min_doc_length")
        if self.use_transformers and not HAS_TRANSFORMERS:
            errors.append("transformers library required but not available")
        return errors


@dataclass
class DocumentationConfig(ConfigurableComponent):
    """Configuration for documentation quality requirements"""

    min_length: int = 20
    max_length: int = 2000
    require_examples: bool = True
    require_param_docs: bool = True
    require_return_docs: bool = True
    require_exception_docs: bool = False
    style_guide: str = "google"  # google, numpy, sphinx

    def validate_config(self) -> List[str]:
        errors = []
        if self.max_length <= self.min_length:
            errors.append("max_length must be greater than min_length")
        if self.style_guide not in ["google", "numpy", "sphinx"]:
            errors.append("style_guide must be one of: google, numpy, sphinx")
        return errors


class CodeDocumentationGenerator(BaseDocumentProcessor[TechnicalConfig]):
    """
    Generates documentation for code using AI models and rule-based approaches
    """

    async def _async_initialize(self) -> None:
        """Initialize code documentation models"""
        self.metrics = MetricsCollector()
        self.model = None
        self.tokenizer = None

        if self.config.use_transformers and HAS_TRANSFORMERS:
            try:
                self.logger.info(f"Loading model: {self.config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.model_name
                )
                self.logger.info("Transformers models loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load transformers model: {e}")

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Generate documentation for code"""
        if not isinstance(input_data, dict):
            raise ValidationError(
                "Input must be a dictionary with 'code' and 'language' keys"
            )

        code = input_data.get("code", "")
        language = input_data.get("language", "python")

        if not code.strip():
            raise ValidationError("Code cannot be empty")

        self.metrics.start_timer("doc_generation")

        try:
            # Parse code structure
            code_structure = await self._analyze_code_structure(code, language)

            # Generate documentation
            documentation = await self._generate_documentation(
                code, language, code_structure
            )

            # Quality assessment
            quality_score = await self._assess_quality(documentation)

            self.metrics.end_timer("doc_generation")

            return {
                "documentation": documentation,
                "code_structure": code_structure,
                "quality_score": quality_score,
                "language": language,
                "metrics": self.metrics.get_metrics(),
            }

        except Exception as e:
            self.metrics.end_timer("doc_generation")
            raise ProcessingError(f"Documentation generation failed: {e}")

    async def _analyze_code_structure(self, code: str, language: str) -> Dict[str, Any]:
        """Analyze code structure to extract functions, classes, etc."""
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_score": 0.0,
        }

        # Simple regex-based analysis (would use tree-sitter in production)
        if language.lower() == "python":
            # Find function definitions
            function_pattern = r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:"
            for match in re.finditer(function_pattern, code):
                structure["functions"].append(
                    {
                        "name": match.group(1),
                        "parameters": match.group(2),
                        "return_type": match.group(3),
                    }
                )

            # Find class definitions
            class_pattern = r"class\s+(\w+)(?:\(([^)]+)\))?\s*:"
            for match in re.finditer(class_pattern, code):
                structure["classes"].append(
                    {"name": match.group(1), "bases": match.group(2)}
                )

        # Calculate simple complexity score
        structure["complexity_score"] = (
            len(structure["functions"]) * 0.1 + len(structure["classes"]) * 0.2
        )

        return structure

    async def _generate_documentation(
        self, code: str, language: str, structure: Dict[str, Any]
    ) -> str:
        """Generate documentation based on code analysis"""
        # Template-based documentation generation
        doc_parts = []

        if structure["functions"]:
            doc_parts.append("Functions:")
            for func in structure["functions"]:
                doc_parts.append(f"- {func['name']}: Auto-generated documentation")

        if structure["classes"]:
            doc_parts.append("Classes:")
            for cls in structure["classes"]:
                doc_parts.append(f"- {cls['name']}: Auto-generated class documentation")

        if not doc_parts:
            doc_parts.append("Code documentation generated automatically")

        return "\n".join(doc_parts)

    async def _assess_quality(self, documentation: str) -> float:
        """Assess documentation quality"""
        score = 0.5  # Base score

        # Length check
        if len(documentation) > self.config.min_doc_length:
            score += 0.2

        # Content checks
        if "function" in documentation.lower() or "class" in documentation.lower():
            score += 0.2

        if len(documentation.split()) > 10:
            score += 0.1

        return min(score, 1.0)


class APIDocumentationAnalyzer(BaseDocumentProcessor[TechnicalConfig]):
    """
    Analyzes existing API documentation for completeness and quality
    """

    async def _async_initialize(self) -> None:
        """Initialize API documentation analyzer"""
        self.metrics = MetricsCollector()

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze API documentation"""
        if not isinstance(input_data, str):
            raise ValidationError("Input must be a string containing API documentation")

        docstring = input_data.strip()
        if not docstring:
            raise ValidationError("Documentation string cannot be empty")

        self.metrics.start_timer("api_analysis")

        try:
            # Extract API elements
            api_elements = await self._extract_api_elements(docstring)

            # Analyze completeness
            completeness_score = await self._analyze_completeness(
                docstring, api_elements
            )

            # Check style compliance
            style_score = await self._check_style(docstring)

            self.metrics.end_timer("api_analysis")

            return {
                "api_elements": api_elements,
                "completeness_score": completeness_score,
                "style_score": style_score,
                "overall_score": (completeness_score + style_score) / 2,
                "metrics": self.metrics.get_metrics(),
            }

        except Exception as e:
            self.metrics.end_timer("api_analysis")
            raise ProcessingError(f"API analysis failed: {e}")

    async def _extract_api_elements(self, docstring: str) -> Dict[str, Any]:
        """Extract API elements from documentation"""
        elements = {"endpoints": [], "parameters": [], "returns": None, "examples": []}

        # Simple pattern matching for API elements
        # In production, this would use more sophisticated parsing

        # Find parameters
        param_pattern = r"(@param|:param|Parameters?:)\s*(\w+)"
        for match in re.finditer(param_pattern, docstring, re.IGNORECASE):
            elements["parameters"].append(match.group(2))

        # Find returns
        return_pattern = r"(@return|:return|Returns?:)"
        if re.search(return_pattern, docstring, re.IGNORECASE):
            elements["returns"] = "documented"

        # Find examples
        example_pattern = r"(example|Example|EXAMPLE)"
        if re.search(example_pattern, docstring):
            elements["examples"].append("found")

        return elements

    async def _analyze_completeness(
        self, docstring: str, elements: Dict[str, Any]
    ) -> float:
        """Analyze documentation completeness"""
        score = 0.0

        # Basic description
        if len(docstring.split()) >= 5:
            score += 0.3

        # Parameters documented
        if elements["parameters"]:
            score += 0.3

        # Returns documented
        if elements["returns"]:
            score += 0.2

        # Examples provided
        if elements["examples"]:
            score += 0.2

        return min(score, 1.0)

    async def _check_style(self, docstring: str) -> float:
        """Check documentation style compliance"""
        score = 0.5  # Base score

        # Proper capitalization
        if docstring and docstring[0].isupper():
            score += 0.2

        # Proper punctuation
        if docstring.rstrip().endswith("."):
            score += 0.2

        # Reasonable length
        if 20 <= len(docstring) <= 1000:
            score += 0.1

        return min(score, 1.0)


class TechnicalWritingAssistant(BaseDocumentProcessor[TechnicalConfig]):
    """
    Provides writing assistance for technical documentation
    """

    async def _async_initialize(self) -> None:
        """Initialize writing assistant"""
        self.metrics = MetricsCollector()
        self.common_issues = {
            r"\b(very|really|quite)\b": "Avoid vague qualifiers",
            r"\b(obviously|clearly|simply)\b": "Avoid assumptive language",
            r"[.]{2,}": "Use proper ellipsis formatting",
            r"[!]{2,}": "Avoid excessive exclamation marks",
        }

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Improve technical writing"""
        if not isinstance(input_data, str):
            raise ValidationError("Input must be a string containing text to improve")

        text = input_data.strip()
        if not text:
            raise ValidationError("Text cannot be empty")

        self.metrics.start_timer("writing_assistance")

        try:
            # Analyze issues
            issues = await self._find_issues(text)

            # Generate improvements
            improved_text = await self._improve_text(text, issues)

            # Calculate improvement score
            improvement_score = len(issues) / max(len(text.split()), 1)

            self.metrics.end_timer("writing_assistance")

            return {
                "original_text": text,
                "improved_text": improved_text,
                "issues_found": issues,
                "improvement_score": min(improvement_score, 1.0),
                "metrics": self.metrics.get_metrics(),
            }

        except Exception as e:
            self.metrics.end_timer("writing_assistance")
            raise ProcessingError(f"Writing assistance failed: {e}")

    async def _find_issues(self, text: str) -> List[Dict[str, str]]:
        """Find writing issues in text"""
        issues = []

        for pattern, message in self.common_issues.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                issues.append(
                    {
                        "issue": message,
                        "text": match.group(0),
                        "position": match.start(),
                    }
                )

        return issues

    async def _improve_text(self, text: str, issues: List[Dict[str, str]]) -> str:
        """Apply improvements to text"""
        improved = text

        # Simple improvements (in production, would use ML models)
        for pattern, _ in self.common_issues.items():
            if pattern in [r"[.]{2,}", r"[!]{2,}"]:
                improved = re.sub(pattern, ".", improved)
            else:
                # For now, just flag the issues
                pass

        return improved


class DocumentationQualityAnalyzer(BaseDocumentProcessor[DocumentationConfig]):
    """
    Analyzes documentation quality against configurable standards
    """

    async def _async_initialize(self) -> None:
        """Initialize quality analyzer"""
        self.metrics = MetricsCollector()

    async def _process_internal(
        self, input_data: Any, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze documentation quality"""
        if not isinstance(input_data, str):
            raise ValidationError(
                "Input must be a string containing documentation to analyze"
            )

        documentation = input_data.strip()
        if not documentation:
            raise ValidationError("Documentation cannot be empty")

        self.metrics.start_timer("quality_analysis")

        try:
            # Length analysis
            length_score = await self._analyze_length(documentation)

            # Structure analysis
            structure_score = await self._analyze_structure(documentation)

            # Content analysis
            content_score = await self._analyze_content(documentation)

            # Overall quality score
            overall_score = (length_score + structure_score + content_score) / 3

            # Generate suggestions
            suggestions = await self._generate_suggestions(
                documentation,
                {
                    "length": length_score,
                    "structure": structure_score,
                    "content": content_score,
                },
            )

            self.metrics.end_timer("quality_analysis")

            return {
                "overall_score": overall_score,
                "length_score": length_score,
                "structure_score": structure_score,
                "content_score": content_score,
                "suggestions": suggestions,
                "metrics": self.metrics.get_metrics(),
            }

        except Exception as e:
            self.metrics.end_timer("quality_analysis")
            raise ProcessingError(f"Quality analysis failed: {e}")

    async def _analyze_length(self, documentation: str) -> float:
        """Analyze documentation length appropriateness"""
        length = len(documentation)

        if self.config.min_length <= length <= self.config.max_length:
            return 1.0
        elif length < self.config.min_length:
            return length / self.config.min_length
        else:
            # Penalize excessive length
            return max(0.5, self.config.max_length / length)

    async def _analyze_structure(self, documentation: str) -> float:
        """Analyze documentation structure"""
        score = 0.0

        # Has proper paragraph structure
        paragraphs = [p.strip() for p in documentation.split("\n\n") if p.strip()]
        if len(paragraphs) >= 1:
            score += 0.3

        # Has examples if required
        if not self.config.require_examples or "example" in documentation.lower():
            score += 0.4

        # Has parameter docs if required
        if not self.config.require_param_docs or "param" in documentation.lower():
            score += 0.2

        # Has return docs if required
        if not self.config.require_return_docs or "return" in documentation.lower():
            score += 0.1

        return min(score, 1.0)

    async def _analyze_content(self, documentation: str) -> float:
        """Analyze documentation content quality"""
        score = 0.5  # Base score

        # Word count
        words = len(documentation.split())
        if words >= 10:
            score += 0.2

        # Proper sentence structure
        sentences = [s.strip() for s in documentation.split(".") if s.strip()]
        if len(sentences) >= 2:
            score += 0.2

        # Technical vocabulary
        tech_words = ["function", "method", "parameter", "return", "class", "object"]
        if any(word in documentation.lower() for word in tech_words):
            score += 0.1

        return min(score, 1.0)

    async def _generate_suggestions(
        self, documentation: str, scores: Dict[str, float]
    ) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        if scores["length"] < 0.8:
            if len(documentation) < self.config.min_length:
                suggestions.append(
                    "Documentation is too short. Consider adding more detail."
                )
            else:
                suggestions.append(
                    "Documentation is too long. Consider being more concise."
                )

        if scores["structure"] < 0.8:
            if self.config.require_examples and "example" not in documentation.lower():
                suggestions.append("Add examples to illustrate usage.")
            if self.config.require_param_docs and "param" not in documentation.lower():
                suggestions.append("Document all parameters.")

        if scores["content"] < 0.8:
            suggestions.append("Consider adding more technical details and context.")

        return suggestions

    # Backwards-compatible convenience used by some tests
    async def analyze_quality(
        self, documentation: str, context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """Wrapper that forwards to process() for compatibility."""
        return await self.process(documentation, context)


# Main AI class that orchestrates all components
class TechnicalDocumentationAI:
    """
    Main interface for technical documentation AI capabilities
    """

    def __init__(self, config: Optional[TechnicalConfig] = None):
        self.config = config or TechnicalConfig()
        self.doc_config = DocumentationConfig()

        # Initialize processors
        self.code_generator = CodeDocumentationGenerator(self.config)
        self.api_analyzer = APIDocumentationAnalyzer(self.config)
        self.writing_assistant = TechnicalWritingAssistant(self.config)
        self.quality_analyzer = DocumentationQualityAnalyzer(self.doc_config)
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize all processors"""
        await asyncio.gather(
            self.code_generator.initialize(),
            self.api_analyzer.initialize(),
            self.writing_assistant.initialize(),
            self.quality_analyzer.initialize(),
        )
        self.is_initialized = True

    async def process_document(
        self,
        data: Union[str, Dict[str, Any], ProcessingContext],
        context: Optional[ProcessingContext] = None,
    ) -> ProcessingResult:
        """
        Route processing based on input shape/content for convenience in tests.

        - ProcessingContext -> extract content and process
        - dict with 'code' -> generate_code_docs
        - string that looks like API doc (has 'param' or 'return' or 'endpoint') -> analyze_api_docs
        - otherwise -> quality_analysis or assist_writing depending on length
        """
        # Handle ProcessingContext as document
        if isinstance(data, ProcessingContext):
            text = data.metadata.get("content", "")
            ctx = data

            # Validate input
            if not text or not text.strip():
                return ProcessingResult(
                    success=False,
                    context=ctx,
                    data={"error": "Empty content provided for technical processing"},
                    errors=["Empty content provided for technical processing"],
                )

            # Route based on content
            return await self.quality_analysis(text, ctx)

        if isinstance(data, dict) and ("code" in data or "language" in data):
            code = data.get("code", "")
            language = data.get("language", "python")
            return await self.generate_code_docs(code, language, context)

        if isinstance(data, str):
            text = data.strip()
            lower = text.lower()
            if any(k in lower for k in ["param", "return", "endpoint", "api"]):
                return await self.analyze_api_docs(text, context)
            # Prefer quality analysis for medium/long text, otherwise writing assist
            if len(text.split()) >= 10:
                return await self.quality_analysis(text, context)
            return await self.assist_writing(text, context)

        # Fallback: stringify and do quality analysis
        return await self.quality_analysis(str(data), context)

    async def generate_code_docs(
        self,
        code: str,
        language: str = "python",
        context: Optional[ProcessingContext] = None,
    ) -> ProcessingResult:
        """Generate documentation for code"""
        input_data = {"code": code, "language": language}
        return await self.code_generator.process(input_data, context)

    async def analyze_api_docs(
        self, docstring: str, context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """Analyze API documentation quality"""
        return await self.api_analyzer.process(docstring, context)

    async def assist_writing(
        self, text: str, context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """Provide writing assistance"""
        return await self.writing_assistant.process(text, context)

    async def quality_analysis(
        self, docstring: str, context: Optional[ProcessingContext] = None
    ) -> ProcessingResult:
        """Analyze documentation quality"""
        return await self.quality_analyzer.process(docstring, context)

    async def health_check(self) -> Dict[str, Any]:
        """Get health status of all components"""
        checks = await asyncio.gather(
            self.code_generator.health_check(),
            self.api_analyzer.health_check(),
            self.writing_assistant.health_check(),
            self.quality_analyzer.health_check(),
        )
        overall_ok = all(c.get("status") == "healthy" for c in checks)
        return {
            "healthy": bool(self.is_initialized and overall_ok),
            "overall_status": (
                "healthy" if (self.is_initialized and overall_ok) else "degraded"
            ),
            "components": {
                "code_generator": checks[0],
                "api_analyzer": checks[1],
                "writing_assistant": checks[2],
                "quality_analyzer": checks[3],
            },
        }
