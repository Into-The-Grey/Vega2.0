"""
Technical Document Module Tests
==============================

Comprehensive test suite for technical documentation AI capabilities.
Tests API documentation analysis, code documentation generation, and technical writing assistance.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
from pathlib import Path

from src.vega.document.technical import (
    TechnicalDocumentationAI,
    CodeDocumentationGenerator,
    APIDocumentationAnalyzer,
    TechnicalWritingAssistant,
    DocumentationQualityAnalyzer,
)
from src.vega.document.base import ProcessingContext, ProcessingResult
from tests.document.fixtures import (
    sample_technical_documents,
    mock_technical_processor,
    create_test_context,
    performance_monitor,
)


class TestTechnicalDocumentationAI:
    """Test suite for TechnicalDocumentationAI class"""

    @pytest.fixture
    async def technical_ai(self):
        """Create and initialize technical AI instance"""
        ai = TechnicalDocumentationAI()
        await ai.initialize()
        return ai

    @pytest.fixture
    def technical_context(self):
        """Create technical document processing context"""
        return create_test_context(
            content=sample_technical_documents["api_reference"],
            document_type="technical",
            processing_mode="analysis",
            metadata={"doc_type": "api_documentation"},
        )

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test technical AI initialization"""
        ai = TechnicalDocumentationAI()

        # Should not be initialized yet
        assert not ai.is_initialized

        # Initialize
        await ai.initialize()

        # Should be initialized now
        assert ai.is_initialized
        assert ai.code_doc_generator is not None
        assert ai.api_doc_analyzer is not None
        assert ai.writing_assistant is not None
        assert ai.quality_analyzer is not None

    @pytest.mark.asyncio
    async def test_process_document_basic(self, technical_ai, technical_context):
        """Test basic technical document processing"""
        result = await technical_ai.process_document(technical_context)

        assert isinstance(result, ProcessingResult)
        assert result.session_id == technical_context.session_id
        assert result.success is True
        assert "technical_analysis" in result.results
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_api_documentation_analysis(self, technical_ai):
        """Test API documentation analysis"""
        context = create_test_context(
            content=sample_technical_documents["rest_api_doc"],
            processing_mode="api_analysis",
            metadata={"api_type": "REST"},
        )

        result = await technical_ai.process_document(context)

        api_analysis = result.results.get("api_analysis", {})

        # Check analysis components
        assert "endpoints" in api_analysis
        assert "authentication" in api_analysis
        assert "data_models" in api_analysis
        assert "completeness_score" in api_analysis

        # Validate endpoints structure
        endpoints = api_analysis["endpoints"]
        assert isinstance(endpoints, list)

        for endpoint in endpoints:
            assert "method" in endpoint
            assert "path" in endpoint
            assert "description" in endpoint

    @pytest.mark.asyncio
    async def test_code_documentation_generation(self, technical_ai):
        """Test code documentation generation"""
        context = create_test_context(
            content=sample_technical_documents["python_code"],
            processing_mode="code_documentation",
            metadata={"language": "python", "style": "docstring"},
        )

        result = await technical_ai.process_document(context)

        code_docs = result.results.get("code_documentation", {})

        assert "generated_docs" in code_docs
        assert "functions" in code_docs
        assert "classes" in code_docs
        assert "documentation_style" in code_docs

    @pytest.mark.asyncio
    async def test_technical_writing_assistance(self, technical_ai):
        """Test technical writing assistance"""
        context = create_test_context(
            content=sample_technical_documents["draft_specification"],
            processing_mode="writing_assistance",
            metadata={"assistance_type": "improvement"},
        )

        result = await technical_ai.process_document(context)

        assistance = result.results.get("writing_assistance", {})

        assert "improvements" in assistance
        assert "clarity_suggestions" in assistance
        assert "structure_recommendations" in assistance
        assert "terminology_check" in assistance

    @pytest.mark.asyncio
    async def test_documentation_quality_analysis(self, technical_ai):
        """Test documentation quality analysis"""
        context = create_test_context(
            content=sample_technical_documents["incomplete_doc"],
            processing_mode="quality_analysis",
        )

        result = await technical_ai.process_document(context)

        quality = result.results.get("quality_analysis", {})

        assert "completeness_score" in quality
        assert "clarity_score" in quality
        assert "consistency_score" in quality
        assert "missing_elements" in quality
        assert "improvement_suggestions" in quality

    @pytest.mark.asyncio
    async def test_architecture_documentation_analysis(self, technical_ai):
        """Test architecture documentation analysis"""
        context = create_test_context(
            content=sample_technical_documents["architecture_doc"],
            processing_mode="architecture_analysis",
            metadata={"doc_type": "system_architecture"},
        )

        result = await technical_ai.process_document(context)

        architecture = result.results.get("architecture_analysis", {})

        assert "components" in architecture
        assert "relationships" in architecture
        assert "data_flow" in architecture
        assert "design_patterns" in architecture

    @pytest.mark.asyncio
    async def test_tutorial_generation(self, technical_ai):
        """Test tutorial and guide generation"""
        context = create_test_context(
            content=sample_technical_documents["api_reference"],
            processing_mode="tutorial_generation",
            metadata={"tutorial_type": "quickstart", "target_audience": "beginner"},
        )

        result = await technical_ai.process_document(context)

        tutorial = result.results.get("tutorial", {})

        assert "steps" in tutorial
        assert "examples" in tutorial
        assert "prerequisites" in tutorial
        assert "learning_objectives" in tutorial

    @pytest.mark.asyncio
    async def test_code_example_generation(self, technical_ai):
        """Test code example generation"""
        context = create_test_context(
            content=sample_technical_documents["api_reference"],
            processing_mode="example_generation",
            metadata={"languages": ["python", "javascript"], "example_type": "usage"},
        )

        result = await technical_ai.process_document(context)

        examples = result.results.get("code_examples", {})

        assert "python" in examples
        assert "javascript" in examples

        for lang, lang_examples in examples.items():
            assert isinstance(lang_examples, list)
            assert len(lang_examples) > 0

    @pytest.mark.asyncio
    async def test_changelog_generation(self, technical_ai):
        """Test changelog generation from documentation"""
        context = create_test_context(
            content=sample_technical_documents["version_comparison"],
            processing_mode="changelog_generation",
            metadata={"version_from": "1.0", "version_to": "2.0"},
        )

        result = await technical_ai.process_document(context)

        changelog = result.results.get("changelog", {})

        assert "added" in changelog
        assert "changed" in changelog
        assert "deprecated" in changelog
        assert "removed" in changelog
        assert "fixed" in changelog

    @pytest.mark.asyncio
    async def test_error_handling(self, technical_ai):
        """Test error handling in technical processing"""
        # Test with empty content
        context = create_test_context(content="", processing_mode="analysis")

        result = await technical_ai.process_document(context)

        assert result.success is False
        assert "error" in result.results
        assert "Empty content" in result.results["error"]

    @pytest.mark.asyncio
    async def test_unsupported_programming_language(self, technical_ai):
        """Test handling of unsupported programming languages"""
        context = create_test_context(
            content=sample_technical_documents["obscure_language_code"],
            processing_mode="code_documentation",
            metadata={"language": "brainfuck"},
        )

        result = await technical_ai.process_document(context)

        # Should handle gracefully with language warning
        assert "language_warning" in result.metadata
        assert "unsupported_language" in result.metadata

    @pytest.mark.asyncio
    async def test_large_technical_document(self, technical_ai):
        """Test processing of large technical documents"""
        large_content = sample_technical_documents["comprehensive_api_doc"] * 20

        context = create_test_context(content=large_content, processing_mode="analysis")

        with performance_monitor() as monitor:
            result = await technical_ai.process_document(context)

        assert result.success is True
        assert monitor.memory_usage < 150  # MB
        assert monitor.processing_time < 45  # seconds

    @pytest.mark.asyncio
    async def test_concurrent_technical_processing(self, technical_ai):
        """Test concurrent technical document processing"""
        contexts = [
            create_test_context(
                content=doc, processing_mode="analysis", session_id=f"tech_session_{i}"
            )
            for i, doc in enumerate(sample_technical_documents.values())
        ]

        # Process multiple documents concurrently
        tasks = [technical_ai.process_document(ctx) for ctx in contexts]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result.success for result in results)

        # Session IDs should be preserved
        for i, result in enumerate(results):
            assert result.session_id == f"tech_session_{i}"

    @pytest.mark.asyncio
    async def test_health_check(self, technical_ai):
        """Test technical AI health check"""
        health = await technical_ai.health_check()

        assert isinstance(health, dict)
        assert "healthy" in health
        assert "ready" in health
        assert health["healthy"] is True
        assert health["ready"] is True
        assert "components" in health

    @pytest.mark.asyncio
    async def test_cleanup(self, technical_ai):
        """Test cleanup functionality"""
        # Verify initial state
        assert technical_ai.is_initialized is True

        # Cleanup
        await technical_ai.cleanup()

        # Verify cleanup
        assert technical_ai.is_initialized is False


class TestCodeDocumentationGenerator:
    """Test suite for CodeDocumentationGenerator component"""

    @pytest.fixture
    def code_doc_generator(self):
        """Create code documentation generator instance"""
        return CodeDocumentationGenerator()

    @pytest.mark.asyncio
    async def test_code_doc_generator_initialization(self, code_doc_generator):
        """Test code documentation generator initialization"""
        await code_doc_generator.initialize()
        assert code_doc_generator.is_initialized is True

    @pytest.mark.asyncio
    async def test_python_docstring_generation(self, code_doc_generator):
        """Test Python docstring generation"""
        await code_doc_generator.initialize()

        context = create_test_context(
            content=sample_technical_documents["python_function"],
            metadata={"language": "python", "style": "google"},
        )

        result = await code_doc_generator.generate_documentation(context)

        assert "docstrings" in result
        assert "functions" in result["docstrings"]

    @pytest.mark.asyncio
    async def test_javascript_jsdoc_generation(self, code_doc_generator):
        """Test JavaScript JSDoc generation"""
        await code_doc_generator.initialize()

        context = create_test_context(
            content=sample_technical_documents["javascript_class"],
            metadata={"language": "javascript", "style": "jsdoc"},
        )

        result = await code_doc_generator.generate_documentation(context)

        assert "jsdoc" in result
        assert "classes" in result["jsdoc"]


class TestAPIDocumentationAnalyzer:
    """Test suite for APIDocumentationAnalyzer component"""

    @pytest.fixture
    def api_doc_analyzer(self):
        """Create API documentation analyzer instance"""
        return APIDocumentationAnalyzer()

    @pytest.mark.asyncio
    async def test_api_doc_analyzer_initialization(self, api_doc_analyzer):
        """Test API documentation analyzer initialization"""
        await api_doc_analyzer.initialize()
        assert api_doc_analyzer.is_initialized is True

    @pytest.mark.asyncio
    async def test_rest_api_analysis(self, api_doc_analyzer):
        """Test REST API documentation analysis"""
        await api_doc_analyzer.initialize()

        context = create_test_context(
            content=sample_technical_documents["openapi_spec"],
            metadata={"api_type": "REST", "format": "openapi"},
        )

        result = await api_doc_analyzer.analyze_api_documentation(context)

        assert "endpoints" in result
        assert "schemas" in result
        assert "authentication" in result

    @pytest.mark.asyncio
    async def test_graphql_api_analysis(self, api_doc_analyzer):
        """Test GraphQL API documentation analysis"""
        await api_doc_analyzer.initialize()

        context = create_test_context(
            content=sample_technical_documents["graphql_schema"],
            metadata={"api_type": "GraphQL"},
        )

        result = await api_doc_analyzer.analyze_api_documentation(context)

        assert "queries" in result
        assert "mutations" in result
        assert "types" in result


class TestTechnicalWritingAssistant:
    """Test suite for TechnicalWritingAssistant component"""

    @pytest.fixture
    def writing_assistant(self):
        """Create technical writing assistant instance"""
        return TechnicalWritingAssistant()

    @pytest.mark.asyncio
    async def test_writing_assistant_initialization(self, writing_assistant):
        """Test technical writing assistant initialization"""
        await writing_assistant.initialize()
        assert writing_assistant.is_initialized is True

    @pytest.mark.asyncio
    async def test_clarity_improvement(self, writing_assistant):
        """Test clarity improvement suggestions"""
        await writing_assistant.initialize()

        context = create_test_context(
            content=sample_technical_documents["unclear_specification"],
            metadata={"focus": "clarity"},
        )

        result = await writing_assistant.provide_assistance(context)

        assert "clarity_improvements" in result
        assert "suggestions" in result["clarity_improvements"]

    @pytest.mark.asyncio
    async def test_structure_optimization(self, writing_assistant):
        """Test document structure optimization"""
        await writing_assistant.initialize()

        context = create_test_context(
            content=sample_technical_documents["unstructured_doc"],
            metadata={"focus": "structure"},
        )

        result = await writing_assistant.provide_assistance(context)

        assert "structure_improvements" in result
        assert "recommended_outline" in result["structure_improvements"]


class TestDocumentationQualityAnalyzer:
    """Test suite for DocumentationQualityAnalyzer component"""

    @pytest.fixture
    def quality_analyzer(self):
        """Create documentation quality analyzer instance"""
        return DocumentationQualityAnalyzer()

    @pytest.mark.asyncio
    async def test_quality_analyzer_initialization(self, quality_analyzer):
        """Test documentation quality analyzer initialization"""
        await quality_analyzer.initialize()
        assert quality_analyzer.is_initialized is True

    @pytest.mark.asyncio
    async def test_comprehensive_quality_analysis(self, quality_analyzer):
        """Test comprehensive quality analysis"""
        await quality_analyzer.initialize()

        context = create_test_context(
            content=sample_technical_documents["api_reference"],
            metadata={"analysis_type": "comprehensive"},
        )

        result = await quality_analyzer.analyze_quality(context)

        assert "overall_score" in result
        assert "completeness" in result
        assert "clarity" in result
        assert "consistency" in result
        assert "accuracy" in result

    @pytest.mark.asyncio
    async def test_accessibility_analysis(self, quality_analyzer):
        """Test accessibility analysis of documentation"""
        await quality_analyzer.initialize()

        context = create_test_context(
            content=sample_technical_documents["user_guide"],
            metadata={"focus": "accessibility"},
        )

        result = await quality_analyzer.analyze_quality(context)

        assert "accessibility_score" in result
        assert "readability_level" in result
        assert "accessibility_issues" in result
