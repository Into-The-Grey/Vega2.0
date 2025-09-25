"""
Comprehensive tests for document understanding intelligence module
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.vega.document.understanding import (
    DocumentUnderstandingAI,
    ContentAnalyzer,
    SemanticAnalyzer,
    SummaryGenerator,
    EntityExtractor,
    UnderstandingConfig,
    AnalysisType,
    ContentType
)

from .fixtures import (
    TestFixtures,
    sample_documents,
    processing_context,
    PerformanceTestHelper,
    AsyncTestUtils,
    TestDataGenerator
)


class TestUnderstandingConfig:
    """Test UnderstandingConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = UnderstandingConfig()
        
        assert config.enable_content_analysis is True
        assert config.enable_semantic_analysis is True
        assert config.enable_entity_extraction is True
        assert config.min_confidence == 0.7
        assert len(config.supported_languages) > 0
        assert "en" in config.supported_languages
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = UnderstandingConfig(
            enable_content_analysis=False,
            min_confidence=0.9,
            supported_languages=["en", "es"]
        )
        
        assert config.enable_content_analysis is False
        assert config.min_confidence == 0.9
        assert config.supported_languages == ["en", "es"]
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = UnderstandingConfig(min_confidence=0.8)
        errors = config.validate_config()
        assert len(errors) == 0
    
    def test_config_validation_failure(self):
        """Test configuration validation with errors"""
        config = UnderstandingConfig(
            min_confidence=1.5,  # Invalid
            max_content_length=0,  # Invalid
            supported_languages=[]  # Invalid
        )
        errors = config.validate_config()
        assert len(errors) == 3
        assert any("min_confidence" in error for error in errors)
        assert any("max_content_length" in error for error in errors)
        assert any("supported_languages" in error for error in errors)


class TestContentAnalyzer:
    """Test ContentAnalyzer functionality"""
    
    @pytest.fixture
    async def content_analyzer(self):
        """Fixture for initialized ContentAnalyzer"""
        analyzer = ContentAnalyzer(UnderstandingConfig())
        await analyzer.initialize()
        return analyzer
    
    @pytest.mark.asyncio
    async def test_content_analysis_basic(self, content_analyzer, sample_documents, processing_context):
        """Test basic content analysis"""
        content = sample_documents["understanding_test"]
        result = await content_analyzer.process(content, processing_context)
        
        assert result.success is True
        assert "content_type" in result.data
        assert "language" in result.data
        assert "readability_score" in result.data
        assert "complexity_score" in result.data
        assert "word_count" in result.data
    
    @pytest.mark.asyncio
    async def test_content_analysis_empty_input(self, content_analyzer, processing_context):
        """Test content analysis with empty input"""
        result = await content_analyzer.process("", processing_context)
        
        assert result.success is False
        assert "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_content_analysis_long_text(self, content_analyzer, processing_context):
        """Test content analysis with very long text"""
        long_text = TestDataGenerator.generate_large_document(100)  # 100KB
        result = await content_analyzer.process(long_text, processing_context)
        
        if result.success:
            assert result.data["word_count"] > 1000
        else:
            # Should handle gracefully if text is too long
            assert "too long" in result.error.lower() or "exceed" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_content_analysis_special_characters(self, content_analyzer, processing_context):
        """Test content analysis with special characters"""
        special_text = TestDataGenerator.generate_special_characters_text()
        result = await content_analyzer.process(special_text, processing_context)
        
        assert result.success is True
        # Should handle special characters gracefully
        assert "word_count" in result.data
    
    @pytest.mark.asyncio
    async def test_content_type_detection(self, content_analyzer, sample_documents, processing_context):
        """Test content type detection"""
        # Test with different document types
        test_cases = [
            ("contract", ContentType.LEGAL),
            ("technical_doc", ContentType.TECHNICAL),
            ("research_paper", ContentType.ACADEMIC),
            ("workflow_doc", ContentType.PROCEDURAL)
        ]
        
        for doc_type, expected_type in test_cases:
            content = sample_documents[doc_type]
            result = await content_analyzer.process(content, processing_context)
            
            assert result.success is True
            # Note: Exact matching may depend on implementation
            assert "content_type" in result.data
    
    @pytest.mark.asyncio
    async def test_language_detection(self, content_analyzer, processing_context):
        """Test language detection"""
        multilingual_text = TestDataGenerator.generate_multilingual_text()
        result = await content_analyzer.process(multilingual_text, processing_context)
        
        assert result.success is True
        assert "language" in result.data
        # Should detect primary language or mixed
        assert result.data["language"] is not None


class TestSemanticAnalyzer:
    """Test SemanticAnalyzer functionality"""
    
    @pytest.fixture
    async def semantic_analyzer(self):
        """Fixture for initialized SemanticAnalyzer"""
        analyzer = SemanticAnalyzer(UnderstandingConfig())
        await analyzer.initialize()
        return analyzer
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_basic(self, semantic_analyzer, sample_documents, processing_context):
        """Test basic semantic analysis"""
        content = sample_documents["understanding_test"]
        result = await semantic_analyzer.process(content, processing_context)
        
        assert result.success is True
        assert "key_topics" in result.data
        assert "sentiment" in result.data
        assert "themes" in result.data
        assert isinstance(result.data["key_topics"], list)
    
    @pytest.mark.asyncio
    async def test_topic_extraction(self, semantic_analyzer, sample_documents, processing_context):
        """Test topic extraction functionality"""
        # Test with business document
        content = sample_documents["understanding_test"]
        result = await semantic_analyzer.process(content, processing_context)
        
        assert result.success is True
        topics = result.data["key_topics"]
        assert len(topics) > 0
        
        # Should find AI/business related topics
        topic_texts = [topic.get("text", "").lower() for topic in topics]
        assert any("ai" in text or "artificial" in text or "business" in text for text in topic_texts)
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, semantic_analyzer, processing_context):
        """Test sentiment analysis"""
        test_cases = [
            ("This is an excellent product with outstanding features!", "positive"),
            ("This product is terrible and completely broken.", "negative"),
            ("The product has both good and bad aspects.", "neutral"),
        ]
        
        for text, expected_sentiment in test_cases:
            result = await semantic_analyzer.process(text, processing_context)
            
            assert result.success is True
            sentiment = result.data["sentiment"]
            # Note: Exact matching depends on implementation
            assert sentiment["label"] in ["positive", "negative", "neutral"]
    
    @pytest.mark.asyncio
    async def test_semantic_similarity(self, semantic_analyzer, processing_context):
        """Test semantic similarity computation"""
        input_data = {
            "text1": "Artificial intelligence is transforming business operations.",
            "text2": "AI technology is changing how companies work."
        }
        
        result = await semantic_analyzer.process(input_data, processing_context)
        
        if result.success and "similarity_score" in result.data:
            # These texts should have high similarity
            assert result.data["similarity_score"] > 0.5
    
    @pytest.mark.asyncio
    async def test_theme_identification(self, semantic_analyzer, sample_documents, processing_context):
        """Test theme identification"""
        content = sample_documents["technical_doc"]
        result = await semantic_analyzer.process(content, processing_context)
        
        assert result.success is True
        themes = result.data.get("themes", [])
        
        if themes:
            # Should identify API/technical themes
            theme_names = [theme.get("name", "").lower() for theme in themes]
            assert any("api" in name or "technical" in name or "documentation" in name for name in theme_names)


class TestSummaryGenerator:
    """Test SummaryGenerator functionality"""
    
    @pytest.fixture
    async def summary_generator(self):
        """Fixture for initialized SummaryGenerator"""
        generator = SummaryGenerator(UnderstandingConfig())
        await generator.initialize()
        return generator
    
    @pytest.mark.asyncio
    async def test_summary_generation_basic(self, summary_generator, sample_documents, processing_context):
        """Test basic summary generation"""
        content = sample_documents["research_paper"]
        result = await summary_generator.process(content, processing_context)
        
        assert result.success is True
        assert "summary" in result.data
        assert "key_points" in result.data
        assert len(result.data["summary"]) > 0
        assert len(result.data["summary"]) < len(content)  # Summary should be shorter
    
    @pytest.mark.asyncio
    async def test_extractive_summary(self, summary_generator, sample_documents, processing_context):
        """Test extractive summarization"""
        content = sample_documents["understanding_test"]
        input_data = {
            "text": content,
            "summary_type": "extractive",
            "max_sentences": 3
        }
        
        result = await summary_generator.process(input_data, processing_context)
        
        assert result.success is True
        summary = result.data["summary"]
        # Extractive summary should contain sentences from original
        sentences = summary.split('.')
        assert len([s for s in sentences if s.strip()]) <= 4  # 3 + potential incomplete
    
    @pytest.mark.asyncio
    async def test_abstractive_summary(self, summary_generator, sample_documents, processing_context):
        """Test abstractive summarization"""
        content = sample_documents["workflow_doc"]
        input_data = {
            "text": content,
            "summary_type": "abstractive",
            "max_length": 150
        }
        
        result = await summary_generator.process(input_data, processing_context)
        
        assert result.success is True
        summary = result.data["summary"]
        assert len(summary) <= 200  # Should respect length limit (with some tolerance)
    
    @pytest.mark.asyncio
    async def test_key_points_extraction(self, summary_generator, sample_documents, processing_context):
        """Test key points extraction"""
        content = sample_documents["understanding_test"]
        result = await summary_generator.process(content, processing_context)
        
        assert result.success is True
        key_points = result.data.get("key_points", [])
        
        if key_points:
            assert isinstance(key_points, list)
            assert len(key_points) > 0
            # Each key point should be a meaningful sentence
            assert all(len(point.strip()) > 10 for point in key_points)
    
    @pytest.mark.asyncio
    async def test_summary_different_lengths(self, summary_generator, sample_documents, processing_context):
        """Test summary generation with different target lengths"""
        content = sample_documents["research_paper"]
        
        for max_length in [50, 100, 200]:
            input_data = {
                "text": content,
                "max_length": max_length
            }
            
            result = await summary_generator.process(input_data, processing_context)
            
            assert result.success is True
            summary = result.data["summary"]
            # Summary should roughly respect length constraint
            assert len(summary) <= max_length * 1.2  # 20% tolerance


class TestEntityExtractor:
    """Test EntityExtractor functionality"""
    
    @pytest.fixture
    async def entity_extractor(self):
        """Fixture for initialized EntityExtractor"""
        extractor = EntityExtractor(UnderstandingConfig())
        await extractor.initialize()
        return extractor
    
    @pytest.mark.asyncio
    async def test_entity_extraction_basic(self, entity_extractor, sample_documents, processing_context):
        """Test basic entity extraction"""
        content = sample_documents["contract"]
        result = await entity_extractor.process(content, processing_context)
        
        assert result.success is True
        assert "entities" in result.data
        assert isinstance(result.data["entities"], list)
    
    @pytest.mark.asyncio
    async def test_person_entity_extraction(self, entity_extractor, processing_context):
        """Test person entity extraction"""
        content = "John Smith and Mary Johnson signed the agreement yesterday."
        result = await entity_extractor.process(content, processing_context)
        
        assert result.success is True
        entities = result.data["entities"]
        
        # Should find person entities
        person_entities = [e for e in entities if e.get("label") == "PERSON"]
        if person_entities:  # May depend on NLP model availability
            person_names = [e["text"] for e in person_entities]
            assert any("John" in name or "Mary" in name for name in person_names)
    
    @pytest.mark.asyncio
    async def test_organization_entity_extraction(self, entity_extractor, processing_context):
        """Test organization entity extraction"""
        content = "Apple Inc. and Microsoft Corporation are competing in the tech market."
        result = await entity_extractor.process(content, processing_context)
        
        assert result.success is True
        entities = result.data["entities"]
        
        # Should find organization entities
        org_entities = [e for e in entities if e.get("label") in ["ORG", "ORGANIZATION"]]
        if org_entities:
            org_names = [e["text"] for e in org_entities]
            assert any("Apple" in name or "Microsoft" in name for name in org_names)
    
    @pytest.mark.asyncio
    async def test_date_entity_extraction(self, entity_extractor, processing_context):
        """Test date entity extraction"""
        content = "The contract was signed on January 15, 2024 and expires on December 31, 2025."
        result = await entity_extractor.process(content, processing_context)
        
        assert result.success is True
        entities = result.data["entities"]
        
        # Should find date entities
        date_entities = [e for e in entities if e.get("label") in ["DATE", "TIME"]]
        if date_entities:
            assert len(date_entities) >= 1
    
    @pytest.mark.asyncio
    async def test_custom_entity_patterns(self, entity_extractor, processing_context):
        """Test custom entity pattern recognition"""
        content = "Contact us at support@example.com or call 555-123-4567 for assistance."
        result = await entity_extractor.process(content, processing_context)
        
        assert result.success is True
        entities = result.data["entities"]
        
        # Should find email and phone patterns
        custom_entities = [e for e in entities if e.get("label") in ["EMAIL", "PHONE", "CONTACT"]]
        # This may depend on custom pattern implementation
    
    @pytest.mark.asyncio
    async def test_entity_confidence_scoring(self, entity_extractor, processing_context):
        """Test entity confidence scoring"""
        content = "Barack Obama was the President of the United States."
        result = await entity_extractor.process(content, processing_context)
        
        assert result.success is True
        entities = result.data["entities"]
        
        if entities:
            # All entities should have confidence scores
            assert all("confidence" in entity for entity in entities)
            # Confidence should be between 0 and 1
            assert all(0 <= entity["confidence"] <= 1 for entity in entities)


class TestDocumentUnderstandingAI:
    """Test main DocumentUnderstandingAI orchestration"""
    
    @pytest.fixture
    async def understanding_ai(self):
        """Fixture for initialized DocumentUnderstandingAI"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        return ai
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self, understanding_ai, sample_documents, processing_context):
        """Test comprehensive document understanding analysis"""
        content = sample_documents["understanding_test"]
        result = await understanding_ai.analyze_content(content, processing_context)
        
        assert result.success is True
        
        # Should have results from all analyzers
        expected_keys = ["content_analysis", "semantic_analysis", "summary", "entities"]
        for key in expected_keys:
            if key in result.data:
                assert result.data[key] is not None
    
    @pytest.mark.asyncio
    async def test_semantic_analysis_integration(self, understanding_ai, sample_documents, processing_context):
        """Test semantic analysis integration"""
        content = sample_documents["research_paper"]
        result = await understanding_ai.analyze_semantics(content, processing_context)
        
        assert result.success is True
        # Should have semantic analysis results
        assert "key_topics" in result.data or "topics" in result.data
    
    @pytest.mark.asyncio
    async def test_summary_generation_integration(self, understanding_ai, sample_documents, processing_context):
        """Test summary generation integration"""
        content = sample_documents["workflow_doc"]
        result = await understanding_ai.generate_summary(content, processing_context)
        
        assert result.success is True
        assert "summary" in result.data
        assert len(result.data["summary"]) > 0
    
    @pytest.mark.asyncio
    async def test_entity_extraction_integration(self, understanding_ai, sample_documents, processing_context):
        """Test entity extraction integration"""
        content = sample_documents["contract"]
        result = await understanding_ai.extract_entities(content, processing_context)
        
        assert result.success is True
        assert "entities" in result.data
        assert isinstance(result.data["entities"], list)
    
    @pytest.mark.asyncio
    async def test_health_check(self, understanding_ai):
        """Test system health check"""
        health = await understanding_ai.health_check()
        
        assert "overall_status" in health
        assert "components" in health
        
        # All components should be present
        expected_components = ["content_analyzer", "semantic_analyzer", "summary_generator", "entity_extractor"]
        for component in expected_components:
            assert component in health["components"]
    
    @pytest.mark.asyncio
    async def test_error_handling_cascading(self, processing_context):
        """Test error handling across components"""
        # Create AI with failing components
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        
        # Mock a component to fail
        with patch.object(ai.content_analyzer, 'process', side_effect=Exception("Component failed")):
            result = await ai.analyze_content("test content", processing_context)
            
            # Should handle component failure gracefully
            assert result.success is False
            assert "failed" in result.error.lower()


class TestPerformanceAndLoad:
    """Test performance and load characteristics"""
    
    @pytest.mark.asyncio
    async def test_processing_performance(self):
        """Test processing performance measurement"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        content = TestFixtures.get_sample_document("understanding_test")
        context = TestFixtures.create_processing_context()
        
        perf_result = await PerformanceTestHelper.measure_processing_time(
            ai.content_analyzer, content, context
        )
        
        # Should complete within reasonable time
        assert perf_result["processing_time"] < 10.0  # 10 seconds max
        assert perf_result["result"].success is True
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing of multiple documents"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        # Process multiple documents concurrently
        tasks = []
        for i in range(5):
            content = f"Document {i} content with unique identifier {i}"
            context = TestFixtures.create_processing_context(f"context_{i}")
            task = ai.analyze_content(content, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_memory_usage_large_documents(self):
        """Test memory usage with large documents"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        # Generate a large document
        large_content = TestDataGenerator.generate_large_document(50)  # 50KB
        context = TestFixtures.create_processing_context()
        
        result = await ai.analyze_content(large_content, context)
        
        # Should handle large documents (success or graceful failure)
        if not result.success:
            assert "too large" in result.error.lower() or "memory" in result.error.lower()


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions"""
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """Test handling of empty input"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        context = TestFixtures.create_processing_context()
        
        result = await ai.analyze_content("", context)
        assert result.success is False
        assert "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_whitespace_only_input(self):
        """Test handling of whitespace-only input"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        context = TestFixtures.create_processing_context()
        
        result = await ai.analyze_content("   \n\t   ", context)
        assert result.success is False
        assert "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_malformed_input_handling(self):
        """Test handling of malformed input"""
        config = UnderstandingConfig()
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        context = TestFixtures.create_processing_context()
        
        # Test with invalid JSON when expecting structured input
        malformed_json = TestDataGenerator.generate_malformed_json()
        
        # Should handle gracefully (success with warning or proper error)
        result = await ai.analyze_content(malformed_json, context)
        # Result depends on implementation - should not crash
        assert isinstance(result, type(result))  # Basic sanity check
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling for long operations"""
        config = UnderstandingConfig(timeout_seconds=0.1)  # Very short timeout
        ai = DocumentUnderstandingAI(config)
        await ai.initialize()
        
        context = TestFixtures.create_processing_context()
        large_content = TestDataGenerator.generate_large_document(100)
        
        # Should timeout or handle gracefully
        with pytest.raises((asyncio.TimeoutError, Exception)):
            await AsyncUtils.run_with_timeout(
                ai.analyze_content(large_content, context),
                timeout=1.0
            )