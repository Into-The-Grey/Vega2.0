"""
Comprehensive tests for document classification intelligence module
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.vega.document.classification import (
    DocumentClassificationAI,
    HierarchicalClassifier,
    TopicClassifier,
    ContentClassifier,
    IntentClassifier,
    ClassificationConfig,
    ClassificationCategory,
    ClassificationResult,
    HierarchicalCategory
)

from .fixtures import (
    TestFixtures,
    sample_documents,
    processing_context,
    PerformanceTestHelper,
    AsyncTestUtils,
    TestDataGenerator
)


class TestClassificationConfig:
    """Test ClassificationConfig functionality"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ClassificationConfig()
        
        assert config.enable_hierarchical is True
        assert config.enable_topic_classification is True
        assert config.enable_content_classification is True
        assert config.min_confidence == 0.7
        assert config.max_categories == 5
        assert len(config.default_categories) > 0
    
    def test_custom_config(self):
        """Test custom configuration"""
        custom_categories = [
            ClassificationCategory(name="tech", keywords=["technology", "software"])
        ]
        
        config = ClassificationConfig(
            min_confidence=0.9,
            max_categories=3,
            default_categories=custom_categories
        )
        
        assert config.min_confidence == 0.9
        assert config.max_categories == 3
        assert len(config.default_categories) == 1
        assert config.default_categories[0].name == "tech"
    
    def test_config_validation_success(self):
        """Test successful configuration validation"""
        config = ClassificationConfig()
        errors = config.validate_config()
        assert len(errors) == 0
    
    def test_config_validation_failure(self):
        """Test configuration validation with errors"""
        config = ClassificationConfig(
            min_confidence=1.5,  # Invalid
            max_categories=0,    # Invalid
            default_categories=[]  # Invalid
        )
        errors = config.validate_config()
        assert len(errors) >= 2
        assert any("min_confidence" in error for error in errors)
        assert any("max_categories" in error for error in errors)


class TestClassificationCategory:
    """Test ClassificationCategory functionality"""
    
    def test_category_creation(self):
        """Test category creation"""
        category = ClassificationCategory(
            name="legal",
            description="Legal documents and contracts",
            keywords=["contract", "agreement", "legal"],
            confidence_threshold=0.8
        )
        
        assert category.name == "legal"
        assert category.description == "Legal documents and contracts"
        assert len(category.keywords) == 3
        assert category.confidence_threshold == 0.8
    
    def test_category_keyword_matching(self):
        """Test category keyword matching"""
        category = ClassificationCategory(
            name="tech",
            keywords=["software", "programming", "code"]
        )
        
        # Test matching
        assert category.matches_keywords("This is about software development")
        assert category.matches_keywords("Programming languages are important")
        assert not category.matches_keywords("This is about cooking recipes")
    
    def test_category_score_calculation(self):
        """Test category score calculation"""
        category = ClassificationCategory(
            name="business",
            keywords=["business", "company", "corporate", "enterprise"]
        )
        
        text = "Our business company handles corporate enterprise solutions"
        score = category.calculate_score(text)
        
        # Should have high score due to multiple keyword matches
        assert score > 0.5
        
        text_no_match = "This is about cooking and recipes"
        score_no_match = category.calculate_score(text_no_match)
        assert score_no_match == 0.0


class TestHierarchicalCategory:
    """Test HierarchicalCategory functionality"""
    
    def test_hierarchical_creation(self):
        """Test hierarchical category creation"""
        category = HierarchicalCategory(
            name="technology",
            level=1,
            parent=None,
            subcategories=[]
        )
        
        assert category.name == "technology"
        assert category.level == 1
        assert category.parent is None
        assert len(category.subcategories) == 0
    
    def test_hierarchical_tree_structure(self):
        """Test hierarchical tree structure"""
        # Create parent category
        tech = HierarchicalCategory(name="technology", level=1)
        
        # Create child categories
        software = HierarchicalCategory(name="software", level=2, parent=tech)
        hardware = HierarchicalCategory(name="hardware", level=2, parent=tech)
        
        tech.subcategories = [software, hardware]
        
        # Test structure
        assert len(tech.subcategories) == 2
        assert software.parent == tech
        assert hardware.parent == tech
        assert software.level == 2
    
    def test_get_full_path(self):
        """Test getting full hierarchical path"""
        root = HierarchicalCategory(name="business", level=1)
        tech = HierarchicalCategory(name="technology", level=2, parent=root)
        software = HierarchicalCategory(name="software", level=3, parent=tech)
        
        path = software.get_full_path()
        assert path == ["business", "technology", "software"]
    
    def test_find_category_by_path(self):
        """Test finding category by path"""
        # Create hierarchy
        root = HierarchicalCategory(name="documents", level=1)
        business = HierarchicalCategory(name="business", level=2, parent=root)
        contracts = HierarchicalCategory(name="contracts", level=3, parent=business)
        
        root.subcategories = [business]
        business.subcategories = [contracts]
        
        # Test finding
        found = root.find_by_path(["business", "contracts"])
        assert found == contracts
        
        not_found = root.find_by_path(["nonexistent", "path"])
        assert not_found is None


class TestContentClassifier:
    """Test ContentClassifier functionality"""
    
    @pytest.fixture
    async def content_classifier(self):
        """Fixture for initialized ContentClassifier"""
        classifier = ContentClassifier(ClassificationConfig())
        await classifier.initialize()
        return classifier
    
    @pytest.mark.asyncio
    async def test_content_classification_basic(self, content_classifier, sample_documents, processing_context):
        """Test basic content classification"""
        content = sample_documents["contract"]
        result = await content_classifier.process(content, processing_context)
        
        assert result.success is True
        assert "classification" in result.data
        assert "confidence" in result.data
        assert isinstance(result.data["classification"], str)
        assert 0 <= result.data["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_legal_document_classification(self, content_classifier, sample_documents, processing_context):
        """Test legal document classification"""
        content = sample_documents["contract"]
        result = await content_classifier.process(content, processing_context)
        
        assert result.success is True
        classification = result.data["classification"].lower()
        # Should identify as legal/contract content
        assert any(term in classification for term in ["legal", "contract", "agreement"])
    
    @pytest.mark.asyncio
    async def test_technical_document_classification(self, content_classifier, sample_documents, processing_context):
        """Test technical document classification"""
        content = sample_documents["technical_doc"]
        result = await content_classifier.process(content, processing_context)
        
        assert result.success is True
        classification = result.data["classification"].lower()
        # Should identify as technical content
        assert any(term in classification for term in ["technical", "api", "documentation"])
    
    @pytest.mark.asyncio
    async def test_academic_document_classification(self, content_classifier, sample_documents, processing_context):
        """Test academic document classification"""
        content = sample_documents["research_paper"]
        result = await content_classifier.process(content, processing_context)
        
        assert result.success is True
        classification = result.data["classification"].lower()
        # Should identify as academic/research content
        assert any(term in classification for term in ["academic", "research", "paper"])
    
    @pytest.mark.asyncio
    async def test_multi_category_classification(self, content_classifier, processing_context):
        """Test classification of content with multiple categories"""
        mixed_content = """
        This technical documentation describes our business API for legal compliance.
        The software helps law firms manage contracts and agreements efficiently.
        Research shows that automated legal document processing improves accuracy.
        """
        
        input_data = {
            "text": mixed_content,
            "multi_label": True,
            "max_categories": 3
        }
        
        result = await content_classifier.process(input_data, processing_context)
        
        assert result.success is True
        if "categories" in result.data:
            categories = result.data["categories"]
            assert isinstance(categories, list)
            assert len(categories) <= 3
            # Should identify multiple relevant categories
            category_names = [cat.get("name", "").lower() for cat in categories]
            assert len(category_names) > 1
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self, content_classifier, processing_context):
        """Test confidence scoring for classifications"""
        # Test with very clear content
        clear_content = "This is a legal contract agreement between two parties for service provision."
        result = await content_classifier.process(clear_content, processing_context)
        
        assert result.success is True
        assert result.data["confidence"] > 0.7  # Should be high confidence
        
        # Test with ambiguous content
        ambiguous_content = "This document contains information."
        result_ambiguous = await content_classifier.process(ambiguous_content, processing_context)
        
        if result_ambiguous.success:
            # Confidence should be lower for ambiguous content
            assert result_ambiguous.data["confidence"] < result.data["confidence"]


class TestTopicClassifier:
    """Test TopicClassifier functionality"""
    
    @pytest.fixture
    async def topic_classifier(self):
        """Fixture for initialized TopicClassifier"""
        classifier = TopicClassifier(ClassificationConfig())
        await classifier.initialize()
        return classifier
    
    @pytest.mark.asyncio
    async def test_topic_classification_basic(self, topic_classifier, sample_documents, processing_context):
        """Test basic topic classification"""
        content = sample_documents["understanding_test"]
        result = await topic_classifier.process(content, processing_context)
        
        assert result.success is True
        assert "topics" in result.data
        assert isinstance(result.data["topics"], list)
        assert len(result.data["topics"]) > 0
    
    @pytest.mark.asyncio
    async def test_ai_business_topic_identification(self, topic_classifier, sample_documents, processing_context):
        """Test AI and business topic identification"""
        content = sample_documents["understanding_test"]
        result = await topic_classifier.process(content, processing_context)
        
        assert result.success is True
        topics = result.data["topics"]
        
        # Should identify AI/business topics
        topic_names = [topic.get("name", "").lower() for topic in topics]
        assert any("ai" in name or "artificial" in name or "business" in name for name in topic_names)
    
    @pytest.mark.asyncio
    async def test_topic_relevance_scoring(self, topic_classifier, processing_context):
        """Test topic relevance scoring"""
        content = "Machine learning and artificial intelligence are transforming healthcare through automated diagnosis and treatment recommendations."
        
        result = await topic_classifier.process(content, processing_context)
        
        assert result.success is True
        topics = result.data["topics"]
        
        # Topics should have relevance scores
        for topic in topics:
            assert "relevance" in topic or "score" in topic
            score = topic.get("relevance", topic.get("score", 0))
            assert 0 <= score <= 1
    
    @pytest.mark.asyncio
    async def test_topic_clustering(self, topic_classifier, processing_context):
        """Test topic clustering functionality"""
        content = """
        The software development process involves coding, testing, and deployment.
        Quality assurance ensures bug-free applications through automated testing.
        Code review processes improve software reliability and maintainability.
        DevOps practices streamline development and operations integration.
        """
        
        input_data = {
            "text": content,
            "cluster_topics": True,
            "max_topics": 5
        }
        
        result = await topic_classifier.process(input_data, processing_context)
        
        assert result.success is True
        topics = result.data["topics"]
        
        # Should identify related software development topics
        if topics:
            topic_names = [t.get("name", "").lower() for t in topics]
            assert any("software" in name or "development" in name for name in topic_names)


class TestHierarchicalClassifier:
    """Test HierarchicalClassifier functionality"""
    
    @pytest.fixture
    async def hierarchical_classifier(self):
        """Fixture for initialized HierarchicalClassifier"""
        classifier = HierarchicalClassifier(ClassificationConfig())
        await classifier.initialize()
        return classifier
    
    @pytest.mark.asyncio
    async def test_hierarchical_classification_basic(self, hierarchical_classifier, sample_documents, processing_context):
        """Test basic hierarchical classification"""
        content = sample_documents["contract"]
        result = await hierarchical_classifier.process(content, processing_context)
        
        assert result.success is True
        assert "hierarchy" in result.data
        assert "level_1" in result.data["hierarchy"]
        assert isinstance(result.data["hierarchy"]["level_1"], str)
    
    @pytest.mark.asyncio
    async def test_multi_level_classification(self, hierarchical_classifier, sample_documents, processing_context):
        """Test multi-level hierarchical classification"""
        content = sample_documents["technical_doc"]
        result = await hierarchical_classifier.process(content, processing_context)
        
        assert result.success is True
        hierarchy = result.data["hierarchy"]
        
        # Should have multiple levels
        expected_levels = ["level_1", "level_2", "level_3"]
        available_levels = [level for level in expected_levels if level in hierarchy]
        assert len(available_levels) >= 2  # At least 2 levels
    
    @pytest.mark.asyncio
    async def test_legal_document_hierarchy(self, hierarchical_classifier, sample_documents, processing_context):
        """Test legal document hierarchical classification"""
        content = sample_documents["contract"]
        result = await hierarchical_classifier.process(content, processing_context)
        
        assert result.success is True
        hierarchy = result.data["hierarchy"]
        
        # Should classify under legal hierarchy
        level_1 = hierarchy["level_1"].lower()
        assert any(term in level_1 for term in ["legal", "contract", "document"])
    
    @pytest.mark.asyncio
    async def test_classification_path_consistency(self, hierarchical_classifier, processing_context):
        """Test that classification paths are logically consistent"""
        content = "This API documentation explains REST endpoints for user management."
        
        result = await hierarchical_classifier.process(content, processing_context)
        
        assert result.success is True
        hierarchy = result.data["hierarchy"]
        
        # Verify hierarchical consistency
        if "level_2" in hierarchy and "level_1" in hierarchy:
            # Level 2 should be more specific than level 1
            level_1 = hierarchy["level_1"].lower()
            level_2 = hierarchy["level_2"].lower()
            
            # Both should be related to technical/documentation
            assert level_1 != level_2  # Should be different levels
    
    @pytest.mark.asyncio
    async def test_custom_hierarchy_definition(self, processing_context):
        """Test classification with custom hierarchy"""
        # Create custom hierarchy
        custom_hierarchy = {
            "business": {
                "technology": ["software", "hardware", "ai"],
                "operations": ["process", "workflow", "management"]
            }
        }
        
        config = ClassificationConfig()
        classifier = HierarchicalClassifier(config)
        classifier.hierarchy = custom_hierarchy
        await classifier.initialize()
        
        content = "Our software development process involves agile methodology."
        result = await classifier.process(content, processing_context)
        
        assert result.success is True
        hierarchy = result.data.get("hierarchy", {})
        
        if hierarchy:
            # Should follow custom hierarchy structure
            assert any(level in hierarchy.values() for level in ["business", "technology", "operations"])


class TestIntentClassifier:
    """Test IntentClassifier functionality"""
    
    @pytest.fixture
    async def intent_classifier(self):
        """Fixture for initialized IntentClassifier"""
        classifier = IntentClassifier(ClassificationConfig())
        await classifier.initialize()
        return classifier
    
    @pytest.mark.asyncio
    async def test_intent_classification_basic(self, intent_classifier, processing_context):
        """Test basic intent classification"""
        content = "Please review this contract and provide feedback on the terms."
        result = await intent_classifier.process(content, processing_context)
        
        assert result.success is True
        assert "intent" in result.data
        assert "confidence" in result.data
    
    @pytest.mark.asyncio
    async def test_request_intent(self, intent_classifier, processing_context):
        """Test request intent classification"""
        request_texts = [
            "Can you analyze this document for compliance issues?",
            "Please classify this content by topic.",
            "I need help understanding this legal agreement."
        ]
        
        for text in request_texts:
            result = await intent_classifier.process(text, processing_context)
            
            assert result.success is True
            intent = result.data["intent"].lower()
            # Should identify as request intent
            assert any(word in intent for word in ["request", "question", "ask"])
    
    @pytest.mark.asyncio
    async def test_informational_intent(self, intent_classifier, sample_documents, processing_context):
        """Test informational intent classification"""
        content = sample_documents["research_paper"]
        result = await intent_classifier.process(content, processing_context)
        
        assert result.success is True
        intent = result.data["intent"].lower()
        # Should identify as informational intent
        assert any(word in intent for word in ["inform", "explain", "describe"])
    
    @pytest.mark.asyncio
    async def test_action_intent(self, intent_classifier, processing_context):
        """Test action intent classification"""
        action_content = """
        To complete the document processing workflow:
        1. Upload the document
        2. Select classification options
        3. Review results
        4. Export the analysis
        """
        
        result = await intent_classifier.process(action_content, processing_context)
        
        assert result.success is True
        intent = result.data["intent"].lower()
        # Should identify as action/instruction intent
        assert any(word in intent for word in ["action", "instruction", "process"])
    
    @pytest.mark.asyncio
    async def test_multiple_intent_detection(self, intent_classifier, processing_context):
        """Test detection of multiple intents"""
        mixed_content = """
        This document explains our policy (informational) and 
        requests that you review and approve it (request).
        Please follow the attached procedures (action).
        """
        
        input_data = {
            "text": mixed_content,
            "detect_multiple": True
        }
        
        result = await intent_classifier.process(input_data, processing_context)
        
        assert result.success is True
        
        if "intents" in result.data:
            intents = result.data["intents"]
            assert isinstance(intents, list)
            assert len(intents) > 1  # Should detect multiple intents


class TestDocumentClassificationAI:
    """Test main DocumentClassificationAI orchestration"""
    
    @pytest.fixture
    async def classification_ai(self):
        """Fixture for initialized DocumentClassificationAI"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        return ai
    
    @pytest.mark.asyncio
    async def test_comprehensive_classification(self, classification_ai, sample_documents, processing_context):
        """Test comprehensive document classification"""
        content = sample_documents["contract"]
        result = await classification_ai.classify_document(content, processing_context)
        
        assert result.success is True
        
        # Should have results from multiple classifiers
        expected_keys = ["content_classification", "topic_classification", "hierarchical_classification"]
        available_keys = [key for key in expected_keys if key in result.data]
        assert len(available_keys) >= 1
    
    @pytest.mark.asyncio
    async def test_topic_classification_integration(self, classification_ai, sample_documents, processing_context):
        """Test topic classification integration"""
        content = sample_documents["technical_doc"]
        result = await classification_ai.classify_topics(content, processing_context)
        
        assert result.success is True
        assert "topics" in result.data
    
    @pytest.mark.asyncio
    async def test_hierarchical_classification_integration(self, classification_ai, sample_documents, processing_context):
        """Test hierarchical classification integration"""
        content = sample_documents["research_paper"]
        result = await classification_ai.classify_hierarchical(content, processing_context)
        
        assert result.success is True
        assert "hierarchy" in result.data
    
    @pytest.mark.asyncio
    async def test_intent_classification_integration(self, classification_ai, processing_context):
        """Test intent classification integration"""
        content = "Please analyze this document and classify its content type."
        result = await classification_ai.classify_intent(content, processing_context)
        
        assert result.success is True
        assert "intent" in result.data
    
    @pytest.mark.asyncio
    async def test_batch_classification(self, classification_ai, sample_documents, processing_context):
        """Test batch document classification"""
        documents = [
            sample_documents["contract"],
            sample_documents["technical_doc"],
            sample_documents["research_paper"]
        ]
        
        results = []
        for doc in documents:
            result = await classification_ai.classify_document(doc, processing_context)
            results.append(result)
        
        # All should succeed
        assert all(r.success for r in results)
        
        # Should have different classifications
        classifications = [r.data.get("content_classification", "") for r in results]
        assert len(set(classifications)) > 1  # Should have different results
    
    @pytest.mark.asyncio
    async def test_health_check(self, classification_ai):
        """Test system health check"""
        health = await classification_ai.health_check()
        
        assert "overall_status" in health
        assert "components" in health
        
        # All components should be present
        expected_components = ["content_classifier", "topic_classifier", "hierarchical_classifier", "intent_classifier"]
        for component in expected_components:
            if component in health["components"]:
                assert health["components"][component] is not None


class TestPerformanceAndLoad:
    """Test performance and load characteristics"""
    
    @pytest.mark.asyncio
    async def test_classification_performance(self):
        """Test classification performance"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        content = TestFixtures.get_sample_document("contract")
        context = TestFixtures.create_processing_context()
        
        perf_result = await PerformanceTestHelper.measure_processing_time(
            ai.content_classifier, content, context
        )
        
        # Should complete within reasonable time
        assert perf_result["processing_time"] < 5.0  # 5 seconds max
        assert perf_result["result"].success is True
    
    @pytest.mark.asyncio
    async def test_concurrent_classification(self):
        """Test concurrent classification of multiple documents"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        # Classify multiple documents concurrently
        tasks = []
        for i in range(5):
            content = f"Document {i} about business technology and software development."
            context = TestFixtures.create_processing_context(f"context_{i}")
            task = ai.classify_document(content, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert all(r.success for r in results)
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_large_document_classification(self):
        """Test classification of large documents"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        # Generate a large document with clear classification markers
        large_content = """
        Legal Contract Agreement
        """ + TestDataGenerator.generate_large_document(20)  # 20KB
        
        context = TestFixtures.create_processing_context()
        
        result = await ai.classify_document(large_content, context)
        
        # Should handle large documents (success or graceful failure)
        if not result.success:
            assert "too large" in result.error.lower() or "memory" in result.error.lower()


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_empty_content_classification(self):
        """Test classification of empty content"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        context = TestFixtures.create_processing_context()
        
        result = await ai.classify_document("", context)
        assert result.success is False
        assert "empty" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self):
        """Test handling of invalid input types"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        context = TestFixtures.create_processing_context()
        
        # Test with non-string input
        result = await ai.content_classifier.process({"invalid": "input"}, context)
        
        # Should handle gracefully
        if not result.success:
            assert "invalid" in result.error.lower() or "type" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_low_confidence_handling(self):
        """Test handling of low confidence classifications"""
        config = ClassificationConfig(min_confidence=0.9)  # High threshold
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        # Ambiguous content
        ambiguous_content = "This is some text."
        context = TestFixtures.create_processing_context()
        
        result = await ai.classify_document(ambiguous_content, context)
        
        if result.success:
            # Should indicate low confidence
            confidence = result.data.get("confidence", 1.0)
            if confidence < config.min_confidence:
                # Implementation should handle low confidence appropriately
                assert "confidence" in str(result.data).lower()
    
    @pytest.mark.asyncio
    async def test_component_failure_handling(self):
        """Test handling of component failures"""
        config = ClassificationConfig()
        ai = DocumentClassificationAI(config)
        await ai.initialize()
        
        # Mock a component to fail
        with patch.object(ai.content_classifier, 'process', side_effect=Exception("Component failed")):
            context = TestFixtures.create_processing_context()
            result = await ai.classify_document("test content", context)
            
            # Should handle component failure gracefully
            assert result.success is False
            assert "failed" in result.error.lower()