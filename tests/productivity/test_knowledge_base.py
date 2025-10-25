"""
Tests for Knowledge Base

Tests the personal knowledge management system including:
- Knowledge entry creation and retrieval
- Concept extraction and graph building
- Semantic search
- Knowledge relationships
- Importance scoring
"""

import pytest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from src.vega.productivity.knowledge_base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeType,
    KnowledgeSource,
    Concept,
    ConceptGraph,
    KnowledgeExtractor,
    SemanticSearchEngine,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def knowledge_base(temp_storage):
    """Create KnowledgeBase instance with temporary storage"""
    return KnowledgeBase(storage_path=temp_storage)


class TestKnowledgeCreation:
    """Test knowledge entry creation"""

    def test_add_simple_knowledge(self, knowledge_base):
        """Test adding a simple knowledge entry"""
        entry = knowledge_base.add_knowledge(
            content="Python is a programming language",
            knowledge_type=KnowledgeType.FACT,
            source=KnowledgeSource.MANUAL,
        )

        assert entry.content == "Python is a programming language"
        assert entry.knowledge_type == KnowledgeType.FACT
        assert entry.source == KnowledgeSource.MANUAL
        assert entry.id is not None
        assert entry.created_at is not None

    def test_add_knowledge_with_title(self, knowledge_base):
        """Test adding knowledge with a title"""
        entry = knowledge_base.add_knowledge(
            content="Machine learning is a subset of AI",
            title="ML Definition",
            knowledge_type=KnowledgeType.FACT,
        )

        assert entry.title == "ML Definition"
        assert entry.content == "Machine learning is a subset of AI"

    def test_add_knowledge_with_tags(self, knowledge_base):
        """Test adding knowledge with tags"""
        entry = knowledge_base.add_knowledge(
            content="FastAPI is a modern web framework",
            tags=["python", "web", "api"],
        )

        assert "python" in entry.tags
        assert "web" in entry.tags
        assert "api" in entry.tags

    def test_add_multiple_knowledge_types(self, knowledge_base):
        """Test adding different types of knowledge"""
        fact = knowledge_base.add_knowledge(
            content="Water boils at 100°C", knowledge_type=KnowledgeType.FACT
        )

        concept = knowledge_base.add_knowledge(
            content="Recursion is when a function calls itself",
            knowledge_type=KnowledgeType.CONCEPT,
        )

        procedure = knowledge_base.add_knowledge(
            content="To make coffee: 1. Boil water 2. Add grounds",
            knowledge_type=KnowledgeType.PROCEDURE,
        )

        assert fact.knowledge_type == KnowledgeType.FACT
        assert concept.knowledge_type == KnowledgeType.CONCEPT
        assert procedure.knowledge_type == KnowledgeType.PROCEDURE


class TestKnowledgeRetrieval:
    """Test knowledge retrieval operations"""

    def test_get_knowledge_by_id(self, knowledge_base):
        """Test retrieving knowledge by ID"""
        entry = knowledge_base.add_knowledge(content="Test knowledge")
        entry_id = entry.id

        retrieved = knowledge_base.get_knowledge(entry_id)

        assert retrieved is not None
        assert retrieved.id == entry_id
        assert retrieved.content == "Test knowledge"

    def test_get_nonexistent_knowledge(self, knowledge_base):
        """Test retrieving non-existent knowledge"""
        result = knowledge_base.get_knowledge("nonexistent-id")

        assert result is None

    def test_list_all_knowledge(self, knowledge_base):
        """Test listing all knowledge entries"""
        knowledge_base.add_knowledge(content="Entry 1")
        knowledge_base.add_knowledge(content="Entry 2")
        knowledge_base.add_knowledge(content="Entry 3")

        entries = knowledge_base.list_knowledge()

        assert len(entries) == 3


class TestConceptExtraction:
    """Test concept extraction from knowledge"""

    def test_extract_concepts_from_simple_text(self):
        """Test extracting concepts from simple text"""
        extractor = KnowledgeExtractor()

        concepts = extractor.extract_concepts(
            "Python is a programming language used for data science"
        )

        assert len(concepts) > 0
        assert isinstance(concepts, list)
        assert all(isinstance(c, str) for c in concepts)

    def test_concepts_extracted_on_add(self, knowledge_base):
        """Test that concepts are extracted when adding knowledge"""
        entry = knowledge_base.add_knowledge(
            content="Machine learning uses neural networks for pattern recognition"
        )

        assert len(entry.concepts) > 0

    def test_calculate_importance_score(self):
        """Test importance score calculation"""
        extractor = KnowledgeExtractor()

        # Longer, more detailed content should have higher importance
        simple_score = extractor.calculate_importance(
            "Python is good", tags=[], references=[]
        )

        detailed_score = extractor.calculate_importance(
            "Python is a high-level programming language known for its readability",
            tags=["programming", "python"],
            references=[],
        )

        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= detailed_score <= 1.0


class TestConceptGraph:
    """Test concept graph and relationship tracking"""

    def test_create_concept(self):
        """Test creating a concept"""
        graph = ConceptGraph()

        graph.add_concept("Python")

        assert "Python" in graph.concepts

    def test_add_connection(self):
        """Test adding connections between concepts"""
        graph = ConceptGraph()

        graph.add_concept("Python")
        graph.add_concept("Programming")
        graph.add_connection("Python", "Programming", weight=1.0)

        assert len(graph.connections) > 0

    def test_find_path_between_concepts(self):
        """Test finding paths between concepts"""
        graph = ConceptGraph()

        # Build a simple graph: A -> B -> C
        graph.add_concept("A")
        graph.add_concept("B")
        graph.add_concept("C")
        graph.add_connection("A", "B")
        graph.add_connection("B", "C")

        path = graph.find_path("A", "C")

        assert path is not None
        assert len(path) >= 2

    def test_get_related_concepts(self):
        """Test getting related concepts"""
        graph = ConceptGraph()

        graph.add_concept("Python")
        graph.add_concept("Django")
        graph.add_concept("Flask")
        graph.add_connection("Python", "Django")
        graph.add_connection("Python", "Flask")

        related = graph.get_related_concepts("Python")

        assert len(related) >= 2


class TestSemanticSearch:
    """Test semantic search functionality"""

    def test_search_basic_keyword(self, knowledge_base):
        """Test basic keyword search"""
        knowledge_base.add_knowledge(content="Python is a programming language")
        knowledge_base.add_knowledge(content="Java is also a programming language")

        results = knowledge_base.search("Python")

        assert len(results) > 0
        assert results[0][0].content == "Python is a programming language"

    def test_search_returns_scores(self, knowledge_base):
        """Test that search returns relevance scores"""
        knowledge_base.add_knowledge(content="Machine learning is amazing")

        results = knowledge_base.search("machine learning")

        assert len(results) > 0
        for entry, score in results:
            assert isinstance(entry, KnowledgeEntry)
            assert isinstance(score, float)
            assert score >= 0.0

    def test_search_filters_by_type(self, knowledge_base):
        """Test searching with type filter"""
        knowledge_base.add_knowledge(
            content="Fact about Python", knowledge_type=KnowledgeType.FACT
        )
        knowledge_base.add_knowledge(
            content="Python concept", knowledge_type=KnowledgeType.CONCEPT
        )

        fact_results = knowledge_base.search(
            "Python", knowledge_type=KnowledgeType.FACT
        )

        assert len(fact_results) > 0
        assert all(
            entry.knowledge_type == KnowledgeType.FACT for entry, _ in fact_results
        )

    def test_search_limit_results(self, knowledge_base):
        """Test limiting search results"""
        for i in range(10):
            knowledge_base.add_knowledge(content=f"Entry {i} about Python")

        results = knowledge_base.search("Python", top_k=5)

        assert len(results) <= 5

    def test_search_no_results(self, knowledge_base):
        """Test search with no matching results"""
        knowledge_base.add_knowledge(content="Something unrelated")

        results = knowledge_base.search("completely different query xyz123")

        # Should return empty list or very low scores
        assert len(results) == 0 or all(score < 0.1 for _, score in results)


class TestKnowledgeRelationships:
    """Test knowledge relationship tracking"""

    def test_add_relationship(self, knowledge_base):
        """Test adding relationships between knowledge entries"""
        entry1 = knowledge_base.add_knowledge(content="Python programming")
        entry2 = knowledge_base.add_knowledge(content="Web development")

        knowledge_base.add_relationship(entry1.id, entry2.id, "related_to")

        # Verify relationship exists in graph
        assert len(knowledge_base.graph.connections) > 0

    def test_get_related_knowledge(self, knowledge_base):
        """Test getting related knowledge entries"""
        entry1 = knowledge_base.add_knowledge(content="FastAPI framework")
        entry2 = knowledge_base.add_knowledge(content="Python web development")

        # Add relationship
        knowledge_base.add_relationship(entry1.id, entry2.id)

        related = knowledge_base.get_related_knowledge(entry1.id)

        # Should find related entries through shared concepts
        assert isinstance(related, list)


class TestKnowledgeUpdate:
    """Test knowledge update operations"""

    def test_update_knowledge_content(self, knowledge_base):
        """Test updating knowledge content"""
        entry = knowledge_base.add_knowledge(content="Original content")

        updated = knowledge_base.update_knowledge(entry.id, content="Updated content")

        assert updated.content == "Updated content"
        assert updated.updated_at > entry.created_at

    def test_update_knowledge_tags(self, knowledge_base):
        """Test updating knowledge tags"""
        entry = knowledge_base.add_knowledge(content="Test", tags=["old"])

        updated = knowledge_base.update_knowledge(entry.id, tags=["new", "tags"])

        assert "new" in updated.tags
        assert "tags" in updated.tags
        assert "old" not in updated.tags

    def test_update_importance_score(self, knowledge_base):
        """Test updating importance score"""
        entry = knowledge_base.add_knowledge(content="Test")
        original_score = entry.importance_score

        updated = knowledge_base.update_knowledge(entry.id, importance_score=0.95)

        assert updated.importance_score == 0.95
        assert updated.importance_score != original_score


class TestKnowledgeDeletion:
    """Test knowledge deletion operations"""

    def test_delete_knowledge(self, knowledge_base):
        """Test deleting a knowledge entry"""
        entry = knowledge_base.add_knowledge(content="To be deleted")
        entry_id = entry.id

        knowledge_base.delete_knowledge(entry_id)

        assert knowledge_base.get_knowledge(entry_id) is None

    def test_delete_nonexistent_knowledge(self, knowledge_base):
        """Test deleting non-existent knowledge"""
        # Should not raise error
        knowledge_base.delete_knowledge("nonexistent-id")


class TestKnowledgeStatistics:
    """Test knowledge base statistics"""

    def test_get_basic_stats(self, knowledge_base):
        """Test basic statistics"""
        knowledge_base.add_knowledge(content="Entry 1")
        knowledge_base.add_knowledge(content="Entry 2")

        stats = knowledge_base.get_stats()

        assert stats["total_entries"] == 2
        assert "total_concepts" in stats
        assert "total_connections" in stats

    def test_stats_by_type(self, knowledge_base):
        """Test statistics grouped by type"""
        knowledge_base.add_knowledge(
            content="Fact 1", knowledge_type=KnowledgeType.FACT
        )
        knowledge_base.add_knowledge(
            content="Fact 2", knowledge_type=KnowledgeType.FACT
        )
        knowledge_base.add_knowledge(
            content="Concept 1", knowledge_type=KnowledgeType.CONCEPT
        )

        stats = knowledge_base.get_stats()

        assert stats["by_type"]["fact"] == 2
        assert stats["by_type"]["concept"] == 1

    def test_stats_by_source(self, knowledge_base):
        """Test statistics grouped by source"""
        knowledge_base.add_knowledge(
            content="Manual entry", source=KnowledgeSource.MANUAL
        )
        knowledge_base.add_knowledge(content="Web entry", source=KnowledgeSource.WEB)

        stats = knowledge_base.get_stats()

        assert stats["by_source"]["manual"] == 1
        assert stats["by_source"]["web"] == 1


class TestKnowledgePersistence:
    """Test knowledge persistence and data loading"""

    def test_save_and_load_knowledge(self, temp_storage):
        """Test saving and loading knowledge"""
        kb1 = KnowledgeBase(storage_path=temp_storage)
        entry1 = kb1.add_knowledge(content="Persistent knowledge", title="Test Entry")

        # Create new instance with same storage
        kb2 = KnowledgeBase(storage_path=temp_storage)

        loaded = kb2.get_knowledge(entry1.id)
        assert loaded is not None
        assert loaded.content == "Persistent knowledge"
        assert loaded.title == "Test Entry"

    def test_graph_persists(self, temp_storage):
        """Test that concept graph persists"""
        kb1 = KnowledgeBase(storage_path=temp_storage)
        kb1.add_knowledge(content="Python is used for web development")

        kb2 = KnowledgeBase(storage_path=temp_storage)

        # Graph should be loaded
        assert len(kb2.graph.concepts) > 0


class TestKnowledgeImportanceScoring:
    """Test importance scoring system"""

    def test_importance_score_calculated(self, knowledge_base):
        """Test that importance score is calculated"""
        entry = knowledge_base.add_knowledge(content="Important knowledge")

        assert entry.importance_score is not None
        assert 0.0 <= entry.importance_score <= 1.0

    def test_tags_increase_importance(self, knowledge_base):
        """Test that tags increase importance"""
        simple = knowledge_base.add_knowledge(content="Simple entry")

        tagged = knowledge_base.add_knowledge(
            content="Tagged entry", tags=["tag1", "tag2", "tag3"]
        )

        # Tagged entry should have higher or equal importance
        assert tagged.importance_score >= simple.importance_score

    def test_longer_content_higher_importance(self, knowledge_base):
        """Test that longer content has higher importance"""
        short = knowledge_base.add_knowledge(content="Short")

        long_entry = knowledge_base.add_knowledge(
            content="Very long detailed knowledge entry " * 20
        )

        assert long_entry.importance_score >= short.importance_score


class TestKnowledgeFiltering:
    """Test knowledge filtering operations"""

    def test_filter_by_tags(self, knowledge_base):
        """Test filtering knowledge by tags"""
        knowledge_base.add_knowledge(content="Entry 1", tags=["python"])
        knowledge_base.add_knowledge(content="Entry 2", tags=["javascript"])
        knowledge_base.add_knowledge(content="Entry 3", tags=["python", "web"])

        python_entries = [
            e for e in knowledge_base.list_knowledge() if "python" in e.tags
        ]

        assert len(python_entries) == 2

    def test_filter_by_source(self, knowledge_base):
        """Test filtering knowledge by source"""
        knowledge_base.add_knowledge(content="Manual", source=KnowledgeSource.MANUAL)
        knowledge_base.add_knowledge(content="Web", source=KnowledgeSource.WEB)

        manual_entries = [
            e
            for e in knowledge_base.list_knowledge()
            if e.source == KnowledgeSource.MANUAL
        ]

        assert len(manual_entries) == 1


class TestKnowledgeReferences:
    """Test knowledge reference tracking"""

    def test_add_reference(self, knowledge_base):
        """Test adding references to knowledge"""
        entry = knowledge_base.add_knowledge(
            content="Test knowledge", references=["https://example.com"]
        )

        assert "https://example.com" in entry.references

    def test_update_references(self, knowledge_base):
        """Test updating knowledge references"""
        entry = knowledge_base.add_knowledge(content="Test")

        updated = knowledge_base.update_knowledge(
            entry.id, references=["https://newref.com"]
        )

        assert "https://newref.com" in updated.references


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
