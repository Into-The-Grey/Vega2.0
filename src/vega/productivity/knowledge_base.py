"""
Vega 2.0 Personal Knowledge Base System

This module provides AI-powered knowledge management with:
- Automated knowledge extraction from documents and conversations
- Concept graph construction with entity and relationship extraction
- Semantic search across personal knowledge
- Intelligent concept linking and discovery
- Knowledge categorization and organization
- Temporal knowledge tracking (evolution over time)

The system builds a personal knowledge graph that grows with use,
enabling intelligent information retrieval and insight discovery.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
import json
import uuid
from collections import defaultdict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge entries"""

    FACT = "fact"  # Factual information
    CONCEPT = "concept"  # Abstract concept or idea
    PROCESS = "process"  # How-to or procedure
    INSIGHT = "insight"  # Personal insight or conclusion
    REFERENCE = "reference"  # External reference or citation
    EXPERIENCE = "experience"  # Personal experience
    DEFINITION = "definition"  # Definition or explanation
    EXAMPLE = "example"  # Example or case study


class KnowledgeSource(Enum):
    """Source of knowledge"""

    DOCUMENT = "document"
    CONVERSATION = "conversation"
    WEB = "web"
    BOOK = "book"
    VIDEO = "video"
    MANUAL = "manual"  # Manually entered
    IMPORTED = "imported"


@dataclass
class KnowledgeEntry:
    """Represents a single piece of knowledge"""

    id: str
    content: str  # The knowledge content
    title: Optional[str]  # Optional title/summary
    knowledge_type: KnowledgeType
    source: KnowledgeSource
    created_at: datetime
    updated_at: datetime
    tags: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)  # Extracted concepts
    entities: List[str] = field(default_factory=list)  # Named entities
    related_ids: List[str] = field(default_factory=list)  # Related knowledge IDs
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None  # Semantic embedding
    importance_score: float = 0.5  # 0.0 to 1.0
    access_count: int = 0  # How often accessed
    last_accessed: Optional[datetime] = None

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "id": self.id,
            "content": self.content,
            "title": self.title,
            "knowledge_type": self.knowledge_type.value,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "concepts": self.concepts,
            "entities": self.entities,
            "related_ids": self.related_ids,
            "source_metadata": self.source_metadata,
            "importance_score": self.importance_score,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
        }

        if include_embedding and self.embedding is not None:
            data["embedding"] = self.embedding.tolist()

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        """Create KnowledgeEntry from dictionary"""
        entry = cls(
            id=data["id"],
            content=data["content"],
            title=data.get("title"),
            knowledge_type=KnowledgeType(data["knowledge_type"]),
            source=KnowledgeSource(data["source"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []),
            concepts=data.get("concepts", []),
            entities=data.get("entities", []),
            related_ids=data.get("related_ids", []),
            source_metadata=data.get("source_metadata", {}),
            importance_score=data.get("importance_score", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
        )

        if "embedding" in data:
            entry.embedding = np.array(data["embedding"])

        return entry


@dataclass
class Concept:
    """Represents a concept in the knowledge graph"""

    name: str
    description: Optional[str]
    category: Optional[str]
    related_concepts: Set[str] = field(default_factory=set)
    knowledge_entries: Set[str] = field(default_factory=set)  # Entry IDs
    first_seen: datetime = field(default_factory=datetime.now)
    frequency: int = 0


class KnowledgeExtractor:
    """Extracts knowledge from text using NLP techniques"""

    def __init__(self):
        # Simple keyword-based extraction (can be enhanced with spaCy/transformers)
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }

    def extract_concepts(self, text: str) -> List[str]:
        """Extract concepts from text"""
        # Simple implementation: extract capitalized phrases and important terms
        words = text.split()
        concepts = []

        # Find capitalized phrases (potential concepts/entities)
        i = 0
        while i < len(words):
            if words[i] and words[i][0].isupper():
                phrase = [words[i]]
                j = i + 1
                while j < len(words) and words[j] and words[j][0].isupper():
                    phrase.append(words[j])
                    j += 1
                if len(phrase) > 0:
                    concept = " ".join(phrase)
                    if len(concept) > 2:  # Filter very short terms
                        concepts.append(concept)
                i = j
            else:
                i += 1

        # TODO: Enhance with spaCy NER or transformer-based extraction
        return list(set(concepts))[:20]  # Limit to top 20

    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        # Simple implementation - can be enhanced with spaCy
        return self.extract_concepts(text)  # For now, same as concepts

    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """Extract key phrases from text"""
        # Simple implementation: extract noun phrases
        # TODO: Use TextRank or RAKE for better phrase extraction
        sentences = text.split(".")
        phrases = []

        for sentence in sentences:
            words = [
                w.strip().lower()
                for w in sentence.split()
                if w.strip().lower() not in self.stop_words
            ]
            if len(words) >= 2:
                # Create 2-3 word phrases
                for i in range(len(words) - 1):
                    phrase = " ".join(words[i : i + 2])
                    phrases.append(phrase)
                    if i < len(words) - 2:
                        phrase = " ".join(words[i : i + 3])
                        phrases.append(phrase)

        # Count frequency and return top phrases
        phrase_freq = defaultdict(int)
        for phrase in phrases:
            phrase_freq[phrase] += 1

        top_phrases = sorted(phrase_freq.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in top_phrases[:max_phrases]]

    def calculate_importance(self, text: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for knowledge entry"""
        score = 0.5  # Base score

        # Factor 1: Length (longer = potentially more detailed)
        length = len(text)
        if length > 500:
            score += 0.1
        if length > 1000:
            score += 0.1

        # Factor 2: Source type
        source_scores = {
            KnowledgeSource.BOOK: 0.15,
            KnowledgeSource.WEB: 0.1,
            KnowledgeSource.DOCUMENT: 0.1,
            KnowledgeSource.CONVERSATION: 0.05,
            KnowledgeSource.MANUAL: 0.2,  # Manually entered = high importance
        }
        if "source" in metadata:
            source = KnowledgeSource(metadata["source"])
            score += source_scores.get(source, 0.0)

        # Factor 3: Has explicit title
        if metadata.get("title"):
            score += 0.05

        # Factor 4: Number of concepts (rich content)
        num_concepts = len(self.extract_concepts(text))
        score += min(num_concepts * 0.01, 0.1)

        return min(score, 1.0)


class ConceptGraph:
    """Graph of concepts and their relationships"""

    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.edges: Dict[Tuple[str, str], float] = {}  # (concept1, concept2) -> weight

    def add_concept(
        self,
        name: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> Concept:
        """Add or update a concept"""
        if name in self.concepts:
            concept = self.concepts[name]
            concept.frequency += 1
            if description and not concept.description:
                concept.description = description
            if category and not concept.category:
                concept.category = category
        else:
            concept = Concept(
                name=name,
                description=description,
                category=category,
                frequency=1,
            )
            self.concepts[name] = concept

        return concept

    def link_concepts(self, concept1: str, concept2: str, weight: float = 1.0) -> None:
        """Create a link between two concepts"""
        if concept1 not in self.concepts:
            self.add_concept(concept1)
        if concept2 not in self.concepts:
            self.add_concept(concept2)

        # Bidirectional edge
        edge = tuple(sorted([concept1, concept2]))
        self.edges[edge] = self.edges.get(edge, 0.0) + weight

        # Update concept relationships
        self.concepts[concept1].related_concepts.add(concept2)
        self.concepts[concept2].related_concepts.add(concept1)

    def get_related_concepts(
        self, concept: str, max_results: int = 10
    ) -> List[Tuple[str, float]]:
        """Get concepts related to the given concept"""
        if concept not in self.concepts:
            return []

        related = []
        for related_concept in self.concepts[concept].related_concepts:
            edge = tuple(sorted([concept, related_concept]))
            weight = self.edges.get(edge, 0.0)
            related.append((related_concept, weight))

        # Sort by weight
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_results]

    def find_path(
        self, start: str, end: str, max_depth: int = 3
    ) -> Optional[List[str]]:
        """Find shortest path between two concepts"""
        if start not in self.concepts or end not in self.concepts:
            return None

        # BFS for shortest path
        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_depth:
                continue

            if current == end:
                return path

            for related in self.concepts[current].related_concepts:
                if related not in visited:
                    visited.add(related)
                    queue.append((related, path + [related]))

        return None


class SemanticSearchEngine:
    """Semantic search over knowledge base using embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic search model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load semantic search model: {e}")
        else:
            logger.warning(
                "sentence-transformers not available, semantic search disabled"
            )

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if self.model is None:
            return None

        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None

    def search(
        self,
        query: str,
        entries: List[KnowledgeEntry],
        top_k: int = 10,
        threshold: float = 0.3,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Semantic search across knowledge entries"""
        if self.model is None:
            # Fall back to keyword search
            return self._keyword_search(query, entries, top_k)

        # Encode query
        query_embedding = self.encode(query)
        if query_embedding is None:
            return self._keyword_search(query, entries, top_k)

        # Ensure all entries have embeddings
        for entry in entries:
            if entry.embedding is None:
                entry.embedding = self.encode(entry.content)

        # Calculate cosine similarities
        results = []
        for entry in entries:
            if entry.embedding is not None:
                # Cosine similarity
                similarity = np.dot(query_embedding, entry.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(entry.embedding)
                )
                if similarity >= threshold:
                    results.append((entry, float(similarity)))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _keyword_search(
        self, query: str, entries: List[KnowledgeEntry], top_k: int
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Fallback keyword-based search"""
        query_words = set(query.lower().split())
        results = []

        for entry in entries:
            content_words = set(entry.content.lower().split())
            # Simple Jaccard similarity
            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            similarity = intersection / union if union > 0 else 0.0

            if similarity > 0:
                results.append((entry, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


class KnowledgeBase:
    """Main knowledge base system coordinating all components"""

    def __init__(self, storage_path: Optional[Path] = None):
        if isinstance(storage_path, str):
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = storage_path or Path.home() / ".vega" / "knowledge"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.entries: Dict[str, KnowledgeEntry] = {}
        self.concept_graph = ConceptGraph()
        self.extractor = KnowledgeExtractor()
        self.search_engine = SemanticSearchEngine()

        self._load_knowledge()

    def _load_knowledge(self) -> None:
        """Load knowledge from storage"""
        entries_file = self.storage_path / "entries.json"
        graph_file = self.storage_path / "concepts.json"

        # Load entries
        if entries_file.exists():
            try:
                with open(entries_file, "r") as f:
                    data = json.load(f)
                    self.entries = {
                        entry_id: KnowledgeEntry.from_dict(entry_data)
                        for entry_id, entry_data in data.items()
                    }
                logger.info(f"Loaded {len(self.entries)} knowledge entries")
            except Exception as e:
                logger.error(f"Error loading knowledge entries: {e}")

        # Load concept graph
        if graph_file.exists():
            try:
                with open(graph_file, "r") as f:
                    data = json.load(f)
                    # Reconstruct concept graph
                    for concept_name, concept_data in data.get("concepts", {}).items():
                        concept = Concept(
                            name=concept_name,
                            description=concept_data.get("description"),
                            category=concept_data.get("category"),
                            related_concepts=set(concept_data.get("related", [])),
                            knowledge_entries=set(concept_data.get("entries", [])),
                            first_seen=datetime.fromisoformat(
                                concept_data["first_seen"]
                            ),
                            frequency=concept_data.get("frequency", 0),
                        )
                        self.concept_graph.concepts[concept_name] = concept

                    # Reconstruct edges
                    for edge_key, weight in data.get("edges", {}).items():
                        c1, c2 = edge_key.split("||")
                        self.concept_graph.edges[(c1, c2)] = weight

                logger.info(f"Loaded {len(self.concept_graph.concepts)} concepts")
            except Exception as e:
                logger.error(f"Error loading concept graph: {e}")

    def _save_knowledge(self) -> None:
        """Save knowledge to storage"""
        entries_file = self.storage_path / "entries.json"
        graph_file = self.storage_path / "concepts.json"

        # Save entries (without embeddings in JSON)
        try:
            with open(entries_file, "w") as f:
                json.dump(
                    {
                        entry_id: entry.to_dict(include_embedding=False)
                        for entry_id, entry in self.entries.items()
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            logger.error(f"Error saving knowledge entries: {e}")

        # Save concept graph
        try:
            graph_data = {
                "concepts": {
                    name: {
                        "description": concept.description,
                        "category": concept.category,
                        "related": list(concept.related_concepts),
                        "entries": list(concept.knowledge_entries),
                        "first_seen": concept.first_seen.isoformat(),
                        "frequency": concept.frequency,
                    }
                    for name, concept in self.concept_graph.concepts.items()
                },
                "edges": {
                    f"{c1}||{c2}": weight
                    for (c1, c2), weight in self.concept_graph.edges.items()
                },
            }

            with open(graph_file, "w") as f:
                json.dump(graph_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving concept graph: {e}")

    def add_knowledge(
        self,
        content: str,
        knowledge_type: KnowledgeType = KnowledgeType.FACT,
        source: KnowledgeSource = KnowledgeSource.MANUAL,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source_metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeEntry:
        """Add knowledge to the base"""
        # Extract concepts and entities
        concepts = self.extractor.extract_concepts(content)
        entities = self.extractor.extract_entities(content)

        # Calculate importance
        metadata = source_metadata or {}
        metadata["source"] = source.value
        metadata["title"] = title
        importance = self.extractor.calculate_importance(content, metadata)

        # Create entry
        entry = KnowledgeEntry(
            id=str(uuid.uuid4()),
            content=content,
            title=title,
            knowledge_type=knowledge_type,
            source=source,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            tags=tags or [],
            concepts=concepts,
            entities=entities,
            source_metadata=metadata,
            importance_score=importance,
        )

        # Generate embedding
        entry.embedding = self.search_engine.encode(content)

        # Store entry
        self.entries[entry.id] = entry

        # Update concept graph
        for concept in concepts:
            concept_obj = self.concept_graph.add_concept(concept)
            concept_obj.knowledge_entries.add(entry.id)

        # Link related concepts (concepts appearing in same entry)
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1 :]:
                self.concept_graph.link_concepts(c1, c2, weight=0.5)

        self._save_knowledge()
        logger.info(f"Added knowledge entry: {title or entry.id}")

        return entry

    def search(
        self,
        query: str,
        knowledge_type: Optional[KnowledgeType] = None,
        top_k: int = 10,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Search knowledge base"""
        entries = list(self.entries.values())

        # Filter by type if specified
        if knowledge_type:
            entries = [e for e in entries if e.knowledge_type == knowledge_type]

        # Update access stats
        results = self.search_engine.search(query, entries, top_k=top_k)

        for entry, _ in results:
            entry.access_count += 1
            entry.last_accessed = datetime.now()

        self._save_knowledge()
        return results

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get knowledge entry by ID"""
        entry = self.entries.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            self._save_knowledge()
        return entry

    def get_related_knowledge(
        self, entry_id: str, max_results: int = 5
    ) -> List[KnowledgeEntry]:
        """Get knowledge related to an entry"""
        entry = self.entries.get(entry_id)
        if not entry:
            return []

        # Find entries sharing concepts
        related_ids = set()
        for concept in entry.concepts:
            if concept in self.concept_graph.concepts:
                related_ids.update(
                    self.concept_graph.concepts[concept].knowledge_entries
                )

        # Remove self
        related_ids.discard(entry_id)

        # Get entries and sort by importance
        related_entries = [
            self.entries[rid] for rid in related_ids if rid in self.entries
        ]
        related_entries.sort(key=lambda e: e.importance_score, reverse=True)

        return related_entries[:max_results]

    def list_knowledge(
        self,
        knowledge_type: Optional[KnowledgeType] = None,
        source: Optional[KnowledgeSource] = None,
    ) -> List[KnowledgeEntry]:
        """List all knowledge entries with optional filtering"""
        entries = list(self.entries.values())

        if knowledge_type:
            entries = [e for e in entries if e.knowledge_type == knowledge_type]
        if source:
            entries = [e for e in entries if e.source == source]

        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def update_knowledge(
        self,
        entry_id: str,
        content: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance_score: Optional[float] = None,
    ) -> Optional[KnowledgeEntry]:
        """Update a knowledge entry"""
        entry = self.entries.get(entry_id)
        if not entry:
            return None

        # Update fields
        if content is not None:
            entry.content = content
            # Re-extract concepts for new content
            entry.concepts = self.extractor.extract_concepts(content)
        if title is not None:
            entry.title = title
        if tags is not None:
            entry.tags = tags
        if importance_score is not None:
            entry.importance_score = importance_score

        entry.updated_at = datetime.now()
        self._save_knowledge()
        return entry

    def delete_knowledge(self, entry_id: str) -> bool:
        """Delete a knowledge entry"""
        if entry_id in self.entries:
            entry = self.entries[entry_id]

            # Remove from concept graph
            for concept in entry.concepts:
                if concept in self.concept_graph.concepts:
                    self.concept_graph.concepts[concept].knowledge_entries.discard(
                        entry_id
                    )

            del self.entries[entry_id]
            self._save_knowledge()
            return True
        return False

    def add_relationship(
        self,
        entry_id1: str,
        entry_id2: str,
        relationship_type: str = "related_to",
    ) -> bool:
        """Add a relationship between two knowledge entries"""
        entry1 = self.entries.get(entry_id1)
        entry2 = self.entries.get(entry_id2)

        if not entry1 or not entry2:
            return False

        # Add connection through shared concepts or create direct link
        # For now, we'll create implicit links through concept graph
        for concept1 in entry1.concepts:
            for concept2 in entry2.concepts:
                if concept1 != concept2:
                    self.concept_graph.link_concepts(concept1, concept2, weight=0.5)

        self._save_knowledge()
        return True

    def get_concept_info(self, concept_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a concept"""
        if concept_name not in self.concept_graph.concepts:
            return None

        concept = self.concept_graph.concepts[concept_name]
        related = self.concept_graph.get_related_concepts(concept_name)

        return {
            "name": concept.name,
            "description": concept.description,
            "category": concept.category,
            "frequency": concept.frequency,
            "first_seen": concept.first_seen.isoformat(),
            "related_concepts": [
                {"name": name, "weight": weight} for name, weight in related
            ],
            "entry_count": len(concept.knowledge_entries),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        return {
            "total_entries": len(self.entries),
            "total_concepts": len(self.concept_graph.concepts),
            "total_connections": len(self.concept_graph.edges),
            "by_type": {
                ktype.value: len(
                    [e for e in self.entries.values() if e.knowledge_type == ktype]
                )
                for ktype in KnowledgeType
            },
            "by_source": {
                source.value: len(
                    [e for e in self.entries.values() if e.source == source]
                )
                for source in KnowledgeSource
            },
        }


# Example usage demonstration
async def demo_knowledge_base():
    """Demonstrate knowledge base capabilities"""
    kb = KnowledgeBase()

    # Add some knowledge
    kb.add_knowledge(
        content="Federated learning is a machine learning technique that trains models across "
        "decentralized devices or servers holding local data samples, without exchanging them.",
        knowledge_type=KnowledgeType.DEFINITION,
        source=KnowledgeSource.WEB,
        title="Federated Learning Definition",
        tags=["machine-learning", "privacy", "distributed"],
    )

    kb.add_knowledge(
        content="PyTorch provides a powerful framework for federated learning implementations. "
        "Key components include torch.nn for model definitions and torch.distributed for communication.",
        knowledge_type=KnowledgeType.PROCESS,
        source=KnowledgeSource.DOCUMENT,
        title="PyTorch for Federated Learning",
        tags=["pytorch", "implementation"],
    )

    # Search knowledge
    print("\n=== Search Results for 'federated learning' ===")
    results = kb.search("federated learning", top_k=5)
    for entry, score in results:
        print(f"- {entry.title} (Score: {score:.3f})")

    # Get concept info
    print("\n=== Concept: Federated Learning ===")
    concept_info = kb.get_concept_info("Federated Learning")
    if concept_info:
        print(f"Frequency: {concept_info['frequency']}")
        print(
            f"Related concepts: {[c['name'] for c in concept_info['related_concepts']]}"
        )

    # Get statistics
    print("\n=== Knowledge Base Statistics ===")
    stats = kb.get_stats()
    print(f"Total entries: {stats['total_entries']}")
    print(f"Total concepts: {stats['total_concepts']}")
    print(f"Connections: {stats['total_connections']}")


if __name__ == "__main__":
    asyncio.run(demo_knowledge_base())
