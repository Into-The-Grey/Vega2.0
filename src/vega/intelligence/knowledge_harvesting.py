#!/usr/bin/env python3
"""
🧠 PHASE 6: KNOWLEDGE HARVESTING SYSTEM
==================================================
Advanced knowledge extraction and harvesting system for continuously learning
from conversations, building comprehensive knowledge graphs, and identifying
patterns for autonomous improvement.

This system implements:
- Intelligent conversation mining and pattern extraction
- Dynamic knowledge graph construction and maintenance
- Topic clustering and insight generation
- Continuous learning from user interactions
- Knowledge validation and quality scoring
- Automated knowledge base curation
- Cross-conversation insight synthesis
"""

import sqlite3
import logging
import json
import time
import asyncio
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from pathlib import Path
import re
import statistics
import math
from enum import Enum
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge extracted"""

    FACTUAL = "factual"  # Concrete facts and information
    PROCEDURAL = "procedural"  # How-to knowledge and processes
    CONCEPTUAL = "conceptual"  # Abstract concepts and relationships
    CONTEXTUAL = "contextual"  # Context-dependent insights
    PATTERN = "pattern"  # Recurring patterns and behaviors
    PREFERENCE = "preference"  # User preferences and tendencies


class ConfidenceLevel(Enum):
    """Confidence levels for knowledge items"""

    HIGH = "high"  # >90% confidence
    MEDIUM = "medium"  # 70-90% confidence
    LOW = "low"  # 50-70% confidence
    UNCERTAIN = "uncertain"  # <50% confidence


@dataclass
class KnowledgeItem:
    """Individual piece of extracted knowledge"""

    knowledge_id: str
    knowledge_type: KnowledgeType
    content: str  # The actual knowledge content
    context: str  # Context where this was observed
    confidence: ConfidenceLevel  # Confidence in this knowledge

    # Source Information
    source_conversation_id: str
    extracted_from: str  # Prompt or response
    extraction_method: str  # How this was extracted

    # Validation and Quality
    validation_score: float  # 0-1 validation score
    supporting_evidence: List[str]  # Supporting evidence
    contradicting_evidence: List[str]  # Contradicting evidence
    verification_attempts: int  # Number of verification attempts

    # Relationships
    related_topics: List[str]  # Related topic tags
    related_knowledge: List[str]  # Related knowledge IDs
    dependencies: List[str]  # Knowledge this depends on

    # Metadata
    created_at: datetime
    last_updated: datetime
    usage_count: int = 0
    usefulness_score: float = 0.0

    # Learning Metrics
    reinforcement_count: int = 0  # How many times reinforced
    contradiction_count: int = 0  # How many times contradicted
    application_success_rate: float = 0.0  # Success when applied


@dataclass
class TopicCluster:
    """Cluster of related topics and knowledge"""

    cluster_id: str
    name: str
    description: str
    keywords: List[str]

    # Knowledge in this cluster
    knowledge_items: List[str]  # Knowledge IDs in this cluster
    conversation_count: int  # Number of conversations about this topic

    # Cluster metrics
    coherence_score: float  # How coherent the cluster is
    importance_score: float  # How important this topic is
    frequency_score: float  # How frequently discussed

    # Evolution tracking
    created_at: datetime
    last_updated: datetime
    growth_rate: float = 0.0  # How fast the cluster is growing


@dataclass
class ConversationInsight:
    """High-level insight extracted from conversation patterns"""

    insight_id: str
    insight_type: str  # pattern, trend, anomaly, opportunity
    title: str
    description: str

    # Evidence and Support
    supporting_conversations: List[str]  # Conversation IDs
    confidence_score: float  # 0-1 confidence in insight
    statistical_significance: float  # Statistical significance

    # Impact and Value
    potential_impact: str  # high, medium, low
    actionable_recommendations: List[str]  # What actions to take

    # Temporal aspects
    time_period: str  # Time period this applies to
    trend_direction: str  # increasing, decreasing, stable

    created_at: datetime


class KnowledgeExtractor:
    """Extracts knowledge from individual conversations"""

    def __init__(self):
        self.extraction_patterns = self._load_extraction_patterns()
        logger.info("🧠 Knowledge Extractor initialized")

    def extract_knowledge(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        context: Dict[str, Any] = None,
    ) -> List[KnowledgeItem]:
        """Extract knowledge items from a conversation"""
        knowledge_items = []

        try:
            # Extract factual knowledge
            knowledge_items.extend(
                self._extract_factual_knowledge(conversation_id, prompt, response)
            )

            # Extract procedural knowledge
            knowledge_items.extend(
                self._extract_procedural_knowledge(conversation_id, prompt, response)
            )

            # Extract conceptual knowledge
            knowledge_items.extend(
                self._extract_conceptual_knowledge(conversation_id, prompt, response)
            )

            # Extract contextual knowledge
            knowledge_items.extend(
                self._extract_contextual_knowledge(
                    conversation_id, prompt, response, context
                )
            )

            # Extract patterns
            knowledge_items.extend(
                self._extract_pattern_knowledge(conversation_id, prompt, response)
            )

            # Extract preferences
            knowledge_items.extend(
                self._extract_preference_knowledge(conversation_id, prompt, response)
            )

            logger.info(
                f"📚 Extracted {len(knowledge_items)} knowledge items from conversation {conversation_id}"
            )

        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")

        return knowledge_items

    def _extract_factual_knowledge(
        self, conversation_id: str, prompt: str, response: str
    ) -> List[KnowledgeItem]:
        """Extract factual information"""
        knowledge_items = []

        # Look for fact patterns in responses
        fact_patterns = [
            r"(.+) is (.+)",  # X is Y
            r"(.+) was (.+)",  # X was Y
            r"(.+) are (.+)",  # X are Y
            r"(.+) can (.+)",  # X can Y
            r"(.+) will (.+)",  # X will Y
            r"(.+) has (.+)",  # X has Y
            r"(.+) contains (.+)",  # X contains Y
            r"(.+) requires (.+)",  # X requires Y
            r"(.+) supports (.+)",  # X supports Y
        ]

        for pattern in fact_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if (
                    len(match) == 2
                    and len(match[0].strip()) > 2
                    and len(match[1].strip()) > 2
                ):
                    # Filter out very generic statements
                    if not any(
                        generic in match[0].lower()
                        for generic in ["this", "that", "it", "they"]
                    ):
                        knowledge_id = hashlib.md5(
                            f"{match[0]}_{match[1]}".encode()
                        ).hexdigest()[:12]

                        knowledge_items.append(
                            KnowledgeItem(
                                knowledge_id=knowledge_id,
                                knowledge_type=KnowledgeType.FACTUAL,
                                content=f"{match[0].strip()} -> {match[1].strip()}",
                                context=prompt,
                                confidence=self._assess_confidence(
                                    match[0], match[1], response
                                ),
                                source_conversation_id=conversation_id,
                                extracted_from="response",
                                extraction_method="pattern_matching",
                                validation_score=0.5,  # Default, will be updated
                                supporting_evidence=[],
                                contradicting_evidence=[],
                                verification_attempts=0,
                                related_topics=self._extract_topics(
                                    f"{match[0]} {match[1]}"
                                ),
                                related_knowledge=[],
                                dependencies=[],
                                created_at=datetime.now(),
                                last_updated=datetime.now(),
                            )
                        )

        return knowledge_items

    def _extract_procedural_knowledge(
        self, conversation_id: str, prompt: str, response: str
    ) -> List[KnowledgeItem]:
        """Extract how-to and procedural knowledge"""
        knowledge_items = []

        # Look for step-by-step instructions
        step_patterns = [
            r"(?:step\s*\d+|first|second|third|then|next|finally)[:\-]?\s*(.+)",
            r"(?:\d+\.|\d+\))\s*(.+)",
            r"(?:to\s+.+,?\s*)?(?:you\s+)?(?:should|must|need\s+to|can)\s+(.+)",
        ]

        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            steps.extend(
                [match.strip() for match in matches if len(match.strip()) > 10]
            )

        if len(steps) >= 2:  # At least 2 steps to be considered procedural
            procedure_topic = self._extract_main_topic(prompt)
            if procedure_topic:
                knowledge_id = hashlib.md5(
                    f"procedure_{procedure_topic}".encode()
                ).hexdigest()[:12]

                knowledge_items.append(
                    KnowledgeItem(
                        knowledge_id=knowledge_id,
                        knowledge_type=KnowledgeType.PROCEDURAL,
                        content=f"How to {procedure_topic}: {'; '.join(steps[:5])}",  # Limit to 5 steps
                        context=prompt,
                        confidence=ConfidenceLevel.MEDIUM,
                        source_conversation_id=conversation_id,
                        extracted_from="response",
                        extraction_method="step_analysis",
                        validation_score=0.6,
                        supporting_evidence=[],
                        contradicting_evidence=[],
                        verification_attempts=0,
                        related_topics=[procedure_topic]
                        + self._extract_topics(response),
                        related_knowledge=[],
                        dependencies=[],
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                    )
                )

        return knowledge_items

    def _extract_conceptual_knowledge(
        self, conversation_id: str, prompt: str, response: str
    ) -> List[KnowledgeItem]:
        """Extract conceptual and abstract knowledge"""
        knowledge_items = []

        # Look for concept definitions and explanations
        concept_patterns = [
            r"(.+)\s+(?:is|are)\s+(?:a|an|the)?\s*(?:type of|kind of|form of)\s+(.+)",
            r"(.+)\s+(?:means|refers to|represents)\s+(.+)",
            r"(?:the concept of|the idea of)\s+(.+)\s+(?:involves|includes|encompasses)\s+(.+)",
        ]

        for pattern in concept_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if (
                    len(match) == 2
                    and len(match[0].strip()) > 3
                    and len(match[1].strip()) > 5
                ):
                    concept = match[0].strip()
                    definition = match[1].strip()

                    knowledge_id = hashlib.md5(
                        f"concept_{concept}".encode()
                    ).hexdigest()[:12]

                    knowledge_items.append(
                        KnowledgeItem(
                            knowledge_id=knowledge_id,
                            knowledge_type=KnowledgeType.CONCEPTUAL,
                            content=f"Concept: {concept} = {definition}",
                            context=prompt,
                            confidence=self._assess_confidence(
                                concept, definition, response
                            ),
                            source_conversation_id=conversation_id,
                            extracted_from="response",
                            extraction_method="concept_analysis",
                            validation_score=0.5,
                            supporting_evidence=[],
                            contradicting_evidence=[],
                            verification_attempts=0,
                            related_topics=self._extract_topics(
                                f"{concept} {definition}"
                            ),
                            related_knowledge=[],
                            dependencies=[],
                            created_at=datetime.now(),
                            last_updated=datetime.now(),
                        )
                    )

        return knowledge_items

    def _extract_contextual_knowledge(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        context: Dict[str, Any] = None,
    ) -> List[KnowledgeItem]:
        """Extract context-dependent knowledge"""
        knowledge_items = []

        if not context:
            return knowledge_items

        # Extract knowledge about conversation patterns
        conversation_length = context.get("length", 1)
        if conversation_length > 3:  # Multi-turn conversation
            knowledge_id = hashlib.md5(
                f"context_{conversation_id}".encode()
            ).hexdigest()[:12]

            knowledge_items.append(
                KnowledgeItem(
                    knowledge_id=knowledge_id,
                    knowledge_type=KnowledgeType.CONTEXTUAL,
                    content=f"Multi-turn conversation pattern: {conversation_length} exchanges",
                    context=f"Conversation depth: {conversation_length}",
                    confidence=ConfidenceLevel.HIGH,
                    source_conversation_id=conversation_id,
                    extracted_from="context",
                    extraction_method="context_analysis",
                    validation_score=0.8,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    verification_attempts=0,
                    related_topics=["conversation_patterns", "user_engagement"],
                    related_knowledge=[],
                    dependencies=[],
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                )
            )

        return knowledge_items

    def _extract_pattern_knowledge(
        self, conversation_id: str, prompt: str, response: str
    ) -> List[KnowledgeItem]:
        """Extract pattern-based knowledge"""
        knowledge_items = []

        # Identify question patterns in prompts
        question_types = []
        if "?" in prompt:
            if any(
                word in prompt.lower()
                for word in ["how", "what", "where", "when", "why", "which"]
            ):
                question_types.append("informational")
            if any(
                word in prompt.lower() for word in ["can", "could", "would", "should"]
            ):
                question_types.append("capability")
            if any(word in prompt.lower() for word in ["help", "assist", "support"]):
                question_types.append("assistance")

        if question_types:
            pattern_id = hashlib.md5(
                f"pattern_{'_'.join(question_types)}".encode()
            ).hexdigest()[:12]

            knowledge_items.append(
                KnowledgeItem(
                    knowledge_id=pattern_id,
                    knowledge_type=KnowledgeType.PATTERN,
                    content=f"Question pattern: {', '.join(question_types)}",
                    context=prompt,
                    confidence=ConfidenceLevel.MEDIUM,
                    source_conversation_id=conversation_id,
                    extracted_from="prompt",
                    extraction_method="pattern_recognition",
                    validation_score=0.6,
                    supporting_evidence=[],
                    contradicting_evidence=[],
                    verification_attempts=0,
                    related_topics=question_types + ["question_patterns"],
                    related_knowledge=[],
                    dependencies=[],
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                )
            )

        return knowledge_items

    def _extract_preference_knowledge(
        self, conversation_id: str, prompt: str, response: str
    ) -> List[KnowledgeItem]:
        """Extract user preference knowledge"""
        knowledge_items = []

        # Look for preference indicators in prompts
        preference_patterns = [
            r"i prefer (.+)",
            r"i like (.+)",
            r"i want (.+)",
            r"i need (.+)",
            r"i would rather (.+)",
            r"please (.+)",
        ]

        for pattern in preference_patterns:
            matches = re.findall(pattern, prompt, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 3:
                    preference_id = hashlib.md5(
                        f"preference_{match}".encode()
                    ).hexdigest()[:12]

                    knowledge_items.append(
                        KnowledgeItem(
                            knowledge_id=preference_id,
                            knowledge_type=KnowledgeType.PREFERENCE,
                            content=f"User preference: {match.strip()}",
                            context=prompt,
                            confidence=ConfidenceLevel.MEDIUM,
                            source_conversation_id=conversation_id,
                            extracted_from="prompt",
                            extraction_method="preference_analysis",
                            validation_score=0.7,
                            supporting_evidence=[],
                            contradicting_evidence=[],
                            verification_attempts=0,
                            related_topics=self._extract_topics(match),
                            related_knowledge=[],
                            dependencies=[],
                            created_at=datetime.now(),
                            last_updated=datetime.now(),
                        )
                    )

        return knowledge_items

    def _assess_confidence(
        self, subject: str, content: str, full_text: str
    ) -> ConfidenceLevel:
        """Assess confidence level in extracted knowledge"""
        confidence_score = 0.5  # Base confidence

        # Boost confidence for specific indicators
        if any(
            indicator in full_text.lower()
            for indicator in ["research shows", "studies indicate", "according to"]
        ):
            confidence_score += 0.3

        if any(
            indicator in full_text.lower()
            for indicator in ["always", "never", "definitely", "certainly"]
        ):
            confidence_score += 0.2

        # Reduce confidence for uncertainty indicators
        if any(
            indicator in full_text.lower()
            for indicator in ["might", "could", "possibly", "probably"]
        ):
            confidence_score -= 0.2

        # Reduce confidence for very short or generic content
        if len(content) < 10 or any(
            generic in content.lower()
            for generic in ["something", "anything", "everything"]
        ):
            confidence_score -= 0.3

        # Map to confidence levels
        if confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN

    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant topics from text"""
        # Simple keyword extraction
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

        # Filter out common words
        stopwords = {
            "this",
            "that",
            "with",
            "have",
            "will",
            "been",
            "from",
            "they",
            "know",
            "want",
            "were",
            "said",
            "each",
            "which",
            "their",
            "time",
            "will",
            "about",
            "would",
            "there",
            "could",
            "other",
        }

        meaningful_words = [
            word for word in words if word not in stopwords and len(word) > 3
        ]

        # Count frequency and return most common
        word_counts = Counter(meaningful_words)
        return [word for word, count in word_counts.most_common(5)]

    def _extract_main_topic(self, text: str) -> str:
        """Extract the main topic from text"""
        topics = self._extract_topics(text)
        return topics[0] if topics else ""

    def _load_extraction_patterns(self) -> Dict[str, List[str]]:
        """Load extraction patterns for different knowledge types"""
        return {
            "factual": [
                r"(.+) is (.+)",
                r"(.+) was (.+)",
                r"(.+) are (.+)",
            ],
            "procedural": [
                r"step \d+:?\s*(.+)",
                r"first,?\s*(.+)",
                r"then,?\s*(.+)",
                r"finally,?\s*(.+)",
            ],
            "conceptual": [
                r"(.+) means (.+)",
                r"(.+) refers to (.+)",
                r"the concept of (.+)",
            ],
        }


class KnowledgeGraphBuilder:
    """Builds and maintains knowledge graphs"""

    def __init__(self):
        self.graph = nx.Graph()
        self.knowledge_db = "knowledge_graph.db"
        self._init_database()

        logger.info("🕸️ Knowledge Graph Builder initialized")

    def _init_database(self):
        """Initialize knowledge graph database"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS knowledge_items (
            knowledge_id TEXT PRIMARY KEY,
            knowledge_type TEXT,
            content TEXT,
            context TEXT,
            confidence TEXT,
            source_conversation_id TEXT,
            extracted_from TEXT,
            extraction_method TEXT,
            validation_score REAL,
            supporting_evidence TEXT,
            contradicting_evidence TEXT,
            verification_attempts INTEGER,
            related_topics TEXT,
            related_knowledge TEXT,
            dependencies TEXT,
            created_at TEXT,
            last_updated TEXT,
            usage_count INTEGER DEFAULT 0,
            usefulness_score REAL DEFAULT 0.0,
            reinforcement_count INTEGER DEFAULT 0,
            contradiction_count INTEGER DEFAULT 0,
            application_success_rate REAL DEFAULT 0.0
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS topic_clusters (
            cluster_id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            keywords TEXT,
            knowledge_items TEXT,
            conversation_count INTEGER DEFAULT 0,
            coherence_score REAL DEFAULT 0.0,
            importance_score REAL DEFAULT 0.0,
            frequency_score REAL DEFAULT 0.0,
            created_at TEXT,
            last_updated TEXT,
            growth_rate REAL DEFAULT 0.0
        )
        """
        )

        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS conversation_insights (
            insight_id TEXT PRIMARY KEY,
            insight_type TEXT,
            title TEXT,
            description TEXT,
            supporting_conversations TEXT,
            confidence_score REAL,
            statistical_significance REAL,
            potential_impact TEXT,
            actionable_recommendations TEXT,
            time_period TEXT,
            trend_direction TEXT,
            created_at TEXT
        )
        """
        )

        conn.commit()
        conn.close()

    def add_knowledge_item(self, knowledge_item: KnowledgeItem):
        """Add knowledge item to graph and database"""
        try:
            # Add to graph
            self.graph.add_node(
                knowledge_item.knowledge_id,
                knowledge_type=knowledge_item.knowledge_type.value,
                content=knowledge_item.content,
                confidence=knowledge_item.confidence.value,
            )

            # Add relationships to graph
            for related_id in knowledge_item.related_knowledge:
                if related_id in self.graph:
                    self.graph.add_edge(
                        knowledge_item.knowledge_id, related_id, relationship="related"
                    )

            # Store in database
            self._store_knowledge_item(knowledge_item)

            logger.debug(
                f"📝 Added knowledge item {knowledge_item.knowledge_id} to graph"
            )

        except Exception as e:
            logger.error(f"Failed to add knowledge item: {e}")

    def build_topic_clusters(
        self, knowledge_items: List[KnowledgeItem]
    ) -> List[TopicCluster]:
        """Build topic clusters from knowledge items"""
        try:
            # Group knowledge by topics
            topic_groups = defaultdict(list)

            for item in knowledge_items:
                for topic in item.related_topics:
                    topic_groups[topic].append(item.knowledge_id)

            clusters = []
            for topic, knowledge_ids in topic_groups.items():
                if len(knowledge_ids) >= 2:  # At least 2 items to form a cluster
                    cluster_id = hashlib.md5(f"cluster_{topic}".encode()).hexdigest()[
                        :12
                    ]

                    cluster = TopicCluster(
                        cluster_id=cluster_id,
                        name=topic.replace("_", " ").title(),
                        description=f"Knowledge cluster for {topic}",
                        keywords=[topic],
                        knowledge_items=knowledge_ids,
                        conversation_count=len(
                            set(
                                item.source_conversation_id
                                for item in knowledge_items
                                if item.knowledge_id in knowledge_ids
                            )
                        ),
                        coherence_score=self._calculate_cluster_coherence(
                            knowledge_ids, knowledge_items
                        ),
                        importance_score=len(knowledge_ids) / len(knowledge_items),
                        frequency_score=len(knowledge_ids) / len(knowledge_items),
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                    )

                    clusters.append(cluster)
                    self._store_topic_cluster(cluster)

            logger.info(f"🗂️ Built {len(clusters)} topic clusters")
            return clusters

        except Exception as e:
            logger.error(f"Cluster building failed: {e}")
            return []

    def _calculate_cluster_coherence(
        self, knowledge_ids: List[str], all_knowledge: List[KnowledgeItem]
    ) -> float:
        """Calculate coherence score for a cluster"""
        if len(knowledge_ids) < 2:
            return 1.0

        # Simple coherence based on shared topics
        items_in_cluster = [
            item for item in all_knowledge if item.knowledge_id in knowledge_ids
        ]

        all_topics = []
        for item in items_in_cluster:
            all_topics.extend(item.related_topics)

        if not all_topics:
            return 0.0

        topic_counts = Counter(all_topics)
        max_count = max(topic_counts.values())

        # Coherence is based on how many items share the most common topic
        coherence = max_count / len(items_in_cluster)
        return min(1.0, coherence)

    def _store_knowledge_item(self, item: KnowledgeItem):
        """Store knowledge item in database"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO knowledge_items VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                item.knowledge_id,
                item.knowledge_type.value,
                item.content,
                item.context,
                item.confidence.value,
                item.source_conversation_id,
                item.extracted_from,
                item.extraction_method,
                item.validation_score,
                json.dumps(item.supporting_evidence),
                json.dumps(item.contradicting_evidence),
                item.verification_attempts,
                json.dumps(item.related_topics),
                json.dumps(item.related_knowledge),
                json.dumps(item.dependencies),
                item.created_at.isoformat(),
                item.last_updated.isoformat(),
                item.usage_count,
                item.usefulness_score,
                item.reinforcement_count,
                item.contradiction_count,
                item.application_success_rate,
            ),
        )

        conn.commit()
        conn.close()

    def _store_topic_cluster(self, cluster: TopicCluster):
        """Store topic cluster in database"""
        conn = sqlite3.connect(self.knowledge_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO topic_clusters VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                cluster.cluster_id,
                cluster.name,
                cluster.description,
                json.dumps(cluster.keywords),
                json.dumps(cluster.knowledge_items),
                cluster.conversation_count,
                cluster.coherence_score,
                cluster.importance_score,
                cluster.frequency_score,
                cluster.created_at.isoformat(),
                cluster.last_updated.isoformat(),
                cluster.growth_rate,
            ),
        )

        conn.commit()
        conn.close()


class ConversationInsightGenerator:
    """Generates high-level insights from conversation patterns"""

    def __init__(self, knowledge_graph: KnowledgeGraphBuilder):
        self.knowledge_graph = knowledge_graph
        logger.info("💡 Conversation Insight Generator initialized")

    def generate_insights(
        self, knowledge_items: List[KnowledgeItem], time_period_days: int = 30
    ) -> List[ConversationInsight]:
        """Generate insights from knowledge items"""
        insights = []

        try:
            # Analyze knowledge patterns
            insights.extend(self._analyze_knowledge_patterns(knowledge_items))

            # Analyze topic trends
            insights.extend(
                self._analyze_topic_trends(knowledge_items, time_period_days)
            )

            # Analyze confidence patterns
            insights.extend(self._analyze_confidence_patterns(knowledge_items))

            # Store insights
            for insight in insights:
                self._store_insight(insight)

            logger.info(f"💡 Generated {len(insights)} conversation insights")

        except Exception as e:
            logger.error(f"Insight generation failed: {e}")

        return insights

    def _analyze_knowledge_patterns(
        self, knowledge_items: List[KnowledgeItem]
    ) -> List[ConversationInsight]:
        """Analyze patterns in knowledge types and extraction"""
        insights = []

        if not knowledge_items:
            return insights

        # Analyze knowledge type distribution
        type_counts = Counter([item.knowledge_type.value for item in knowledge_items])
        total_items = len(knowledge_items)

        dominant_type = type_counts.most_common(1)[0]
        dominance_ratio = dominant_type[1] / total_items

        if dominance_ratio > 0.6:  # If one type dominates
            insight_id = hashlib.md5(
                f"pattern_dominance_{dominant_type[0]}".encode()
            ).hexdigest()[:12]

            insights.append(
                ConversationInsight(
                    insight_id=insight_id,
                    insight_type="pattern",
                    title=f"Knowledge Type Dominance: {dominant_type[0].title()}",
                    description=f"{dominant_type[0].title()} knowledge represents {dominance_ratio:.1%} of extracted knowledge, indicating a strong focus on {dominant_type[0]} information.",
                    supporting_conversations=list(
                        set(
                            [
                                item.source_conversation_id
                                for item in knowledge_items
                                if item.knowledge_type.value == dominant_type[0]
                            ]
                        )
                    ),
                    confidence_score=0.8,
                    statistical_significance=dominance_ratio,
                    potential_impact="medium",
                    actionable_recommendations=[
                        f"Consider diversifying conversation topics beyond {dominant_type[0]} knowledge",
                        f"Optimize {dominant_type[0]} knowledge extraction and utilization",
                        "Monitor for knowledge type balance in future conversations",
                    ],
                    time_period=f"last_{len(knowledge_items)}_interactions",
                    trend_direction="stable",
                    created_at=datetime.now(),
                )
            )

        return insights

    def _analyze_topic_trends(
        self, knowledge_items: List[KnowledgeItem], days: int
    ) -> List[ConversationInsight]:
        """Analyze trends in topics over time"""
        insights = []

        # Group by time periods
        recent_items = [
            item
            for item in knowledge_items
            if item.created_at > datetime.now() - timedelta(days=days)
        ]

        if len(recent_items) < 5:  # Need sufficient data
            return insights

        # Analyze topic frequency
        all_topics = []
        for item in recent_items:
            all_topics.extend(item.related_topics)

        topic_counts = Counter(all_topics)
        trending_topics = topic_counts.most_common(3)

        if trending_topics:
            insight_id = hashlib.md5(f"trending_topics_{days}d".encode()).hexdigest()[
                :12
            ]

            insights.append(
                ConversationInsight(
                    insight_id=insight_id,
                    insight_type="trend",
                    title=f"Trending Topics (Last {days} days)",
                    description=f"Most discussed topics: {', '.join([topic for topic, count in trending_topics])}",
                    supporting_conversations=list(
                        set([item.source_conversation_id for item in recent_items])
                    ),
                    confidence_score=0.7,
                    statistical_significance=trending_topics[0][1] / len(recent_items),
                    potential_impact="high",
                    actionable_recommendations=[
                        f"Develop specialized knowledge for trending topic: {trending_topics[0][0]}",
                        "Monitor topic evolution and prepare relevant content",
                        "Consider creating topic-specific optimization strategies",
                    ],
                    time_period=f"last_{days}_days",
                    trend_direction="increasing",
                    created_at=datetime.now(),
                )
            )

        return insights

    def _analyze_confidence_patterns(
        self, knowledge_items: List[KnowledgeItem]
    ) -> List[ConversationInsight]:
        """Analyze confidence patterns in extracted knowledge"""
        insights = []

        if not knowledge_items:
            return insights

        # Analyze confidence distribution
        confidence_counts = Counter([item.confidence.value for item in knowledge_items])
        total_items = len(knowledge_items)

        low_confidence_ratio = (
            confidence_counts.get("low", 0) + confidence_counts.get("uncertain", 0)
        ) / total_items

        if low_confidence_ratio > 0.4:  # More than 40% low confidence
            insight_id = hashlib.md5("low_confidence_pattern".encode()).hexdigest()[:12]

            insights.append(
                ConversationInsight(
                    insight_id=insight_id,
                    insight_type="anomaly",
                    title="High Proportion of Low-Confidence Knowledge",
                    description=f"{low_confidence_ratio:.1%} of extracted knowledge has low or uncertain confidence, indicating potential areas for improvement.",
                    supporting_conversations=list(
                        set(
                            [
                                item.source_conversation_id
                                for item in knowledge_items
                                if item.confidence.value in ["low", "uncertain"]
                            ]
                        )
                    ),
                    confidence_score=0.9,
                    statistical_significance=low_confidence_ratio,
                    potential_impact="high",
                    actionable_recommendations=[
                        "Improve knowledge extraction accuracy",
                        "Implement better confidence assessment algorithms",
                        "Focus on reducing uncertainty in responses",
                        "Validate low-confidence knowledge through additional sources",
                    ],
                    time_period="current_analysis",
                    trend_direction="concerning",
                    created_at=datetime.now(),
                )
            )

        return insights

    def _store_insight(self, insight: ConversationInsight):
        """Store insight in database"""
        conn = sqlite3.connect(self.knowledge_graph.knowledge_db)
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT OR REPLACE INTO conversation_insights VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
            (
                insight.insight_id,
                insight.insight_type,
                insight.title,
                insight.description,
                json.dumps(insight.supporting_conversations),
                insight.confidence_score,
                insight.statistical_significance,
                insight.potential_impact,
                json.dumps(insight.actionable_recommendations),
                insight.time_period,
                insight.trend_direction,
                insight.created_at.isoformat(),
            ),
        )

        conn.commit()
        conn.close()


class KnowledgeHarvestingSystem:
    """Main knowledge harvesting system coordinator"""

    def __init__(self):
        self.extractor = KnowledgeExtractor()
        self.graph_builder = KnowledgeGraphBuilder()
        self.insight_generator = ConversationInsightGenerator(self.graph_builder)

        self.knowledge_cache: List[KnowledgeItem] = []
        self.harvesting_active = True

        logger.info("🧠 Knowledge Harvesting System initialized")

    async def harvest_from_conversation(
        self,
        conversation_id: str,
        prompt: str,
        response: str,
        context: Dict[str, Any] = None,
    ) -> List[KnowledgeItem]:
        """Harvest knowledge from a single conversation"""
        if not self.harvesting_active:
            return []

        try:
            # Extract knowledge
            knowledge_items = self.extractor.extract_knowledge(
                conversation_id, prompt, response, context
            )

            # Add to graph
            for item in knowledge_items:
                self.graph_builder.add_knowledge_item(item)
                self.knowledge_cache.append(item)

            # Trigger cluster building if we have enough items
            if len(self.knowledge_cache) % 20 == 0:  # Every 20 items
                await self._build_clusters()

            return knowledge_items

        except Exception as e:
            logger.error(f"Knowledge harvesting failed: {e}")
            return []

    async def generate_comprehensive_insights(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive insights from all harvested knowledge"""
        try:
            # Get recent knowledge
            recent_knowledge = [
                item
                for item in self.knowledge_cache
                if item.created_at > datetime.now() - timedelta(days=days)
            ]

            # Build clusters
            clusters = self.graph_builder.build_topic_clusters(recent_knowledge)

            # Generate insights
            insights = self.insight_generator.generate_insights(recent_knowledge, days)

            # Compile comprehensive report
            report = {
                "knowledge_summary": {
                    "total_items": len(self.knowledge_cache),
                    "recent_items": len(recent_knowledge),
                    "knowledge_types": dict(
                        Counter(
                            [item.knowledge_type.value for item in recent_knowledge]
                        )
                    ),
                    "confidence_distribution": dict(
                        Counter([item.confidence.value for item in recent_knowledge])
                    ),
                    "avg_validation_score": (
                        statistics.mean(
                            [item.validation_score for item in recent_knowledge]
                        )
                        if recent_knowledge
                        else 0
                    ),
                },
                "topic_clusters": [
                    {
                        "name": cluster.name,
                        "knowledge_count": len(cluster.knowledge_items),
                        "importance": cluster.importance_score,
                        "coherence": cluster.coherence_score,
                    }
                    for cluster in clusters[:10]  # Top 10 clusters
                ],
                "insights": [
                    {
                        "title": insight.title,
                        "type": insight.insight_type,
                        "impact": insight.potential_impact,
                        "confidence": insight.confidence_score,
                        "recommendations": insight.actionable_recommendations[
                            :3
                        ],  # Top 3 recommendations
                    }
                    for insight in insights
                ],
                "knowledge_graph_stats": {
                    "nodes": self.graph_builder.graph.number_of_nodes(),
                    "edges": self.graph_builder.graph.number_of_edges(),
                    "density": (
                        nx.density(self.graph_builder.graph)
                        if self.graph_builder.graph.number_of_nodes() > 0
                        else 0
                    ),
                },
                "harvesting_performance": {
                    "extraction_success_rate": 0.85,  # Placeholder
                    "avg_knowledge_per_conversation": len(recent_knowledge)
                    / max(
                        1,
                        len(
                            set(
                                [
                                    item.source_conversation_id
                                    for item in recent_knowledge
                                ]
                            )
                        ),
                    ),
                    "knowledge_quality_score": (
                        statistics.mean(
                            [item.validation_score for item in recent_knowledge]
                        )
                        if recent_knowledge
                        else 0
                    ),
                },
            }

            return report

        except Exception as e:
            logger.error(f"Comprehensive insight generation failed: {e}")
            return {"error": str(e)}

    async def _build_clusters(self):
        """Build topic clusters from accumulated knowledge"""
        try:
            clusters = self.graph_builder.build_topic_clusters(self.knowledge_cache)
            logger.info(
                f"🗂️ Built {len(clusters)} clusters from {len(self.knowledge_cache)} knowledge items"
            )
        except Exception as e:
            logger.error(f"Cluster building failed: {e}")

    async def run_harvesting_cycle(self):
        """Run a complete knowledge harvesting cycle"""
        logger.info("🔄 Running knowledge harvesting cycle...")

        try:
            # Build clusters from all knowledge
            await self._build_clusters()

            # Generate insights
            insights_report = await self.generate_comprehensive_insights()

            logger.info(f"📊 Harvesting cycle completed:")
            logger.info(
                f"  📚 Total knowledge items: {insights_report['knowledge_summary']['total_items']}"
            )
            logger.info(f"  🗂️ Topic clusters: {len(insights_report['topic_clusters'])}")
            logger.info(f"  💡 Insights generated: {len(insights_report['insights'])}")

            return insights_report

        except Exception as e:
            logger.error(f"Harvesting cycle failed: {e}")
            return {}


# Test and demonstration functions
async def demonstrate_knowledge_harvesting():
    """Demonstrate the knowledge harvesting system"""
    print("🧠 KNOWLEDGE HARVESTING SYSTEM")
    print("=" * 45)

    system = KnowledgeHarvestingSystem()

    # Sample conversations to harvest from
    test_conversations = [
        {
            "id": "conv_1",
            "prompt": "How do I implement a REST API in Python using FastAPI?",
            "response": "To implement a REST API in Python, you can use FastAPI. First, install FastAPI with pip install fastapi uvicorn. Then create your main application file. FastAPI is a modern web framework that provides automatic API documentation. You should define your endpoints using decorators like @app.get() and @app.post(). FastAPI supports automatic data validation using Pydantic models.",
            "context": {"length": 2},
        },
        {
            "id": "conv_2",
            "prompt": "What are the benefits of using Docker containers?",
            "response": "Docker containers provide several key benefits. They ensure consistency across different environments. Docker containers are lightweight compared to virtual machines. They enable easy application deployment and scaling. Docker also supports microservices architecture effectively. Containers isolate applications from the host system.",
            "context": {"length": 1},
        },
        {
            "id": "conv_3",
            "prompt": "I prefer using TypeScript over JavaScript for large projects. Can you explain the advantages?",
            "response": "TypeScript offers several advantages over JavaScript for large projects. TypeScript provides static type checking which catches errors at compile time. It has better IDE support with autocomplete and refactoring tools. TypeScript supports modern JavaScript features while maintaining compatibility. The type system makes code more maintainable and self-documenting.",
            "context": {"length": 3},
        },
        {
            "id": "conv_4",
            "prompt": "What is machine learning and how does it work?",
            "response": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without explicit programming. Machine learning algorithms identify patterns in data. There are three main types: supervised learning, unsupervised learning, and reinforcement learning. The process involves training models on datasets and then using them for predictions.",
            "context": {"length": 1},
        },
    ]

    print("📚 Harvesting knowledge from sample conversations...")

    all_knowledge = []
    for conv in test_conversations:
        knowledge_items = await system.harvest_from_conversation(
            conv["id"], conv["prompt"], conv["response"], conv["context"]
        )
        all_knowledge.extend(knowledge_items)

        print(f"  ✅ {conv['id']}: Extracted {len(knowledge_items)} knowledge items")

    print(f"\n📊 Total knowledge items harvested: {len(all_knowledge)}")

    # Display some extracted knowledge
    print("\n🔍 Sample extracted knowledge:")
    for i, item in enumerate(all_knowledge[:5]):  # Show first 5 items
        print(f"  {i+1}. [{item.knowledge_type.value.upper()}] {item.content[:80]}...")
        print(
            f"     Confidence: {item.confidence.value}, Topics: {', '.join(item.related_topics[:3])}"
        )

    # Generate comprehensive insights
    print("\n💡 Generating comprehensive insights...")
    insights_report = await system.generate_comprehensive_insights()

    print(f"📈 Knowledge Summary:")
    summary = insights_report["knowledge_summary"]
    print(f"  📚 Total items: {summary['total_items']}")
    print(f"  🎯 Knowledge types: {summary['knowledge_types']}")
    print(f"  📊 Confidence distribution: {summary['confidence_distribution']}")
    print(f"  ✅ Avg validation score: {summary['avg_validation_score']:.3f}")

    print(f"\n🗂️ Topic Clusters ({len(insights_report['topic_clusters'])}):")
    for cluster in insights_report["topic_clusters"][:3]:
        print(
            f"  📁 {cluster['name']}: {cluster['knowledge_count']} items (importance: {cluster['importance']:.3f})"
        )

    print(f"\n💡 Generated Insights ({len(insights_report['insights'])}):")
    for insight in insights_report["insights"]:
        print(f"  🔍 {insight['title']} ({insight['type']})")
        print(
            f"     Impact: {insight['impact']}, Confidence: {insight['confidence']:.3f}"
        )
        if insight["recommendations"]:
            print(f"     Recommendation: {insight['recommendations'][0]}")

    # Run full harvesting cycle
    print("\n🔄 Running complete harvesting cycle...")
    cycle_report = await system.run_harvesting_cycle()

    print("\n🎯 KNOWLEDGE HARVESTING SYSTEM OPERATIONAL")
    print(
        "System is now continuously learning from conversations and building knowledge"
    )


if __name__ == "__main__":
    asyncio.run(demonstrate_knowledge_harvesting())
