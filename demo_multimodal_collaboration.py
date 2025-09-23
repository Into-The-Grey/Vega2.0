#!/usr/bin/env python3
"""
Multi-Modal Search with Real-Time Collaboration Demo
====================================================

Demonstration of Vega2.0's integrated multi-modal search and real-time collaboration
capabilities for shared content discovery and team collaboration.

Features demonstrated:
- Cross-modal content search (text, image, audio, video, documents)
- Real-time collaborative search sessions
- Shared discovery and annotation systems
- Team-based content curation and filtering
- Live search result sharing and synchronization
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
import json
import time
import uuid
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search mode types"""

    INDIVIDUAL = "individual"
    COLLABORATIVE = "collaborative"
    TEAM_FILTERED = "team_filtered"
    SHARED_SESSION = "shared_session"


class ContentType(Enum):
    """Content types for multi-modal search"""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    MIXED = "mixed"


@dataclass
class SearchResult:
    """Search result with multi-modal content"""

    id: str
    title: str
    content_type: ContentType
    content_path: str
    description: str
    relevance_score: float
    metadata: Dict[str, Any]
    embedding: np.ndarray
    tags: List[str] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    discovered_by: Optional[str] = None
    shared_with: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CollaborativeSession:
    """Real-time collaborative search session"""

    session_id: str
    name: str
    participants: Set[str]
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    shared_results: List[SearchResult] = field(default_factory=list)
    annotations: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    filters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


@dataclass
class UserProfile:
    """User profile for collaboration features"""

    user_id: str
    username: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    favorite_content: List[str] = field(default_factory=list)
    collaboration_settings: Dict[str, Any] = field(default_factory=dict)


class MockMultiModalSearchEngine:
    """Mock multi-modal search engine for demo purposes"""

    def __init__(self):
        self.content_database = self._initialize_content_database()
        self.search_index = self._build_search_index()

    def _initialize_content_database(self) -> List[SearchResult]:
        """Initialize mock content database with diverse multi-modal content"""

        content_items = [
            # Text content
            SearchResult(
                id="text_001",
                title="Advanced Machine Learning Techniques",
                content_type=ContentType.TEXT,
                content_path="/content/texts/ml_techniques.txt",
                description="Comprehensive guide covering neural networks, deep learning, and ensemble methods",
                relevance_score=0.0,  # Will be calculated
                metadata={
                    "author": "Dr. AI Expert",
                    "category": "technology",
                    "difficulty": "advanced",
                },
                embedding=self._generate_embedding(
                    "machine learning neural networks deep learning AI"
                ),
                tags=["AI", "machine learning", "neural networks", "technology"],
            ),
            SearchResult(
                id="text_002",
                title="Climate Change Research Findings",
                content_type=ContentType.TEXT,
                content_path="/content/texts/climate_research.txt",
                description="Latest research on global warming impacts and mitigation strategies",
                relevance_score=0.0,
                metadata={
                    "source": "Environmental Journal",
                    "category": "environment",
                    "year": 2024,
                },
                embedding=self._generate_embedding(
                    "climate change global warming environment research"
                ),
                tags=["climate", "environment", "research", "sustainability"],
            ),
            # Image content
            SearchResult(
                id="img_001",
                title="Neural Network Architecture Diagram",
                content_type=ContentType.IMAGE,
                content_path="/content/images/neural_network_diagram.png",
                description="Detailed visualization of deep neural network architecture with layer specifications",
                relevance_score=0.0,
                metadata={
                    "format": "PNG",
                    "dimensions": "1920x1080",
                    "category": "technology",
                },
                embedding=self._generate_embedding(
                    "neural network architecture diagram visualization AI"
                ),
                tags=["neural networks", "diagram", "AI", "visualization"],
            ),
            SearchResult(
                id="img_002",
                title="Mountain Ecosystem Photography",
                content_type=ContentType.IMAGE,
                content_path="/content/images/mountain_ecosystem.jpg",
                description="Stunning landscape photography showcasing alpine biodiversity and climate effects",
                relevance_score=0.0,
                metadata={
                    "photographer": "Nature Lens",
                    "location": "Rocky Mountains",
                    "category": "environment",
                },
                embedding=self._generate_embedding(
                    "mountain landscape nature environment ecosystem photography"
                ),
                tags=["nature", "mountains", "ecosystem", "photography"],
            ),
            # Audio content
            SearchResult(
                id="audio_001",
                title="AI Ethics Podcast Discussion",
                content_type=ContentType.AUDIO,
                content_path="/content/audio/ai_ethics_podcast.mp3",
                description="Expert panel discussion on ethical implications of artificial intelligence",
                relevance_score=0.0,
                metadata={
                    "duration": "45:30",
                    "format": "MP3",
                    "speakers": 3,
                    "category": "technology",
                },
                embedding=self._generate_embedding(
                    "artificial intelligence ethics discussion podcast technology"
                ),
                tags=["AI", "ethics", "podcast", "discussion", "technology"],
            ),
            SearchResult(
                id="audio_002",
                title="Nature Sounds - Forest Ambience",
                content_type=ContentType.AUDIO,
                content_path="/content/audio/forest_ambience.wav",
                description="High-quality forest soundscape with birds, wind, and wildlife",
                relevance_score=0.0,
                metadata={
                    "duration": "30:00",
                    "format": "WAV",
                    "location": "Pacific Northwest",
                    "category": "nature",
                },
                embedding=self._generate_embedding(
                    "forest nature sounds birds wildlife ambience environment"
                ),
                tags=["nature", "forest", "sounds", "ambience", "wildlife"],
            ),
            # Video content
            SearchResult(
                id="video_001",
                title="Machine Learning Tutorial Series",
                content_type=ContentType.VIDEO,
                content_path="/content/videos/ml_tutorial_series.mp4",
                description="Complete video course on machine learning fundamentals and applications",
                relevance_score=0.0,
                metadata={
                    "duration": "3:45:20",
                    "resolution": "4K",
                    "instructor": "ML Academy",
                    "category": "education",
                },
                embedding=self._generate_embedding(
                    "machine learning tutorial education course video training"
                ),
                tags=["machine learning", "tutorial", "education", "video", "training"],
            ),
            SearchResult(
                id="video_002",
                title="Wildlife Documentary - Arctic Changes",
                content_type=ContentType.VIDEO,
                content_path="/content/videos/arctic_wildlife_doc.mp4",
                description="Documentary exploring how climate change affects Arctic wildlife populations",
                relevance_score=0.0,
                metadata={
                    "duration": "58:15",
                    "production": "Nature Films",
                    "year": 2024,
                    "category": "environment",
                },
                embedding=self._generate_embedding(
                    "arctic wildlife climate change documentary environment animals"
                ),
                tags=["wildlife", "arctic", "climate change", "documentary", "animals"],
            ),
            # Document content
            SearchResult(
                id="doc_001",
                title="AI Implementation Strategy Guide",
                content_type=ContentType.DOCUMENT,
                content_path="/content/documents/ai_strategy_guide.pdf",
                description="Comprehensive business guide for implementing AI solutions in enterprise environments",
                relevance_score=0.0,
                metadata={
                    "pages": 127,
                    "format": "PDF",
                    "publisher": "Tech Strategy Press",
                    "category": "business",
                },
                embedding=self._generate_embedding(
                    "artificial intelligence business implementation strategy enterprise guide"
                ),
                tags=["AI", "business", "strategy", "implementation", "enterprise"],
            ),
            SearchResult(
                id="doc_002",
                title="Environmental Impact Assessment Report",
                content_type=ContentType.DOCUMENT,
                content_path="/content/documents/environmental_impact_report.docx",
                description="Detailed assessment of industrial activities' environmental impact and recommendations",
                relevance_score=0.0,
                metadata={
                    "pages": 89,
                    "format": "DOCX",
                    "agency": "Environmental Protection",
                    "category": "environment",
                },
                embedding=self._generate_embedding(
                    "environmental impact assessment industrial pollution report analysis"
                ),
                tags=["environment", "impact", "assessment", "industrial", "pollution"],
            ),
        ]

        return content_items

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate consistent embedding for text content"""
        hash_val = hash(text) % 10000
        np.random.seed(hash_val)
        embedding = np.random.normal(0, 1, 512).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    def _build_search_index(self) -> Dict[str, Any]:
        """Build search index for fast retrieval"""
        return {
            "embeddings": np.array([item.embedding for item in self.content_database]),
            "metadata_index": {item.id: item for item in self.content_database},
            "tag_index": self._build_tag_index(),
            "content_type_index": self._build_content_type_index(),
        }

    def _build_tag_index(self) -> Dict[str, List[str]]:
        """Build tag-based index"""
        tag_index = {}
        for item in self.content_database:
            for tag in item.tags:
                if tag not in tag_index:
                    tag_index[tag] = []
                tag_index[tag].append(item.id)
        return tag_index

    def _build_content_type_index(self) -> Dict[ContentType, List[str]]:
        """Build content type index"""
        type_index = {}
        for item in self.content_database:
            if item.content_type not in type_index:
                type_index[item.content_type] = []
            type_index[item.content_type].append(item.id)
        return type_index

    async def search(
        self,
        query: str,
        content_types: Optional[List[ContentType]] = None,
        tags: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.1,
    ) -> List[SearchResult]:
        """Perform multi-modal search"""

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Calculate similarities
        similarities = np.dot(self.search_index["embeddings"], query_embedding)

        # Apply filters
        candidate_items = list(self.content_database)

        if content_types:
            candidate_items = [
                item for item in candidate_items if item.content_type in content_types
            ]

        if tags:
            candidate_items = [
                item
                for item in candidate_items
                if any(tag in item.tags for tag in tags)
            ]

        # Score and rank results
        results = []
        for item in candidate_items:
            item_idx = next(
                i
                for i, db_item in enumerate(self.content_database)
                if db_item.id == item.id
            )
            similarity = similarities[item_idx]

            if similarity >= threshold:
                # Create result copy with updated score
                result = SearchResult(
                    id=item.id,
                    title=item.title,
                    content_type=item.content_type,
                    content_path=item.content_path,
                    description=item.description,
                    relevance_score=float(similarity),
                    metadata=item.metadata.copy(),
                    embedding=item.embedding.copy(),
                    tags=item.tags.copy(),
                    annotations=item.annotations.copy(),
                    discovered_by=item.discovered_by,
                    shared_with=item.shared_with.copy(),
                    timestamp=item.timestamp,
                )
                results.append(result)

        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results[:top_k]


class CollaborationManager:
    """Manages real-time collaboration features"""

    def __init__(self):
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.users: Dict[str, UserProfile] = {}
        self.shared_workspaces: Dict[str, Dict[str, Any]] = {}

    def create_user(self, username: str) -> UserProfile:
        """Create a new user profile"""
        user_id = str(uuid.uuid4())
        user = UserProfile(
            user_id=user_id,
            username=username,
            collaboration_settings={
                "auto_share_discoveries": True,
                "receive_notifications": True,
                "privacy_level": "team",
            },
        )
        self.users[user_id] = user
        return user

    def create_collaborative_session(
        self, name: str, creator_id: str
    ) -> CollaborativeSession:
        """Create a new collaborative search session"""
        session_id = str(uuid.uuid4())
        session = CollaborativeSession(
            session_id=session_id, name=name, participants={creator_id}
        )
        self.active_sessions[session_id] = session
        return session

    def join_session(self, session_id: str, user_id: str) -> bool:
        """Join an existing collaborative session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].participants.add(user_id)
            self.active_sessions[session_id].last_activity = datetime.now()
            return True
        return False

    def share_search_results(
        self, session_id: str, user_id: str, results: List[SearchResult], query: str
    ) -> bool:
        """Share search results with session participants"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Add to search history
        search_entry = {
            "user_id": user_id,
            "query": query,
            "results_count": len(results),
            "timestamp": datetime.now(),
            "result_ids": [r.id for r in results],
        }
        session.search_history.append(search_entry)

        # Add unique results to shared results
        existing_ids = {r.id for r in session.shared_results}
        for result in results:
            if result.id not in existing_ids:
                result.discovered_by = user_id
                result.shared_with = session.participants.copy()
                session.shared_results.append(result)

        session.last_activity = datetime.now()
        return True

    def add_annotation(
        self,
        session_id: str,
        user_id: str,
        content_id: str,
        annotation: str,
        annotation_type: str = "comment",
    ) -> bool:
        """Add annotation to shared content"""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if content_id not in session.annotations:
            session.annotations[content_id] = []

        annotation_data = {
            "user_id": user_id,
            "text": annotation,
            "type": annotation_type,
            "timestamp": datetime.now(),
        }

        session.annotations[content_id].append(annotation_data)
        session.last_activity = datetime.now()

        return True

    def get_session_analytics(self, session_id: str) -> Dict[str, Any]:
        """Get analytics for a collaborative session"""
        if session_id not in self.active_sessions:
            return {}

        session = self.active_sessions[session_id]

        # Calculate analytics
        total_searches = len(session.search_history)
        unique_content_discovered = len(session.shared_results)
        total_annotations = sum(
            len(annotations) for annotations in session.annotations.values()
        )

        # Participant activity
        user_activity = {}
        for search in session.search_history:
            user_id = search["user_id"]
            if user_id not in user_activity:
                user_activity[user_id] = {"searches": 0, "discoveries": 0}
            user_activity[user_id]["searches"] += 1

        for result in session.shared_results:
            if result.discovered_by:
                if result.discovered_by not in user_activity:
                    user_activity[result.discovered_by] = {
                        "searches": 0,
                        "discoveries": 0,
                    }
                user_activity[result.discovered_by]["discoveries"] += 1

        # Content type distribution
        content_type_dist = {}
        for result in session.shared_results:
            content_type = result.content_type.value
            content_type_dist[content_type] = content_type_dist.get(content_type, 0) + 1

        return {
            "session_duration": (
                session.last_activity - session.created_at
            ).total_seconds(),
            "total_participants": len(session.participants),
            "total_searches": total_searches,
            "unique_content_discovered": unique_content_discovered,
            "total_annotations": total_annotations,
            "user_activity": user_activity,
            "content_type_distribution": content_type_dist,
            "most_relevant_content": sorted(
                session.shared_results, key=lambda x: x.relevance_score, reverse=True
            )[:3],
        }


async def demo_individual_search():
    """Demonstrate individual multi-modal search"""
    print("🔍 Individual Multi-Modal Search Demo")
    print("=" * 40)

    search_engine = MockMultiModalSearchEngine()

    # Test different search queries
    search_queries = [
        {
            "query": "artificial intelligence machine learning",
            "content_types": None,
            "description": "General AI/ML search",
        },
        {
            "query": "climate change environment",
            "content_types": [ContentType.TEXT, ContentType.VIDEO],
            "description": "Environmental content (text & video only)",
        },
        {
            "query": "neural networks visualization",
            "content_types": [ContentType.IMAGE],
            "description": "Visual content search",
        },
    ]

    for search_config in search_queries:
        print(f"\n🔎 {search_config['description']}")
        print(f"   Query: '{search_config['query']}'")

        results = await search_engine.search(
            query=search_config["query"],
            content_types=search_config["content_types"],
            top_k=5,
        )

        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"      {i}. {result.title} ({result.content_type.value})")
            print(
                f"         Score: {result.relevance_score:.3f} | Tags: {', '.join(result.tags[:3])}"
            )


async def demo_collaborative_search():
    """Demonstrate collaborative search session"""
    print("\n🤝 Collaborative Search Session Demo")
    print("=" * 40)

    search_engine = MockMultiModalSearchEngine()
    collaboration_manager = CollaborationManager()

    # Create users
    alice = collaboration_manager.create_user("Alice")
    bob = collaboration_manager.create_user("Bob")
    charlie = collaboration_manager.create_user("Charlie")

    print(f"   👥 Created users: {alice.username}, {bob.username}, {charlie.username}")

    # Create collaborative session
    session = collaboration_manager.create_collaborative_session(
        "AI Research Project", alice.user_id
    )
    print(f"   🆕 Created session: '{session.name}' by {alice.username}")

    # Users join session
    collaboration_manager.join_session(session.session_id, bob.user_id)
    collaboration_manager.join_session(session.session_id, charlie.user_id)
    print(f"   ✅ {bob.username} and {charlie.username} joined the session")

    # Simulate collaborative search activities
    search_activities = [
        (
            alice.user_id,
            "machine learning algorithms",
            "Alice searches for ML algorithms",
        ),
        (
            bob.user_id,
            "neural network architectures",
            "Bob looks for network architectures",
        ),
        (charlie.user_id, "AI ethics and fairness", "Charlie explores AI ethics"),
        (alice.user_id, "deep learning tutorials", "Alice finds learning resources"),
        (
            bob.user_id,
            "computer vision applications",
            "Bob discovers vision applications",
        ),
    ]

    print(f"\n   🔍 Collaborative search activities:")

    for user_id, query, description in search_activities:
        user = next(u for u in [alice, bob, charlie] if u.user_id == user_id)

        # Perform search
        results = await search_engine.search(query, top_k=3)

        # Share results with session
        collaboration_manager.share_search_results(
            session.session_id, user_id, results, query
        )

        print(f"      • {description}: {len(results)} results shared")

        # Add some annotations
        if results:
            collaboration_manager.add_annotation(
                session.session_id,
                user_id,
                results[0].id,
                f"This looks relevant for our project - {user.username}",
                "comment",
            )

    # Get session analytics
    analytics = collaboration_manager.get_session_analytics(session.session_id)

    print(f"\n   📊 Session Analytics:")
    print(f"      • Total searches: {analytics['total_searches']}")
    print(
        f"      • Unique content discovered: {analytics['unique_content_discovered']}"
    )
    print(f"      • Total annotations: {analytics['total_annotations']}")
    print(f"      • Session duration: {analytics['session_duration']:.1f} seconds")

    print(f"   📈 User Activity:")
    for user_id, activity in analytics["user_activity"].items():
        user = next(u for u in [alice, bob, charlie] if u.user_id == user_id)
        print(
            f"      • {user.username}: {activity['searches']} searches, {activity['discoveries']} discoveries"
        )

    print(f"   📁 Content Type Distribution:")
    for content_type, count in analytics["content_type_distribution"].items():
        print(f"      • {content_type}: {count} items")


async def demo_real_time_features():
    """Demonstrate real-time collaboration features"""
    print("\n⚡ Real-Time Collaboration Features Demo")
    print("=" * 45)

    search_engine = MockMultiModalSearchEngine()
    collaboration_manager = CollaborationManager()

    # Create users and session
    team_lead = collaboration_manager.create_user("TeamLead")
    researcher1 = collaboration_manager.create_user("Researcher1")
    researcher2 = collaboration_manager.create_user("Researcher2")

    session = collaboration_manager.create_collaborative_session(
        "Live Research Session", team_lead.user_id
    )
    collaboration_manager.join_session(session.session_id, researcher1.user_id)
    collaboration_manager.join_session(session.session_id, researcher2.user_id)

    print(f"   🔴 Live session '{session.name}' with 3 participants")

    # Simulate real-time activities
    print(f"\n   📡 Real-time activity simulation:")

    # TeamLead searches for content
    results1 = await search_engine.search(
        "artificial intelligence applications", top_k=2
    )
    collaboration_manager.share_search_results(
        session.session_id, team_lead.user_id, results1, "AI applications"
    )
    print(f"      • TeamLead shared {len(results1)} AI application results")

    # Researcher1 adds annotations
    if results1:
        collaboration_manager.add_annotation(
            session.session_id,
            researcher1.user_id,
            results1[0].id,
            "This aligns with our project goals",
            "approval",
        )
        print(
            f"      • Researcher1 annotated content: 'This aligns with our project goals'"
        )

    # Researcher2 discovers related content
    results2 = await search_engine.search("machine learning business strategy", top_k=2)
    collaboration_manager.share_search_results(
        session.session_id, researcher2.user_id, results2, "ML business strategy"
    )
    print(f"      • Researcher2 discovered {len(results2)} business strategy resources")

    # TeamLead reviews and annotates
    if results2:
        collaboration_manager.add_annotation(
            session.session_id,
            team_lead.user_id,
            results2[0].id,
            "Excellent find! Let's prioritize this",
            "priority",
        )
        print(
            f"      • TeamLead marked content as priority: 'Excellent find! Let's prioritize this'"
        )

    # Show live session state
    current_session = collaboration_manager.active_sessions[session.session_id]
    print(f"\n   📊 Live Session State:")
    print(f"      • Active participants: {len(current_session.participants)}")
    print(f"      • Shared content items: {len(current_session.shared_results)}")
    print(f"      • Total search activities: {len(current_session.search_history)}")
    print(f"      • Annotated items: {len(current_session.annotations)}")

    # Show recent annotations
    print(f"   💬 Recent Annotations:")
    for content_id, annotations in current_session.annotations.items():
        content = next(
            (r for r in current_session.shared_results if r.id == content_id), None
        )
        if content:
            print(f"      • {content.title[:40]}...")
            for annotation in annotations[-2:]:  # Show last 2 annotations
                user = next(
                    u
                    for u in [team_lead, researcher1, researcher2]
                    if u.user_id == annotation["user_id"]
                )
                print(f"        - {user.username}: {annotation['text']}")


async def demo_content_curation():
    """Demonstrate team-based content curation"""
    print("\n📚 Team-Based Content Curation Demo")
    print("=" * 40)

    search_engine = MockMultiModalSearchEngine()
    collaboration_manager = CollaborationManager()

    # Create curation team
    curator = collaboration_manager.create_user("ContentCurator")
    expert1 = collaboration_manager.create_user("DomainExpert1")
    expert2 = collaboration_manager.create_user("DomainExpert2")

    session = collaboration_manager.create_collaborative_session(
        "Content Curation Hub", curator.user_id
    )
    collaboration_manager.join_session(session.session_id, expert1.user_id)
    collaboration_manager.join_session(session.session_id, expert2.user_id)

    print(f"   👥 Curation team assembled: Curator + 2 Domain Experts")

    # Multi-phase curation process
    curation_phases = [
        {
            "phase": "Discovery",
            "activities": [
                (curator.user_id, "comprehensive AI research materials"),
                (expert1.user_id, "technical machine learning resources"),
                (expert2.user_id, "AI ethics and policy documents"),
            ],
        },
        {
            "phase": "Evaluation",
            "activities": [
                ("annotation", "Quality assessment and relevance scoring"),
                ("annotation", "Technical accuracy verification"),
                ("annotation", "Bias and fairness evaluation"),
            ],
        },
        {
            "phase": "Curation",
            "activities": [
                ("curation", "Content organization and categorization"),
                ("curation", "Priority ranking and recommendations"),
                ("curation", "Final collection assembly"),
            ],
        },
    ]

    all_discovered_content = []

    for phase in curation_phases:
        print(f"\n   📋 {phase['phase']} Phase:")

        if phase["phase"] == "Discovery":
            for user_id, query in phase["activities"]:
                user = next(
                    u for u in [curator, expert1, expert2] if u.user_id == user_id
                )
                results = await search_engine.search(query, top_k=4)
                collaboration_manager.share_search_results(
                    session.session_id, user_id, results, query
                )
                all_discovered_content.extend(results)
                print(f"      • {user.username}: {len(results)} items discovered")

        elif phase["phase"] == "Evaluation":
            # Simulate evaluation annotations
            for i, (activity_type, description) in enumerate(phase["activities"]):
                evaluator = [curator, expert1, expert2][i]
                if all_discovered_content:
                    sample_content = all_discovered_content[
                        i % len(all_discovered_content)
                    ]
                    collaboration_manager.add_annotation(
                        session.session_id,
                        evaluator.user_id,
                        sample_content.id,
                        f"{description} - {evaluator.username}",
                        "evaluation",
                    )
                print(f"      • {description} completed")

        elif phase["phase"] == "Curation":
            for activity_type, description in phase["activities"]:
                print(f"      • {description} in progress")

    # Final curation results
    final_analytics = collaboration_manager.get_session_analytics(session.session_id)

    print(f"\n   ✅ Curation Results:")
    print(
        f"      • Total content evaluated: {final_analytics['unique_content_discovered']} items"
    )
    print(
        f"      • Quality annotations: {final_analytics['total_annotations']} assessments"
    )
    print(
        f"      • Team collaboration score: {len(final_analytics['user_activity'])}/3 active participants"
    )

    # Show top curated content
    if final_analytics["most_relevant_content"]:
        print(f"   🌟 Top Curated Content:")
        for i, content in enumerate(final_analytics["most_relevant_content"], 1):
            print(f"      {i}. {content.title} (Score: {content.relevance_score:.3f})")
            print(
                f"         Type: {content.content_type.value} | Tags: {', '.join(content.tags[:3])}"
            )


async def demo_performance_metrics():
    """Demonstrate performance metrics for collaborative search"""
    print("\n📈 Performance Metrics Demo")
    print("=" * 30)

    search_engine = MockMultiModalSearchEngine()
    collaboration_manager = CollaborationManager()

    # Create performance test session
    users = [collaboration_manager.create_user(f"User{i}") for i in range(5)]
    session = collaboration_manager.create_collaborative_session(
        "Performance Test", users[0].user_id
    )

    for user in users[1:]:
        collaboration_manager.join_session(session.session_id, user.user_id)

    print(f"   🚀 Performance test with {len(users)} concurrent users")

    # Measure search performance
    search_queries = [
        "artificial intelligence",
        "machine learning applications",
        "neural network architectures",
        "data science methodologies",
        "computer vision systems",
    ]

    start_time = time.time()
    total_results = 0

    # Simulate concurrent search activities
    for i, query in enumerate(search_queries):
        user = users[i % len(users)]
        results = await search_engine.search(query, top_k=5)
        collaboration_manager.share_search_results(
            session.session_id, user.user_id, results, query
        )
        total_results += len(results)

    search_time = time.time() - start_time

    # Measure annotation performance
    annotation_start = time.time()

    current_session = collaboration_manager.active_sessions[session.session_id]
    annotation_count = 0

    for result in current_session.shared_results[:10]:  # Annotate first 10 results
        user = users[annotation_count % len(users)]
        collaboration_manager.add_annotation(
            session.session_id,
            user.user_id,
            result.id,
            f"Performance test annotation {annotation_count + 1}",
            "test",
        )
        annotation_count += 1

    annotation_time = time.time() - annotation_start

    # Calculate performance metrics
    print(f"\n   ⚡ Performance Results:")
    print(f"      • Search operations: {len(search_queries)} queries")
    print(f"      • Total search time: {search_time:.3f}s")
    print(
        f"      • Average search time: {search_time/len(search_queries):.3f}s per query"
    )
    print(
        f"      • Search throughput: {len(search_queries)/search_time:.1f} queries/second"
    )
    print(f"      • Total results retrieved: {total_results}")

    print(f"   💬 Annotation Performance:")
    print(f"      • Annotations created: {annotation_count}")
    print(f"      • Annotation time: {annotation_time:.3f}s")
    print(
        f"      • Annotation rate: {annotation_count/annotation_time:.1f} annotations/second"
    )

    # Session efficiency metrics
    final_analytics = collaboration_manager.get_session_analytics(session.session_id)
    efficiency_score = (
        final_analytics["unique_content_discovered"] / final_analytics["total_searches"]
    ) * 100

    print(f"   📊 Collaboration Efficiency:")
    print(f"      • Unique content discovery rate: {efficiency_score:.1f}%")
    print(
        f"      • User engagement: {len(final_analytics['user_activity'])}/{len(users)} active"
    )
    print(
        f"      • Content diversity: {len(final_analytics['content_type_distribution'])} types"
    )


async def main():
    """Main demo function"""
    print("🌟 Vega2.0 Multi-Modal Search with Real-Time Collaboration Demo")
    print("=" * 80)
    print("Cross-Modal Search | Team Collaboration | Shared Content Discovery")
    print(f"Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Run all collaboration and search demos
        await demo_individual_search()
        await demo_collaborative_search()
        await demo_real_time_features()
        await demo_content_curation()
        await demo_performance_metrics()

        print("\n" + "=" * 80)
        print("✨ Multi-Modal Search with Real-Time Collaboration Demo Completed!")
        print("🔍 Cross-modal content discovery across text, image, audio, video")
        print("🤝 Real-time collaborative search sessions with live sharing")
        print("💬 Interactive annotation and content evaluation systems")
        print("📚 Team-based content curation and quality assessment")
        print("⚡ High-performance concurrent search and collaboration")
        print("📈 Comprehensive analytics and engagement metrics")
        print("🌐 Seamless integration with existing Vega2.0 infrastructure")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Multi-modal collaboration demo execution failed")


if __name__ == "__main__":
    asyncio.run(main())
