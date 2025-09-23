#!/usr/bin/env python3
"""
Personal Multimodal Workspace Demo
==================================

Demonstration of Vega2.0's personal multimodal workspace capabilities
for individual content discovery, organization, and management.

Features demonstrated:
- Personal content search across multiple formats (text, image, audio, video)
- Personal workspace organization and management
- Content annotation and personal curation
- Search history and favorites management
- Individual productivity features
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from personal multimodal search."""
    
    content: str
    content_type: str  # 'text', 'image', 'audio', 'video'
    score: float
    source: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'content': self.content,
            'content_type': self.content_type,
            'score': self.score,
            'source': self.source,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class PersonalMultimodalSearch:
    """Personal multimodal search engine for individual workspace."""
    
    def __init__(self):
        self.search_history: List[SearchResult] = []
        self.content_database = self._initialize_personal_content()
    
    def _initialize_personal_content(self) -> List[SearchResult]:
        """Initialize personal content collection"""
        
        return [
            SearchResult(
                content="Advanced Machine Learning Techniques - Personal notes on neural networks and deep learning",
                content_type="text",
                score=0.95,
                source="personal_notes.txt",
                metadata={"category": "technology", "date_added": "2024-09-20"}
            ),
            SearchResult(
                content="Climate Research Findings - Personal collection of environmental studies",
                content_type="text", 
                score=0.88,
                source="climate_notes.txt",
                metadata={"category": "environment", "date_added": "2024-09-18"}
            ),
            SearchResult(
                content="Neural Network Architecture Diagram - Personal AI visualization",
                content_type="image",
                score=0.92,
                source="diagrams/neural_network.png",
                metadata={"category": "technology", "format": "PNG"}
            ),
            SearchResult(
                content="Mountain Ecosystem Photography - Personal nature collection",
                content_type="image",
                score=0.85,
                source="photos/mountain_ecosystem.jpg", 
                metadata={"category": "nature", "format": "JPEG"}
            ),
            SearchResult(
                content="AI Ethics Podcast - Personal educational content",
                content_type="audio",
                score=0.90,
                source="podcasts/ai_ethics.mp3",
                metadata={"category": "technology", "duration": "45:30"}
            ),
            SearchResult(
                content="Forest Ambience - Personal relaxation sounds",
                content_type="audio",
                score=0.80,
                source="sounds/forest_ambience.wav",
                metadata={"category": "nature", "duration": "30:00"}
            ),
            SearchResult(
                content="Machine Learning Tutorial - Personal learning videos",
                content_type="video",
                score=0.87,
                source="tutorials/ml_basics.mp4",
                metadata={"category": "education", "duration": "25:15"}
            ),
            SearchResult(
                content="Nature Documentary - Personal favorite films",
                content_type="video",
                score=0.83,
                source="videos/nature_doc.mp4",
                metadata={"category": "nature", "duration": "58:30"}
            )
        ]

    def search(self, query: str, content_types: Optional[List[str]] = None) -> List[SearchResult]:
        """Search personal content collection"""
        if content_types is None:
            content_types = ["text", "image", "audio", "video"]
        
        # Simple text matching for demo
        results = []
        query_lower = query.lower()
        
        for item in self.content_database:
            if item.content_type in content_types:
                if query_lower in item.content.lower():
                    results.append(item)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Add to search history
        self.search_history.extend(results)
        
        return results


class PersonalWorkspace:
    """Personal workspace for content management and organization"""
    
    def __init__(self):
        self.search_engine = PersonalMultimodalSearch()
        self.saved_items = []
        self.search_sessions = []
        self.annotations = {}
        
    def search_content(self, query: str, content_types: Optional[List[str]] = None) -> List[SearchResult]:
        """Search through personal content collection"""
        results = self.search_engine.search(query, content_types)
        
        # Log search session
        session = {
            'query': query,
            'timestamp': datetime.now(),
            'results_count': len(results),
            'content_types': content_types or ["all"]
        }
        self.search_sessions.append(session)
        
        return results
    
    def save_item(self, result: SearchResult, tags: Optional[List[str]] = None):
        """Save an item to personal collection"""
        saved_item = {
            'item': result,
            'saved_at': datetime.now(),
            'tags': tags or [],
            'personal_notes': ""
        }
        self.saved_items.append(saved_item)
    
    def add_annotation(self, result: SearchResult, annotation: str):
        """Add personal annotation to content"""
        item_key = f"{result.source}_{result.content_type}"
        if item_key not in self.annotations:
            self.annotations[item_key] = []
        
        self.annotations[item_key].append({
            'annotation': annotation,
            'timestamp': datetime.now()
        })
    
    def get_search_history(self) -> List[Dict]:
        """Get personal search history"""
        return self.search_sessions
    
    def get_saved_items(self) -> List[Dict]:
        """Get personally saved items"""
        return self.saved_items
    
    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of personal workspace content"""
        content_types = {}
        for item in self.search_engine.content_database:
            content_type = item.content_type
            if content_type not in content_types:
                content_types[content_type] = 0
            content_types[content_type] += 1
        
        return {
            'total_items': len(self.search_engine.content_database),
            'content_types': content_types,
            'search_sessions': len(self.search_sessions),
            'saved_items': len(self.saved_items),
            'annotations': len(self.annotations)
        }


def run_personal_workspace_demo():
    """Demonstrate personal workspace features"""
    print("🏠 Personal Multimodal Workspace Demo")
    print("=" * 50)
    
    # Initialize personal workspace
    workspace = PersonalWorkspace()
    
    # Show content summary
    summary = workspace.get_content_summary()
    print(f"\n📊 Personal Content Summary:")
    print(f"   Total items: {summary['total_items']}")
    print(f"   Content types: {summary['content_types']}")
    
    # Demo searches with different content types
    search_scenarios = [
        ("machine learning", ["text", "image"]),
        ("climate environment", None),
        ("forest sounds", ["audio"]),
        ("neural network", ["image", "video"]),
        ("podcast tutorial", ["audio", "video"])
    ]
    
    print(f"\n🔍 Personal Search Demonstrations:")
    for query, content_types in search_scenarios:
        print(f"\n   Query: '{query}'")
        if content_types:
            print(f"   Filtering by: {content_types}")
        
        results = workspace.search_content(query, content_types)
        
        print(f"   Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"     {i}. [{result.content_type.upper()}] {result.content[:50]}...")
            print(f"        Score: {result.score:.2f} | Source: {result.source}")
            
            # Save and annotate first result of each search
            if i == 1:
                workspace.save_item(result, tags=["demo", query.split()[0]])
                workspace.add_annotation(result, f"Found through '{query}' search - highly relevant")
                print(f"        💾 Saved to personal collection")
                print(f"        📝 Added personal annotation")
    
    # Show workspace statistics
    print(f"\n📈 Personal Workspace Statistics:")
    history = workspace.get_search_history()
    saved = workspace.get_saved_items()
    
    print(f"   Search sessions: {len(history)}")
    print(f"   Saved items: {len(saved)}")
    print(f"   Personal annotations: {len(workspace.annotations)}")
    
    # Show recent search history
    print(f"\n📋 Recent Search History:")
    for session in history[-3:]:  # Show last 3 searches
        print(f"   • '{session['query']}' → {session['results_count']} results")
        print(f"     {session['timestamp'].strftime('%H:%M:%S')} | Types: {session['content_types']}")
    
    # Show saved items
    print(f"\n💾 Saved Items:")
    for saved_item in saved:
        item = saved_item['item']
        print(f"   • [{item.content_type.upper()}] {item.content[:40]}...")
        print(f"     Tags: {saved_item['tags']} | Saved: {saved_item['saved_at'].strftime('%H:%M:%S')}")
    
    # Show personal annotations
    print(f"\n📝 Personal Annotations:")
    for item_key, annotations in workspace.annotations.items():
        print(f"   {item_key}:")
        for annotation in annotations:
            print(f"     - {annotation['annotation']} ({annotation['timestamp'].strftime('%H:%M:%S')})")
    
    print(f"\n✨ Personal Workspace Demo Complete!")
    print("   Your content is organized, searchable, and personally annotated")


if __name__ == "__main__":
    run_personal_workspace_demo()
