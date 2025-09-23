"""
Vector Database Demo Configuration
=================================

Configuration settings for the vector database integration demo.
Adjust these settings to test different scenarios and configurations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum

from src.vega.multimodal import VectorDBType, IndexType, DistanceMetric


class DemoScenario(Enum):
    """Available demo scenarios"""

    BASIC = "basic"
    PERFORMANCE = "performance"
    BATCH_OPERATIONS = "batch"
    CROSS_MODAL = "cross_modal"
    METADATA_FILTERING = "metadata"
    ALL = "all"


@dataclass
class DemoConfig:
    """Configuration for vector database demo"""

    # Demo scenario selection
    scenario: DemoScenario = DemoScenario.ALL

    # Dataset settings
    sample_size: int = 100
    large_dataset_size: int = 1000
    embedding_dimension: int = 512

    # FAISS configuration options
    faiss_configs: List[Dict[str, Any]] = None

    # Pinecone configuration
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    pinecone_index_name: str = "vega-demo"

    # Performance testing
    performance_iterations: int = 10
    batch_size: int = 50
    search_top_k: int = 5

    # Content generation
    content_categories: List[str] = None
    content_modalities: List[str] = None

    # Logging and output
    verbose_output: bool = True
    save_results: bool = False
    results_path: str = "./demo_results"

    def __post_init__(self):
        """Initialize default configurations"""

        if self.faiss_configs is None:
            self.faiss_configs = [
                {
                    "name": "FLAT Index (Exact Search)",
                    "type": VectorDBType.FAISS,
                    "index_type": IndexType.FLAT,
                    "distance_metric": DistanceMetric.COSINE,
                    "params": {},
                },
                {
                    "name": "IVF_FLAT Index (Fast Approximate)",
                    "type": VectorDBType.FAISS,
                    "index_type": IndexType.IVF_FLAT,
                    "distance_metric": DistanceMetric.COSINE,
                    "params": {"nlist": 20, "nprobe": 5},
                },
                {
                    "name": "IVF_PQ Index (Memory Efficient)",
                    "type": VectorDBType.FAISS,
                    "index_type": IndexType.IVF_PQ,
                    "distance_metric": DistanceMetric.COSINE,
                    "params": {"nlist": 20, "nprobe": 5, "m": 8},
                },
                {
                    "name": "HNSW Index (Fast Approximate)",
                    "type": VectorDBType.FAISS,
                    "index_type": IndexType.HNSW,
                    "distance_metric": DistanceMetric.COSINE,
                    "params": {"M": 16, "efConstruction": 200, "efSearch": 50},
                },
            ]

        if self.content_categories is None:
            self.content_categories = [
                "nature",
                "technology",
                "music",
                "art",
                "sports",
                "science",
                "travel",
                "food",
                "education",
                "entertainment",
            ]

        if self.content_modalities is None:
            self.content_modalities = ["text", "image", "document", "audio", "video"]


# Sample content templates for different categories and modalities
CONTENT_TEMPLATES = {
    "nature": {
        "text": [
            "beautiful mountain landscape with snow-capped peaks",
            "serene forest with tall pine trees and wildlife",
            "crystal clear lake reflecting the blue sky",
            "colorful sunset over the ocean horizon",
            "wildflower meadow in full spring bloom",
        ],
        "image": [
            "mountain_vista.jpg",
            "forest_path.jpg",
            "lake_reflection.jpg",
            "ocean_sunset.jpg",
            "wildflower_field.jpg",
        ],
        "document": [
            "National Park hiking guide and trail information",
            "Wildlife conservation research paper",
            "Environmental protection policy document",
            "Nature photography techniques manual",
            "Botanical species identification guide",
        ],
        "audio": [
            "forest_sounds.mp3",
            "ocean_waves.mp3",
            "bird_songs.mp3",
            "rainfall_ambience.mp3",
            "wind_through_trees.mp3",
        ],
        "video": [
            "nature_documentary.mp4",
            "wildlife_footage.mp4",
            "landscape_timelapse.mp4",
            "underwater_exploration.mp4",
            "mountain_climbing.mp4",
        ],
    },
    "technology": {
        "text": [
            "artificial intelligence and machine learning research",
            "quantum computing breakthrough discoveries",
            "blockchain and cryptocurrency innovations",
            "robotics and automation advances",
            "virtual reality immersive experiences",
        ],
        "image": [
            "ai_conference.jpg",
            "quantum_computer.jpg",
            "blockchain_diagram.jpg",
            "robot_assembly.jpg",
            "vr_headset.jpg",
        ],
        "document": [
            "Machine learning algorithm research paper",
            "Software development best practices guide",
            "Cybersecurity threat analysis report",
            "Cloud computing architecture whitepaper",
            "Data science methodology handbook",
        ],
        "audio": [
            "tech_podcast.mp3",
            "coding_tutorial.mp3",
            "startup_pitch.mp3",
            "webinar_recording.mp3",
            "conference_talk.mp3",
        ],
        "video": [
            "tech_conference.mp4",
            "coding_tutorial.mp4",
            "product_demo.mp4",
            "startup_presentation.mp4",
            "algorithm_explanation.mp4",
        ],
    },
    "music": {
        "text": [
            "classical symphony orchestra performance",
            "jazz improvisation and musical creativity",
            "electronic music production techniques",
            "world music cultural traditions",
            "rock concert live performance energy",
        ],
        "image": [
            "orchestra_performance.jpg",
            "jazz_club.jpg",
            "electronic_studio.jpg",
            "world_instruments.jpg",
            "rock_concert.jpg",
        ],
        "document": [
            "Music theory and composition textbook",
            "Audio engineering production guide",
            "Music history cultural analysis",
            "Instrument technique instruction manual",
            "Concert venue acoustics study",
        ],
        "audio": [
            "classical_symphony.mp3",
            "jazz_improvisation.mp3",
            "electronic_track.mp3",
            "world_music.mp3",
            "rock_anthem.mp3",
        ],
        "video": [
            "concert_performance.mp4",
            "music_lesson.mp4",
            "studio_session.mp4",
            "music_documentary.mp4",
            "instrument_tutorial.mp4",
        ],
    },
    "art": {
        "text": [
            "renaissance painting masterpiece analysis",
            "modern sculpture contemporary interpretation",
            "digital art creative expression techniques",
            "abstract expressionism artistic movement",
            "street art urban cultural significance",
        ],
        "image": [
            "renaissance_painting.jpg",
            "modern_sculpture.jpg",
            "digital_artwork.jpg",
            "abstract_painting.jpg",
            "street_art.jpg",
        ],
        "document": [
            "Art history comprehensive survey",
            "Painting techniques instructional guide",
            "Museum exhibition catalog",
            "Artist biography and works analysis",
            "Art criticism theoretical framework",
        ],
        "audio": [
            "gallery_tour.mp3",
            "artist_interview.mp3",
            "art_lecture.mp3",
            "museum_guide.mp3",
            "critique_discussion.mp3",
        ],
        "video": [
            "art_documentary.mp4",
            "painting_process.mp4",
            "gallery_exhibition.mp4",
            "sculpture_creation.mp4",
            "art_history_lesson.mp4",
        ],
    },
    "sports": {
        "text": [
            "olympic athletic competition excellence",
            "professional football championship strategy",
            "marathon running endurance training",
            "tennis tournament competitive dynamics",
            "swimming technique performance optimization",
        ],
        "image": [
            "olympic_ceremony.jpg",
            "football_stadium.jpg",
            "marathon_race.jpg",
            "tennis_match.jpg",
            "swimming_pool.jpg",
        ],
        "document": [
            "Athletic training methodology guide",
            "Sports psychology performance manual",
            "Nutrition for athletes handbook",
            "Equipment safety standards document",
            "Competition rules and regulations",
        ],
        "audio": [
            "sports_commentary.mp3",
            "training_motivation.mp3",
            "coach_interview.mp3",
            "stadium_atmosphere.mp3",
            "workout_music.mp3",
        ],
        "video": [
            "sports_highlights.mp4",
            "training_session.mp4",
            "competition_footage.mp4",
            "athlete_documentary.mp4",
            "technique_analysis.mp4",
        ],
    },
}


# Search query templates for testing
SEARCH_QUERIES = {
    "nature": [
        "mountain landscape nature scenery",
        "forest trees wildlife environment",
        "ocean water marine life",
        "flowers plants botanical garden",
        "wilderness outdoor adventure",
    ],
    "technology": [
        "artificial intelligence machine learning",
        "computer programming software development",
        "robotics automation engineering",
        "data science analytics research",
        "innovation startup technology",
    ],
    "music": [
        "classical symphony orchestra performance",
        "jazz blues improvisation music",
        "electronic digital music production",
        "concert live performance entertainment",
        "instruments musical composition",
    ],
    "art": [
        "painting drawing visual arts",
        "sculpture installation contemporary art",
        "museum gallery exhibition culture",
        "creative expression artistic vision",
        "design aesthetic beautiful artwork",
    ],
    "sports": [
        "athletic competition championship tournament",
        "training exercise physical fitness",
        "team strategy competitive sports",
        "performance excellence achievement",
        "outdoor recreation activity",
    ],
}


# Performance benchmark configurations
BENCHMARK_CONFIGS = {
    "small_scale": {
        "vector_count": 100,
        "search_queries": 10,
        "top_k": 5,
        "description": "Small scale testing (100 vectors)",
    },
    "medium_scale": {
        "vector_count": 1000,
        "search_queries": 50,
        "top_k": 10,
        "description": "Medium scale testing (1K vectors)",
    },
    "large_scale": {
        "vector_count": 10000,
        "search_queries": 100,
        "top_k": 20,
        "description": "Large scale testing (10K vectors)",
    },
}


# Default demo configuration instance
DEFAULT_DEMO_CONFIG = DemoConfig(
    scenario=DemoScenario.ALL,
    sample_size=50,
    large_dataset_size=500,
    performance_iterations=5,
    verbose_output=True,
)
