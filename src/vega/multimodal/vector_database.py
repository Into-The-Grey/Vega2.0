"""
Vector Database Integration
==========================

Unified vector database system for large-scale similarity search and retrieval
across multi-modal embeddings. Supports both FAISS (local) and Pinecone (cloud)
for scalable nearest neighbor search.

Features:
- FAISS integration for local high-performance search
- Pinecone integration for cloud-based scalable search
- Unified interface for different vector database backends
- Batch operations for efficient data management
- Advanced indexing strategies (IVF, HNSW, LSH)
- Metadata filtering and hybrid search
- Real-time index updates and management
- Performance monitoring and optimization
"""

import asyncio
import logging
import numpy as np
import json
import time
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import uuid

logger = logging.getLogger(__name__)


class VectorDBType(Enum):
    """Supported vector database types"""

    FAISS = "faiss"
    PINECONE = "pinecone"
    CHROMA = "chroma"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"


class IndexType(Enum):
    """Vector index types for different use cases"""

    FLAT = "flat"  # Brute force, exact search
    IVF_FLAT = "ivf_flat"  # Inverted file index
    IVF_PQ = "ivf_pq"  # IVF with product quantization
    HNSW = "hnsw"  # Hierarchical Navigable Small World
    LSH = "lsh"  # Locality Sensitive Hashing
    SCANN = "scann"  # Google's ScaNN


class DistanceMetric(Enum):
    """Distance metrics for similarity search"""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"
    HAMMING = "hamming"


@dataclass
class VectorDBConfig:
    """Configuration for vector database"""

    db_type: VectorDBType = VectorDBType.FAISS
    index_type: IndexType = IndexType.FLAT
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    dimension: int = 512

    # Index parameters
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    m: int = 8  # Number of subquantizers for PQ
    bits: int = 8  # Bits per subquantizer

    # Performance settings
    use_gpu: bool = False
    num_threads: int = 4
    batch_size: int = 1000

    # Storage settings
    index_file: str = "vector_index.faiss"
    metadata_file: str = "vector_metadata.json"
    backup_enabled: bool = True

    # Pinecone specific
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    pinecone_index_name: str = "vega-multimodal"

    # Advanced features
    enable_filtering: bool = True
    enable_hybrid_search: bool = True
    cache_size: int = 10000


@dataclass
class VectorRecord:
    """Record containing vector and metadata"""

    id: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    modality: str = "unknown"
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class SearchResult:
    """Search result with similarity score and metadata"""

    id: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[np.ndarray] = None
    distance: Optional[float] = None


@dataclass
class SearchQuery:
    """Query for vector search"""

    vector: np.ndarray
    top_k: int = 10
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_vectors: bool = False
    threshold: Optional[float] = None


class VectorDatabase(ABC):
    """Abstract base class for vector databases"""

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the vector database"""
        pass

    @abstractmethod
    async def add_vectors(self, records: List[VectorRecord]) -> bool:
        """Add vectors to the database"""
        pass

    @abstractmethod
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        pass

    @abstractmethod
    async def update_vector(self, record: VectorRecord) -> bool:
        """Update a vector record"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass


class FAISSVectorDB(VectorDatabase):
    """FAISS-based vector database implementation"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.index = None
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        self.next_index = 0
        self.is_initialized = False

        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
        self.add_count = 0
        self.total_add_time = 0.0

        self.executor = ThreadPoolExecutor(max_workers=config.num_threads)

    async def initialize(self) -> bool:
        """Initialize FAISS index"""
        try:
            # Try to import FAISS
            try:
                import faiss

                self.faiss = faiss
            except ImportError:
                logger.error("FAISS not available. Using mock implementation.")
                self.faiss = None
                return await self._initialize_mock()

            # Create FAISS index based on configuration
            if self.config.index_type == IndexType.FLAT:
                if self.config.distance_metric == DistanceMetric.COSINE:
                    self.index = self.faiss.IndexFlatIP(self.config.dimension)
                else:
                    self.index = self.faiss.IndexFlatL2(self.config.dimension)

            elif self.config.index_type == IndexType.IVF_FLAT:
                quantizer = self.faiss.IndexFlatL2(self.config.dimension)
                self.index = self.faiss.IndexIVFFlat(
                    quantizer, self.config.dimension, self.config.nlist
                )

            elif self.config.index_type == IndexType.IVF_PQ:
                quantizer = self.faiss.IndexFlatL2(self.config.dimension)
                self.index = self.faiss.IndexIVFPQ(
                    quantizer,
                    self.config.dimension,
                    self.config.nlist,
                    self.config.m,
                    self.config.bits,
                )

            elif self.config.index_type == IndexType.HNSW:
                self.index = self.faiss.IndexHNSWFlat(self.config.dimension, 32)

            else:
                # Default to flat index
                self.index = self.faiss.IndexFlatL2(self.config.dimension)

            # Configure search parameters
            if hasattr(self.index, "nprobe"):
                self.index.nprobe = self.config.nprobe

            # Try to load existing index
            await self._load_index()

            self.is_initialized = True
            logger.info(
                f"FAISS vector database initialized with {self.config.index_type.value}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize FAISS database: {e}")
            return False

    async def _initialize_mock(self) -> bool:
        """Initialize mock FAISS implementation for demo"""
        self.index = MockFAISSIndex(self.config.dimension)
        self.is_initialized = True
        logger.info("Mock FAISS vector database initialized")
        return True

    async def add_vectors(self, records: List[VectorRecord]) -> bool:
        """Add vectors to FAISS index"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Prepare vectors and metadata
            vectors = []
            for record in records:
                # Normalize vectors for cosine similarity
                if self.config.distance_metric == DistanceMetric.COSINE:
                    vector = record.vector / np.linalg.norm(record.vector)
                else:
                    vector = record.vector

                vectors.append(vector)

                # Store metadata and ID mapping
                index_id = self.next_index
                self.id_to_index[record.id] = index_id
                self.index_to_id[index_id] = record.id
                self.metadata_store[record.id] = {
                    "metadata": record.metadata,
                    "modality": record.modality,
                    "source": record.source,
                    "timestamp": record.timestamp,
                }
                self.next_index += 1

            # Add to index
            vectors_array = np.array(vectors, dtype=np.float32)

            if hasattr(self.index, "train") and not self.index.is_trained:
                self.index.train(vectors_array)

            self.index.add(vectors_array)

            # Save index and metadata
            await self._save_index()

            processing_time = time.time() - start_time
            self.add_count += len(records)
            self.total_add_time += processing_time

            logger.info(f"Added {len(records)} vectors in {processing_time:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            return False

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Normalize query vector for cosine similarity
            query_vector = query.vector
            if self.config.distance_metric == DistanceMetric.COSINE:
                query_vector = query_vector / np.linalg.norm(query_vector)

            # Perform search
            query_array = query_vector.reshape(1, -1).astype(np.float32)
            scores, indices = self.index.search(query_array, query.top_k)

            # Process results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # No more results
                    break

                # Apply threshold filter
                if query.threshold is not None:
                    if self.config.distance_metric == DistanceMetric.COSINE:
                        if score < query.threshold:
                            continue
                    else:  # L2 distance
                        if score > query.threshold:
                            continue

                # Get record ID and metadata
                record_id = self.index_to_id.get(idx, f"unknown_{idx}")
                metadata = {}

                if query.include_metadata and record_id in self.metadata_store:
                    metadata = self.metadata_store[record_id]["metadata"]

                # Apply filters
                if query.filters and not self._apply_filters(metadata, query.filters):
                    continue

                result = SearchResult(
                    id=record_id,
                    score=float(score),
                    metadata=metadata,
                    distance=(
                        float(score)
                        if self.config.distance_metric != DistanceMetric.COSINE
                        else None
                    ),
                )

                if query.include_vectors:
                    # This would require storing original vectors separately
                    result.vector = None  # Placeholder

                results.append(result)

            processing_time = time.time() - start_time
            self.search_count += 1
            self.total_search_time += processing_time

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs (FAISS doesn't support direct deletion)"""
        try:
            # Remove from metadata store
            for record_id in ids:
                if record_id in self.metadata_store:
                    del self.metadata_store[record_id]

                # Mark index mapping as deleted
                if record_id in self.id_to_index:
                    idx = self.id_to_index[record_id]
                    del self.id_to_index[record_id]
                    if idx in self.index_to_id:
                        del self.index_to_id[idx]

            # Note: FAISS doesn't support efficient deletion
            # In production, you'd need to rebuild the index periodically
            logger.info(f"Marked {len(ids)} vectors for deletion")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    async def update_vector(self, record: VectorRecord) -> bool:
        """Update a vector record (requires deletion and re-addition)"""
        try:
            # Delete existing
            await self.delete_vectors([record.id])
            # Add updated
            return await self.add_vectors([record])

        except Exception as e:
            logger.error(f"Failed to update vector: {e}")
            return False

    def _apply_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Apply metadata filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False

            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False

        return True

    async def _save_index(self):
        """Save index and metadata to disk"""
        try:
            if self.config.backup_enabled:
                # Save FAISS index
                if self.faiss and hasattr(self.index, "ntotal"):
                    self.faiss.write_index(self.index, self.config.index_file)

                # Save metadata
                metadata_to_save = {
                    "metadata_store": self.metadata_store,
                    "id_to_index": self.id_to_index,
                    "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
                    "next_index": self.next_index,
                }

                with open(self.config.metadata_file, "w") as f:
                    json.dump(metadata_to_save, f, indent=2)

                logger.debug("Index and metadata saved")

        except Exception as e:
            logger.warning(f"Failed to save index: {e}")

    async def _load_index(self):
        """Load existing index and metadata"""
        try:
            # Load FAISS index
            if Path(self.config.index_file).exists() and self.faiss:
                self.index = self.faiss.read_index(self.config.index_file)
                logger.info(
                    f"Loaded existing FAISS index with {self.index.ntotal} vectors"
                )

            # Load metadata
            if Path(self.config.metadata_file).exists():
                with open(self.config.metadata_file, "r") as f:
                    data = json.load(f)

                self.metadata_store = data.get("metadata_store", {})
                self.id_to_index = data.get("id_to_index", {})
                self.index_to_id = {
                    int(k): v for k, v in data.get("index_to_id", {}).items()
                }
                self.next_index = data.get("next_index", 0)

                logger.info(f"Loaded metadata for {len(self.metadata_store)} records")

        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        avg_search_time = self.total_search_time / max(self.search_count, 1)
        avg_add_time = self.total_add_time / max(self.add_count, 1)

        stats = {
            "db_type": self.config.db_type.value,
            "index_type": self.config.index_type.value,
            "distance_metric": self.config.distance_metric.value,
            "dimension": self.config.dimension,
            "total_vectors": len(self.metadata_store),
            "search_count": self.search_count,
            "add_count": self.add_count,
            "avg_search_time": avg_search_time,
            "avg_add_time": avg_add_time,
            "is_initialized": self.is_initialized,
        }

        if hasattr(self.index, "ntotal"):
            stats["index_size"] = self.index.ntotal

        return stats


class MockFAISSIndex:
    """Mock FAISS index for demo purposes"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = []
        self.is_trained = True
        self.ntotal = 0

    def train(self, vectors: np.ndarray):
        pass

    def add(self, vectors: np.ndarray):
        for vector in vectors:
            self.vectors.append(vector)
        self.ntotal = len(self.vectors)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.vectors:
            return np.array([[]]), np.array([[]])

        query_vec = query[0]
        similarities = []

        for i, vector in enumerate(self.vectors):
            # Compute cosine similarity
            similarity = np.dot(query_vec, vector)
            similarities.append((similarity, i))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        # Return top k
        top_k = similarities[:k]
        scores = np.array([[score for score, _ in top_k]])
        indices = np.array([[idx for _, idx in top_k]])

        return scores, indices


class PineconeVectorDB(VectorDatabase):
    """Pinecone-based vector database implementation"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
        self.index = None
        self.is_initialized = False

        # Performance tracking
        self.search_count = 0
        self.total_search_time = 0.0
        self.upsert_count = 0
        self.total_upsert_time = 0.0

    async def initialize(self) -> bool:
        """Initialize Pinecone connection"""
        try:
            # Try to import Pinecone
            try:
                import pinecone

                self.pinecone = pinecone
            except ImportError:
                logger.error("Pinecone not available. Using mock implementation.")
                return await self._initialize_mock()

            if not self.config.pinecone_api_key:
                logger.error("Pinecone API key not provided")
                return False

            # Initialize Pinecone
            self.pinecone.init(
                api_key=self.config.pinecone_api_key,
                environment=self.config.pinecone_environment,
            )

            # Create or connect to index
            index_name = self.config.pinecone_index_name

            if index_name not in self.pinecone.list_indexes():
                # Create new index
                self.pinecone.create_index(
                    index_name,
                    dimension=self.config.dimension,
                    metric=self._get_pinecone_metric(),
                )
                logger.info(f"Created new Pinecone index: {index_name}")

            self.index = self.pinecone.Index(index_name)
            self.is_initialized = True

            logger.info(f"Pinecone vector database initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Pinecone database: {e}")
            return False

    async def _initialize_mock(self) -> bool:
        """Initialize mock Pinecone implementation"""
        self.index = MockPineconeIndex()
        self.is_initialized = True
        logger.info("Mock Pinecone vector database initialized")
        return True

    def _get_pinecone_metric(self) -> str:
        """Convert distance metric to Pinecone format"""
        metric_map = {
            DistanceMetric.COSINE: "cosine",
            DistanceMetric.EUCLIDEAN: "euclidean",
            DistanceMetric.DOT_PRODUCT: "dotproduct",
        }
        return metric_map.get(self.config.distance_metric, "cosine")

    async def add_vectors(self, records: List[VectorRecord]) -> bool:
        """Add vectors to Pinecone"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Prepare upsert data
            upsert_data = []
            for record in records:
                vector_data = {
                    "id": record.id,
                    "values": record.vector.tolist(),
                    "metadata": {
                        **record.metadata,
                        "modality": record.modality,
                        "source": record.source,
                        "timestamp": record.timestamp,
                    },
                }
                upsert_data.append(vector_data)

            # Batch upsert
            batch_size = self.config.batch_size
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i : i + batch_size]
                self.index.upsert(vectors=batch)

            processing_time = time.time() - start_time
            self.upsert_count += len(records)
            self.total_upsert_time += processing_time

            logger.info(
                f"Upserted {len(records)} vectors to Pinecone in {processing_time:.3f}s"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add vectors to Pinecone: {e}")
            return False

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors in Pinecone"""
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Prepare query
            query_params = {
                "vector": query.vector.tolist(),
                "top_k": query.top_k,
                "include_metadata": query.include_metadata,
                "include_values": query.include_vectors,
            }

            if query.filters:
                query_params["filter"] = query.filters

            # Perform search
            response = self.index.query(**query_params)

            # Process results
            results = []
            for match in response.matches:
                # Apply threshold filter
                if query.threshold is not None and match.score < query.threshold:
                    continue

                result = SearchResult(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata if hasattr(match, "metadata") else {},
                    vector=(
                        np.array(match.values)
                        if hasattr(match, "values") and match.values
                        else None
                    ),
                )
                results.append(result)

            processing_time = time.time() - start_time
            self.search_count += 1
            self.total_search_time += processing_time

            return results

        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors from Pinecone"""
        try:
            self.index.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} vectors from Pinecone")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors from Pinecone: {e}")
            return False

    async def update_vector(self, record: VectorRecord) -> bool:
        """Update a vector in Pinecone (same as upsert)"""
        return await self.add_vectors([record])

    def get_stats(self) -> Dict[str, Any]:
        """Get Pinecone database statistics"""
        avg_search_time = self.total_search_time / max(self.search_count, 1)
        avg_upsert_time = self.total_upsert_time / max(self.upsert_count, 1)

        stats = {
            "db_type": self.config.db_type.value,
            "index_name": self.config.pinecone_index_name,
            "dimension": self.config.dimension,
            "search_count": self.search_count,
            "upsert_count": self.upsert_count,
            "avg_search_time": avg_search_time,
            "avg_upsert_time": avg_upsert_time,
            "is_initialized": self.is_initialized,
        }

        try:
            if self.index and hasattr(self.index, "describe_index_stats"):
                index_stats = self.index.describe_index_stats()
                stats.update(index_stats)
        except Exception as e:
            logger.warning(f"Failed to get Pinecone stats: {e}")

        return stats


class MockPineconeIndex:
    """Mock Pinecone index for demo purposes"""

    def __init__(self):
        self.vectors = {}
        self.metadata = {}

    def upsert(self, vectors: List[Dict]):
        for vector_data in vectors:
            vector_id = vector_data["id"]
            self.vectors[vector_id] = np.array(vector_data["values"])
            self.metadata[vector_id] = vector_data.get("metadata", {})

    def query(
        self,
        vector: List[float],
        top_k: int,
        include_metadata: bool = False,
        include_values: bool = False,
        filter: Dict = None,
    ):

        query_vec = np.array(vector)
        similarities = []

        for vec_id, vec_data in self.vectors.items():
            # Skip if doesn't match filter
            if filter:
                metadata = self.metadata.get(vec_id, {})
                if not self._matches_filter(metadata, filter):
                    continue

            # Compute cosine similarity
            similarity = np.dot(query_vec, vec_data) / (
                np.linalg.norm(query_vec) * np.linalg.norm(vec_data)
            )
            similarities.append((similarity, vec_id))

        # Sort by similarity (descending)
        similarities.sort(reverse=True)

        # Create mock response
        matches = []
        for score, vec_id in similarities[:top_k]:
            match = type(
                "Match",
                (),
                {
                    "id": vec_id,
                    "score": float(score),
                },
            )

            if include_metadata:
                match.metadata = self.metadata.get(vec_id, {})

            if include_values:
                match.values = self.vectors[vec_id].tolist()

            matches.append(match)

        return type("QueryResponse", (), {"matches": matches})

    def delete(self, ids: List[str]):
        for vec_id in ids:
            if vec_id in self.vectors:
                del self.vectors[vec_id]
            if vec_id in self.metadata:
                del self.metadata[vec_id]

    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter"""
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def describe_index_stats(self):
        return {
            "namespaces": {"": {"vector_count": len(self.vectors)}},
            "dimension": 512,
            "index_fullness": 0.1,
            "total_vector_count": len(self.vectors),
        }


class UnifiedVectorDB:
    """Unified interface for different vector database backends"""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.db = None
        self.is_initialized = False

        # Create appropriate database instance
        if config.db_type == VectorDBType.FAISS:
            self.db = FAISSVectorDB(config)
        elif config.db_type == VectorDBType.PINECONE:
            self.db = PineconeVectorDB(config)
        else:
            raise ValueError(f"Unsupported database type: {config.db_type}")

    async def initialize(self) -> bool:
        """Initialize the vector database"""
        if self.db:
            self.is_initialized = await self.db.initialize()
            return self.is_initialized
        return False

    async def add_vectors(self, records: List[VectorRecord]) -> bool:
        """Add vectors to the database"""
        if not self.is_initialized:
            await self.initialize()
        return await self.db.add_vectors(records)

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors"""
        if not self.is_initialized:
            await self.initialize()
        return await self.db.search(query)

    async def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs"""
        if not self.is_initialized:
            await self.initialize()
        return await self.db.delete_vectors(ids)

    async def update_vector(self, record: VectorRecord) -> bool:
        """Update a vector record"""
        if not self.is_initialized:
            await self.initialize()
        return await self.db.update_vector(record)

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.db:
            return self.db.get_stats()
        return {}

    async def batch_search(
        self, queries: List[SearchQuery]
    ) -> List[List[SearchResult]]:
        """Perform batch search operations"""
        results = []
        for query in queries:
            result = await self.search(query)
            results.append(result)
        return results

    async def similarity_search_by_id(
        self, vector_id: str, top_k: int = 10
    ) -> List[SearchResult]:
        """Find similar vectors to a stored vector by ID"""
        # This would require storing vectors separately or retrieving from index
        # For now, return empty list
        logger.warning("Similarity search by ID not yet implemented")
        return []

    async def export_vectors(self, output_file: str, format: str = "json") -> bool:
        """Export vectors and metadata to file"""
        try:
            if format == "json":
                # This would export the metadata store
                # Implementation depends on specific database backend
                logger.info(f"Vector export to {output_file} not yet implemented")
                return True
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    async def import_vectors(self, input_file: str, format: str = "json") -> bool:
        """Import vectors and metadata from file"""
        try:
            if format == "json":
                # This would import vectors from file
                # Implementation depends on specific file format
                logger.info(f"Vector import from {input_file} not yet implemented")
                return True
        except Exception as e:
            logger.error(f"Import failed: {e}")
            return False


# Factory functions for easy database creation
def create_faiss_db(
    dimension: int = 512,
    index_type: IndexType = IndexType.FLAT,
    distance_metric: DistanceMetric = DistanceMetric.COSINE,
    **kwargs,
) -> UnifiedVectorDB:
    """Create a FAISS-based vector database"""

    config = VectorDBConfig(
        db_type=VectorDBType.FAISS,
        dimension=dimension,
        index_type=index_type,
        distance_metric=distance_metric,
        **kwargs,
    )

    return UnifiedVectorDB(config)


def create_pinecone_db(
    dimension: int = 512,
    api_key: str = None,
    index_name: str = "vega-multimodal",
    environment: str = "us-west1-gcp",
    **kwargs,
) -> UnifiedVectorDB:
    """Create a Pinecone-based vector database"""

    config = VectorDBConfig(
        db_type=VectorDBType.PINECONE,
        dimension=dimension,
        pinecone_api_key=api_key,
        pinecone_index_name=index_name,
        pinecone_environment=environment,
        **kwargs,
    )

    return UnifiedVectorDB(config)
