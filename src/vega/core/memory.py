"""
memory.py - Dynamic memory system for Vega2.0

A comprehensive memory system that:
- Uses local database as dynamic memory store
- Maintains live index synchronization
- Implements load-on-demand with topic isolation
- Tracks all access with timestamps
- Provides automatic update detection
- Manages favorites based on usage patterns
- Comprehensive logging and error handling
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Boolean,
    create_engine,
    select,
    update,
    delete,
    Index,
    ForeignKey,
    func,
)
from sqlalchemy.orm import declarative_base, Session, relationship
from sqlalchemy.exc import SQLAlchemyError
import threading
import time

# Memory database path (separate from main conversation db)
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "vega_memory.db")
MEMORY_DATABASE_URL = f"sqlite:///{MEMORY_DB_PATH}"

Base = declarative_base()

# Configure logging for memory operations
memory_logger = logging.getLogger("vega.memory")
memory_logger.setLevel(logging.INFO)
if not memory_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - MEMORY - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    memory_logger.addHandler(handler)


class KnowledgeItem(Base):
    """Core knowledge storage with versioning and metadata"""

    __tablename__ = "knowledge_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(255), nullable=False, index=True)  # Unique identifier
    topic = Column(String(100), nullable=False, index=True)  # Topic/domain
    content = Column(Text, nullable=False)  # The actual knowledge
    content_hash = Column(String(64), nullable=False, index=True)  # SHA256 of content
    meta_data = Column(Text, nullable=True)  # JSON metadata (renamed to avoid conflict)
    version = Column(Integer, nullable=False, default=1)
    source = Column(String(100), nullable=False)  # Where it came from
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_used_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    usage_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True)

    # Relationships
    access_logs = relationship("AccessLog", back_populates="knowledge_item")
    favorite = relationship(
        "FavoriteItem", back_populates="knowledge_item", uselist=False
    )

    # Composite index for fast topic-based queries
    __table_args__ = (
        Index("idx_topic_active_used", "topic", "is_active", "last_used_at"),
        Index("idx_key_topic", "key", "topic"),
    )


class FavoriteItem(Base):
    """Fast-access favorites based on usage patterns"""

    __tablename__ = "favorite_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_item_id = Column(
        Integer, ForeignKey("knowledge_items.id"), nullable=False, unique=True
    )
    frequency_score = Column(Float, nullable=False, default=0.0)  # Usage frequency
    recency_score = Column(Float, nullable=False, default=0.0)  # Recent access
    combined_score = Column(Float, nullable=False, default=0.0)  # Combined ranking
    promoted_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    last_score_update = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationship
    knowledge_item = relationship("KnowledgeItem", back_populates="favorite")

    # Index for fast score-based retrieval
    __table_args__ = (Index("idx_combined_score", "combined_score"),)


class AccessLog(Base):
    """Comprehensive access tracking for all memory operations"""

    __tablename__ = "access_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_item_id = Column(Integer, ForeignKey("knowledge_items.id"), nullable=True)
    action = Column(String(20), nullable=False)  # read/write/cache/update/delete
    item_key = Column(String(255), nullable=True)  # Key if item exists
    topic = Column(String(100), nullable=True)  # Topic if applicable
    source = Column(String(100), nullable=False)  # Where action originated
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    outcome = Column(String(20), nullable=False)  # success/failure/partial
    error_message = Column(Text, nullable=True)  # Error details if failed
    meta_data = Column(Text, nullable=True)  # Additional context as JSON (renamed)

    # Relationship
    knowledge_item = relationship("KnowledgeItem", back_populates="access_logs")

    # Index for time-based queries
    __table_args__ = (
        Index("idx_timestamp_outcome", "timestamp", "outcome"),
        Index("idx_action_topic", "action", "topic"),
    )


class MemoryIndex(Base):
    """Live index for fast knowledge lookup and search"""

    __tablename__ = "memory_index"

    id = Column(Integer, primary_key=True, autoincrement=True)
    knowledge_item_id = Column(
        Integer, ForeignKey("knowledge_items.id"), nullable=False
    )
    topic = Column(String(100), nullable=False, index=True)
    search_terms = Column(
        Text, nullable=False
    )  # Searchable terms extracted from content
    content_preview = Column(String(500), nullable=False)  # First 500 chars for preview
    updated_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Composite index for fast search
    __table_args__ = (Index("idx_topic_terms", "topic", "search_terms"),)


# Create the engine with optimizations for memory operations
memory_engine = create_engine(
    MEMORY_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    connect_args={
        "check_same_thread": False,
        "timeout": 30,
    },
    pool_timeout=30,
    pool_recycle=3600,
)

# Create all tables
Base.metadata.create_all(memory_engine)

# Enable WAL mode for better concurrency
with memory_engine.connect() as conn:
    from sqlalchemy import text

    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.execute(text("PRAGMA synchronous=NORMAL"))
    conn.execute(text("PRAGMA temp_store=MEMORY"))
    conn.execute(text("PRAGMA mmap_size=268435456"))  # 256MB mmap
    conn.commit()


@dataclass
class MemoryItem:
    """In-memory representation of a knowledge item"""

    key: str
    topic: str
    content: str
    metadata: Optional[Dict[str, Any]] = None
    version: int = 1
    source: str = "unknown"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    usage_count: int = 0
    content_hash: Optional[str] = None

    def __post_init__(self):
        now = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
        if self.last_used_at is None:
            self.last_used_at = now
        if self.content_hash is None:
            self.content_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA256 hash of content for change detection"""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ["created_at", "updated_at", "last_used_at"]:
            if result[field]:
                result[field] = result[field].isoformat()
        return result


@contextmanager
def get_memory_session():
    """Context manager for database sessions with proper cleanup"""
    session = Session(memory_engine)
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        memory_logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def log_memory_action(
    action: str,
    item_key: Optional[str] = None,
    topic: Optional[str] = None,
    source: str = "memory",
    outcome: str = "success",
    error_message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Log memory system actions with full context"""
    try:
        with get_memory_session() as session:
            log_entry = AccessLog(
                action=action,
                item_key=item_key,
                topic=topic,
                source=source,
                outcome=outcome,
                error_message=error_message,
                meta_data=json.dumps(metadata) if metadata else None,
            )
            session.add(log_entry)
            session.commit()

        # Also log to Python logger
        log_msg = f"{action.upper()} key={item_key} topic={topic} outcome={outcome}"
        if error_message:
            log_msg += f" error={error_message}"
        memory_logger.info(log_msg)

    except Exception as e:
        memory_logger.error(f"Failed to log memory action: {e}")


def extract_search_terms(content: str) -> str:
    """Extract searchable terms from content"""
    # Simple term extraction - can be enhanced with NLP
    import re

    # Remove special characters and convert to lowercase
    terms = re.sub(r"[^\w\s]", " ", content.lower())

    # Split into words and filter out short ones
    words = [word for word in terms.split() if len(word) > 2]

    # Return unique terms joined by space
    return " ".join(sorted(set(words)))


class MemoryManager:
    """
    Core memory management system with load-on-demand, topic isolation,
    and automatic synchronization between database and index.
    """

    def __init__(self, retry_attempts: int = 3, retry_delay: float = 0.5):
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self._topic_cache: Dict[str, Dict[str, MemoryItem]] = {}
        self._cache_lock = threading.RLock()
        self._favorites_threshold = 5  # Usage count threshold for favorites
        memory_logger.info("MemoryManager initialized")

    def _retry_operation(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute operation with retry logic and error handling"""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                result = operation_func(*args, **kwargs)
                if attempt > 0:
                    memory_logger.info(
                        f"{operation_name} succeeded on attempt {attempt + 1}"
                    )
                return result
            except Exception as e:
                last_error = e
                memory_logger.warning(
                    f"{operation_name} attempt {attempt + 1} failed: {e}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff

        # All attempts failed
        log_memory_action(
            action=operation_name.lower(),
            outcome="failure",
            error_message=str(last_error),
            source="memory_manager",
        )
        raise last_error

    def _update_index_sync(
        self, session: Session, knowledge_item: KnowledgeItem
    ) -> None:
        """Update search index immediately after data change"""
        try:
            # Remove existing index entries
            session.execute(
                delete(MemoryIndex).where(
                    MemoryIndex.knowledge_item_id == knowledge_item.id
                )
            )

            # Create new index entry
            search_terms = extract_search_terms(knowledge_item.content)
            content_preview = knowledge_item.content[:500]

            index_entry = MemoryIndex(
                knowledge_item_id=knowledge_item.id,
                topic=knowledge_item.topic,
                search_terms=search_terms,
                content_preview=content_preview,
                updated_at=datetime.now(timezone.utc),
            )
            session.add(index_entry)

            memory_logger.debug(f"Index updated for item {knowledge_item.key}")

        except Exception as e:
            memory_logger.error(
                f"Failed to update index for item {knowledge_item.key}: {e}"
            )
            raise

    def _calculate_favorites_score(
        self, usage_count: int, last_used_at: datetime
    ) -> tuple[float, float, float]:
        """Calculate frequency, recency, and combined scores for favorites"""
        now = datetime.now(timezone.utc)

        # Frequency score (logarithmic to prevent runaway growth)
        frequency_score = min(100.0, 10.0 * (usage_count**0.5))

        # Recency score (decay over time)
        hours_since_use = (now - last_used_at).total_seconds() / 3600
        recency_score = max(0.0, 100.0 * (0.95**hours_since_use))

        # Combined score (weighted average)
        combined_score = (0.6 * frequency_score) + (0.4 * recency_score)

        return frequency_score, recency_score, combined_score

    def _update_favorites(
        self, session: Session, knowledge_item: KnowledgeItem
    ) -> None:
        """Update favorites table based on usage patterns"""
        try:
            freq_score, rec_score, combined_score = self._calculate_favorites_score(
                knowledge_item.usage_count, knowledge_item.last_used_at
            )

            # Check if item should be in favorites
            should_be_favorite = (
                knowledge_item.usage_count >= self._favorites_threshold
                or combined_score >= 50.0
            )

            existing_favorite = session.execute(
                select(FavoriteItem).where(
                    FavoriteItem.knowledge_item_id == knowledge_item.id
                )
            ).scalar_one_or_none()

            if should_be_favorite:
                if existing_favorite:
                    # Update existing favorite
                    existing_favorite.frequency_score = freq_score
                    existing_favorite.recency_score = rec_score
                    existing_favorite.combined_score = combined_score
                    existing_favorite.last_score_update = datetime.now(timezone.utc)
                else:
                    # Add new favorite
                    new_favorite = FavoriteItem(
                        knowledge_item_id=knowledge_item.id,
                        frequency_score=freq_score,
                        recency_score=rec_score,
                        combined_score=combined_score,
                    )
                    session.add(new_favorite)

                    log_memory_action(
                        action="favorite_promoted",
                        item_key=knowledge_item.key,
                        topic=knowledge_item.topic,
                        source="memory_manager",
                        metadata={"combined_score": combined_score},
                    )
            else:
                if existing_favorite:
                    # Remove from favorites if score dropped
                    session.delete(existing_favorite)
                    log_memory_action(
                        action="favorite_demoted",
                        item_key=knowledge_item.key,
                        topic=knowledge_item.topic,
                        source="memory_manager",
                    )

        except Exception as e:
            memory_logger.error(
                f"Failed to update favorites for item {knowledge_item.key}: {e}"
            )

    def _load_topic_cache(self, topic: str) -> None:
        """Load all items for a topic into memory cache"""
        with self._cache_lock:
            if topic in self._topic_cache:
                return  # Already loaded

            try:
                with get_memory_session() as session:
                    items = (
                        session.execute(
                            select(KnowledgeItem)
                            .where(
                                KnowledgeItem.topic == topic,
                                KnowledgeItem.is_active == True,
                            )
                            .order_by(KnowledgeItem.last_used_at.desc())
                        )
                        .scalars()
                        .all()
                    )

                    topic_cache = {}
                    for item in items:
                        metadata = (
                            json.loads(item.meta_data) if item.meta_data else None
                        )

                        memory_item = MemoryItem(
                            key=item.key,
                            topic=item.topic,
                            content=item.content,
                            metadata=metadata,
                            version=item.version,
                            source=item.source,
                            created_at=item.created_at,
                            updated_at=item.updated_at,
                            last_used_at=item.last_used_at,
                            usage_count=item.usage_count,
                            content_hash=item.content_hash,
                        )
                        topic_cache[item.key] = memory_item

                    self._topic_cache[topic] = topic_cache

                    log_memory_action(
                        action="cache_load",
                        topic=topic,
                        source="memory_manager",
                        metadata={"items_loaded": len(topic_cache)},
                    )

            except Exception as e:
                memory_logger.error(f"Failed to load topic cache for {topic}: {e}")
                self._topic_cache[topic] = (
                    {}
                )  # Create empty cache to prevent repeated failures

    def _clear_topic_cache(self, topic: str) -> None:
        """Clear topic cache to release memory"""
        with self._cache_lock:
            if topic in self._topic_cache:
                items_count = len(self._topic_cache[topic])
                del self._topic_cache[topic]

                log_memory_action(
                    action="cache_clear",
                    topic=topic,
                    source="memory_manager",
                    metadata={"items_cleared": items_count},
                )

    def store_knowledge(self, item: MemoryItem, source: str = "user") -> bool:
        """Store or update a knowledge item with automatic index sync"""

        def _store_operation():
            with get_memory_session() as session:
                # Check if item already exists
                existing = session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.key == item.key, KnowledgeItem.topic == item.topic
                    )
                ).scalar_one_or_none()

                now = datetime.now(timezone.utc)
                content_hash = item._calculate_hash()

                if existing:
                    # Check if content has changed
                    if existing.content_hash != content_hash:
                        # Update existing item
                        existing.content = item.content
                        existing.content_hash = content_hash
                        existing.meta_data = (
                            json.dumps(item.metadata) if item.metadata else None
                        )
                        existing.version += 1
                        existing.updated_at = now
                        existing.last_used_at = now
                        existing.usage_count += 1

                        # Update index immediately
                        self._update_index_sync(session, existing)

                        # Update favorites
                        self._update_favorites(session, existing)

                        # Update cache if loaded
                        with self._cache_lock:
                            if (
                                item.topic in self._topic_cache
                                and item.key in self._topic_cache[item.topic]
                            ):
                                self._topic_cache[item.topic][item.key] = item

                        log_memory_action(
                            action="update",
                            item_key=item.key,
                            topic=item.topic,
                            source=source,
                            metadata={"version": existing.version},
                        )
                        return True
                    else:
                        # Content unchanged, just update usage
                        existing.last_used_at = now
                        existing.usage_count += 1
                        self._update_favorites(session, existing)

                        log_memory_action(
                            action="touch",
                            item_key=item.key,
                            topic=item.topic,
                            source=source,
                        )
                        return True
                else:
                    # Create new item
                    new_item = KnowledgeItem(
                        key=item.key,
                        topic=item.topic,
                        content=item.content,
                        content_hash=content_hash,
                        meta_data=json.dumps(item.metadata) if item.metadata else None,
                        source=source,
                        created_at=now,
                        updated_at=now,
                        last_used_at=now,
                        usage_count=1,
                    )
                    session.add(new_item)
                    session.flush()  # Get the ID

                    # Update index immediately
                    self._update_index_sync(session, new_item)

                    # Update favorites
                    self._update_favorites(session, new_item)

                    # Update cache if loaded
                    with self._cache_lock:
                        if item.topic in self._topic_cache:
                            self._topic_cache[item.topic][item.key] = item

                    log_memory_action(
                        action="create",
                        item_key=item.key,
                        topic=item.topic,
                        source=source,
                        metadata={"id": new_item.id},
                    )
                    return True

        return self._retry_operation("store_knowledge", _store_operation)

    def get_knowledge(
        self, key: str, topic: str, source: str = "user"
    ) -> Optional[MemoryItem]:
        """Retrieve a specific knowledge item, updating access time"""

        def _get_operation():
            # Check cache first
            with self._cache_lock:
                if topic not in self._topic_cache:
                    self._load_topic_cache(topic)

                if key in self._topic_cache[topic]:
                    item = self._topic_cache[topic][key]

                    # Update last_used_at in database
                    try:
                        with get_memory_session() as session:
                            session.execute(
                                update(KnowledgeItem)
                                .where(
                                    KnowledgeItem.key == key,
                                    KnowledgeItem.topic == topic,
                                )
                                .values(
                                    last_used_at=datetime.now(timezone.utc),
                                    usage_count=KnowledgeItem.usage_count + 1,
                                )
                            )
                            session.commit()
                    except Exception as e:
                        memory_logger.warning(f"Failed to update usage for {key}: {e}")

                    log_memory_action(
                        action="read", item_key=key, topic=topic, source=source
                    )
                    return item

            # Not in cache, try database
            with get_memory_session() as session:
                db_item = session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.key == key,
                        KnowledgeItem.topic == topic,
                        KnowledgeItem.is_active == True,
                    )
                ).scalar_one_or_none()

                if db_item:
                    # Update usage
                    db_item.last_used_at = datetime.now(timezone.utc)
                    db_item.usage_count += 1
                    self._update_favorites(session, db_item)

                    # Convert to MemoryItem
                    metadata = (
                        json.loads(db_item.meta_data) if db_item.meta_data else None
                    )
                    memory_item = MemoryItem(
                        key=db_item.key,
                        topic=db_item.topic,
                        content=db_item.content,
                        metadata=metadata,
                        version=db_item.version,
                        source=db_item.source,
                        created_at=db_item.created_at,
                        updated_at=db_item.updated_at,
                        last_used_at=db_item.last_used_at,
                        usage_count=db_item.usage_count,
                        content_hash=db_item.content_hash,
                    )

                    # Add to cache
                    with self._cache_lock:
                        if topic not in self._topic_cache:
                            self._topic_cache[topic] = {}
                        self._topic_cache[topic][key] = memory_item

                    log_memory_action(
                        action="read", item_key=key, topic=topic, source=source
                    )
                    return memory_item

            log_memory_action(
                action="read",
                item_key=key,
                topic=topic,
                source=source,
                outcome="failure",
                error_message="Item not found",
            )
            return None

        return self._retry_operation("get_knowledge", _get_operation)

    def search_knowledge(
        self,
        query: str,
        topic: Optional[str] = None,
        limit: int = 10,
        source: str = "user",
    ) -> List[MemoryItem]:
        """Search knowledge items using the index"""

        def _search_operation():
            with get_memory_session() as session:
                # Build search query
                search_query = select(MemoryIndex).join(KnowledgeItem)

                if topic:
                    search_query = search_query.where(MemoryIndex.topic == topic)

                # Simple text search in search_terms
                search_terms = extract_search_terms(query)
                if search_terms:
                    # Use LIKE for simple substring matching
                    for term in search_terms.split()[:3]:  # Limit to first 3 terms
                        search_query = search_query.where(
                            MemoryIndex.search_terms.like(f"%{term}%")
                        )

                search_query = search_query.where(KnowledgeItem.is_active == True)
                search_query = search_query.order_by(KnowledgeItem.last_used_at.desc())
                search_query = search_query.limit(limit)

                results = session.execute(search_query).scalars().all()

                # Convert to MemoryItems
                memory_items = []
                for index_entry in results:
                    # Get the full knowledge item
                    knowledge_item = session.get(
                        KnowledgeItem, index_entry.knowledge_item_id
                    )
                    if knowledge_item:
                        metadata = (
                            json.loads(knowledge_item.meta_data)
                            if knowledge_item.meta_data
                            else None
                        )
                        memory_item = MemoryItem(
                            key=knowledge_item.key,
                            topic=knowledge_item.topic,
                            content=knowledge_item.content,
                            metadata=metadata,
                            version=knowledge_item.version,
                            source=knowledge_item.source,
                            created_at=knowledge_item.created_at,
                            updated_at=knowledge_item.updated_at,
                            last_used_at=knowledge_item.last_used_at,
                            usage_count=knowledge_item.usage_count,
                            content_hash=knowledge_item.content_hash,
                        )
                        memory_items.append(memory_item)

                log_memory_action(
                    action="search",
                    topic=topic,
                    source=source,
                    metadata={"query": query, "results": len(memory_items)},
                )
                return memory_items

        return self._retry_operation("search_knowledge", _search_operation)

    def get_favorites(
        self, topic: Optional[str] = None, limit: int = 20
    ) -> List[MemoryItem]:
        """Get favorite items sorted by combined score"""

        def _favorites_operation():
            with get_memory_session() as session:
                query = select(KnowledgeItem).join(FavoriteItem)

                if topic:
                    query = query.where(KnowledgeItem.topic == topic)

                query = query.where(KnowledgeItem.is_active == True)
                query = query.order_by(FavoriteItem.combined_score.desc())
                query = query.limit(limit)

                results = session.execute(query).scalars().all()

                memory_items = []
                for item in results:
                    metadata = json.loads(item.meta_data) if item.meta_data else None
                    memory_item = MemoryItem(
                        key=item.key,
                        topic=item.topic,
                        content=item.content,
                        metadata=metadata,
                        version=item.version,
                        source=item.source,
                        created_at=item.created_at,
                        updated_at=item.updated_at,
                        last_used_at=item.last_used_at,
                        usage_count=item.usage_count,
                        content_hash=item.content_hash,
                    )
                    memory_items.append(memory_item)

                log_memory_action(
                    action="get_favorites",
                    topic=topic,
                    source="memory_manager",
                    metadata={"results": len(memory_items)},
                )
                return memory_items

        return self._retry_operation("get_favorites", _favorites_operation)

    def delete_knowledge(self, key: str, topic: str, source: str = "user") -> bool:
        """Soft delete a knowledge item"""

        def _delete_operation():
            with get_memory_session() as session:
                result = session.execute(
                    update(KnowledgeItem)
                    .where(KnowledgeItem.key == key, KnowledgeItem.topic == topic)
                    .values(is_active=False, updated_at=datetime.now(timezone.utc))
                )

                if result.rowcount > 0:
                    # Remove from index
                    session.execute(
                        delete(MemoryIndex).where(
                            MemoryIndex.knowledge_item_id.in_(
                                select(KnowledgeItem.id).where(
                                    KnowledgeItem.key == key,
                                    KnowledgeItem.topic == topic,
                                )
                            )
                        )
                    )

                    # Remove from cache
                    with self._cache_lock:
                        if (
                            topic in self._topic_cache
                            and key in self._topic_cache[topic]
                        ):
                            del self._topic_cache[topic][key]

                    log_memory_action(
                        action="delete", item_key=key, topic=topic, source=source
                    )
                    return True

                return False

        return self._retry_operation("delete_knowledge", _delete_operation)

    def cleanup_topic(self, topic: str) -> None:
        """Release memory for a topic when done working with it"""
        self._clear_topic_cache(topic)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        try:
            with get_memory_session() as session:
                # Basic counts
                total_items = (
                    session.execute(
                        select(KnowledgeItem).where(KnowledgeItem.is_active == True)
                    ).rowcount
                    or 0
                )

                total_favorites = session.execute(select(FavoriteItem)).rowcount or 0

                # Topic distribution
                topic_counts = session.execute(
                    select(KnowledgeItem.topic, func.count(KnowledgeItem.id))
                    .where(KnowledgeItem.is_active == True)
                    .group_by(KnowledgeItem.topic)
                ).all()

                # Cache stats
                cache_stats = {}
                with self._cache_lock:
                    for topic, items in self._topic_cache.items():
                        cache_stats[topic] = len(items)

                return {
                    "total_items": total_items,
                    "total_favorites": total_favorites,
                    "topics": dict(topic_counts),
                    "cached_topics": cache_stats,
                    "database_path": MEMORY_DB_PATH,
                }
        except Exception as e:
            memory_logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
