"""
db.py - SQLite persistence for Vega2.0

- Uses SQLAlchemy 2.0 to manage a simple SQLite database (vega.db)
- Table 'conversations' stores prompt/response pairs with timestamps and source labels
- Functions:
    log_conversation(prompt: str, response: str, source: str) -> None
    get_history(limit: int = 50) -> list[dict]
- Notes:
    * The database file is created next to this script by default
    * For concurrency, SQLite is sufficient for local use; WAL enabled
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import asyncio
import re

from sqlalchemy import Column, Integer, String, Text, DateTime, create_engine, select
from sqlalchemy.orm import declarative_base, Session
from sqlalchemy import text as sqltext
import shutil
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "vega.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    source = Column(String(32), nullable=False, default="api")  # api|cli|integration
    session_id = Column(String(64), nullable=True)
    # Feedback/meta (added via lightweight migration if missing)
    rating = Column(Integer, nullable=True)  # 1-5 scale
    tags = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    reviewed = Column(Integer, nullable=False, default=0)  # 0/1 flag


class MemoryFact(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    key = Column(String(64), nullable=False)
    value = Column(Text, nullable=False)


# Create the engine with SQLite pragmas suitable for local logging
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False,  # allow access from different threads
        "timeout": 20,  # Wait up to 20s for lock
    },
    pool_size=10,  # Connection pool for concurrent requests
    max_overflow=20,  # Allow 20 extra connections under load
    pool_pre_ping=True,  # Verify connections before using
    future=True,
)


def _init_db() -> None:
    # Create tables if not exist
    Base.metadata.create_all(engine)
    # Enable WAL for better concurrent reads
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        # Lightweight migration: add feedback columns if missing
        cols = {
            row[1]  # name
            for row in conn.exec_driver_sql(
                "PRAGMA table_info(conversations);"
            ).fetchall()
        }
        if "rating" not in cols:
            conn.exec_driver_sql("ALTER TABLE conversations ADD COLUMN rating INTEGER;")
        if "tags" not in cols:
            conn.exec_driver_sql("ALTER TABLE conversations ADD COLUMN tags TEXT;")
        if "notes" not in cols:
            conn.exec_driver_sql("ALTER TABLE conversations ADD COLUMN notes TEXT;")
        if "reviewed" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE conversations ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0;"
            )
        if "session_id" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE conversations ADD COLUMN session_id VARCHAR(64);"
            )
        # Indexes - Composite indexes for common query patterns
        try:
            # Single-column indexes (existing)
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_ts ON conversations (ts DESC);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_session ON conversations (session_id, id DESC);"
            )
            # Composite indexes for optimal query performance
            # For session-based time-range queries (get recent conversations in session)
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_session_ts ON conversations (session_id, ts DESC, id DESC);"
            )
            # For time-range queries across all sessions
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_ts_session ON conversations (ts DESC, session_id, id DESC);"
            )
            # For reviewed/unreviewed filtering (learning pipeline)
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_reviewed_ts ON conversations (reviewed, ts DESC);"
            )
            # For source-based filtering (api/cli/integration)
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_source_ts ON conversations (source, ts DESC);"
            )
        except Exception:
            pass


_init_db()


def log_conversation(
    prompt: str, response: str, source: str = "api", session_id: Optional[str] = None
) -> int:
    """Insert a new conversation row and return its id."""
    from .db_profiler import profile_db_function

    @profile_db_function
    def _do_log():
        # Optional PII masking
        prompt_masked = prompt
        response_masked = response
        try:
            from ..config import get_config
            from ..security import mask_pii

            if get_config().pii_masking:
                prompt_masked = mask_pii(prompt)
                response_masked = mask_pii(response)
        except Exception:
            pass

        with Session(engine) as sess:
            obj = Conversation(
                prompt=prompt_masked,
                response=response_masked,
                source=source,
                session_id=session_id,
            )
            sess.add(obj)
            sess.commit()
            result_id = int(obj.id)

        # Invalidate conversation history cache after write
        try:
            from .query_cache import get_query_cache_sync

            cache = get_query_cache_sync()
            cache.invalidate_pattern("conversation_history")
        except Exception:
            pass  # Cache invalidation failure shouldn't break logging

        return result_id

    return _do_log()


async def bulk_log_conversations(conversations: List[Dict[str, Any]]):
    """Bulk insert conversations asynchronously.

    Uses a threadpool to avoid blocking the event loop while performing
    synchronous SQLAlchemy operations.
    """

    def _insert_all():
        with Session(engine) as sess:
            objs = [
                Conversation(
                    prompt=conv.get("prompt", ""),
                    response=conv.get("response", ""),
                    source=conv.get("source", "api"),
                    session_id=conv.get("session_id"),
                )
                for conv in conversations
            ]
            if objs:
                sess.add_all(objs)
                sess.commit()

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _insert_all)


def purge_old(days: int) -> int:
    """Delete conversations older than N days. Returns rows deleted."""
    from .db_profiler import profile_db_function

    @profile_db_function
    def _do_purge():
        if days <= 0:
            return 0
        with Session(engine) as sess:
            delta = f"-{int(days)} days"
            res = sess.execute(
                sqltext("DELETE FROM conversations WHERE ts < datetime('now', :delta)"),
                {"delta": delta},
            )
            sess.commit()
            return res.rowcount or 0

    return _do_purge()


def get_history(limit: int = 50) -> List[Dict]:
    """Fetch the most recent N conversation rows as dictionaries."""
    from .db_profiler import profile_db_function

    @profile_db_function
    def _do_get():
        with Session(engine) as sess:
            stmt = select(Conversation).order_by(Conversation.id.desc()).limit(limit)
            rows = sess.execute(stmt).scalars().all()
            # newest first; convert to dicts
            return [
                {
                    "id": r.id,
                    "ts": r.ts.isoformat(),
                    "prompt": r.prompt,
                    "response": r.response,
                    "source": r.source,
                    "rating": r.rating,
                    "tags": r.tags,
                    "notes": r.notes,
                    "reviewed": int(r.reviewed or 0),
                }
                for r in rows
            ]

    return _do_get()


async def get_history_cached(limit: int = 50, ttl: float = 60.0) -> List[Dict]:
    """
    Fetch conversation history with intelligent caching.

    Uses query cache to avoid repeated database queries.
    Automatically invalidated when new conversations are logged.

    Args:
        limit: Number of recent conversations to fetch
        ttl: Cache TTL in seconds (default 60s)

    Returns:
        List of conversation dictionaries
    """
    try:
        from .query_cache import get_query_cache
    except ImportError:
        # Fallback to non-cached version if cache unavailable
        return get_history(limit)

    cache = await get_query_cache()

    async def _fetch():
        # Run synchronous DB query in executor to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, get_history, limit)

    return await cache.get_or_fetch(
        "conversation_history",
        _fetch,
        limit=limit,
        ttl=ttl,
    )


def get_history_page(limit: int = 50, before_id: Optional[int] = None) -> List[Dict]:
    """Fetch a page of conversations.

    - If before_id is provided, return rows with id < before_id, newest first.
    - Otherwise, return the latest rows.
    """
    from .db_profiler import profile_db_function

    @profile_db_function
    def _do_get_page():
        with Session(engine) as sess:
            stmt = select(Conversation)
            if before_id is not None:
                stmt = stmt.where(Conversation.id < int(before_id))
            stmt = stmt.order_by(Conversation.id.desc()).limit(limit)
            rows = sess.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "ts": r.ts.isoformat(),
                    "prompt": r.prompt,
                    "response": r.response,
                    "source": r.source,
                    "rating": r.rating,
                    "tags": r.tags,
                    "notes": r.notes,
                    "reviewed": int(r.reviewed or 0),
                }
                for r in rows
            ]

    return _do_get_page()


def get_session_history(session_id: str, limit: int = 8) -> List[Dict]:
    """Fetch last N rows for a given session id, newest first."""
    from .db_profiler import profile_db_function

    @profile_db_function
    def _do_get_session():
        with Session(engine) as sess:
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.id.desc())
                .limit(limit)
            )
            rows = sess.execute(stmt).scalars().all()
            return [
                {
                    "id": r.id,
                    "ts": r.ts.isoformat(),
                    "prompt": r.prompt,
                    "response": r.response,
                    "source": r.source,
                    "session_id": r.session_id,
                }
                for r in rows
            ]

    return _do_get_session()


def set_feedback(
    conversation_id: int,
    rating: Optional[int] = None,
    tags: Optional[str] = None,
    notes: Optional[str] = None,
    reviewed: Optional[bool] = None,
) -> bool:
    """Update feedback fields for a conversation row.

    Returns True if a row was updated.
    """
    with Session(engine) as sess:
        obj = sess.get(Conversation, conversation_id)
        if not obj:
            return False
        if rating is not None:
            obj.rating = int(rating)
        if tags is not None:
            obj.tags = tags
        if notes is not None:
            obj.notes = notes
        if reviewed is not None:
            obj.reviewed = 1 if reviewed else 0
        sess.commit()
        return True


def get_for_training(
    min_rating: int = 4, reviewed_only: bool = False, limit: Optional[int] = None
) -> List[Dict]:
    """Return conversations suitable for training, filtered by rating and review flag."""
    with Session(engine) as sess:
        stmt = select(Conversation)
        if min_rating is not None:
            from sqlalchemy import or_

            # include rows with rating >= min_rating
            stmt = stmt.where(Conversation.rating >= int(min_rating))
        if reviewed_only:
            stmt = stmt.where(Conversation.reviewed == 1)
        stmt = stmt.order_by(Conversation.id.desc())
        if limit:
            stmt = stmt.limit(int(limit))
        rows = sess.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "prompt": r.prompt,
                "response": r.response,
                "source": r.source,
                "rating": r.rating,
                "tags": r.tags,
                "notes": r.notes,
                "reviewed": int(r.reviewed or 0),
            }
            for r in rows
        ]


def backup_db(out_path: Optional[str] = None) -> str:
    """Create a timestamped backup copy of the SQLite database."""
    src = DB_PATH
    if out_path is None:
        base, ext = os.path.splitext(DB_PATH)
        out_path = f"{base}.backup"
    shutil.copy2(src, out_path)
    return out_path


def vacuum_db() -> None:
    """Run VACUUM to reclaim space."""
    with engine.connect() as conn:
        conn.exec_driver_sql("VACUUM;")


def export_jsonl(path: str, limit: Optional[int] = None) -> str:
    """Export conversations to a JSONL file."""
    rows = get_history(limit=limit or 1000000)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def import_jsonl(path: str) -> int:
    """Import conversations from a JSONL file (prompt/response/source optional)."""
    count = 0
    with open(path, "r", encoding="utf-8") as f, Session(engine) as sess:
        for line in f:
            try:
                o = json.loads(line)
                prompt = o.get("prompt")
                response = o.get("response")
                source = o.get("source", "import")
                session_id = o.get("session_id")
                if not prompt or not response:
                    continue
                obj = Conversation(
                    prompt=prompt,
                    response=response,
                    source=source,
                    session_id=session_id,
                )
                sess.add(obj)
                count += 1
            except Exception:
                continue
        sess.commit()
    return count


def search_conversations(q: str, limit: int = 50) -> List[Dict]:
    """Simple LIKE-based search across prompt and response."""
    qlike = f"%{q}%"
    with Session(engine) as sess:
        stmt = (
            select(Conversation)
            .where(
                (Conversation.prompt.like(qlike)) | (Conversation.response.like(qlike))
            )
            .order_by(Conversation.id.desc())
            .limit(limit)
        )
        rows = sess.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "prompt": r.prompt,
                "response": r.response,
                "source": r.source,
            }
            for r in rows
        ]


def get_persistent_session_id() -> str:
    """
    Get or create a persistent session ID for continuing conversations.
    Returns the most recent session_id or creates a new one if none exists.
    """
    with Session(engine) as sess:
        # Get the most recent conversation's session_id
        stmt = (
            select(Conversation.session_id)
            .where(Conversation.session_id.isnot(None))
            .order_by(Conversation.id.desc())
            .limit(1)
        )
        result = sess.execute(stmt).scalar_one_or_none()

        if result:
            return result
        else:
            # No existing session, create a new persistent one
            import uuid

            return f"persistent-{uuid.uuid4()}"


def get_recent_context(
    session_id: Optional[str] = None, limit: int = 10, max_chars: int = 4000
) -> List[Dict]:
    """
    Efficiently retrieve recent conversation history for LLM context.

    Args:
        session_id: If provided, get context from this session only.
                   If None, get the most recent global context.
        limit: Maximum number of exchanges (prompt+response pairs) to retrieve
        max_chars: Maximum total characters to include (prevents context overflow)

    Returns:
        List of conversations in chronological order (oldest first) suitable for LLM context
    """
    with Session(engine) as sess:
        stmt = select(Conversation)

        if session_id:
            stmt = stmt.where(Conversation.session_id == session_id)

        # Get recent conversations, newest first
        stmt = stmt.order_by(Conversation.id.desc()).limit(limit)
        rows = sess.execute(stmt).scalars().all()

        # Reverse to get chronological order (oldest to newest)
        rows = list(reversed(rows))

        # Build context list with character limit
        context = []
        total_chars = 0

        for r in rows:
            entry = {
                "id": r.id,
                "ts": r.ts.isoformat(),
                "prompt": r.prompt,
                "response": r.response,
                "session_id": r.session_id,
            }

            entry_size = len(r.prompt) + len(r.response)

            # Stop if adding this would exceed character limit
            if total_chars + entry_size > max_chars and context:
                break

            context.append(entry)
            total_chars += entry_size

        return context


def get_conversation_summary(
    session_id: Optional[str] = None,
    older_than_id: Optional[int] = None,
    max_entries: int = 100,
) -> str:
    """
    Generate a compact summary of older conversations for compressed context.
    Useful for very long conversation histories.

    Args:
        session_id: Session to summarize
        older_than_id: Only summarize conversations with id < this value
        max_entries: Maximum number of old conversations to include in summary

    Returns:
        A text summary of the older conversation history
    """
    with Session(engine) as sess:
        stmt = select(Conversation)

        if session_id:
            stmt = stmt.where(Conversation.session_id == session_id)

        if older_than_id:
            stmt = stmt.where(Conversation.id < older_than_id)

        stmt = stmt.order_by(Conversation.id.desc()).limit(max_entries)
        rows = sess.execute(stmt).scalars().all()

        if not rows:
            return ""

        # Build a compact summary
        summary_lines = [f"[Earlier conversation context: {len(rows)} exchanges]"]

        # Group by topic/theme (simple keyword extraction)
        topics = {}
        for r in rows:
            # Extract first few words of prompt as topic indicator
            words = r.prompt.split()[:5]
            topic_key = " ".join(words)
            if topic_key not in topics:
                topics[topic_key] = 0
            topics[topic_key] += 1

        # Add topic summary
        if topics:
            summary_lines.append("Topics discussed:")
            for topic, count in sorted(topics.items(), key=lambda x: -x[1])[:5]:
                summary_lines.append(f"  - {topic}... ({count} exchanges)")

        return "\n".join(summary_lines)


def _sanitize_utf8(s: str) -> str:
    """Remove invalid UTF-8 sequences and surrogate pairs"""
    try:
        return s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _normalize_key(s: str) -> str:
    """Normalize memory fact keys.

    - Trim leading/trailing whitespace
    - Collapse internal whitespace to a single space
    - Remove zero-width and control characters
    - Lowercase for consistency
    """
    if s is None:
        return ""
    # Ensure string and sanitize encoding first
    try:
        if not isinstance(s, str):
            s = str(s)
    except Exception:
        s = ""
    s = _sanitize_utf8(s)
    # Remove zero-width and BOM-like chars
    zero_width_pattern = r"[\u200B\u200C\u200D\u2060\uFEFF]"
    s = re.sub(zero_width_pattern, "", s)
    # Remove control chars
    s = re.sub(r"[\x00-\x1F\x7F]", "", s)
    # Collapse whitespace and trim
    s = re.sub(r"\s+", " ", s).strip()
    # Lowercase
    s = s.lower()
    return s


def set_memory_fact(session_id: Optional[str], key: str, value: str) -> None:
    """Store or update a simple memory fact for the given session.

    If session_id is None, stores as a global fact.
    """
    # Sanitize inputs to prevent encoding errors
    key = _sanitize_utf8(key)
    value = _sanitize_utf8(value)
    if session_id:
        session_id = _sanitize_utf8(session_id)

    # Optional key normalization via config toggle (defaults to True)
    try:
        from ..config import get_config

        if getattr(get_config(), "memory_normalize_keys", True):
            key = _normalize_key(key)
    except Exception:
        # If config unavailable, normalize by default for safety
        key = _normalize_key(key)

    with Session(engine) as sess:
        # delete existing fact with same key/session
        if session_id is None:
            sess.execute(
                sqltext("DELETE FROM memories WHERE session_id IS NULL AND key = :k"),
                {"k": key},
            )
        else:
            sess.execute(
                sqltext("DELETE FROM memories WHERE session_id = :sid AND key = :k"),
                {"sid": session_id, "k": key},
            )
        obj = MemoryFact(session_id=session_id, key=key, value=value)
        sess.add(obj)
        sess.commit()


def get_memory_facts(session_id: Optional[str]) -> Dict[str, str]:
    """Return memory facts for the session merged with global facts (None session)."""
    facts: Dict[str, str] = {}
    with Session(engine) as sess:
        # global facts
        stmt = select(MemoryFact).where(MemoryFact.session_id.is_(None))
        for r in sess.execute(stmt).scalars().all():
            facts[r.key] = r.value
        # session facts
        if session_id is not None:
            stmt = select(MemoryFact).where(MemoryFact.session_id == session_id)
            for r in sess.execute(stmt).scalars().all():
                facts[r.key] = r.value
    return facts


def compress_old_context(cutoff: datetime, keep_recent: int = 50) -> int:
    """
    Compress old conversation context by removing response text for old entries.
    Keeps prompts intact so we can still reference what was asked.

    Args:
        cutoff: Compress conversations older than this timestamp
        keep_recent: Always keep this many most recent conversations uncompressed

    Returns:
        Number of conversations compressed
    """
    with Session(engine) as sess:
        # Find conversations to compress (older than cutoff, excluding recent N)
        recent_ids_stmt = (
            select(Conversation.id).order_by(Conversation.id.desc()).limit(keep_recent)
        )
        recent_ids = [r[0] for r in sess.execute(recent_ids_stmt).all()]

        compress_stmt = (
            select(Conversation)
            .where(Conversation.ts < cutoff)
            .where(Conversation.id.notin_(recent_ids))
        )
        to_compress = sess.execute(compress_stmt).scalars().all()

        compressed_count = 0
        for conv in to_compress:
            # Keep prompt, compress response to summary
            if len(conv.response) > 100:
                conv.response = conv.response[:100] + "... [compressed]"
                compressed_count += 1

        sess.commit()
        return compressed_count


def summarize_and_archive_old(cutoff: datetime, keep_recent: int = 20) -> int:
    """
    Summarize very old conversations and remove their full text.
    Creates a summary entry and removes detailed conversations.

    Args:
        cutoff: Archive conversations older than this timestamp
        keep_recent: Always keep this many most recent conversations

    Returns:
        Number of conversations archived
    """
    with Session(engine) as sess:
        # Find conversations to archive (older than cutoff, excluding recent N)
        recent_ids_stmt = (
            select(Conversation.id).order_by(Conversation.id.desc()).limit(keep_recent)
        )
        recent_ids = [r[0] for r in sess.execute(recent_ids_stmt).all()]

        archive_stmt = (
            select(Conversation)
            .where(Conversation.ts < cutoff)
            .where(Conversation.id.notin_(recent_ids))
        )
        to_archive = sess.execute(archive_stmt).scalars().all()

        if not to_archive:
            return 0

        # Create a summary of archived conversations
        summary_lines = [
            f"[Archived {len(to_archive)} conversations from before {cutoff.isoformat()}]",
            "Topics discussed:",
        ]

        # Group by session and summarize
        session_groups = {}
        for conv in to_archive:
            sid = conv.session_id or "main"
            if sid not in session_groups:
                session_groups[sid] = []
            session_groups[sid].append(conv.prompt[:50])

        for sid, prompts in list(session_groups.items())[:10]:  # Max 10 sessions
            summary_lines.append(f"  {sid}: {len(prompts)} exchanges")

        summary_text = "\n".join(summary_lines)

        # Store summary as a memory fact
        oldest_session = to_archive[0].session_id if to_archive else None
        if oldest_session:
            set_memory_fact(
                oldest_session,
                f"archive_{cutoff.strftime('%Y%m%d_%H%M%S')}",
                summary_text,
            )

        # Delete the archived conversations
        archived_count = len(to_archive)
        for conv in to_archive:
            sess.delete(conv)

        sess.commit()
        return archived_count


def vacuum_database():
    """
    Vacuum the SQLite database to reclaim space after deletions.
    """
    try:
        with Session(engine) as sess:
            sess.execute(sqltext("VACUUM"))
            sess.commit()
    except Exception as e:
        logger.error(f"Error vacuuming database: {e}")


def get_db_size() -> int:
    """
    Get the size of the database file in bytes.

    Returns:
        Size in bytes, or 0 if error
    """
    try:
        import os

        db_path = "vega.db"  # Adjust if your DB path is different
        if os.path.exists(db_path):
            return os.path.getsize(db_path)
        return 0
    except Exception:
        return 0
