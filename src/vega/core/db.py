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
from typing import List, Dict, Optional

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


# Create the engine with SQLite pragmas suitable for local logging
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # allow access from different threads
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
        # Indexes
        try:
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_ts ON conversations (ts DESC);"
            )
            conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS ix_conv_session ON conversations (session_id, id DESC);"
            )
        except Exception:
            pass


_init_db()


def log_conversation(
    prompt: str, response: str, source: str = "api", session_id: Optional[str] = None
) -> int:
    """Insert a new conversation row and return its id."""
    # Optional PII masking
    try:
        from ..config import get_config
        from ..security import mask_pii

        if get_config().pii_masking:
            prompt = mask_pii(prompt)
            response = mask_pii(response)
    except Exception:
        pass
    with Session(engine) as sess:
        obj = Conversation(
            prompt=prompt,
            response=response,
            source=source,
            session_id=session_id,
        )
        sess.add(obj)
        sess.commit()
        return int(obj.id)


def purge_old(days: int) -> int:
    """Delete conversations older than N days. Returns rows deleted."""
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


def get_history(limit: int = 50) -> List[Dict]:
    """Fetch the most recent N conversation rows as dictionaries."""
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


def get_history_page(limit: int = 50, before_id: Optional[int] = None) -> List[Dict]:
    """Fetch a page of conversations.

    - If before_id is provided, return rows with id < before_id, newest first.
    - Otherwise, return the latest rows.
    """
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


def get_session_history(session_id: str, limit: int = 8) -> List[Dict]:
    """Fetch last N rows for a given session id, newest first."""
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
