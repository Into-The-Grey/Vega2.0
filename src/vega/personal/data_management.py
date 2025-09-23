"""
Personal Data Management System

Provides local database management, personal schema organization,
and data governance for single-user environment.
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import aiosqlite
import hashlib
import shutil

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Categories for organizing personal data"""

    WORKSPACE = "workspace"
    DOCUMENTS = "documents"
    MEDIA = "media"
    CONVERSATIONS = "conversations"
    ANALYTICS = "analytics"
    TRAINING = "training"
    PREFERENCES = "preferences"
    SYSTEM = "system"


class DataAccessLevel(Enum):
    """Access levels for personal data organization"""

    PUBLIC = "public"  # Shareable data
    PRIVATE = "private"  # Personal only
    SENSITIVE = "sensitive"  # Encrypted storage
    TEMPORARY = "temporary"  # Auto-cleanup


@dataclass
class DataSchema:
    """Schema definition for personal data tables"""

    table_name: str
    category: DataCategory
    access_level: DataAccessLevel
    columns: Dict[str, str]  # column_name: sql_type
    indexes: List[str] = None
    retention_days: Optional[int] = None

    def __post_init__(self):
        if self.indexes is None:
            self.indexes = []


@dataclass
class DataRecord:
    """Individual data record with metadata"""

    id: Optional[str]
    category: DataCategory
    access_level: DataAccessLevel
    data: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PersonalDataManager:
    """
    Comprehensive personal data management system
    Handles local database operations, schema management, and data governance
    """

    def __init__(self, database_path: str = "data/personal.db"):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.schemas: Dict[str, DataSchema] = {}
        self._connection_pool = {}

        # Initialize default schemas
        self._initialize_default_schemas()

    def _initialize_default_schemas(self):
        """Initialize default schemas for personal data"""

        # Workspace data schema
        workspace_schema = DataSchema(
            table_name="workspace_items",
            category=DataCategory.WORKSPACE,
            access_level=DataAccessLevel.PRIVATE,
            columns={
                "id": "TEXT PRIMARY KEY",
                "workspace_id": "TEXT NOT NULL",
                "item_type": "TEXT NOT NULL",
                "name": "TEXT NOT NULL",
                "path": "TEXT",
                "content": "TEXT",
                "tags": "TEXT",  # JSON array
                "metadata": "TEXT",  # JSON object
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            indexes=["workspace_id", "item_type", "tags"],
            retention_days=365,
        )

        # Document management schema
        documents_schema = DataSchema(
            table_name="personal_documents",
            category=DataCategory.DOCUMENTS,
            access_level=DataAccessLevel.PRIVATE,
            columns={
                "id": "TEXT PRIMARY KEY",
                "title": "TEXT NOT NULL",
                "content": "TEXT",
                "format": "TEXT NOT NULL",
                "file_path": "TEXT",
                "version": "INTEGER DEFAULT 1",
                "parent_id": "TEXT",  # For versioning
                "tags": "TEXT",  # JSON array
                "summary": "TEXT",
                "word_count": "INTEGER",
                "metadata": "TEXT",  # JSON object
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            indexes=["title", "format", "tags", "parent_id"],
            retention_days=None,  # Keep forever
        )

        # Media processing schema
        media_schema = DataSchema(
            table_name="media_items",
            category=DataCategory.MEDIA,
            access_level=DataAccessLevel.PRIVATE,
            columns={
                "id": "TEXT PRIMARY KEY",
                "file_path": "TEXT NOT NULL",
                "media_type": "TEXT NOT NULL",  # image, video, audio
                "format": "TEXT NOT NULL",
                "size_bytes": "INTEGER",
                "duration": "REAL",  # For video/audio
                "resolution": "TEXT",  # For video/images
                "analysis_data": "TEXT",  # JSON object
                "embeddings": "BLOB",  # Vector embeddings
                "tags": "TEXT",  # JSON array
                "metadata": "TEXT",  # JSON object
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "processed_at": "TIMESTAMP",
            },
            indexes=["media_type", "format", "tags"],
            retention_days=180,
        )

        # Personal preferences schema
        preferences_schema = DataSchema(
            table_name="user_preferences",
            category=DataCategory.PREFERENCES,
            access_level=DataAccessLevel.PRIVATE,
            columns={
                "id": "TEXT PRIMARY KEY",
                "category": "TEXT NOT NULL",
                "key": "TEXT NOT NULL",
                "value": "TEXT NOT NULL",
                "data_type": "TEXT DEFAULT 'string'",
                "description": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            indexes=["category", "key"],
            retention_days=None,
        )

        # Analytics data schema
        analytics_schema = DataSchema(
            table_name="personal_analytics",
            category=DataCategory.ANALYTICS,
            access_level=DataAccessLevel.PRIVATE,
            columns={
                "id": "TEXT PRIMARY KEY",
                "event_type": "TEXT NOT NULL",
                "event_data": "TEXT",  # JSON object
                "timestamp": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "session_id": "TEXT",
                "user_agent": "TEXT",
                "metadata": "TEXT",  # JSON object
            },
            indexes=["event_type", "timestamp", "session_id"],
            retention_days=90,
        )

        # Training data schema
        training_schema = DataSchema(
            table_name="training_data",
            category=DataCategory.TRAINING,
            access_level=DataAccessLevel.SENSITIVE,
            columns={
                "id": "TEXT PRIMARY KEY",
                "dataset_name": "TEXT NOT NULL",
                "prompt": "TEXT NOT NULL",
                "response": "TEXT NOT NULL",
                "quality_score": "REAL",
                "source": "TEXT",
                "metadata": "TEXT",  # JSON object
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "reviewed_at": "TIMESTAMP",
            },
            indexes=["dataset_name", "quality_score", "source"],
            retention_days=365,
        )

        # Register all schemas
        for schema in [
            workspace_schema,
            documents_schema,
            media_schema,
            preferences_schema,
            analytics_schema,
            training_schema,
        ]:
            self.schemas[schema.table_name] = schema

    async def initialize_database(self):
        """Initialize database with all schemas"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                # Enable WAL mode for better concurrency
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA synchronous=NORMAL")
                await db.execute("PRAGMA cache_size=10000")
                await db.execute("PRAGMA temp_store=memory")

                # Create tables for all schemas
                for schema in self.schemas.values():
                    await self._create_table(db, schema)

                await db.commit()
                logger.info(
                    f"Initialized personal database with {len(self.schemas)} schemas"
                )

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def _create_table(self, db: aiosqlite.Connection, schema: DataSchema):
        """Create table from schema definition"""
        columns_sql = ", ".join(
            [f"{name} {sql_type}" for name, sql_type in schema.columns.items()]
        )

        create_sql = f"CREATE TABLE IF NOT EXISTS {schema.table_name} ({columns_sql})"
        await db.execute(create_sql)

        # Create indexes
        for index_column in schema.indexes:
            index_name = f"idx_{schema.table_name}_{index_column}"
            index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {schema.table_name} ({index_column})"
            await db.execute(index_sql)

    async def insert_data(self, table_name: str, data: Dict[str, Any]) -> str:
        """Insert data into specified table"""
        if table_name not in self.schemas:
            raise ValueError(f"Unknown table: {table_name}")

        schema = self.schemas[table_name]

        # Generate ID if not provided
        if "id" not in data:
            data["id"] = self._generate_id(table_name, data)

        # Add timestamps
        now = datetime.now().isoformat()
        if "created_at" not in data:
            data["created_at"] = now
        data["updated_at"] = now

        # Prepare SQL
        columns = list(data.keys())
        placeholders = ["?" for _ in columns]
        values = [self._serialize_value(data[col]) for col in columns]

        sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"

        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute(sql, values)
                await db.commit()

                logger.debug(f"Inserted data into {table_name}: {data['id']}")
                return data["id"]

        except Exception as e:
            logger.error(f"Failed to insert data into {table_name}: {e}")
            raise

    async def query_data(
        self,
        table_name: str,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query data from specified table"""
        if table_name not in self.schemas:
            raise ValueError(f"Unknown table: {table_name}")

        sql = f"SELECT * FROM {table_name}"
        params = []

        # Add filters
        if filters:
            where_clauses = []
            for column, value in filters.items():
                if isinstance(value, list):
                    placeholders = ",".join(["?" for _ in value])
                    where_clauses.append(f"{column} IN ({placeholders})")
                    params.extend(value)
                else:
                    where_clauses.append(f"{column} = ?")
                    params.append(value)

            if where_clauses:
                sql += f" WHERE {' AND '.join(where_clauses)}"

        # Add ordering
        if order_by:
            sql += f" ORDER BY {order_by}"

        # Add limit
        if limit:
            sql += f" LIMIT {limit}"

        try:
            async with aiosqlite.connect(self.database_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to query {table_name}: {e}")
            raise

    async def update_data(
        self, table_name: str, record_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update existing data record"""
        if table_name not in self.schemas:
            raise ValueError(f"Unknown table: {table_name}")

        # Add updated timestamp
        updates["updated_at"] = datetime.now().isoformat()

        # Prepare SQL
        set_clauses = [f"{column} = ?" for column in updates.keys()]
        values = [self._serialize_value(updates[col]) for col in updates.keys()]
        values.append(record_id)

        sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE id = ?"

        try:
            async with aiosqlite.connect(self.database_path) as db:
                cursor = await db.execute(sql, values)
                await db.commit()

                updated = cursor.rowcount > 0
                logger.debug(f"Updated {table_name} record {record_id}: {updated}")
                return updated

        except Exception as e:
            logger.error(f"Failed to update {table_name} record {record_id}: {e}")
            raise

    async def delete_data(self, table_name: str, record_id: str) -> bool:
        """Delete data record"""
        if table_name not in self.schemas:
            raise ValueError(f"Unknown table: {table_name}")

        try:
            async with aiosqlite.connect(self.database_path) as db:
                cursor = await db.execute(
                    f"DELETE FROM {table_name} WHERE id = ?", (record_id,)
                )
                await db.commit()

                deleted = cursor.rowcount > 0
                logger.debug(f"Deleted {table_name} record {record_id}: {deleted}")
                return deleted

        except Exception as e:
            logger.error(f"Failed to delete {table_name} record {record_id}: {e}")
            raise

    async def cleanup_expired_data(self):
        """Clean up expired data based on retention policies"""
        for schema in self.schemas.values():
            if schema.retention_days is None:
                continue

            cutoff_date = datetime.now() - timedelta(days=schema.retention_days)
            cutoff_str = cutoff_date.isoformat()

            try:
                async with aiosqlite.connect(self.database_path) as db:
                    cursor = await db.execute(
                        f"DELETE FROM {schema.table_name} WHERE created_at < ?",
                        (cutoff_str,),
                    )
                    await db.commit()

                    if cursor.rowcount > 0:
                        logger.info(
                            f"Cleaned up {cursor.rowcount} expired records from {schema.table_name}"
                        )

            except Exception as e:
                logger.error(f"Failed to cleanup {schema.table_name}: {e}")

    async def backup_database(self, backup_path: Optional[str] = None) -> str:
        """Create backup of personal database"""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"data/backups/personal_backup_{timestamp}.db"

        backup_file = Path(backup_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.copy2(self.database_path, backup_file)
            logger.info(f"Database backed up to {backup_file}")
            return str(backup_file)

        except Exception as e:
            logger.error(f"Failed to backup database: {e}")
            raise

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {}

        try:
            async with aiosqlite.connect(self.database_path) as db:
                # Get file size
                file_size = self.database_path.stat().st_size
                stats["file_size_bytes"] = file_size
                stats["file_size_mb"] = round(file_size / (1024 * 1024), 2)

                # Get table counts
                for schema in self.schemas.values():
                    async with db.execute(
                        f"SELECT COUNT(*) FROM {schema.table_name}"
                    ) as cursor:
                        count = await cursor.fetchone()
                        stats[f"{schema.table_name}_count"] = count[0]

                # Get database info
                async with db.execute("PRAGMA database_list") as cursor:
                    db_list = await cursor.fetchall()
                    stats["database_list"] = [dict(row) for row in db_list]

                stats["last_updated"] = datetime.now().isoformat()

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats["error"] = str(e)

        return stats

    def _generate_id(self, table_name: str, data: Dict[str, Any]) -> str:
        """Generate unique ID for data record"""
        content = f"{table_name}_{json.dumps(data, sort_keys=True)}_{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _serialize_value(self, value: Any) -> str:
        """Serialize complex values for database storage"""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return str(value)


# High-level convenience functions for common operations


async def save_workspace_item(
    manager: PersonalDataManager,
    workspace_id: str,
    item_type: str,
    name: str,
    content: Any,
    tags: List[str] = None,
) -> str:
    """Save item to personal workspace"""
    data = {
        "workspace_id": workspace_id,
        "item_type": item_type,
        "name": name,
        "content": content,
        "tags": json.dumps(tags or []),
        "metadata": json.dumps({}),
    }
    return await manager.insert_data("workspace_items", data)


async def save_document(
    manager: PersonalDataManager,
    title: str,
    content: str,
    format: str,
    tags: List[str] = None,
    file_path: str = None,
) -> str:
    """Save document to personal collection"""
    word_count = len(content.split()) if content else 0

    data = {
        "title": title,
        "content": content,
        "format": format,
        "file_path": file_path,
        "tags": json.dumps(tags or []),
        "word_count": word_count,
        "metadata": json.dumps({}),
    }
    return await manager.insert_data("personal_documents", data)


async def save_user_preference(
    manager: PersonalDataManager,
    category: str,
    key: str,
    value: Any,
    description: str = None,
) -> str:
    """Save user preference"""
    data_type = type(value).__name__

    data = {
        "category": category,
        "key": key,
        "value": json.dumps(value) if isinstance(value, (dict, list)) else str(value),
        "data_type": data_type,
        "description": description or "",
    }
    return await manager.insert_data("user_preferences", data)


async def track_analytics_event(
    manager: PersonalDataManager,
    event_type: str,
    event_data: Dict[str, Any],
    session_id: str = None,
) -> str:
    """Track analytics event"""
    data = {
        "event_type": event_type,
        "event_data": json.dumps(event_data),
        "session_id": session_id,
        "metadata": json.dumps({}),
    }
    return await manager.insert_data("personal_analytics", data)


# Example usage
async def demo_personal_data_management():
    """Demonstrate personal data management capabilities"""

    manager = PersonalDataManager()
    await manager.initialize_database()

    # Save workspace items
    doc_id = await save_workspace_item(
        manager,
        "main",
        "document",
        "My Notes",
        "Important personal notes",
        ["personal", "notes"],
    )

    # Save documents
    article_id = await save_document(
        manager,
        "Research Article",
        "Content of the article...",
        "markdown",
        ["research", "ai"],
    )

    # Save preferences
    pref_id = await save_user_preference(
        manager, "ui", "theme", "dark", "UI theme preference"
    )

    # Track analytics
    event_id = await track_analytics_event(
        manager, "document_created", {"document_id": article_id}
    )

    # Query data
    documents = await manager.query_data(
        "personal_documents", filters={"format": "markdown"}, order_by="created_at DESC"
    )

    # Get stats
    stats = await manager.get_database_stats()

    print(f"Personal Data Management Demo Complete:")
    print(f"- Workspace items: {len(await manager.query_data('workspace_items'))}")
    print(f"- Documents: {len(documents)}")
    print(f"- Database size: {stats.get('file_size_mb', 0)} MB")

    return manager


if __name__ == "__main__":
    asyncio.run(demo_personal_data_management())
