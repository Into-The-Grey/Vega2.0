"""
Database Optimization System

Provides query optimization, connection pooling, and enhanced local
database performance for single-user workloads.
"""

import asyncio
import sqlite3
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import aiosqlite
from queue import Queue, Empty
import statistics
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries"""

    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    INDEX = "index"


class OptimizationLevel(Enum):
    """Database optimization levels"""

    BASIC = "basic"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class QueryMetrics:
    """Query performance metrics"""

    query_id: str
    query_type: QueryType
    sql: str
    execution_time_ms: float
    rows_affected: int
    timestamp: datetime
    optimization_applied: bool = False
    cache_hit: bool = False

    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""

    pool_size: int
    active_connections: int
    idle_connections: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_wait_time_ms: float
    peak_usage: int
    last_updated: datetime

    def __post_init__(self):
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)


class AsyncConnectionPool:
    """
    Async SQLite connection pool for optimal performance
    """

    def __init__(self, database_path: str, pool_size: int = 10, timeout: float = 30.0):
        self.database_path = database_path
        self.pool_size = pool_size
        self.timeout = timeout
        self.connections: Queue = Queue(maxsize=pool_size)
        self.active_count = 0
        self.stats = ConnectionPoolStats(
            pool_size=pool_size,
            active_connections=0,
            idle_connections=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_wait_time_ms=0.0,
            peak_usage=0,
            last_updated=datetime.now(),
        )
        self.wait_times: List[float] = []
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize connection pool"""
        for _ in range(self.pool_size):
            conn = await aiosqlite.connect(self.database_path)

            # Apply optimizations
            await self._optimize_connection(conn)

            self.connections.put(conn)

        self.stats.idle_connections = self.pool_size
        logger.info(f"Initialized connection pool with {self.pool_size} connections")

    async def _optimize_connection(self, conn: aiosqlite.Connection):
        """Apply SQLite optimizations to connection"""
        optimizations = [
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA cache_size=10000",
            "PRAGMA temp_store=memory",
            "PRAGMA mmap_size=268435456",  # 256MB
            "PRAGMA page_size=4096",
            "PRAGMA auto_vacuum=INCREMENTAL",
            "PRAGMA busy_timeout=30000",
            "PRAGMA foreign_keys=ON",
        ]

        for pragma in optimizations:
            await conn.execute(pragma)

        await conn.commit()

    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with context manager"""
        start_time = time.time()

        async with self._lock:
            self.stats.total_requests += 1

        connection = None
        try:
            # Try to get connection from pool
            try:
                connection = self.connections.get_nowait()
                async with self._lock:
                    self.stats.idle_connections -= 1
                    self.stats.active_connections += 1
                    if self.stats.active_connections > self.stats.peak_usage:
                        self.stats.peak_usage = self.stats.active_connections
            except Empty:
                # Pool is empty, create new connection if under limit
                if self.active_count < self.pool_size * 2:  # Allow overflow
                    connection = await aiosqlite.connect(self.database_path)
                    await self._optimize_connection(connection)
                    self.active_count += 1
                    async with self._lock:
                        self.stats.active_connections += 1
                else:
                    # Wait for connection to become available
                    await asyncio.sleep(0.1)
                    return await self.get_connection()

            wait_time = (time.time() - start_time) * 1000
            self.wait_times.append(wait_time)
            if len(self.wait_times) > 1000:
                self.wait_times = self.wait_times[-1000:]  # Keep last 1000

            async with self._lock:
                self.stats.successful_requests += 1
                self.stats.average_wait_time_ms = statistics.mean(self.wait_times)
                self.stats.last_updated = datetime.now()

            yield connection

        except Exception as e:
            async with self._lock:
                self.stats.failed_requests += 1
            logger.error(f"Connection pool error: {e}")
            raise
        finally:
            if connection:
                try:
                    # Return connection to pool
                    self.connections.put_nowait(connection)
                    async with self._lock:
                        self.stats.active_connections -= 1
                        self.stats.idle_connections += 1
                except:
                    # Pool is full, close connection
                    await connection.close()
                    self.active_count -= 1

    async def close_all(self):
        """Close all connections in pool"""
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                await conn.close()
            except Empty:
                break

        self.stats.active_connections = 0
        self.stats.idle_connections = 0
        logger.info("Closed all connections in pool")


class QueryOptimizer:
    """
    Intelligent query optimizer for SQLite
    """

    def __init__(
        self, optimization_level: OptimizationLevel = OptimizationLevel.MODERATE
    ):
        self.optimization_level = optimization_level
        self.query_patterns: Dict[str, str] = {}
        self.index_suggestions: List[str] = []
        self.query_cache: Dict[str, Any] = {}
        self.metrics: List[QueryMetrics] = []

        # Load optimization patterns
        self._initialize_optimization_patterns()

    def _initialize_optimization_patterns(self):
        """Initialize query optimization patterns"""

        # Common optimization patterns
        self.query_patterns = {
            # Remove unnecessary DISTINCT
            r"SELECT\s+DISTINCT\s+(.*?)\s+FROM\s+(\w+)\s+WHERE\s+(\w+)\s*=\s*\?": r"SELECT \1 FROM \2 WHERE \3 = ?",
            # Use LIMIT for large result sets
            r"SELECT\s+(.*?)\s+FROM\s+(\w+)(?!\s+.*LIMIT)$": r"SELECT \1 FROM \2 LIMIT 1000",
            # Optimize COUNT queries
            r"SELECT\s+COUNT\(\*\)\s+FROM\s+(\w+)\s+WHERE": r"SELECT COUNT(1) FROM \1 WHERE",
            # Use covering indexes hint
            r"SELECT\s+(.*?)\s+FROM\s+(\w+)\s+WHERE\s+(\w+)": r"SELECT \1 FROM \2 INDEXED BY idx_\2_\3 WHERE \3",
        }

        # Index suggestions based on common patterns
        self.index_suggestions = [
            "CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})",
            "CREATE INDEX IF NOT EXISTS idx_{table}_{column1}_{column2} ON {table}({column1}, {column2})",
            "CREATE INDEX IF NOT EXISTS idx_{table}_created_at ON {table}(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_{table}_updated_at ON {table}(updated_at)",
        ]

    def optimize_query(
        self, sql: str, params: Optional[Tuple] = None
    ) -> Tuple[str, bool]:
        """Optimize SQL query"""
        original_sql = sql
        optimized = False

        # Apply optimization patterns
        if self.optimization_level in [
            OptimizationLevel.MODERATE,
            OptimizationLevel.AGGRESSIVE,
        ]:
            for pattern, replacement in self.query_patterns.items():
                import re

                if re.search(pattern, sql, re.IGNORECASE):
                    new_sql = re.sub(pattern, replacement, sql, flags=re.IGNORECASE)
                    if new_sql != sql:
                        sql = new_sql
                        optimized = True
                        break

        # Add query hints for aggressive optimization
        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            if sql.upper().startswith("SELECT") and "ORDER BY" not in sql.upper():
                # Add optimization hints
                sql = sql.replace("SELECT", "SELECT /*+ USE_INDEX */")
                optimized = True

        return sql, optimized

    def suggest_indexes(self, sql: str, table_name: str) -> List[str]:
        """Suggest indexes based on query patterns"""
        suggestions = []

        # Extract WHERE conditions
        import re

        where_match = re.search(
            r"WHERE\s+(.*?)(?:\s+ORDER\s+BY|\s+GROUP\s+BY|\s+LIMIT|$)",
            sql,
            re.IGNORECASE,
        )

        if where_match:
            where_clause = where_match.group(1)

            # Find column references
            column_matches = re.findall(r"(\w+)\s*[=<>!]", where_clause)
            for column in column_matches:
                if column.lower() not in ["and", "or", "not"]:
                    index_sql = f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{column} ON {table_name}({column})"
                    suggestions.append(index_sql)

        # Suggest timestamp indexes
        if "created_at" in sql or "updated_at" in sql:
            suggestions.append(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamps ON {table_name}(created_at, updated_at)"
            )

        return suggestions

    def analyze_query_performance(self, query_metrics: QueryMetrics):
        """Analyze query performance and suggest optimizations"""
        self.metrics.append(query_metrics)

        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        # Analyze slow queries
        if query_metrics.execution_time_ms > 1000:  # Slower than 1 second
            logger.warning(
                f"Slow query detected: {query_metrics.execution_time_ms:.2f}ms - {query_metrics.sql[:100]}..."
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary"""
        if not self.metrics:
            return {"error": "No metrics available"}

        # Calculate statistics
        execution_times = [m.execution_time_ms for m in self.metrics]

        summary = {
            "total_queries": len(self.metrics),
            "avg_execution_time_ms": statistics.mean(execution_times),
            "median_execution_time_ms": statistics.median(execution_times),
            "max_execution_time_ms": max(execution_times),
            "min_execution_time_ms": min(execution_times),
            "slow_queries": len(
                [m for m in self.metrics if m.execution_time_ms > 1000]
            ),
            "optimized_queries": len(
                [m for m in self.metrics if m.optimization_applied]
            ),
            "cache_hits": len([m for m in self.metrics if m.cache_hit]),
            "query_types": {},
        }

        # Query type distribution
        for metric in self.metrics:
            query_type = metric.query_type.value
            if query_type not in summary["query_types"]:
                summary["query_types"][query_type] = 0
            summary["query_types"][query_type] += 1

        return summary


class DatabaseOptimizer:
    """
    Comprehensive database optimization system
    """

    def __init__(
        self,
        database_path: str,
        optimization_level: OptimizationLevel = OptimizationLevel.MODERATE,
        pool_size: int = 10,
    ):
        self.database_path = database_path
        self.optimization_level = optimization_level
        self.connection_pool = AsyncConnectionPool(database_path, pool_size)
        self.query_optimizer = QueryOptimizer(optimization_level)
        self.query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self.cache_ttl = 300  # 5 minutes default

        # Performance monitoring
        self.total_queries = 0
        self.total_execution_time = 0.0
        self.optimization_enabled = True

    async def initialize(self):
        """Initialize database optimizer"""
        await self.connection_pool.initialize()
        await self._apply_database_optimizations()
        logger.info("Database optimizer initialized")

    async def _apply_database_optimizations(self):
        """Apply database-level optimizations"""
        optimizations = []

        if self.optimization_level in [
            OptimizationLevel.MODERATE,
            OptimizationLevel.AGGRESSIVE,
        ]:
            optimizations.extend(
                [
                    "ANALYZE",  # Update query planner statistics
                    "PRAGMA optimize",  # Apply automatic optimizations
                ]
            )

        if self.optimization_level == OptimizationLevel.AGGRESSIVE:
            optimizations.extend(
                [
                    "VACUUM",  # Rebuild database file
                    "REINDEX",  # Rebuild all indexes
                ]
            )

        async with self.connection_pool.get_connection() as conn:
            for optimization in optimizations:
                try:
                    await conn.execute(optimization)
                    await conn.commit()
                    logger.info(f"Applied optimization: {optimization}")
                except Exception as e:
                    logger.warning(f"Failed to apply optimization {optimization}: {e}")

    async def execute_query(
        self, sql: str, params: Optional[Tuple] = None, fetch_all: bool = True
    ) -> Union[List[Dict], int]:
        """Execute optimized query"""
        start_time = time.time()

        # Generate cache key
        cache_key = self._generate_cache_key(sql, params)

        # Check cache first for SELECT queries
        if sql.strip().upper().startswith("SELECT"):
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                # Record cache hit
                execution_time = (time.time() - start_time) * 1000
                query_metrics = QueryMetrics(
                    query_id=cache_key[:16],
                    query_type=self._get_query_type(sql),
                    sql=sql,
                    execution_time_ms=execution_time,
                    rows_affected=(
                        len(cached_result) if isinstance(cached_result, list) else 1
                    ),
                    timestamp=datetime.now(),
                    cache_hit=True,
                )
                self.query_optimizer.analyze_query_performance(query_metrics)
                return cached_result

        # Optimize query
        optimized_sql, was_optimized = self.query_optimizer.optimize_query(sql, params)

        # Execute query
        result = None
        rows_affected = 0

        try:
            async with self.connection_pool.get_connection() as conn:
                if fetch_all and sql.strip().upper().startswith("SELECT"):
                    # SELECT query
                    conn.row_factory = aiosqlite.Row
                    async with conn.execute(optimized_sql, params or ()) as cursor:
                        rows = await cursor.fetchall()
                        result = [dict(row) for row in rows]
                        rows_affected = len(result)
                else:
                    # Non-SELECT query
                    cursor = await conn.execute(optimized_sql, params or ())
                    rows_affected = cursor.rowcount
                    result = rows_affected
                    await conn.commit()

                # Cache SELECT results
                if sql.strip().upper().startswith("SELECT") and result is not None:
                    self._cache_result(cache_key, result)

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

        # Record metrics
        execution_time = (time.time() - start_time) * 1000
        query_metrics = QueryMetrics(
            query_id=cache_key[:16],
            query_type=self._get_query_type(sql),
            sql=sql,
            execution_time_ms=execution_time,
            rows_affected=rows_affected,
            timestamp=datetime.now(),
            optimization_applied=was_optimized,
            cache_hit=False,
        )

        self.query_optimizer.analyze_query_performance(query_metrics)

        # Update global stats
        self.total_queries += 1
        self.total_execution_time += execution_time

        return result

    def _generate_cache_key(self, sql: str, params: Optional[Tuple]) -> str:
        """Generate cache key for query"""
        import hashlib

        content = f"{sql}{params or ()}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached query result"""
        if cache_key in self.query_cache:
            result, timestamp = self.query_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return result
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: Any):
        """Cache query result"""
        self.query_cache[cache_key] = (result, datetime.now())

        # Limit cache size
        if len(self.query_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(self.query_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_cache[:100]:  # Remove 100 oldest
                del self.query_cache[key]

    def _get_query_type(self, sql: str) -> QueryType:
        """Determine query type from SQL"""
        sql_upper = sql.strip().upper()

        if sql_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif sql_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif sql_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif sql_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif sql_upper.startswith("CREATE"):
            return QueryType.CREATE
        elif sql_upper.startswith("DROP"):
            return QueryType.DROP
        else:
            return QueryType.SELECT  # Default

    async def create_suggested_indexes(
        self, table_name: str, sample_queries: List[str]
    ):
        """Create indexes based on query analysis"""
        all_suggestions = set()

        for sql in sample_queries:
            suggestions = self.query_optimizer.suggest_indexes(sql, table_name)
            all_suggestions.update(suggestions)

        created_indexes = []
        async with self.connection_pool.get_connection() as conn:
            for index_sql in all_suggestions:
                try:
                    await conn.execute(index_sql)
                    await conn.commit()
                    created_indexes.append(index_sql)
                    logger.info(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {e}")

        return created_indexes

    async def vacuum_analyze(self):
        """Perform database maintenance"""
        maintenance_ops = ["VACUUM", "ANALYZE", "PRAGMA optimize"]

        async with self.connection_pool.get_connection() as conn:
            for op in maintenance_ops:
                try:
                    await conn.execute(op)
                    await conn.commit()
                    logger.info(f"Completed maintenance: {op}")
                except Exception as e:
                    logger.error(f"Maintenance failed {op}: {e}")

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            "total_queries": self.total_queries,
            "avg_execution_time_ms": (
                (self.total_execution_time / self.total_queries)
                if self.total_queries > 0
                else 0
            ),
            "cache_size": len(self.query_cache),
            "connection_pool": asdict(self.connection_pool.stats),
            "query_performance": self.query_optimizer.get_performance_summary(),
            "optimization_level": self.optimization_level.value,
            "last_updated": datetime.now().isoformat(),
        }

        # Get database file size
        try:
            import os

            file_size = os.path.getsize(self.database_path)
            stats["database_size_bytes"] = file_size
            stats["database_size_mb"] = round(file_size / (1024 * 1024), 2)
        except Exception as e:
            stats["database_size_error"] = str(e)

        return stats

    async def close(self):
        """Close database optimizer"""
        await self.connection_pool.close_all()
        logger.info("Database optimizer closed")


# Demo and testing functions
async def demo_database_optimization():
    """Demonstrate database optimization capabilities"""

    # Initialize optimizer
    optimizer = DatabaseOptimizer(
        database_path="data/demo_optimized.db",
        optimization_level=OptimizationLevel.MODERATE,
        pool_size=5,
    )

    await optimizer.initialize()

    print("Database Optimization Demo")

    # Create test table
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS test_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        category TEXT,
        value REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """

    await optimizer.execute_query(create_table_sql, fetch_all=False)
    print("Created test table")

    # Insert test data
    print("Inserting test data...")
    for i in range(100):
        insert_sql = """
        INSERT INTO test_performance (name, category, value)
        VALUES (?, ?, ?)
        """
        await optimizer.execute_query(
            insert_sql, (f"Item {i}", f"Category {i % 10}", i * 1.5), fetch_all=False
        )

    # Test queries with optimization
    test_queries = [
        "SELECT * FROM test_performance WHERE category = 'Category 1'",
        "SELECT COUNT(*) FROM test_performance",
        "SELECT * FROM test_performance ORDER BY value DESC LIMIT 10",
        "SELECT category, AVG(value) FROM test_performance GROUP BY category",
    ]

    print("\nExecuting test queries...")
    for i, sql in enumerate(test_queries):
        start_time = time.time()
        result = await optimizer.execute_query(sql)
        execution_time = (time.time() - start_time) * 1000

        result_count = len(result) if isinstance(result, list) else result
        print(f"Query {i+1}: {result_count} results in {execution_time:.2f}ms")

    # Test cache effectiveness (run same query again)
    print("\nTesting query cache...")
    start_time = time.time()
    cached_result = await optimizer.execute_query(test_queries[0])
    cache_time = (time.time() - start_time) * 1000
    print(f"Cached query: {len(cached_result)} results in {cache_time:.2f}ms")

    # Create suggested indexes
    print("\nCreating suggested indexes...")
    created_indexes = await optimizer.create_suggested_indexes(
        "test_performance", test_queries
    )
    print(f"Created {len(created_indexes)} indexes")

    # Get performance statistics
    stats = await optimizer.get_database_stats()
    print(f"\nDatabase Statistics:")
    print(f"- Total queries: {stats['total_queries']}")
    print(f"- Average execution time: {stats['avg_execution_time_ms']:.2f}ms")
    print(f"- Cache size: {stats['cache_size']} entries")
    print(f"- Database size: {stats.get('database_size_mb', 'Unknown')} MB")
    print(
        f"- Connection pool stats: {stats['connection_pool']['successful_requests']} successful requests"
    )

    await optimizer.close()
    return optimizer


if __name__ == "__main__":
    asyncio.run(demo_database_optimization())
