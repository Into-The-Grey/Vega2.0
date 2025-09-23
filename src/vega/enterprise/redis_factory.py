"""
Cluster-aware Redis client factory for Vega2.0
Supports standalone, cluster, and sentinel modes using redis.asyncio
"""

try:
    import redis.asyncio as redis
except ImportError:
    import redis
from typing import Any


def create_redis_client(config: Any, db_override: int | None = None):
    """
    Create a Redis client (standalone, cluster, or sentinel) based on config.
    config: result of get_config() or dict with redis_mode, redis_cluster_nodes, etc.
    db_override: if set, overrides the db index (for RBAC, etc)
    """
    mode = getattr(config, "redis_mode", None) or config.get("redis_mode", "standalone")
    username = getattr(config, "redis_username", None) or config.get("redis_username")
    password = getattr(config, "redis_password", None) or config.get("redis_password")
    ssl = (
        getattr(config, "redis_ssl", False)
        if hasattr(config, "redis_ssl")
        else config.get("redis_ssl", False)
    )
    db = (
        db_override
        if db_override is not None
        else (
            getattr(config, "redis_db", 0)
            if hasattr(config, "redis_db")
            else config.get("redis_db", 0)
        )
    )
    if mode == "cluster":
        # redis_cluster_nodes: tuple[str, ...] or list[str]
        nodes = getattr(config, "redis_cluster_nodes", ()) or config.get(
            "redis_cluster_nodes", ()
        )
        if not nodes:
            raise ValueError("REDIS_CLUSTER_NODES must be set for cluster mode")
        startup_nodes = []
        for node in nodes:
            host, port = node.split(":")
            startup_nodes.append({"host": host, "port": int(port)})
        return redis.RedisCluster(
            startup_nodes=startup_nodes,
            username=username,
            password=password,
            ssl=ssl,
            decode_responses=True,
        )
    elif mode == "sentinel":
        # Not implemented: add Sentinel support if needed
        raise NotImplementedError("Redis Sentinel mode is not yet implemented.")
    else:
        # Standalone
        host = None
        port = None
        if hasattr(config, "redis_cluster_nodes") and config.redis_cluster_nodes:
            # Use first node for standalone
            node = config.redis_cluster_nodes[0]
            host, port = node.split(":")
        else:
            host = getattr(config, "redis_host", None) or config.get(
                "redis_host", "localhost"
            )
            port = getattr(config, "redis_port", None) or config.get("redis_port", 6379)
        return redis.Redis(
            host=host or "localhost",
            port=int(port) if port else 6379,
            db=db,
            username=username,
            password=password,
            ssl=ssl,
            decode_responses=True,
        )
