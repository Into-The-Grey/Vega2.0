"""
Personal Load Balancing System

Provides intelligent request routing, resource optimization, and workload
distribution for single-user environments and local services.
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import asynccontextmanager
import json
import hashlib
import random
import statistics
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


class RoutingStrategy(Enum):
    """Request routing strategies"""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    INTELLIGENT = "intelligent"
    FAILOVER = "failover"


class HealthStatus(Enum):
    """Service health status"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RequestType(Enum):
    """Types of requests"""

    HTTP_API = "http_api"
    DATABASE = "database"
    COMPUTE = "compute"
    FILE_IO = "file_io"
    NETWORK = "network"
    CACHE = "cache"
    CUSTOM = "custom"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""

    endpoint_id: str
    name: str
    address: str
    port: int
    weight: int = 1
    max_connections: int = 100
    timeout_ms: int = 30000
    health_check_url: Optional[str] = None
    health_check_interval: int = 30
    tags: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ServiceHealth:
    """Service health information"""

    endpoint_id: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int
    success_count: int
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    active_connections: int = 0

    def __post_init__(self):
        if isinstance(self.last_check, str):
            self.last_check = datetime.fromisoformat(self.last_check)


@dataclass
class RequestMetrics:
    """Request performance metrics"""

    request_id: str
    endpoint_id: str
    request_type: RequestType
    start_time: datetime
    end_time: Optional[datetime] = None
    response_time_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    bytes_sent: int = 0
    bytes_received: int = 0

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = datetime.fromisoformat(self.end_time)


@dataclass
class LoadBalancerStats:
    """Load balancer statistics"""

    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    requests_per_second: float
    active_connections: int
    endpoint_distribution: Dict[str, int]
    health_status: Dict[str, str]
    last_updated: datetime

    def __post_init__(self):
        if isinstance(self.last_updated, str):
            self.last_updated = datetime.fromisoformat(self.last_updated)


class CircuitBreaker:
    """
    Circuit breaker for service protection
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Exception = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and (
                time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True

        return False

    def record_success(self):
        """Record successful execution"""
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None

    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
        }


class HealthChecker:
    """
    Service health monitoring
    """

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.health_data: Dict[str, ServiceHealth] = {}
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.is_running = False

    async def start_monitoring(self, endpoints: List[ServiceEndpoint]):
        """Start health monitoring for endpoints"""
        self.is_running = True

        for endpoint in endpoints:
            task = asyncio.create_task(self._monitor_endpoint_health(endpoint))
            self.check_tasks[endpoint.endpoint_id] = task

        logger.info(f"Started health monitoring for {len(endpoints)} endpoints")

    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.is_running = False

        for task in self.check_tasks.values():
            task.cancel()

        await asyncio.gather(*self.check_tasks.values(), return_exceptions=True)
        self.check_tasks.clear()

        logger.info("Stopped health monitoring")

    async def _monitor_endpoint_health(self, endpoint: ServiceEndpoint):
        """Monitor individual endpoint health"""
        while self.is_running:
            try:
                health = await self._check_endpoint_health(endpoint)
                self.health_data[endpoint.endpoint_id] = health

                await asyncio.sleep(endpoint.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {endpoint.endpoint_id}: {e}")

                # Mark as unhealthy
                self.health_data[endpoint.endpoint_id] = ServiceHealth(
                    endpoint_id=endpoint.endpoint_id,
                    status=HealthStatus.UNHEALTHY,
                    last_check=datetime.now(),
                    response_time_ms=float("inf"),
                    error_count=1,
                    success_count=0,
                )

                await asyncio.sleep(endpoint.health_check_interval)

    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> ServiceHealth:
        """Check individual endpoint health"""
        start_time = time.time()

        try:
            # Simulate health check (in real implementation, this would make HTTP request)
            await asyncio.sleep(0.01)  # Simulate network delay

            # Mock health check logic
            import random

            success = random.random() > 0.1  # 90% success rate

            response_time = (time.time() - start_time) * 1000

            if success:
                status = HealthStatus.HEALTHY
                if response_time > 1000:  # Slow response
                    status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            # Get current health or create new
            current_health = self.health_data.get(endpoint.endpoint_id)
            error_count = 0
            success_count = 1

            if current_health:
                if success:
                    error_count = max(0, current_health.error_count - 1)
                    success_count = current_health.success_count + 1
                else:
                    error_count = current_health.error_count + 1
                    success_count = current_health.success_count

            return ServiceHealth(
                endpoint_id=endpoint.endpoint_id,
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time,
                error_count=error_count,
                success_count=success_count,
                cpu_usage_percent=random.uniform(10, 80),
                memory_usage_percent=random.uniform(20, 70),
                active_connections=random.randint(0, endpoint.max_connections),
            )

        except Exception as e:
            return ServiceHealth(
                endpoint_id=endpoint.endpoint_id,
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                response_time_ms=float("inf"),
                error_count=1,
                success_count=0,
            )

    def get_health_status(self, endpoint_id: str) -> Optional[ServiceHealth]:
        """Get health status for endpoint"""
        return self.health_data.get(endpoint_id)

    def get_healthy_endpoints(self, endpoint_ids: List[str]) -> List[str]:
        """Get list of healthy endpoints"""
        healthy = []

        for endpoint_id in endpoint_ids:
            health = self.health_data.get(endpoint_id)
            if health and health.status in [
                HealthStatus.HEALTHY,
                HealthStatus.DEGRADED,
            ]:
                healthy.append(endpoint_id)

        return healthy


class LoadBalancingRouter:
    """
    Intelligent load balancing router
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self.health_checker = HealthChecker()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Routing state
        self.round_robin_index = 0
        self.connection_counts: Dict[str, int] = defaultdict(int)
        self.response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Request tracking
        self.request_metrics: List[RequestMetrics] = []
        self.active_requests: Dict[str, RequestMetrics] = {}

        # Statistics
        self.stats = LoadBalancerStats(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_response_time_ms=0.0,
            requests_per_second=0.0,
            active_connections=0,
            endpoint_distribution={},
            health_status={},
            last_updated=datetime.now(),
        )

        self.is_started = False

    async def start(self):
        """Start load balancer"""
        if self.endpoints:
            await self.health_checker.start_monitoring(list(self.endpoints.values()))

        self.is_started = True
        logger.info("Load balancer started")

    async def stop(self):
        """Stop load balancer"""
        await self.health_checker.stop_monitoring()
        self.is_started = False
        logger.info("Load balancer stopped")

    def add_endpoint(self, endpoint: ServiceEndpoint):
        """Add service endpoint"""
        self.endpoints[endpoint.endpoint_id] = endpoint

        # Initialize circuit breaker
        self.circuit_breakers[endpoint.endpoint_id] = CircuitBreaker(
            failure_threshold=5, recovery_timeout=60.0
        )

        # Initialize routing state
        self.connection_counts[endpoint.endpoint_id] = 0
        self.response_times[endpoint.endpoint_id] = deque(maxlen=100)

        logger.info(
            f"Added endpoint: {endpoint.name} ({endpoint.address}:{endpoint.port})"
        )

    def remove_endpoint(self, endpoint_id: str):
        """Remove service endpoint"""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            del self.circuit_breakers[endpoint_id]
            del self.connection_counts[endpoint_id]
            del self.response_times[endpoint_id]

            logger.info(f"Removed endpoint: {endpoint_id}")

    async def route_request(
        self,
        request_type: RequestType = RequestType.HTTP_API,
        tags: Optional[List[str]] = None,
    ) -> Optional[ServiceEndpoint]:
        """Route request to optimal endpoint"""
        if not self.endpoints:
            return None

        # Filter endpoints by tags if specified
        candidate_endpoints = list(self.endpoints.values())
        if tags:
            candidate_endpoints = [
                ep for ep in candidate_endpoints if any(tag in ep.tags for tag in tags)
            ]

        if not candidate_endpoints:
            return None

        # Get healthy endpoints
        healthy_endpoint_ids = self.health_checker.get_healthy_endpoints(
            [ep.endpoint_id for ep in candidate_endpoints]
        )

        healthy_endpoints = [
            ep for ep in candidate_endpoints if ep.endpoint_id in healthy_endpoint_ids
        ]

        if not healthy_endpoints:
            # No healthy endpoints, try circuit breaker recovery
            for ep in candidate_endpoints:
                cb = self.circuit_breakers[ep.endpoint_id]
                if cb.can_execute():
                    return ep
            return None

        # Apply routing strategy
        selected_endpoint = self._apply_routing_strategy(
            healthy_endpoints, request_type
        )

        if selected_endpoint:
            # Update connection count
            self.connection_counts[selected_endpoint.endpoint_id] += 1

        return selected_endpoint

    def _apply_routing_strategy(
        self, endpoints: List[ServiceEndpoint], request_type: RequestType
    ) -> Optional[ServiceEndpoint]:
        """Apply routing strategy to select endpoint"""
        if not endpoints:
            return None

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            endpoint = endpoints[self.round_robin_index % len(endpoints)]
            self.round_robin_index += 1
            return endpoint

        elif self.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            total_weight = sum(ep.weight for ep in endpoints)
            if total_weight == 0:
                return endpoints[0]

            # Create weighted list
            weighted_endpoints = []
            for ep in endpoints:
                weighted_endpoints.extend([ep] * ep.weight)

            index = self.round_robin_index % len(weighted_endpoints)
            self.round_robin_index += 1
            return weighted_endpoints[index]

        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return min(endpoints, key=lambda ep: self.connection_counts[ep.endpoint_id])

        elif self.strategy == RoutingStrategy.LEAST_RESPONSE_TIME:

            def avg_response_time(ep):
                times = self.response_times[ep.endpoint_id]
                return statistics.mean(times) if times else 0

            return min(endpoints, key=avg_response_time)

        elif self.strategy == RoutingStrategy.RESOURCE_BASED:
            return self._select_by_resources(endpoints)

        elif self.strategy == RoutingStrategy.INTELLIGENT:
            return self._intelligent_selection(endpoints, request_type)

        elif self.strategy == RoutingStrategy.FAILOVER:
            # Try endpoints in order of health and performance
            endpoints_by_score = sorted(
                endpoints, key=self._calculate_endpoint_score, reverse=True
            )
            return endpoints_by_score[0] if endpoints_by_score else None

        return endpoints[0]  # Default fallback

    def _select_by_resources(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Select endpoint based on resource usage"""

        def resource_score(ep):
            health = self.health_checker.get_health_status(ep.endpoint_id)
            if not health:
                return 0

            # Lower CPU/memory usage = higher score
            cpu_score = 100 - (health.cpu_usage_percent or 50)
            memory_score = 100 - (health.memory_usage_percent or 50)
            connection_score = 100 - (
                health.active_connections / ep.max_connections * 100
            )

            return (cpu_score + memory_score + connection_score) / 3

        return max(endpoints, key=resource_score)

    def _intelligent_selection(
        self, endpoints: List[ServiceEndpoint], request_type: RequestType
    ) -> ServiceEndpoint:
        """Intelligent endpoint selection using multiple factors"""
        scores = {}

        for ep in endpoints:
            score = self._calculate_endpoint_score(ep)

            # Adjust score based on request type
            if request_type == RequestType.COMPUTE:
                # Prefer endpoints with low CPU usage
                health = self.health_checker.get_health_status(ep.endpoint_id)
                if health and health.cpu_usage_percent:
                    score *= (100 - health.cpu_usage_percent) / 100

            elif request_type == RequestType.DATABASE:
                # Prefer endpoints with low connection count
                connection_ratio = (
                    self.connection_counts[ep.endpoint_id] / ep.max_connections
                )
                score *= 1 - connection_ratio

            scores[ep.endpoint_id] = score

        # Select endpoint with highest score
        best_endpoint_id = max(scores.keys(), key=lambda x: scores[x])
        return next(ep for ep in endpoints if ep.endpoint_id == best_endpoint_id)

    def _calculate_endpoint_score(self, endpoint: ServiceEndpoint) -> float:
        """Calculate comprehensive endpoint score"""
        score = 0.0

        # Health score
        health = self.health_checker.get_health_status(endpoint.endpoint_id)
        if health:
            if health.status == HealthStatus.HEALTHY:
                score += 40
            elif health.status == HealthStatus.DEGRADED:
                score += 20

            # Response time score (inverse)
            if health.response_time_ms < float("inf"):
                response_score = max(0, 20 - (health.response_time_ms / 100))
                score += response_score

        # Connection load score
        connection_ratio = (
            self.connection_counts[endpoint.endpoint_id] / endpoint.max_connections
        )
        connection_score = (1 - connection_ratio) * 20
        score += connection_score

        # Weight factor
        score *= endpoint.weight

        return score

    async def record_request_start(
        self, endpoint_id: str, request_type: RequestType
    ) -> str:
        """Record request start"""
        request_id = f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"

        metrics = RequestMetrics(
            request_id=request_id,
            endpoint_id=endpoint_id,
            request_type=request_type,
            start_time=datetime.now(),
        )

        self.active_requests[request_id] = metrics
        return request_id

    async def record_request_end(
        self,
        request_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ):
        """Record request completion"""
        if request_id not in self.active_requests:
            return

        metrics = self.active_requests.pop(request_id)
        metrics.end_time = datetime.now()
        metrics.response_time_ms = (
            metrics.end_time - metrics.start_time
        ).total_seconds() * 1000
        metrics.success = success
        metrics.error_message = error_message
        metrics.bytes_sent = bytes_sent
        metrics.bytes_received = bytes_received

        # Store metrics
        self.request_metrics.append(metrics)

        # Update endpoint response times
        if metrics.response_time_ms:
            self.response_times[metrics.endpoint_id].append(metrics.response_time_ms)

        # Update circuit breaker
        cb = self.circuit_breakers[metrics.endpoint_id]
        if success:
            cb.record_success()
        else:
            cb.record_failure()

        # Update connection count
        self.connection_counts[metrics.endpoint_id] -= 1

        # Update statistics
        self._update_stats()

        # Keep only recent metrics
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.request_metrics = [
            m for m in self.request_metrics if m.start_time >= cutoff_time
        ]

    def _update_stats(self):
        """Update load balancer statistics"""
        if not self.request_metrics:
            return

        # Calculate basic stats
        total_requests = len(self.request_metrics)
        successful_requests = sum(1 for m in self.request_metrics if m.success)
        failed_requests = total_requests - successful_requests

        # Average response time
        response_times = [
            m.response_time_ms
            for m in self.request_metrics
            if m.response_time_ms is not None
        ]
        avg_response_time = statistics.mean(response_times) if response_times else 0.0

        # Requests per second (last hour)
        current_time = datetime.now()
        recent_requests = [
            m
            for m in self.request_metrics
            if current_time - m.start_time <= timedelta(seconds=60)
        ]
        rps = len(recent_requests) / 60.0

        # Endpoint distribution
        endpoint_distribution = defaultdict(int)
        for m in self.request_metrics:
            endpoint_distribution[m.endpoint_id] += 1

        # Health status
        health_status = {}
        for endpoint_id in self.endpoints.keys():
            health = self.health_checker.get_health_status(endpoint_id)
            health_status[endpoint_id] = health.status.value if health else "unknown"

        # Update stats
        self.stats = LoadBalancerStats(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=avg_response_time,
            requests_per_second=rps,
            active_connections=sum(self.connection_counts.values()),
            endpoint_distribution=dict(endpoint_distribution),
            health_status=health_status,
            last_updated=current_time,
        )

    @asynccontextmanager
    async def request_context(
        self,
        request_type: RequestType = RequestType.HTTP_API,
        tags: Optional[List[str]] = None,
    ):
        """Context manager for request routing"""
        endpoint = await self.route_request(request_type, tags)

        if not endpoint:
            raise RuntimeError("No available endpoints")

        request_id = await self.record_request_start(endpoint.endpoint_id, request_type)

        try:
            yield endpoint
            await self.record_request_end(request_id, success=True)
        except Exception as e:
            await self.record_request_end(
                request_id, success=False, error_message=str(e)
            )
            raise

    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        stats_dict = asdict(self.stats)

        # Add endpoint details
        stats_dict["endpoints"] = {}
        for endpoint_id, endpoint in self.endpoints.items():
            health = self.health_checker.get_health_status(endpoint_id)
            cb_state = self.circuit_breakers[endpoint_id].get_state()

            stats_dict["endpoints"][endpoint_id] = {
                "name": endpoint.name,
                "address": f"{endpoint.address}:{endpoint.port}",
                "weight": endpoint.weight,
                "health": asdict(health) if health else None,
                "circuit_breaker": cb_state,
                "active_connections": self.connection_counts[endpoint_id],
                "recent_response_times": list(self.response_times[endpoint_id])[
                    -10:
                ],  # Last 10
            }

        # Add routing configuration
        stats_dict["configuration"] = {
            "strategy": self.strategy.value,
            "total_endpoints": len(self.endpoints),
            "healthy_endpoints": len(
                self.health_checker.get_healthy_endpoints(list(self.endpoints.keys()))
            ),
        }

        return stats_dict


# Demo and testing functions
async def demo_load_balancing():
    """Demonstrate load balancing capabilities"""

    print("Personal Load Balancing Demo")

    # Create load balancer
    router = LoadBalancingRouter(RoutingStrategy.INTELLIGENT)

    # Add service endpoints
    endpoints = [
        ServiceEndpoint(
            endpoint_id="api-1",
            name="API Server 1",
            address="127.0.0.1",
            port=8001,
            weight=2,
            max_connections=50,
            tags=["api", "primary"],
        ),
        ServiceEndpoint(
            endpoint_id="api-2",
            name="API Server 2",
            address="127.0.0.1",
            port=8002,
            weight=1,
            max_connections=30,
            tags=["api", "secondary"],
        ),
        ServiceEndpoint(
            endpoint_id="compute-1",
            name="Compute Server 1",
            address="127.0.0.1",
            port=8003,
            weight=3,
            max_connections=20,
            tags=["compute", "gpu"],
        ),
    ]

    for endpoint in endpoints:
        router.add_endpoint(endpoint)

    print(f"Added {len(endpoints)} endpoints")

    # Start load balancer
    await router.start()

    # Simulate requests
    print("\nSimulating requests...")

    request_types = [RequestType.HTTP_API, RequestType.COMPUTE, RequestType.DATABASE]

    for i in range(50):
        request_type = random.choice(request_types)
        tags = ["api"] if request_type == RequestType.HTTP_API else ["compute"]

        try:
            async with router.request_context(request_type, tags) as endpoint:
                # Simulate request processing
                processing_time = random.uniform(0.01, 0.5)
                await asyncio.sleep(processing_time)

                if i % 10 == 0:
                    print(f"Request {i+1}: Routed to {endpoint.name}")

        except Exception as e:
            logger.error(f"Request {i+1} failed: {e}")

    # Get statistics
    stats = router.get_load_balancer_stats()
    print(f"\nLoad Balancer Statistics:")
    print(f"- Total requests: {stats['total_requests']}")
    print(f"- Success rate: {stats['successful_requests']}/{stats['total_requests']}")
    print(f"- Average response time: {stats['average_response_time_ms']:.2f}ms")
    print(f"- Requests per second: {stats['requests_per_second']:.1f}")
    print(f"- Active connections: {stats['active_connections']}")

    print(f"\nEndpoint Distribution:")
    for endpoint_id, count in stats["endpoint_distribution"].items():
        endpoint_name = router.endpoints[endpoint_id].name
        percentage = (
            (count / stats["total_requests"]) * 100
            if stats["total_requests"] > 0
            else 0
        )
        print(f"- {endpoint_name}: {count} requests ({percentage:.1f}%)")

    print(f"\nEndpoint Health:")
    for endpoint_id, endpoint_info in stats["endpoints"].items():
        health_status = (
            endpoint_info["health"]["status"] if endpoint_info["health"] else "unknown"
        )
        cb_state = endpoint_info["circuit_breaker"]["state"]
        print(f"- {endpoint_info['name']}: {health_status} (CB: {cb_state})")

    # Stop load balancer
    await router.stop()

    return router


if __name__ == "__main__":
    asyncio.run(demo_load_balancing())
