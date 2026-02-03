"""Intelligent Edge Network for Opifex production optimization.

This module implements global distribution with sub-millisecond latency optimization,
edge caching, and regional failover for the Phase 7.4 Production Optimization system.

Part of: Hybrid Performance Platform + Intelligent Edge + Adaptive Optimization
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class EdgeRegion(Enum):
    """Global edge regions for intelligent distribution."""

    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_PACIFIC = "asia-pacific"
    ASIA_NORTHEAST = "asia-northeast"
    SOUTH_AMERICA = "south-america"
    AFRICA = "africa"
    OCEANIA = "oceania"
    CUSTOM = "custom"


class FailoverStrategy(Enum):
    """Failover strategies for edge network."""

    NEAREST_REGION = "nearest_region"
    LOWEST_LATENCY = "lowest_latency"
    HIGHEST_CAPACITY = "highest_capacity"
    ROUND_ROBIN = "round_robin"
    WEIGHTED_DISTRIBUTION = "weighted_distribution"


@dataclass
class EdgeNodeMetrics:
    """Performance metrics for edge nodes."""

    node_id: str
    region: EdgeRegion
    latency_ms: float
    throughput_rps: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    error_rate: float
    cache_hit_ratio: float
    bandwidth_mbps: float
    concurrent_connections: int
    health_score: float
    last_update: float = field(default_factory=time.time)


@dataclass
class LatencyProfile:
    """Latency optimization profile for different workload types."""

    workload_type: str  # "inference", "training", "simulation", "analysis"
    target_latency_ms: float
    latency_sla_ms: float
    latency_percentile: float  # P99, P95, etc.
    optimization_priority: float  # 0-1 scale
    geographic_distribution: dict[EdgeRegion, float]  # Weight by region


@dataclass
class CacheEntry:
    """Cache entry for edge model and result caching."""

    cache_key: str
    model_hash: str
    input_hash: str
    cached_result: jnp.ndarray
    creation_time: float
    last_access_time: float
    access_count: int
    expiry_time: float
    size_bytes: int
    compression_ratio: float = 1.0


@dataclass
class FailoverResult:
    """Result of failover operation."""

    original_region: EdgeRegion
    failover_region: EdgeRegion
    failover_reason: str
    latency_impact_ms: float
    success: bool
    failover_time_ms: float


class EdgeGateway(nnx.Module):
    """Intelligent edge gateway for global request distribution."""

    def __init__(
        self,
        primary_regions: list[EdgeRegion] | None = None,
        latency_target_ms: float = 0.5,  # Sub-millisecond target
        max_failover_attempts: int = 3,
    ):
        super().__init__()
        self.primary_regions = primary_regions or [
            EdgeRegion.US_EAST,
            EdgeRegion.EU_WEST,
            EdgeRegion.ASIA_PACIFIC,
        ]
        self.latency_target_ms = latency_target_ms
        self.max_failover_attempts = max_failover_attempts

        # Edge node registry and metrics
        self.edge_nodes: dict[str, EdgeNodeMetrics] = {}
        self.regional_capacity: dict[EdgeRegion, float] = {}
        self.latency_matrix: dict[tuple[EdgeRegion, EdgeRegion], float] = {}

    def register_edge_node(
        self, node_id: str, region: EdgeRegion, initial_metrics: EdgeNodeMetrics
    ) -> bool:
        """Register a new edge node with the gateway."""
        self.edge_nodes[node_id] = initial_metrics
        if region not in self.regional_capacity:
            self.regional_capacity[region] = 0.0
        self.regional_capacity[region] += initial_metrics.throughput_rps
        return True

    def update_node_metrics(self, node_id: str, metrics: EdgeNodeMetrics) -> bool:
        """Update metrics for an existing edge node."""
        if node_id in self.edge_nodes:
            old_metrics = self.edge_nodes[node_id]
            self.edge_nodes[node_id] = metrics

            # Update regional capacity
            region = metrics.region
            if region in self.regional_capacity:
                capacity_delta = metrics.throughput_rps - old_metrics.throughput_rps
                self.regional_capacity[region] += capacity_delta

            return True
        return False

    def select_optimal_region(
        self,
        client_region: EdgeRegion,
        workload_profile: LatencyProfile,
        required_capacity: float,
    ) -> EdgeRegion:
        """Select optimal edge region for request routing."""

        candidate_regions = []

        # Filter regions by capacity
        for region, capacity in self.regional_capacity.items():
            if capacity >= required_capacity:
                candidate_regions.append(region)

        if not candidate_regions:
            # Fallback to primary regions if no capacity available
            candidate_regions = self.primary_regions

        # Score regions based on latency and workload profile
        region_scores = {}
        for region in candidate_regions:
            score = self._calculate_region_score(
                client_region, region, workload_profile
            )
            region_scores[region] = score

        # Select region with highest score
        return max(region_scores, key=lambda x: region_scores[x])

    def _calculate_region_score(
        self,
        client_region: EdgeRegion,
        candidate_region: EdgeRegion,
        workload_profile: LatencyProfile,
    ) -> float:
        """Calculate scoring for region selection."""

        # Base score from geographic distribution preferences
        geographic_score = workload_profile.geographic_distribution.get(
            candidate_region, 0.5
        )

        # Latency score
        latency_key = (client_region, candidate_region)
        base_latency = self.latency_matrix.get(latency_key, 50.0)  # Default 50ms
        latency_score = max(
            0.0, 1.0 - (base_latency / workload_profile.target_latency_ms)
        )

        # Capacity score
        region_nodes = [
            node for node in self.edge_nodes.values() if node.region == candidate_region
        ]
        if region_nodes:
            avg_utilization = sum(node.cpu_utilization for node in region_nodes) / len(
                region_nodes
            )
            capacity_score = max(0.0, 1.0 - avg_utilization)
        else:
            capacity_score = 0.0

        # Health score
        if region_nodes:
            avg_health = sum(node.health_score for node in region_nodes) / len(
                region_nodes
            )
            health_score = avg_health
        else:
            health_score = 0.0

        # Weighted final score
        return (
            0.4 * latency_score
            + 0.3 * capacity_score
            + 0.2 * health_score
            + 0.1 * geographic_score
        )


class LatencyOptimizer(nnx.Module):
    """Sub-millisecond latency optimizer for edge distribution."""

    def __init__(
        self,
        target_latency_ms: float = 0.5,
        optimization_window_seconds: int = 60,
        learning_rate: float = 0.001,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.target_latency_ms = target_latency_ms
        self.optimization_window_seconds = optimization_window_seconds

        # Neural network for latency prediction and optimization
        self.latency_predictor = nnx.Sequential(
            nnx.Linear(16, 64, rngs=rngs),  # Input: region, load, network metrics
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.gelu,
            nnx.Linear(32, 1, rngs=rngs),  # Output: predicted latency
        )

        # Route optimization network
        self.route_optimizer = nnx.Sequential(
            nnx.Linear(20, 128, rngs=rngs),  # Input: route characteristics
            nnx.gelu,
            nnx.Linear(128, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 10, rngs=rngs),  # Output: route selection probabilities
        )

    def predict_route_latency(self, route_features: jnp.ndarray) -> jnp.ndarray:
        """Predict latency for given route configuration."""
        return self.latency_predictor(route_features)

    def optimize_route_selection(self, available_routes: jnp.ndarray) -> jnp.ndarray:
        """Optimize route selection for minimal latency."""
        route_probabilities = self.route_optimizer(available_routes)
        return jax.nn.softmax(route_probabilities)

    def adaptive_latency_optimization(
        self,
        current_latencies: jnp.ndarray,
        target_latency: float,
        route_options: jnp.ndarray,
    ) -> dict[str, Any]:
        """Perform adaptive latency optimization."""

        # Predict latencies for all route options
        predicted_latencies = jax.vmap(self.predict_route_latency)(route_options)

        # Find routes that meet target latency
        viable_routes = predicted_latencies <= target_latency

        if jnp.any(viable_routes):
            # Select optimal route among viable options
            viable_indices = jnp.where(viable_routes)[0]
            optimal_route_idx = viable_indices[
                jnp.argmin(predicted_latencies[viable_indices])
            ]
        else:
            # Fallback: select route with minimum predicted latency
            optimal_route_idx = jnp.argmin(predicted_latencies)

        return {
            "optimal_route_index": int(optimal_route_idx),
            "predicted_latency_ms": float(predicted_latencies[optimal_route_idx]),
            "meets_target": bool(
                predicted_latencies[optimal_route_idx] <= target_latency
            ),
            "latency_improvement": float(
                jnp.mean(current_latencies) - predicted_latencies[optimal_route_idx]
            ),
        }


class EdgeCache:
    """High-performance edge cache for models and results."""

    def __init__(
        self,
        max_cache_size_gb: float = 10.0,
        max_entries: int = 10000,
        ttl_seconds: int = 3600,  # 1 hour default TTL
        compression_threshold_mb: float = 100.0,
    ):
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.compression_threshold_bytes = int(compression_threshold_mb * 1024 * 1024)

        self.cache: dict[str, CacheEntry] = {}
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0

    def generate_cache_key(
        self, model_hash: str, input_hash: str, parameters: dict[str, Any] | None = None
    ) -> str:
        """Generate cache key for model inference."""
        param_hash = hash(str(sorted((parameters or {}).items())))
        return f"{model_hash}:{input_hash}:{param_hash}"

    def put(
        self,
        model_hash: str,
        input_hash: str,
        result: jnp.ndarray,
        parameters: dict[str, Any] | None = None,
    ) -> bool:
        """Store result in cache."""
        cache_key = self.generate_cache_key(model_hash, input_hash, parameters)

        # Calculate size
        result_size = result.nbytes
        current_time = time.time()

        # Check if compression is needed
        compression_ratio = 1.0
        if result_size > self.compression_threshold_bytes:
            # Simulate compression (in practice, would use actual compression)
            compression_ratio = 0.6  # Assume 40% compression
            result_size = int(result_size * compression_ratio)

        # Evict if necessary
        while (
            self.current_size_bytes + result_size > self.max_cache_size_bytes
            or len(self.cache) >= self.max_entries
        ) and self.cache:
            self._evict_lru()

        # Create cache entry
        entry = CacheEntry(
            cache_key=cache_key,
            model_hash=model_hash,
            input_hash=input_hash,
            cached_result=result,
            creation_time=current_time,
            last_access_time=current_time,
            access_count=1,
            expiry_time=current_time + self.ttl_seconds,
            size_bytes=result_size,
            compression_ratio=compression_ratio,
        )

        self.cache[cache_key] = entry
        self.current_size_bytes += result_size
        return True

    def get(
        self,
        model_hash: str,
        input_hash: str,
        parameters: dict[str, Any] | None = None,
    ) -> jnp.ndarray | None:
        """Retrieve result from cache."""
        cache_key = self.generate_cache_key(model_hash, input_hash, parameters)

        if cache_key in self.cache:
            entry = self.cache[cache_key]
            current_time = time.time()

            # Check if entry has expired
            if current_time > entry.expiry_time:
                self._remove_entry(cache_key)
                self.miss_count += 1
                return None

            # Update access statistics
            entry.last_access_time = current_time
            entry.access_count += 1
            self.hit_count += 1

            return entry.cached_result

        self.miss_count += 1
        return None

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return

        # Find LRU entry
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k].last_access_time,
        )

        self._remove_entry(lru_key)
        self.eviction_count += 1

    def _remove_entry(self, cache_key: str) -> None:
        """Remove entry from cache."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            self.current_size_bytes -= entry.size_bytes
            del self.cache[cache_key]

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0.0

        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_ratio": hit_ratio,
            "eviction_count": self.eviction_count,
            "current_entries": len(self.cache),
            "current_size_gb": self.current_size_bytes / (1024 * 1024 * 1024),
            "utilization": self.current_size_bytes / self.max_cache_size_bytes,
        }


class RegionalFailover:
    """Regional failover system for edge network resilience."""

    def __init__(
        self,
        edge_gateway: EdgeGateway,
        failover_strategy: FailoverStrategy = FailoverStrategy.LOWEST_LATENCY,
        health_check_interval: float = 10.0,
        failover_threshold: float = 0.5,  # Health score threshold
    ):
        self.edge_gateway = edge_gateway
        self.failover_strategy = failover_strategy
        self.health_check_interval = health_check_interval
        self.failover_threshold = failover_threshold

        self.region_health: dict[EdgeRegion, float] = {}
        self.failover_history: list[FailoverResult] = []
        self.is_monitoring = False

    async def start_health_monitoring(self) -> None:
        """Start continuous health monitoring for failover detection."""
        self.is_monitoring = True

        while self.is_monitoring:
            await self._update_region_health()
            await self._check_failover_triggers()
            await asyncio.sleep(self.health_check_interval)

    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        self.is_monitoring = False

    async def _update_region_health(self) -> None:
        """Update health scores for all regions."""
        region_health = {}

        for region in EdgeRegion:
            region_nodes = [
                node
                for node in self.edge_gateway.edge_nodes.values()
                if node.region == region
            ]

            if region_nodes:
                # Calculate average health score
                avg_health = sum(node.health_score for node in region_nodes) / len(
                    region_nodes
                )
                region_health[region] = avg_health
            else:
                region_health[region] = 0.0

        self.region_health = region_health

    async def _check_failover_triggers(self) -> None:
        """Check if any region needs failover."""
        for region, health in self.region_health.items():
            if health < self.failover_threshold:
                await self._trigger_failover(
                    region, f"Health score {health:.2f} below threshold"
                )

    async def _trigger_failover(
        self, failing_region: EdgeRegion, reason: str
    ) -> FailoverResult:
        """Trigger failover from failing region to healthy alternative."""
        start_time = time.time()

        # Find best failover target
        failover_target = self._select_failover_target(failing_region)

        if failover_target is None:
            return FailoverResult(
                original_region=failing_region,
                failover_region=failing_region,  # No change
                failover_reason=f"{reason} - No viable failover target",
                latency_impact_ms=0.0,
                success=False,
                failover_time_ms=(time.time() - start_time) * 1000,
            )

        # Calculate latency impact
        latency_impact = self._calculate_latency_impact(failing_region, failover_target)

        # Execute failover (would trigger actual infrastructure changes)
        success = await self._execute_failover(failing_region, failover_target)

        failover_time_ms = (time.time() - start_time) * 1000

        result = FailoverResult(
            original_region=failing_region,
            failover_region=failover_target,
            failover_reason=reason,
            latency_impact_ms=latency_impact,
            success=success,
            failover_time_ms=failover_time_ms,
        )

        self.failover_history.append(result)
        return result

    def _select_failover_target(self, failing_region: EdgeRegion) -> EdgeRegion | None:
        """Select best region for failover based on strategy."""
        healthy_regions = [
            region
            for region, health in self.region_health.items()
            if health >= self.failover_threshold and region != failing_region
        ]

        if not healthy_regions:
            return None

        if self.failover_strategy == FailoverStrategy.NEAREST_REGION:
            # Select geographically nearest region (simplified)
            return healthy_regions[0]  # Would use actual geographic distance

        if self.failover_strategy == FailoverStrategy.LOWEST_LATENCY:
            # Select region with lowest latency
            latency_scores = {}
            for region in healthy_regions:
                latency_key = (failing_region, region)
                latency = self.edge_gateway.latency_matrix.get(latency_key, 1000.0)
                latency_scores[region] = latency

            return min(latency_scores, key=lambda x: latency_scores[x])

        if self.failover_strategy == FailoverStrategy.HIGHEST_CAPACITY:
            # Select region with highest available capacity
            return max(
                healthy_regions,
                key=lambda r: self.edge_gateway.regional_capacity.get(r, 0.0),
            )

        # Round robin or weighted distribution
        return healthy_regions[0]

    def _calculate_latency_impact(
        self, original_region: EdgeRegion, failover_region: EdgeRegion
    ) -> float:
        """Calculate expected latency impact of failover."""
        latency_key = (original_region, failover_region)
        return self.edge_gateway.latency_matrix.get(latency_key, 0.0)

    async def _execute_failover(
        self, from_region: EdgeRegion, to_region: EdgeRegion
    ) -> bool:
        """Execute the actual failover operation."""
        # In practice, this would:
        # 1. Update load balancer configurations
        # 2. Migrate active connections
        # 3. Update DNS records
        # 4. Notify monitoring systems
        # 5. Update service mesh routing

        # Simulate execution delay
        await asyncio.sleep(0.1)
        return True


class IntelligentEdgeNetwork:
    """Main orchestrator for intelligent edge network with global distribution."""

    def __init__(
        self,
        edge_gateway: EdgeGateway,
        latency_optimizer: LatencyOptimizer,
        edge_cache: EdgeCache,
        regional_failover: RegionalFailover,
        target_latency_ms: float = 0.5,
    ):
        self.edge_gateway = edge_gateway
        self.latency_optimizer = latency_optimizer
        self.edge_cache = edge_cache
        self.regional_failover = regional_failover
        self.target_latency_ms = target_latency_ms

        self.request_count = 0
        self.total_latency_ms = 0.0
        self.cache_enabled = True

    async def process_inference_request(
        self,
        model_hash: str,
        input_data: jnp.ndarray,
        client_region: EdgeRegion,
        workload_profile: LatencyProfile,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process inference request with intelligent edge routing and caching."""
        start_time = time.time()
        input_hash = str(hash(input_data.tobytes()))

        # Try cache first
        cached_result = None
        if self.cache_enabled:
            cached_result = self.edge_cache.get(model_hash, input_hash, parameters)

        if cached_result is not None:
            cache_latency_ms = (time.time() - start_time) * 1000
            return {
                "result": cached_result,
                "latency_ms": cache_latency_ms,
                "cache_hit": True,
                "edge_region": client_region,
                "optimization_applied": False,
            }

        # Select optimal edge region
        required_capacity = workload_profile.optimization_priority * 100  # Simplified
        optimal_region = self.edge_gateway.select_optimal_region(
            client_region, workload_profile, required_capacity
        )

        # Simulate model inference (in practice, would route to actual edge node)
        inference_latency_ms = await self._simulate_edge_inference(
            optimal_region, input_data, workload_profile
        )

        # Generate simulated result for caching
        result_shape = (*input_data.shape[:-1], 1)  # Typical output shape
        simulated_result = jax.random.normal(jax.random.PRNGKey(42), result_shape)

        # Cache result if enabled
        if self.cache_enabled:
            self.edge_cache.put(model_hash, input_hash, simulated_result, parameters)

        total_latency_ms = (time.time() - start_time) * 1000
        self.request_count += 1
        self.total_latency_ms += total_latency_ms

        return {
            "result": simulated_result,
            "latency_ms": total_latency_ms,
            "inference_latency_ms": inference_latency_ms,
            "cache_hit": False,
            "edge_region": optimal_region,
            "optimization_applied": True,
            "meets_target": total_latency_ms <= self.target_latency_ms,
        }

    async def _simulate_edge_inference(
        self,
        region: EdgeRegion,
        input_data: jnp.ndarray,
        workload_profile: LatencyProfile,
    ) -> float:
        """Simulate inference latency for edge region."""
        # Base latency by region (simplified)
        base_latencies = {
            EdgeRegion.US_EAST: 0.3,
            EdgeRegion.US_WEST: 0.4,
            EdgeRegion.EU_WEST: 0.5,
            EdgeRegion.EU_CENTRAL: 0.4,
            EdgeRegion.ASIA_PACIFIC: 0.6,
            EdgeRegion.ASIA_NORTHEAST: 0.5,
        }

        base_latency = base_latencies.get(region, 1.0)

        # Add workload complexity factor
        complexity_factor = {
            "inference": 1.0,
            "training": 3.0,
            "simulation": 2.0,
            "analysis": 1.5,
        }.get(workload_profile.workload_type, 1.0)

        # Simulate processing delay
        processing_time = base_latency * complexity_factor
        await asyncio.sleep(processing_time / 1000)  # Convert to seconds

        return processing_time

    def get_network_statistics(self) -> dict[str, Any]:
        """Get comprehensive network performance statistics."""
        avg_latency = (
            self.total_latency_ms / self.request_count
            if self.request_count > 0
            else 0.0
        )

        cache_stats = self.edge_cache.get_cache_statistics()

        return {
            "request_count": self.request_count,
            "average_latency_ms": avg_latency,
            "target_latency_ms": self.target_latency_ms,
            "meets_target_percentage": (
                100.0 * (avg_latency <= self.target_latency_ms)
                if self.request_count > 0
                else 0.0
            ),
            "cache_statistics": cache_stats,
            "regional_capacity": dict(self.edge_gateway.regional_capacity),
            "region_health": dict(self.regional_failover.region_health),
            "failover_count": len(self.regional_failover.failover_history),
        }

    async def start_edge_services(self) -> None:
        """Start all edge network services."""
        await self.regional_failover.start_health_monitoring()

    async def stop_edge_services(self) -> None:
        """Stop all edge network services."""
        await self.regional_failover.stop_health_monitoring()
