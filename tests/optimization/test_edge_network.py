"""Comprehensive tests for Edge Network production optimization.

This test suite provides enterprise-grade testing for the intelligent edge network
using pytest, pytest-mock, pytest-asyncio, and pytest-benchmark.

Coverage Enhancement: 34% → 80%+ target
Enterprise Testing Strategy: Using existing robust libraries and industry patterns
"""

from unittest.mock import AsyncMock, MagicMock, patch

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.edge_network import (
    EdgeCache,
    EdgeGateway,
    EdgeNodeMetrics,
    EdgeRegion,
    FailoverResult,
    FailoverStrategy,
    IntelligentEdgeNetwork,
    LatencyOptimizer,
    LatencyProfile,
    RegionalFailover,
)


class TestEdgeGateway:
    """Test suite for EdgeGateway with comprehensive coverage."""

    def test_edge_gateway_initialization(self):
        """Test EdgeGateway initialization with default and custom parameters."""
        # Test default initialization (ASIA_PACIFIC is included by default)
        gateway = EdgeGateway()
        assert EdgeRegion.US_EAST in gateway.primary_regions
        assert EdgeRegion.EU_WEST in gateway.primary_regions
        assert EdgeRegion.ASIA_PACIFIC in gateway.primary_regions
        assert gateway.latency_target_ms == 0.5
        assert gateway.max_failover_attempts == 3
        assert len(gateway.edge_nodes) == 0

        # Test custom initialization
        custom_regions = [EdgeRegion.ASIA_PACIFIC, EdgeRegion.US_WEST]
        gateway_custom = EdgeGateway(
            primary_regions=custom_regions,
            latency_target_ms=1.0,
            max_failover_attempts=5,
        )
        assert gateway_custom.primary_regions == custom_regions
        assert gateway_custom.latency_target_ms == 1.0
        assert gateway_custom.max_failover_attempts == 5

    def test_register_edge_node_success(self):
        """Test successful edge node registration."""
        gateway = EdgeGateway()
        metrics = EdgeNodeMetrics(
            node_id="node_1",
            region=EdgeRegion.US_EAST,
            latency_ms=0.3,
            throughput_rps=1000.0,
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )

        result = gateway.register_edge_node("node_1", EdgeRegion.US_EAST, metrics)
        assert result is True
        assert "node_1" in gateway.edge_nodes
        assert gateway.edge_nodes["node_1"] == metrics
        assert EdgeRegion.US_EAST in gateway.regional_capacity
        assert gateway.regional_capacity[EdgeRegion.US_EAST] == 1000.0

    def test_register_multiple_nodes_same_region(self):
        """Test registering multiple nodes in the same region."""
        gateway = EdgeGateway()
        metrics1 = EdgeNodeMetrics(
            node_id="node_1",
            region=EdgeRegion.US_EAST,
            latency_ms=0.3,
            throughput_rps=1000.0,
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )

        metrics2 = EdgeNodeMetrics(
            node_id="node_2",
            region=EdgeRegion.US_EAST,
            latency_ms=0.4,
            throughput_rps=800.0,
            cpu_utilization=0.7,
            memory_utilization=0.6,
            gpu_utilization=0.7,
            error_rate=0.02,
            cache_hit_ratio=0.90,
            bandwidth_mbps=800.0,
            concurrent_connections=80,
            health_score=0.90,
        )

        gateway.register_edge_node("node_1", EdgeRegion.US_EAST, metrics1)
        gateway.register_edge_node("node_2", EdgeRegion.US_EAST, metrics2)

        # Regional capacity should be sum of both nodes
        assert gateway.regional_capacity[EdgeRegion.US_EAST] == 1800.0

    def test_update_node_metrics_existing(self):
        """Test updating metrics for existing node."""
        gateway = EdgeGateway()
        initial_metrics = EdgeNodeMetrics(
            node_id="node_1",
            region=EdgeRegion.US_EAST,
            latency_ms=0.3,
            throughput_rps=1000.0,
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )

        gateway.register_edge_node("node_1", EdgeRegion.US_EAST, initial_metrics)

        updated_metrics = EdgeNodeMetrics(
            node_id="node_1",
            region=EdgeRegion.US_EAST,
            latency_ms=0.4,
            throughput_rps=1200.0,
            cpu_utilization=0.7,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )

        result = gateway.update_node_metrics("node_1", updated_metrics)
        assert result is True
        assert gateway.edge_nodes["node_1"].latency_ms == 0.4
        assert gateway.edge_nodes["node_1"].throughput_rps == 1200.0
        # Regional capacity should be updated too (1000 -> 1200)
        assert gateway.regional_capacity[EdgeRegion.US_EAST] == 1200.0

    def test_update_node_metrics_nonexistent(self):
        """Test updating metrics for non-existent node."""
        gateway = EdgeGateway()
        metrics = EdgeNodeMetrics(
            node_id="nonexistent",
            region=EdgeRegion.US_EAST,
            latency_ms=0.3,
            throughput_rps=1000.0,
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )

        result = gateway.update_node_metrics("nonexistent", metrics)
        assert result is False

    def test_select_optimal_region_basic(self):
        """Test basic optimal region selection."""
        gateway = EdgeGateway()

        # Register nodes
        us_metrics = EdgeNodeMetrics(
            node_id="us_node",
            region=EdgeRegion.US_EAST,
            latency_ms=0.3,
            throughput_rps=1000.0,
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )

        gateway.register_edge_node("us_node", EdgeRegion.US_EAST, us_metrics)

        workload_profile = LatencyProfile(
            workload_type="inference",
            target_latency_ms=0.5,
            latency_sla_ms=1.0,
            latency_percentile=95.0,
            optimization_priority=0.8,
            geographic_distribution={EdgeRegion.US_EAST: 0.7, EdgeRegion.EU_WEST: 0.3},
        )

        optimal_region = gateway.select_optimal_region(
            EdgeRegion.US_EAST, workload_profile, 100.0
        )

        # Should return a valid region
        assert isinstance(optimal_region, EdgeRegion)

    def test_select_optimal_region_no_nodes(self):
        """Test optimal region selection with no registered nodes."""
        gateway = EdgeGateway()
        workload_profile = LatencyProfile(
            workload_type="inference",
            target_latency_ms=0.5,
            latency_sla_ms=1.0,
            latency_percentile=95.0,
            optimization_priority=0.8,
            geographic_distribution={EdgeRegion.US_EAST: 1.0},
        )

        optimal_region = gateway.select_optimal_region(
            EdgeRegion.US_EAST, workload_profile, 100.0
        )

        # Should return default primary region
        assert optimal_region in gateway.primary_regions


class TestEdgeCache:
    """Test suite for EdgeCache with comprehensive coverage."""

    def test_edge_cache_initialization(self):
        """Test EdgeCache initialization with different configurations."""
        # Test default initialization
        cache = EdgeCache()
        assert cache.max_cache_size_bytes == int(
            10.0 * 1024 * 1024 * 1024
        )  # 10GB in bytes
        assert cache.max_entries == 10000
        assert cache.ttl_seconds == 3600
        assert len(cache.cache) == 0
        assert cache.current_size_bytes == 0

        # Test custom initialization
        cache_custom = EdgeCache(
            max_cache_size_gb=20.0,
            max_entries=5000,
            ttl_seconds=7200,
            compression_threshold_mb=50.0,
        )
        assert cache_custom.max_cache_size_bytes == int(20.0 * 1024 * 1024 * 1024)
        assert cache_custom.max_entries == 5000
        assert cache_custom.ttl_seconds == 7200

    def test_generate_cache_key(self):
        """Test cache key generation."""
        cache = EdgeCache()

        # Test basic key generation
        key1 = cache.generate_cache_key("model_hash_123", "input_hash_456")
        assert isinstance(key1, str)
        assert "model_hash_123" in key1
        assert "input_hash_456" in key1

        # Test key generation with parameters
        parameters = {"temperature": 0.8, "max_length": 100}
        key2 = cache.generate_cache_key("model_hash_123", "input_hash_456", parameters)
        assert isinstance(key2, str)
        assert key1 != key2  # Should be different with parameters

        # Test consistent key generation
        key3 = cache.generate_cache_key("model_hash_123", "input_hash_456", parameters)
        assert key2 == key3  # Should be same with same inputs

    def test_cache_put_and_get_basic(self):
        """Test basic cache put and get operations."""
        cache = EdgeCache()

        model_hash = "model_123"
        input_hash = "input_456"
        result = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Put data in cache
        put_success = cache.put(model_hash, input_hash, result)
        assert put_success is True

        # Get data from cache
        cached_result = cache.get(model_hash, input_hash)
        assert cached_result is not None
        assert jnp.allclose(cached_result, result)

        # Verify cache statistics updated
        stats = cache.get_cache_statistics()
        assert stats["current_entries"] == 1
        assert stats["hit_count"] == 1

    def test_cache_get_nonexistent(self):
        """Test getting non-existent data from cache."""
        cache = EdgeCache()

        # Try to get non-existent data
        result = cache.get("nonexistent_model", "nonexistent_input")
        assert result is None

        # Verify miss is recorded in statistics
        stats = cache.get_cache_statistics()
        assert stats["miss_count"] == 1

    def test_cache_put_with_parameters(self):
        """Test cache operations with parameters."""
        cache = EdgeCache()

        model_hash = "model_123"
        input_hash = "input_456"
        result = jnp.array([1.0, 2.0, 3.0, 4.0])
        parameters = {"temperature": 0.8}

        # Put with parameters
        put_success = cache.put(model_hash, input_hash, result, parameters)
        assert put_success is True

        # Get with same parameters - should hit
        cached_result = cache.get(model_hash, input_hash, parameters)
        assert cached_result is not None
        assert jnp.allclose(cached_result, result)

        # Get without parameters - should miss
        cached_result_no_params = cache.get(model_hash, input_hash)
        assert cached_result_no_params is None

    @patch("time.time")
    def test_cache_expiry(self, mock_time):
        """Test cache entry expiry functionality."""
        cache = EdgeCache(ttl_seconds=10)  # 10 second TTL

        # Mock initial time
        mock_time.return_value = 1000.0

        model_hash = "model_123"
        input_hash = "input_456"
        result = jnp.array([1.0, 2.0, 3.0])

        # Put data in cache
        cache.put(model_hash, input_hash, result)

        # Should be available immediately
        cached_result = cache.get(model_hash, input_hash)
        assert cached_result is not None

        # Mock time after expiry
        mock_time.return_value = 1015.0  # 15 seconds later

        # Should be expired now
        cached_result_expired = cache.get(model_hash, input_hash)
        assert cached_result_expired is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when entry limit is reached."""
        # Small cache for testing eviction
        cache = EdgeCache(max_entries=2)

        result1 = jnp.array([1.0])
        result2 = jnp.array([2.0])
        result3 = jnp.array([3.0])

        # Put first entry
        cache.put("model_1", "input_1", result1)
        assert len(cache.cache) == 1

        # Put second entry
        cache.put("model_2", "input_2", result2)
        assert len(cache.cache) == 2

        # Access first entry to make it more recently used
        cache.get("model_1", "input_1")

        # Put third entry - should evict model_2 (LRU)
        cache.put("model_3", "input_3", result3)
        assert len(cache.cache) == 2

        # model_2 should be evicted, model_1 and model_3 should remain
        assert cache.get("model_2", "input_2") is None
        assert cache.get("model_1", "input_1") is not None
        assert cache.get("model_3", "input_3") is not None

    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        cache = EdgeCache()

        # Initial statistics
        stats = cache.get_cache_statistics()
        assert stats["current_entries"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["hit_ratio"] == 0.0

        # Add entry and test hits/misses
        result = jnp.array([1.0, 2.0, 3.0])
        cache.put("model_1", "input_1", result)

        # Hit
        cache.get("model_1", "input_1")

        # Miss
        cache.get("model_2", "input_2")

        # Check updated statistics
        stats = cache.get_cache_statistics()
        assert stats["current_entries"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_ratio"] == 0.5  # 50% hit rate

    def test_cache_compression_threshold(self):
        """Test cache compression behavior for large entries."""
        cache = EdgeCache(compression_threshold_mb=0.001)  # Very low threshold

        # Large array that should trigger compression
        large_result = jnp.ones((1000, 1000))  # Large array

        cache.put("model_large", "input_large", large_result)

        # Should still be able to retrieve
        cached_result = cache.get("model_large", "input_large")
        assert cached_result is not None
        assert jnp.allclose(cached_result, large_result)


class TestRegionalFailover:
    """Test suite for RegionalFailover with comprehensive coverage."""

    @pytest.fixture
    def mock_edge_gateway(self):
        """Create mock EdgeGateway for testing."""
        gateway = MagicMock(spec=EdgeGateway)
        gateway.edge_nodes = {
            "node_us": EdgeNodeMetrics(
                node_id="node_us",
                region=EdgeRegion.US_EAST,
                latency_ms=0.3,
                throughput_rps=1000.0,
                cpu_utilization=0.6,
                memory_utilization=0.7,
                gpu_utilization=0.8,
                error_rate=0.01,
                cache_hit_ratio=0.95,
                bandwidth_mbps=1000.0,
                concurrent_connections=100,
                health_score=0.95,
            ),
            "node_eu": EdgeNodeMetrics(
                node_id="node_eu",
                region=EdgeRegion.EU_WEST,
                latency_ms=0.5,
                throughput_rps=800.0,
                cpu_utilization=0.8,
                memory_utilization=0.6,
                gpu_utilization=0.7,
                error_rate=0.02,
                cache_hit_ratio=0.90,
                bandwidth_mbps=800.0,
                concurrent_connections=80,
                health_score=0.85,
            ),
        }
        # Mock the latency_matrix attribute
        gateway.latency_matrix = {
            (EdgeRegion.EU_WEST, EdgeRegion.US_EAST): 150.0,
            (EdgeRegion.US_EAST, EdgeRegion.EU_WEST): 150.0,
        }
        return gateway

    def test_regional_failover_initialization(self, mock_edge_gateway):
        """Test RegionalFailover initialization."""
        failover = RegionalFailover(
            edge_gateway=mock_edge_gateway,
            failover_strategy=FailoverStrategy.LOWEST_LATENCY,
            health_check_interval=10.0,
            failover_threshold=0.5,
        )

        assert failover.edge_gateway == mock_edge_gateway
        assert failover.failover_strategy == FailoverStrategy.LOWEST_LATENCY
        assert failover.health_check_interval == 10.0
        assert failover.failover_threshold == 0.5

    def test_select_failover_target_lowest_latency(self, mock_edge_gateway):
        """Test failover target selection with lowest latency strategy."""
        failover = RegionalFailover(
            edge_gateway=mock_edge_gateway,
            failover_strategy=FailoverStrategy.LOWEST_LATENCY,
        )

        # Set up region health and latency data
        failover.region_health = {
            EdgeRegion.US_EAST: 0.9,
            EdgeRegion.US_WEST: 0.8,
            EdgeRegion.EU_WEST: 0.3,  # Failing region
        }
        mock_edge_gateway.latency_matrix = {
            (EdgeRegion.EU_WEST, EdgeRegion.US_EAST): 50.0,
            (EdgeRegion.EU_WEST, EdgeRegion.US_WEST): 80.0,
        }

        # Should select US_EAST (lower latency than EU_WEST)
        target = failover._select_failover_target(EdgeRegion.EU_WEST)
        assert target == EdgeRegion.US_EAST

    def test_select_failover_target_highest_capacity(self, mock_edge_gateway):
        """Test failover target selection with highest capacity strategy."""
        failover = RegionalFailover(
            edge_gateway=mock_edge_gateway,
            failover_strategy=FailoverStrategy.HIGHEST_CAPACITY,
        )

        # Set up region health and capacity data
        failover.region_health = {
            EdgeRegion.US_EAST: 0.9,
            EdgeRegion.US_WEST: 0.8,
            EdgeRegion.EU_WEST: 0.3,  # Failing region
        }
        mock_edge_gateway.regional_capacity = {
            EdgeRegion.US_EAST: 1000.0,
            EdgeRegion.US_WEST: 800.0,
        }

        # Should select US_EAST (higher throughput)
        target = failover._select_failover_target(EdgeRegion.EU_WEST)
        assert target == EdgeRegion.US_EAST

    def test_calculate_latency_impact(self, mock_edge_gateway):
        """Test latency impact calculation."""
        failover = RegionalFailover(edge_gateway=mock_edge_gateway)

        impact = failover._calculate_latency_impact(
            EdgeRegion.EU_WEST, EdgeRegion.US_EAST
        )

        # Should match the mocked latency matrix
        assert impact == 150.0

    @pytest.mark.asyncio
    async def test_trigger_failover_success(self, mock_edge_gateway):
        """Test successful failover operation."""
        failover = RegionalFailover(
            edge_gateway=mock_edge_gateway,
            failover_strategy=FailoverStrategy.LOWEST_LATENCY,
        )

        # Set up region health and latency data for successful failover
        failover.region_health = {
            EdgeRegion.US_EAST: 0.9,
            EdgeRegion.US_WEST: 0.8,
            EdgeRegion.EU_WEST: 0.3,  # Failing region
        }
        mock_edge_gateway.latency_matrix = {
            (EdgeRegion.EU_WEST, EdgeRegion.US_EAST): 50.0,
            (EdgeRegion.EU_WEST, EdgeRegion.US_WEST): 80.0,
        }

        # Mock successful failover execution
        with patch.object(
            failover, "_execute_failover", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = True

            result = await failover._trigger_failover(
                EdgeRegion.EU_WEST, "High latency"
            )

            assert isinstance(result, FailoverResult)
            assert result.original_region == EdgeRegion.EU_WEST
            assert result.failover_region == EdgeRegion.US_EAST
            assert "High latency" in result.failover_reason
            assert result.success is True
            mock_execute.assert_called_once()


class TestIntelligentEdgeNetwork:
    """Test suite for IntelligentEdgeNetwork integration with comprehensive coverage."""

    @pytest.fixture
    def edge_network_components(self):
        """Create all components for IntelligentEdgeNetwork."""
        gateway = EdgeGateway()

        # Add test node
        us_metrics = EdgeNodeMetrics(
            node_id="us_node",
            region=EdgeRegion.US_EAST,
            latency_ms=0.3,
            throughput_rps=1000.0,
            cpu_utilization=0.6,
            memory_utilization=0.7,
            gpu_utilization=0.8,
            error_rate=0.01,
            cache_hit_ratio=0.95,
            bandwidth_mbps=1000.0,
            concurrent_connections=100,
            health_score=0.95,
        )
        gateway.register_edge_node("us_node", EdgeRegion.US_EAST, us_metrics)

        # Create other components without causing JAX dimension issues
        rngs = nnx.Rngs(42)
        latency_optimizer = LatencyOptimizer(rngs=rngs)
        edge_cache = EdgeCache()
        regional_failover = RegionalFailover(gateway)

        return gateway, latency_optimizer, edge_cache, regional_failover

    def test_intelligent_edge_network_initialization(self, edge_network_components):
        """Test IntelligentEdgeNetwork initialization."""
        gateway, optimizer, cache, failover = edge_network_components

        network = IntelligentEdgeNetwork(
            edge_gateway=gateway,
            latency_optimizer=optimizer,
            edge_cache=cache,
            regional_failover=failover,
            target_latency_ms=0.5,
        )

        assert network.edge_gateway == gateway
        assert network.latency_optimizer == optimizer
        assert network.edge_cache == cache
        assert network.regional_failover == failover
        assert network.target_latency_ms == 0.5

    def test_get_network_statistics(self, edge_network_components):
        """Test network statistics collection."""
        gateway, optimizer, cache, failover = edge_network_components
        network = IntelligentEdgeNetwork(gateway, optimizer, cache, failover)

        stats = network.get_network_statistics()

        assert isinstance(stats, dict)
        assert "cache_statistics" in stats
        assert "regional_capacity" in stats

        # Verify structure matches expected format
        assert isinstance(stats["cache_statistics"], dict)
        assert isinstance(stats["regional_capacity"], dict)


@pytest.mark.benchmark(group="edge_network_performance")
class TestEdgeNetworkPerformance:
    """Performance benchmarks for Edge Network components."""

    def test_cache_put_performance(self, benchmark):
        """Benchmark cache put operations."""
        cache = EdgeCache()
        result = jnp.ones((50, 50))  # Medium-sized array

        def cache_put_operation():
            return cache.put("benchmark_model", "benchmark_input", result)

        result = benchmark(cache_put_operation)
        assert result is True

    def test_cache_get_performance(self, benchmark):
        """Benchmark cache get operations."""
        cache = EdgeCache()
        result = jnp.ones((50, 50))
        cache.put("benchmark_model", "benchmark_input", result)

        def cache_get_operation():
            return cache.get("benchmark_model", "benchmark_input")

        cached_result = benchmark(cache_get_operation)
        assert cached_result is not None

    def test_region_selection_performance(self, benchmark):
        """Benchmark optimal region selection."""
        gateway = EdgeGateway()

        # Add multiple nodes for realistic benchmarking
        regions = [EdgeRegion.US_EAST, EdgeRegion.EU_WEST, EdgeRegion.ASIA_PACIFIC]
        for i, region in enumerate(regions):
            metrics = EdgeNodeMetrics(
                node_id=f"node_{i}",
                region=region,
                latency_ms=0.3 + i * 0.1,
                throughput_rps=1000.0 - i * 100,
                cpu_utilization=0.6 + i * 0.1,
                memory_utilization=0.7 - i * 0.05,
                gpu_utilization=0.8 - i * 0.05,
                error_rate=0.01 + i * 0.005,
                cache_hit_ratio=0.95 - i * 0.01,
                bandwidth_mbps=1000.0 - i * 50,
                concurrent_connections=100 - i * 10,
                health_score=0.95 - i * 0.02,
            )
            gateway.register_edge_node(f"node_{i}", region, metrics)

        workload_profile = LatencyProfile(
            workload_type="inference",
            target_latency_ms=0.5,
            latency_sla_ms=1.0,
            latency_percentile=95.0,
            optimization_priority=0.8,
            geographic_distribution={EdgeRegion.US_EAST: 0.4, EdgeRegion.EU_WEST: 0.6},
        )

        def region_selection():
            return gateway.select_optimal_region(
                EdgeRegion.US_EAST, workload_profile, 100.0
            )

        selected_region = benchmark(region_selection)
        assert isinstance(selected_region, EdgeRegion)
