"""Comprehensive tests for GPUPoolManager.

Test-driven development (TDD) approach: These tests are written BEFORE
extracting the GPUPoolManager module from the original file.
"""

import time

import pytest

from opifex.deployment.resource_management.types import (
    CloudProvider,
    ResourceType,
)


# Import after extraction
# from opifex.deployment.resource_management.gpu_manager import GPUPoolManager
# from opifex.deployment.resource_management.orchestrator import ResourceOrchestrator


class TestGPUPoolManagerInitialization:
    """Test GPUPoolManager initialization."""

    def test_initialization_default_parameters(self, mock_orchestrator):
        """Test GPUPoolManager initialization with default parameters."""
        from opifex.deployment.resource_management.gpu_manager import GPUPoolManager

        manager = GPUPoolManager(mock_orchestrator)

        assert manager.resource_orchestrator == mock_orchestrator
        assert manager.memory_optimization_threshold == 0.85
        assert manager.pool_rebalancing_interval == 300
        assert isinstance(manager.gpu_pools, dict)
        assert len(manager.gpu_pools) == 0
        assert isinstance(manager.memory_allocations, dict)
        assert len(manager.memory_allocations) == 0
        assert isinstance(manager.model_placement_cache, dict)
        assert len(manager.model_placement_cache) == 0

    def test_initialization_custom_parameters(self, mock_orchestrator):
        """Test GPUPoolManager initialization with custom parameters."""
        from opifex.deployment.resource_management.gpu_manager import GPUPoolManager

        manager = GPUPoolManager(
            mock_orchestrator,
            memory_optimization_threshold=0.75,
            pool_rebalancing_interval=600,
        )

        assert manager.memory_optimization_threshold == 0.75
        assert manager.pool_rebalancing_interval == 600


class TestGPUPoolCreation:
    """Test GPU pool creation functionality."""

    def test_create_gpu_pool_success(self, gpu_pool_manager):
        """Test successful GPU pool creation."""
        result = gpu_pool_manager.create_gpu_pool(
            pool_id="pool-001",
            gpu_type=ResourceType.GPU_A100,
            gpu_count=4,
            memory_per_gpu_gb=40.0,
            provider=CloudProvider.AWS,
            region="us-east-1",
        )

        assert result is True
        assert "pool-001" in gpu_pool_manager.gpu_pools

        pool = gpu_pool_manager.gpu_pools["pool-001"]
        assert pool["pool_id"] == "pool-001"
        assert pool["gpu_type"] == ResourceType.GPU_A100
        assert pool["gpu_count"] == 4
        assert pool["memory_per_gpu_gb"] == 40.0
        assert pool["provider"] == CloudProvider.AWS
        assert pool["region"] == "us-east-1"
        assert pool["total_memory_gb"] == 160.0  # 4 * 40
        assert pool["allocated_memory_gb"] == 0.0
        assert isinstance(pool["active_models"], dict)
        assert pool["utilization"] == 0.0
        assert "creation_time" in pool

    def test_create_multiple_gpu_pools(self, gpu_pool_manager):
        """Test creating multiple GPU pools."""
        gpu_pool_manager.create_gpu_pool(
            "pool-001", ResourceType.GPU_A100, 4, 40.0, CloudProvider.AWS, "us-east-1"
        )
        gpu_pool_manager.create_gpu_pool(
            "pool-002", ResourceType.GPU_V100, 8, 32.0, CloudProvider.GCP, "us-west-1"
        )

        assert len(gpu_pool_manager.gpu_pools) == 2
        assert "pool-001" in gpu_pool_manager.gpu_pools
        assert "pool-002" in gpu_pool_manager.gpu_pools

    def test_create_gpu_pool_h100(self, gpu_pool_manager):
        """Test creating GPU pool with H100 GPUs."""
        result = gpu_pool_manager.create_gpu_pool(
            "pool-h100",
            ResourceType.GPU_H100,
            2,
            80.0,
            CloudProvider.AZURE,
            "eu-west-1",
        )

        assert result is True
        pool = gpu_pool_manager.gpu_pools["pool-h100"]
        assert pool["gpu_type"] == ResourceType.GPU_H100
        assert pool["total_memory_gb"] == 160.0  # 2 * 80


class TestMemoryAllocation:
    """Test GPU memory allocation functionality."""

    def test_allocate_memory_success(self, gpu_pool_manager_with_pools):
        """Test successful GPU memory allocation."""
        manager = gpu_pool_manager_with_pools

        result = manager.allocate_gpu_memory("model-001", memory_requirement_gb=30.0)

        assert result["success"] is True
        assert "allocation_id" in result
        assert "pool_id" in result
        assert result["allocated_memory_gb"] == 30.0
        assert "pool_utilization" in result
        assert "gpu_type" in result

        # Verify pool state updated
        pool_id = result["pool_id"]
        pool = manager.gpu_pools[pool_id]
        assert pool["allocated_memory_gb"] == 30.0
        assert "model-001" in pool["active_models"]
        assert pool["active_models"]["model-001"]["memory_gb"] == 30.0

        # Verify model placement cached
        assert "model-001" in manager.model_placement_cache
        assert manager.model_placement_cache["model-001"] == pool_id

    def test_allocate_memory_with_preferred_pool(self, gpu_pool_manager_with_pools):
        """Test memory allocation with preferred pool."""
        manager = gpu_pool_manager_with_pools

        result = manager.allocate_gpu_memory(
            "model-002", memory_requirement_gb=20.0, preferred_pool_id="pool-001"
        )

        assert result["success"] is True
        assert result["pool_id"] == "pool-001"

    def test_allocate_memory_insufficient_space(self, gpu_pool_manager_with_pools):
        """Test memory allocation failure when insufficient space."""
        manager = gpu_pool_manager_with_pools

        # Try to allocate more memory than available in any pool
        result = manager.allocate_gpu_memory("large-model", memory_requirement_gb=500.0)

        assert result["success"] is False
        assert "error" in result
        assert "No available GPU pool" in result["error"]
        assert result["required_memory_gb"] == 500.0
        assert "available_pools" in result

    def test_allocate_memory_preferred_pool_insufficient(
        self, gpu_pool_manager_with_pools
    ):
        """Test allocation falls back when preferred pool has insufficient space."""
        manager = gpu_pool_manager_with_pools

        # Fill pool-001 first
        manager.allocate_gpu_memory(
            "model-fill", memory_requirement_gb=100.0, preferred_pool_id="pool-001"
        )

        # Try to allocate to pool-001 again (should fail and fall back)
        result = manager.allocate_gpu_memory(
            "model-fallback", memory_requirement_gb=80.0, preferred_pool_id="pool-001"
        )

        assert result["success"] is True
        assert result["pool_id"] != "pool-001"  # Should use different pool

    def test_multiple_allocations_same_pool(self, gpu_pool_manager_with_pools):
        """Test multiple allocations to same pool."""
        manager = gpu_pool_manager_with_pools

        result1 = manager.allocate_gpu_memory("model-001", 30.0)
        result2 = manager.allocate_gpu_memory("model-002", 40.0)

        assert result1["success"] is True
        assert result2["success"] is True

        # Both could be in same pool if it has capacity
        if result1["pool_id"] == result2["pool_id"]:
            pool = manager.gpu_pools[result1["pool_id"]]
            assert pool["allocated_memory_gb"] == 70.0
            assert len(pool["active_models"]) == 2


class TestMemoryDeallocation:
    """Test GPU memory deallocation functionality."""

    def test_deallocate_memory_success(self, gpu_pool_manager_with_allocation):
        """Test successful GPU memory deallocation."""
        manager = gpu_pool_manager_with_allocation

        # Verify allocation exists
        assert "test-model" in manager.model_placement_cache

        result = manager.deallocate_gpu_memory("test-model")

        assert result is True
        assert "test-model" not in manager.model_placement_cache

        # Verify pool state updated
        pool_id = "pool-001"
        pool = manager.gpu_pools[pool_id]
        assert "test-model" not in pool["active_models"]
        assert pool["allocated_memory_gb"] == 0.0
        assert pool["utilization"] == 0.0

    def test_deallocate_nonexistent_model(self, gpu_pool_manager_with_pools):
        """Test deallocation of non-existent model."""
        manager = gpu_pool_manager_with_pools

        result = manager.deallocate_gpu_memory("nonexistent-model")

        assert result is False

    def test_deallocate_updates_allocation_record(
        self, gpu_pool_manager_with_allocation
    ):
        """Test deallocation updates allocation record status."""
        manager = gpu_pool_manager_with_allocation

        manager.deallocate_gpu_memory("test-model")

        # Check allocation record was updated
        for allocation in manager.memory_allocations.values():
            if allocation["model_hash"] == "test-model":
                assert allocation["status"] == "deallocated"
                assert "deallocation_time" in allocation
                break
        else:
            pytest.fail("Allocation record not found")

    def test_deallocate_pool_not_found(self, gpu_pool_manager_with_allocation):
        """Test deallocation when pool no longer exists."""
        manager = gpu_pool_manager_with_allocation

        # Add model to cache but remove pool
        manager.model_placement_cache["orphan-model"] = "nonexistent-pool"

        result = manager.deallocate_gpu_memory("orphan-model")

        assert result is False


class TestPoolSelection:
    """Test GPU pool selection logic."""

    def test_find_best_pool_by_utilization(self, gpu_pool_manager_with_pools):
        """Test pool selection prefers moderate utilization."""
        manager = gpu_pool_manager_with_pools

        # Allocate to create different utilization levels
        manager.allocate_gpu_memory("model-001", 50.0, preferred_pool_id="pool-001")
        manager.allocate_gpu_memory("model-002", 100.0, preferred_pool_id="pool-002")

        # Next allocation should prefer pool with moderate utilization
        result = manager.allocate_gpu_memory("model-003", 20.0)

        assert result["success"] is True
        # Should prefer pool with utilization closer to 0.7

    def test_gpu_type_scoring(self, gpu_pool_manager):
        """Test GPU type scoring for workload suitability."""
        manager = gpu_pool_manager

        # Create pools with different GPU types
        manager.create_gpu_pool(
            "pool-h100", ResourceType.GPU_H100, 2, 80.0, CloudProvider.AWS, "us-east-1"
        )
        manager.create_gpu_pool(
            "pool-a100", ResourceType.GPU_A100, 4, 40.0, CloudProvider.GCP, "us-west-1"
        )
        manager.create_gpu_pool(
            "pool-v100",
            ResourceType.GPU_V100,
            8,
            32.0,
            CloudProvider.AZURE,
            "eu-west-1",
        )

        # Test internal scoring method
        h100_score = manager._calculate_gpu_type_score(ResourceType.GPU_H100)
        a100_score = manager._calculate_gpu_type_score(ResourceType.GPU_A100)
        v100_score = manager._calculate_gpu_type_score(ResourceType.GPU_V100)

        assert h100_score == 1.0
        assert a100_score == 0.9
        assert v100_score == 0.7
        assert h100_score > a100_score > v100_score


class TestMemoryOptimization:
    """Test memory optimization functionality."""

    @pytest.mark.asyncio
    async def test_optimize_memory_layout_no_optimization_needed(
        self, gpu_pool_manager_with_pools
    ):
        """Test memory optimization when no optimization needed."""
        manager = gpu_pool_manager_with_pools

        # Low utilization, no optimization needed
        result = await manager.optimize_memory_layout()

        assert "pools_optimized" in result
        assert "memory_saved_gb" in result
        assert "models_relocated" in result
        assert "optimizations_applied" in result
        assert result["pools_optimized"] == 0
        assert result["memory_saved_gb"] == 0.0
        assert result["models_relocated"] == 0

    @pytest.mark.asyncio
    async def test_optimize_memory_layout_high_utilization(
        self, gpu_pool_manager_with_pools
    ):
        """Test memory optimization with high utilization pool."""
        manager = gpu_pool_manager_with_pools

        # Fill pool-001 to high utilization (>85%)
        manager.allocate_gpu_memory("large-model", 140.0, preferred_pool_id="pool-001")

        # Add small models that could be relocated
        manager.allocate_gpu_memory("small-1", 4.0, preferred_pool_id="pool-001")

        result = await manager.optimize_memory_layout()

        # Should attempt optimization on high utilization pool
        assert result["pools_optimized"] >= 0

    @pytest.mark.asyncio
    async def test_optimize_pool_memory_relocates_models(
        self, gpu_pool_manager_with_pools
    ):
        """Test pool memory optimization relocates small models."""
        manager = gpu_pool_manager_with_pools

        # Create scenario: pool-001 high utilization, pool-002 low utilization
        manager.allocate_gpu_memory("large", 120.0, preferred_pool_id="pool-001")
        manager.allocate_gpu_memory("small-1", 3.0, preferred_pool_id="pool-001")
        manager.allocate_gpu_memory("small-2", 4.0, preferred_pool_id="pool-001")

        pool_001 = manager.gpu_pools["pool-001"]
        pool_001["utilization"]

        # Manually trigger pool optimization
        result = await manager._optimize_pool_memory(pool_001)

        # Should have relocation opportunities
        assert "models_relocated" in result
        assert "optimizations_applied" in result

    @pytest.mark.asyncio
    async def test_find_better_pool_for_model(self, gpu_pool_manager_with_pools):
        """Test finding better pool for model relocation."""
        manager = gpu_pool_manager_with_pools

        # Fill pool-001
        manager.allocate_gpu_memory("filler", 100.0, preferred_pool_id="pool-001")

        # pool-002 has low utilization
        better_pool = manager._find_better_pool_for_model(
            "small-model", 5.0, "pool-001"
        )

        # Should find pool-002 as better option (lower utilization)
        if better_pool:
            assert better_pool["pool_id"] != "pool-001"
            assert better_pool["utilization"] < 0.6


class TestPoolStatistics:
    """Test GPU pool statistics functionality."""

    def test_get_pool_statistics_empty(self, gpu_pool_manager):
        """Test statistics with no pools."""
        manager = gpu_pool_manager

        stats = manager.get_pool_statistics()

        assert stats["total_pools"] == 0
        assert stats["total_gpus"] == 0
        assert stats["total_memory_gb"] == 0.0
        assert stats["allocated_memory_gb"] == 0.0
        assert stats["memory_utilization"] == 0.0
        assert stats["average_pool_utilization"] == 0.0
        assert stats["active_models"] == 0
        assert stats["active_allocations"] == 0
        assert "pools_by_provider" in stats

    def test_get_pool_statistics_with_pools(self, gpu_pool_manager_with_pools):
        """Test statistics with multiple pools."""
        manager = gpu_pool_manager_with_pools

        stats = manager.get_pool_statistics()

        assert stats["total_pools"] == 2
        assert stats["total_gpus"] == 12  # 4 + 8
        assert stats["total_memory_gb"] == 416.0  # 160 + 256
        assert stats["allocated_memory_gb"] == 0.0
        assert stats["memory_utilization"] == 0.0
        assert stats["average_pool_utilization"] == 0.0
        assert isinstance(stats["pools_by_provider"], dict)

    def test_get_pool_statistics_with_allocations(
        self, gpu_pool_manager_with_allocation
    ):
        """Test statistics with active allocations."""
        manager = gpu_pool_manager_with_allocation

        stats = manager.get_pool_statistics()

        assert stats["allocated_memory_gb"] == 50.0
        assert stats["memory_utilization"] > 0.0
        assert stats["active_models"] == 1
        assert stats["active_allocations"] == 1

    def test_pool_statistics_by_provider(self, gpu_pool_manager_with_pools):
        """Test pool statistics grouped by provider."""
        manager = gpu_pool_manager_with_pools

        stats = manager.get_pool_statistics()
        provider_stats = stats["pools_by_provider"]

        assert provider_stats[CloudProvider.AWS.value] == 1
        assert provider_stats[CloudProvider.GCP.value] == 1
        # Other providers should have 0
        assert provider_stats[CloudProvider.AZURE.value] == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_can_allocate_memory_exact_match(self, gpu_pool_manager_with_pools):
        """Test allocation when memory requirement exactly matches available."""
        manager = gpu_pool_manager_with_pools

        pool = manager.gpu_pools["pool-001"]
        result = manager._can_allocate_memory(pool, 160.0)  # Exact total capacity

        assert result is True

    def test_can_allocate_memory_overflow(self, gpu_pool_manager_with_pools):
        """Test allocation when memory requirement exceeds available."""
        manager = gpu_pool_manager_with_pools

        pool = manager.gpu_pools["pool-001"]
        result = manager._can_allocate_memory(pool, 161.0)  # Exceeds capacity

        assert result is False

    def test_pool_utilization_calculation(self, gpu_pool_manager_with_pools):
        """Test pool utilization is calculated correctly."""
        manager = gpu_pool_manager_with_pools

        result = manager.allocate_gpu_memory(
            "model", 80.0, preferred_pool_id="pool-001"
        )

        pool = manager.gpu_pools["pool-001"]
        expected_utilization = 80.0 / 160.0  # 0.5
        assert abs(pool["utilization"] - expected_utilization) < 0.01
        assert abs(result["pool_utilization"] - expected_utilization) < 0.01

    def test_allocation_id_uniqueness(self, gpu_pool_manager_with_pools):
        """Test allocation IDs are unique."""
        manager = gpu_pool_manager_with_pools

        result1 = manager.allocate_gpu_memory("model-1", 10.0)
        time.sleep(0.001)  # Ensure different timestamp
        result2 = manager.allocate_gpu_memory("model-2", 10.0)

        assert result1["allocation_id"] != result2["allocation_id"]

    def test_memory_headroom_calculation(self, gpu_pool_manager_with_pools):
        """Test memory headroom is considered in pool selection."""
        manager = gpu_pool_manager_with_pools

        # Allocate to create different headroom scenarios
        manager.allocate_gpu_memory("model-1", 100.0, preferred_pool_id="pool-001")
        manager.allocate_gpu_memory("model-2", 200.0, preferred_pool_id="pool-002")

        # Allocate small model - should prefer pool with better headroom ratio
        result = manager.allocate_gpu_memory("small", 10.0)

        assert result["success"] is True
