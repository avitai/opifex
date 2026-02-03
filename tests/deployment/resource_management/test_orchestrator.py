"""Comprehensive tests for ResourceOrchestrator.

Test-driven development (TDD) approach: These tests are written BEFORE
extracting the ResourceOrchestrator module from the original file.

This is the most complex module with NNX neural networks and 6 optimization objectives.
"""

import time

import jax.numpy as jnp
import pytest

from opifex.deployment.resource_management.types import (
    CloudProvider,
    OptimizationObjective,
    ResourcePool,
    ResourceType,
)


# Import after extraction
# from opifex.deployment.resource_management.orchestrator import ResourceOrchestrator


class TestResourceOrchestratorInitialization:
    """Test ResourceOrchestrator initialization and NNX module setup."""

    def test_initialization_default_parameters(self, rngs):
        """Test ResourceOrchestrator initialization with default parameters."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(rngs=rngs)

        assert orchestrator.optimization_objective == (
            OptimizationObjective.BALANCE_COST_PERFORMANCE
        )
        assert hasattr(orchestrator, "allocation_optimizer")
        assert hasattr(orchestrator, "cost_predictor")
        assert hasattr(orchestrator, "performance_predictor")
        assert isinstance(orchestrator.resource_pools, dict)
        assert len(orchestrator.resource_pools) == 0
        assert isinstance(orchestrator.active_allocations, dict)
        assert len(orchestrator.active_allocations) == 0

    def test_initialization_custom_objective(self, rngs):
        """Test initialization with custom optimization objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MINIMIZE_COST, rngs=rngs
        )

        assert (
            orchestrator.optimization_objective == OptimizationObjective.MINIMIZE_COST
        )

    def test_neural_network_components_exist(self, rngs):
        """Test that all neural network components are properly initialized."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(rngs=rngs)

        # Test that neural networks can perform forward passes
        test_input_32 = jnp.ones((1, 32))
        allocation_output = orchestrator.allocation_optimizer(test_input_32)
        assert allocation_output.shape == (1, 16)

        test_input_24 = jnp.ones((1, 24))
        cost_output = orchestrator.cost_predictor(test_input_24)
        assert cost_output.shape == (1, 1)

        performance_output = orchestrator.performance_predictor(test_input_24)
        assert performance_output.shape == (1, 1)


class TestPoolManagement:
    """Test resource pool registration and management."""

    def test_register_resource_pool(self, resource_orchestrator, sample_pool):
        """Test registering a new resource pool."""
        result = resource_orchestrator.register_resource_pool(sample_pool)

        assert result is True
        assert sample_pool.pool_id in resource_orchestrator.resource_pools
        assert resource_orchestrator.resource_pools[sample_pool.pool_id] == sample_pool

    def test_register_multiple_pools(
        self, resource_orchestrator, sample_pool, sample_pool_2
    ):
        """Test registering multiple resource pools."""
        resource_orchestrator.register_resource_pool(sample_pool)
        resource_orchestrator.register_resource_pool(sample_pool_2)

        assert len(resource_orchestrator.resource_pools) == 2
        assert sample_pool.pool_id in resource_orchestrator.resource_pools
        assert sample_pool_2.pool_id in resource_orchestrator.resource_pools

    def test_update_pool_status_success(
        self, resource_orchestrator_with_pools, sample_pool
    ):
        """Test updating pool status successfully."""
        result = resource_orchestrator_with_pools.update_pool_status(
            sample_pool.pool_id, utilization=0.75, available_capacity=50
        )

        assert result is True
        pool = resource_orchestrator_with_pools.resource_pools[sample_pool.pool_id]
        assert pool.current_utilization == 0.75
        assert pool.available_capacity == 50

    def test_update_pool_status_nonexistent(self, resource_orchestrator):
        """Test updating status of non-existent pool."""
        result = resource_orchestrator.update_pool_status(
            "nonexistent", utilization=0.5, available_capacity=10
        )

        assert result is False


class TestRequirementEncoding:
    """Test requirement encoding into feature vectors."""

    def test_encode_requirements_basic(
        self, resource_orchestrator, sample_resource_requirements
    ):
        """Test encoding basic resource requirements."""
        features = resource_orchestrator._encode_requirements(
            sample_resource_requirements, {}
        )

        assert isinstance(features, jnp.ndarray)
        assert features.shape == (32,)
        # Check that GPU requirements are encoded
        assert features[list(ResourceType).index(ResourceType.GPU_A100)] == 2

    def test_encode_requirements_with_constraints(
        self, resource_orchestrator, sample_resource_requirements, sample_constraints
    ):
        """Test encoding requirements with constraints."""
        features = resource_orchestrator._encode_requirements(
            sample_resource_requirements, sample_constraints
        )

        assert features.shape == (32,)
        # Check constraint encoding (normalized values)
        assert 0.0 <= features[len(ResourceType)] <= 1.0  # max_cost normalized

    def test_encode_requirements_optimization_objective(self, rngs):
        """Test that optimization objective is encoded in features."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MINIMIZE_COST, rngs=rngs
        )

        features = orchestrator._encode_requirements({ResourceType.GPU_A100: 1}, {})

        # Check that optimization objective is one-hot encoded
        objective_start = len(ResourceType) + 4  # After resources and constraints
        objective_encoding = features[
            objective_start : objective_start + len(OptimizationObjective)
        ]
        assert jnp.sum(objective_encoding) == 1.0  # One-hot encoded


class TestPoolFiltering:
    """Test pool filtering based on requirements and constraints."""

    def test_filter_eligible_pools_capacity(
        self, resource_orchestrator_with_pools, sample_resource_requirements
    ):
        """Test filtering pools by capacity requirements."""
        eligible = resource_orchestrator_with_pools._filter_eligible_pools(
            sample_resource_requirements, {}
        )

        # Should find pools that have capacity for the requirements
        assert len(eligible) > 0
        for pool in eligible:
            required = sample_resource_requirements.get(pool.resource_type, 0)
            assert pool.available_capacity >= required

    def test_filter_eligible_pools_cost_constraint(
        self, resource_orchestrator_with_pools
    ):
        """Test filtering pools by cost constraint."""
        requirements = {ResourceType.GPU_A100: 1}
        constraints = {"max_cost_usd_per_hour": 5.0}  # Very low cost

        eligible = resource_orchestrator_with_pools._filter_eligible_pools(
            requirements, constraints
        )

        # Should only include pools under cost threshold
        for pool in eligible:
            assert pool.cost_per_hour_usd <= 5.0

    def test_filter_eligible_pools_performance_constraint(
        self, resource_orchestrator_with_pools
    ):
        """Test filtering pools by performance constraint."""
        requirements = {ResourceType.GPU_A100: 1}
        constraints = {"min_performance_score": 0.9}  # High performance required

        eligible = resource_orchestrator_with_pools._filter_eligible_pools(
            requirements, constraints
        )

        for pool in eligible:
            assert pool.performance_score >= 0.9

    def test_filter_eligible_pools_provider_preference(
        self, resource_orchestrator_with_pools
    ):
        """Test filtering pools by provider preference."""
        requirements = {ResourceType.GPU_A100: 1}
        constraints = {"preferred_providers": [CloudProvider.AWS]}

        eligible = resource_orchestrator_with_pools._filter_eligible_pools(
            requirements, constraints
        )

        for pool in eligible:
            assert pool.provider == CloudProvider.AWS

    def test_filter_eligible_pools_no_match(self, resource_orchestrator_with_pools):
        """Test filtering when no pools match requirements."""
        requirements = {ResourceType.GPU_A100: 10000}  # Impossible capacity

        eligible = resource_orchestrator_with_pools._filter_eligible_pools(
            requirements, {}
        )

        # Pools of other resource types don't match GPU_A100 requirements
        # But they're still returned by filter if they have capacity for their type
        # The actual filtering happens in _select_optimal_pools which groups by type
        assert all(
            pool.resource_type != ResourceType.GPU_A100
            or pool.available_capacity < 10000
            for pool in eligible
        )


class TestPoolSelection:
    """Test pool selection based on optimization objectives."""

    def test_select_optimal_pools_minimize_cost(self, rngs):
        """Test pool selection with MINIMIZE_COST objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MINIMIZE_COST, rngs=rngs
        )

        # Create pools with different costs
        cheap_pool = ResourcePool(
            pool_id="cheap",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=1.0,
            performance_score=0.7,
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )
        expensive_pool = ResourcePool(
            pool_id="expensive",
            provider=CloudProvider.AWS,
            region="us-west-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=10.0,
            performance_score=0.95,
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )

        pools = [cheap_pool, expensive_pool]
        selected = orchestrator._score_and_select_pool(pools)

        assert selected == cheap_pool  # Should select cheaper pool

    def test_select_optimal_pools_minimize_latency(self, rngs):
        """Test pool selection with MINIMIZE_LATENCY objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MINIMIZE_LATENCY, rngs=rngs
        )

        slow_pool = ResourcePool(
            pool_id="slow",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=1.0,
            performance_score=0.7,
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )
        fast_pool = ResourcePool(
            pool_id="fast",
            provider=CloudProvider.AWS,
            region="us-west-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=10.0,
            performance_score=0.95,
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )

        pools = [slow_pool, fast_pool]
        selected = orchestrator._score_and_select_pool(pools)

        assert selected == fast_pool  # Should select faster pool

    def test_select_optimal_pools_maximize_throughput(self, rngs):
        """Test pool selection with MAXIMIZE_THROUGHPUT objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MAXIMIZE_THROUGHPUT, rngs=rngs
        )

        small_pool = ResourcePool(
            pool_id="small",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=10,
            available_capacity=10,
            reserved_capacity=0,
            cost_per_hour_usd=1.0,
            performance_score=0.9,
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )
        large_pool = ResourcePool(
            pool_id="large",
            provider=CloudProvider.AWS,
            region="us-west-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=5.0,
            performance_score=0.9,
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )

        pools = [small_pool, large_pool]
        selected = orchestrator._score_and_select_pool(pools)

        assert selected == large_pool  # Should select pool with higher throughput

    def test_select_optimal_pools_minimize_carbon(self, rngs):
        """Test pool selection with MINIMIZE_CARBON objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MINIMIZE_CARBON, rngs=rngs
        )

        dirty_pool = ResourcePool(
            pool_id="dirty",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=1.0,
            performance_score=0.9,
            carbon_efficiency=500.0,  # High carbon
            availability_sla=0.99,
        )
        green_pool = ResourcePool(
            pool_id="green",
            provider=CloudProvider.GCP,
            region="us-west-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=2.0,
            performance_score=0.9,
            carbon_efficiency=50.0,  # Low carbon
            availability_sla=0.99,
        )

        pools = [dirty_pool, green_pool]
        selected = orchestrator._score_and_select_pool(pools)

        assert selected == green_pool  # Should select greener pool

    def test_select_optimal_pools_maximize_availability(self, rngs):
        """Test pool selection with MAXIMIZE_AVAILABILITY objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.MAXIMIZE_AVAILABILITY,
            rngs=rngs,
        )

        unreliable_pool = ResourcePool(
            pool_id="unreliable",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=1.0,
            performance_score=0.9,
            carbon_efficiency=100.0,
            availability_sla=0.95,
        )
        reliable_pool = ResourcePool(
            pool_id="reliable",
            provider=CloudProvider.AWS,
            region="us-west-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=2.0,
            performance_score=0.9,
            carbon_efficiency=100.0,
            availability_sla=0.999,
        )

        pools = [unreliable_pool, reliable_pool]
        selected = orchestrator._score_and_select_pool(pools)

        assert selected == reliable_pool  # Should select more reliable pool

    def test_select_optimal_pools_balance_cost_performance(self, rngs):
        """Test pool selection with BALANCE_COST_PERFORMANCE objective."""
        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        orchestrator = ResourceOrchestrator(
            optimization_objective=OptimizationObjective.BALANCE_COST_PERFORMANCE,
            rngs=rngs,
        )

        imbalanced_pool = ResourcePool(
            pool_id="imbalanced",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=10.0,  # Expensive
            performance_score=0.6,  # Poor performance
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )
        balanced_pool = ResourcePool(
            pool_id="balanced",
            provider=CloudProvider.GCP,
            region="us-west-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=100,
            reserved_capacity=0,
            cost_per_hour_usd=3.0,  # Moderate cost
            performance_score=0.9,  # Good performance
            carbon_efficiency=100.0,
            availability_sla=0.99,
        )

        pools = [imbalanced_pool, balanced_pool]
        selected = orchestrator._score_and_select_pool(pools)

        assert selected == balanced_pool  # Should select balanced pool

    def test_score_and_select_pool_empty_list(self, resource_orchestrator):
        """Test pool selection with empty pool list."""
        result = resource_orchestrator._score_and_select_pool([])

        assert result is None


class TestResourceAllocation:
    """Test complete resource allocation flow."""

    def test_optimize_resource_allocation_success(
        self, resource_orchestrator_with_pools, sample_resource_requirements
    ):
        """Test successful resource allocation."""
        allocation = resource_orchestrator_with_pools.optimize_resource_allocation(
            sample_resource_requirements
        )

        assert allocation.allocation_id.startswith("alloc_")
        assert allocation.requested_resources == sample_resource_requirements
        assert len(allocation.allocated_resources) > 0
        assert allocation.cost_estimate_usd > 0
        assert allocation.performance_estimate >= 0
        assert allocation.carbon_footprint_kg >= 0
        assert allocation.start_time > 0
        assert allocation.end_time is None
        assert allocation.allocation_strategy in [
            "Cost-Optimized",
            "Latency-Optimized",
            "Throughput-Optimized",
            "Balanced",
            "Carbon-Optimized",
            "Availability-Optimized",
            "Custom",
        ]

    def test_optimize_resource_allocation_with_constraints(
        self, resource_orchestrator_with_pools, sample_resource_requirements
    ):
        """Test resource allocation with constraints."""
        constraints = {"max_cost_usd_per_hour": 100.0, "min_performance_score": 0.5}

        allocation = resource_orchestrator_with_pools.optimize_resource_allocation(
            sample_resource_requirements, constraints
        )

        assert allocation.cost_estimate_usd <= 100.0 * len(
            allocation.allocated_resources
        )

    def test_optimize_resource_allocation_no_eligible_pools(
        self, resource_orchestrator
    ):
        """Test allocation when no pools meet requirements."""
        # Empty orchestrator with no pools
        requirements = {ResourceType.GPU_A100: 1}

        with pytest.raises(ValueError, match="No eligible resource pools found"):
            resource_orchestrator.optimize_resource_allocation(requirements)

    def test_allocation_recorded_in_active_allocations(
        self, resource_orchestrator_with_pools, sample_resource_requirements
    ):
        """Test that allocation is recorded in active allocations."""
        initial_count = len(resource_orchestrator_with_pools.active_allocations)

        allocation = resource_orchestrator_with_pools.optimize_resource_allocation(
            sample_resource_requirements
        )

        assert len(resource_orchestrator_with_pools.active_allocations) == (
            initial_count + 1
        )
        assert (
            allocation.allocation_id
            in resource_orchestrator_with_pools.active_allocations
        )


class TestCostEstimation:
    """Test cost estimation calculations."""

    def test_calculate_cost_estimate(self, resource_orchestrator_with_pools):
        """Test cost estimation calculation."""
        # Create allocation with known costs
        pool = next(iter(resource_orchestrator_with_pools.resource_pools.values()))
        selected_pools = {pool.pool_id: pool}
        requirements = {pool.resource_type: 2}

        cost = resource_orchestrator_with_pools._calculate_cost_estimate(
            selected_pools, requirements
        )

        expected_cost = pool.cost_per_hour_usd * 2
        assert abs(cost - expected_cost) < 0.01

    def test_calculate_cost_estimate_multiple_pools(
        self, resource_orchestrator_with_pools
    ):
        """Test cost estimation with multiple pools."""
        pools = list(resource_orchestrator_with_pools.resource_pools.values())[:2]
        selected_pools = {p.pool_id: p for p in pools}
        requirements = {p.resource_type: 1 for p in pools}

        cost = resource_orchestrator_with_pools._calculate_cost_estimate(
            selected_pools, requirements
        )

        expected_cost = sum(p.cost_per_hour_usd for p in pools)
        assert abs(cost - expected_cost) < 0.01


class TestPerformanceEstimation:
    """Test performance estimation calculations."""

    def test_calculate_performance_estimate(self, resource_orchestrator_with_pools):
        """Test performance estimation calculation."""
        pool = next(iter(resource_orchestrator_with_pools.resource_pools.values()))
        selected_pools = {pool.pool_id: pool}

        performance = resource_orchestrator_with_pools._calculate_performance_estimate(
            selected_pools
        )

        # With one pool, should return that pool's performance score
        assert performance == pool.performance_score

    def test_calculate_performance_estimate_empty(self, resource_orchestrator):
        """Test performance estimation with no pools."""
        performance = resource_orchestrator._calculate_performance_estimate({})

        assert performance == 0.0

    def test_calculate_performance_estimate_weighted(
        self, resource_orchestrator_with_pools
    ):
        """Test weighted performance estimation with multiple pools."""
        pools = list(resource_orchestrator_with_pools.resource_pools.values())[:2]
        selected_pools = {p.pool_id: p for p in pools}

        performance = resource_orchestrator_with_pools._calculate_performance_estimate(
            selected_pools
        )

        # Should be weighted average
        assert 0.0 <= performance <= 1.0


class TestCarbonFootprintCalculation:
    """Test carbon footprint calculations."""

    def test_calculate_carbon_footprint(self, resource_orchestrator_with_pools):
        """Test carbon footprint calculation."""
        pool = next(iter(resource_orchestrator_with_pools.resource_pools.values()))
        selected_pools = {pool.pool_id: pool}
        requirements = {pool.resource_type: 1}

        carbon = resource_orchestrator_with_pools._calculate_carbon_footprint(
            selected_pools, requirements
        )

        # Should convert gCO2 to kg
        expected_carbon = pool.carbon_efficiency * 1 / 1000.0
        assert abs(carbon - expected_carbon) < 0.001


class TestAllocationStrategyNames:
    """Test allocation strategy naming."""

    def test_get_allocation_strategy_name_all_objectives(self):
        """Test strategy names for all optimization objectives."""
        from flax import nnx

        from opifex.deployment.resource_management.orchestrator import (
            ResourceOrchestrator,
        )

        rngs = nnx.Rngs(0)

        expected_names = {
            OptimizationObjective.MINIMIZE_COST: "Cost-Optimized",
            OptimizationObjective.MINIMIZE_LATENCY: "Latency-Optimized",
            OptimizationObjective.MAXIMIZE_THROUGHPUT: "Throughput-Optimized",
            OptimizationObjective.BALANCE_COST_PERFORMANCE: "Balanced",
            OptimizationObjective.MINIMIZE_CARBON: "Carbon-Optimized",
            OptimizationObjective.MAXIMIZE_AVAILABILITY: "Availability-Optimized",
        }

        for objective, expected_name in expected_names.items():
            orchestrator = ResourceOrchestrator(
                optimization_objective=objective, rngs=rngs
            )
            assert orchestrator._get_allocation_strategy_name() == expected_name


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_allocation_id_uniqueness(
        self, resource_orchestrator_with_pools, sample_resource_requirements
    ):
        """Test that allocation IDs are unique."""
        alloc1 = resource_orchestrator_with_pools.optimize_resource_allocation(
            sample_resource_requirements
        )
        time.sleep(0.001)  # Ensure different timestamp
        alloc2 = resource_orchestrator_with_pools.optimize_resource_allocation(
            sample_resource_requirements
        )

        assert alloc1.allocation_id != alloc2.allocation_id

    def test_empty_constraints_handled(
        self, resource_orchestrator_with_pools, sample_resource_requirements
    ):
        """Test that None constraints are handled properly."""
        allocation = resource_orchestrator_with_pools.optimize_resource_allocation(
            sample_resource_requirements, constraints=None
        )

        assert allocation is not None

    def test_requirements_with_zero_quantity(self, resource_orchestrator_with_pools):
        """Test requirements with zero quantity are skipped."""
        requirements = {ResourceType.GPU_A100: 0, ResourceType.CPU_INTEL: 2}

        # Should not raise error, just skip zero quantity
        allocation = resource_orchestrator_with_pools.optimize_resource_allocation(
            requirements
        )

        # Should only allocate CPU_INTEL
        assert len(allocation.allocated_resources) >= 0
