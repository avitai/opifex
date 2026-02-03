"""Pytest configuration and fixtures for resource management tests."""

import pytest
from flax import nnx


@pytest.fixture
def rngs():
    """Provide RNG for NNX modules."""
    return nnx.Rngs(0)


@pytest.fixture
def sample_resource_requirements():
    """Provide sample resource requirements for testing."""
    from opifex.deployment.resource_management.types import ResourceType

    return {
        ResourceType.GPU_A100: 2,
        ResourceType.CPU_INTEL: 8,
        ResourceType.MEMORY: 64,
    }


@pytest.fixture
def sample_constraints():
    """Provide sample constraints for testing."""
    return {
        "max_cost_usd_per_hour": 50.0,
        "min_performance_score": 0.8,
        "max_latency_ms": 10.0,
    }


@pytest.fixture
def mock_orchestrator(rngs):
    """Provide a mock ResourceOrchestrator for testing."""
    from opifex.deployment.resource_management.types import OptimizationObjective

    # Create a minimal mock that doesn't require full ResourceOrchestrator
    class MockOrchestrator:
        def __init__(self):
            self.optimization_objective = OptimizationObjective.BALANCE_COST_PERFORMANCE
            self.resource_pools = {}

    return MockOrchestrator()


@pytest.fixture
def gpu_pool_manager(mock_orchestrator):
    """Provide a GPUPoolManager instance for testing."""
    from opifex.deployment.resource_management.gpu_manager import GPUPoolManager

    return GPUPoolManager(mock_orchestrator)


@pytest.fixture
def gpu_pool_manager_with_pools(gpu_pool_manager):
    """Provide a GPUPoolManager with pre-created pools."""
    from opifex.deployment.resource_management.types import CloudProvider, ResourceType

    # Create two test pools
    gpu_pool_manager.create_gpu_pool(
        pool_id="pool-001",
        gpu_type=ResourceType.GPU_A100,
        gpu_count=4,
        memory_per_gpu_gb=40.0,
        provider=CloudProvider.AWS,
        region="us-east-1",
    )

    gpu_pool_manager.create_gpu_pool(
        pool_id="pool-002",
        gpu_type=ResourceType.GPU_V100,
        gpu_count=8,
        memory_per_gpu_gb=32.0,
        provider=CloudProvider.GCP,
        region="us-west-1",
    )

    return gpu_pool_manager


@pytest.fixture
def gpu_pool_manager_with_allocation(gpu_pool_manager_with_pools):
    """Provide a GPUPoolManager with an active allocation."""
    manager = gpu_pool_manager_with_pools

    # Allocate memory for a test model
    manager.allocate_gpu_memory(
        "test-model", memory_requirement_gb=50.0, preferred_pool_id="pool-001"
    )

    return manager


@pytest.fixture
def resource_orchestrator(rngs):
    """Provide a ResourceOrchestrator instance for testing."""
    from opifex.deployment.resource_management.orchestrator import ResourceOrchestrator

    return ResourceOrchestrator(rngs=rngs)


@pytest.fixture
def sample_pool():
    """Provide a sample ResourcePool for testing."""
    from opifex.deployment.resource_management.types import (
        CloudProvider,
        ResourcePool,
        ResourceType,
    )

    return ResourcePool(
        pool_id="test-pool-001",
        provider=CloudProvider.AWS,
        region="us-east-1",
        resource_type=ResourceType.GPU_A100,
        total_capacity=100,
        available_capacity=100,
        reserved_capacity=0,
        cost_per_hour_usd=3.0,
        performance_score=0.85,
        carbon_efficiency=150.0,
        availability_sla=0.99,
    )


@pytest.fixture
def sample_pool_2():
    """Provide a second sample ResourcePool for testing."""
    from opifex.deployment.resource_management.types import (
        CloudProvider,
        ResourcePool,
        ResourceType,
    )

    return ResourcePool(
        pool_id="test-pool-002",
        provider=CloudProvider.GCP,
        region="us-west-1",
        resource_type=ResourceType.CPU_INTEL,
        total_capacity=200,
        available_capacity=150,
        reserved_capacity=50,
        cost_per_hour_usd=1.5,
        performance_score=0.75,
        carbon_efficiency=80.0,
        availability_sla=0.995,
    )


@pytest.fixture
def resource_orchestrator_with_pools(resource_orchestrator, sample_pool, sample_pool_2):
    """Provide a ResourceOrchestrator with registered pools."""
    resource_orchestrator.register_resource_pool(sample_pool)
    resource_orchestrator.register_resource_pool(sample_pool_2)

    # Add a third pool for MEMORY type
    from opifex.deployment.resource_management.types import (
        CloudProvider,
        ResourcePool,
        ResourceType,
    )

    memory_pool = ResourcePool(
        pool_id="test-pool-003",
        provider=CloudProvider.AZURE,
        region="eu-west-1",
        resource_type=ResourceType.MEMORY,
        total_capacity=1000,
        available_capacity=800,
        reserved_capacity=200,
        cost_per_hour_usd=0.5,
        performance_score=0.9,
        carbon_efficiency=50.0,
        availability_sla=0.999,
    )
    resource_orchestrator.register_resource_pool(memory_pool)

    return resource_orchestrator


@pytest.fixture
def cost_controller():
    """Provide a CostController instance for testing."""
    from opifex.deployment.resource_management.cost_controller import CostController

    return CostController()


@pytest.fixture
def sustainability_tracker():
    """Provide a SustainabilityTracker instance for testing."""
    from opifex.deployment.resource_management.sustainability import (
        SustainabilityTracker,
    )

    return SustainabilityTracker()


@pytest.fixture
def global_resource_manager(
    resource_orchestrator, gpu_pool_manager, cost_controller, sustainability_tracker
):
    """Provide a GlobalResourceManager instance for testing."""
    from opifex.deployment.resource_management.global_manager import (
        GlobalResourceManager,
    )

    return GlobalResourceManager(
        resource_orchestrator, gpu_pool_manager, cost_controller, sustainability_tracker
    )


@pytest.fixture
def global_resource_manager_with_setup(
    resource_orchestrator_with_pools,
    gpu_pool_manager,
    cost_controller,
    sustainability_tracker,
):
    """Provide a GlobalResourceManager with pools registered."""
    from opifex.deployment.resource_management.global_manager import (
        GlobalResourceManager,
    )

    return GlobalResourceManager(
        resource_orchestrator_with_pools,
        gpu_pool_manager,
        cost_controller,
        sustainability_tracker,
    )


@pytest.fixture
def global_resource_manager_with_gpu_pools(
    resource_orchestrator_with_pools,
    gpu_pool_manager_with_pools,
    cost_controller,
    sustainability_tracker,
):
    """Provide a GlobalResourceManager with GPU pools created."""
    from opifex.deployment.resource_management.global_manager import (
        GlobalResourceManager,
    )

    return GlobalResourceManager(
        resource_orchestrator_with_pools,
        gpu_pool_manager_with_pools,
        cost_controller,
        sustainability_tracker,
    )
