"""Tests for resource management types module.

Following TDD principles: Write ALL tests BEFORE extracting the module.
"""

from opifex.deployment.resource_management.types import (
    CloudProvider,
    CostOptimization,
    OptimizationObjective,
    ResourceAllocation,
    ResourcePool,
    ResourceType,
    SustainabilityMetrics,
)


class TestCloudProviderEnum:
    """Tests for CloudProvider enum."""

    def test_cloud_provider_enum_values(self):
        """Test CloudProvider enum has all expected values."""
        assert hasattr(CloudProvider, "AWS")
        assert hasattr(CloudProvider, "GCP")
        assert hasattr(CloudProvider, "AZURE")
        assert hasattr(CloudProvider, "IBM_CLOUD")
        assert hasattr(CloudProvider, "ORACLE_CLOUD")
        assert hasattr(CloudProvider, "ON_PREMISE")
        assert hasattr(CloudProvider, "HYBRID")

    def test_cloud_provider_enum_count(self):
        """Test CloudProvider enum has exactly 7 values."""
        assert len(CloudProvider) == 7

    def test_cloud_provider_string_values(self):
        """Test CloudProvider enum string representations."""
        assert CloudProvider.AWS.value == "aws"
        assert CloudProvider.GCP.value == "gcp"
        assert CloudProvider.AZURE.value == "azure"
        assert CloudProvider.IBM_CLOUD.value == "ibm_cloud"
        assert CloudProvider.ORACLE_CLOUD.value == "oracle_cloud"
        assert CloudProvider.ON_PREMISE.value == "on_premise"
        assert CloudProvider.HYBRID.value == "hybrid"


class TestResourceTypeEnum:
    """Tests for ResourceType enum."""

    def test_resource_type_enum_values(self):
        """Test ResourceType enum has all expected values."""
        assert hasattr(ResourceType, "GPU_A100")
        assert hasattr(ResourceType, "GPU_V100")
        assert hasattr(ResourceType, "GPU_H100")
        assert hasattr(ResourceType, "CPU_INTEL")
        assert hasattr(ResourceType, "CPU_AMD")
        assert hasattr(ResourceType, "CPU_ARM")
        assert hasattr(ResourceType, "TPU_V4")
        assert hasattr(ResourceType, "MEMORY")
        assert hasattr(ResourceType, "STORAGE")
        assert hasattr(ResourceType, "NETWORK")

    def test_resource_type_enum_count(self):
        """Test ResourceType enum has exactly 10 values."""
        assert len(ResourceType) == 10

    def test_resource_type_gpu_types(self):
        """Test GPU resource types have correct values."""
        assert ResourceType.GPU_A100.value == "gpu_a100"
        assert ResourceType.GPU_V100.value == "gpu_v100"
        assert ResourceType.GPU_H100.value == "gpu_h100"

    def test_resource_type_cpu_types(self):
        """Test CPU resource types have correct values."""
        assert ResourceType.CPU_INTEL.value == "cpu_intel"
        assert ResourceType.CPU_AMD.value == "cpu_amd"
        assert ResourceType.CPU_ARM.value == "cpu_arm"


class TestOptimizationObjectiveEnum:
    """Tests for OptimizationObjective enum."""

    def test_optimization_objective_enum_values(self):
        """Test OptimizationObjective enum has all expected values."""
        assert hasattr(OptimizationObjective, "MINIMIZE_COST")
        assert hasattr(OptimizationObjective, "MINIMIZE_LATENCY")
        assert hasattr(OptimizationObjective, "MAXIMIZE_THROUGHPUT")
        assert hasattr(OptimizationObjective, "BALANCE_COST_PERFORMANCE")
        assert hasattr(OptimizationObjective, "MINIMIZE_CARBON")
        assert hasattr(OptimizationObjective, "MAXIMIZE_AVAILABILITY")

    def test_optimization_objective_enum_count(self):
        """Test OptimizationObjective enum has exactly 6 values."""
        assert len(OptimizationObjective) == 6

    def test_optimization_objective_string_values(self):
        """Test OptimizationObjective enum string representations."""
        assert OptimizationObjective.MINIMIZE_COST.value == "minimize_cost"
        assert OptimizationObjective.MINIMIZE_LATENCY.value == "minimize_latency"
        assert OptimizationObjective.MAXIMIZE_THROUGHPUT.value == "maximize_throughput"
        assert (
            OptimizationObjective.BALANCE_COST_PERFORMANCE.value
            == "balance_cost_performance"
        )
        assert OptimizationObjective.MINIMIZE_CARBON.value == "minimize_carbon"
        assert (
            OptimizationObjective.MAXIMIZE_AVAILABILITY.value == "maximize_availability"
        )


class TestResourcePoolDataclass:
    """Tests for ResourcePool dataclass."""

    def test_resource_pool_creation(self):
        """Test ResourcePool dataclass initialization."""
        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=5.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        assert pool.pool_id == "pool-1"
        assert pool.provider == CloudProvider.AWS
        assert pool.region == "us-east-1"
        assert pool.resource_type == ResourceType.GPU_A100
        assert pool.total_capacity == 100
        assert pool.available_capacity == 50
        assert pool.reserved_capacity == 10
        assert pool.cost_per_hour_usd == 5.0
        assert pool.performance_score == 0.9
        assert pool.carbon_efficiency == 50.0
        assert pool.availability_sla == 0.995

    def test_resource_pool_default_values(self):
        """Test ResourcePool dataclass has correct default values."""
        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=5.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        # Test default values
        assert pool.current_utilization == 0.0
        assert pool.maintenance_window == "02:00-04:00 UTC"

    def test_resource_pool_with_custom_defaults(self):
        """Test ResourcePool dataclass with custom default values."""
        pool = ResourcePool(
            pool_id="pool-2",
            provider=CloudProvider.GCP,
            region="us-central1",
            resource_type=ResourceType.GPU_H100,
            total_capacity=200,
            available_capacity=150,
            reserved_capacity=20,
            cost_per_hour_usd=10.0,
            performance_score=0.95,
            carbon_efficiency=30.0,
            availability_sla=0.999,
            current_utilization=0.5,
            maintenance_window="03:00-05:00 UTC",
        )

        assert pool.current_utilization == 0.5
        assert pool.maintenance_window == "03:00-05:00 UTC"


class TestResourceAllocationDataclass:
    """Tests for ResourceAllocation dataclass."""

    def test_resource_allocation_creation(self):
        """Test ResourceAllocation dataclass initialization."""
        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=5.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        allocation = ResourceAllocation(
            allocation_id="alloc-1",
            requested_resources={ResourceType.GPU_A100: 4},
            allocated_resources={"pool-1": pool},
            start_time=1000.0,
            end_time=2000.0,
            cost_estimate_usd=100.0,
            performance_estimate=0.9,
            carbon_footprint_kg=2.5,
            allocation_strategy="cost-optimized",
        )

        assert allocation.allocation_id == "alloc-1"
        assert allocation.requested_resources == {ResourceType.GPU_A100: 4}
        assert allocation.allocated_resources == {"pool-1": pool}
        assert allocation.start_time == 1000.0
        assert allocation.end_time == 2000.0
        assert allocation.cost_estimate_usd == 100.0
        assert allocation.performance_estimate == 0.9
        assert allocation.carbon_footprint_kg == 2.5
        assert allocation.allocation_strategy == "cost-optimized"

    def test_resource_allocation_with_none_end_time(self):
        """Test ResourceAllocation with None end_time (ongoing allocation)."""
        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=5.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        allocation = ResourceAllocation(
            allocation_id="alloc-2",
            requested_resources={ResourceType.GPU_A100: 2},
            allocated_resources={"pool-1": pool},
            start_time=1000.0,
            end_time=None,  # Ongoing allocation
            cost_estimate_usd=50.0,
            performance_estimate=0.85,
            carbon_footprint_kg=1.2,
            allocation_strategy="balanced",
        )

        assert allocation.end_time is None


class TestCostOptimizationDataclass:
    """Tests for CostOptimization dataclass."""

    def test_cost_optimization_creation(self):
        """Test CostOptimization dataclass initialization."""
        cost_opt = CostOptimization(
            current_cost_usd_per_hour=100.0,
            optimized_cost_usd_per_hour=80.0,
            potential_savings_percentage=20.0,
            recommendations=["Switch to spot instances", "Use reserved capacity"],
            alternative_configurations=[{"provider": "GCP", "cost": 75.0}],
            cost_breakdown_by_provider={
                CloudProvider.AWS: 60.0,
                CloudProvider.GCP: 40.0,
            },
            roi_analysis={"monthly_savings": 14400.0, "payback_period_months": 3},
        )

        assert cost_opt.current_cost_usd_per_hour == 100.0
        assert cost_opt.optimized_cost_usd_per_hour == 80.0
        assert cost_opt.potential_savings_percentage == 20.0
        assert len(cost_opt.recommendations) == 2
        assert len(cost_opt.alternative_configurations) == 1
        assert len(cost_opt.cost_breakdown_by_provider) == 2
        assert cost_opt.roi_analysis["monthly_savings"] == 14400.0

    def test_cost_optimization_empty_collections(self):
        """Test CostOptimization with empty collections."""
        cost_opt = CostOptimization(
            current_cost_usd_per_hour=100.0,
            optimized_cost_usd_per_hour=100.0,
            potential_savings_percentage=0.0,
            recommendations=[],
            alternative_configurations=[],
            cost_breakdown_by_provider={},
            roi_analysis={},
        )

        assert len(cost_opt.recommendations) == 0
        assert len(cost_opt.alternative_configurations) == 0
        assert len(cost_opt.cost_breakdown_by_provider) == 0
        assert len(cost_opt.roi_analysis) == 0


class TestSustainabilityMetricsDataclass:
    """Tests for SustainabilityMetrics dataclass."""

    def test_sustainability_metrics_creation(self):
        """Test SustainabilityMetrics dataclass initialization."""
        metrics = SustainabilityMetrics(
            total_carbon_footprint_kg=100.0,
            carbon_per_compute_unit=2.5,
            renewable_energy_percentage=0.8,
            carbon_offset_cost_usd=20.0,
            sustainability_score=0.85,
            green_computing_recommendations=[
                "Use renewable energy data centers",
                "Optimize workload scheduling",
            ],
        )

        assert metrics.total_carbon_footprint_kg == 100.0
        assert metrics.carbon_per_compute_unit == 2.5
        assert metrics.renewable_energy_percentage == 0.8
        assert metrics.carbon_offset_cost_usd == 20.0
        assert metrics.sustainability_score == 0.85
        assert len(metrics.green_computing_recommendations) == 2

    def test_sustainability_metrics_zero_carbon(self):
        """Test SustainabilityMetrics with zero carbon footprint."""
        metrics = SustainabilityMetrics(
            total_carbon_footprint_kg=0.0,
            carbon_per_compute_unit=0.0,
            renewable_energy_percentage=1.0,
            carbon_offset_cost_usd=0.0,
            sustainability_score=1.0,
            green_computing_recommendations=[],
        )

        assert metrics.total_carbon_footprint_kg == 0.0
        assert metrics.renewable_energy_percentage == 1.0
        assert metrics.sustainability_score == 1.0
        assert len(metrics.green_computing_recommendations) == 0

    def test_sustainability_metrics_high_carbon(self):
        """Test SustainabilityMetrics with high carbon footprint."""
        metrics = SustainabilityMetrics(
            total_carbon_footprint_kg=500.0,
            carbon_per_compute_unit=10.0,
            renewable_energy_percentage=0.2,
            carbon_offset_cost_usd=100.0,
            sustainability_score=0.3,
            green_computing_recommendations=[
                "Migrate to green data centers",
                "Implement carbon-aware scheduling",
                "Purchase carbon offsets",
            ],
        )

        assert metrics.total_carbon_footprint_kg == 500.0
        assert metrics.sustainability_score == 0.3
        assert len(metrics.green_computing_recommendations) == 3
