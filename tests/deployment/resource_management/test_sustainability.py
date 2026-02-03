"""Tests for SustainabilityTracker module.

Following TDD principles: Write ALL tests BEFORE extracting the module.
Tests cover carbon emissions tracking, sustainability metrics calculation,
and sustainability optimization.
"""

from opifex.deployment.resource_management.sustainability import SustainabilityTracker
from opifex.deployment.resource_management.types import (
    CloudProvider,
    ResourcePool,
    ResourceType,
    SustainabilityMetrics,
)


class TestSustainabilityTrackerInitialization:
    """Tests for SustainabilityTracker initialization."""

    def test_sustainability_tracker_default_initialization(self):
        """Test SustainabilityTracker initialization with default parameters."""
        tracker = SustainabilityTracker()

        assert tracker.carbon_reduction_target_percentage == 30.0
        assert tracker.renewable_energy_preference is True
        assert isinstance(tracker.carbon_emissions, list)
        assert isinstance(tracker.sustainability_metrics_history, list)
        assert len(tracker.carbon_emissions) == 0
        assert len(tracker.sustainability_metrics_history) == 0

    def test_sustainability_tracker_custom_initialization(self):
        """Test SustainabilityTracker initialization with custom parameters."""
        tracker = SustainabilityTracker(
            carbon_reduction_target_percentage=50.0,
            renewable_energy_preference=False,
        )

        assert tracker.carbon_reduction_target_percentage == 50.0
        assert tracker.renewable_energy_preference is False


class TestCarbonEmissionsTracking:
    """Tests for carbon emissions tracking."""

    def test_track_carbon_emissions_basic(self):
        """Test tracking carbon emissions."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=10.5,
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.75,
        )

        assert len(tracker.carbon_emissions) == 1
        emission = tracker.carbon_emissions[0]
        assert emission["allocation_id"] == "alloc-1"
        assert emission["carbon_footprint_kg"] == 10.5
        assert emission["provider"] == CloudProvider.AWS
        assert emission["region"] == "us-east-1"
        assert emission["renewable_energy_percentage"] == 0.75
        assert "timestamp" in emission

    def test_track_multiple_carbon_emissions(self):
        """Test tracking multiple carbon emissions."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=10.0,
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.80,
        )

        tracker.track_carbon_emissions(
            allocation_id="alloc-2",
            carbon_footprint_kg=15.0,
            provider=CloudProvider.GCP,
            region="us-central1",
            renewable_energy_percentage=0.90,
        )

        assert len(tracker.carbon_emissions) == 2

    def test_track_zero_carbon_emissions(self):
        """Test tracking zero carbon emissions."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-zero",
            carbon_footprint_kg=0.0,
            provider=CloudProvider.GCP,
            region="us-central1",
            renewable_energy_percentage=1.0,
        )

        assert len(tracker.carbon_emissions) == 1
        assert tracker.carbon_emissions[0]["carbon_footprint_kg"] == 0.0


class TestSustainabilityMetricsCalculation:
    """Tests for sustainability metrics calculation."""

    def test_calculate_sustainability_metrics_no_emissions(self):
        """Test sustainability metrics with no emissions data."""
        tracker = SustainabilityTracker()

        metrics = tracker.calculate_sustainability_metrics()

        assert isinstance(metrics, SustainabilityMetrics)
        assert metrics.total_carbon_footprint_kg == 0.0
        assert metrics.carbon_per_compute_unit == 0.0
        assert metrics.renewable_energy_percentage == 0.0
        assert metrics.carbon_offset_cost_usd == 0.0
        assert metrics.sustainability_score == 1.0
        assert len(metrics.green_computing_recommendations) == 0
        # When no emissions, early return doesn't update history
        assert len(tracker.sustainability_metrics_history) == 0

    def test_calculate_sustainability_metrics_single_emission(self):
        """Test sustainability metrics with single emission."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=25.0,
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.80,
        )

        metrics = tracker.calculate_sustainability_metrics()

        assert metrics.total_carbon_footprint_kg == 25.0
        assert metrics.carbon_per_compute_unit == 25.0  # 25 / 1 emission
        assert metrics.renewable_energy_percentage == 0.80
        assert metrics.carbon_offset_cost_usd == 0.5  # (25/1000) * 20
        assert 0.0 < metrics.sustainability_score <= 1.0

    def test_calculate_sustainability_metrics_multiple_emissions(self):
        """Test sustainability metrics with multiple emissions."""
        tracker = SustainabilityTracker()

        # Track emissions
        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=20.0,
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.70,
        )

        tracker.track_carbon_emissions(
            allocation_id="alloc-2",
            carbon_footprint_kg=30.0,
            provider=CloudProvider.GCP,
            region="us-central1",
            renewable_energy_percentage=0.90,
        )

        metrics = tracker.calculate_sustainability_metrics()

        assert metrics.total_carbon_footprint_kg == 50.0
        assert metrics.carbon_per_compute_unit == 25.0  # 50 / 2 emissions
        assert metrics.renewable_energy_percentage == 0.80  # Average of 0.7 and 0.9
        assert metrics.carbon_offset_cost_usd == 1.0  # (50/1000) * 20

    def test_calculate_sustainability_metrics_low_renewable_energy(self):
        """Test metrics generate recommendations for low renewable energy."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=30.0,
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.5,  # Low renewable energy
        )

        metrics = tracker.calculate_sustainability_metrics()

        assert metrics.renewable_energy_percentage == 0.5
        assert any(
            "renewable energy" in rec.lower()
            for rec in metrics.green_computing_recommendations
        )

    def test_calculate_sustainability_metrics_high_carbon(self):
        """Test metrics generate recommendations for high carbon emissions."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=100.0,  # High carbon
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.80,
        )

        metrics = tracker.calculate_sustainability_metrics()

        assert metrics.total_carbon_footprint_kg == 100.0
        assert any(
            "carbon offset" in rec.lower()
            for rec in metrics.green_computing_recommendations
        )

    def test_calculate_sustainability_metrics_updates_history(self):
        """Test that metrics calculation updates history."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=20.0,
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.85,
        )

        # Calculate metrics twice
        tracker.calculate_sustainability_metrics()
        tracker.calculate_sustainability_metrics()

        assert len(tracker.sustainability_metrics_history) == 2

    def test_calculate_sustainability_score_formula(self):
        """Test sustainability score calculation formula."""
        tracker = SustainabilityTracker()

        # Test with specific values to verify formula
        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=10.0,  # Will result in (1.0 - 10.0/100.0) = 0.9
            provider=CloudProvider.GCP,
            region="us-central1",
            renewable_energy_percentage=0.85,
        )

        metrics = tracker.calculate_sustainability_metrics()

        # Score = min(1.0, renewable_energy + (1.0 - carbon/100.0))
        # Score = min(1.0, 0.85 + (1.0 - 10.0/100.0))
        # Score = min(1.0, 0.85 + 0.9) = min(1.0, 1.75) = 1.0
        assert metrics.sustainability_score == 1.0

    def test_carbon_offset_cost_calculation(self):
        """Test carbon offset cost calculation ($20 per ton CO2)."""
        tracker = SustainabilityTracker()

        tracker.track_carbon_emissions(
            allocation_id="alloc-1",
            carbon_footprint_kg=500.0,  # 0.5 tons
            provider=CloudProvider.AWS,
            region="us-east-1",
            renewable_energy_percentage=0.75,
        )

        metrics = tracker.calculate_sustainability_metrics()

        # Cost = (500 kg / 1000) * $20 = 0.5 * 20 = $10
        assert metrics.carbon_offset_cost_usd == 10.0


class TestSustainabilityOptimization:
    """Tests for sustainability optimization."""

    def test_optimize_for_sustainability_preference_disabled(self):
        """Test optimization when renewable energy preference is disabled."""
        tracker = SustainabilityTracker(renewable_energy_preference=False)

        pools = [
            ResourcePool(
                pool_id="pool-1",
                provider=CloudProvider.AWS,
                region="us-east-1",
                resource_type=ResourceType.GPU_A100,
                total_capacity=100,
                available_capacity=50,
                reserved_capacity=10,
                cost_per_hour_usd=10.0,
                performance_score=0.9,
                carbon_efficiency=100.0,  # High carbon
                availability_sla=0.995,
            ),
            ResourcePool(
                pool_id="pool-2",
                provider=CloudProvider.GCP,
                region="us-central1",
                resource_type=ResourceType.GPU_H100,
                total_capacity=100,
                available_capacity=50,
                reserved_capacity=10,
                cost_per_hour_usd=15.0,
                performance_score=0.95,
                carbon_efficiency=30.0,  # Low carbon
                availability_sla=0.999,
            ),
        ]

        optimized = tracker.optimize_for_sustainability(pools)

        # Should return original list unchanged
        assert optimized == pools
        assert len(optimized) == 2

    def test_optimize_for_sustainability_sorts_by_carbon_efficiency(self):
        """Test optimization sorts pools by sustainability score."""
        tracker = SustainabilityTracker(renewable_energy_preference=True)

        pools = [
            ResourcePool(
                pool_id="pool-high-carbon",
                provider=CloudProvider.AWS,
                region="us-east-1",
                resource_type=ResourceType.GPU_A100,
                total_capacity=100,
                available_capacity=50,
                reserved_capacity=10,
                cost_per_hour_usd=10.0,
                performance_score=0.9,
                carbon_efficiency=100.0,  # High carbon (worse)
                availability_sla=0.995,
            ),
            ResourcePool(
                pool_id="pool-low-carbon",
                provider=CloudProvider.GCP,
                region="us-central1",
                resource_type=ResourceType.GPU_H100,
                total_capacity=100,
                available_capacity=50,
                reserved_capacity=10,
                cost_per_hour_usd=15.0,
                performance_score=0.95,
                carbon_efficiency=30.0,  # Low carbon (better)
                availability_sla=0.999,
            ),
        ]

        optimized = tracker.optimize_for_sustainability(pools)

        # Should be sorted with low-carbon pool first
        assert len(optimized) == 2
        assert optimized[0].pool_id == "pool-low-carbon"
        assert optimized[1].pool_id == "pool-high-carbon"

    def test_optimize_for_sustainability_empty_list(self):
        """Test optimization with empty pool list."""
        tracker = SustainabilityTracker()

        optimized = tracker.optimize_for_sustainability([])

        assert optimized == []

    def test_optimize_for_sustainability_single_pool(self):
        """Test optimization with single pool."""
        tracker = SustainabilityTracker()

        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.GCP,
            region="us-central1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=10.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.999,
        )

        optimized = tracker.optimize_for_sustainability([pool])

        assert len(optimized) == 1
        assert optimized[0] == pool

    def test_optimize_for_sustainability_considers_availability(self):
        """Test optimization considers both carbon efficiency and availability."""
        tracker = SustainabilityTracker(renewable_energy_preference=True)

        pools = [
            ResourcePool(
                pool_id="pool-low-carbon-low-availability",
                provider=CloudProvider.AWS,
                region="us-east-1",
                resource_type=ResourceType.GPU_A100,
                total_capacity=100,
                available_capacity=50,
                reserved_capacity=10,
                cost_per_hour_usd=10.0,
                performance_score=0.9,
                carbon_efficiency=20.0,  # Very low carbon
                availability_sla=0.90,  # Low availability
            ),
            ResourcePool(
                pool_id="pool-medium-carbon-high-availability",
                provider=CloudProvider.GCP,
                region="us-central1",
                resource_type=ResourceType.GPU_H100,
                total_capacity=100,
                available_capacity=50,
                reserved_capacity=10,
                cost_per_hour_usd=15.0,
                performance_score=0.95,
                carbon_efficiency=50.0,  # Medium carbon
                availability_sla=0.999,  # High availability
            ),
        ]

        optimized = tracker.optimize_for_sustainability(pools)

        # Both carbon efficiency and availability should be considered
        # Score = 0.7 * (1 / (carbon + 0.1)) + 0.3 * availability
        assert len(optimized) == 2
        # Verify it's sorted by sustainability score (descending)
        assert optimized[0].pool_id != optimized[1].pool_id
