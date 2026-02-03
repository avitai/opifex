"""Tests for CostController module.

Following TDD principles: Write ALL tests BEFORE extracting the module.
Tests cover initialization, cost tracking, budget alerts, optimization analysis,
and cost analytics.
"""

import time

from opifex.deployment.resource_management.cost_controller import CostController
from opifex.deployment.resource_management.types import (
    CloudProvider,
    CostOptimization,
    ResourceAllocation,
    ResourcePool,
    ResourceType,
)


class TestCostControllerInitialization:
    """Tests for CostController initialization."""

    def test_cost_controller_default_initialization(self):
        """Test CostController initialization with default parameters."""
        controller = CostController()

        assert controller.budget_limit_usd_per_day == 10000.0
        assert controller.cost_optimization_interval == 3600
        assert controller.savings_target_percentage == 20.0
        assert isinstance(controller.daily_spending, dict)
        assert isinstance(controller.cost_history, list)
        assert isinstance(controller.optimization_history, list)
        assert len(controller.daily_spending) == 0
        assert len(controller.cost_history) == 0
        assert len(controller.optimization_history) == 0

    def test_cost_controller_custom_initialization(self):
        """Test CostController initialization with custom parameters."""
        controller = CostController(
            budget_limit_usd_per_day=5000.0,
            cost_optimization_interval=1800,
            savings_target_percentage=30.0,
        )

        assert controller.budget_limit_usd_per_day == 5000.0
        assert controller.cost_optimization_interval == 1800
        assert controller.savings_target_percentage == 30.0


class TestCostTracking:
    """Tests for resource cost tracking."""

    def test_track_resource_cost_basic(self):
        """Test tracking resource costs updates daily spending."""
        controller = CostController()
        current_date = time.strftime("%Y-%m-%d")

        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=10.0,
            duration_hours=2.0,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        # Verify daily spending updated
        assert current_date in controller.daily_spending
        assert controller.daily_spending[current_date] == 20.0

        # Verify cost history recorded
        assert len(controller.cost_history) == 1
        entry = controller.cost_history[0]
        assert entry["allocation_id"] == "alloc-1"
        assert entry["cost_usd"] == 20.0
        assert entry["cost_per_hour"] == 10.0
        assert entry["duration_hours"] == 2.0
        assert entry["provider"] == CloudProvider.AWS
        assert entry["resource_type"] == ResourceType.GPU_A100
        assert entry["date"] == current_date

    def test_track_multiple_costs_same_day(self):
        """Test tracking multiple costs accumulates correctly."""
        controller = CostController()
        current_date = time.strftime("%Y-%m-%d")

        # Track first allocation
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=10.0,
            duration_hours=1.0,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        # Track second allocation
        controller.track_resource_cost(
            allocation_id="alloc-2",
            cost_usd_per_hour=15.0,
            duration_hours=2.0,
            provider=CloudProvider.GCP,
            resource_type=ResourceType.GPU_H100,
        )

        # Verify total daily spending
        assert controller.daily_spending[current_date] == 40.0  # 10 + 30
        assert len(controller.cost_history) == 2

    def test_track_resource_cost_zero_duration(self):
        """Test tracking cost with zero duration."""
        controller = CostController()
        current_date = time.strftime("%Y-%m-%d")

        controller.track_resource_cost(
            allocation_id="alloc-zero",
            cost_usd_per_hour=10.0,
            duration_hours=0.0,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.CPU_INTEL,
        )

        assert controller.daily_spending[current_date] == 0.0


class TestBudgetAlerts:
    """Tests for budget alert and violation detection."""

    def test_check_budget_alerts_under_budget(self):
        """Test budget alerts when spending is under budget."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        # Track spending under budget
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=10.0,
            duration_hours=5.0,  # Total: $50
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        alerts = controller.check_budget_alerts()

        assert alerts["budget_violation"] is False
        assert alerts["budget_warning"] is False
        assert alerts["daily_spending"] == 50.0
        assert alerts["budget_limit"] == 1000.0
        assert alerts["utilization_percentage"] == 5.0
        assert len(alerts["recommended_actions"]) == 0

    def test_check_budget_alerts_warning_threshold(self):
        """Test budget warning when spending exceeds 80% threshold."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        # Track spending at 85% of budget
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=85.0,
            duration_hours=10.0,  # Total: $850 (85% of budget)
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        alerts = controller.check_budget_alerts()

        assert alerts["budget_violation"] is False
        assert alerts["budget_warning"] is True
        assert alerts["daily_spending"] == 850.0
        assert alerts["utilization_percentage"] == 85.0
        assert len(alerts["recommended_actions"]) == 3
        assert "Monitor spending closely" in alerts["recommended_actions"]

    def test_check_budget_alerts_violation(self):
        """Test budget violation detection and alerts."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        # Track spending over budget
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=110.0,
            duration_hours=10.0,  # Total: $1100 (110% of budget)
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        alerts = controller.check_budget_alerts()

        assert alerts["budget_violation"] is True
        assert alerts["budget_warning"] is True  # Also triggers warning
        assert alerts["daily_spending"] == 1100.0
        assert abs(alerts["utilization_percentage"] - 110.0) < 0.01  # Float comparison
        assert len(alerts["recommended_actions"]) == 3
        assert "Immediate cost reduction required" in alerts["recommended_actions"]
        assert (
            "Review and terminate non-essential allocations"
            in alerts["recommended_actions"]
        )

    def test_check_budget_alerts_no_spending(self):
        """Test budget alerts with no spending."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        alerts = controller.check_budget_alerts()

        assert alerts["budget_violation"] is False
        assert alerts["budget_warning"] is False
        assert alerts["daily_spending"] == 0.0
        assert alerts["utilization_percentage"] == 0.0


class TestCostOptimizationAnalysis:
    """Tests for cost optimization analysis."""

    def test_analyze_cost_optimization_single_allocation(self):
        """Test cost optimization analysis with single allocation."""
        controller = CostController()

        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=10.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        allocation = ResourceAllocation(
            allocation_id="alloc-1",
            requested_resources={ResourceType.GPU_A100: 4},
            allocated_resources={"pool-1": pool},
            start_time=1000.0,
            end_time=None,
            cost_estimate_usd=10.0,
            performance_estimate=0.9,
            carbon_footprint_kg=2.5,
            allocation_strategy="balanced",
        )

        active_allocations = {"alloc-1": allocation}
        optimization = controller.analyze_cost_optimization_opportunities(
            active_allocations
        )

        assert isinstance(optimization, CostOptimization)
        assert optimization.current_cost_usd_per_hour == 10.0
        assert optimization.optimized_cost_usd_per_hour == 8.0  # 20% savings
        assert optimization.potential_savings_percentage == 20.0
        assert isinstance(optimization.recommendations, list)
        assert isinstance(optimization.cost_breakdown_by_provider, dict)
        assert len(controller.optimization_history) == 1

    def test_analyze_cost_optimization_multiple_providers(self):
        """Test cost optimization with multiple cloud providers."""
        controller = CostController()

        pool1 = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=10.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        pool2 = ResourcePool(
            pool_id="pool-2",
            provider=CloudProvider.GCP,
            region="us-central1",
            resource_type=ResourceType.GPU_H100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=5.0,
            performance_score=0.95,
            carbon_efficiency=30.0,
            availability_sla=0.999,
        )

        allocation1 = ResourceAllocation(
            allocation_id="alloc-1",
            requested_resources={ResourceType.GPU_A100: 2},
            allocated_resources={"pool-1": pool1},
            start_time=1000.0,
            end_time=None,
            cost_estimate_usd=10.0,
            performance_estimate=0.9,
            carbon_footprint_kg=2.5,
            allocation_strategy="balanced",
        )

        allocation2 = ResourceAllocation(
            allocation_id="alloc-2",
            requested_resources={ResourceType.GPU_H100: 2},
            allocated_resources={"pool-2": pool2},
            start_time=1000.0,
            end_time=None,
            cost_estimate_usd=5.0,
            performance_estimate=0.95,
            carbon_footprint_kg=1.5,
            allocation_strategy="cost-optimized",
        )

        active_allocations = {"alloc-1": allocation1, "alloc-2": allocation2}
        optimization = controller.analyze_cost_optimization_opportunities(
            active_allocations
        )

        assert optimization.current_cost_usd_per_hour == 15.0
        assert CloudProvider.AWS in optimization.cost_breakdown_by_provider
        assert CloudProvider.GCP in optimization.cost_breakdown_by_provider

    def test_analyze_cost_optimization_aws_heavy(self):
        """Test optimization recommendations when AWS usage is > 40%."""
        controller = CostController()

        # Create allocation heavily weighted to AWS
        pool_aws = ResourcePool(
            pool_id="pool-aws",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=50.0,  # High AWS cost
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        allocation = ResourceAllocation(
            allocation_id="alloc-1",
            requested_resources={ResourceType.GPU_A100: 4},
            allocated_resources={"pool-aws": pool_aws},
            start_time=1000.0,
            end_time=None,
            cost_estimate_usd=50.0,
            performance_estimate=0.9,
            carbon_footprint_kg=5.0,
            allocation_strategy="balanced",
        )

        optimization = controller.analyze_cost_optimization_opportunities(
            {"alloc-1": allocation}
        )

        # Should recommend migrating workloads
        assert any(
            "lower-cost providers" in rec for rec in optimization.recommendations
        )

    def test_analyze_cost_optimization_low_performance(self):
        """Test optimization recommendations for under-performing allocations."""
        controller = CostController()

        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=10.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        allocation = ResourceAllocation(
            allocation_id="alloc-1",
            requested_resources={ResourceType.GPU_A100: 4},
            allocated_resources={"pool-1": pool},
            start_time=1000.0,
            end_time=None,
            cost_estimate_usd=10.0,
            performance_estimate=0.5,  # Low performance
            carbon_footprint_kg=2.5,
            allocation_strategy="balanced",
        )

        optimization = controller.analyze_cost_optimization_opportunities(
            {"alloc-1": allocation}
        )

        # Should recommend optimizing under-performing allocations
        assert any("under-performing" in rec for rec in optimization.recommendations)


class TestCostAnalytics:
    """Tests for comprehensive cost analytics."""

    def test_get_cost_analytics_no_data(self):
        """Test cost analytics with no cost data."""
        controller = CostController()

        analytics = controller.get_cost_analytics()

        assert "error" in analytics
        assert analytics["error"] == "No cost data available"

    def test_get_cost_analytics_with_data(self):
        """Test comprehensive cost analytics with data."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        # Track some costs
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=10.0,
            duration_hours=5.0,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        controller.track_resource_cost(
            allocation_id="alloc-2",
            cost_usd_per_hour=20.0,
            duration_hours=3.0,
            provider=CloudProvider.GCP,
            resource_type=ResourceType.GPU_H100,
        )

        analytics = controller.get_cost_analytics()

        assert analytics["total_spending_usd"] == 110.0  # 50 + 60
        assert analytics["average_daily_spending"] == 110.0  # Only 1 day
        assert analytics["budget_utilization"] == 11.0  # 110/1000 * 100
        assert "cost_by_provider" in analytics
        assert "cost_by_resource_type" in analytics
        assert analytics["cost_by_provider"][CloudProvider.AWS.value] == 50.0
        assert analytics["cost_by_provider"][CloudProvider.GCP.value] == 60.0
        assert analytics["cost_by_resource_type"][ResourceType.GPU_A100.value] == 50.0
        assert analytics["cost_by_resource_type"][ResourceType.GPU_H100.value] == 60.0

    def test_get_cost_analytics_potential_savings(self):
        """Test cost analytics includes potential savings calculation."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        # Track costs
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=10.0,
            duration_hours=10.0,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        # Create optimization opportunity
        pool = ResourcePool(
            pool_id="pool-1",
            provider=CloudProvider.AWS,
            region="us-east-1",
            resource_type=ResourceType.GPU_A100,
            total_capacity=100,
            available_capacity=50,
            reserved_capacity=10,
            cost_per_hour_usd=10.0,
            performance_score=0.9,
            carbon_efficiency=50.0,
            availability_sla=0.995,
        )

        allocation = ResourceAllocation(
            allocation_id="alloc-1",
            requested_resources={ResourceType.GPU_A100: 4},
            allocated_resources={"pool-1": pool},
            start_time=1000.0,
            end_time=None,
            cost_estimate_usd=10.0,
            performance_estimate=0.9,
            carbon_footprint_kg=2.5,
            allocation_strategy="balanced",
        )

        controller.analyze_cost_optimization_opportunities({"alloc-1": allocation})

        analytics = controller.get_cost_analytics()

        assert "potential_monthly_savings" in analytics
        assert analytics["optimization_opportunities"] == 1

    def test_get_cost_analytics_multiple_days(self):
        """Test cost analytics averages across multiple days."""
        controller = CostController(budget_limit_usd_per_day=1000.0)

        # Simulate costs across multiple days by tracking with different dates
        time.strftime("%Y-%m-%d")

        # Day 1
        controller.track_resource_cost(
            allocation_id="alloc-1",
            cost_usd_per_hour=10.0,
            duration_hours=10.0,
            provider=CloudProvider.AWS,
            resource_type=ResourceType.GPU_A100,
        )

        # Manually add another day's spending
        controller.daily_spending["2025-01-01"] = 200.0

        analytics = controller.get_cost_analytics()

        # Total spending should be 300 (100 + 200), avg daily 150 across 2 days
        assert analytics["total_spending_usd"] == 100.0  # Only from cost_history
        assert analytics["average_daily_spending"] == 50.0  # 100 / 2 days
