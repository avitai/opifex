"""Comprehensive tests for GlobalResourceManager.

Test-driven development (TDD) approach: These tests are written BEFORE
extracting the GlobalResourceManager module.

This module integrates all other managers for end-to-end resource management.
"""

import asyncio
import contextlib

import pytest

from opifex.deployment.resource_management.types import ResourceType


# Import after extraction
# from opifex.deployment.resource_management.global_manager import GlobalResourceManager


class TestGlobalResourceManagerInitialization:
    """Test GlobalResourceManager initialization."""

    def test_initialization(self, global_resource_manager):
        """Test GlobalResourceManager initialization with all dependencies."""
        assert global_resource_manager.resource_orchestrator is not None
        assert global_resource_manager.gpu_pool_manager is not None
        assert global_resource_manager.cost_controller is not None
        assert global_resource_manager.sustainability_tracker is not None
        assert global_resource_manager.is_monitoring is False

    def test_all_managers_connected(self, global_resource_manager):
        """Test that all sub-managers are properly connected."""
        # Check resource orchestrator
        assert hasattr(global_resource_manager.resource_orchestrator, "resource_pools")

        # Check GPU pool manager
        assert hasattr(global_resource_manager.gpu_pool_manager, "gpu_pools")

        # Check cost controller
        assert hasattr(global_resource_manager.cost_controller, "daily_spending")

        # Check sustainability tracker
        assert hasattr(
            global_resource_manager.sustainability_tracker, "carbon_emissions"
        )


class TestIntelligentResourceAllocation:
    """Test intelligent resource allocation with full integration."""

    @pytest.mark.asyncio
    async def test_allocate_resources_success(
        self, global_resource_manager_with_setup, sample_resource_requirements
    ):
        """Test successful end-to-end resource allocation."""
        result = await global_resource_manager_with_setup.allocate_resources_with_intelligence(
            sample_resource_requirements
        )

        # Verify result structure
        assert "allocation" in result
        assert "gpu_allocations" in result
        assert "cost_estimate_usd" in result
        assert "carbon_footprint_kg" in result
        assert "performance_estimate" in result
        assert "sustainability_optimized" in result
        assert "allocation_strategy" in result

        # Verify allocation was created
        allocation = result["allocation"]
        assert allocation.allocation_id is not None
        assert allocation.requested_resources == sample_resource_requirements

    @pytest.mark.asyncio
    async def test_allocate_resources_with_constraints(
        self, global_resource_manager_with_setup
    ):
        """Test resource allocation with specific constraints."""
        requirements = {ResourceType.GPU_A100: 1}
        constraints = {"max_cost_usd_per_hour": 50.0}

        result = await global_resource_manager_with_setup.allocate_resources_with_intelligence(
            requirements, constraints
        )

        assert result["allocation"] is not None
        assert result["cost_estimate_usd"] <= 50.0

    @pytest.mark.asyncio
    async def test_allocate_resources_without_sustainability(
        self, global_resource_manager_with_setup, sample_resource_requirements
    ):
        """Test allocation without sustainability priority."""
        result = await global_resource_manager_with_setup.allocate_resources_with_intelligence(
            sample_resource_requirements, sustainability_priority=False
        )

        assert result["sustainability_optimized"] is False

    @pytest.mark.asyncio
    async def test_gpu_allocation_integration(
        self, global_resource_manager_with_gpu_pools
    ):
        """Test that GPU allocations are handled properly."""
        requirements = {ResourceType.GPU_A100: 2}

        result = await global_resource_manager_with_gpu_pools.allocate_resources_with_intelligence(
            requirements
        )

        # Should have GPU allocations
        assert "gpu_allocations" in result
        # GPU allocation happens if GPU pools exist
        if result["gpu_allocations"]:
            assert ResourceType.GPU_A100.value in result["gpu_allocations"]

    @pytest.mark.asyncio
    async def test_cost_tracking_integration(
        self, global_resource_manager_with_setup, sample_resource_requirements
    ):
        """Test that costs are tracked during allocation."""
        initial_cost_history_len = len(
            global_resource_manager_with_setup.cost_controller.cost_history
        )

        await global_resource_manager_with_setup.allocate_resources_with_intelligence(
            sample_resource_requirements
        )

        # Cost should be tracked
        assert (
            len(global_resource_manager_with_setup.cost_controller.cost_history)
            > initial_cost_history_len
        )

    @pytest.mark.asyncio
    async def test_sustainability_tracking_integration(
        self, global_resource_manager_with_setup, sample_resource_requirements
    ):
        """Test that sustainability is tracked during allocation."""
        initial_emissions_len = len(
            global_resource_manager_with_setup.sustainability_tracker.carbon_emissions
        )

        await global_resource_manager_with_setup.allocate_resources_with_intelligence(
            sample_resource_requirements
        )

        # Carbon emissions should be tracked
        assert (
            len(
                global_resource_manager_with_setup.sustainability_tracker.carbon_emissions
            )
            > initial_emissions_len
        )


class TestResourceMonitoring:
    """Test asynchronous resource monitoring."""

    @pytest.mark.asyncio
    async def test_start_and_stop_monitoring(self, global_resource_manager):
        """Test starting and stopping resource monitoring."""
        assert global_resource_manager.is_monitoring is False

        # Start monitoring in background
        monitoring_task = asyncio.create_task(
            global_resource_manager.start_resource_monitoring()
        )

        # Give it a moment to start
        await asyncio.sleep(0.1)
        assert global_resource_manager.is_monitoring is True

        # Stop monitoring
        await global_resource_manager.stop_resource_monitoring()
        await asyncio.sleep(0.1)

        # Wait for task to complete
        monitoring_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitoring_task

        assert global_resource_manager.is_monitoring is False

    @pytest.mark.asyncio
    async def test_monitoring_runs_continuously(
        self, global_resource_manager_with_setup
    ):
        """Test that monitoring runs continuously until stopped."""
        # Start monitoring
        monitoring_task = asyncio.create_task(
            global_resource_manager_with_setup.start_resource_monitoring()
        )

        # Let it run for a short time
        await asyncio.sleep(0.2)

        # Should still be running
        assert global_resource_manager_with_setup.is_monitoring is True
        assert not monitoring_task.done()

        # Stop it
        await global_resource_manager_with_setup.stop_resource_monitoring()
        await asyncio.sleep(0.1)

        # Clean up
        monitoring_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitoring_task


class TestComprehensiveStatus:
    """Test comprehensive status reporting."""

    def test_get_comprehensive_status_empty(self, global_resource_manager):
        """Test status with no activity."""
        status = global_resource_manager.get_comprehensive_resource_status()

        # Check all required sections
        assert "resource_orchestration" in status
        assert "gpu_management" in status
        assert "cost_optimization" in status
        assert "cost_analytics" in status
        assert "sustainability" in status
        assert "monitoring_active" in status
        assert "status" in status

        assert status["status"] == "operational"
        assert status["monitoring_active"] is False

    def test_get_comprehensive_status_with_activity(
        self, global_resource_manager_with_setup, sample_resource_requirements
    ):
        """Test status after some activity."""
        # Create some activity
        asyncio.run(
            global_resource_manager_with_setup.allocate_resources_with_intelligence(
                sample_resource_requirements
            )
        )

        status = global_resource_manager_with_setup.get_comprehensive_resource_status()

        # Should have activity data
        assert status["resource_orchestration"]["active_allocations"] > 0
        assert status["resource_orchestration"]["available_pools"] > 0

    def test_status_includes_orchestrator_info(
        self, global_resource_manager_with_setup
    ):
        """Test that status includes orchestrator information."""
        status = global_resource_manager_with_setup.get_comprehensive_resource_status()

        orchestration = status["resource_orchestration"]
        assert "active_allocations" in orchestration
        assert "available_pools" in orchestration
        assert "optimization_objective" in orchestration

    def test_status_includes_gpu_stats(self, global_resource_manager_with_gpu_pools):
        """Test that status includes GPU management stats."""
        status = (
            global_resource_manager_with_gpu_pools.get_comprehensive_resource_status()
        )

        gpu_mgmt = status["gpu_management"]
        assert "total_pools" in gpu_mgmt
        assert "total_gpus" in gpu_mgmt
        assert "total_memory_gb" in gpu_mgmt

    def test_status_includes_cost_info(self, global_resource_manager_with_setup):
        """Test that status includes cost information."""
        status = global_resource_manager_with_setup.get_comprehensive_resource_status()

        cost_opt = status["cost_optimization"]
        assert "current_cost_usd_per_hour" in cost_opt
        assert "potential_savings_percentage" in cost_opt
        assert "recommendations_count" in cost_opt

        cost_analytics = status["cost_analytics"]
        # Should have either data or error message
        assert cost_analytics is not None

    def test_status_includes_sustainability_metrics(
        self, global_resource_manager_with_setup
    ):
        """Test that status includes sustainability metrics."""
        status = global_resource_manager_with_setup.get_comprehensive_resource_status()

        sustainability = status["sustainability"]
        assert "total_carbon_footprint_kg" in sustainability
        assert "renewable_energy_percentage" in sustainability
        assert "sustainability_score" in sustainability
        assert "carbon_offset_cost_usd" in sustainability


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_scenario(
        self, global_resource_manager_with_gpu_pools
    ):
        """Test complete lifecycle: allocate, monitor status, deallocate."""
        manager = global_resource_manager_with_gpu_pools

        # Step 1: Allocate resources
        requirements = {ResourceType.GPU_A100: 1, ResourceType.MEMORY: 10}
        result = await manager.allocate_resources_with_intelligence(requirements)

        assert result["allocation"] is not None
        allocation_id = result["allocation"].allocation_id

        # Step 2: Check status
        status = manager.get_comprehensive_resource_status()
        assert status["resource_orchestration"]["active_allocations"] > 0

        # Step 3: Verify tracking happened
        assert len(manager.cost_controller.cost_history) > 0
        assert len(manager.sustainability_tracker.carbon_emissions) > 0

        # Verify allocation_id is tracked
        assert allocation_id in [
            entry["allocation_id"] for entry in manager.cost_controller.cost_history
        ]

    @pytest.mark.asyncio
    async def test_multiple_allocations(self, global_resource_manager_with_setup):
        """Test handling multiple concurrent allocations."""
        manager = global_resource_manager_with_setup

        # Create multiple allocations
        req1 = {ResourceType.GPU_A100: 1}
        req2 = {ResourceType.CPU_INTEL: 4}

        result1 = await manager.allocate_resources_with_intelligence(req1)
        result2 = await manager.allocate_resources_with_intelligence(req2)

        assert result1["allocation"] is not None
        assert result2["allocation"] is not None
        assert (
            result1["allocation"].allocation_id != result2["allocation"].allocation_id
        )

        # Check that both are tracked
        status = manager.get_comprehensive_resource_status()
        assert status["resource_orchestration"]["active_allocations"] >= 2

    @pytest.mark.asyncio
    async def test_cost_budget_monitoring(self, global_resource_manager_with_setup):
        """Test that budget alerts work in integrated scenario."""
        manager = global_resource_manager_with_setup

        # Allocate resources
        requirements = {ResourceType.GPU_A100: 2}
        await manager.allocate_resources_with_intelligence(requirements)

        # Check budget alerts
        alerts = manager.cost_controller.check_budget_alerts()
        assert "budget_violation" in alerts
        assert "budget_warning" in alerts
        assert "daily_spending" in alerts


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_allocation_with_no_pools(self, global_resource_manager):
        """Test allocation when no pools are registered."""
        requirements = {ResourceType.GPU_A100: 1}

        with pytest.raises(ValueError, match=r".*"):
            await global_resource_manager.allocate_resources_with_intelligence(
                requirements
            )

    @pytest.mark.asyncio
    async def test_empty_resource_requirements(
        self, global_resource_manager_with_setup
    ):
        """Test allocation with empty requirements."""
        # This should still work, just return empty allocation
        result = await global_resource_manager_with_setup.allocate_resources_with_intelligence(
            {}
        )

        # Should handle gracefully
        assert "allocation" in result or result is not None

    def test_status_before_any_activity(self, global_resource_manager):
        """Test getting status before any allocations."""
        status = global_resource_manager.get_comprehensive_resource_status()

        # Should return valid structure even with no activity
        assert status["status"] == "operational"
        assert status["resource_orchestration"]["active_allocations"] == 0
