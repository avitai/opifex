"""Cost optimization and budget management for multi-cloud resources.

This module implements cost tracking, budget alerts, optimization analysis,
and comprehensive cost analytics for resource management.
"""

import time
from typing import Any

from .types import (
    CloudProvider,
    CostOptimization,
    ResourceAllocation,
    ResourceType,
)


class CostController:
    """Cost optimization and budget management for multi-cloud resources."""

    def __init__(
        self,
        budget_limit_usd_per_day: float = 10000.0,
        cost_optimization_interval: int = 3600,  # 1 hour
        savings_target_percentage: float = 20.0,
    ):
        self.budget_limit_usd_per_day = budget_limit_usd_per_day
        self.cost_optimization_interval = cost_optimization_interval
        self.savings_target_percentage = savings_target_percentage

        self.daily_spending: dict[str, float] = {}  # date -> spending
        self.cost_history: list[dict[str, Any]] = []
        self.optimization_history: list[CostOptimization] = []

    def track_resource_cost(
        self,
        allocation_id: str,
        cost_usd_per_hour: float,
        duration_hours: float,
        provider: CloudProvider,
        resource_type: ResourceType,
    ) -> None:
        """Track cost for resource allocation."""
        total_cost = cost_usd_per_hour * duration_hours
        current_date = time.strftime("%Y-%m-%d")

        # Update daily spending
        if current_date not in self.daily_spending:
            self.daily_spending[current_date] = 0.0
        self.daily_spending[current_date] += total_cost

        # Record in cost history
        self.cost_history.append(
            {
                "timestamp": time.time(),
                "allocation_id": allocation_id,
                "cost_usd": total_cost,
                "cost_per_hour": cost_usd_per_hour,
                "duration_hours": duration_hours,
                "provider": provider,
                "resource_type": resource_type,
                "date": current_date,
            }
        )

    def analyze_cost_optimization_opportunities(
        self, active_allocations: dict[str, ResourceAllocation]
    ) -> CostOptimization:
        """Analyze opportunities for cost optimization."""
        current_hourly_cost = sum(
            alloc.cost_estimate_usd for alloc in active_allocations.values()
        )

        # Calculate potential savings
        recommendations = []
        alternative_configs = []
        cost_breakdown = dict.fromkeys(CloudProvider, 0.0)

        # Analyze by provider
        for allocation in active_allocations.values():
            for _pool_id, pool in allocation.allocated_resources.items():
                cost_breakdown[pool.provider] += pool.cost_per_hour_usd

        # Generate recommendations
        if cost_breakdown[CloudProvider.AWS] > current_hourly_cost * 0.4:
            recommendations.append(
                "Consider migrating some workloads to lower-cost providers"
            )

        if any(
            alloc.performance_estimate < 0.7 for alloc in active_allocations.values()
        ):
            recommendations.append(
                "Optimize under-performing allocations for better cost efficiency"
            )

        # Calculate optimized cost (simplified)
        optimized_cost = current_hourly_cost * 0.8  # Assume 20% savings potential

        cost_optimization = CostOptimization(
            current_cost_usd_per_hour=current_hourly_cost,
            optimized_cost_usd_per_hour=optimized_cost,
            potential_savings_percentage=20.0,
            recommendations=recommendations,
            alternative_configurations=alternative_configs,
            cost_breakdown_by_provider=cost_breakdown,
            roi_analysis={"current_efficiency": 0.75, "optimized_efficiency": 0.90},
        )

        self.optimization_history.append(cost_optimization)
        return cost_optimization

    def check_budget_alerts(self) -> dict[str, Any]:
        """Check for budget alerts and violations."""
        current_date = time.strftime("%Y-%m-%d")
        daily_spending = self.daily_spending.get(current_date, 0.0)

        alerts = {
            "budget_violation": daily_spending > self.budget_limit_usd_per_day,
            "budget_warning": daily_spending > self.budget_limit_usd_per_day * 0.8,
            "daily_spending": daily_spending,
            "budget_limit": self.budget_limit_usd_per_day,
            "utilization_percentage": (daily_spending / self.budget_limit_usd_per_day)
            * 100,
            "recommended_actions": [],
        }

        if alerts["budget_violation"]:
            alerts["recommended_actions"].extend(
                [
                    "Immediate cost reduction required",
                    "Review and terminate non-essential allocations",
                    "Implement emergency cost controls",
                ]
            )
        elif alerts["budget_warning"]:
            alerts["recommended_actions"].extend(
                [
                    "Monitor spending closely",
                    "Consider deferring non-critical workloads",
                    "Review allocation efficiency",
                ]
            )

        return alerts

    def get_cost_analytics(self) -> dict[str, Any]:
        """Get comprehensive cost analytics and insights."""
        if not self.cost_history:
            return {"error": "No cost data available"}

        # Calculate trends
        total_spending = sum(entry["cost_usd"] for entry in self.cost_history)
        avg_daily_spending = total_spending / max(len(self.daily_spending), 1)

        # Provider breakdown
        provider_costs = {provider.value: 0.0 for provider in CloudProvider}
        for entry in self.cost_history:
            provider_costs[entry["provider"].value] += entry["cost_usd"]

        # Resource type breakdown
        resource_costs = {resource.value: 0.0 for resource in ResourceType}
        for entry in self.cost_history:
            resource_costs[entry["resource_type"].value] += entry["cost_usd"]

        return {
            "total_spending_usd": total_spending,
            "average_daily_spending": avg_daily_spending,
            "budget_utilization": (avg_daily_spending / self.budget_limit_usd_per_day)
            * 100,
            "cost_by_provider": provider_costs,
            "cost_by_resource_type": resource_costs,
            "optimization_opportunities": len(self.optimization_history),
            "potential_monthly_savings": sum(
                opt.potential_savings_percentage for opt in self.optimization_history
            )
            / max(len(self.optimization_history), 1)
            * avg_daily_spending
            * 30
            / 100,
        }
