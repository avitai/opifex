"""Multi-cloud resource orchestrator with intelligent allocation.

This module provides intelligent resource orchestration across multiple cloud
providers using neural network-based optimization for allocation decisions.
"""

import time
from typing import Any

import jax.numpy as jnp
from flax import nnx

from .types import (
    CloudProvider,
    OptimizationObjective,
    ResourceAllocation,
    ResourcePool,
    ResourceType,
)


class ResourceOrchestrator(nnx.Module):
    """Multi-cloud resource orchestrator with intelligent allocation."""

    def __init__(
        self,
        optimization_objective: OptimizationObjective = (
            OptimizationObjective.BALANCE_COST_PERFORMANCE
        ),
        learning_rate: float = 0.001,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize ResourceOrchestrator with neural network components.

        Args:
            optimization_objective: The optimization objective for
                resource allocation
            learning_rate: Learning rate for neural network optimization
                (currently unused)
            rngs: Random number generators for neural network initialization
        """
        super().__init__()
        self.optimization_objective = optimization_objective

        # Neural network for resource allocation optimization
        self.allocation_optimizer = nnx.Sequential(
            nnx.Linear(
                32, 128, rngs=rngs
            ),  # Input: resource requirements + constraints
            nnx.gelu,
            nnx.Linear(128, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(
                64, 16, rngs=rngs
            ),  # Output: allocation scores for different pools
        )

        # Cost prediction network
        self.cost_predictor = nnx.Sequential(
            nnx.Linear(24, 64, rngs=rngs),  # Input: allocation configuration
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.gelu,
            nnx.Linear(32, 1, rngs=rngs),  # Output: predicted cost
        )

        # Performance prediction network
        self.performance_predictor = nnx.Sequential(
            nnx.Linear(24, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.gelu,
            nnx.Linear(32, 1, rngs=rngs),  # Output: predicted performance
        )

        self.resource_pools: dict[str, ResourcePool] = {}
        self.active_allocations: dict[str, ResourceAllocation] = {}

    def register_resource_pool(self, pool: ResourcePool) -> bool:
        """Register a new resource pool.

        Args:
            pool: ResourcePool to register

        Returns:
            True if registration was successful
        """
        self.resource_pools[pool.pool_id] = pool
        return True

    def update_pool_status(
        self, pool_id: str, utilization: float, available_capacity: int
    ) -> bool:
        """Update resource pool status.

        Args:
            pool_id: ID of the pool to update
            utilization: Current utilization (0-1)
            available_capacity: Available capacity in the pool

        Returns:
            True if update was successful, False if pool not found
        """
        if pool_id in self.resource_pools:
            pool = self.resource_pools[pool_id]
            pool.current_utilization = utilization
            pool.available_capacity = available_capacity
            return True
        return False

    def optimize_resource_allocation(
        self,
        resource_requirements: dict[ResourceType, int],
        constraints: dict[str, Any] | None = None,
    ) -> ResourceAllocation:
        """Optimize resource allocation across multi-cloud infrastructure.

        Args:
            resource_requirements: Dictionary mapping resource types to quantities
            constraints: Optional constraints for allocation (cost, performance, etc.)

        Returns:
            ResourceAllocation with selected pools and estimates

        Raises:
            ValueError: If no eligible resource pools found
        """
        constraints = constraints or {}

        # Convert requirements to feature vector
        requirement_features = self._encode_requirements(
            resource_requirements, constraints
        )

        # Get allocation scores for available pools
        allocation_scores = self.allocation_optimizer(requirement_features)

        # Filter and rank eligible pools
        eligible_pools = self._filter_eligible_pools(resource_requirements, constraints)

        if not eligible_pools:
            raise ValueError("No eligible resource pools found for requirements")

        # Select optimal allocation strategy
        selected_pools = self._select_optimal_pools(
            eligible_pools, allocation_scores, resource_requirements
        )

        # Calculate cost and performance estimates
        cost_estimate = self._calculate_cost_estimate(
            selected_pools, resource_requirements
        )
        performance_estimate = self._calculate_performance_estimate(selected_pools)
        carbon_footprint = self._calculate_carbon_footprint(
            selected_pools, resource_requirements
        )

        # Create allocation
        allocation = ResourceAllocation(
            allocation_id=f"alloc_{int(time.time() * 1000)}",
            requested_resources=resource_requirements,
            allocated_resources=selected_pools,
            start_time=time.time(),
            end_time=None,
            cost_estimate_usd=cost_estimate,
            performance_estimate=performance_estimate,
            carbon_footprint_kg=carbon_footprint,
            allocation_strategy=self._get_allocation_strategy_name(),
        )

        self.active_allocations[allocation.allocation_id] = allocation
        return allocation

    def _encode_requirements(
        self, requirements: dict[ResourceType, int], constraints: dict[str, Any]
    ) -> jnp.ndarray:
        """Encode resource requirements and constraints into feature vector.

        Args:
            requirements: Resource requirements dictionary
            constraints: Constraints dictionary

        Returns:
            JAX array of size 32 with encoded features
        """
        # Create feature vector for neural network input
        features = []

        # Resource requirements (one-hot + quantities)
        for resource_type in ResourceType:
            features.append(requirements.get(resource_type, 0))

        # Constraints
        max_cost = constraints.get("max_cost_usd_per_hour", 1000.0)
        min_performance = constraints.get("min_performance_score", 0.5)
        max_latency = constraints.get("max_latency_ms", 100.0)
        preferred_providers = constraints.get("preferred_providers", [])

        features.extend(
            [
                max_cost / 1000.0,  # Normalize
                min_performance,
                max_latency / 100.0,  # Normalize
                len(preferred_providers)
                / len(CloudProvider),  # Provider preference ratio
            ]
        )

        # Optimization objective encoding
        objective_encoding = [0.0] * len(OptimizationObjective)
        objective_idx = list(OptimizationObjective).index(self.optimization_objective)
        objective_encoding[objective_idx] = 1.0
        features.extend(objective_encoding)

        # Pad or truncate to expected input size
        while len(features) < 32:
            features.append(0.0)
        features = features[:32]

        return jnp.array(features)

    def _filter_eligible_pools(
        self, requirements: dict[ResourceType, int], constraints: dict[str, Any]
    ) -> list[ResourcePool]:
        """Filter resource pools that meet requirements and constraints.

        Args:
            requirements: Resource requirements
            constraints: Allocation constraints

        Returns:
            List of eligible ResourcePool objects
        """
        eligible_pools = []

        for pool in self.resource_pools.values():
            # Check capacity
            required_capacity = requirements.get(pool.resource_type, 0)
            if pool.available_capacity < required_capacity:
                continue

            # Check cost constraints
            max_cost = constraints.get("max_cost_usd_per_hour", float("inf"))
            if pool.cost_per_hour_usd > max_cost:
                continue

            # Check performance constraints
            min_performance = constraints.get("min_performance_score", 0.0)
            if pool.performance_score < min_performance:
                continue

            # Check provider preferences
            preferred_providers = constraints.get("preferred_providers", [])
            if preferred_providers and pool.provider not in preferred_providers:
                continue

            eligible_pools.append(pool)

        return eligible_pools

    def _select_optimal_pools(
        self,
        eligible_pools: list[ResourcePool],
        allocation_scores: jnp.ndarray,
        requirements: dict[ResourceType, int],
    ) -> dict[str, ResourcePool]:
        """Select optimal pools based on allocation scores and optimization objective.

        Analyzes eligible pools and selects the most optimal ones based on
        the current optimization objective and resource requirements.

        Args:
            eligible_pools: List of eligible pools
            allocation_scores: Neural network allocation scores
            requirements: Resource requirements

        Returns:
            Dictionary mapping pool IDs to selected ResourcePool objects
        """
        selected_pools = {}

        # Group pools by resource type
        pools_by_type: dict[ResourceType, list[ResourcePool]] = {}
        for pool in eligible_pools:
            if pool.resource_type not in pools_by_type:
                pools_by_type[pool.resource_type] = []
            pools_by_type[pool.resource_type].append(pool)

        # Select best pool for each required resource type
        for resource_type, quantity in requirements.items():
            if quantity <= 0:
                continue

            available_pools = pools_by_type.get(resource_type, [])
            if not available_pools:
                continue

            # Score pools based on optimization objective
            best_pool = self._score_and_select_pool(available_pools)
            if best_pool:
                selected_pools[best_pool.pool_id] = best_pool

        return selected_pools

    def _score_and_select_pool(self, pools: list[ResourcePool]) -> ResourcePool | None:
        """Score and select best pool based on optimization objective.

        Args:
            pools: List of candidate pools

        Returns:
            Best pool or None if no pools available
        """
        if not pools:
            return None

        # Define selection strategies for each objective
        selection_strategies = {
            OptimizationObjective.MINIMIZE_COST: lambda p: min(
                p, key=lambda x: x.cost_per_hour_usd
            ),
            OptimizationObjective.MINIMIZE_LATENCY: lambda p: max(
                p, key=lambda x: x.performance_score
            ),
            OptimizationObjective.MAXIMIZE_THROUGHPUT: lambda p: max(
                p, key=lambda x: x.performance_score * x.available_capacity
            ),
            OptimizationObjective.MINIMIZE_CARBON: lambda p: min(
                p, key=lambda x: x.carbon_efficiency
            ),
            OptimizationObjective.MAXIMIZE_AVAILABILITY: lambda p: max(
                p, key=lambda x: x.availability_sla
            ),
        }

        # Use strategy if available, otherwise use BALANCE_COST_PERFORMANCE logic
        if self.optimization_objective in selection_strategies:
            return selection_strategies[self.optimization_objective](pools)

        # BALANCE_COST_PERFORMANCE - weighted score combining cost and performance
        scores = []
        for pool in pools:
            # Normalize and combine (lower cost is better, higher performance is better)
            cost_score = 1.0 / (pool.cost_per_hour_usd + 0.01)
            performance_score = pool.performance_score
            combined_score = 0.6 * performance_score + 0.4 * cost_score
            scores.append(combined_score)

        best_idx = scores.index(max(scores))
        return pools[best_idx]

    def _calculate_cost_estimate(
        self,
        selected_pools: dict[str, ResourcePool],
        requirements: dict[ResourceType, int],
    ) -> float:
        """Calculate cost estimate for allocation.

        Args:
            selected_pools: Selected resource pools
            requirements: Resource requirements

        Returns:
            Estimated cost in USD per hour
        """
        total_cost = 0.0

        for _pool_id, pool in selected_pools.items():
            required_quantity = requirements.get(pool.resource_type, 0)
            pool_cost = pool.cost_per_hour_usd * required_quantity
            total_cost += pool_cost

        return total_cost

    def _calculate_performance_estimate(
        self, selected_pools: dict[str, ResourcePool]
    ) -> float:
        """Calculate performance estimate for allocation.

        Args:
            selected_pools: Selected resource pools

        Returns:
            Weighted average performance score (0-1)
        """
        if not selected_pools:
            return 0.0

        # Weighted average of performance scores
        total_weight = 0.0
        weighted_performance = 0.0

        for pool in selected_pools.values():
            weight = pool.available_capacity  # Use capacity as weight
            weighted_performance += pool.performance_score * weight
            total_weight += weight

        return weighted_performance / total_weight if total_weight > 0 else 0.0

    def _calculate_carbon_footprint(
        self,
        selected_pools: dict[str, ResourcePool],
        requirements: dict[ResourceType, int],
    ) -> float:
        """Calculate carbon footprint for allocation.

        Args:
            selected_pools: Selected resource pools
            requirements: Resource requirements

        Returns:
            Carbon footprint in kg CO2
        """
        total_carbon = 0.0

        for _pool_id, pool in selected_pools.items():
            required_quantity = requirements.get(pool.resource_type, 0)
            pool_carbon = pool.carbon_efficiency * required_quantity
            total_carbon += pool_carbon

        return total_carbon / 1000.0  # Convert to kg

    def _get_allocation_strategy_name(self) -> str:
        """Get human-readable allocation strategy name.

        Returns:
            Strategy name string
        """
        strategy_names = {
            OptimizationObjective.MINIMIZE_COST: "Cost-Optimized",
            OptimizationObjective.MINIMIZE_LATENCY: "Latency-Optimized",
            OptimizationObjective.MAXIMIZE_THROUGHPUT: "Throughput-Optimized",
            OptimizationObjective.BALANCE_COST_PERFORMANCE: "Balanced",
            OptimizationObjective.MINIMIZE_CARBON: "Carbon-Optimized",
            OptimizationObjective.MAXIMIZE_AVAILABILITY: "Availability-Optimized",
        }
        return strategy_names.get(self.optimization_objective, "Custom")
