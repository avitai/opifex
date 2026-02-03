"""GPU pool management with multi-model inference optimization.

This module provides intelligent GPU pool management, memory allocation,
and optimization for multi-model inference across cloud providers.
"""

import time
from typing import Any

from .types import CloudProvider, ResourceType


class GPUPoolManager:
    """Intelligent GPU pool management with multi-model inference optimization."""

    def __init__(
        self,
        resource_orchestrator: Any,
        memory_optimization_threshold: float = 0.85,
        pool_rebalancing_interval: int = 300,  # 5 minutes
    ):
        """Initialize GPU pool manager.

        Args:
            resource_orchestrator: ResourceOrchestrator instance for coordination
            memory_optimization_threshold: Utilization threshold to trigger
                optimization (0-1)
            pool_rebalancing_interval: Interval between rebalancing attempts
                in seconds
        """
        self.resource_orchestrator = resource_orchestrator
        self.memory_optimization_threshold = memory_optimization_threshold
        self.pool_rebalancing_interval = pool_rebalancing_interval

        self.gpu_pools: dict[str, dict[str, Any]] = {}
        self.memory_allocations: dict[str, dict[str, Any]] = {}
        self.model_placement_cache: dict[str, str] = {}  # model_hash -> pool_id

    def create_gpu_pool(
        self,
        pool_id: str,
        gpu_type: ResourceType,
        gpu_count: int,
        memory_per_gpu_gb: float,
        provider: CloudProvider,
        region: str,
    ) -> bool:
        """Create a new GPU pool.

        Args:
            pool_id: Unique identifier for the pool
            gpu_type: Type of GPU resources
            gpu_count: Number of GPUs in the pool
            memory_per_gpu_gb: Memory per GPU in GB
            provider: Cloud provider hosting the pool
            region: Geographic region of the pool

        Returns:
            True if pool was created successfully
        """
        gpu_pool = {
            "pool_id": pool_id,
            "gpu_type": gpu_type,
            "gpu_count": gpu_count,
            "memory_per_gpu_gb": memory_per_gpu_gb,
            "provider": provider,
            "region": region,
            "total_memory_gb": gpu_count * memory_per_gpu_gb,
            "allocated_memory_gb": 0.0,
            "active_models": {},
            "utilization": 0.0,
            "creation_time": time.time(),
        }

        self.gpu_pools[pool_id] = gpu_pool
        return True

    def allocate_gpu_memory(
        self,
        model_hash: str,
        memory_requirement_gb: float,
        preferred_pool_id: str | None = None,
    ) -> dict[str, Any]:
        """Allocate GPU memory for model deployment.

        Args:
            model_hash: Unique identifier for the model
            memory_requirement_gb: Required memory in GB
            preferred_pool_id: Optional preferred pool for allocation

        Returns:
            Dictionary with allocation result and details
        """
        # Try preferred pool first
        if preferred_pool_id and preferred_pool_id in self.gpu_pools:
            pool = self.gpu_pools[preferred_pool_id]
            if self._can_allocate_memory(pool, memory_requirement_gb):
                return self._execute_memory_allocation(
                    pool, model_hash, memory_requirement_gb
                )

        # Find best available pool
        best_pool = self._find_best_gpu_pool(memory_requirement_gb)
        if best_pool:
            return self._execute_memory_allocation(
                best_pool, model_hash, memory_requirement_gb
            )

        # No suitable pool found
        return {
            "success": False,
            "error": "No available GPU pool with sufficient memory",
            "required_memory_gb": memory_requirement_gb,
            "available_pools": list(self.gpu_pools.keys()),
        }

    def _can_allocate_memory(self, pool: dict[str, Any], memory_gb: float) -> bool:
        """Check if pool can allocate the required memory.

        Args:
            pool: Pool dictionary
            memory_gb: Required memory in GB

        Returns:
            True if pool has sufficient available memory
        """
        available_memory = pool["total_memory_gb"] - pool["allocated_memory_gb"]
        return available_memory >= memory_gb

    def _find_best_gpu_pool(
        self, memory_requirement_gb: float
    ) -> dict[str, Any] | None:
        """Find the best GPU pool for memory allocation.

        Args:
            memory_requirement_gb: Required memory in GB

        Returns:
            Best pool dictionary or None if no suitable pool found
        """
        eligible_pools = []

        for pool in self.gpu_pools.values():
            if self._can_allocate_memory(pool, memory_requirement_gb):
                eligible_pools.append(pool)

        if not eligible_pools:
            return None

        # Score pools based on utilization and efficiency
        best_pool = None
        best_score = -1.0

        for pool in eligible_pools:
            # Prefer pools with moderate utilization (avoid both idle and overloaded)
            utilization_score = 1.0 - abs(pool["utilization"] - 0.7)

            # Prefer pools with matching GPU types for the workload
            gpu_type_score = self._calculate_gpu_type_score(pool["gpu_type"])

            # Prefer pools with sufficient headroom
            memory_headroom = (
                pool["total_memory_gb"]
                - pool["allocated_memory_gb"]
                - memory_requirement_gb
            )
            headroom_score = min(1.0, memory_headroom / memory_requirement_gb)

            combined_score = (
                0.4 * utilization_score + 0.3 * gpu_type_score + 0.3 * headroom_score
            )

            if combined_score > best_score:
                best_score = combined_score
                best_pool = pool

        return best_pool

    def _calculate_gpu_type_score(self, gpu_type: ResourceType) -> float:
        """Calculate GPU type score for workload suitability.

        Args:
            gpu_type: Type of GPU resource

        Returns:
            Score between 0.0 and 1.0
        """
        # Simple scoring based on GPU capabilities
        gpu_scores = {
            ResourceType.GPU_H100: 1.0,  # Best for large models
            ResourceType.GPU_A100: 0.9,  # Excellent for most workloads
            ResourceType.GPU_V100: 0.7,  # Good for medium models
        }
        return gpu_scores.get(gpu_type, 0.5)

    def _execute_memory_allocation(
        self, pool: dict[str, Any], model_hash: str, memory_gb: float
    ) -> dict[str, Any]:
        """Execute memory allocation in the selected pool.

        Args:
            pool: Pool dictionary
            model_hash: Model identifier
            memory_gb: Memory to allocate in GB

        Returns:
            Allocation result dictionary
        """
        pool_id = pool["pool_id"]

        # Update pool allocation
        pool["allocated_memory_gb"] += memory_gb
        pool["active_models"][model_hash] = {
            "memory_gb": memory_gb,
            "allocation_time": time.time(),
        }
        pool["utilization"] = pool["allocated_memory_gb"] / pool["total_memory_gb"]

        # Cache model placement
        self.model_placement_cache[model_hash] = pool_id

        # Record allocation
        allocation_id = f"gpu_alloc_{int(time.time() * 1000)}"
        self.memory_allocations[allocation_id] = {
            "allocation_id": allocation_id,
            "model_hash": model_hash,
            "pool_id": pool_id,
            "memory_gb": memory_gb,
            "allocation_time": time.time(),
            "status": "active",
        }

        return {
            "success": True,
            "allocation_id": allocation_id,
            "pool_id": pool_id,
            "allocated_memory_gb": memory_gb,
            "pool_utilization": pool["utilization"],
            "gpu_type": pool["gpu_type"],
        }

    def deallocate_gpu_memory(self, model_hash: str) -> bool:
        """Deallocate GPU memory for a model.

        Args:
            model_hash: Model identifier

        Returns:
            True if deallocation was successful
        """
        if model_hash not in self.model_placement_cache:
            return False

        pool_id = self.model_placement_cache[model_hash]
        if pool_id not in self.gpu_pools:
            return False

        pool = self.gpu_pools[pool_id]
        if model_hash not in pool["active_models"]:
            return False

        # Deallocate memory
        model_info = pool["active_models"][model_hash]
        memory_gb = model_info["memory_gb"]

        pool["allocated_memory_gb"] -= memory_gb
        del pool["active_models"][model_hash]
        pool["utilization"] = pool["allocated_memory_gb"] / pool["total_memory_gb"]

        # Remove from cache
        del self.model_placement_cache[model_hash]

        # Update allocation record
        for allocation in self.memory_allocations.values():
            if (
                allocation["model_hash"] == model_hash
                and allocation["pool_id"] == pool_id
            ):
                allocation["status"] = "deallocated"
                allocation["deallocation_time"] = time.time()
                break

        return True

    async def optimize_memory_layout(self) -> dict[str, Any]:
        """Optimize memory layout across GPU pools.

        Returns:
            Dictionary with optimization results
        """
        optimization_results = {
            "pools_optimized": 0,
            "memory_saved_gb": 0.0,
            "models_relocated": 0,
            "optimizations_applied": [],
        }

        for _pool_id, pool in self.gpu_pools.items():
            if pool["utilization"] > self.memory_optimization_threshold:
                # Try to optimize this pool
                pool_optimization = await self._optimize_pool_memory(pool)

                if pool_optimization["memory_saved_gb"] > 0:
                    optimization_results["pools_optimized"] += 1
                    optimization_results["memory_saved_gb"] += pool_optimization[
                        "memory_saved_gb"
                    ]
                    optimization_results["models_relocated"] += pool_optimization[
                        "models_relocated"
                    ]
                    optimization_results["optimizations_applied"].extend(
                        pool_optimization["optimizations_applied"]
                    )

        return optimization_results

    async def _optimize_pool_memory(self, pool: dict[str, Any]) -> dict[str, Any]:
        """Optimize memory usage for a specific pool.

        Args:
            pool: Pool dictionary

        Returns:
            Optimization result dictionary
        """
        optimization_result = {
            "memory_saved_gb": 0.0,
            "models_relocated": 0,
            "optimizations_applied": [],
        }

        # Strategy 1: Relocate smaller models to less utilized pools
        small_models = [
            (model_hash, info)
            for model_hash, info in pool["active_models"].items()
            if info["memory_gb"] < 5.0  # Models under 5GB
        ]

        for model_hash, model_info in small_models:
            # Find a better pool for this model
            better_pool = self._find_better_pool_for_model(
                model_hash, model_info["memory_gb"], pool["pool_id"]
            )

            # Relocate model if better pool found and memory can be deallocated
            if better_pool and self.deallocate_gpu_memory(model_hash):
                new_allocation = self.allocate_gpu_memory(
                    model_hash, model_info["memory_gb"], better_pool["pool_id"]
                )

                if new_allocation["success"]:
                    optimization_result["models_relocated"] += 1
                    optimization_result["optimizations_applied"].append(
                        f"Relocated model {model_hash[:8]} from "
                        f"{pool['pool_id']} to {better_pool['pool_id']}"
                    )

        return optimization_result

    def _find_better_pool_for_model(
        self, model_hash: str, memory_gb: float, current_pool_id: str
    ) -> dict[str, Any] | None:
        """Find a better pool for model relocation.

        Args:
            model_hash: Model identifier
            memory_gb: Memory requirement in GB
            current_pool_id: Current pool ID

        Returns:
            Better pool dictionary or None
        """
        for pool_id, pool in self.gpu_pools.items():
            if pool_id == current_pool_id:
                continue

            # Check if pool has capacity and lower utilization
            if (
                self._can_allocate_memory(pool, memory_gb) and pool["utilization"] < 0.6
            ):  # Target pools under 60% utilization
                return pool

        return None

    def get_pool_statistics(self) -> dict[str, Any]:
        """Get comprehensive GPU pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        total_gpus = sum(pool["gpu_count"] for pool in self.gpu_pools.values())
        total_memory_gb = sum(
            pool["total_memory_gb"] for pool in self.gpu_pools.values()
        )
        allocated_memory_gb = sum(
            pool["allocated_memory_gb"] for pool in self.gpu_pools.values()
        )

        pool_utilizations = [pool["utilization"] for pool in self.gpu_pools.values()]
        avg_utilization = (
            sum(pool_utilizations) / len(pool_utilizations)
            if pool_utilizations
            else 0.0
        )

        return {
            "total_pools": len(self.gpu_pools),
            "total_gpus": total_gpus,
            "total_memory_gb": total_memory_gb,
            "allocated_memory_gb": allocated_memory_gb,
            "memory_utilization": allocated_memory_gb / total_memory_gb
            if total_memory_gb > 0
            else 0.0,
            "average_pool_utilization": avg_utilization,
            "active_models": len(self.model_placement_cache),
            "active_allocations": len(
                [a for a in self.memory_allocations.values() if a["status"] == "active"]
            ),
            "pools_by_provider": {
                provider.value: len(
                    [p for p in self.gpu_pools.values() if p["provider"] == provider]
                )
                for provider in CloudProvider
            },
        }
