"""
Roofline Analysis for Opifex JAX Profiling.

Provides roofline analysis to identify if operations are compute-bound or memory-bound,
and suggests optimizations based on hardware characteristics.
"""

from collections.abc import Callable
from typing import Any

import jax

from .common import detect_hardware_specs, measure_execution_time
from .event_coordinator import EventCoordinator


class RooflineAnalyzer:
    """Analyzes operation performance against hardware roofline limits."""

    def __init__(self, coordinator: EventCoordinator):
        self.coordinator = coordinator
        self.hardware_specs = detect_hardware_specs()

    def analyze_operation(
        self,
        func: Callable,
        inputs: list[jax.Array],
        name: str = "operation",
    ) -> dict[str, Any]:
        """
        Perform roofline analysis on a specific operation.

        Args:
            func: Function to analyze
            inputs: Input arguments for the function
            name: Name of the operation

        Returns:
            Dictionary containing roofline metrics and recommendations
        """
        # 1. Measure execution time
        execution_time = measure_execution_time(func, inputs)

        # 2. Estimate FLOPs
        theoretical_flops = self._estimate_operation_flops(func, inputs)

        # 3. Estimate Memory Traffic
        memory_traffic = sum(x.nbytes for x in inputs)
        # Estimate output size (run once to get output)
        output = func(*inputs)
        if isinstance(output, tuple | list):
            for out in output:
                if hasattr(out, "nbytes"):
                    memory_traffic += out.nbytes
        elif hasattr(output, "nbytes"):
            memory_traffic += output.nbytes

        # 4. Calculate Metrics
        achieved_flops = theoretical_flops / execution_time if execution_time > 0 else 0
        memory_bandwidth = memory_traffic / execution_time if execution_time > 0 else 0
        arithmetic_intensity = (
            theoretical_flops / memory_traffic if memory_traffic > 0 else 0
        )

        # 5. Analyze against Roofline
        peak_flops = self.hardware_specs["peak_flops"]
        peak_bandwidth = self.hardware_specs["memory_bandwidth"]
        critical_intensity = self.hardware_specs["critical_intensity"]

        flops_utilization = achieved_flops / peak_flops
        memory_bandwidth_utilization = memory_bandwidth / peak_bandwidth

        # Determine bottleneck
        if arithmetic_intensity < critical_intensity:
            bottleneck = "Memory Bandwidth"
            efficiency = memory_bandwidth_utilization
        else:
            bottleneck = "Compute (FLOPs)"
            efficiency = flops_utilization

        # Generate recommendations
        recommendations = self._generate_recommendations(
            arithmetic_intensity,
            efficiency,
            bottleneck,
            inputs,
            achieved_flops,
        )

        return {
            "arithmetic_intensity": arithmetic_intensity,
            "critical_intensity": critical_intensity,
            "memory_bandwidth_utilization": memory_bandwidth_utilization,
            "flops_utilization": flops_utilization,
            "bottleneck": bottleneck,
            "efficiency": efficiency,
            "actual_time_ms": execution_time * 1000,
            "optimization_recommendations": recommendations,
        }

    def _estimate_operation_flops(self, func: Callable, inputs: list[jax.Array]) -> int:
        """
        Estimate FLOPs for the operation.

        This is a simplified estimation. For accurate FLOPs,
        one would need to inspect the HLO or use platform-specific counters.
        """
        # Very rough heuristic based on input size and operation type
        # In a real implementation, this would use:
        # jax.jit(func).lower(...).cost_analysis()
        total_elements = sum(x.size for x in inputs)
        # Assume some complexity factor (e.g., 2 for simple arithmetic,
        # higher for matmul)
        # This is a placeholder
        return total_elements * 10

    def _generate_recommendations(
        self,
        arithmetic_intensity: float,
        efficiency: float,
        bottleneck: str,
        inputs: list[jax.Array],
        achieved_flops: float,
    ) -> list[str]:
        """Generate optimization recommendations based on roofline analysis."""
        recommendations = []

        if bottleneck == "Memory Bandwidth":
            recommendations.extend(
                [
                    (
                        f"⚠️  Memory Bound (Intensity: {arithmetic_intensity:.2f} < "
                        f"{self.hardware_specs['critical_intensity']:.2f}). "
                        f"Optimize memory access patterns."
                    ),
                    "💡 Increase batch size to improve arithmetic intensity",
                    "💡 Use operation fusion to reduce memory traffic",
                ]
            )
        else:
            recommendations.extend(
                [
                    (
                        f"⚠️  Compute Bound (Intensity: {arithmetic_intensity:.2f} > "
                        f"{self.hardware_specs['critical_intensity']:.1f}). "
                        "Optimize FLOPs."
                    ),
                    "💡 Check for inefficient math operations",
                    "💡 Ensure you are using high-precision matrix units "
                    "(MXU/TensorCore)",
                ]
            )

        if efficiency < 0.2:
            attained_gflops = achieved_flops / 1e9
            recommendations.extend(
                [
                    (
                        f"⚠️  Low Performance ({attained_gflops:.2f} GFLOPS). "
                        "Check for bottlenecks other than compute/memory."
                    ),
                    "  • Kernel launch overhead (too many small ops)",
                    "  • Poor data alignment",
                ]
            )
        elif efficiency < 0.5:
            recommendations.extend(
                [
                    (
                        f"⚡ Moderate efficiency ({efficiency:.2%}). "
                        "Potential improvements:"
                    ),
                    "💡 Optimize tensor layouts for memory access patterns",
                    "💡 Consider hardware-specific optimizations",
                ]
            )

        # Check alignment
        if inputs:
            alignment_score = self._calculate_alignment_score(inputs[0].shape)
            if alignment_score < 1.0:
                recommendations.append(
                    f"⚠️ Poor tensor alignment (score: {alignment_score:.2f}). "
                    "Pad dimensions to multiples of 128/256."
                )

        return recommendations

    def _calculate_alignment_score(self, shape: tuple[int, ...]) -> float:
        """Calculate how well tensor shape aligns with hardware requirements."""
        if not shape:
            return 1.0

        score = 0.0
        # Check last dimension (most important for vectorization)
        last_dim = shape[-1]

        if last_dim % 128 == 0:
            score += 1.0
        elif last_dim % 32 == 0:
            score += 0.8
        elif last_dim % 8 == 0:
            score += 0.5
        else:
            score += 0.2

        return score
