"""
Hardware-Aware Performance Profiler for JAX.

Provides hardware-specific performance analysis including TPU MXU utilization,
GPU TensorCore usage, memory coalescing analysis, and platform-specific
optimization recommendations.
"""

from collections.abc import Callable
from typing import Any

import jax
import numpy as np

from opifex.benchmarking.profiling.common import (
    detect_hardware_specs,
    measure_execution_time,
)
from opifex.benchmarking.profiling.event_coordinator import EventCoordinator


class TPUProfiler:
    """TPU-specific performance analysis."""

    def __init__(self, coordinator: EventCoordinator | None = None):
        self.coordinator = coordinator
        self.hardware_specs = detect_hardware_specs()

    def analyze_mxu_utilization(
        self, func: Callable, inputs: list[jax.Array]
    ) -> dict[str, Any]:
        """Analyze TPU Matrix Multiply Unit (MXU) utilization."""

        # Estimate MXU operations based on input shapes
        mxu_ops = self._estimate_mxu_operations(inputs)

        # Time execution
        if self.coordinator:
            _, execution_time = self.coordinator.time_function(
                func, *inputs, profiler_id="tpu_profiler", operation_name="mxu_analysis"
            )
        else:
            execution_time = measure_execution_time(func, inputs)
            _ = func(*inputs)

        # Calculate MXU utilization
        peak_mxu_throughput = self.hardware_specs["peak_flops_bf16"]
        actual_throughput = mxu_ops / execution_time if execution_time > 0 else 0
        mxu_utilization = actual_throughput / peak_mxu_throughput

        # Analyze shape alignment for MXU efficiency
        shape_analysis = self._analyze_mxu_shape_alignment(inputs)

        return {
            "mxu_utilization": mxu_utilization,
            "estimated_mxu_ops": mxu_ops,
            "execution_time_ms": execution_time * 1000,
            "actual_throughput_flops": actual_throughput,
            "peak_throughput_flops": peak_mxu_throughput,
            "shape_alignment": shape_analysis,
            "recommendations": self._generate_mxu_recommendations(
                mxu_utilization, shape_analysis
            ),
        }

    def analyze_vmem_usage(
        self, func: Callable, inputs: list[jax.Array]
    ) -> dict[str, Any]:
        """Analyze TPU Vector Memory (VMEM) usage patterns."""

        # Calculate total input size
        total_input_size = sum(
            inp.size * inp.dtype.itemsize
            for inp in inputs
            if hasattr(inp, "size") and hasattr(inp, "dtype")
        )

        # TPU v5e has 16MB VMEM
        vmem_threshold = 16 * 1024 * 1024  # 16MB
        vmem_eligible = (
            total_input_size < vmem_threshold * 0.8
        )  # 80% threshold for safety

        # Estimate speedup potential
        if vmem_eligible:
            # VMEM is 22x faster than HBM on TPU
            speedup_potential = 22.0
            recommendation = (
                "✅ Operation likely uses VMEM - optimal for TPU performance"
            )
        else:
            speedup_potential = 1.0
            recommendation = (
                "⚠️ Operation too large for VMEM. Consider tiling or reducing data size"
            )

        return {
            "vmem_eligible": vmem_eligible,
            "total_input_size_mb": total_input_size / (1024 * 1024),
            "vmem_threshold_mb": vmem_threshold / (1024 * 1024),
            "speedup_potential": speedup_potential,
            "recommendation": recommendation,
            "optimization_suggestions": self._generate_vmem_optimizations(
                total_input_size, vmem_threshold
            ),
        }

    def _estimate_mxu_operations(self, inputs: list[jax.Array]) -> int:
        """Estimate operations that can utilize the MXU."""

        total_ops = 0
        # Check for matrix multiplication patterns (e.g., (..., M, K) @ (..., K, N))
        for i, inp in enumerate(inputs):
            if hasattr(inp, "shape") and len(inp.shape) >= 2 and i < len(inputs) - 1:
                # Estimate matrix operations based on shape
                next_inp = inputs[i + 1]
                if (
                    hasattr(next_inp, "shape")
                    and len(next_inp.shape) >= 2
                    and inp.shape[-1] == next_inp.shape[-2]
                ):
                    m, k = inp.shape[-2], inp.shape[-1]
                    n = next_inp.shape[-1]

                    batch_size = 1
                    for dim in inp.shape[:-2]:
                        batch_size *= dim

                    total_ops += batch_size * 2 * m * k * n

        return total_ops

    def _analyze_mxu_shape_alignment(self, inputs: list[jax.Array]) -> dict[str, Any]:
        """Analyze how well input shapes align with TPU MXU requirements."""

        alignment_scores = []
        shape_recommendations = []

        for inp in inputs:
            if hasattr(inp, "shape") and len(inp.shape) >= 2:
                # TPU MXU prefers multiples of 128
                m, n = inp.shape[-2], inp.shape[-1]

                m_aligned = m % 128 == 0
                n_aligned = n % 128 == 0

                if m_aligned and n_aligned:
                    score = 1.0
                    recommendation = f"✅ Shape {inp.shape} perfectly aligned for MXU"
                elif m % 8 == 0 and n % 8 == 0:
                    score = 0.6
                    recommendation = (
                        f"⚡ Shape {inp.shape} partially aligned. "
                        f"Consider padding to multiples of 128"
                    )
                else:
                    score = 0.2
                    recommendation = (
                        f"⚠️ Shape {inp.shape} poorly aligned. "
                        f"Pad to multiples of 8 or 128"
                    )

                alignment_scores.append(score)
                shape_recommendations.append(recommendation)

        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

        return {
            "average_alignment_score": avg_alignment,
            "individual_scores": alignment_scores,
            "shape_recommendations": shape_recommendations,
            "overall_rating": "excellent"
            if avg_alignment > 0.8
            else "good"
            if avg_alignment > 0.5
            else "poor",
        }

    def _generate_mxu_recommendations(
        self, utilization: float, shape_analysis: dict[str, Any]
    ) -> list[str]:
        """Generate MXU optimization recommendations."""

        recommendations = []

        if utilization < 0.3:
            recommendations.extend(
                [
                    (
                        f"⚠️  Low SIMD utilization ({utilization:.1f}%). "
                        f"Use vectorized operations (vmap) and avoid scalar loops."
                    ),
                    "💡 Increase batch size above 240 for better MXU engagement",
                    "💡 Ensure operations are matrix-multiplication heavy",
                ]
            )
        elif utilization < 0.6:
            recommendations.extend(
                [
                    f"⚡ Moderate MXU utilization ({utilization:.2%})",
                    "💡 Fine-tune batch size and tensor shapes for optimal MXU usage",
                ]
            )
        else:
            recommendations.append(f"✅ Good MXU utilization ({utilization:.2%})")

        # Add shape-specific recommendations
        if shape_analysis["average_alignment_score"] < 0.5:
            recommendations.extend(
                [
                    "🔧 Shape optimization needed:",
                    "  • Pad tensor dimensions to multiples of 128",
                    "  • Use TPU-optimized data layouts",
                    "  • Consider reshaping operations to improve alignment",
                ]
            )

        return recommendations

    def _generate_vmem_optimizations(
        self, total_size: int, threshold: int
    ) -> list[str]:
        """Generate VMEM optimization suggestions."""

        suggestions = []

        if total_size > threshold:
            suggestions.extend(
                [
                    "💡 VMEM optimization strategies:",
                    f"  • Current size: {total_size / (1024 * 1024):.1f}MB, "
                    f"VMEM limit: {threshold / (1024 * 1024):.1f}MB",
                    "  • Use operation tiling to fit in VMEM",
                    "  • Consider gradient checkpointing to reduce memory usage",
                    "  • Split large operations into smaller chunks",
                ]
            )
        else:
            suggestions.extend(
                [
                    "✅ Operation fits in VMEM - excellent for performance",
                    "💡 Ensure XLA compiler utilizes VMEM effectively",
                ]
            )

        return suggestions


class GPUProfiler:
    """GPU-specific performance analysis."""

    def __init__(self, coordinator: EventCoordinator | None = None):
        self.coordinator = coordinator
        self.hardware_specs = detect_hardware_specs()

    def analyze_tensorcore_utilization(
        self, func: Callable, inputs: list[jax.Array]
    ) -> dict[str, Any]:
        """Analyze GPU TensorCore utilization."""

        # Check shape alignment for TensorCore usage
        tensorcore_analysis = self._analyze_tensorcore_alignment(inputs)

        # Estimate TensorCore operations
        tensorcore_ops = self._estimate_tensorcore_operations(inputs)

        # Time execution
        if self.coordinator:
            _, execution_time = self.coordinator.time_function(
                func,
                *inputs,
                profiler_id="gpu_profiler",
                operation_name="tensorcore_analysis",
            )
        else:
            execution_time = measure_execution_time(func, inputs)
            _ = func(*inputs)

        # Calculate utilization
        peak_tensorcore_throughput = self.hardware_specs["peak_flops_bf16"]
        actual_throughput = tensorcore_ops / execution_time if execution_time > 0 else 0
        tensorcore_utilization = actual_throughput / peak_tensorcore_throughput

        return {
            "tensorcore_utilization": tensorcore_utilization,
            "estimated_tensorcore_ops": tensorcore_ops,
            "execution_time_ms": execution_time * 1000,
            "shape_alignment": tensorcore_analysis,
            "recommendations": self._generate_tensorcore_recommendations(
                tensorcore_utilization, tensorcore_analysis
            ),
        }

    def analyze_memory_coalescing(self, inputs: list[jax.Array]) -> dict[str, Any]:
        """Analyze GPU memory access patterns for coalescing efficiency."""

        coalescing_analysis = []

        for inp in inputs:
            if hasattr(inp, "shape") and len(inp.shape) >= 2:
                # Analyze memory layout efficiency
                shape = inp.shape

                # Check if last dimension is aligned for coalesced access
                last_dim = shape[-1]

                if last_dim % 32 == 0:
                    coalescing_score = 1.0
                    efficiency = "excellent"
                elif last_dim % 16 == 0:
                    coalescing_score = 0.8
                    efficiency = "good"
                elif last_dim % 8 == 0:
                    coalescing_score = 0.6
                    efficiency = "moderate"
                else:
                    coalescing_score = 0.3
                    efficiency = "poor"

                coalescing_analysis.append(
                    {
                        "shape": shape,
                        "coalescing_score": coalescing_score,
                        "efficiency": efficiency,
                        "last_dim_alignment": last_dim,
                    }
                )

        avg_coalescing = (
            np.mean([a["coalescing_score"] for a in coalescing_analysis])
            if coalescing_analysis
            else 0.0
        )

        return {
            "average_coalescing_efficiency": avg_coalescing,
            "individual_analysis": coalescing_analysis,
            "memory_throughput_loss": (1 - avg_coalescing) * 100,
            "optimization_suggestions": self._generate_coalescing_optimizations(
                coalescing_analysis
            ),
        }

    def _analyze_tensorcore_alignment(self, inputs: list[jax.Array]) -> dict[str, Any]:
        """Analyze tensor shape alignment for TensorCore usage."""

        alignment_scores = []
        recommendations = []

        # TensorCore preferred shapes for different precisions
        tensorcore_shapes = self.hardware_specs.get("tensor_core_shapes", [(16, 16, 8)])

        for inp in inputs:
            if hasattr(inp, "shape") and len(inp.shape) >= 2:
                m, n = inp.shape[-2], inp.shape[-1]

                # Check alignment with TensorCore requirements
                best_score = 0.0
                best_shape = None

                for tc_m, tc_n, tc_k in tensorcore_shapes:
                    if m % tc_m == 0 and n % tc_n == 0:
                        score = 1.0
                        best_shape = (tc_m, tc_n, tc_k)
                        best_score = score
                        break
                    if m % (tc_m // 2) == 0 and n % (tc_n // 2) == 0:
                        score = 0.7
                        if score > best_score:
                            best_score = score
                            best_shape = (tc_m, tc_n, tc_k)

                if best_score > 0.8:
                    recommendation = (
                        f"✅ Shape {inp.shape} well-aligned for "
                        f"TensorCores ({best_shape})"
                    )
                elif best_score > 0.5:
                    recommendation = (
                        f"⚡ Shape {inp.shape} partially aligned. "
                        f"Consider padding for {best_shape}"
                    )
                else:
                    recommendation = (
                        f"✅ Shape {inp.shape} well-aligned for "
                        f"TensorCores ({best_shape})"
                    )

                alignment_scores.append(best_score)
                recommendations.append(recommendation)

        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

        return {
            "average_alignment_score": avg_alignment,
            "individual_scores": alignment_scores,
            "shape_recommendations": recommendations,
            "tensorcore_shapes": tensorcore_shapes,
        }

    def _estimate_tensorcore_operations(self, inputs: list[jax.Array]) -> int:
        """Estimate operations that can use TensorCores."""

        total_ops = 0

        for i, inp in enumerate(inputs):
            if hasattr(inp, "shape") and len(inp.shape) >= 2 and i < len(inputs) - 1:
                next_inp = inputs[i + 1]
                if hasattr(next_inp, "shape") and len(next_inp.shape) >= 2:
                    # Matrix multiplication that could use TensorCores
                    m, k = inp.shape[-2], inp.shape[-1]
                    n = next_inp.shape[-1]

                    batch_size = 1
                    for dim in inp.shape[:-2]:
                        batch_size *= dim

                    total_ops += batch_size * 2 * m * k * n

        return total_ops

    def _generate_tensorcore_recommendations(
        self, utilization: float, alignment: dict[str, Any]
    ) -> list[str]:
        """Generate TensorCore optimization recommendations."""

        recommendations = []

        if utilization < 0.3:
            recommendations.extend(
                [
                    (
                        f"⚠️  Low TensorCore utilization ({utilization:.1f}%). "
                        f"Ensure shapes are multiples of 8/16 (FP16) or 16 (INT8)."
                    ),
                    "💡 Use mixed precision (bfloat16/float16) to enable TensorCores",
                    "💡 Increase batch size above 298 for H100 optimal performance",
                ]
            )
        elif utilization < 0.7:
            recommendations.extend(
                [
                    f"⚡ Moderate TensorCore utilization ({utilization:.2%})",
                    "💡 Fine-tune tensor shapes for optimal TensorCore alignment",
                ]
            )
        else:
            recommendations.append(
                f"✅ Good TensorCore utilization ({utilization:.2%})"
            )

        if alignment["average_alignment_score"] < 0.5:
            recommendations.extend(
                [
                    "🔧 TensorCore alignment optimization:",
                    "  • Pad tensor dimensions to multiples of 8, 16, or 32",
                    "  • Use NHWC layout for convolutions",
                    "  • Consider tensor reshaping for better alignment",
                ]
            )

        return recommendations

    def _generate_coalescing_optimizations(
        self, analysis: list[dict[str, Any]]
    ) -> list[str]:
        """Generate memory coalescing optimization suggestions."""

        suggestions = []

        poor_coalescing = [a for a in analysis if a["coalescing_score"] < 0.5]

        if poor_coalescing:
            suggestions.extend(
                [
                    "💡 Memory coalescing optimizations:",
                    "  • Ensure last dimension is multiple of 32 for optimal "
                    "coalescing",
                    "  • Consider tensor transposition to improve access patterns",
                    "  • Use contiguous memory layouts where possible",
                ]
            )
        else:
            suggestions.append("✅ Good memory coalescing patterns detected")

        return suggestions


class CPUProfiler:
    """CPU-specific performance analysis."""

    def __init__(self, coordinator: EventCoordinator | None = None):
        self.coordinator = coordinator
        self.hardware_specs = detect_hardware_specs()

    def analyze_simd_utilization(
        self, func: Callable, inputs: list[jax.Array]
    ) -> dict[str, Any]:
        """Analyze CPU SIMD utilization."""

        # Analyze vectorization potential
        simd_analysis = self._analyze_simd_alignment(inputs)

        # Time execution
        if self.coordinator:
            _, execution_time = self.coordinator.time_function(
                func,
                *inputs,
                profiler_id="cpu_profiler",
                operation_name="simd_analysis",
            )
        else:
            execution_time = measure_execution_time(func, inputs)
            _ = func(*inputs)

        return {
            "execution_time_ms": execution_time * 1000,
            "simd_alignment": simd_analysis,
            "recommendations": self._generate_cpu_recommendations(simd_analysis),
        }

    def _analyze_simd_alignment(self, inputs: list[jax.Array]) -> dict[str, Any]:
        """Analyze tensor alignment for SIMD operations."""

        simd_width = self.hardware_specs.get("simd_width", 8)
        alignment_scores = []

        for inp in inputs:
            if hasattr(inp, "shape"):
                # Check if dimensions are SIMD-friendly
                last_dim = inp.shape[-1] if inp.shape else 1

                if last_dim % simd_width == 0:
                    score = 1.0
                elif last_dim % (simd_width // 2) == 0:
                    score = 0.7
                else:
                    score = 0.3

                alignment_scores.append(score)

        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

        return {
            "average_simd_alignment": avg_alignment,
            "simd_width": simd_width,
            "individual_scores": alignment_scores,
        }

    def _generate_cpu_recommendations(self, simd_analysis: dict[str, Any]) -> list[str]:
        """Generate CPU optimization recommendations."""

        recommendations = []

        if simd_analysis["average_simd_alignment"] < 0.5:
            recommendations.extend(
                [
                    "💡 CPU SIMD optimizations:",
                    (
                        f"  • Align tensor dimensions to multiples of "
                        f"{simd_analysis['simd_width']}"
                    ),
                    "  • Use vectorized operations where possible",
                    "  • Consider smaller batch sizes for better cache utilization",
                ]
            )
        else:
            recommendations.append("✅ Good SIMD alignment for CPU execution")

        return recommendations


class HardwareAwareProfiler:
    """Main hardware-aware profiler that delegates to platform-specific profilers."""

    def __init__(self, coordinator: EventCoordinator | None = None):
        self.coordinator = coordinator
        self.backend = jax.default_backend()

        # Initialize platform-specific profiler
        if self.backend == "tpu":
            self.platform_profiler = TPUProfiler(coordinator)
        elif self.backend == "gpu":
            self.platform_profiler = GPUProfiler(coordinator)
        else:
            self.platform_profiler = CPUProfiler(coordinator)

        if coordinator:
            coordinator.register_profiler("hardware_profiler")

    def profile_operation(self, func: Callable, *inputs) -> dict[str, Any]:
        """Profile operation with hardware-specific analysis."""

        input_list = list(inputs)

        # Common analysis
        hardware_info = {
            "backend": self.backend,
            "device_count": jax.device_count(),
            "devices": [str(device) for device in jax.devices()],
        }

        # Platform-specific analysis
        # Platform-specific analysis
        if isinstance(self.platform_profiler, TPUProfiler):
            platform_analysis = {
                "mxu_analysis": self.platform_profiler.analyze_mxu_utilization(
                    func, input_list
                ),
                "vmem_analysis": self.platform_profiler.analyze_vmem_usage(
                    func, input_list
                ),
            }
        elif isinstance(self.platform_profiler, GPUProfiler):
            tensorcore_analysis = self.platform_profiler.analyze_tensorcore_utilization(
                func, input_list
            )
            platform_analysis = {
                "tensorcore_analysis": tensorcore_analysis,
                "memory_coalescing": self.platform_profiler.analyze_memory_coalescing(
                    input_list
                ),
            }
        elif isinstance(self.platform_profiler, CPUProfiler):
            platform_analysis = {
                "simd_analysis": self.platform_profiler.analyze_simd_utilization(
                    func, input_list
                )
            }
        else:
            # Fallback or empty analysis if profiler type is unknown
            platform_analysis = {}

        return {
            "hardware_info": hardware_info,
            "platform_analysis": platform_analysis,
            "backend": self.backend,
        }
