"""
Memory Usage Analysis for JAX Neural Operators.

Provides detailed memory profiling capabilities for neural operators,
including peak memory usage, memory efficiency analysis, and optimization suggestions.
"""

import gc
import tracemalloc
from typing import Any

import jax
import jax.numpy as jnp
import psutil
from flax import nnx


class MemoryProfiler:
    """JAX memory profiler for neural operators."""

    def __init__(self):
        self.baseline_memory: float | None = None
        self.peak_memory: float = 0
        self.memory_timeline: list[tuple[int, float] | tuple[int, float, str]] = []

    def start_profiling(self):
        """Start memory profiling."""
        tracemalloc.start()
        self.baseline_memory = self._get_current_memory()
        self.peak_memory = self.baseline_memory
        self.memory_timeline = [(0, self.baseline_memory)]

    def stop_profiling(self) -> dict[str, Any]:
        """Stop profiling and return results."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        baseline = self.baseline_memory or 0.0
        return {
            "baseline_memory_mb": baseline,
            "peak_memory_mb": self.peak_memory,
            "memory_increase_mb": self.peak_memory - baseline,
            "timeline": self.memory_timeline,
        }

    def _get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)

    def checkpoint(self, label: str):
        """Add a memory checkpoint with label."""
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)
        timestamp = len(self.memory_timeline)
        self.memory_timeline.append((timestamp, current_memory, label))


def memory_usage_analysis(
    model: nnx.Module, input_data: jnp.ndarray, include_gradients: bool = True
) -> dict[str, Any]:
    """
    Comprehensive memory usage analysis for a neural operator.

    Args:
        model: Flax NNX model to analyze
        input_data: Sample input data
        include_gradients: Whether to include gradient computation in analysis

    Returns:
        Detailed memory usage analysis
    """
    profiler = MemoryProfiler()
    profiler.start_profiling()

    # Force garbage collection before starting
    gc.collect()

    # Analyze model parameters memory
    param_memory = _analyze_parameter_memory(model)
    profiler.checkpoint("parameters_loaded")

    # Analyze input data memory
    input_memory = _analyze_tensor_memory(input_data)
    profiler.checkpoint("input_loaded")

    # Forward pass memory analysis
    try:
        output = model(input_data)  # pyright: ignore[reportCallIssue]
        output_memory = _analyze_tensor_memory(output)
        profiler.checkpoint("forward_pass")
    except Exception as e:
        return {
            "error": f"Forward pass failed: {e!s}",
            "parameter_memory": param_memory,
            "input_memory": input_memory,
        }

    # Gradient computation memory (if requested)
    gradient_memory = None
    if include_gradients:
        try:

            def loss_fn(model_params: nnx.Module, x: jax.Array):
                return jnp.mean(model_params(x) ** 2)  # pyright: ignore[reportCallIssue]

            grad_fn = jax.grad(loss_fn)
            gradients = grad_fn(model, input_data)
            gradient_memory = _analyze_gradient_memory(gradients)
            profiler.checkpoint("gradients_computed")
        except Exception as e:
            gradient_memory = {"error": f"Gradient computation failed: {e!s}"}

    # Stop profiling and get timeline
    profiling_results = profiler.stop_profiling()

    # Memory efficiency analysis
    efficiency_analysis = _analyze_memory_efficiency(
        param_memory, input_memory, output_memory, gradient_memory
    )

    return {
        "parameter_memory": param_memory,
        "input_memory": input_memory,
        "output_memory": output_memory,
        "gradient_memory": gradient_memory,
        "profiling_timeline": profiling_results,
        "efficiency_analysis": efficiency_analysis,
        "optimization_suggestions": _generate_optimization_suggestions(
            efficiency_analysis
        ),
    }


def _analyze_parameter_memory(model: nnx.Module) -> dict[str, Any]:
    """Analyze memory usage of model parameters."""
    params = nnx.state(model, nnx.Param)
    # Convert NNX State to pytree for iteration
    params_tree = nnx.to_tree(params)

    total_bytes = 0
    param_breakdown: dict[str, dict[str, Any]] = {}

    # Use JAX tree utilities to iterate over parameters with paths
    flat_with_path = jax.tree_util.tree_leaves_with_path(params_tree)

    for path, value in flat_with_path:
        if hasattr(value, "nbytes"):
            # Build a readable key from the path
            key = "/".join(str(k.key if hasattr(k, "key") else k) for k in path)
            bytes_used = value.nbytes
            total_bytes += bytes_used

            param_breakdown[key] = {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "bytes": bytes_used,
                "mb": bytes_used / (1024 * 1024),
            }

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "parameter_breakdown": param_breakdown,
        "largest_parameter": max(
            param_breakdown.items(),
            key=lambda x: x[1]["bytes"],
            default=("none", {"bytes": 0}),
        )[0],
    }


def _analyze_tensor_memory(tensor: jnp.ndarray) -> dict[str, Any]:
    """Analyze memory usage of a tensor."""
    return {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "bytes": tensor.nbytes,
        "mb": tensor.nbytes / (1024 * 1024),
        "elements": tensor.size,
    }


def _analyze_gradient_memory(gradients) -> dict[str, Any]:
    """Analyze memory usage of gradients."""
    total_bytes = 0
    grad_breakdown = {}

    def analyze_grad_pytree(pytree, prefix=""):
        nonlocal total_bytes

        for key, value in pytree.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if hasattr(value, "nbytes"):
                bytes_used = value.nbytes
                total_bytes += bytes_used

                grad_breakdown[full_key] = {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "bytes": bytes_used,
                    "mb": bytes_used / (1024 * 1024),
                }
            elif isinstance(value, dict):
                analyze_grad_pytree(value, full_key)

    if hasattr(gradients, "items"):
        analyze_grad_pytree(gradients)

    return {
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "gradient_breakdown": grad_breakdown,
    }


def _analyze_memory_efficiency(
    param_memory: dict[str, Any],
    input_memory: dict[str, Any],
    output_memory: dict[str, Any],
    gradient_memory: dict[str, Any] | None,
) -> dict[str, Any]:
    """Analyze overall memory efficiency."""
    total_model_memory = param_memory["total_mb"]
    total_data_memory = input_memory["mb"] + output_memory["mb"]

    if gradient_memory and "total_mb" in gradient_memory:
        total_grad_memory = gradient_memory["total_mb"]
    else:
        total_grad_memory = 0

    total_memory = total_model_memory + total_data_memory + total_grad_memory

    # Calculate efficiency ratios
    param_ratio = total_model_memory / total_memory
    data_ratio = total_data_memory / total_memory
    grad_ratio = total_grad_memory / total_memory if total_grad_memory > 0 else 0

    # Memory efficiency categories
    efficiency_category = "unknown"
    if total_memory < 100:  # < 100 MB
        efficiency_category = "very_efficient"
    elif total_memory < 500:  # < 500 MB
        efficiency_category = "efficient"
    elif total_memory < 1000:  # < 1 GB
        efficiency_category = "moderate"
    else:
        efficiency_category = "memory_intensive"

    return {
        "total_memory_mb": total_memory,
        "parameter_ratio": param_ratio,
        "data_ratio": data_ratio,
        "gradient_ratio": grad_ratio,
        "efficiency_category": efficiency_category,
        "memory_breakdown": {
            "parameters_mb": total_model_memory,
            "data_mb": total_data_memory,
            "gradients_mb": total_grad_memory,
        },
    }


def _generate_optimization_suggestions(
    efficiency_analysis: dict[str, Any],
) -> list[str]:
    """Generate memory optimization suggestions."""
    suggestions = []

    total_memory = efficiency_analysis["total_memory_mb"]
    param_ratio = efficiency_analysis["parameter_ratio"]
    data_ratio = efficiency_analysis["data_ratio"]

    # High memory usage suggestions
    if total_memory > 1000:  # > 1 GB
        suggestions.append(
            "Consider using gradient checkpointing to reduce memory usage"
        )
        suggestions.append("Use mixed precision (bfloat16) to reduce memory footprint")

    # Parameter-heavy model suggestions
    if param_ratio > 0.7:  # Parameters use >70% of memory
        suggestions.append(
            "Model is parameter-heavy - consider parameter sharing or pruning"
        )
        suggestions.append("Use quantization techniques to reduce parameter memory")

    # Data-heavy computation suggestions
    if data_ratio > 0.5:  # Data uses >50% of memory
        suggestions.append("Consider processing data in smaller batches")
        suggestions.append("Use data pipeline optimization to reduce memory overhead")

    # JAX-specific suggestions
    suggestions.append("Use jax.jit compilation to optimize memory layout")
    suggestions.append("Consider using jax.remat for memory-compute tradeoffs")

    # Neural operator specific suggestions
    suggestions.append("For FNO models, consider reducing the number of Fourier modes")
    suggestions.append(
        "Use spectral convolution optimizations for better memory efficiency"
    )

    return suggestions


def compare_memory_usage(
    models: dict[str, nnx.Module], input_data: jnp.ndarray
) -> dict[str, Any]:
    """
    Compare memory usage across multiple models.

    Args:
        models: Dictionary of {name: model} to compare
        input_data: Sample input for all models

    Returns:
        Memory usage comparison results
    """
    results = {}

    for name, model in models.items():
        try:
            analysis = memory_usage_analysis(model, input_data, include_gradients=False)
            results[name] = analysis
        except Exception as e:
            results[name] = {"error": str(e)}

    # Add comparison metrics
    if len(results) > 1:
        memory_values = [
            r["efficiency_analysis"]["total_memory_mb"]
            for r in results.values()
            if "efficiency_analysis" in r
        ]

        if memory_values:
            min_memory = min(memory_values)
            results["_comparison"] = {
                "memory_efficiency_ratios": {
                    name: results[name]["efficiency_analysis"]["total_memory_mb"]
                    / min_memory
                    for name in results
                    if not name.startswith("_")
                    and "efficiency_analysis" in results[name]
                },
                "most_memory_efficient": min(
                    results.keys(),
                    key=lambda k: results[k]
                    .get("efficiency_analysis", {})
                    .get("total_memory_mb", float("inf")),
                ),
            }

    return results


def generate_memory_report(analysis: dict[str, Any]) -> str:
    """Generate a human-readable memory analysis report."""
    report = []
    report.append("=" * 60)
    report.append("MEMORY USAGE ANALYSIS REPORT")
    report.append("=" * 60)

    # Overall efficiency
    eff = analysis["efficiency_analysis"]
    report.append(f"Total Memory Usage: {eff['total_memory_mb']:.2f} MB")
    report.append(
        f"Efficiency Category: {eff['efficiency_category'].replace('_', ' ').title()}"
    )
    report.append("")

    # Memory breakdown
    report.append("MEMORY BREAKDOWN:")
    report.append(
        f"  Parameters: {eff['memory_breakdown']['parameters_mb']:.2f} MB "
        f"({eff['parameter_ratio']:.1%})"
    )
    report.append(
        f"  Data (I/O): {eff['memory_breakdown']['data_mb']:.2f} MB "
        f"({eff['data_ratio']:.1%})"
    )
    if eff["gradient_ratio"] > 0:
        report.append(
            f"  Gradients: {eff['memory_breakdown']['gradients_mb']:.2f} MB "
            f"({eff['gradient_ratio']:.1%})"
        )
    report.append("")

    # Optimization suggestions
    if "optimization_suggestions" in analysis:
        report.append("OPTIMIZATION SUGGESTIONS:")
        for i, suggestion in enumerate(analysis["optimization_suggestions"], 1):
            report.append(f"  {i}. {suggestion}")

    return "\n".join(report)
