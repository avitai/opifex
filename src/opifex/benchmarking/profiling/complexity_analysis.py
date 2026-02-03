"""
Model Complexity Analysis for Opifex Neural Operators.

Provides comprehensive analysis of model complexity including parameter counts,
memory usage, computational complexity, and scaling characteristics.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


def model_complexity_analysis(
    model: nnx.Module, input_shape: tuple[int, ...]
) -> dict[str, Any]:
    """
    Comprehensive model complexity analysis.

    Args:
        model: Flax NNX model to analyze
        input_shape: Shape of input data (including batch dimension)

    Returns:
        Dictionary with detailed complexity metrics
    """
    # Generate sample input
    key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(key, input_shape)

    # Parameter analysis
    param_analysis = _analyze_parameters(model)

    # Memory analysis
    memory_analysis = _analyze_memory_usage(model, sample_input)

    # Computational complexity
    comp_analysis = _analyze_computational_complexity(model, input_shape)

    # Scaling characteristics
    scaling_analysis = _analyze_scaling_properties(model, input_shape)

    return {
        "parameters": param_analysis,
        "memory": memory_analysis,
        "computational": comp_analysis,
        "scaling": scaling_analysis,
        "input_shape": input_shape,
        "model_type": _identify_model_type(model),
    }


def _analyze_parameters(model: nnx.Module) -> dict[str, Any]:
    """Analyze model parameters in detail."""
    params = nnx.state(model, nnx.Param)
    # Convert NNX State to pytree for iteration
    params_tree = nnx.to_tree(params)

    total_params = 0
    param_breakdown: dict[str, dict[str, Any]] = {}
    largest_layer = {"name": "", "params": 0}

    # Use JAX tree utilities to iterate over parameters with paths
    flat_with_path = jax.tree_util.tree_leaves_with_path(params_tree)

    for path, value in flat_with_path:
        if hasattr(value, "shape"):
            # Build a readable key from the path
            key = "/".join(str(k.key if hasattr(k, "key") else k) for k in path)
            param_count = int(jnp.prod(jnp.array(value.shape)))
            total_params += param_count

            param_breakdown[key] = {
                "shape": value.shape,
                "parameters": param_count,
                "dtype": str(value.dtype),
            }

            # Track largest layer (simplified - treat each param as its own layer)
            if param_count > largest_layer["params"]:
                largest_layer = {"name": key, "params": param_count}

    # Calculate memory usage of parameters
    # Assume float32 parameters (4 bytes each)
    param_memory_mb = (total_params * 4) / (1024 * 1024)

    return {
        "total_parameters": total_params,
        "parameter_memory_mb": param_memory_mb,
        "largest_layer": largest_layer,
        "parameter_breakdown": param_breakdown,
        "memory_efficient": param_memory_mb < 100,  # Arbitrary threshold
    }


def _analyze_memory_usage(model: nnx.Module, sample_input: jax.Array) -> dict[str, Any]:
    """Analyze memory usage during forward pass."""
    input_memory = sample_input.nbytes / (1024 * 1024)  # MB

    # Estimate intermediate activations
    # This is approximate - real profiling would require JAX memory profiling
    try:
        output = model(sample_input)  # type: ignore[misc]
        output_memory = output.nbytes / (1024 * 1024)  # MB

        # Rough estimate of intermediate activations
        # Neural operators typically have similar-sized intermediates
        estimated_intermediate = input_memory * 3  # Heuristic

        total_memory = input_memory + output_memory + estimated_intermediate

        return {
            "estimated_intermediate_mb": estimated_intermediate,
            "total_estimated_mb": total_memory,
            "memory_efficient": total_memory < 1000,  # 1GB threshold
        }
    except Exception as e:
        return {"input_memory_mb": input_memory, "error": f"Could not analyze: {e!s}"}


def _analyze_computational_complexity(
    model: nnx.Module, input_shape: tuple[int, ...]
) -> dict[str, Any]:
    """Analyze computational complexity."""
    batch_size = input_shape[0]
    spatial_dims = input_shape[1:]
    spatial_size = jnp.prod(jnp.array(spatial_dims))

    # Theoretical complexity analysis for common neural operator components
    complexity_analysis: dict[str, dict[str, Any]] = {}
    # FFT-based operations (common in FNO)
    if len(spatial_dims) >= 2:
        fft_complexity = spatial_size * math.log2(spatial_size)
        complexity_analysis["fft_operations"] = {
            "complexity_class": "O(N log N)",
            "estimated_ops": int(fft_complexity),
            "scaling_factor": "N log N where N is spatial size",
        }

    # Convolution operations
    conv_complexity = spatial_size**2  # Simplified estimate
    complexity_analysis["convolution_operations"] = {
        "complexity_class": "O(N²)",
        "estimated_ops": int(conv_complexity),
        "scaling_factor": "N² where N is spatial size",
    }

    # Linear operations
    linear_complexity = spatial_size
    complexity_analysis["linear_operations"] = {
        "complexity_class": "O(N)",
        "estimated_ops": int(linear_complexity),
        "scaling_factor": "N where N is spatial size",
    }

    total_ops = sum(op["estimated_ops"] for op in complexity_analysis.values())

    return {
        "total_estimated_operations": total_ops,
        "operations_per_sample": total_ops // batch_size,
        "complexity_breakdown": complexity_analysis,
        "dominant_complexity": _determine_dominant_complexity(complexity_analysis),
    }


def _analyze_scaling_properties(
    model: nnx.Module, input_shape: tuple[int, ...]
) -> dict[str, Any]:
    """Analyze how the model scales with input size."""
    base_spatial_size = jnp.prod(jnp.array(input_shape[1:]))

    # Test scaling at different resolutions
    scaling_tests: list[dict[str, Any]] = []

    for scale_factor in [0.5, 1.0, 2.0]:
        if scale_factor != 1.0:
            # Create scaled input shape
            scaled_spatial = [int(dim * scale_factor) for dim in input_shape[1:]]
            scaled_shape = (input_shape[0], *tuple(scaled_spatial))
            scaled_size = jnp.prod(jnp.array(scaled_spatial))
        else:
            scaled_shape = input_shape
            scaled_size = base_spatial_size

        scaling_tests.append(
            {
                "scale_factor": scale_factor,
                "spatial_size": int(scaled_size),
                "scaled_shape": scaled_shape,
                "theoretical_fft_ops": int(scaled_size * math.log2(scaled_size)),
                "theoretical_conv_ops": int(scaled_size**2),
                "memory_scaling": int(scaled_size * 4 / (1024 * 1024)),  # MB
            }
        )

    return {
        "base_spatial_size": int(base_spatial_size),
        "scaling_tests": scaling_tests,
        "scaling_characteristics": {
            "fft_scaling": "O(N log N)",
            "convolution_scaling": "O(N²)",
            "memory_scaling": "O(N)",
            "parameter_scaling": "O(1)",  # Parameters don't scale with input
        },
    }


def _identify_model_type(model: nnx.Module) -> str:
    """Identify the type of neural operator model."""
    model_name = type(model).__name__.lower()

    if "fno" in model_name or "fourier" in model_name:
        return "Fourier Neural Operator"
    if "deeponet" in model_name:
        return "Deep Operator Network"
    if "uno" in model_name:
        return "U-Neural Operator"
    if "gino" in model_name:
        return "Graph-Informed Neural Operator"
    if "uqno" in model_name:
        return "Uncertainty Quantification Neural Operator"
    return "Unknown Neural Operator"


def _determine_dominant_complexity(complexity_breakdown: dict[str, Any]) -> str:
    """Determine which operation dominates computational complexity."""
    max_ops = 0
    dominant_op = "unknown"

    for op_name, op_info in complexity_breakdown.items():
        if op_info["estimated_ops"] > max_ops:
            max_ops = op_info["estimated_ops"]
            dominant_op = op_name

    return dominant_op


def compare_model_complexities(
    models: dict[str, nnx.Module], input_shape: tuple[int, ...]
) -> dict[str, Any]:
    """
    Compare complexity across multiple models.

    Args:
        models: Dictionary of {name: model} to compare
        input_shape: Input shape for analysis

    Returns:
        Comparison results across all models
    """
    results: dict[str, dict[str, Any]] = {}

    for name, model in models.items():
        try:
            analysis = model_complexity_analysis(model, input_shape)
            results[name] = analysis
        except Exception as e:
            results[name] = {"error": str(e)}

    # Add comparative metrics
    if len(results) > 1:
        param_counts = [
            r["parameters"]["total_parameters"]
            for r in results.values()
            if "parameters" in r
        ]

        if param_counts:
            min_params = min(param_counts)
            results["_comparison"] = {
                "parameter_efficiency": {
                    name: results[name]["parameters"]["total_parameters"] / min_params
                    for name in results
                    if not name.startswith("_") and "parameters" in results[name]
                },
                "most_efficient": min(
                    results.keys(),
                    key=lambda k: (
                        results[k]
                        .get("parameters", {})
                        .get("total_parameters", float("inf"))
                    ),
                ),
            }

    return results


def generate_complexity_report(analysis: dict[str, Any]) -> str:
    """Generate a human-readable complexity report."""
    report: list[str] = []
    report.append("=" * 60)
    report.append("MODEL COMPLEXITY ANALYSIS REPORT")
    report.append("=" * 60)

    # Model info
    report.append(f"Model Type: {analysis['model_type']}")
    report.append(f"Input Shape: {analysis['input_shape']}")
    report.append("")

    # Parameters
    params = analysis["parameters"]
    report.append("PARAMETER ANALYSIS:")
    report.append(f"  Total Parameters: {params['total_parameters']:,}")
    report.append(f"  Parameter Memory: {params['parameter_memory_mb']:.2f} MB")
    report.append(
        f"  Largest Layer: {params['largest_layer']['name']} "
        f"({params['largest_layer']['params']:,} params)"
    )
    report.append("")

    # Memory
    memory = analysis["memory"]
    if "total_estimated_mb" in memory:
        report.append("MEMORY ANALYSIS:")
        report.append(f"  Input Memory: {memory['input_memory_mb']:.2f} MB")
        report.append(f"  Output Memory: {memory['output_memory_mb']:.2f} MB")
        report.append(f"  Total Estimated: {memory['total_estimated_mb']:.2f} MB")
        report.append("")

    # Computational
    comp = analysis["computational"]
    report.append("COMPUTATIONAL COMPLEXITY:")
    report.append(f"  Total Operations: {comp['total_estimated_operations']:,}")
    report.append(f"  Dominant Operation: {comp['dominant_complexity']}")
    report.append("")

    # Scaling
    scaling = analysis["scaling"]
    report.append("SCALING CHARACTERISTICS:")
    for op_type, scaling_behavior in scaling["scaling_characteristics"].items():
        report.append(f"  {op_type.replace('_', ' ').title()}: {scaling_behavior}")

    return "\n".join(report)
