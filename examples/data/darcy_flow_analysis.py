#!/usr/bin/env python3
"""Comprehensive Darcy Flow Analysis with FNO Validation.

This module provides comprehensive analysis tools for Darcy flow problems,
including gradient computation, validation metrics, and comparative analysis
of different solution methods.
"""

import argparse
import time
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Opifex Framework imports
from opifex.data.sources import DarcyDataSource


# Simplified data processing functions (inline implementation)
def normalize_field(field: jax.Array) -> jax.Array:
    """Normalize a field to zero mean and unit variance."""
    mean = jnp.mean(field)
    std = jnp.std(field)
    return (field - mean) / (std + 1e-8)  # Add small epsilon for numerical stability


def create_grid_coordinates(resolution: int) -> tuple[jax.Array, jax.Array]:
    """Create 2D grid coordinates for embedding visualization."""
    x = jnp.linspace(0, 1, resolution)
    y = jnp.linspace(0, 1, resolution)
    X, Y = jnp.meshgrid(x, y)
    return X, Y


def analyze_darcy_flow_dataset(
    n_samples: int = 100,
    resolutions: list[int] | None = None,
    sub_resolution: int = 8,
    viscosity_range: tuple[float, float] = (1e-5, 1e-3),
    force_coefficient: float = 1.0,
    output_dir: str = "darcy_analysis_output",
) -> dict[str, Any]:
    """
    Analyze Darcy flow dataset characteristics across multiple resolutions.

    Args:
        n_samples: Number of samples to generate for analysis
        resolutions: List of grid resolutions to test
        sub_resolution: Subsampling factor for coarse-graining
        viscosity_range: Range of viscosity values to sample
        force_coefficient: Force scaling coefficient
        output_dir: Directory to save analysis results

    Returns:
        Dictionary containing comprehensive analysis results
    """
    if resolutions is None:
        resolutions = [64, 128]

    print("=" * 80)
    print("🌊 DARCY FLOW DATASET ANALYSIS")
    print("=" * 80)

    results = {
        "parameters": {
            "n_samples": n_samples,
            "resolutions": resolutions,
            "sub_resolution": sub_resolution,
            "viscosity_range": viscosity_range,
            "force_coefficient": force_coefficient,
        },
        "datasets": {},
        "comparisons": {},
        "timing": {},
    }

    # Analyze each resolution
    for resolution in resolutions:
        print(f"\n📊 Analyzing resolution: {resolution}x{resolution}")

        # Create data source (Grain-based)
        data_source = DarcyDataSource(
            resolution=resolution,
            n_samples=n_samples,
            viscosity_range=viscosity_range,
            seed=42,
        )

        # Generate samples and measure timing
        start_time = time.time()
        samples = [data_source[i] for i in range(n_samples)]
        generation_time = time.time() - start_time

        # Analyze samples
        resolution_results = _analyze_resolution_samples(samples, resolution)
        resolution_results["generation_time"] = generation_time
        resolution_results["samples_per_second"] = n_samples / generation_time

        results["datasets"][resolution] = resolution_results

        print(f"  ✅ Generated {n_samples} samples in {generation_time:.2f}s")
        print(f"  ✅ Rate: {n_samples / generation_time:.1f} samples/second")

    # Cross-resolution comparisons
    if len(resolutions) > 1:
        results["comparisons"] = _compare_resolutions(results["datasets"])

    return results


def _analyze_resolution_samples(samples: list[dict], resolution: int) -> dict[str, Any]:
    """Analyze samples for a specific resolution."""
    inputs = jnp.stack([sample["input"] for sample in samples])
    outputs = jnp.stack([sample["output"] for sample in samples])

    return {
        "resolution": resolution,
        "input_stats": _compute_field_statistics(inputs),
        "output_stats": _compute_field_statistics(outputs),
        "spatial_patterns": _analyze_spatial_patterns(inputs, outputs),
        "data_quality": _assess_data_quality(inputs, outputs),
    }


def _compute_field_statistics(fields: jax.Array) -> dict[str, float]:
    """Compute comprehensive statistics for field data."""
    return {
        "mean": float(jnp.mean(fields)),
        "std": float(jnp.std(fields)),
        "min": float(jnp.min(fields)),
        "max": float(jnp.max(fields)),
        "median": float(jnp.median(fields)),
        "q25": float(jnp.percentile(fields, 25)),
        "q75": float(jnp.percentile(fields, 75)),
        "dynamic_range": float(jnp.max(fields) - jnp.min(fields)),
        "coefficient_of_variation": float(jnp.std(fields) / (jnp.mean(fields) + 1e-8)),
    }


def _analyze_spatial_patterns(inputs: jax.Array, outputs: jax.Array) -> dict[str, Any]:
    """Analyze spatial patterns in the data."""
    # Compute spatial gradients for each axis separately
    input_grad_x = jnp.asarray(jnp.gradient(inputs, axis=-1))
    input_grad_y = jnp.asarray(jnp.gradient(inputs, axis=-2))
    input_grad_magnitude = jnp.sqrt(jnp.square(input_grad_x) + jnp.square(input_grad_y))

    output_grad_x = jnp.asarray(jnp.gradient(outputs, axis=-1))
    output_grad_y = jnp.asarray(jnp.gradient(outputs, axis=-2))
    output_grad_magnitude = jnp.sqrt(
        jnp.square(output_grad_x) + jnp.square(output_grad_y)
    )

    # Compute correlation between inputs and outputs
    flat_inputs = inputs.reshape(inputs.shape[0], -1)
    flat_outputs = outputs.reshape(outputs.shape[0], -1)
    correlations = []

    for i in range(flat_inputs.shape[0]):
        corr = jnp.corrcoef(flat_inputs[i], flat_outputs[i])[0, 1]
        if not jnp.isnan(corr):
            correlations.append(float(corr))

    return {
        "input_gradient_stats": _compute_field_statistics(input_grad_magnitude),
        "output_gradient_stats": _compute_field_statistics(output_grad_magnitude),
        "input_output_correlation": {
            "mean": float(np.mean(correlations)) if correlations else 0.0,
            "std": float(np.std(correlations)) if correlations else 0.0,
        },
        "gradient_correlation": float(
            jnp.corrcoef(
                input_grad_magnitude.flatten(), output_grad_magnitude.flatten()
            )[0, 1]
        )
        if input_grad_magnitude.size > 0
        else 0.0,
    }


def _assess_data_quality(inputs: jax.Array, outputs: jax.Array) -> dict[str, Any]:
    """Assess data quality metrics."""
    return {
        "has_nan": bool(jnp.any(jnp.isnan(inputs)) or jnp.any(jnp.isnan(outputs))),
        "has_inf": bool(jnp.any(jnp.isinf(inputs)) or jnp.any(jnp.isinf(outputs))),
        "input_range_valid": bool(jnp.all(inputs >= 0)),
        "output_finite": bool(jnp.all(jnp.isfinite(outputs))),
        "shape_consistency": inputs.shape[:-1] == outputs.shape[:-1],
    }


def _compare_resolutions(datasets: dict[int, dict]) -> dict[str, Any]:
    """Compare datasets across different resolutions."""
    resolutions = sorted(datasets.keys())

    if len(resolutions) < 2:
        return {}

    comparisons = {
        "resolution_scaling": {},
        "performance_scaling": {},
        "quality_comparison": {},
    }

    # Resolution scaling analysis
    for i, res in enumerate(resolutions[1:], 1):
        prev_res = resolutions[i - 1]
        scale_factor = res / prev_res

        # Performance scaling
        prev_time = datasets[prev_res]["generation_time"]
        curr_time = datasets[res]["generation_time"]
        time_scaling = curr_time / prev_time

        comparisons["performance_scaling"][f"{prev_res}_to_{res}"] = {
            "resolution_scale": scale_factor,
            "time_scale": time_scaling,
            "efficiency_ratio": scale_factor**2 / time_scaling,
        }

    return comparisons


def create_visualization(
    results: dict[str, Any],
    preprocessing_results: dict[str, Any],
    save_path: str | None = None,
) -> None:
    """Create comprehensive visualization of analysis results."""
    _create_resolution_comparison_plots(results, save_path)
    _create_statistical_summary_plots(results, save_path)
    _create_performance_analysis_plots(results, save_path)


def _create_resolution_comparison_plots(
    results: dict[str, Any], save_path: str | None
) -> None:
    """Create plots comparing different resolutions."""
    datasets = results["datasets"]
    if len(datasets) < 2:
        return

    _, axes = plt.subplots(2, 2, figsize=(15, 12))
    resolutions = list(datasets.keys())

    # Plot 1: Mean values comparison
    input_means = [datasets[res]["input_stats"]["mean"] for res in resolutions]
    output_means = [datasets[res]["output_stats"]["mean"] for res in resolutions]

    axes[0, 0].plot(resolutions, input_means, "o-", label="Input (Permeability)")
    axes[0, 0].plot(resolutions, output_means, "s-", label="Output (Pressure)")
    axes[0, 0].set_xlabel("Resolution")
    axes[0, 0].set_ylabel("Mean Value")
    axes[0, 0].set_title("Mean Values vs Resolution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Standard deviation comparison
    input_stds = [datasets[res]["input_stats"]["std"] for res in resolutions]
    output_stds = [datasets[res]["output_stats"]["std"] for res in resolutions]

    axes[0, 1].plot(resolutions, input_stds, "o-", label="Input (Permeability)")
    axes[0, 1].plot(resolutions, output_stds, "s-", label="Output (Pressure)")
    axes[0, 1].set_xlabel("Resolution")
    axes[0, 1].set_ylabel("Standard Deviation")
    axes[0, 1].set_title("Variability vs Resolution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Dynamic range comparison
    input_ranges = [
        datasets[res]["input_stats"]["dynamic_range"] for res in resolutions
    ]
    output_ranges = [
        datasets[res]["output_stats"]["dynamic_range"] for res in resolutions
    ]

    axes[1, 0].plot(resolutions, input_ranges, "o-", label="Input (Permeability)")
    axes[1, 0].plot(resolutions, output_ranges, "s-", label="Output (Pressure)")
    axes[1, 0].set_xlabel("Resolution")
    axes[1, 0].set_ylabel("Dynamic Range")
    axes[1, 0].set_title("Dynamic Range vs Resolution")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Performance scaling
    generation_times = [datasets[res]["generation_time"] for res in resolutions]
    samples_per_sec = [datasets[res]["samples_per_second"] for res in resolutions]

    ax4_twin = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(
        resolutions, generation_times, "ro-", label="Generation Time (s)"
    )
    line2 = ax4_twin.plot(resolutions, samples_per_sec, "bs-", label="Samples/Second")

    axes[1, 1].set_xlabel("Resolution")
    axes[1, 1].set_ylabel("Generation Time (s)", color="red")
    ax4_twin.set_ylabel("Samples per Second", color="blue")
    axes[1, 1].set_title("Performance vs Resolution")

    lines = line1 + line2
    axes[1, 1].legend(lines, [l.get_label() for l in lines], loc="center right")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(
            f"{save_path}_resolution_comparison.png", dpi=300, bbox_inches="tight"
        )
    plt.show()


def _create_statistical_summary_plots(
    results: dict[str, Any], save_path: str | None
) -> None:
    """Create statistical summary visualizations."""
    _ = results["datasets"]

    _, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Summary statistics will be added based on available data
    axes[0, 0].text(
        0.5,
        0.5,
        "Statistical Summary\n(Implementation Placeholder)",
        ha="center",
        va="center",
        transform=axes[0, 0].transAxes,
    )
    axes[0, 0].set_title("Input Statistics Summary")

    plt.tight_layout()
    if save_path:
        plt.savefig(
            f"{save_path}_statistical_summary.png", dpi=300, bbox_inches="tight"
        )
    plt.show()


def _create_performance_analysis_plots(
    results: dict[str, Any], save_path: str | None
) -> None:
    """Create performance analysis visualizations."""
    if "comparisons" not in results or not results["comparisons"]:
        return

    _, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Performance analysis will be added based on available data
    axes[0].text(
        0.5,
        0.5,
        "Performance Analysis\n(Implementation Placeholder)",
        ha="center",
        va="center",
        transform=axes[0].transAxes,
    )
    axes[0].set_title("Scaling Performance")

    axes[1].text(
        0.5,
        0.5,
        "Efficiency Metrics\n(Implementation Placeholder)",
        ha="center",
        va="center",
        transform=axes[1].transAxes,
    )
    axes[1].set_title("Computational Efficiency")

    plt.tight_layout()
    if save_path:
        plt.savefig(
            f"{save_path}_performance_analysis.png", dpi=300, bbox_inches="tight"
        )
    plt.show()


def main():
    """Run comprehensive Darcy flow analysis validation."""
    parser = argparse.ArgumentParser(
        description="Analyze Darcy flow dataset characteristics"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to generate for analysis",
    )
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[64, 128],
        help="Grid resolutions to analyze",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="darcy_analysis_output",
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_plots", action="store_true", help="Save generated plots to disk"
    )

    args = parser.parse_args()

    # Run analysis
    print("Starting comprehensive Darcy flow dataset analysis...")
    results = analyze_darcy_flow_dataset(
        n_samples=args.n_samples,
        resolutions=args.resolutions,
        output_dir=args.output_dir,
    )

    # Create visualizations
    preprocessing_results = {}  # Placeholder for future preprocessing analysis

    save_path = f"{args.output_dir}/darcy_analysis" if args.save_plots else None
    create_visualization(results, preprocessing_results, save_path)

    # Print summary
    print("\n" + "=" * 80)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 80)
    for res, data in results["datasets"].items():
        print(f"Resolution {res}x{res}:")
        print(f"  Generation time: {data['generation_time']:.2f}s")
        print(f"  Samples/second: {data['samples_per_second']:.1f}")
        print(f"  Input mean: {data['input_stats']['mean']:.4f}")
        print(f"  Output mean: {data['output_stats']['mean']:.4f}")


if __name__ == "__main__":
    main()
