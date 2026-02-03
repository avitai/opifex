"""
Performance visualization tools for Opifex framework.

Provides plotting utilities for FLOPS analysis, memory usage,
model complexity comparison, and performance benchmarking results.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_flops_analysis(
    flops_results: dict[str, Any],
    title: str = "FLOPS Analysis",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot FLOPS analysis results including breakdown and comparisons.

    Args:
        flops_results: Results from JAXFlopCounter or benchmark_neural_operator
        title: Plot title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Extract data based on results structure
    if "total_flops" in flops_results or "forward_flops" in flops_results:
        # Single model results
        _plot_single_model_flops(flops_results, axes)
    else:
        # Multiple models or benchmark results
        _plot_multi_model_flops(flops_results, axes)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def _plot_single_model_flops(results: dict[str, Any], axes: list[plt.Axes]):
    """Plot FLOPS analysis for a single model."""
    # FLOPS breakdown pie chart
    ax = axes[0]
    if "total_flops" in results:
        backward_flops = results.get("backward_flops", 0)
        # Use existing forward_flops or derive from total (total = forward + backward)
        forward_flops = results.get(
            "forward_flops", results.get("total_flops", 0) - backward_flops
        )

        if backward_flops > 0:
            labels = ["Forward Pass", "Backward Pass"]
            sizes = [forward_flops, backward_flops]
            colors = ["lightblue", "lightcoral"]
        else:
            labels = ["Forward Pass"]
            sizes = [forward_flops]
            colors = ["lightblue"]

        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("FLOPS Breakdown")
    else:
        ax.text(
            0.5,
            0.5,
            "No FLOPS data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("FLOPS Breakdown")

    # Timing comparison
    ax = axes[1]
    if "forward_time" in results:
        forward_time = results.get("forward_time", 0)
        backward_time = results.get("backward_time", 0)

        categories = ["Forward", "Backward"] if backward_time > 0 else ["Forward"]
        times = (
            [forward_time * 1000, backward_time * 1000]
            if backward_time > 0
            else [forward_time * 1000]
        )

        bars = ax.bar(
            categories,
            times,
            color=["skyblue", "salmon"] if len(times) > 1 else ["skyblue"],
        )
        ax.set_ylabel("Time (ms)")
        ax.set_title("Execution Time")

        # Add value labels on bars
        for bar, time in zip(bars, times, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{time:.2f}ms",
                ha="center",
                va="bottom",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No timing data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Execution Time")

    # Model parameters info
    ax = axes[2]
    if "model_parameters" in results:
        params = results["model_parameters"]
        total_params = params.get("total_parameters", 0)
        trainable_params = params.get("trainable_parameters", 0)

        # Simple bar chart
        categories = ["Total", "Trainable"]
        param_counts = [total_params, trainable_params]

        bars = ax.bar(categories, param_counts, color=["lightgreen", "darkgreen"])
        ax.set_ylabel("Parameter Count")
        ax.set_title("Model Parameters")

        # Add value labels
        for bar, count in zip(bars, param_counts, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{count:,}",
                ha="center",
                va="bottom",
            )
    else:
        ax.text(
            0.5,
            0.5,
            "No parameter data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Model Parameters")

    # FLOPS efficiency (FLOPS per parameter)
    ax = axes[3]
    if "total_flops" in results and "model_parameters" in results:
        total_flops = results["total_flops"]
        total_params = results["model_parameters"]["total_parameters"]

        if total_params > 0:
            efficiency = total_flops / total_params

            # Simple efficiency visualization
            ax.bar(["FLOPS/Parameter"], [efficiency], color="orange")
            ax.set_ylabel("FLOPS per Parameter")
            ax.set_title("Computational Efficiency")
            ax.text(
                0,
                efficiency + efficiency * 0.01,
                f"{efficiency:.2e}",
                ha="center",
                va="bottom",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Cannot compute efficiency",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Computational Efficiency")
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data for efficiency",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Computational Efficiency")


def _plot_multi_model_flops(results: dict[str, Any], axes: list[plt.Axes]):
    """Plot FLOPS analysis for multiple models or benchmark results."""
    # Extract model names and FLOPS data
    model_names = []
    forward_flops = []
    backward_flops = []
    total_flops = []

    for name, data in results.items():
        if name.startswith("_"):  # Skip comparison metadata
            continue
        if isinstance(data, dict) and (
            "forward_flops" in data or "total_flops" in data
        ):
            model_names.append(name)
            backward_val = data.get("backward_flops", 0)
            total_val = data.get("total_flops", 0)
            forward_val = data.get("forward_flops", total_val - backward_val)

            forward_flops.append(forward_val)
            backward_flops.append(backward_val)
            total_flops.append(total_val)

    if not model_names:
        # Handle benchmark results format
        for shape_key, data in results.items():
            if isinstance(data, dict) and "avg_total_flops" in data:
                model_names.append(shape_key)
                forward_flops.append(data.get("avg_forward_flops", 0))
                backward_flops.append(data.get("avg_backward_flops", 0))
                total_flops.append(data.get("avg_total_flops", 0))

    if model_names:
        # FLOPS comparison bar chart
        ax = axes[0]
        x = np.arange(len(model_names))
        width = 0.35

        ax.bar(x - width / 2, forward_flops, width, label="Forward", color="skyblue")
        ax.bar(x + width / 2, backward_flops, width, label="Backward", color="salmon")

        ax.set_xlabel("Models")
        ax.set_ylabel("FLOPS")
        ax.set_title("FLOPS Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()

        # Total FLOPS comparison
        ax = axes[1]
        ax.bar(model_names, total_flops, color="lightgreen")
        ax.set_xlabel("Models")
        ax.set_ylabel("Total FLOPS")
        ax.set_title("Total FLOPS Comparison")
        ax.tick_params(axis="x", rotation=45)

        # Efficiency ratios (if available)
        ax = axes[2]
        if "_comparison" in results and "efficiency_ratios" in results["_comparison"]:
            ratios = results["_comparison"]["efficiency_ratios"]
            names = list(ratios.keys())
            ratio_values = list(ratios.values())

            ax.bar(names, ratio_values, color="orange")
            ax.set_xlabel("Models")
            ax.set_ylabel("Efficiency Ratio")
            ax.set_title("Relative Efficiency (vs. most efficient)")
            ax.tick_params(axis="x", rotation=45)
            ax.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="Baseline")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No efficiency comparison available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Relative Efficiency")

        # FLOPS scaling analysis
        ax = axes[3]
        if len(forward_flops) > 1:
            # Plot FLOPS vs model index (proxy for complexity)
            ax.plot(
                range(len(total_flops)),
                total_flops,
                "o-",
                color="purple",
                linewidth=2,
                markersize=8,
            )
            ax.set_xlabel("Model Index")
            ax.set_ylabel("Total FLOPS")
            ax.set_title("FLOPS Scaling")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "Need multiple models for scaling analysis",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("FLOPS Scaling")


def plot_memory_usage(
    memory_results: dict[str, Any],
    title: str = "Memory Usage Analysis",
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot memory usage analysis results.

    Args:
        memory_results: Results from memory_usage_analysis
        title: Plot title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Memory breakdown pie chart
    ax = axes[0]
    if "efficiency_analysis" in memory_results:
        eff = memory_results["efficiency_analysis"]
        breakdown = eff.get("memory_breakdown", {})

        if breakdown:
            labels = []
            sizes = []
            colors = ["lightblue", "lightcoral", "lightgreen"]

            for _i, (key, value) in enumerate(breakdown.items()):
                if value > 0:
                    labels.append(key.replace("_mb", "").replace("_", " ").title())
                    sizes.append(value)

            if sizes:
                ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors[: len(sizes)],
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title("Memory Breakdown")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No memory data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No memory breakdown available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # Memory timeline (if available)
    ax = axes[1]
    if "profiling_timeline" in memory_results:
        timeline = memory_results["profiling_timeline"]
        timeline_data = timeline.get("timeline", [])

        if timeline_data and len(timeline_data) > 1:
            times = [point[0] for point in timeline_data]
            memories = [point[1] for point in timeline_data]

            ax.plot(times, memories, "b-", linewidth=2, marker="o", markersize=6)
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Memory Usage (MB)")
            ax.set_title("Memory Timeline")
            ax.grid(True, alpha=0.3)

            # Add labels for checkpoints with labels
            for point in timeline_data:
                if len(point) > 2:  # Has label
                    ax.annotate(
                        point[2],
                        (point[0], point[1]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                    )
        else:
            ax.text(
                0.5,
                0.5,
                "No timeline data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Memory Timeline")

    # Memory efficiency category
    ax = axes[2]
    if "efficiency_analysis" in memory_results:
        eff = memory_results["efficiency_analysis"]
        category = eff.get("efficiency_category", "unknown")
        total_memory = eff.get("total_memory_mb", 0)

        # Color code efficiency
        colors = {
            "very_efficient": "green",
            "efficient": "lightgreen",
            "moderate": "orange",
            "memory_intensive": "red",
            "unknown": "gray",
        }

        color = colors.get(category, "gray")

        # Simple bar showing total memory with color coding
        bar = ax.bar(["Total Memory"], [total_memory], color=color)
        ax.set_ylabel("Memory (MB)")
        ax.set_title(f"Memory Efficiency: {category.replace('_', ' ').title()}")

        # Add value label
        ax.text(
            0,
            total_memory + total_memory * 0.01,
            f"{total_memory:.2f} MB",
            ha="center",
            va="bottom",
        )

        # Add efficiency thresholds as horizontal lines
        ax.axhline(
            y=100,
            color="green",
            linestyle="--",
            alpha=0.5,
            label="Very Efficient (<100MB)",
        )
        ax.axhline(
            y=500, color="orange", linestyle="--", alpha=0.5, label="Moderate (<500MB)"
        )
        ax.axhline(
            y=1000, color="red", linestyle="--", alpha=0.5, label="Intensive (<1GB)"
        )
        ax.legend(fontsize=8)

    # Optimization suggestions
    ax = axes[3]
    if "optimization_suggestions" in memory_results:
        suggestions = memory_results["optimization_suggestions"]

        if suggestions:
            # Count suggestions by category
            categories = {
                "Memory Reduction": 0,
                "JAX Optimization": 0,
                "Model Optimization": 0,
                "Data Pipeline": 0,
            }

            for suggestion in suggestions:
                if "memory" in suggestion.lower() or "batch" in suggestion.lower():
                    categories["Memory Reduction"] += 1
                elif "jax" in suggestion.lower() or "jit" in suggestion.lower():
                    categories["JAX Optimization"] += 1
                elif "model" in suggestion.lower() or "parameter" in suggestion.lower():
                    categories["Model Optimization"] += 1
                else:
                    categories["Data Pipeline"] += 1

            # Filter out zero categories
            categories = {k: v for k, v in categories.items() if v > 0}

            if categories:
                bars = ax.bar(
                    categories.keys(),
                    categories.values(),
                    color=["lightblue", "lightcoral", "lightgreen", "orange"],
                )
                ax.set_ylabel("Number of Suggestions")
                ax.set_title("Optimization Suggestions by Category")
                ax.tick_params(axis="x", rotation=45)

                # Add value labels
                for bar, count in zip(bars, categories.values(), strict=False):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        str(count),
                        ha="center",
                        va="bottom",
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No optimization suggestions",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No optimization suggestions available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_model_complexity_comparison(
    complexity_results: dict[str, Any],
    title: str = "Model Complexity Comparison",
    figsize: tuple[int, int] = (14, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot comprehensive model complexity comparison.

    Args:
        complexity_results: Results from compare_model_complexities
        title: Plot title
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Extract model data
    model_names = []
    total_params = []
    memory_usage = []
    total_ops = []

    for name, data in complexity_results.items():
        if name.startswith("_"):  # Skip comparison metadata
            continue
        if isinstance(data, dict) and "parameters" in data:
            model_names.append(name)
            total_params.append(data["parameters"]["total_parameters"])

            if "memory" in data and "total_estimated_mb" in data["memory"]:
                memory_usage.append(data["memory"]["total_estimated_mb"])
            else:
                memory_usage.append(0)

            if "computational" in data:
                total_ops.append(data["computational"]["total_estimated_operations"])
            else:
                total_ops.append(0)

    if not model_names:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No model data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.suptitle(title, fontsize=16)
        return fig

    # Parameter comparison
    ax = axes[0]
    ax.bar(model_names, total_params, color="lightblue")
    ax.set_ylabel("Total Parameters")
    ax.set_title("Model Parameters")
    ax.tick_params(axis="x", rotation=45)

    # Memory usage comparison
    ax = axes[1]
    ax.bar(model_names, memory_usage, color="lightcoral")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Memory Usage")
    ax.tick_params(axis="x", rotation=45)

    # Computational complexity
    ax = axes[2]
    ax.bar(model_names, total_ops, color="lightgreen")
    ax.set_ylabel("Total Operations")
    ax.set_title("Computational Complexity")
    ax.tick_params(axis="x", rotation=45)

    # Efficiency scatter plot (Parameters vs Memory)
    ax = axes[3]
    scatter = ax.scatter(
        total_params, memory_usage, c=total_ops, cmap="viridis", s=100, alpha=0.7
    )
    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Efficiency Analysis")
    plt.colorbar(scatter, ax=ax, label="Operations")

    # Add model name labels
    for i, name in enumerate(model_names):
        ax.annotate(
            name,
            (total_params[i], memory_usage[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Relative efficiency (if comparison data available)
    ax = axes[4]
    if (
        "_comparison" in complexity_results
        and "parameter_efficiency" in complexity_results["_comparison"]
    ):
        ratios = complexity_results["_comparison"]["parameter_efficiency"]
        names = list(ratios.keys())
        ratio_values = list(ratios.values())

        ax.bar(names, ratio_values, color="orange")
        ax.set_ylabel("Parameter Efficiency Ratio")
        ax.set_title("Relative Parameter Efficiency")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="Baseline")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No efficiency comparison available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Relative Efficiency")

    # Model type distribution (if available)
    ax = axes[5]
    model_types = []
    for name, data in complexity_results.items():
        if not name.startswith("_") and isinstance(data, dict) and "model_type" in data:
            model_types.append(data["model_type"])

    if model_types:
        type_counts = {}
        for model_type in model_types:
            type_counts[model_type] = type_counts.get(model_type, 0) + 1

        ax.pie(
            type_counts.values(),
            labels=type_counts.keys(),
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Model Type Distribution")
    else:
        ax.text(
            0.5,
            0.5,
            "No model type data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Model Type Distribution")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
