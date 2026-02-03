"""Grid Embeddings Example - Opifex Framework.

Comprehensive demonstration of grid embedding techniques for neural operators,
reproducing and extending neuraloperator plot_embeddings.py with Opifex framework.

This example demonstrates:
- GridEmbedding2D for 2D spatial coordinate injection
- GridEmbeddingND for N-dimensional coordinate embedding
- SinusoidalEmbedding for frequency-based positional encoding
- Comparison of different embedding methods
- Visualization of embedding effects on data

Usage:
    python examples/layers/grid_embeddings_example.py
"""

import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from opifex.neural.operators.common.embeddings import (
    GridEmbedding2D,
    GridEmbeddingND,
    regular_grid_2d,
    SinusoidalEmbedding,
)


def demonstrate_grid_embedding_2d(
    spatial_shape: tuple[int, int] = (32, 32),
    batch_size: int = 4,
    in_channels: int = 3,
    grid_boundaries: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Demonstrate 2D grid embedding functionality.

    Args:
        spatial_shape: Spatial dimensions (height, width)
        batch_size: Batch size for demonstration
        in_channels: Number of input channels
        grid_boundaries: Coordinate boundaries [[x_min, x_max], [y_min, y_max]]

    Returns:
        Dictionary containing embedded data, coordinate grids, and embedding info
    """
    if grid_boundaries is None:
        grid_boundaries = [[0.0, 1.0], [0.0, 1.0]]

    print("\n🔲 Grid Embedding 2D Demonstration")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Input Channels: {in_channels}")
    print(f"   Grid Boundaries: {grid_boundaries}")

    # Create grid embedding
    embedding = GridEmbedding2D(
        in_channels=in_channels, grid_boundaries=grid_boundaries
    )

    # Generate sample input data
    rng_key = jax.random.PRNGKey(42)
    sample_input = jax.random.normal(rng_key, (batch_size, *spatial_shape, in_channels))

    # Apply embedding
    start_time = time.time()
    embedded_data = embedding(sample_input)
    embedding_time = time.time() - start_time

    # Get coordinate grids
    x_grid, y_grid = embedding.get_grid(spatial_shape)

    # Gather embedding information
    embedding_info = {
        "input_shape": sample_input.shape,
        "output_shape": embedded_data.shape,
        "in_channels": embedding.in_channels,
        "out_channels": embedding.out_channels,
        "coordinate_channels": 2,
        "grid_boundaries": grid_boundaries,
        "embedding_time_ms": embedding_time * 1000,
        "memory_increase_factor": embedded_data.size / sample_input.size,
    }

    print(f"   ✅ Input Shape: {sample_input.shape}")
    print(f"   ✅ Output Shape: {embedded_data.shape}")
    print(f"   ✅ Output Channels: {embedding.out_channels}")
    print(f"   ✅ Embedding Time: {embedding_time * 1000:.2f} ms")

    return {
        "embedded_data": embedded_data,
        "coordinate_grids": (x_grid, y_grid),
        "embedding_info": embedding_info,
        "sample_input": sample_input,
        "embedding": embedding,
    }


def demonstrate_grid_embedding_nd(
    spatial_shape: tuple[int, ...],
    batch_size: int = 2,
    in_channels: int = 2,
    grid_boundaries: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Demonstrate N-dimensional grid embedding functionality.

    Args:
        spatial_shape: Spatial dimensions (d1, d2, ..., dn)
        batch_size: Batch size for demonstration
        in_channels: Number of input channels
        grid_boundaries: List of coordinate boundaries for each dimension

    Returns:
        Dictionary containing embedded data, coordinate grids, and embedding info
    """
    dim = len(spatial_shape)
    if grid_boundaries is None:
        grid_boundaries = [[0.0, 1.0] for _ in range(dim)]

    print(f"\n📐 Grid Embedding {dim}D Demonstration")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Input Channels: {in_channels}")
    print(f"   Dimensions: {dim}")
    print(f"   Grid Boundaries: {grid_boundaries}")

    # Create N-dimensional grid embedding
    embedding = GridEmbeddingND(
        in_channels=in_channels, dim=dim, grid_boundaries=grid_boundaries
    )

    # Generate sample input data
    rng_key = jax.random.PRNGKey(43)
    sample_input = jax.random.normal(rng_key, (batch_size, *spatial_shape, in_channels))

    # Apply embedding
    start_time = time.time()
    embedded_data = embedding(sample_input)
    embedding_time = time.time() - start_time

    # Get coordinate grids
    coordinate_grids = embedding.get_grid(spatial_shape)

    # Gather embedding information
    embedding_info = {
        "input_shape": sample_input.shape,
        "output_shape": embedded_data.shape,
        "in_channels": embedding.in_channels,
        "out_channels": embedding.out_channels,
        "coordinate_channels": dim,
        "dimensions": dim,
        "grid_boundaries": grid_boundaries,
        "embedding_time_ms": embedding_time * 1000,
        "memory_increase_factor": embedded_data.size / sample_input.size,
    }

    print(f"   ✅ Input Shape: {sample_input.shape}")
    print(f"   ✅ Output Shape: {embedded_data.shape}")
    print(f"   ✅ Output Channels: {embedding.out_channels}")
    print(f"   ✅ Coordinate Channels: {dim}")
    print(f"   ✅ Embedding Time: {embedding_time * 1000:.2f} ms")

    return {
        "embedded_data": embedded_data,
        "coordinate_grids": coordinate_grids,
        "embedding_info": embedding_info,
        "sample_input": sample_input,
        "embedding": embedding,
    }


def demonstrate_sinusoidal_embedding(
    spatial_shape: tuple[int, int] = (32, 32),
    batch_size: int = 4,
    in_channels: int = 3,
    num_frequencies: int = 8,
) -> dict[str, Any]:
    """Demonstrate sinusoidal embedding functionality.

    Args:
        spatial_shape: Spatial dimensions (height, width)
        batch_size: Batch size for demonstration
        in_channels: Number of input channels
        num_frequencies: Number of frequency components for embedding

    Returns:
        Dictionary containing embedded data, embedding info, and frequency analysis
    """
    print("\n🌊 Sinusoidal Embedding Demonstration")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Input Channels: {in_channels}")
    print(f"   Frequencies: {num_frequencies}")

    # Create sinusoidal embedding
    embedding = SinusoidalEmbedding(
        in_channels=in_channels,
        num_frequencies=num_frequencies,
        embedding_type="transformer",
    )

    # Generate sample input data - SinusoidalEmbedding expects (batch, n_points, channels)
    rng_key = jax.random.PRNGKey(44)
    h, w = spatial_shape
    n_points = h * w

    # Create coordinate data for sinusoidal embedding
    x_coords = jnp.linspace(0, 1, w)
    y_coords = jnp.linspace(0, 1, h)
    x_grid, y_grid = jnp.meshgrid(x_coords, y_coords, indexing="xy")

    # Flatten spatial coordinates to (n_points, 2)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    coord_flat = jnp.stack([x_flat, y_flat], axis=-1)  # (n_points, 2)

    # Expand for batch dimension: (batch, n_points, 2)
    coord_batched = jnp.repeat(coord_flat[None, :, :], batch_size, axis=0)

    # Add additional input channels if needed
    if in_channels > 2:
        extra_channels = jax.random.normal(
            rng_key, (batch_size, n_points, in_channels - 2)
        )
        sample_input = jnp.concatenate([coord_batched, extra_channels], axis=-1)
    else:
        sample_input = coord_batched[:, :, :in_channels]

    # Apply embedding
    start_time = time.time()
    embedded_data = embedding(sample_input)
    embedding_time = time.time() - start_time

    # Reshape embedded data back to spatial format for consistency
    embedded_spatial = embedded_data.reshape(batch_size, h, w, -1)

    # Analyze frequency content
    frequencies = jnp.arange(num_frequencies)

    # Extract sin and cos components correctly
    # The embedding interleaves sin and cos: [sin0, cos0, sin1, cos1, ...]
    # For multiple input channels: [ch0_sin0, ch0_cos0, ch0_sin1, ch0_cos1, ch1_sin0, ch1_cos0, ...]
    _ = embedded_spatial.shape[-1]
    frequency_analysis = {
        "frequencies": frequencies,
        "embedding_patterns": {
            "sin_components": embedded_spatial[:, :, :, ::2],  # Every even index (sin)
            "cos_components": embedded_spatial[:, :, :, 1::2],  # Every odd index (cos)
        },
        "frequency_distribution": {
            "low_freq": jnp.sum(frequencies < num_frequencies // 3),
            "mid_freq": jnp.sum(
                (frequencies >= num_frequencies // 3)
                & (frequencies < 2 * num_frequencies // 3)
            ),
            "high_freq": jnp.sum(frequencies >= 2 * num_frequencies // 3),
        },
    }

    # Gather embedding information
    embedding_info = {
        "input_shape": sample_input.shape,
        "output_shape": embedded_data.shape,
        "in_channels": embedding.in_channels,
        "out_channels": embedding.out_channels,
        "num_frequencies": num_frequencies,
        "embedding_type": "transformer",
        "embedding_time_ms": embedding_time * 1000,
        "memory_increase_factor": embedded_data.size / sample_input.size,
    }

    print(f"   ✅ Input Shape: {sample_input.shape}")
    print(f"   ✅ Output Shape: {embedded_data.shape}")
    print(f"   ✅ Output Channels: {embedding.out_channels}")
    print(f"   ✅ Frequency Components: {num_frequencies}")
    print(f"   ✅ Embedding Time: {embedding_time * 1000:.2f} ms")

    return {
        "embedded_data": embedded_spatial,
        "embedding_info": embedding_info,
        "frequency_analysis": frequency_analysis,
        "sample_input": sample_input,
        "embedding": embedding,
    }


def compare_embedding_methods(
    spatial_shape: tuple[int, int] = (24, 24),
    batch_size: int = 2,
    in_channels: int = 2,
) -> dict[str, Any]:
    """Compare different embedding methods side by side.

    Args:
        spatial_shape: Spatial dimensions for comparison
        batch_size: Batch size for comparison
        in_channels: Number of input channels

    Returns:
        Dictionary containing comparison results for all embedding methods
    """
    print("\n🔬 Embedding Methods Comparison")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Input Channels: {in_channels}")

    # Grid Embedding 2D
    grid_2d_result = demonstrate_grid_embedding_2d(
        spatial_shape=spatial_shape, batch_size=batch_size, in_channels=in_channels
    )

    # Grid Embedding ND (2D case for comparison)
    grid_nd_result = demonstrate_grid_embedding_nd(
        spatial_shape=spatial_shape,
        batch_size=batch_size,
        in_channels=in_channels,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
    )

    # Sinusoidal Embedding
    sinusoidal_result = demonstrate_sinusoidal_embedding(
        spatial_shape=spatial_shape,
        batch_size=batch_size,
        in_channels=in_channels,
        num_frequencies=8,
    )

    # Create comparison metrics
    methods = {
        "grid_2d": grid_2d_result,
        "grid_nd": grid_nd_result,
        "sinusoidal": sinusoidal_result,
    }

    comparison_metrics = {
        "channel_comparison": {
            method: result["embedding_info"]["out_channels"]
            for method, result in methods.items()
        },
        "performance_comparison": {
            method: result["embedding_info"]["embedding_time_ms"]
            for method, result in methods.items()
        },
        "memory_comparison": {
            method: result["embedding_info"]["memory_increase_factor"]
            for method, result in methods.items()
        },
        "use_case_recommendations": {
            "grid_2d": "Best for regular 2D grids, simple coordinate injection",
            "grid_nd": "Best for general N-dimensional problems, flexible boundaries",
            "sinusoidal": "Best for positional encoding, frequency-aware problems",
        },
    }

    print("\n📊 Comparison Summary:")
    print(
        f"   Grid 2D: {comparison_metrics['channel_comparison']['grid_2d']} channels, "
        f"{comparison_metrics['performance_comparison']['grid_2d']:.2f} ms"
    )
    print(
        f"   Grid ND: {comparison_metrics['channel_comparison']['grid_nd']} channels, "
        f"{comparison_metrics['performance_comparison']['grid_nd']:.2f} ms"
    )
    print(
        f"   Sinusoidal: {comparison_metrics['channel_comparison']['sinusoidal']} channels, "
        f"{comparison_metrics['performance_comparison']['sinusoidal']:.2f} ms"
    )

    # Structure output for testing
    comparison_output = {}
    for method, result in methods.items():
        comparison_output[method] = {
            "embedded_data": result["embedded_data"],
            "out_channels": result["embedding_info"]["out_channels"],
            "computational_info": {
                "embedding_time_ms": result["embedding_info"]["embedding_time_ms"],
                "memory_factor": result["embedding_info"]["memory_increase_factor"],
            },
        }

    comparison_output["comparison_metrics"] = comparison_metrics

    return comparison_output


def visualize_coordinate_grids(
    spatial_shape: tuple[int, int] = (16, 16),
    grid_boundaries: list[list[float]] | None = None,
    save_path: str | None = None,
) -> None:
    """Visualize coordinate grids and embedding effects.

    Args:
        spatial_shape: Spatial dimensions for visualization
        grid_boundaries: Coordinate boundaries for grid generation
        save_path: Path to save visualization (if None, display only)
    """
    if grid_boundaries is None:
        grid_boundaries = [[-1.0, 1.0], [-1.0, 1.0]]

    print("\n📈 Coordinate Grid Visualization")
    print(f"   Spatial Shape: {spatial_shape}")
    print(f"   Grid Boundaries: {grid_boundaries}")

    # Generate coordinate grids
    x_grid, y_grid = regular_grid_2d(spatial_shape, grid_boundaries)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Grid Embedding Coordinate Visualization", fontsize=16)

    # X coordinates
    im1 = axes[0, 0].imshow(x_grid, cmap="viridis", aspect="equal")
    axes[0, 0].set_title("X Coordinates")
    axes[0, 0].set_xlabel("Width")
    axes[0, 0].set_ylabel("Height")
    plt.colorbar(im1, ax=axes[0, 0])

    # Y coordinates
    im2 = axes[0, 1].imshow(y_grid, cmap="plasma", aspect="equal")
    axes[0, 1].set_title("Y Coordinates")
    axes[0, 1].set_xlabel("Width")
    axes[0, 1].set_ylabel("Height")
    plt.colorbar(im2, ax=axes[0, 1])

    # Combined coordinate magnitude
    coord_magnitude = jnp.sqrt(x_grid**2 + y_grid**2)
    im3 = axes[1, 0].imshow(coord_magnitude, cmap="coolwarm", aspect="equal")
    axes[1, 0].set_title("Coordinate Magnitude")
    axes[1, 0].set_xlabel("Width")
    axes[1, 0].set_ylabel("Height")
    plt.colorbar(im3, ax=axes[1, 0])

    # Coordinate contours
    axes[1, 1].contour(x_grid, levels=10, colors="blue", alpha=0.6, linewidths=1)
    axes[1, 1].contour(y_grid, levels=10, colors="red", alpha=0.6, linewidths=1)
    axes[1, 1].set_title("Coordinate Contours")
    axes[1, 1].set_xlabel("Width")
    axes[1, 1].set_ylabel("Height")
    axes[1, 1].legend(["X contours", "Y contours"])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Visualization saved to: {save_path}")
    else:
        print("   ✅ Visualization generated (not saved)")

    # Don't show during testing
    if save_path is not None:
        plt.close()


def _setup_demo_environment(
    save_outputs: bool, verbose: bool
) -> tuple[Path | None, str]:
    """Set up the demonstration environment and output directory."""
    if verbose:
        print("=" * 70)
        print("🔲 Opifex Grid Embeddings Layer Example")
        print("=" * 70)
        print("Reproducing neuraloperator plot_embeddings.py with Opifex framework")
        print(f"Timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

    output_dir: Path | None = None
    if save_outputs:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"examples_output/grid_embeddings_demo_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Output Directory: {output_dir}")

    return output_dir, "setup_complete"


def _run_embedding_demonstrations(verbose: bool) -> dict[str, Any]:
    """Run all embedding demonstrations."""
    demonstrations = {}

    # 1. Grid Embedding 2D Demo
    if verbose:
        print("\n" + "=" * 50)
        print("🔲 PHASE 1: Grid Embedding 2D")
        print("=" * 50)

    demonstrations["grid_2d_demo"] = demonstrate_grid_embedding_2d(
        spatial_shape=(64, 64),
        batch_size=8,
        in_channels=3,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
    )

    # 2. Grid Embedding ND Demo
    if verbose:
        print("\n" + "=" * 50)
        print("📐 PHASE 2: Grid Embedding N-Dimensional")
        print("=" * 50)

    demonstrations["grid_nd_demo"] = demonstrate_grid_embedding_nd(
        spatial_shape=(32, 32, 32),
        batch_size=4,
        in_channels=2,
        grid_boundaries=[[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
    )

    demonstrations["grid_1d_demo"] = demonstrate_grid_embedding_nd(
        spatial_shape=(128,), batch_size=6, in_channels=1, grid_boundaries=[[-2.0, 2.0]]
    )

    # 3. Sinusoidal Embedding Demo
    if verbose:
        print("\n" + "=" * 50)
        print("🌊 PHASE 3: Sinusoidal Embedding")
        print("=" * 50)

    demonstrations["sinusoidal_demo"] = demonstrate_sinusoidal_embedding(
        spatial_shape=(64, 64), batch_size=4, in_channels=2, num_frequencies=16
    )

    return demonstrations


def _create_demo_summary(
    demonstrations: dict, comparisons: dict, verbose: bool, output_dir: Path | None
) -> dict[str, Any]:
    """Create demonstration summary and print results."""
    embedding_methods_tested = ["Grid2D", "GridND", "Sinusoidal"]
    spatial_dimensions_tested = ["1D", "2D", "3D"]

    summary = {
        "total_demonstrations": len(demonstrations),
        "embedding_methods_tested": embedding_methods_tested,
        "spatial_dimensions_tested": spatial_dimensions_tested,
        "key_findings": {
            "grid_2d": "Efficient for regular 2D coordinate injection",
            "grid_nd": "Flexible for arbitrary dimensions",
            "sinusoidal": "Rich frequency content for positional encoding",
        },
        "performance_summary": {
            method: comparisons["comparison_metrics"]["performance_comparison"][method]
            for method in ["grid_2d", "grid_nd", "sinusoidal"]
        },
        "memory_summary": {
            method: comparisons["comparison_metrics"]["memory_comparison"][method]
            for method in ["grid_2d", "grid_nd", "sinusoidal"]
        },
    }

    if verbose:
        print("\n" + "=" * 70)
        print("📊 DEMONSTRATION SUMMARY")
        print("=" * 70)
        print(f"✅ Total Demonstrations: {summary['total_demonstrations']}")
        print(
            f"✅ Methods Tested: {', '.join(str(method) for method in summary['embedding_methods_tested'])}"
        )
        print(
            f"✅ Dimensions Tested: {', '.join(str(dim) for dim in summary['spatial_dimensions_tested'])}"
        )
        print("\n🚀 Grid Embeddings Demo Complete!")
        if output_dir:
            print(f"📁 Results saved to: {output_dir}")

    return summary


def run_grid_embeddings_demo(
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run complete grid embeddings demonstration.

    Args:
        save_outputs: Whether to save output files
        verbose: Whether to print detailed information

    Returns:
        Dictionary containing all demonstration results
    """
    # Setup environment
    output_dir, _ = _setup_demo_environment(save_outputs, verbose)

    # Run demonstrations
    demonstrations = _run_embedding_demonstrations(verbose)

    # Methods comparison
    if verbose:
        print("\n" + "=" * 50)
        print("🔬 PHASE 4: Methods Comparison")
        print("=" * 50)

    comparisons = compare_embedding_methods(
        spatial_shape=(48, 48), batch_size=3, in_channels=2
    )

    # Visualization
    if verbose:
        print("\n" + "=" * 50)
        print("📈 PHASE 5: Visualization")
        print("=" * 50)

    if save_outputs and output_dir is not None:
        viz_path = output_dir / "coordinate_grids_visualization.png"
        visualize_coordinate_grids(
            spatial_shape=(32, 32),
            grid_boundaries=[[-2.0, 2.0], [-2.0, 2.0]],
            save_path=str(viz_path),
        )
    else:
        visualize_coordinate_grids(save_path=None)

    # Create summary
    summary = _create_demo_summary(demonstrations, comparisons, verbose, output_dir)

    return {
        "demonstrations": demonstrations,
        "comparisons": comparisons,
        "summary": summary,
        "output_directory": str(output_dir) if save_outputs else None,
    }


if __name__ == "__main__":
    # Run the complete demonstration
    results = run_grid_embeddings_demo(save_outputs=True, verbose=True)

    print("\n🎉 Grid Embeddings Example completed successfully!")
    print(f"📁 Check output directory: {results['output_directory']}")
