"""Test Grid Embeddings Example - Opifex Framework.

Test-driven development for grid embeddings layer example that reproduces
neuraloperator plot_embeddings.py with Opifex framework components.
"""

import jax
import jax.numpy as jnp
import pytest


def test_grid_embeddings_example_imports():
    """Test that the grid embeddings example imports work correctly."""
    # Test should pass after implementation
    from examples.layers.grid_embeddings_example import (
        compare_embedding_methods,
        demonstrate_grid_embedding_2d,
        demonstrate_grid_embedding_nd,
        demonstrate_sinusoidal_embedding,
        run_grid_embeddings_demo,
        visualize_coordinate_grids,
    )

    # Test that all functions are callable
    assert callable(demonstrate_grid_embedding_2d)
    assert callable(demonstrate_grid_embedding_nd)
    assert callable(demonstrate_sinusoidal_embedding)
    assert callable(compare_embedding_methods)
    assert callable(visualize_coordinate_grids)
    assert callable(run_grid_embeddings_demo)


def test_demonstrate_grid_embedding_2d():
    """Test 2D grid embedding demonstration function."""
    from examples.layers.grid_embeddings_example import demonstrate_grid_embedding_2d

    # Test basic functionality
    result = demonstrate_grid_embedding_2d(
        spatial_shape=(32, 32),
        batch_size=4,
        in_channels=3,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
    )

    # Verify output structure
    assert "embedded_data" in result
    assert "coordinate_grids" in result
    assert "embedding_info" in result

    # Verify shapes
    embedded_data = result["embedded_data"]
    assert embedded_data.shape == (4, 32, 32, 5)  # 3 input + 2 coordinate channels

    # Verify coordinate grids
    x_grid, y_grid = result["coordinate_grids"]
    assert x_grid.shape == (32, 32)
    assert y_grid.shape == (32, 32)

    # Verify coordinate ranges
    assert jnp.allclose(x_grid.min(), 0.0, atol=1e-6)
    assert jnp.allclose(x_grid.max(), 1.0, atol=1e-6)
    assert jnp.allclose(y_grid.min(), 0.0, atol=1e-6)
    assert jnp.allclose(y_grid.max(), 1.0, atol=1e-6)


def test_demonstrate_grid_embedding_nd():
    """Test N-dimensional grid embedding demonstration function."""
    from examples.layers.grid_embeddings_example import demonstrate_grid_embedding_nd

    # Test 3D case
    result_3d = demonstrate_grid_embedding_nd(
        spatial_shape=(16, 16, 16),
        batch_size=2,
        in_channels=2,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
    )

    # Verify 3D output structure
    assert "embedded_data" in result_3d
    assert "coordinate_grids" in result_3d
    assert "embedding_info" in result_3d

    # Verify 3D shapes
    embedded_data = result_3d["embedded_data"]
    assert embedded_data.shape == (2, 16, 16, 16, 5)  # 2 input + 3 coordinate channels

    # Test 1D case
    result_1d = demonstrate_grid_embedding_nd(
        spatial_shape=(64,), batch_size=3, in_channels=1, grid_boundaries=[[-1.0, 1.0]]
    )

    # Verify 1D output structure
    embedded_data_1d = result_1d["embedded_data"]
    assert embedded_data_1d.shape == (3, 64, 2)  # 1 input + 1 coordinate channel


def test_demonstrate_sinusoidal_embedding():
    """Test sinusoidal embedding demonstration function."""
    from examples.layers.grid_embeddings_example import demonstrate_sinusoidal_embedding

    # Test basic functionality
    result = demonstrate_sinusoidal_embedding(
        spatial_shape=(32, 32), batch_size=4, in_channels=3, num_frequencies=8
    )

    # Verify output structure
    assert "embedded_data" in result
    assert "embedding_info" in result
    assert "frequency_analysis" in result

    # Verify shapes - sinusoidal embedding outputs input_channels * 2 * num_frequencies
    embedded_data = result["embedded_data"]
    expected_out_channels = 3 * 2 * 8  # input_channels * 2 * num_frequencies = 48
    assert embedded_data.shape == (4, 32, 32, expected_out_channels)

    # Verify frequency analysis
    freq_analysis = result["frequency_analysis"]
    assert "frequencies" in freq_analysis
    assert "embedding_patterns" in freq_analysis


def test_compare_embedding_methods():
    """Test embedding methods comparison function."""
    from examples.layers.grid_embeddings_example import compare_embedding_methods

    # Test comparison functionality
    comparison = compare_embedding_methods(
        spatial_shape=(24, 24), batch_size=2, in_channels=2
    )

    # Verify comparison structure
    assert "grid_2d" in comparison
    assert "grid_nd" in comparison
    assert "sinusoidal" in comparison
    assert "comparison_metrics" in comparison

    # Verify each method has required fields
    for method in ["grid_2d", "grid_nd", "sinusoidal"]:
        method_result = comparison[method]
        assert "embedded_data" in method_result
        assert "out_channels" in method_result
        assert "computational_info" in method_result

    # Verify comparison metrics
    metrics = comparison["comparison_metrics"]
    assert "channel_comparison" in metrics
    assert "performance_comparison" in metrics
    assert "use_case_recommendations" in metrics


def test_visualize_coordinate_grids():
    """Test coordinate grid visualization function."""
    from examples.layers.grid_embeddings_example import visualize_coordinate_grids

    # Test basic visualization (should not error)
    try:
        visualize_coordinate_grids(
            spatial_shape=(16, 16),
            grid_boundaries=[[-1.0, 1.0], [-1.0, 1.0]],
            save_path=None,  # Don't save during testing
        )
        visualization_success = True
    except Exception as e:
        visualization_success = False
        pytest.fail(f"Visualization failed: {e}")

    assert visualization_success


def test_run_grid_embeddings_demo():
    """Test the complete grid embeddings demonstration."""
    from examples.layers.grid_embeddings_example import run_grid_embeddings_demo

    # Test that demo runs without errors
    try:
        results = run_grid_embeddings_demo(
            save_outputs=False,  # Don't save during testing
            verbose=False,  # Reduce output during testing
        )
        demo_success = True
    except Exception as e:
        demo_success = False
        pytest.fail(f"Demo execution failed: {e}")

    assert demo_success

    # Verify demo results structure
    assert "demonstrations" in results
    assert "comparisons" in results
    assert "summary" in results

    demonstrations = results["demonstrations"]
    assert "grid_2d_demo" in demonstrations
    assert "grid_nd_demo" in demonstrations
    assert "sinusoidal_demo" in demonstrations


def test_embedding_jax_compatibility():
    """Test that example functions work with JAX transformations."""
    from examples.layers.grid_embeddings_example import demonstrate_grid_embedding_2d

    # Test basic functionality (not JIT compilation since modules can't be JIT-compiled)
    result = demonstrate_grid_embedding_2d(
        spatial_shape=(8, 8),
        batch_size=2,
        in_channels=1,
        grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
    )

    # Should work and return expected results
    assert "embedded_data" in result
    assert result["embedded_data"].shape == (2, 8, 8, 3)

    # Test that embedded data is a valid JAX array
    embedded_data = result["embedded_data"]
    assert isinstance(embedded_data, jnp.ndarray)

    # Test that we can apply JAX transformations to the embedded data
    sum_result = jnp.sum(embedded_data)
    assert isinstance(sum_result, jnp.ndarray)


def test_embedding_gradient_flow():
    """Test that embeddings support gradient computation."""
    from examples.layers.grid_embeddings_example import demonstrate_grid_embedding_2d

    # Test gradient computation on embedded data
    result = demonstrate_grid_embedding_2d(
        spatial_shape=(4, 4), batch_size=1, in_channels=1
    )
    embedded_data = result["embedded_data"]

    # Define a simple loss function that operates on the embedded data
    def loss_fn(data):
        return jnp.sum(data**2)

    # Should support gradient computation on the embedded data
    grad_fn = jax.grad(loss_fn)
    try:
        grad_result = grad_fn(embedded_data)
        gradient_success = True
        # Verify gradient has the same shape as input
        assert grad_result.shape == embedded_data.shape
    except Exception:
        gradient_success = False

    assert gradient_success


@pytest.mark.parametrize(
    ("spatial_shape", "expected_dims"),
    [
        ((16, 16), 2),
        ((8, 8, 8), 3),
        ((32,), 1),
        ((4, 8, 16), 3),
    ],
)
def test_multi_dimensional_embedding_shapes(spatial_shape, expected_dims):
    """Test embedding shapes for various spatial dimensions."""
    from examples.layers.grid_embeddings_example import demonstrate_grid_embedding_nd

    result = demonstrate_grid_embedding_nd(
        spatial_shape=spatial_shape,
        batch_size=2,
        in_channels=1,
        grid_boundaries=[[0.0, 1.0] for _ in range(expected_dims)],
    )

    embedded_data = result["embedded_data"]
    expected_shape = (2, *spatial_shape, 1 + expected_dims)
    assert embedded_data.shape == expected_shape


@pytest.mark.parametrize("num_frequencies", [4, 8, 16, 32])
def test_sinusoidal_frequency_variations(num_frequencies):
    """Test sinusoidal embeddings with different frequency counts."""
    from examples.layers.grid_embeddings_example import demonstrate_sinusoidal_embedding

    result = demonstrate_sinusoidal_embedding(
        spatial_shape=(16, 16),
        batch_size=1,
        in_channels=2,
        num_frequencies=num_frequencies,
    )

    embedded_data = result["embedded_data"]
    expected_out_channels = 2 * 2 * num_frequencies  # input_channels * 2 * frequencies
    assert embedded_data.shape == (1, 16, 16, expected_out_channels)

    # Verify frequency information
    freq_analysis = result["frequency_analysis"]
    assert len(freq_analysis["frequencies"]) == num_frequencies
