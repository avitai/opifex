"""Tests for field plotting visualization functionality.

This module tests the field_plotting module, covering:
- 2D field plotting with various inputs
- 3D input handling (channel extraction)
- Invalid dimension rejection
- Custom colormap/vmin/vmax
- Field comparison (3-panel with error metrics)
- Spectral analysis (FFT, radial spectrum, energy bands)
- Field evolution snapshot selection
"""

import jax.numpy as jnp
import matplotlib as mpl
import pytest


# Use non-interactive backend for testing
mpl.use("Agg")

from opifex.visualization.field_plotting import (
    plot_2d_field,
    plot_field_comparison,
    plot_field_evolution,
    plot_spectral_analysis,
)


class TestPlot2DField:
    """Test plot_2d_field function."""

    def test_basic_2d_field(self):
        """Test basic 2D field plotting."""
        field = jnp.ones((16, 16))
        fig = plot_2d_field(field)

        assert fig is not None
        # Clean up
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_2d_field_with_custom_title(self):
        """Test 2D field with custom title."""
        field = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 64).reshape(8, 8))
        fig = plot_2d_field(field, title="Test Field")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_2d_field_with_custom_labels(self):
        """Test 2D field with custom axis labels."""
        field = jnp.ones((16, 16))
        fig = plot_2d_field(field, xlabel="Custom X", ylabel="Custom Y")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_2d_field_with_custom_colormap(self):
        """Test 2D field with custom colormap."""
        field = jnp.ones((16, 16))
        fig = plot_2d_field(field, colormap="plasma")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_2d_field_with_vmin_vmax(self):
        """Test 2D field with explicit vmin/vmax."""
        field = jnp.ones((16, 16)) * 0.5
        fig = plot_2d_field(field, vmin=0.0, vmax=1.0)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_3d_field_extracts_first_channel(self):
        """Test that 3D input extracts first channel."""
        # 3D array with channels
        field = jnp.ones((16, 16, 3))
        fig = plot_2d_field(field)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_invalid_dimension_raises_error(self):
        """Test that invalid dimension raises ValueError."""
        field = jnp.ones((16,))  # 1D array

        with pytest.raises(ValueError, match="2D or 3D"):
            plot_2d_field(field)

    def test_4d_field_raises_error(self):
        """Test that 4D array raises ValueError."""
        field = jnp.ones((2, 16, 16, 3))  # 4D array

        with pytest.raises(ValueError, match="2D or 3D"):
            plot_2d_field(field)

    def test_save_path(self, temp_directory):
        """Test saving plot to file."""
        field = jnp.ones((16, 16))
        save_path = str(temp_directory / "test_field.png")

        fig = plot_2d_field(field, save_path=save_path)

        assert (temp_directory / "test_field.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotFieldComparison:
    """Test plot_field_comparison function."""

    def test_basic_comparison(self):
        """Test basic field comparison plotting."""
        ground_truth = jnp.ones((16, 16))
        prediction = jnp.ones((16, 16)) * 1.1

        fig = plot_field_comparison(ground_truth, prediction)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_comparison_with_custom_titles(self):
        """Test comparison with custom subplot titles."""
        ground_truth = jnp.ones((16, 16))
        prediction = jnp.ones((16, 16))

        fig = plot_field_comparison(
            ground_truth, prediction, titles=["GT", "Pred", "Error"]
        )

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_comparison_with_3d_inputs(self):
        """Test comparison handles 3D inputs."""
        ground_truth = jnp.ones((16, 16, 3))
        prediction = jnp.ones((16, 16, 3)) * 1.1

        fig = plot_field_comparison(ground_truth, prediction)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_comparison_calculates_error_metrics(self):
        """Test that comparison calculates MSE, MAE, max_error."""
        ground_truth = jnp.zeros((16, 16))
        prediction = jnp.ones((16, 16)) * 0.1

        fig = plot_field_comparison(ground_truth, prediction)

        # The suptitle should contain MSE, MAE, Max Error
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""  # pyright: ignore[reportAttributeAccessIssue]
        assert "MSE" in suptitle
        assert "MAE" in suptitle
        assert "Max Error" in suptitle
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_comparison_save_path(self, temp_directory):
        """Test saving comparison plot to file."""
        ground_truth = jnp.ones((16, 16))
        prediction = jnp.ones((16, 16))
        save_path = str(temp_directory / "comparison.png")

        fig = plot_field_comparison(ground_truth, prediction, save_path=save_path)

        assert (temp_directory / "comparison.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotSpectralAnalysis:
    """Test plot_spectral_analysis function."""

    def test_basic_spectral_analysis(self):
        """Test basic spectral analysis plotting."""
        # Create a simple periodic signal
        x = jnp.linspace(0, 4 * jnp.pi, 64)
        field = jnp.sin(x.reshape(8, 8))

        fig = plot_spectral_analysis(field)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_spectral_analysis_log_scale(self):
        """Test spectral analysis with log scale."""
        field = jnp.ones((16, 16))
        fig = plot_spectral_analysis(field, log_scale=True)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_spectral_analysis_linear_scale(self):
        """Test spectral analysis with linear scale."""
        field = jnp.ones((16, 16))
        fig = plot_spectral_analysis(field, log_scale=False)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_spectral_analysis_with_3d_input(self):
        """Test spectral analysis with 3D input (extracts first channel)."""
        field = jnp.ones((16, 16, 3))
        fig = plot_spectral_analysis(field)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_spectral_analysis_save_path(self, temp_directory):
        """Test saving spectral analysis to file."""
        field = jnp.ones((16, 16))
        save_path = str(temp_directory / "spectral.png")

        fig = plot_spectral_analysis(field, save_path=save_path)

        assert (temp_directory / "spectral.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotFieldEvolution:
    """Test plot_field_evolution function."""

    def test_basic_field_evolution(self):
        """Test basic field evolution plotting."""
        # Create sequence of fields
        field_sequence = jnp.ones((10, 16, 16))

        fig = plot_field_evolution(field_sequence)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_field_evolution_with_n_snapshots(self):
        """Test field evolution with custom number of snapshots."""
        field_sequence = jnp.ones((20, 16, 16))

        fig = plot_field_evolution(field_sequence, n_snapshots=4)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_field_evolution_with_time_points(self):
        """Test field evolution with time points."""
        field_sequence = jnp.ones((10, 16, 16))
        time_points = jnp.linspace(0, 1, 10)

        fig = plot_field_evolution(field_sequence, time_points=time_points)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_field_evolution_with_4d_input(self):
        """Test field evolution with 4D input (channels)."""
        field_sequence = jnp.ones((10, 16, 16, 3))

        fig = plot_field_evolution(field_sequence)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_field_evolution_save_path(self, temp_directory):
        """Test saving field evolution to file."""
        field_sequence = jnp.ones((10, 16, 16))
        save_path = str(temp_directory / "evolution.png")

        fig = plot_field_evolution(field_sequence, save_path=save_path)

        assert (temp_directory / "evolution.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestErrorMetricsInComparison:
    """Test error metrics calculation in field comparison."""

    def test_perfect_prediction_zero_errors(self):
        """Test that identical fields have zero errors."""
        field = jnp.ones((16, 16)) * 0.5
        fig = plot_field_comparison(field, field)

        # Check suptitle contains zero or very small MSE
        suptitle = fig._suptitle.get_text() if fig._suptitle else ""  # pyright: ignore[reportAttributeAccessIssue]
        # MSE should be 0.000000
        assert "MSE: 0" in suptitle
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_known_error_values(self):
        """Test error calculation with known values."""
        ground_truth = jnp.zeros((4, 4))
        prediction = jnp.ones((4, 4)) * 0.5

        # Expected: MSE = 0.25, MAE = 0.5, Max Error = 0.5
        fig = plot_field_comparison(ground_truth, prediction)

        suptitle = fig._suptitle.get_text() if fig._suptitle else ""  # pyright: ignore[reportAttributeAccessIssue]
        assert "MSE: 0.25" in suptitle
        assert "MAE: 0.5" in suptitle
        import matplotlib.pyplot as plt

        plt.close(fig)
