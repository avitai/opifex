"""Tests for performance visualization functionality.

This module tests the performance visualization module, covering:
- FLOPS analysis charts
- Memory usage visualization
- Model complexity comparison
- Missing data handling
"""

import matplotlib as mpl


# Use non-interactive backend for testing
mpl.use("Agg")

from opifex.visualization.performance import (
    plot_flops_analysis,
)


class TestPlotFlopsAnalysis:
    """Test plot_flops_analysis function."""

    def test_single_model_flops(self):
        """Test FLOPS analysis with single model results."""
        flops_results = {
            "total_flops": 1e9,
            "forward_flops": 6e8,
            "backward_flops": 4e8,
            "forward_time": 0.001,
            "backward_time": 0.002,
        }

        fig = plot_flops_analysis(flops_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_model_forward_only(self):
        """Test FLOPS analysis with forward pass only."""
        flops_results = {
            "total_flops": 1e9,
            "forward_flops": 1e9,
            "forward_time": 0.001,
        }

        fig = plot_flops_analysis(flops_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_flops_with_custom_title(self):
        """Test FLOPS analysis with custom title."""
        flops_results = {"total_flops": 1e9}

        fig = plot_flops_analysis(flops_results, title="Custom FLOPS Analysis")

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_flops_save_path(self, temp_directory):
        """Test saving FLOPS analysis to file."""
        flops_results = {"total_flops": 1e9}
        save_path = str(temp_directory / "flops.png")

        fig = plot_flops_analysis(flops_results, save_path=save_path)

        assert (temp_directory / "flops.png").exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_multi_model_flops(self):
        """Test FLOPS analysis with multiple model results."""
        flops_results = {
            "FNO": {"total_flops": 1e9, "forward_time": 0.001},
            "DeepONet": {"total_flops": 2e9, "forward_time": 0.002},
        }

        fig = plot_flops_analysis(flops_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_empty_results(self):
        """Test handling of empty results."""
        flops_results = {}

        fig = plot_flops_analysis(flops_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_missing_data_handling(self):
        """Test handling of missing data fields."""
        # Results with some missing fields
        flops_results = {"total_flops": 1e9}  # Missing forward_flops, backward_flops

        fig = plot_flops_analysis(flops_results)

        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)
