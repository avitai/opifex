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
    plot_memory_usage,
    plot_model_complexity_comparison,
)


class TestPlotMemoryUsage:
    """Smoke tests pinning the four-panel memory-usage figure."""

    def test_empty_results_returns_four_axis_figure(self):
        """No data renders the four no-data panels and returns a Figure."""
        import matplotlib.pyplot as plt

        fig = plot_memory_usage({})
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_with_minimal_data_returns_figure(self):
        """Populated results render without error and keep four panels."""
        import matplotlib.pyplot as plt

        results = {
            "efficiency_analysis": {
                "memory_breakdown": {"weights_mb": 10.0, "activations_mb": 5.0},
                "efficiency_category": "efficient",
                "total_memory_mb": 15.0,
            },
            "optimization_suggestions": ["Reduce batch size to lower memory"],
        }
        fig = plot_memory_usage(results)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)


class TestPlotModelComplexityComparison:
    """Smoke tests pinning the six-panel model-complexity figure."""

    def test_empty_results_returns_six_axis_figure(self):
        """No model data renders the six no-data panels and returns a Figure."""
        import matplotlib.pyplot as plt

        fig = plot_model_complexity_comparison({})
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 6
        plt.close(fig)

    def test_with_models_returns_figure(self):
        """Two models render every panel (scatter colorbar adds an axis)."""
        import matplotlib.pyplot as plt

        results = {
            "model_a": {
                "parameters": {"total_parameters": 1000},
                "memory": {"total_estimated_mb": 4.0},
                "computational": {"total_estimated_operations": 5000},
                "model_type": "fno",
            },
            "model_b": {
                "parameters": {"total_parameters": 2000},
                "memory": {"total_estimated_mb": 8.0},
                "computational": {"total_estimated_operations": 9000},
                "model_type": "deeponet",
            },
        }
        fig = plot_model_complexity_comparison(results)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 6
        plt.close(fig)


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
