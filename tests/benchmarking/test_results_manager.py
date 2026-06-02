"""Tests for ResultsManager publication plot generators.

Covers the scaling and convergence plot generators exposed through
``ResultsManager.export_publication_plots``. Each generator must render a real
matplotlib figure and write a non-empty image file (no silent no-ops).
"""

import tempfile

import matplotlib as mpl
from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric


mpl.use("Agg")


def _make_result(
    name: str,
    dataset: str,
    metrics: dict[str, float],
    execution_time: float = 1.0,
    problem_size: int | None = None,
    loss_history: list[float] | None = None,
) -> BenchmarkResult:
    """Build a BenchmarkResult with optional scaling/convergence metadata."""
    metadata: dict[str, object] = {"execution_time": execution_time}
    if problem_size is not None:
        metadata["problem_size"] = problem_size
    if loss_history is not None:
        metadata["loss_history"] = loss_history
    return BenchmarkResult(
        name=name,
        tags={"dataset": dataset},
        metrics={k: Metric(value=v) for k, v in metrics.items()},
        metadata=metadata,
    )


class TestScalingPlots:
    """Test scaling behaviour plot generation."""

    def test_scaling_plot_written_and_non_empty(self):
        """A model evaluated at several problem sizes yields a non-empty PNG."""
        from opifex.benchmarking.results_manager import ResultsManager

        sizes = [64, 128, 256, 512]
        results = [
            _make_result(
                "FNO",
                "darcy",
                {"mse": 1.0 / size},
                execution_time=float(size) / 64.0,
                problem_size=size,
            )
            for size in sizes
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(storage_path=tmpdir)
            files = manager.export_publication_plots(results, plot_type="scaling")

            assert files, "scaling generator produced no files"
            for plot_file in files:
                assert plot_file.exists()
                assert plot_file.stat().st_size > 0

    def test_scaling_plot_skips_without_problem_sizes(self):
        """Results lacking problem sizes produce no scaling plots (not a crash)."""
        from opifex.benchmarking.results_manager import ResultsManager

        results = [
            _make_result("FNO", "darcy", {"mse": 0.1}),
            _make_result("FNO", "darcy", {"mse": 0.2}),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(storage_path=tmpdir)
            files = manager.export_publication_plots(results, plot_type="scaling")

            assert files == []


class TestConvergencePlots:
    """Test convergence analysis plot generation."""

    def test_convergence_plot_written_and_non_empty(self):
        """A result carrying a loss history yields a non-empty convergence PNG."""
        from opifex.benchmarking.results_manager import ResultsManager

        history = [float(10.0 / (epoch + 1)) for epoch in range(20)]
        results = [
            _make_result("FNO", "darcy", {"mse": 0.05}, loss_history=history),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(storage_path=tmpdir)
            files = manager.export_publication_plots(results, plot_type="convergence")

            assert files, "convergence generator produced no files"
            for plot_file in files:
                assert plot_file.exists()
                assert plot_file.stat().st_size > 0

    def test_convergence_plot_skips_without_history(self):
        """Results lacking a loss history produce no convergence plots."""
        from opifex.benchmarking.results_manager import ResultsManager

        results = [_make_result("FNO", "darcy", {"mse": 0.05})]

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ResultsManager(storage_path=tmpdir)
            files = manager.export_publication_plots(results, plot_type="convergence")

            assert files == []
