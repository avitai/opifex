"""Tests for BaselineRepository with calibrax Store integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from opifex.benchmarking.baseline_repository import BaselineRepository


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def repo(tmp_path: Path) -> BaselineRepository:
    """Create a BaselineRepository with a temporary data path."""
    return BaselineRepository(
        baseline_data_path=str(tmp_path / "baselines.json"),
        store_path=tmp_path / "store",
    )


class TestDefaultBaselines:
    """Tests for default baseline data."""

    def test_has_default_datasets(self, repo: BaselineRepository) -> None:
        """Repository ships with default datasets."""
        datasets = repo.get_available_datasets()
        assert "advection" in datasets
        assert "burgers" in datasets
        assert "darcy_flow" in datasets

    def test_has_default_model_types(self, repo: BaselineRepository) -> None:
        """Default datasets have FNO and DeepONet baselines."""
        model_types = repo.get_available_model_types("darcy_flow")
        assert "fno" in model_types
        assert "deeponet" in model_types


class TestGetBaseline:
    """Tests for retrieving baseline metrics."""

    def test_get_baseline_metrics(self, repo: BaselineRepository) -> None:
        """Retrieve numeric metrics for a baseline."""
        metrics = repo.get_baseline_metrics("darcy_flow", "fno")
        assert "mse" in metrics
        assert "relative_error" in metrics
        assert isinstance(metrics["mse"], float)

    def test_metrics_exclude_metadata_fields(self, repo: BaselineRepository) -> None:
        """Returned metrics exclude non-numeric fields like source."""
        metrics = repo.get_baseline_metrics("darcy_flow", "fno")
        assert "source" not in metrics
        assert "model_config" not in metrics

    def test_unknown_dataset_raises(self, repo: BaselineRepository) -> None:
        """Raises ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="No baselines"):
            repo.get_baseline_metrics("nonexistent", "fno")

    def test_unknown_model_raises(self, repo: BaselineRepository) -> None:
        """Raises ValueError for unknown model type."""
        with pytest.raises(ValueError, match="No baselines"):
            repo.get_baseline_metrics("darcy_flow", "nonexistent")


class TestAddBaseline:
    """Tests for adding new baselines."""

    def test_add_and_retrieve(self, repo: BaselineRepository) -> None:
        """Add a baseline and retrieve it."""
        repo.add_baseline("navier_stokes", "fno", {"mse": 0.003, "mae": 0.02})
        metrics = repo.get_baseline_metrics("navier_stokes", "fno")
        assert metrics["mse"] == pytest.approx(0.003)

    def test_add_to_existing_dataset(self, repo: BaselineRepository) -> None:
        """Add a new model type to an existing dataset."""
        repo.add_baseline("darcy_flow", "tfno", {"mse": 0.0005})
        assert "tfno" in repo.get_available_model_types("darcy_flow")

    def test_add_with_metadata(self, repo: BaselineRepository) -> None:
        """Add baseline with source and config metadata."""
        repo.add_baseline(
            "darcy_flow",
            "tfno",
            {"mse": 0.0005},
            source="Custom",
            model_config={"modes": 16},
            notes="Factorized FNO",
        )
        # Verify metrics don't leak metadata
        metrics = repo.get_baseline_metrics("darcy_flow", "tfno")
        assert "source" not in metrics

    def test_add_baseline_persists_to_store(self, repo: BaselineRepository) -> None:
        """New baselines are saved to the calibrax Store."""
        repo.add_baseline("navier_stokes", "fno", {"mse": 0.003})
        # The store should contain runs
        runs = repo._store.list_runs()
        assert len(runs) >= 1


class TestCompareToBaseline:
    """Tests for comparing test metrics against baselines."""

    def test_compare_metrics(self, repo: BaselineRepository) -> None:
        """Compare returns absolute difference and relative improvement."""
        test_metrics = {"mse": 0.0005, "mae": 0.006}
        comparison = repo.compare_to_baseline("darcy_flow", "fno", test_metrics)
        assert "absolute_difference" in comparison
        assert "relative_improvement" in comparison
        assert "is_better" in comparison

    def test_lower_mse_is_better(self, repo: BaselineRepository) -> None:
        """Lower MSE is flagged as improvement."""
        # darcy_flow/fno has mse=0.0008
        test_metrics = {"mse": 0.0005}
        comparison = repo.compare_to_baseline("darcy_flow", "fno", test_metrics)
        assert comparison["is_better"]["mse"] is True
        assert comparison["relative_improvement"]["mse"] > 0

    def test_higher_r2_is_better(self, repo: BaselineRepository) -> None:
        """Higher R2 is flagged as improvement."""
        test_metrics = {"r2_score": 0.99}
        comparison = repo.compare_to_baseline("darcy_flow", "fno", test_metrics)
        assert comparison["is_better"]["r2_score"] is True


class TestBestBaseline:
    """Tests for finding the best baseline."""

    def test_best_by_mse(self, repo: BaselineRepository) -> None:
        """Get best model by MSE (lower is better)."""
        model, metrics = repo.get_best_baseline("darcy_flow", "mse")
        assert model == "fno"  # fno has lower MSE than deeponet
        assert "mse" in metrics

    def test_best_by_r2(self, repo: BaselineRepository) -> None:
        """Get best model by R2 (higher is better)."""
        model, _ = repo.get_best_baseline("darcy_flow", "r2_score")
        assert model == "fno"  # fno has higher R2

    def test_unknown_dataset_raises(self, repo: BaselineRepository) -> None:
        """Raises ValueError for unknown dataset."""
        with pytest.raises(ValueError, match="No baselines"):
            repo.get_best_baseline("nonexistent")


class TestSummary:
    """Tests for baseline summary generation."""

    def test_summary_structure(self, repo: BaselineRepository) -> None:
        """Summary includes dataset counts and model coverage."""
        summary = repo.generate_baseline_summary()
        assert summary["total_datasets"] == 3
        assert summary["total_baselines"] == 6  # 3 datasets * 2 models
        assert "fno" in summary["available_model_types"]

    def test_summary_after_add(self, repo: BaselineRepository) -> None:
        """Summary reflects newly added baselines."""
        repo.add_baseline("navier_stokes", "fno", {"mse": 0.003})
        summary = repo.generate_baseline_summary()
        assert summary["total_datasets"] == 4


class TestPersistence:
    """Tests for save/load of baselines."""

    def test_save_and_reload(self, tmp_path: Path) -> None:
        """Baselines persist across repository instances."""
        path = str(tmp_path / "baselines.json")
        store_path = tmp_path / "store"
        repo = BaselineRepository(baseline_data_path=path, store_path=store_path)
        repo.add_baseline("test_data", "model_a", {"mse": 0.01})
        repo.save_baselines()

        repo2 = BaselineRepository(baseline_data_path=path, store_path=store_path)
        metrics = repo2.get_baseline_metrics("test_data", "model_a")
        assert metrics["mse"] == pytest.approx(0.01)
