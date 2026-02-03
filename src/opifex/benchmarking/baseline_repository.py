"""Baseline Repository Module.

Stores and retrieves baseline performance metrics for PDEBench datasets.
Delegates persistence to ``calibrax.storage.Store`` while retaining
domain-specific comparison and reporting logic.
"""

import json
import logging
from pathlib import Path
from typing import Any

from calibrax.core.models import Metric, Point, Run
from calibrax.storage.store import Store

from opifex.benchmarking._shared import LOWER_IS_BETTER
from opifex.benchmarking.adapters import default_metric_defs


logger = logging.getLogger(__name__)


class BaselineRepository:
    """Repository for storing and retrieving baseline performance metrics.

    Manages a database of baseline performance metrics for standard PDEBench
    datasets, enabling comparison of new models against established benchmarks.
    New baselines are persisted via a ``calibrax.storage.Store``.
    """

    def __init__(
        self,
        baseline_data_path: str | None = None,
        store_path: Path | str | None = None,
    ) -> None:
        """Initialize baseline repository.

        Args:
            baseline_data_path: Path to baseline data file (JSON format).
            store_path: Directory for calibrax Store persistence.
        """
        self.baseline_data_path = (
            Path(baseline_data_path) if baseline_data_path else Path("baselines.json")
        )

        store_dir = (
            Path(store_path) if store_path else self.baseline_data_path.parent / "store"
        )
        self._store = Store(store_dir)

        # Initialize with standard baselines if no file exists
        self._default_baselines = {
            "advection": {
                "fno": {
                    "mse": 0.001,
                    "mae": 0.01,
                    "r2_score": 0.95,
                    "relative_error": 0.02,
                    "source": "PDEBench Standard",
                    "model_config": {
                        "hidden_channels": 64,
                        "modes": 12,
                        "num_layers": 4,
                    },
                },
                "deeponet": {
                    "mse": 0.0015,
                    "mae": 0.012,
                    "r2_score": 0.93,
                    "relative_error": 0.025,
                    "source": "PDEBench Standard",
                    "model_config": {
                        "branch_hidden_dims": [128, 128],
                        "trunk_hidden_dims": [128, 128],
                        "latent_dim": 64,
                    },
                },
            },
            "burgers": {
                "fno": {
                    "mse": 0.002,
                    "mae": 0.015,
                    "r2_score": 0.92,
                    "relative_error": 0.03,
                    "source": "PDEBench Standard",
                },
                "deeponet": {
                    "mse": 0.0025,
                    "mae": 0.018,
                    "r2_score": 0.90,
                    "relative_error": 0.035,
                    "source": "PDEBench Standard",
                },
            },
            "darcy_flow": {
                "fno": {
                    "mse": 0.0008,
                    "mae": 0.008,
                    "r2_score": 0.97,
                    "relative_error": 0.015,
                    "source": "PDEBench Standard",
                },
                "deeponet": {
                    "mse": 0.0012,
                    "mae": 0.011,
                    "r2_score": 0.95,
                    "relative_error": 0.02,
                    "source": "PDEBench Standard",
                },
            },
        }

        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load baseline data from file or initialize with defaults."""
        if self.baseline_data_path.exists():
            try:
                with open(self.baseline_data_path) as f:
                    self.baselines = json.load(f)
            except (OSError, json.JSONDecodeError):
                # Use defaults if file is corrupted
                self.baselines = self._default_baselines.copy()
        else:
            self.baselines = self._default_baselines.copy()

    def save_baselines(self) -> None:
        """Save baseline data to file."""
        self.baseline_data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_data_path, "w") as f:
            json.dump(self.baselines, f, indent=2)

    def get_baseline_metrics(
        self, dataset_name: str, model_type: str
    ) -> dict[str, float]:
        """
        Get baseline metrics for a specific dataset and model type.

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model (e.g., "fno", "deeponet")

        Returns:
            Dictionary of baseline metrics

        Raises:
            ValueError: If dataset or model type not found
        """
        if dataset_name not in self.baselines:
            raise ValueError(f"No baselines available for dataset: {dataset_name}")

        if model_type not in self.baselines[dataset_name]:
            raise ValueError(
                f"No baselines available for model type: {model_type} on "
                f"dataset: {dataset_name}"
            )

        baseline_data = self.baselines[dataset_name][model_type].copy()

        # Filter out non-metric fields
        return {
            k: v
            for k, v in baseline_data.items()
            if k not in ["source", "model_config", "notes"]
        }

    def get_available_datasets(self) -> list[str]:
        """Get list of datasets with baseline data."""
        return list(self.baselines.keys())

    def get_available_model_types(self, dataset_name: str) -> list[str]:
        """
        Get list of model types with baselines for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of available model types
        """
        if dataset_name not in self.baselines:
            return []

        return list(self.baselines[dataset_name].keys())

    def add_baseline(
        self,
        dataset_name: str,
        model_type: str,
        metrics: dict[str, float],
        source: str = "User Added",
        model_config: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        """Add a new baseline to the repository.

        Persists both to the JSON file and to the calibrax Store.

        Args:
            dataset_name: Name of the dataset.
            model_type: Type of model.
            metrics: Performance metrics.
            source: Source of the baseline data.
            model_config: Model configuration details.
            notes: Additional notes.
        """
        if dataset_name not in self.baselines:
            self.baselines[dataset_name] = {}

        baseline_entry: dict[str, Any] = metrics.copy()
        baseline_entry["source"] = source

        if model_config is not None:
            baseline_entry["model_config"] = model_config

        if notes is not None:
            baseline_entry["notes"] = notes

        self.baselines[dataset_name][model_type] = baseline_entry

        # Persist to calibrax Store
        self._persist_to_store(dataset_name, model_type, metrics)

    def _persist_to_store(
        self,
        dataset_name: str,
        model_type: str,
        metrics: dict[str, float],
    ) -> None:
        """Save a baseline as a calibrax Run to the Store.

        Args:
            dataset_name: Name of the dataset.
            model_type: Name of the model.
            metrics: Numeric metric values.
        """
        point = Point(
            name=model_type,
            scenario=dataset_name,
            tags={"dataset": dataset_name, "model_type": model_type},
            metrics={k: Metric(value=v) for k, v in metrics.items()},
        )
        run = Run(
            points=(point,),
            metadata={"source": "baseline_repository"},
            metric_defs=default_metric_defs(),
        )
        self._store.save(run)

    def compare_to_baseline(
        self,
        dataset_name: str,
        model_type: str,
        test_metrics: dict[str, float],
        metrics_to_compare: list[str] | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compare test metrics to baseline metrics.

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model
            test_metrics: Metrics to compare against baseline
            metrics_to_compare: Specific metrics to compare (None for all)

        Returns:
            Dictionary with comparison results including relative improvements
        """
        baseline_metrics = self.get_baseline_metrics(dataset_name, model_type)

        if metrics_to_compare is None:
            # Compare all metrics that are in both test and baseline
            metrics_to_compare = list(
                set(test_metrics.keys()) & set(baseline_metrics.keys())
            )

        comparison = {
            "absolute_difference": {},
            "relative_improvement": {},
            "is_better": {},
        }

        for metric in metrics_to_compare:
            if metric not in test_metrics or metric not in baseline_metrics:
                continue

            test_value = test_metrics[metric]
            baseline_value = baseline_metrics[metric]

            # Calculate absolute difference
            abs_diff = test_value - baseline_value
            comparison["absolute_difference"][metric] = abs_diff

            # Calculate relative improvement (positive means test is better)
            # For metrics like MSE, MAE (lower is better), improvement is
            # negative of relative change
            # For metrics like R2 (higher is better), improvement is positive
            # relative change
            if metric.lower() in LOWER_IS_BETTER:
                # Lower is better - negative relative change is improvement
                rel_improvement = (baseline_value - test_value) / (
                    baseline_value + 1e-8
                )
                is_better = test_value < baseline_value
            else:
                # Higher is better - positive relative change is improvement
                rel_improvement = (test_value - baseline_value) / (
                    baseline_value + 1e-8
                )
                is_better = test_value > baseline_value

            comparison["relative_improvement"][metric] = rel_improvement
            comparison["is_better"][metric] = is_better

        return comparison

    def get_best_baseline(
        self, dataset_name: str, metric: str = "mse"
    ) -> tuple[str, dict[str, float]]:
        """
        Get the best baseline for a dataset based on a specific metric.

        Args:
            dataset_name: Name of the dataset
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_type, metrics) for the best baseline
        """
        if dataset_name not in self.baselines:
            raise ValueError(f"No baselines available for dataset: {dataset_name}")

        best_model = None
        best_value = None
        best_metrics = None

        # Determine if metric is better when lower or higher
        lower_is_better = metric.lower() in LOWER_IS_BETTER

        for model_type, baseline_data in self.baselines[dataset_name].items():
            if metric not in baseline_data:
                continue

            value = baseline_data[metric]

            if (
                best_value is None
                or (lower_is_better and value < best_value)
                or (not lower_is_better and value > best_value)
            ):
                best_model = model_type
                best_value = value
                best_metrics = self.get_baseline_metrics(dataset_name, model_type)

        if best_model is None:
            raise ValueError(
                f"No baseline found with metric: {metric} for dataset: {dataset_name}"
            )

        # This is guaranteed to be non-None due to the check above
        if best_metrics is None:
            raise ValueError("No valid metrics found - this should not happen")
        return best_model, best_metrics

    def generate_baseline_summary(self) -> dict[str, Any]:
        """
        Generate a comprehensive summary of all baselines.

        Returns:
            Dictionary with baseline summary statistics
        """
        summary = {
            "total_datasets": len(self.baselines),
            "total_baselines": 0,
            "datasets": {},
            "model_coverage": {},
        }

        all_model_types = set()

        for dataset_name, dataset_baselines in self.baselines.items():
            dataset_info = {
                "model_types": list(dataset_baselines.keys()),
                "num_baselines": len(dataset_baselines),
            }

            summary["datasets"][dataset_name] = dataset_info
            summary["total_baselines"] += len(dataset_baselines)

            for model_type in dataset_baselines:
                all_model_types.add(model_type)
                if model_type not in summary["model_coverage"]:
                    summary["model_coverage"][model_type] = []
                summary["model_coverage"][model_type].append(dataset_name)

        summary["available_model_types"] = list(all_model_types)

        return summary
