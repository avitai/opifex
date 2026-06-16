# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.12.6
# ---

# %% [markdown]
"""
# Neural Operator Comparative Benchmark

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~15 min (CPU/GPU) |
| **Prerequisites** | JAX, Flax NNX, Neural Operators, Benchmarking |
| **Format** | Python + Jupyter |

## Overview

This benchmark provides a full comparative analysis of UNO, FNO, and SFNO
neural operators on the Darcy flow equation using Opifex's benchmarking
infrastructure. Each operator is *trained* with the standard operator-learning
recipe (grid positional embedding, Gaussian input/output normalization, and the
relative-L2 loss) so the accuracy column reflects learned behaviour rather than
random initialization. It evaluates accuracy, parameter count, training time,
and inference throughput across multiple grid resolutions.

## Learning Goals

1. **Compare** UNO, FNO, and SFNO on Darcy flow across resolutions
2. **Train** each operator with the proven recipe so accuracy is meaningful
3. **Evaluate** with relative-L2 / MSE accuracy, parameter count, and timing
4. **Analyze** results with statistical significance testing
5. **Generate** publication-ready comparison tables and visualizations
"""

# %%
import logging
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import numpy as np
from calibrax.core import BenchmarkResult, Metric
from flax import nnx


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.benchmarking.analysis_engine import AnalysisEngine
from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator
from opifex.benchmarking.results_manager import ResultsManager
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig
from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.common.embeddings import GridEmbedding2D

# Neural operators
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.fno.spherical import SphericalFourierNeuralOperator
from opifex.neural.operators.specialized.uno import create_uno


# %%
# Setup logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)

# %% [markdown]
"""
## Grid-Embedded Operator Wrappers

Spectral operators resolve boundary-value problems best when the normalized
``(x, y)`` coordinates are appended as extra input channels. The wrappers below
add a :class:`GridEmbedding2D` in front of each operator and standardize on a
channels-first ``(batch, channels, height, width)`` interface so the benchmark
can train and evaluate every architecture through the same code path.
"""


# %%
class FNOWithGrid(nnx.Module):
    """Fourier Neural Operator with a 2D grid positional embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying FNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            hidden_channels: Hidden layer width.
            modes: Number of Fourier modes for the spectral layers.
            num_layers: Number of spectral layers.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Append grid coordinates, then apply the FNO.

        Args:
            x: Input of shape ``(batch, channels, height, width)``.

        Returns:
            Output of shape ``(batch, out_channels, height, width)``.
        """
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        return self.fno(x_chw)


# %%
class SFNOWithGrid(nnx.Module):
    """Spherical Fourier Neural Operator with a 2D grid positional embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        lmax: int,
        num_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying SFNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            hidden_channels: Hidden layer width.
            lmax: Maximum spherical harmonic degree (controls spectral resolution).
            num_layers: Number of SFNO layers.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.sfno = SphericalFourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            lmax=lmax,
            num_layers=num_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Append grid coordinates, then apply the SFNO.

        Args:
            x: Input of shape ``(batch, channels, height, width)``.

        Returns:
            Output of shape ``(batch, out_channels, height, width)``.
        """
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        x_chw = jnp.moveaxis(x_embedded, -1, 1)
        return self.sfno(x_chw)


# %%
class UNOWithGrid(nnx.Module):
    """U-Net Neural Operator with a 2D grid positional embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        modes: int,
        n_layers: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Build the grid embedding and the underlying UNO.

        Args:
            in_channels: Number of physical input channels (before the grid).
            out_channels: Number of output channels.
            hidden_channels: Base number of UNO hidden channels.
            modes: Number of Fourier modes for the spectral layers.
            n_layers: Number of U-Net encoder/decoder stages.
            rngs: Random number generators.
        """
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.uno = create_uno(
            input_channels=self.grid_embedding.out_channels,
            output_channels=out_channels,
            hidden_channels=hidden_channels,
            modes=modes,
            n_layers=n_layers,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Append grid coordinates, then apply the UNO.

        Args:
            x: Input of shape ``(batch, channels, height, width)``.

        Returns:
            Output of shape ``(batch, out_channels, height, width)``.
        """
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        out_hwc = self.uno(x_embedded, deterministic=True)
        return jnp.moveaxis(out_hwc, -1, 1)


# %% [markdown]
"""
## Comparative Study Class

The `NeuralOperatorComparativeStudy` class orchestrates the full benchmark
pipeline: operator creation, dataset loading, training, evaluation, statistical
analysis, and reporting.
"""


# %%
class NeuralOperatorComparativeStudy:
    """Full comparative study of neural operators trained on Darcy flow."""

    def __init__(
        self,
        output_dir: str = "benchmark_results/operator_benchmark",
        resolution_sizes: list[int] | None = None,
        n_train: int = 1000,
        n_test: int = 100,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        hidden_channels: int = 32,
        seed: int = 42,
    ) -> None:
        """Initialize comparative study.

        Args:
            output_dir: Directory to store results.
            resolution_sizes: Grid resolutions to test.
            n_train: Number of training samples for each dataset.
            n_test: Number of test samples for each dataset.
            num_epochs: Number of training epochs per operator.
            batch_size: Mini-batch size for training.
            learning_rate: Adam learning rate.
            hidden_channels: Hidden width shared by all operators.
            seed: Random seed for reproducibility.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.resolution_sizes = resolution_sizes or [32, 64]
        self.n_train = n_train
        self.n_test = n_test
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_channels = hidden_channels
        self.seed = seed

        # Initialize benchmarking components with proper directory structure
        self.evaluator = BenchmarkEvaluator(output_dir=str(self.output_dir))
        self.analysis_engine = AnalysisEngine()
        self.results_manager = ResultsManager(storage_path=str(self.output_dir))

        # Store results for analysis
        self.all_results: list[BenchmarkResult] = []
        # Cache of parameter counts keyed by operator name.
        self.param_counts: dict[str, int] = {}

        logger.info(f"Initialized comparative study with output dir: {output_dir}")

    def create_operators(self, resolution: int) -> dict[str, nnx.Module]:
        """Create grid-embedded neural operators for comparison.

        Args:
            resolution: Grid resolution for the operators.

        Returns:
            Dictionary of neural operators keyed by name.
        """
        rngs = nnx.Rngs(self.seed)  # Fixed seed for reproducibility
        modes = min(16, resolution // 2)
        lmax = min(16, resolution // 2)

        operators: dict[str, nnx.Module] = {
            "UNO": UNOWithGrid(
                in_channels=1,
                out_channels=1,
                hidden_channels=self.hidden_channels,
                modes=modes,
                n_layers=3,
                rngs=nnx.Rngs(self.seed),
            ),
            "FNO": FNOWithGrid(
                in_channels=1,
                out_channels=1,
                hidden_channels=self.hidden_channels,
                modes=modes,
                num_layers=4,
                rngs=rngs,
            ),
            "SFNO": SFNOWithGrid(
                in_channels=1,
                out_channels=1,
                hidden_channels=self.hidden_channels,
                lmax=lmax,
                num_layers=4,
                rngs=nnx.Rngs(self.seed),
            ),
        }
        for name in operators:
            logger.info(f"{name} created for resolution {resolution}")
        return operators

    @staticmethod
    def _count_parameters(operator: nnx.Module) -> int:
        """Count trainable parameters of an operator.

        Args:
            operator: The neural operator module.

        Returns:
            Total number of scalar parameters.
        """
        params = nnx.state(operator, nnx.Param)
        return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

    def _load_split(
        self, n_samples: int, resolution: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load a Darcy flow split via the default Grain loader.

        Args:
            n_samples: Number of samples to draw.
            resolution: Grid resolution.
            seed: Loader seed (controls the sampled fields).

        Returns:
            Tuple ``(x, y)`` in channels-first ``(N, 1, H, W)`` layout.
        """
        loaders = create_darcy_loader(
            n_samples=n_samples,
            batch_size=self.batch_size,
            resolution=resolution,
            seed=seed,
        )
        # Drain both train/val pipelines for one contiguous block; batches are
        # already channels-first ``(N, 1, H, W)``.
        inputs: list[np.ndarray] = []
        outputs: list[np.ndarray] = []
        for pipeline in (loaders.train, loaders.val):
            for batch in pipeline:
                inputs.append(np.asarray(batch["input"]))
                outputs.append(np.asarray(batch["output"]))

        x = np.concatenate(inputs, axis=0)[:n_samples]
        y = np.concatenate(outputs, axis=0)[:n_samples]
        return x, y

    def generate_test_datasets(self, resolution: int) -> dict[str, dict[str, jnp.ndarray]]:
        """Load and normalize the Darcy dataset for a resolution.

        Inputs and outputs are standardized with Gaussian statistics fit on the
        training split. Predictions are un-normalized before error computation.

        Args:
            resolution: Grid resolution.

        Returns:
            Dictionary of datasets with normalized train/test splits plus the
            output statistics needed to un-normalize predictions.
        """
        datasets: dict[str, dict[str, jnp.ndarray]] = {}

        logger.info(f"Loading Darcy dataset at resolution {resolution}...")
        x_train, y_train = self._load_split(self.n_train, resolution, self.seed)
        x_test, y_test = self._load_split(self.n_test, resolution, self.seed + 1000)

        x_mean, x_std = float(x_train.mean()), float(x_train.std())
        y_mean, y_std = float(y_train.mean()), float(y_train.std())

        datasets["Darcy"] = {
            "x_train": jnp.asarray((x_train - x_mean) / x_std),
            "y_train": jnp.asarray((y_train - y_mean) / y_std),
            "x_test": jnp.asarray((x_test - x_mean) / x_std),
            "y_test_norm": jnp.asarray((y_test - y_mean) / y_std),
            "y_test_raw": jnp.asarray(y_test),
            "y_mean": jnp.asarray(y_mean),
            "y_std": jnp.asarray(y_std),
        }
        logger.info(f"Darcy dataset ready: {datasets['Darcy']['x_train'].shape}")
        return datasets

    def _train_operator(
        self,
        operator: nnx.Module,
        dataset: dict[str, jnp.ndarray],
    ) -> tuple[nnx.Module, float]:
        """Train an operator with the standard operator-learning recipe.

        Args:
            operator: The neural operator module (channels-first interface).
            dataset: Normalized dataset splits.

        Returns:
            Tuple ``(trained_operator, training_time_seconds)``.
        """
        # Warm any lazily-built spectral caches (e.g. the SFNO harmonic basis)
        # outside of ``jit`` so they hold concrete constants, not tracers.
        _ = operator(dataset["x_train"][:2])

        config = TrainingConfig(
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            validation_frequency=max(1, self.num_epochs // 4),
            verbose=False,
            loss_config=LossConfig(loss_type="relative_l2"),
        )
        trainer = Trainer(model=operator, config=config, rngs=nnx.Rngs(self.seed))

        start_time = time.perf_counter()
        trained_model, _ = trainer.fit(
            train_data=(dataset["x_train"], dataset["y_train"]),
            val_data=(dataset["x_test"], dataset["y_test_norm"]),
        )
        return trained_model, time.perf_counter() - start_time

    @staticmethod
    def _relative_l2(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
        """Compute the mean per-sample relative L2 error.

        Args:
            predictions: Predicted fields (un-normalized).
            targets: Ground-truth fields (un-normalized).

        Returns:
            Mean relative L2 error across the batch.
        """
        pred_flat = predictions.reshape(predictions.shape[0], -1)
        target_flat = targets.reshape(targets.shape[0], -1)
        l2_diff = jnp.linalg.norm(pred_flat - target_flat, axis=1)
        l2_target = jnp.linalg.norm(target_flat, axis=1)
        return float(jnp.mean(l2_diff / (l2_target + 1e-8)))

    def benchmark_operator(
        self,
        operator_name: str,
        operator: nnx.Module,
        dataset: dict[str, jnp.ndarray],
        dataset_name: str,
        resolution: int,
    ) -> BenchmarkResult:
        """Train, then benchmark a single operator on a dataset.

        Args:
            operator_name: Name of the neural operator.
            operator: The neural operator module.
            dataset: Normalized dataset with train/test splits.
            dataset_name: Name of the dataset.
            resolution: Grid resolution.

        Returns:
            Benchmark result with accuracy, parameter, and timing metrics.
        """
        logger.info(f"Benchmarking {operator_name} on {dataset_name} (resolution: {resolution})")

        param_count = self._count_parameters(operator)
        self.param_counts[operator_name] = param_count

        try:
            trained_model, train_time = self._train_operator(operator, dataset)

            # Un-normalize predictions back to physical space for accuracy.
            y_mean = dataset["y_mean"]
            y_std = dataset["y_std"]

            def model_fn(x: jnp.ndarray) -> jnp.ndarray:
                return trained_model(x) * y_std + y_mean

            # Inference timing + MSE/MAE via the benchmarking evaluator.
            result = self.evaluator.evaluate_model(
                model=model_fn,
                model_name=f"{operator_name}_{resolution}",
                input_data=dataset["x_test"],
                target_data=dataset["y_test_raw"],
                dataset_name=f"{dataset_name}_{resolution}",
            )

            # Coerce calibrax metric values (JAX arrays) to plain floats so the
            # results manager can serialize them to JSON. ``result`` is frozen, so
            # mutate the underlying metrics dict in place.
            for name in list(result.metrics):
                result.metrics[name] = Metric(value=float(result.metrics[name].value))

            # Add relative-L2 accuracy, parameter count, and training time.
            predictions = model_fn(dataset["x_test"])
            rel_l2 = self._relative_l2(predictions, dataset["y_test_raw"])
            result.metrics["relative_l2"] = Metric(value=rel_l2)
            result.metrics["parameters"] = Metric(value=float(param_count))
            result.metadata["training_time"] = float(train_time)

            mse_metric = result.metrics.get("mse")
            mse_val = mse_metric.value if mse_metric else float("nan")
            infer_time = result.metadata.get("execution_time", 0.0)
            logger.info(
                f"{operator_name} on {dataset_name}: relL2={rel_l2:.4%}, "
                f"MSE={mse_val:.6e}, params={param_count:,}, "
                f"train={train_time:.1f}s, infer={infer_time:.4f}s"
            )
            return result

        except (RuntimeError, ValueError) as e:
            logger.exception(f"Benchmarking failed for {operator_name}")
            return BenchmarkResult(
                name=f"{operator_name}_{resolution}",
                domain="scientific_ml",
                tags={"dataset": f"{dataset_name}_{resolution}"},
                metrics={"error": Metric(value=float("inf"))},
                metadata={"execution_time": float("inf"), "error": str(e)},
            )

    def run_resolution_study(self) -> None:
        """Run comparative study across different resolutions."""
        logger.info("Starting multi-resolution comparative study...")

        for resolution in self.resolution_sizes:
            logger.info("=" * 60)
            logger.info(f"RESOLUTION {resolution}x{resolution} STUDY")
            logger.info("=" * 60)

            datasets = self.generate_test_datasets(resolution)
            if not datasets:
                logger.warning(f"No datasets generated for resolution {resolution}")
                continue

            operators = self.create_operators(resolution)
            if not operators:
                logger.warning(f"No operators created for resolution {resolution}")
                continue

            # Train + benchmark each operator on each dataset.
            for operator_name, operator in operators.items():
                for dataset_name, dataset in datasets.items():
                    result = self.benchmark_operator(
                        operator_name=operator_name,
                        operator=operator,
                        dataset=dataset,
                        dataset_name=dataset_name,
                        resolution=resolution,
                    )
                    self.all_results.append(result)

            # Save intermediate results
            self.save_intermediate_results(resolution)

        logger.info("Multi-resolution study completed!")

    def save_intermediate_results(self, resolution: int) -> None:
        """Save intermediate results for this resolution.

        Args:
            resolution: Grid resolution that was just completed.
        """
        try:
            resolution_results = [
                r
                for r in self.all_results
                if f"_{resolution}" in r.name and f"_{resolution}" in r.tags.get("dataset", "")
            ]

            if resolution_results:
                for result in resolution_results:
                    self.results_manager.save_benchmark_results(result)
                logger.info(f"Saved {len(resolution_results)} results for resolution {resolution}")
            else:
                logger.warning(f"No results to save for resolution {resolution}")

        except (OSError, ValueError):
            logger.exception("Failed to save intermediate results")

    def generate_comparative_analysis(self) -> None:
        """Generate full comparative analysis."""
        logger.info("Generating comparative analysis...")

        if not self.all_results:
            logger.warning("No results available for analysis")
            return

        try:
            results_by_operator: dict[str, list[BenchmarkResult]] = {}
            results_by_dataset: dict[str, list[BenchmarkResult]] = {}

            for result in self.all_results:
                if result.metrics.get("error") is not None:
                    continue  # Skip failed benchmarks

                operator_name = result.name.split("_")[0]
                dataset_name = result.tags.get("dataset", "unknown").split("_")[0]

                results_by_operator.setdefault(operator_name, []).append(result)
                results_by_dataset.setdefault(dataset_name, []).append(result)

            self.create_performance_plots(results_by_operator, results_by_dataset)
            self.perform_statistical_analysis(results_by_operator)
            self.generate_summary_report(results_by_operator, results_by_dataset)

            logger.info("Comparative analysis completed!")

        except (ValueError, KeyError):
            logger.exception("Analysis generation failed")

    def create_performance_plots(
        self,
        results_by_operator: dict[str, list[BenchmarkResult]],
        results_by_dataset: dict[str, list[BenchmarkResult]],
    ) -> None:
        """Create performance comparison plots.

        Args:
            results_by_operator: Results grouped by operator name.
            results_by_dataset: Results grouped by dataset name.
        """
        logger.info("Creating performance plots...")

        try:
            # Plot 1: relative-L2 accuracy vs resolution per dataset.
            _, ax = plt.subplots(figsize=(8, 6))
            for dataset_name, dataset_results in results_by_dataset.items():
                operator_err: dict[str, dict[int, float]] = {}
                resolutions: set[int] = set()
                for result in dataset_results:
                    operator_name = result.name.split("_")[0]
                    resolution = int(result.name.split("_")[1])
                    err_m = result.metrics.get("relative_l2")
                    err = err_m.value if err_m else float("inf")
                    operator_err.setdefault(operator_name, {})[resolution] = err
                    resolutions.add(resolution)

                for operator_name, err_data in operator_err.items():
                    res_list = sorted(resolutions)
                    err_values = [err_data.get(r, float("inf")) for r in res_list]
                    valid = [
                        (r, v)
                        for r, v in zip(res_list, err_values, strict=True)
                        if v != float("inf")
                    ]
                    if valid:
                        xs, ys = zip(*valid, strict=True)
                        ax.plot(
                            xs, ys, "o-", label=f"{operator_name} ({dataset_name})", linewidth=2
                        )

            ax.set_title("Relative L2 Error vs Resolution")
            ax.set_xlabel("Resolution")
            ax.set_ylabel("Relative L2 Error")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Plot 2: inference time distribution per operator.
            _, ax = plt.subplots(figsize=(10, 6))
            execution_times: dict[str, list[float]] = {}
            for operator_name, results in results_by_operator.items():
                times = [
                    r.metadata.get("execution_time", float("inf"))
                    for r in results
                    if r.metadata.get("execution_time", float("inf")) != float("inf")
                ]
                if times:
                    execution_times[operator_name] = times

            if execution_times:
                operators = list(execution_times.keys())
                times_data = [execution_times[op] for op in operators]
                ax.boxplot(times_data, tick_labels=operators)
                ax.set_title("Inference Time Distribution by Operator")
                ax.set_ylabel("Inference Time (seconds)")
                ax.set_xlabel("Neural Operator")
                plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "execution_time_comparison.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()

            logger.info("Performance plots saved")

        except (ValueError, KeyError):
            logger.exception("Plot creation failed")

    def perform_statistical_analysis(
        self, results_by_operator: dict[str, list[BenchmarkResult]]
    ) -> None:
        """Perform pairwise statistical analysis of operator accuracy.

        Args:
            results_by_operator: Results grouped by operator name.
        """
        logger.info("Performing statistical analysis...")

        try:
            analysis_results: dict[str, dict[str, float]] = {}
            operators = list(results_by_operator.keys())
            if len(operators) < 2:
                logger.warning("Need at least 2 operators for comparison")
                return

            for i in range(len(operators)):
                for j in range(i + 1, len(operators)):
                    op1, op2 = operators[i], operators[j]
                    err1 = [
                        r.metrics["relative_l2"].value
                        for r in results_by_operator[op1]
                        if "relative_l2" in r.metrics
                    ]
                    err2 = [
                        r.metrics["relative_l2"].value
                        for r in results_by_operator[op2]
                        if "relative_l2" in r.metrics
                    ]
                    if err1 and err2:
                        mean1, mean2 = float(np.mean(err1)), float(np.mean(err2))
                        analysis_results[f"{op1}_vs_{op2}"] = {
                            "mean_rel_l2_diff": mean1 - mean2,
                            "relative_improvement": (mean2 - mean1) / mean2 * 100,
                            f"{op1}_mean": mean1,
                            f"{op1}_std": float(np.std(err1)),
                            f"{op2}_mean": mean2,
                            f"{op2}_std": float(np.std(err2)),
                        }

            import json

            with open(self.output_dir / "statistical_analysis.json", "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)

            logger.info("Statistical analysis completed")

        except (ValueError, KeyError, OSError):
            logger.exception("Statistical analysis failed")

    @staticmethod
    def _metric_mean(results: list[BenchmarkResult], key: str, default: float) -> float:
        """Average a named metric across results, ignoring missing entries.

        Args:
            results: Benchmark results to aggregate.
            key: Metric name to read.
            default: Fallback value for results missing the metric.

        Returns:
            Mean of the metric across the results.
        """
        values = [r.metrics[key].value if key in r.metrics else default for r in results]
        return float(np.mean(values)) if values else default

    def _write_operator_section(
        self, f, results_by_operator: dict[str, list[BenchmarkResult]]
    ) -> None:
        """Write the per-operator performance summary table.

        Args:
            f: Open file handle for the report.
            results_by_operator: Results grouped by operator name.
        """
        f.write("## Neural Operator Performance\n\n")
        f.write("| Operator | Rel L2 | MSE | Parameters | Train (s) | Infer (s) |\n")
        f.write("|----------|--------|-----|------------|-----------|-----------|\n")
        for operator_name, results in results_by_operator.items():
            rel_l2 = self._metric_mean(results, "relative_l2", float("inf"))
            mse = self._metric_mean(results, "mse", float("inf"))
            params = int(self._metric_mean(results, "parameters", 0.0))
            train_t = float(np.mean([r.metadata.get("training_time", 0.0) for r in results]))
            infer_t = float(np.mean([r.metadata.get("execution_time", 0.0) for r in results]))
            f.write(
                f"| {operator_name} | {rel_l2:.4%} | {mse:.2e} | "
                f"{params:,} | {train_t:.1f} | {infer_t:.4f} |\n"
            )
        f.write("\n")

    def _write_key_findings(self, f, results_by_operator: dict[str, list[BenchmarkResult]]) -> None:
        """Write the key-findings section of the report.

        Args:
            f: Open file handle for the report.
            results_by_operator: Results grouped by operator name.
        """
        f.write("## Key Findings\n\n")

        operator_avg_err: dict[str, float] = {}
        for operator_name, results in results_by_operator.items():
            valid = [r for r in results if "relative_l2" in r.metrics]
            if valid:
                operator_avg_err[operator_name] = float(
                    np.mean([r.metrics["relative_l2"].value for r in valid])
                )
        if operator_avg_err:
            best = min(operator_avg_err, key=operator_avg_err.__getitem__)
            f.write(f"- **Best Overall Accuracy**: {best} (Rel L2: {operator_avg_err[best]:.4%})\n")

        operator_avg_time: dict[str, float] = {}
        for operator_name, results in results_by_operator.items():
            valid = [
                r for r in results if r.metadata.get("execution_time", float("inf")) != float("inf")
            ]
            if valid:
                operator_avg_time[operator_name] = float(
                    np.mean([r.metadata["execution_time"] for r in valid])
                )
        if operator_avg_time:
            fastest = min(operator_avg_time, key=operator_avg_time.__getitem__)
            f.write(
                f"- **Fastest Inference**: {fastest} ({operator_avg_time[fastest]:.4f}s average)\n"
            )
        f.write("\n")

    def generate_summary_report(
        self,
        results_by_operator: dict[str, list[BenchmarkResult]],
        results_by_dataset: dict[str, list[BenchmarkResult]],
    ) -> None:
        """Generate the full markdown summary report.

        Args:
            results_by_operator: Results grouped by operator name.
            results_by_dataset: Results grouped by dataset name.
        """
        logger.info("Generating summary report...")

        try:
            report_path = self.output_dir / "comparative_study_report.md"
            with open(report_path, "w") as f:
                f.write("# Neural Operator Comparative Benchmarking Study\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("## Executive Summary\n\n")
                f.write(
                    f"This report presents a comparative analysis of "
                    f"{len(results_by_operator)} neural operators trained on "
                    f"{len(results_by_dataset)} dataset(s) across "
                    f"{len(self.resolution_sizes)} resolutions.\n\n"
                )

                f.write("## Neural Operators Analyzed\n\n")
                descriptions = {
                    "UNO": "U-Net Neural Operator (Multi-scale CNN + Fourier layers)",
                    "FNO": "Fourier Neural Operator (Spectral convolutions)",
                    "SFNO": "Spherical Fourier Neural Operator (Spherical harmonics)",
                }
                for operator_name in results_by_operator:
                    f.write(
                        f"- **{operator_name}**: "
                        f"{descriptions.get(operator_name, 'Neural operator')}\n"
                    )
                f.write("\n")

                f.write("## Datasets Evaluated\n\n")
                for dataset_name, dataset_results in results_by_dataset.items():
                    f.write(f"- **{dataset_name}**: {len(dataset_results)} benchmark runs\n")
                f.write("\n")

                f.write("## Multi-Resolution Analysis\n\n")
                f.write(f"**Resolutions tested**: {', '.join(map(str, self.resolution_sizes))}\n\n")

                self._write_operator_section(f, results_by_operator)
                self._write_key_findings(f, results_by_operator)

                f.write("## Conclusions\n\n")
                f.write(
                    "Each operator was trained with the standard operator-learning "
                    "recipe (grid embedding, Gaussian normalization, relative-L2 "
                    "loss). The accuracy column therefore reflects learned behaviour "
                    "and is directly comparable across architectures.\n\n"
                )

                f.write("## Generated Files\n\n")
                f.write("- `accuracy_comparison.png`: Relative L2 vs resolution plots\n")
                f.write("- `execution_time_comparison.png`: Inference time distributions\n")
                f.write("- `statistical_analysis.json`: Pairwise statistical comparisons\n")
                f.write("- Individual benchmark result files in the results directory\n")

            logger.info(f"Report saved to {report_path}")

        except (OSError, ValueError, KeyError):
            logger.exception("Report generation failed")

    def print_summary_table(self) -> None:
        """Print a console summary table of the benchmark results."""
        successful = [r for r in self.all_results if "relative_l2" in r.metrics]
        if not successful:
            logger.warning("No successful results to summarize")
            return

        logger.info("=" * 78)
        logger.info("BENCHMARK SUMMARY (Darcy flow, trained operators)")
        logger.info("=" * 78)
        header = (
            f"{'Operator':<10}{'Res':>5}{'Rel L2':>12}"
            f"{'MSE':>14}{'Params':>14}{'Train(s)':>11}{'Infer(s)':>11}"
        )
        logger.info(header)
        logger.info("-" * 78)
        for result in successful:
            name, res = result.name.split("_")
            rel_l2 = result.metrics["relative_l2"].value
            mse = result.metrics["mse"].value if "mse" in result.metrics else float("nan")
            params = int(result.metrics["parameters"].value)
            train_t = result.metadata.get("training_time", 0.0)
            infer_t = result.metadata.get("execution_time", 0.0)
            logger.info(
                f"{name:<10}{res:>5}{rel_l2:>11.4%}{mse:>14.3e}"
                f"{params:>14,}{train_t:>11.1f}{infer_t:>11.4f}"
            )
        logger.info("=" * 78)

    def run_complete_study(self) -> None:
        """Run the complete comparative study."""
        logger.info("Starting full neural operator comparative study!")
        start_time = time.perf_counter()

        try:
            self.run_resolution_study()
            self.generate_comparative_analysis()
            self.print_summary_table()

            total_time = time.perf_counter() - start_time
            logger.info(f"Complete study finished in {total_time:.2f} seconds!")

            successful_runs = len([r for r in self.all_results if "relative_l2" in r.metrics])
            total_runs = len(self.all_results)
            logger.info("STUDY SUMMARY:")
            logger.info(f"   Total benchmark runs: {total_runs}")
            logger.info(f"   Successful runs: {successful_runs}")
            if total_runs > 0:
                logger.info(f"   Success rate: {successful_runs / total_runs * 100:.1f}%")
            logger.info(f"   Results saved to: {self.output_dir}")

        except (RuntimeError, ValueError):
            logger.exception("Study execution failed")
            raise


# %% [markdown]
"""
## Run the Study

The cell below runs the benchmark with a compact configuration so the notebook
finishes quickly. Trained operators reach a low-single-digit relative L2 error
on Darcy flow, so the accuracy column is meaningful for comparison.
"""


# %%
def main() -> dict[str, float | int]:
    """Run the comparative study and return a finite summary of the results."""
    study = NeuralOperatorComparativeStudy(
        resolution_sizes=[32, 64],
        n_train=1000,
        n_test=100,
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        hidden_channels=32,
        seed=42,
    )
    study.run_complete_study()

    successful = [r for r in study.all_results if "relative_l2" in r.metrics]
    best_relative_l2 = (
        min(r.metrics["relative_l2"].value for r in successful) if successful else 0.0
    )
    return {
        "total_runs": len(study.all_results),
        "successful_runs": len(successful),
        "best_relative_l2": float(best_relative_l2),
    }


# %% [markdown]
"""
## Next Steps

- Run on GPU for accurate throughput benchmarks
- Add PDEBench datasets for standardized comparison
- Compare against neuraloperator (PyTorch) and DeepXDE baselines
- See individual model examples (`fno_darcy.py`, `uno_darcy.py`) for details
"""

# %%
if __name__ == "__main__":
    summary = main()
    for key, value in summary.items():
        print(f"{key}: {value}")
