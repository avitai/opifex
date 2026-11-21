"""
PDEBench Integration Module

This module provides full integration with PDEBench datasets for
standardized evaluation of neural operators. It includes dataset loading,
preprocessing, and automated evaluation pipelines.

Key Features:
- Support for major PDEBench datasets (Advection, Burgers, Darcy Flow, etc.)
- Standardized data preprocessing for neural operator compatibility
- Automated evaluation pipelines with statistical analysis
- Integration with existing benchmarking infrastructure

Following Critical Technical Guidelines:
- JAX-native data processing for GPU compatibility
- FLAX NNX integration for neural operator evaluation
- Test-driven development with full coverage
- Type hints and documentation for all public APIs
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from calibrax.core import BenchmarkResult
from jaxtyping import Array
from substrax.rng import fold_in_name

from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator
from opifex.data.sources import load_pdebench, PDEBenchConfig


@dataclass(frozen=True, kw_only=True)
class _Family:
    """A PDEBench dataset family: where its files are and what one sample looks like.

    Attributes:
        file_pattern: Glob of the family's file names, as PDEBench publishes them
            (``pdebench/data_download/pdebench_data_urls.csv``); ``None`` for a family whose
            files no PDEBench loader reads.
        spatial_dims: Number of spatial axes.
        channels: Fields per point.
        initial_step: Input time steps of PDEBench's training (1 for Darcy flow, whose two
            steps are the coefficient and the solution).
        default_resolution: Spatial shape of synthetic data at the "low" setting.
        physics_type: Category of the equation.
    """

    file_pattern: str | None
    spatial_dims: int
    channels: int
    initial_step: int
    default_resolution: tuple[int, ...]
    physics_type: str


_FAMILIES: dict[str, _Family] = {
    "advection": _Family(
        file_pattern="1D_Advection_Sols_*.hdf5",
        spatial_dims=1,
        channels=1,
        initial_step=10,
        default_resolution=(64,),
        physics_type="hyperbolic",
    ),
    "burgers": _Family(
        file_pattern="1D_Burgers_Sols_*.hdf5",
        spatial_dims=1,
        channels=1,
        initial_step=10,
        default_resolution=(256,),
        physics_type="nonlinear_hyperbolic",
    ),
    "darcy_flow": _Family(
        file_pattern="2D_DarcyFlow_*.hdf5",
        spatial_dims=2,
        channels=1,
        initial_step=1,
        default_resolution=(64, 64),
        physics_type="elliptic",
    ),
    "shallow_water": _Family(
        file_pattern="2D_rdb_*.h5",
        spatial_dims=2,
        channels=2,
        initial_step=10,
        default_resolution=(64, 64),
        physics_type="hyperbolic_system",
    ),
    "navier_stokes": _Family(
        file_pattern=None,
        spatial_dims=2,
        channels=2,
        initial_step=10,
        default_resolution=(64, 64),
        physics_type="parabolic_nonlinear",
    ),
    "diffusion_reaction": _Family(
        file_pattern="2D_diff-react_*.h5",
        spatial_dims=2,
        channels=2,
        initial_step=10,
        default_resolution=(64, 64),
        physics_type="parabolic_system",
    ),
    "compressible_navier_stokes": _Family(
        file_pattern="*_CFD_*.hdf5",
        spatial_dims=2,
        channels=4,
        initial_step=10,
        default_resolution=(64, 64),
        physics_type="hyperbolic_parabolic_system",
    ),
}

_NO_READER = "ns_incom_*.h5"
"""PDEBench's incompressible Navier-Stokes files, which no PDEBench loader reads."""


def _first_training_pair(field: Array, initial_step: int) -> tuple[Array, Array]:
    """Return PDEBench's first autoregressive training pair, channels first.

    PDEBench's FNO training (``pdebench/models/fno/train.py``) stacks the ``initial_step``
    input frames of a ``[N, x..., t, V]`` field as channels and predicts frame
    ``initial_step``.

    Args:
        field: ``[N, x..., t, V]``.
        initial_step: Input frames.

    Returns:
        ``(N, initial_step * V, x...)`` inputs and ``(N, V, x...)`` targets.
    """
    frames = field[..., :initial_step, :]
    inputs = frames.reshape(*frames.shape[:-2], -1)
    target = field[..., initial_step, :]
    return jnp.moveaxis(inputs, -1, 1), jnp.moveaxis(target, -1, 1)


class PDEBenchLoader:
    """
    Loads PDEBench datasets for neural operator evaluation.

    Real files are read through :func:`opifex.data.sources.load_pdebench`, as PDEBench's own
    loader reads them, and returned as PDEBench's first training pair; synthetic data, for
    tests, is drawn only when a ``synthetic_key`` is given.
    """

    def __init__(self, data_root: str | None = None) -> None:
        """
        Initialize PDEBench dataset loader.

        Args:
            data_root: Directory holding PDEBench files under their published names;
                ``./data/pdebench`` when ``None``. Nothing is created.
        """
        self.data_root = Path(data_root) if data_root else Path.cwd() / "data" / "pdebench"
        self.supported_datasets = list(_FAMILIES)

    def list_available_datasets(self) -> list[str]:
        """List all supported PDEBench datasets."""
        return self.supported_datasets.copy()

    def get_dataset_info(self, dataset_name: str) -> dict[str, Any]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary containing dataset metadata and characteristics

        Raises:
            ValueError: If ``dataset_name`` is not supported.
        """
        if dataset_name not in _FAMILIES:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported datasets: {self.supported_datasets}"
            )
        family = _FAMILIES[dataset_name]
        return {
            "equation": dataset_name,
            "spatial_dims": family.spatial_dims,
            "temporal": dataset_name != "darcy_flow",
            "channels": family.channels,
            "initial_step": family.initial_step,
            "default_resolution": family.default_resolution,
            "physics_type": family.physics_type,
        }

    def load_dataset(
        self,
        dataset_name: str,
        subset_size: int | None = None,
        resolution: str = "low",
        split: str = "test",
        format_for_model: str = "auto",
        *,
        file_name: str | None = None,
        synthetic_key: jax.Array | None = None,
    ) -> dict[str, Any]:
        """
        Load a PDEBench dataset as PDEBench's first training pair.

        A real file under ``data_root`` is read with PDEBench's layout and split (the test
        split is the first tenth of the samples): inputs are the first ``initial_step`` frames
        stacked as channels, ``(n, initial_step * V, x...)``, targets the next frame,
        ``(n, V, x...)``. Without a real file a ``synthetic_key`` opts in to synthetic data of
        the same shapes, with a :class:`UserWarning` and ``is_synthetic=True`` in the metadata.

        Args:
            dataset_name: Name of the dataset to load
            subset_size: Cap on the file's samples before the split (PDEBench's
                ``num_samples_max``); for synthetic data, the number of samples (100 when
                ``None``).
            resolution: Synthetic resolution ("low", "medium", "high"); a real file is read at
                its own resolution.
            split: "train" or "test" (PDEBench has no validation split).
            format_for_model: Target model format ("fno", "deeponet", "auto")
            file_name: The file to read when several under ``data_root`` match the dataset.
            synthetic_key: Key that opts in to clearly-flagged synthetic data when no
                real PDEBench file is available, folded by the dataset's name before drawing.
                ``None`` (the default) fails fast.

        Returns:
            Dictionary containing:
                - input_data: Input arrays
                - target_data: Target arrays
                - metadata: Dataset metadata (including ``data_source`` and
                  ``is_synthetic``)

        Raises:
            ValueError: If ``dataset_name`` is not supported, ``split`` is neither "train" nor
                "test", several files match and none is named, or the dataset's files have no
                PDEBench reader.
            FileNotFoundError: If no real PDEBench file is found for the dataset and no
                ``synthetic_key`` is given.
        """
        if dataset_name not in _FAMILIES:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        if split not in ("train", "test"):
            raise ValueError(
                f"split must be 'train' or 'test' (PDEBench has no other), got {split!r}"
            )
        family = _FAMILIES[dataset_name]
        info = self.get_dataset_info(dataset_name)

        real_path = self._real_data_path(dataset_name, family, file_name)
        if real_path is not None:
            data = load_pdebench(
                PDEBenchConfig(
                    file_path=real_path,
                    split=split,
                    initial_step=family.initial_step,
                    num_samples_max=-1 if subset_size is None else subset_size,
                )
            )
            input_data, target_data = _first_training_pair(
                jnp.asarray(data.targets), family.initial_step
            )
            grid = jnp.asarray(data.grid).reshape(-1, family.spatial_dims)
            spatial_shape = tuple(data.grid.shape[:-1])
            source = {"data_source": str(real_path), "is_synthetic": False}
        else:
            if synthetic_key is None:
                raise FileNotFoundError(
                    f"No real PDEBench file found for dataset {dataset_name!r} under "
                    f"{self.data_root}. Download the PDEBench data files first, or pass "
                    "a synthetic_key to generate clearly-flagged synthetic data "
                    "(metadata['is_synthetic'] == True) for testing only."
                )
            warnings.warn(
                f"Returning SYNTHETIC data for PDEBench dataset {dataset_name!r}: no real "
                f"file was found under {self.data_root}. This data does NOT reflect the "
                "true PDE solution and must not be used for reported benchmark numbers. "
                "Metadata is flagged with is_synthetic=True.",
                UserWarning,
                stacklevel=2,
            )
            multiplier = {"low": 1, "medium": 2, "high": 4}.get(resolution, 1)
            spatial_shape = tuple(int(dim * multiplier) for dim in family.default_resolution)
            n_samples = subset_size if subset_size is not None else 100
            input_key, target_key = jax.random.split(fold_in_name(synthetic_key, dataset_name))
            input_data = jax.random.normal(
                input_key, (n_samples, family.initial_step * family.channels, *spatial_shape)
            )
            target_data = jax.random.normal(
                target_key, (n_samples, family.channels, *spatial_shape)
            )
            grid = self._generate_spatial_coordinates(spatial_shape)
            source = {"data_source": "synthetic", "is_synthetic": True}

        if format_for_model == "deeponet":
            n_samples = input_data.shape[0]
            input_data = (
                input_data.reshape(n_samples, -1),
                jnp.tile(grid[None, :, :], (n_samples, 1, 1)),
            )
            target_data = target_data.reshape(n_samples, -1)

        metadata = {
            **info,
            "batch_size": int(target_data.shape[0]),
            "resolution": resolution,
            "split": split,
            "format": format_for_model,
            "spatial_shape": spatial_shape,
            **source,
        }
        return {"input_data": input_data, "target_data": target_data, "metadata": metadata}

    def _real_data_path(
        self, dataset_name: str, family: _Family, file_name: str | None
    ) -> Path | None:
        """Locate the real PDEBench file for ``dataset_name`` under ``data_root``.

        Args:
            dataset_name: Name of the dataset.
            family: Its family entry.
            file_name: The file to read when several match.

        Returns:
            The file, or ``None`` when none is present.

        Raises:
            ValueError: If the dataset's files have no PDEBench reader, or several match and
                ``file_name`` names none of them.
        """
        if not self.data_root.exists():
            return None
        if file_name is not None:
            named = self.data_root / file_name
            return named if named.exists() else None
        if family.file_pattern is None:
            unreadable = sorted(self.data_root.glob(_NO_READER))
            if unreadable:
                raise ValueError(
                    f"{unreadable[0].name}: PDEBench's incompressible Navier-Stokes files have "
                    f"no reader in PDEBench's loaders, so {dataset_name!r} has none here either"
                )
            return None
        matches = sorted(self.data_root.glob(family.file_pattern))
        if len(matches) > 1:
            names = ", ".join(match.name for match in matches)
            raise ValueError(
                f"Several PDEBench files match {dataset_name!r} under {self.data_root}: "
                f"{names}; pass file_name to choose one"
            )
        return matches[0] if matches else None

    def _generate_spatial_coordinates(self, spatial_shape: tuple[int, ...]) -> Array:
        """Generate spatial coordinate grid for the given shape."""
        if len(spatial_shape) == 1:
            # 1D coordinates
            x = jnp.linspace(0, 1, spatial_shape[0])
            return x.reshape(-1, 1)
        if len(spatial_shape) == 2:
            # 2D coordinates
            x = jnp.linspace(0, 1, spatial_shape[0])
            y = jnp.linspace(0, 1, spatial_shape[1])
            X, Y = jnp.meshgrid(x, y, indexing="ij")
            return jnp.stack([X.flatten(), Y.flatten()], axis=1)
        raise NotImplementedError(f"Spatial coordinates for {len(spatial_shape)}D not implemented")


class PDEBenchEvaluationPipeline:
    """
    Automated evaluation pipeline for PDEBench datasets.

    This class provides end-to-end evaluation workflows that integrate
    dataset loading, model evaluation, and result analysis.
    """

    def __init__(
        self, output_dir: str | None = None, *, synthetic_key: jax.Array | None = None
    ) -> None:
        """
        Initialize evaluation pipeline.

        Args:
            output_dir: Directory for saving evaluation results
            synthetic_key: Propagated to :meth:`PDEBenchLoader.load_dataset`. Given a
                key, the pipeline may run on clearly-flagged synthetic data drawn from
                it (with a loud warning) if real PDEBench files are absent. ``None``
                (the default) fails fast on missing real data.
        """
        self.loader = PDEBenchLoader()
        self.synthetic_key = synthetic_key
        self.evaluator = BenchmarkEvaluator(
            output_dir=output_dir or "./pdebench_results", save_detailed_results=True
        )

    def evaluate_model_on_datasets(
        self,
        model: Any,
        model_name: str,
        datasets: list[str],
        subset_size: int = 10,
        resolution: str = "low",
        **kwargs: Any,
    ) -> list[BenchmarkResult]:
        """
        Evaluate a model on multiple PDEBench datasets.

        Args:
            model: Neural operator model to evaluate
            model_name: Name identifier for the model
            datasets: List of dataset names to evaluate on
            subset_size: Number of samples per dataset
            resolution: Resolution setting for datasets
            **kwargs: Additional arguments for evaluation

        Returns:
            List of benchmark results for each dataset
        """
        results = []

        for dataset_name in datasets:
            dataset = self.loader.load_dataset(
                dataset_name=dataset_name,
                subset_size=subset_size,
                resolution=resolution,
                format_for_model="auto",
                synthetic_key=self.synthetic_key,
            )
            forward_fn = None
            if isinstance(dataset["input_data"], tuple):
                # DeepONet-style input
                forward_fn = lambda model, inputs: model(inputs[0], inputs[1])
            results.append(
                self.evaluator.evaluate_model(
                    model=model,
                    model_name=model_name,
                    input_data=dataset["input_data"],
                    target_data=dataset["target_data"],
                    dataset_name=dataset_name,
                    forward_fn=forward_fn,
                    **kwargs,
                )
            )

        return results

    def run_comprehensive_evaluation(
        self,
        models: list[tuple[str, Any]],
        datasets: list[str] | None = None,
        resolutions: list[str] | None = None,
        subset_size: int = 10,
    ) -> dict[str, list[BenchmarkResult]]:
        """
        Run full evaluation across multiple models and datasets.

        Args:
            models: List of (model_name, model) tuples
            datasets: List of datasets to evaluate (None for all supported)
            resolutions: List of resolutions to test (None for just "low")
            subset_size: Number of samples per dataset

        Returns:
            Dictionary mapping model names to their evaluation results
        """
        if datasets is None:
            datasets = self.loader.list_available_datasets()

        if resolutions is None:
            resolutions = ["low"]

        all_results = {}

        for model_name, model in models:
            model_results = []

            for resolution in resolutions:
                results = self.evaluate_model_on_datasets(
                    model=model,
                    model_name=f"{model_name}_{resolution}",
                    datasets=datasets,
                    subset_size=subset_size,
                    resolution=resolution,
                )
                model_results.extend(results)

            all_results[model_name] = model_results

        return all_results
