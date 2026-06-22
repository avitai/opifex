"""Scientific simulation data sources for Opifex.

Provides eager-loading sources for HDF5-based scientific datasets
(PDEBench, VTK meshes) following the same patterns as HFEagerSource.

Datarax pattern reference:
    - Config: @dataclass inheriting StructuralConfig (frozen, validated)
    - Module: class extending DataSourceModule → StructuralModule → nnx.Module
    - Data: annotate JAX array storage with nnx.data(value)
    - I/O: all file I/O at __init__, pure JAX after
    - Deps: lazy-import optional deps (h5py, pyvista)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from datarax.core.config import MapOperatorConfig, StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.operators.map_operator import MapOperator
from datarax.pipeline import Pipeline
from flax import nnx


logger = logging.getLogger(__name__)

# =============================================================================
# Known PDEBench multi-field datasets (separate HDF5 keys per field)
# =============================================================================

_MULTIFIELD_DATASETS: dict[str, tuple[str, ...]] = {
    "1D_CFD": ("Vx", "density", "pressure"),
    "2D_CFD": ("Vx", "Vy", "density", "pressure"),
    "3D_CFD": ("Vx", "Vy", "Vz", "density", "pressure"),
}


# =============================================================================
# PDEBench Configuration
# =============================================================================


@dataclass(frozen=True)
class PDEBenchConfig(StructuralConfig):
    """Configuration for PDEBench HDF5 data source.

    PDEBench datasets are HDF5 files with structure:
        /tensor: shape (N, T, X[, Y[, Z]], C) — simulation trajectories
        /x, /y, /z: spatial coordinate grids (optional)
        /t: time coordinates (optional)

    Attributes:
        file_path: Path to HDF5 file (use pathlib.Path, not str — SWE Rule 13)
        dataset_name: Name of the PDE problem (e.g., "1D_Burgers", "2D_DarcyFlow")
        train_split: Fraction of data for training (default: 0.9)
        split: Which split to load ("train" or "test")
        input_steps: Number of input time steps (default: 1)
        output_steps: Number of output time steps to predict (default: 1)
        normalize: Whether to attach a per-channel min-max normalisation operator
            (a datarax ``MapOperator``) in :func:`create_pdebench_loader` (default: True)
        shuffle: Whether ``get_batch_at`` shuffles via the supplied key (default: False)
        dtype: JAX dtype for arrays (default: jnp.float32)
        field_keys: Tuple of HDF5 dataset keys for multi-field datasets (optional).
            If provided, these keys are stacked along the last axis.
            If None, autodetermined from dataset_name or defaults to ("tensor",).
    """

    file_path: Path | None = None
    dataset_name: str | None = None
    train_split: float = 0.9
    split: str = "train"
    input_steps: int = 1
    output_steps: int = 1
    normalize: bool = True
    shuffle: bool = False
    dtype: Any = jnp.float32
    field_keys: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate and freeze config (SWE Rule 6: fail-fast)."""
        object.__setattr__(self, "stochastic", False)
        super().__post_init__()
        if self.file_path is None:
            raise ValueError("file_path is required for PDEBenchConfig")
        if self.dataset_name is None:
            raise ValueError("dataset_name is required for PDEBenchConfig")
        if self.split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{self.split}'")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if self.input_steps < 1:
            raise ValueError(f"input_steps must be >= 1, got {self.input_steps}")
        if self.output_steps < 1:
            raise ValueError(f"output_steps must be >= 1, got {self.output_steps}")


# =============================================================================
# PDEBench Data Source
# =============================================================================


class PDEBenchSource(DataSourceModule):
    """Eager-loading datarax source for PDEBench HDF5 datasets.

    Reads the HDF5 file at ``__init__``, splits train/test, and performs the PDE-specific
    input/target time-window pairing (the one piece datarax has no operator for). After that it is
    a standard datarax :class:`~datarax.core.data_source.DataSourceModule`: it implements the
    stateless :meth:`get_batch_at` / :meth:`element_spec` contract so it can be driven by a datarax
    :class:`~datarax.pipeline.Pipeline`. Each element is a dict with:
        - "input": jax.Array of shape (input_steps, *spatial, channels)
        - "target": jax.Array of shape (output_steps, *spatial, channels)

    Coordinate grids (if present) are exposed via :attr:`coordinates`. Normalisation is *not* baked
    into the stored arrays — it is a datarax :class:`~datarax.operators.map_operator.MapOperator`
    obtained from :meth:`normalize_operator` and attached as a pipeline stage (see
    :func:`create_pdebench_loader`), keeping the source pure data and transforms composable.

    Example:
        >>> config = PDEBenchConfig(
        ...     file_path=Path("/data/pdebench/1D_Burgers.hdf5"),
        ...     dataset_name="1D_Burgers",
        ...     input_steps=10,
        ...     output_steps=10,
        ... )
        >>> loader = create_pdebench_loader(config, batch_size=32)
        >>> batch = loader.step()
        >>> batch["input"].shape   # (32, 10, 1024, 1)
    """

    def __init__(
        self,
        config: PDEBenchConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load HDF5 data, split into train/test, create input/target pairs."""
        super().__init__(config, rngs=rngs, name=name)

        file_path = Path(config.file_path)  # type: ignore[arg-type]
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")

        # Lazy import h5py (optional dependency, SWE Rule 13)
        try:
            import h5py
        except ImportError as e:
            raise ImportError(
                "h5py is required for PDEBenchSource. "
                "Install with: pip install opifex[scientific-data]"
            ) from e

        # Determine field keys
        field_keys = config.field_keys
        if field_keys is None:
            dataset_name: str = config.dataset_name or ""
            field_keys = _MULTIFIELD_DATASETS.get(
                dataset_name,
                ("tensor",),
            )

        # Read HDF5 file (all I/O at init)
        with h5py.File(file_path, "r") as f:
            raw_data = self._read_hdf5_data(f, field_keys)
            coords = self._read_coordinates(f)

        # Split into train/test along sample axis (axis=0)
        n_samples = raw_data.shape[0]
        split_idx = int(n_samples * config.train_split)

        data = raw_data[:split_idx] if config.split == "train" else raw_data[split_idx:]

        # Determine if dataset has time dimension
        has_time = self._has_time_dimension(data, config.dataset_name)

        # Create sliding window pairs or direct input/target
        if has_time:
            inputs, targets = self._create_sliding_windows(
                data,
                config.input_steps,
                config.output_steps,
            )
        else:
            # No time dimension (e.g. DarcyFlow): each sample is both input and target
            inputs = data
            targets = data

        # Store raw arrays; normalisation is a composable MapOperator, not baked in here.
        self.inputs = jnp.array(inputs, dtype=config.dtype)
        self.targets = jnp.array(targets, dtype=config.dtype)
        coord_dict = (
            {k: jnp.array(v, dtype=config.dtype) for k, v in coords.items()} if coords else None
        )
        self.coordinates = nnx.data(coord_dict)
        self._shuffle = config.shuffle

        # Per-channel min-max statistics for the optional normalisation operator.
        norm_min, norm_scale = self._compute_norm_stats(self.inputs, self.targets)
        self._norm_min = nnx.data(norm_min)
        self._norm_scale = nnx.data(norm_scale)

        logger.info(
            "Loaded %s: %d samples (%s split)",
            config.dataset_name,
            len(self),
            config.split,
        )

    @staticmethod
    def _read_hdf5_data(
        f: Any,
        field_keys: tuple[str, ...],
    ) -> np.ndarray:
        """Read and stack field data from HDF5 file."""
        if len(field_keys) == 1 and field_keys[0] == "tensor":
            # Standard single-tensor format
            return np.array(f["tensor"])

        # Multi-field: stack along last axis
        arrays = []
        for key in field_keys:
            arr = np.array(f[key])
            if arr.ndim < 3:
                # Ensure at least (N, T, X) shape
                arr = arr[..., np.newaxis]
            arrays.append(arr)

        return np.stack(arrays, axis=-1)

    @staticmethod
    def _read_coordinates(f: Any) -> dict[str, np.ndarray]:
        """Read optional coordinate arrays (/x, /y, /z, /t)."""
        coords: dict[str, np.ndarray] = {}
        for key in ("x", "y", "z", "t"):
            if key in f:
                coords[key] = np.array(f[key])
        return coords

    @staticmethod
    def _has_time_dimension(
        data: np.ndarray,
        dataset_name: str | None,
    ) -> bool:
        """Determine if dataset has a time dimension.

        DarcyFlow and similar steady-state problems have shape (N, X, Y, C)
        with no time axis. Time-varying problems have shape (N, T, X, ..., C).
        """
        if dataset_name and "Darcy" in dataset_name:
            return False
        # Heuristic: if ndim >= 4 and second dim > 1, likely has time
        return data.ndim >= 4

    @staticmethod
    def _create_sliding_windows(
        data: np.ndarray,
        input_steps: int,
        output_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create sliding window input/target pairs over time axis.

        Args:
            data: shape (N, T, *spatial, C)
            input_steps: number of input time steps
            output_steps: number of output time steps

        Returns:
            inputs: shape (N_windows, input_steps, *spatial, C)
            targets: shape (N_windows, output_steps, *spatial, C)
        """
        n_samples, time_steps = data.shape[0], data.shape[1]
        window_size = input_steps + output_steps
        n_windows_per_sample = max(0, time_steps - window_size + 1)

        if n_windows_per_sample == 0:
            raise ValueError(
                f"input_steps ({input_steps}) + output_steps ({output_steps}) "
                f"= {window_size} exceeds time steps ({time_steps})"
            )

        all_inputs = []
        all_targets = []

        for sample_idx in range(n_samples):
            for t_start in range(n_windows_per_sample):
                inp = data[sample_idx, t_start : t_start + input_steps]
                tgt = data[
                    sample_idx,
                    t_start + input_steps : t_start + input_steps + output_steps,
                ]
                all_inputs.append(inp)
                all_targets.append(tgt)

        return np.stack(all_inputs), np.stack(all_targets)

    @staticmethod
    def _compute_norm_stats(
        inputs: jax.Array,
        targets: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Per-channel min-max statistics ``(min, scale)`` over inputs and targets.

        Returned arrays have shape ``(channels,)`` so they broadcast against both a single element
        ``(steps, *spatial, C)`` and a batch ``(size, steps, *spatial, C)`` on the last axis.
        ``scale`` is ``max - min`` with zeros replaced by one to avoid division by zero.
        """
        channels = inputs.shape[-1]
        combined = jnp.concatenate(
            [inputs.reshape(-1, channels), targets.reshape(-1, channels)], axis=0
        )
        ch_min = combined.min(axis=0)
        ch_scale = combined.max(axis=0) - ch_min
        ch_scale = jnp.where(ch_scale == 0, 1.0, ch_scale)
        return ch_min, ch_scale

    def normalize_operator(self, *, rngs: nnx.Rngs | None = None) -> MapOperator | None:
        """Return the per-channel min-max normalisation as a datarax ``MapOperator``.

        The operator rescales the ``"input"`` and ``"target"`` fields (leaving ``"coordinates"``
        untouched) to ``[0, 1]`` using the statistics computed at load. Returns ``None`` when the
        config has ``normalize=False`` so the loader attaches no normalisation stage.
        """
        if not self.config.normalize:  # type: ignore[attr-defined]
            return None
        ch_min, ch_scale = self._norm_min, self._norm_scale

        def rescale(value: jax.Array, key: jax.Array) -> jax.Array:  # noqa: ARG001 - deterministic
            return (value - ch_min) / ch_scale

        # ``precomputed_stats={}`` keeps the operator on the static-stats path (the normalisation
        # constants are baked into ``rescale``); no per-batch statistics are needed.
        return MapOperator(
            MapOperatorConfig(
                subtree={"input": None, "target": None},
                precomputed_stats={},
            ),
            fn=rescale,
            rngs=rngs,
        )

    def __len__(self) -> int:
        """Return the total number of windowed elements."""
        return self.inputs.shape[0]

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Per-element ``{"input", "target"}`` shapes/dtypes (datarax contract)."""
        return {
            "input": jax.ShapeDtypeStruct(self.inputs.shape[1:], self.inputs.dtype),
            "target": jax.ShapeDtypeStruct(self.targets.shape[1:], self.targets.dtype),
        }

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Stateless indexed batch access for ``Pipeline`` iteration (mirrors ``MemorySource``).

        Returns ``size`` records from logical position ``start`` (wrapping at the end). When the
        source is configured with ``shuffle=True`` and a ``key`` is supplied, indices are drawn
        from a per-call permutation of ``key`` (deterministic for a given ``(start, size, key)``).
        Internal state is never mutated, so the call is JAX-traceable under ``nnx.scan``.
        """
        n = len(self)
        start_arr = jnp.asarray(start, dtype=jnp.int32)
        indices = (start_arr + jnp.arange(size, dtype=jnp.int32)) % jnp.int32(n)
        if self._shuffle and key is not None:
            indices = jax.random.permutation(key, n)[indices]
        return {
            "input": jnp.take(self.inputs, indices, axis=0, mode="wrap"),
            "target": jnp.take(self.targets, indices, axis=0, mode="wrap"),
        }

    def __getitem__(self, index: int) -> dict[str, jax.Array]:  # type: ignore[override]
        """Return the raw ``{"input", "target"}`` element at ``index`` (supports negatives)."""
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Index {index} out of range [0, {n})")
        return {"input": self.inputs[index], "target": self.targets[index]}

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:  # type: ignore[override]
        """Iterate raw elements in order (debug/inspection; training uses the ``Pipeline``)."""
        for i in range(len(self)):
            yield self[i]


def create_pdebench_loader(
    config: PDEBenchConfig,
    *,
    batch_size: int,
    seed: int = 0,
) -> Pipeline:
    """Build a datarax ``Pipeline`` over a :class:`PDEBenchSource` (canonical loader pattern).

    Mirrors the sibling loaders (``rmd17``/``qh9``): the source supplies the windowed
    ``{"input", "target"}`` records via its ``get_batch_at`` contract and the pipeline drives
    batched iteration. When ``config.normalize`` is set, the per-channel min-max
    :class:`~datarax.operators.map_operator.MapOperator` from
    :meth:`PDEBenchSource.normalize_operator` is attached as the single transform stage.

    Args:
        config: PDEBench source configuration.
        batch_size: Records per batch.
        seed: Seed for the source's shuffle stream and the pipeline rngs.

    Returns:
        A configured datarax ``Pipeline`` (drive it with ``.step()`` or ``.scan()``).
    """
    source = PDEBenchSource(config, rngs=nnx.Rngs(seed))
    normalize_op = source.normalize_operator(rngs=nnx.Rngs(seed))
    stages: list[nnx.Module] = [normalize_op] if normalize_op is not None else []
    return Pipeline(
        source=source,
        stages=stages,
        batch_size=batch_size,
        rngs=nnx.Rngs(seed),
    )


# =============================================================================
# VTK Mesh Configuration
# =============================================================================


@dataclass(frozen=True)
class VTKMeshConfig(StructuralConfig):
    """Configuration for VTK unstructured mesh data source.

    Attributes:
        directory: Directory containing .vtu/.vtp files (pathlib.Path)
        file_pattern: Glob pattern for files (default: "*.vtu")
        node_features: Tuple of point data array names
        cell_features: Tuple of cell data array names
        include_connectivity: Whether to build edge lists (default: True)
        shuffle: Whether ``get_batch_at`` shuffles via the supplied key (default: False)
    """

    directory: Path | None = None
    file_pattern: str = "*.vtu"
    node_features: tuple[str, ...] | None = None
    cell_features: tuple[str, ...] | None = None
    shuffle: bool = False
    include_connectivity: bool = True

    def __post_init__(self) -> None:
        """Validate and freeze config."""
        object.__setattr__(self, "stochastic", False)
        super().__post_init__()
        if self.directory is None:
            raise ValueError("directory is required for VTKMeshConfig")


# =============================================================================
# VTK Mesh Data Source
# =============================================================================


def _stack_padded(meshes: list[dict[str, Any]], key: str, axis: int, target: int) -> jax.Array:
    """Pad each mesh's ``key`` array to ``target`` along ``axis`` (zeros) and stack over meshes."""

    def pad(array: jax.Array) -> jax.Array:
        widths = [(0, 0)] * array.ndim
        widths[axis] = (0, target - array.shape[axis])
        return jnp.pad(array, widths)

    return jnp.stack([pad(mesh[key]) for mesh in meshes])


def _stack_masks(meshes: list[dict[str, Any]], key: str, axis: int, target: int) -> jax.Array:
    """Stack per-mesh masks (1.0 over each mesh's real extent along ``axis``, 0.0 for padding)."""
    return jnp.stack(
        [(jnp.arange(target) < mesh[key].shape[axis]).astype(jnp.float32) for mesh in meshes]
    )


class VTKMeshSource(DataSourceModule):
    """Eager-loading datarax source for VTK unstructured mesh files.

    Meshes are ragged (per-file node/edge counts differ), so at load each ragged axis is padded to
    the dataset maximum and a boolean mask is carried, giving uniform arrays that satisfy the
    datarax static-shape contract (``get_batch_at`` / ``element_spec``) and JIT. Each element is a
    dict with (``max_nodes``/``max_edges``/``max_cells`` are the dataset maxima):
        - "node_positions": jax.Array of shape (max_nodes, 3)
        - "node_mask": jax.Array of shape (max_nodes,) — 1.0 for real nodes, 0.0 for padding
        - "node_features": jax.Array of shape (max_nodes, F) (if ``node_features`` configured)
        - "edge_index": jax.Array of shape (2, max_edges)  # COO (if ``include_connectivity``)
        - "edge_mask": jax.Array of shape (max_edges,) (with ``edge_index``)
        - "cell_features"/"cell_mask": shape (max_cells, G)/(max_cells,) (if ``cell_features``)

    Drive batched iteration with :func:`create_vtk_mesh_loader` (a datarax ``Pipeline``).

    Implementation notes:
        - Uses meshio for VTK I/O (lazy import)
        - Converts connectivity to COO edge_index for GNN consumption
    """

    def __init__(
        self,
        config: VTKMeshConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ) -> None:
        """Load VTK files from directory."""
        super().__init__(config, rngs=rngs, name=name)

        directory = Path(config.directory)  # type: ignore[arg-type]

        # Lazy import meshio
        try:
            import meshio  # type: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "meshio is required for VTKMeshSource. "
                "Install with: pip install opifex[scientific-data]"
            ) from e

        # Find and load all matching files
        files = sorted(directory.glob(config.file_pattern))
        mesh_list: list[dict[str, Any]] = []

        for file_path in files:
            mesh = meshio.read(file_path)
            element: dict[str, Any] = {}

            # Node positions
            element["node_positions"] = jnp.array(
                mesh.points,
                dtype=jnp.float32,
            )

            # Node features
            if config.node_features:
                node_feat = self._extract_features(config.node_features, mesh.point_data)
                if node_feat is not None:
                    element["node_features"] = node_feat

            # Cell features
            if config.cell_features:
                cell_data = {k: v[0] for k, v in mesh.cell_data.items()}
                cell_feat = self._extract_features(config.cell_features, cell_data)
                if cell_feat is not None:
                    element["cell_features"] = cell_feat

            # Edge connectivity (COO format)
            if config.include_connectivity and mesh.cells:
                edge_index = self._cells_to_coo(mesh.cells)
                element["edge_index"] = jnp.array(
                    edge_index,
                    dtype=jnp.int32,
                )

            mesh_list.append(element)

        # Meshes are ragged (per-file node/edge counts differ). Pad each ragged axis to the
        # dataset maximum and carry boolean masks, so the records stack into uniform arrays that
        # satisfy the datarax static-shape contract (get_batch_at / element_spec) and JIT.
        self._fields = nnx.data(self._pad_and_stack(mesh_list))
        self._shuffle = config.shuffle

        logger.info(
            "Loaded %d meshes from %s",
            len(self),
            directory,
        )

    @staticmethod
    def _extract_features(
        feature_names: Sequence[str],
        data_dict: dict[str, Any],
    ) -> jax.Array | None:
        """Extract and concatenate named features from a data dictionary.

        Args:
            feature_names: Names of features to extract.
            data_dict: Mapping of feature name to numpy array.

        Returns:
            Concatenated feature array, or None if no features found.
        """
        features = []
        for feat_name in feature_names:
            if feat_name in data_dict:
                feat = np.array(data_dict[feat_name])
                if feat.ndim == 1:
                    feat = feat[:, np.newaxis]
                features.append(feat)
        if not features:
            return None
        return jnp.array(np.concatenate(features, axis=-1), dtype=jnp.float32)

    @staticmethod
    def _cells_to_coo(cells: list[Any]) -> np.ndarray:
        """Convert cell connectivity to COO edge_index format.

        Creates bidirectional edges from cell connectivity.

        Returns:
            np.ndarray of shape (2, num_edges)
        """
        edges_set: set[tuple[int, int]] = set()

        for cell_block in cells:
            # cell_block is a CellBlock with .data array
            cell_data = cell_block.data if hasattr(cell_block, "data") else cell_block[1]
            for cell in cell_data:
                # Add edges between consecutive vertices of each cell
                for i in range(len(cell)):
                    for j in range(i + 1, len(cell)):
                        edges_set.add((int(cell[i]), int(cell[j])))
                        edges_set.add((int(cell[j]), int(cell[i])))

        if not edges_set:
            return np.zeros((2, 0), dtype=np.int32)

        edges = np.array(sorted(edges_set), dtype=np.int32)
        return edges.T  # (2, num_edges)

    @staticmethod
    def _pad_and_stack(meshes: list[dict[str, Any]]) -> dict[str, jax.Array]:
        """Pad each mesh's ragged arrays to the dataset maximum and stack into uniform arrays.

        Node-indexed fields (``node_positions``, ``node_features``) are padded along axis 0 to
        ``max_nodes`` and gain a ``node_mask``; ``edge_index`` is padded along axis 1 to
        ``max_edges`` and gains an ``edge_mask``; ``cell_features`` is padded to ``max_cells`` with
        a ``cell_mask``. Returns a dict of stacked arrays each with a leading mesh axis. An empty
        mesh list yields an empty dict (``len == 0``).
        """
        if not meshes:
            return {}

        fields: dict[str, jax.Array] = {}
        max_nodes = max(m["node_positions"].shape[0] for m in meshes)
        fields["node_positions"] = _stack_padded(meshes, "node_positions", 0, max_nodes)
        fields["node_mask"] = _stack_masks(meshes, "node_positions", 0, max_nodes)

        if all("node_features" in m for m in meshes):
            fields["node_features"] = _stack_padded(meshes, "node_features", 0, max_nodes)

        if all("edge_index" in m for m in meshes):
            max_edges = max(m["edge_index"].shape[1] for m in meshes)
            fields["edge_index"] = _stack_padded(meshes, "edge_index", 1, max_edges)
            fields["edge_mask"] = _stack_masks(meshes, "edge_index", 1, max_edges)

        if all("cell_features" in m for m in meshes):
            max_cells = max(m["cell_features"].shape[0] for m in meshes)
            fields["cell_features"] = _stack_padded(meshes, "cell_features", 0, max_cells)
            fields["cell_mask"] = _stack_masks(meshes, "cell_features", 0, max_cells)

        return fields

    def __len__(self) -> int:
        """Return the number of meshes."""
        if not self._fields:
            return 0
        return next(iter(self._fields.values())).shape[0]

    def element_spec(self) -> dict[str, jax.ShapeDtypeStruct]:
        """Per-mesh padded field shapes/dtypes, including masks (datarax contract)."""
        return {
            key: jax.ShapeDtypeStruct(value.shape[1:], value.dtype)
            for key, value in self._fields.items()
        }

    def get_batch_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Stateless indexed batch of padded meshes (datarax contract, mirrors ``MemorySource``).

        Returns ``size`` meshes from logical position ``start`` (wrapping). With ``shuffle=True``
        and a ``key`` the indices come from a per-call permutation of ``key``. Stateless and
        JAX-traceable, so it composes with ``Pipeline`` / ``nnx.scan``.
        """
        n = len(self)
        indices = (jnp.asarray(start, dtype=jnp.int32) + jnp.arange(size, dtype=jnp.int32)) % (
            jnp.int32(n)
        )
        if self._shuffle and key is not None:
            indices = jax.random.permutation(key, n)[indices]
        return {
            key_name: jnp.take(value, indices, axis=0, mode="wrap")
            for key_name, value in self._fields.items()
        }

    def __getitem__(self, index: int) -> dict[str, jax.Array]:  # type: ignore[override]
        """Return the padded fields (+ masks) for the mesh at ``index`` (supports negatives)."""
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Index {index} out of range [0, {n})")
        return {key: value[index] for key, value in self._fields.items()}

    def __iter__(self) -> Iterator[dict[str, jax.Array]]:  # type: ignore[override]
        """Iterate padded meshes in order (training uses the ``Pipeline``)."""
        for i in range(len(self)):
            yield self[i]


def create_vtk_mesh_loader(
    config: VTKMeshConfig,
    *,
    batch_size: int,
    seed: int = 0,
) -> Pipeline:
    """Build a datarax ``Pipeline`` over a :class:`VTKMeshSource` (canonical loader pattern).

    The source supplies padded, masked mesh records via its ``get_batch_at`` contract and the
    pipeline drives batched iteration; no transform stages are attached by default. Drive it with
    ``.step()`` or ``.scan()``.

    Args:
        config: VTK mesh source configuration.
        batch_size: Meshes per batch.
        seed: Seed for the source's shuffle stream and the pipeline rngs.

    Returns:
        A configured datarax ``Pipeline``.
    """
    source = VTKMeshSource(config, rngs=nnx.Rngs(seed))
    return Pipeline(source=source, stages=[], batch_size=batch_size, rngs=nnx.Rngs(seed))
