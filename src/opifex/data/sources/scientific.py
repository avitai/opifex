"""Scientific simulation data sources for Opifex.

Provides eager-loading sources for HDF5-based scientific datasets
(PDEBench, VTK meshes) following the same patterns as HFEagerSource.

Datarax pattern reference:
    - Config: @dataclass inheriting StructuralConfig (frozen, validated)
    - Module: class extending DataSourceModule → StructuralModule → nnx.Module
    - Data: annotate JAX array storage with nnx.data()
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
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
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


@dataclass
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
        normalize: Whether to normalize data to [0, 1] (default: True)
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
    """Eager-loading source for PDEBench HDF5 datasets.

    Loads entire HDF5 dataset to JAX arrays at init, then provides
    pure-JAX iteration. Each element is a dict with:
        - "input": jax.Array of shape (input_steps, *spatial, channels)
        - "target": jax.Array of shape (output_steps, *spatial, channels)
        - "coordinates": dict of spatial/temporal grids (if available)

    Follows the same patterns as HFEagerSource:
        - All I/O at __init__, pure JAX after
        - Index-based __getitem__ and __iter__
        - get_batch() for batched access

    Example:
        >>> config = PDEBenchConfig(
        ...     file_path=Path("/data/pdebench/1D_Burgers.hdf5"),
        ...     dataset_name="1D_Burgers",
        ...     input_steps=10,
        ...     output_steps=10,
        ... )
        >>> source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        >>> batch = source.get_batch(32)
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

        # Normalize if requested (per-channel min-max to [0, 1])
        if config.normalize:
            inputs, targets = self._normalize(inputs, targets)

        # Convert to JAX arrays
        self.inputs = jnp.array(inputs, dtype=config.dtype)
        self.targets = jnp.array(targets, dtype=config.dtype)
        coord_dict = (
            {k: jnp.array(v, dtype=config.dtype) for k, v in coords.items()}
            if coords
            else None
        )
        self.coordinates = nnx.data(coord_dict)

        # Internal iteration state
        self._position = nnx.Variable(0)

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
    def _normalize(
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel min-max normalization to [0, 1]."""
        # Compute global min/max across both inputs and targets
        combined = np.concatenate(
            [
                inputs.reshape(-1, inputs.shape[-1]),
                targets.reshape(-1, targets.shape[-1]),
            ],
            axis=0,
        )
        ch_min = combined.min(axis=0, keepdims=True)
        ch_max = combined.max(axis=0, keepdims=True)
        denom = ch_max - ch_min
        denom = np.where(denom == 0, 1.0, denom)  # avoid division by zero

        # Reshape for broadcasting
        shape = [1] * (inputs.ndim - 1) + [inputs.shape[-1]]
        ch_min_r = ch_min.reshape(shape)
        denom_r = denom.reshape(shape)

        inputs_norm = (inputs - ch_min_r) / denom_r
        targets_norm = (targets - ch_min_r) / denom_r

        return inputs_norm, targets_norm

    def __len__(self) -> int:
        """Return the total number of data elements."""
        return self.inputs.shape[0]

    def __iter__(self) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        """Iterate over data elements."""
        self._position.value = 0
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        """Get element by index.

        Args:
            index: Index (supports negative indexing).

        Returns:
            Dict with 'input', 'target', and optionally 'coordinates'.

        Raises:
            IndexError: If index is out of bounds.
        """
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Index {index} out of range [0, {n})")

        element: dict[str, Any] = {
            "input": self.inputs[index],
            "target": self.targets[index],
        }
        if self.coordinates is not None:
            element["coordinates"] = self.coordinates

        return element

    def get_batch(
        self,
        batch_size: int,
        key: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Get a batch of data.

        Args:
            batch_size: Number of elements in the batch.
            key: Optional RNG key for stateless random sampling.
                If None, uses sequential sampling from current position.

        Returns:
            Batch dict with arrays having batch_size as first dimension.
        """
        n = len(self)
        if key is not None:
            # Stateless: random indices from key
            indices = jax.random.randint(
                key,
                shape=(batch_size,),
                minval=0,
                maxval=n,
            )
        else:
            # Sequential from current position
            pos = self._position.value
            indices = jnp.arange(pos, pos + batch_size) % n
            self._position.value = (pos + batch_size) % n

        batch: dict[str, Any] = {
            "input": self.inputs[indices],
            "target": self.targets[indices],
        }
        if self.coordinates is not None:
            batch["coordinates"] = self.coordinates

        return batch

    def reset(self, seed: int | None = None) -> None:
        """Reset iteration to the beginning.

        Args:
            seed: Optional seed (unused, kept for API compat).
        """
        self._position.value = 0


# =============================================================================
# VTK Mesh Configuration
# =============================================================================


@dataclass
class VTKMeshConfig(StructuralConfig):
    """Configuration for VTK unstructured mesh data source.

    Attributes:
        directory: Directory containing .vtu/.vtp files (pathlib.Path)
        file_pattern: Glob pattern for files (default: "*.vtu")
        node_features: Tuple of point data array names
        cell_features: Tuple of cell data array names
        include_connectivity: Whether to build edge lists (default: True)
    """

    directory: Path | None = None
    file_pattern: str = "*.vtu"
    node_features: tuple[str, ...] | None = None
    cell_features: tuple[str, ...] | None = None
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


class VTKMeshSource(DataSourceModule):
    """Eager-loading source for VTK unstructured mesh files.

    Each element is a dict with:
        - "node_positions": jax.Array of shape (num_nodes, 3)
        - "node_features": jax.Array of shape (num_nodes, F)
        - "edge_index": jax.Array of shape (2, num_edges)  # COO format
        - "cell_features": jax.Array of shape (num_cells, G) (optional)

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
                node_feat = self._extract_features(
                    config.node_features, mesh.point_data
                )
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

        self.meshes = nnx.data(mesh_list)

        # Iteration state
        self._position = nnx.Variable(0)

        logger.info(
            "Loaded %d meshes from %s",
            len(self.meshes),
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
            cell_data = (
                cell_block.data if hasattr(cell_block, "data") else cell_block[1]
            )
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

    def __len__(self) -> int:
        """Return number of meshes."""
        return len(self.meshes)

    def __iter__(self) -> Iterator[dict[str, Any]]:  # type: ignore[override]
        """Iterate over meshes."""
        self._position.value = 0
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        """Get mesh element by index.

        Args:
            index: Index (supports negative indexing).

        Raises:
            IndexError: If index is out of bounds.
        """
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError(f"Index {index} out of range [0, {n})")
        return self.meshes[index]

    def reset(self, seed: int | None = None) -> None:
        """Reset iteration state."""
        self._position.value = 0
