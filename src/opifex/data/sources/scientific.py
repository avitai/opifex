"""Scientific simulation data sources for Opifex.

Provides an eager-loading source for VTK meshes following the same patterns as
HFEagerSource. The PDEBench HDF5 source is :mod:`opifex.data.sources.pdebench`.

Datarax pattern reference:
    - Config: @dataclass inheriting StructuralConfig (frozen, validated)
    - Module: class extending DataSourceModule → StructuralModule → nnx.Module
    - Data: annotate JAX array storage with nnx.data(value)
    - I/O: all file I/O at __init__, pure JAX after
    - Deps: lazy-import optional deps (meshio)
"""

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from datarax.core.config import StructuralConfig
from datarax.core.data_source import DataSourceModule
from datarax.pipeline import Pipeline
from datarax.sources import resolve_wrapped_indices
from flax import nnx


logger = logging.getLogger(__name__)


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
        shuffle: Whether ``record_indices_at`` shuffles by the supplied key (default: False)
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
    datarax static-shape contract (``get_records`` / ``element_spec``) and JIT. Each element is a
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
        # satisfy the datarax static-shape contract (get_records / element_spec) and JIT.
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

    def record_indices_at(
        self,
        start: int | jax.Array,
        size: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """The index of each mesh at positions ``start .. start + size`` (datarax contract).

        In stored order, or with ``shuffle=True`` and a ``key`` the keyed bijection datarax
        shuffles by: O(1) per mesh and no stored permutation, so a batch costs O(size).
        """
        return resolve_wrapped_indices(start, size, len(self), self._shuffle, key)

    def get_records(self, indices: jax.Array) -> dict[str, jax.Array]:
        """The padded meshes (fields and masks) at ``indices`` (datarax contract).

        Stateless and JAX-traceable, so ``Pipeline`` iterates it in a compiled session and
        ``step()`` / ``scan`` drive it.
        """
        return {
            key_name: jnp.take(value, indices, axis=0) for key_name, value in self._fields.items()
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
    seed: int,
) -> Pipeline:
    """Build a datarax ``Pipeline`` over a :class:`VTKMeshSource` (canonical loader pattern).

    The source supplies padded, masked mesh records via its ``get_records`` contract and the
    pipeline drives batched iteration; no transform stages are attached by default. Iterate it
    (``for batch in loader``), or drive it with ``.step()`` or ``.scan()``.

    Args:
        config: VTK mesh source configuration.
        batch_size: Meshes per batch.
        seed: Seed for the source's shuffle stream and the pipeline rngs.

    Returns:
        A configured datarax ``Pipeline``.
    """
    source = VTKMeshSource(config, rngs=nnx.Rngs(seed))
    return Pipeline(source=source, stages=[], batch_size=batch_size, rngs=nnx.Rngs(seed))
