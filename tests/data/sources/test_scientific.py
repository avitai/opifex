"""Tests for the VTK mesh data source (PDEBench is tested in ``test_pdebench.py``).

Following datarax test patterns from:
- tests/sources/test_source_architecture.py (shared eager source tests)
- tests/sources/test_memory_source_module.py (full coverage patterns)
- tests/sources/test_hf_source.py (config-based source tests)
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from datarax.pipeline import Pipeline
from flax import nnx


def _assert_epoch_matches_step(make_loader: Callable[[], Pipeline]) -> None:
    """``for batch in loader`` serves one epoch: the records successive ``step()`` calls serve.

    datarax iterates a source that implements ``get_records`` inside a compiled session. The
    epoch holds ``ceil(len(source) / batch_size)`` batches and every record once, so its last
    batch is short; ``step()`` never runs out and completes that batch from the next epoch.
    """
    batches = list(make_loader())
    reference = make_loader()
    records = len(reference.source)
    steps = [reference.step() for _ in range(len(batches))]

    assert len(batches) == math.ceil(records / reference.batch_size)
    assert sum(len(batch[next(iter(batch))]) for batch in batches) == records
    for name in batches[0]:
        epoch = np.concatenate([np.asarray(batch[name]) for batch in batches])
        stepped = np.concatenate([np.asarray(batch[name]) for batch in steps])[:records]
        np.testing.assert_array_equal(epoch, stepped)


def _assert_shuffle_is_a_keyed_bijection(source: Any) -> None:
    """``record_indices_at`` orders an epoch by a permutation of the records drawn from ``key``.

    Each index costs O(1): the traced program holds no ``sort``, which a stored or per-batch
    ``jax.random.permutation`` needs (positive control below).
    """
    records = len(source)
    key = jax.random.key(1)
    order = np.asarray(source.record_indices_at(0, records, key))

    assert sorted(order.tolist()) == list(range(records))
    assert order.tolist() != list(range(records))
    np.testing.assert_array_equal(order, np.asarray(source.record_indices_at(0, records, key)))
    assert (
        order.tolist()
        != np.asarray(source.record_indices_at(0, records, jax.random.key(2))).tolist()
    )

    def primitives(fn: Callable[..., Any], *args: Any) -> set[str]:
        """Every primitive in ``fn``'s program, nested programs (pjit, loops, cond) included."""
        found: set[str] = set()
        pending = [jax.make_jaxpr(fn)(*args).jaxpr]
        while pending:
            program = pending.pop()
            for equation in program.eqns:
                found.add(equation.primitive.name)
                for value in equation.params.values():
                    for item in value if isinstance(value, (list, tuple)) else (value,):
                        inner = getattr(item, "jaxpr", item)
                        if hasattr(inner, "eqns"):
                            pending.append(inner)
        return found

    assert "sort" in primitives(lambda k: jax.random.permutation(k, records), key)
    assert "sort" not in primitives(
        lambda start, k: source.record_indices_at(start, 2, k), jnp.int32(0), key
    )


# =============================================================================
# VTKMeshConfig Validation Tests
# =============================================================================


class TestVTKMeshConfigValidation:
    """VTK config validation tests."""

    def test_config_requires_directory(self) -> None:
        """VTKMeshConfig with no directory should raise ValueError."""
        from opifex.data.sources.scientific import VTKMeshConfig

        with pytest.raises(ValueError, match="directory is required"):
            VTKMeshConfig()

    def test_config_valid(self, tmp_path: Path) -> None:
        """Valid VTKMeshConfig should not raise."""
        from opifex.data.sources.scientific import VTKMeshConfig

        config = VTKMeshConfig(
            directory=tmp_path,
            node_features=("velocity",),
        )
        assert config.file_pattern == "*.vtu"
        assert config.include_connectivity is True

    def test_config_is_frozen(self, tmp_path: Path) -> None:
        """VTKMeshConfig should be frozen."""
        from datarax.core.config import FrozenInstanceError

        from opifex.data.sources.scientific import VTKMeshConfig

        config = VTKMeshConfig(directory=tmp_path)
        with pytest.raises(FrozenInstanceError):
            config.directory = Path("/other")  # type: ignore[misc]


# =============================================================================
# VTKMeshSource Tests
# VTKMeshConfig validation tests don't need meshio installed (config-only)
# VTKMeshSource tests DO need meshio — use class-level marker below

_meshio_available = True
try:
    import meshio  # type: ignore[reportMissingImports]  # noqa: F401
except ImportError:
    _meshio_available = False


class TestVTKMeshSource:
    """VTK mesh source tests — uses meshio fixtures."""

    pytestmark = pytest.mark.skipif(
        not _meshio_available,
        reason="meshio not installed",
    )

    @pytest.fixture
    def vtu_directory(self, tmp_path: Path) -> Path:
        """Create synthetic VTU files for testing."""
        import meshio  # type: ignore[reportMissingImports]

        for i in range(3):
            points = (
                np.random.default_rng(i)
                .standard_normal((50, 3))
                .astype(
                    np.float32,
                )
            )
            cells: list[Any] = [("triangle", np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]]))]
            point_data: dict[str, Any] = {
                "velocity": np.random.default_rng(i + 10)
                .standard_normal(
                    (50, 3),
                )
                .astype(np.float32),
            }
            mesh = meshio.Mesh(
                points=points,
                cells=cells,
                point_data=point_data,
            )
            mesh.write(tmp_path / f"mesh_{i}.vtu")

        return tmp_path

    def test_loads_vtu_files(self, vtu_directory: Path) -> None:
        """VTKMeshSource should load VTU files from directory."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        assert len(source) == 3

    def test_element_has_node_positions(self, vtu_directory: Path) -> None:
        """Each element should have 'node_positions' as JAX array."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert "node_positions" in element
        assert isinstance(element["node_positions"], jax.Array)
        assert element["node_positions"].shape == (50, 3)

    def test_element_has_node_features(self, vtu_directory: Path) -> None:
        """Node features should be loaded and stacked."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert "node_features" in element
        assert isinstance(element["node_features"], jax.Array)

    def test_element_has_edge_index(self, vtu_directory: Path) -> None:
        """With include_connectivity=True, COO edge_index should be present."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
            include_connectivity=True,
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert "edge_index" in element
        assert isinstance(element["edge_index"], jax.Array)
        assert element["edge_index"].shape[0] == 2  # COO format: (2, num_edges)

    def test_is_datasource_module(self, vtu_directory: Path) -> None:
        """VTKMeshSource should be a DataSourceModule."""
        from datarax.core.data_source import DataSourceModule

        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        assert isinstance(source, DataSourceModule)

    def test_iteration(self, vtu_directory: Path) -> None:
        """Iteration should yield all meshes."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        elements = list(source)
        assert len(elements) == 3

    def test_getitem_out_of_bounds(self, vtu_directory: Path) -> None:
        """Out-of-bounds index should raise IndexError."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(
            directory=vtu_directory,
            node_features=("velocity",),
        )
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        with pytest.raises(IndexError):
            source[9999]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should produce source with length 0."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(directory=tmp_path)
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        assert len(source) == 0

    def test_element_has_node_mask(self, vtu_directory: Path) -> None:
        """Each padded element carries a node_mask flagging real (non-padded) nodes."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(directory=vtu_directory, node_features=("velocity",))
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert "node_mask" in element
        # All meshes are 50 nodes => max=50, no padding, mask all ones.
        assert element["node_mask"].shape == (50,)
        assert float(element["node_mask"].sum()) == 50.0

    def test_element_spec_and_get_records(self, vtu_directory: Path) -> None:
        """element_spec declares padded shapes; get_records stacks the meshes it names."""
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(directory=vtu_directory, node_features=("velocity",))
        source = VTKMeshSource(config, rngs=nnx.Rngs(0))
        spec = source.element_spec()
        assert spec["node_positions"].shape == (50, 3)
        batch = source.get_records(jnp.array([2, 0], dtype=jnp.int32))
        assert batch["node_positions"].shape == (2, 50, 3)
        assert batch["edge_index"].shape[1] == 2  # (batch, 2, max_edges)
        np.testing.assert_array_equal(
            np.asarray(batch["node_positions"][0]), np.asarray(source[2]["node_positions"])
        )

    def test_shuffle_is_a_keyed_bijection(self, vtu_directory: Path) -> None:
        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        config = VTKMeshConfig(directory=vtu_directory, node_features=("velocity",), shuffle=True)
        _assert_shuffle_is_a_keyed_bijection(VTKMeshSource(config, rngs=nnx.Rngs(0)))

    def test_varying_size_meshes_are_padded_with_masks(self, tmp_path: Path) -> None:
        """Ragged meshes pad to the dataset max; node_mask records each mesh's real node count."""
        import meshio  # type: ignore[reportMissingImports]

        from opifex.data.sources.scientific import VTKMeshConfig, VTKMeshSource

        sizes = [30, 50]
        for index, num_nodes in enumerate(sizes):
            points = np.random.default_rng(index).standard_normal((num_nodes, 3)).astype(np.float32)
            cells: list[Any] = [("triangle", np.array([[0, 1, 2], [2, 3, 4]]))]
            meshio.Mesh(points=points, cells=cells).write(tmp_path / f"mesh_{index}.vtu")

        source = VTKMeshSource(VTKMeshConfig(directory=tmp_path), rngs=nnx.Rngs(0))
        # Both meshes padded to max=50 nodes.
        assert source[0]["node_positions"].shape == (50, 3)
        assert source[1]["node_positions"].shape == (50, 3)
        # The 30-node mesh's mask has exactly 30 real nodes; the 50-node mesh has 50.
        assert float(source[0]["node_mask"].sum()) == 30.0
        assert float(source[1]["node_mask"].sum()) == 50.0

    def test_loader_returns_pipeline(self, vtu_directory: Path) -> None:
        """create_vtk_mesh_loader builds a datarax Pipeline whose step() yields batched meshes."""
        from opifex.data.sources.scientific import create_vtk_mesh_loader, VTKMeshConfig

        config = VTKMeshConfig(directory=vtu_directory, node_features=("velocity",))
        loader = create_vtk_mesh_loader(config, batch_size=2, seed=0)
        assert isinstance(loader, Pipeline)
        batch = loader.step()
        assert batch["node_positions"].shape[0] == 2

    def test_iterating_the_loader_matches_step(self, vtu_directory: Path) -> None:
        """A mesh loader iterates one epoch equal to successive step() calls."""
        from opifex.data.sources.scientific import create_vtk_mesh_loader, VTKMeshConfig

        config = VTKMeshConfig(directory=vtu_directory, node_features=("velocity",))
        _assert_epoch_matches_step(lambda: create_vtk_mesh_loader(config, batch_size=2, seed=0))
