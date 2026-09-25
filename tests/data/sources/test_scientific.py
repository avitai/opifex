"""Tests for scientific data sources (PDEBench HDF5, VTK mesh).

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

from opifex.data.sources.scientific import create_pdebench_loader


# Lazy-import guards for optional deps
h5py = pytest.importorskip("h5py")


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
# HDF5 Fixtures
# =============================================================================


@pytest.fixture
def pdebench_hdf5_file(tmp_path: Path) -> Path:
    """Create a minimal synthetic PDEBench-style HDF5 file.

    Simulates `1D_Burgers` format: /tensor shape (N, T, X, C).
    Also includes /t and /x coordinate arrays.
    """
    file_path = tmp_path / "1D_Burgers_test.hdf5"
    n_samples, time_steps, spatial_x, channels = 20, 11, 64, 1
    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((n_samples, time_steps, spatial_x, channels)).astype(np.float32)
    t = np.linspace(0.0, 1.0, time_steps, dtype=np.float32)
    x = np.linspace(0.0, 1.0, spatial_x, dtype=np.float32)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("tensor", data=tensor)
        f.create_dataset("t", data=t)
        f.create_dataset("x", data=x)

    return file_path


@pytest.fixture
def pdebench_darcy_hdf5_file(tmp_path: Path) -> Path:
    """Create a synthetic PDEBench Darcy Flow HDF5—no time dimension.

    Simulates `2D_DarcyFlow` format: /tensor shape (N, X, Y, C).
    """
    file_path = tmp_path / "2D_DarcyFlow_test.hdf5"
    n_samples, spatial_x, spatial_y, channels = 12, 32, 32, 1
    rng = np.random.default_rng(99)
    tensor = rng.standard_normal((n_samples, spatial_x, spatial_y, channels)).astype(np.float32)

    with h5py.File(file_path, "w") as f:
        f.create_dataset("tensor", data=tensor)

    return file_path


@pytest.fixture
def pdebench_multifield_hdf5_file(tmp_path: Path) -> Path:
    """Create a synthetic HDF5 with separate field keys (CFD-style).

    Simulates `1D_CFD` format: /Vx, /density, /pressure as separate datasets.
    """
    file_path = tmp_path / "1D_CFD_test.hdf5"
    n_samples, time_steps, spatial_x = 10, 11, 64
    rng = np.random.default_rng(77)

    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            "Vx",
            data=rng.standard_normal((n_samples, time_steps, spatial_x)).astype(np.float32),
        )
        f.create_dataset(
            "density",
            data=rng.standard_normal((n_samples, time_steps, spatial_x)).astype(np.float32),
        )
        f.create_dataset(
            "pressure",
            data=rng.standard_normal((n_samples, time_steps, spatial_x)).astype(np.float32),
        )
        f.create_dataset(
            "t",
            data=np.linspace(0, 1, time_steps, dtype=np.float32),
        )

    return file_path


# =============================================================================
# PDEBenchConfig Validation Tests
# =============================================================================


class TestPDEBenchConfigValidation:
    """Config validation tests following datarax StructuralConfig patterns."""

    def test_config_requires_file_path(self) -> None:
        """PDEBenchConfig with no file_path should raise ValueError."""
        from opifex.data.sources.scientific import PDEBenchConfig

        with pytest.raises(ValueError, match="file_path is required"):
            PDEBenchConfig(dataset_name="1D_Burgers")

    def test_config_requires_dataset_name(self) -> None:
        """PDEBenchConfig with no dataset_name should raise ValueError."""
        from opifex.data.sources.scientific import PDEBenchConfig

        with pytest.raises(ValueError, match="dataset_name is required"):
            PDEBenchConfig(file_path=Path("nonexistent.hdf5"))

    def test_config_invalid_split(self) -> None:
        """Split must be 'train' or 'test'."""
        from opifex.data.sources.scientific import PDEBenchConfig

        with pytest.raises(ValueError, match="split must be"):
            PDEBenchConfig(
                file_path=Path("nonexistent.hdf5"),
                dataset_name="1D_Burgers",
                split="val",
            )

    def test_config_invalid_train_split_ratio(self) -> None:
        """train_split must be in (0, 1)."""
        from opifex.data.sources.scientific import PDEBenchConfig

        with pytest.raises(ValueError, match="train_split must be in"):
            PDEBenchConfig(
                file_path=Path("nonexistent.hdf5"),
                dataset_name="1D_Burgers",
                train_split=1.5,
            )

    def test_config_invalid_input_steps(self) -> None:
        """input_steps must be >= 1."""
        from opifex.data.sources.scientific import PDEBenchConfig

        with pytest.raises(ValueError, match="input_steps must be"):
            PDEBenchConfig(
                file_path=Path("nonexistent.hdf5"),
                dataset_name="1D_Burgers",
                input_steps=0,
            )

    def test_config_invalid_output_steps(self) -> None:
        """output_steps must be >= 1."""
        from opifex.data.sources.scientific import PDEBenchConfig

        with pytest.raises(ValueError, match="output_steps must be"):
            PDEBenchConfig(
                file_path=Path("nonexistent.hdf5"),
                dataset_name="1D_Burgers",
                output_steps=0,
            )

    def test_config_valid_defaults(self) -> None:
        """Valid config with defaults should not raise."""
        from opifex.data.sources.scientific import PDEBenchConfig

        config = PDEBenchConfig(
            file_path=Path("nonexistent.hdf5"),
            dataset_name="1D_Burgers",
        )
        assert config.train_split == 0.9
        assert config.split == "train"
        assert config.input_steps == 1
        assert config.output_steps == 1
        assert config.normalize is True

    def test_config_is_frozen(self) -> None:
        """StructuralConfig should be frozen after init."""
        from opifex.data.sources.scientific import PDEBenchConfig

        config = PDEBenchConfig(
            file_path=Path("nonexistent.hdf5"),
            dataset_name="1D_Burgers",
        )
        from datarax.core.config import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            config.train_split = 0.5  # type: ignore[misc]


# =============================================================================
# PDEBenchSource Initialization Tests
# =============================================================================


class TestPDEBenchSourceInit:
    """Initialization tests — datarax pattern: all I/O at __init__."""

    def test_loads_hdf5_at_init(self, pdebench_hdf5_file: Path) -> None:
        """Data should be loaded to JAX arrays at initialization."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            input_steps=1,
            output_steps=1,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))

        assert len(source) > 0
        # Data should be JAX arrays (HFEagerSource pattern)
        element = source[0]
        assert isinstance(element["input"], jax.Array)
        assert isinstance(element["target"], jax.Array)

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        """Loading a nonexistent HDF5 file should raise FileNotFoundError."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=tmp_path / "nonexistent.hdf5",
            dataset_name="1D_Burgers",
        )
        with pytest.raises(FileNotFoundError):
            PDEBenchSource(config, rngs=nnx.Rngs(0))

    def test_is_datasource_module(self, pdebench_hdf5_file: Path) -> None:
        """PDEBenchSource should be a DataSourceModule subclass."""
        from datarax.core.data_source import DataSourceModule

        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        assert isinstance(source, DataSourceModule)

    def test_is_nnx_module(self, pdebench_hdf5_file: Path) -> None:
        """PDEBenchSource should be an nnx.Module."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        assert isinstance(source, nnx.Module)


# =============================================================================
# PDEBenchSource Train/Test Split Tests
# =============================================================================


class TestPDEBenchSplit:
    """Train/test split tests."""

    def test_train_split_size(self, pdebench_hdf5_file: Path) -> None:
        """Train split should contain ~90% of sliding window samples."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            train_split=0.9,
            split="train",
            input_steps=1,
            output_steps=1,
        )
        train_source = PDEBenchSource(config, rngs=nnx.Rngs(0))

        # Total samples = 20. Train = int(20 * 0.9) = 18
        assert len(train_source) > 0

    def test_test_split_size(self, pdebench_hdf5_file: Path) -> None:
        """Test split should contain ~10% of sliding window samples."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            train_split=0.9,
            split="test",
            input_steps=1,
            output_steps=1,
        )
        test_source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        assert len(test_source) > 0

    def test_train_test_no_overlap(self, pdebench_hdf5_file: Path) -> None:
        """Train and test splits should not overlap in sample indices."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        kwargs = {
            "file_path": pdebench_hdf5_file,
            "dataset_name": "1D_Burgers",
            "train_split": 0.5,  # 50/50 for clearer test
            "input_steps": 1,
            "output_steps": 1,
        }
        train_source = PDEBenchSource(
            PDEBenchConfig(**kwargs, split="train"),
            rngs=nnx.Rngs(0),
        )
        test_source = PDEBenchSource(
            PDEBenchConfig(**kwargs, split="test"),
            rngs=nnx.Rngs(0),
        )
        total = len(train_source) + len(test_source)
        # Total should equal number of valid windows from all 20 samples
        assert total > 0

    def test_split_is_deterministic(self, pdebench_hdf5_file: Path) -> None:
        """Same config should produce same data."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        s1 = PDEBenchSource(config, rngs=nnx.Rngs(0))
        s2 = PDEBenchSource(config, rngs=nnx.Rngs(0))
        np.testing.assert_array_equal(
            np.array(s1[0]["input"]),
            np.array(s2[0]["input"]),
        )


# =============================================================================
# PDEBenchSource Sliding Window Tests
# =============================================================================


class TestPDEBenchSlidingWindow:
    """Sliding window input/target pairing tests."""

    def test_input_shape(self, pdebench_hdf5_file: Path) -> None:
        """Input should have (input_steps, *spatial, channels) shape."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            input_steps=3,
            output_steps=2,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert element["input"].shape[0] == 3  # input_steps
        assert element["input"].shape[-1] == 1  # channels

    def test_target_shape(self, pdebench_hdf5_file: Path) -> None:
        """Target should have (output_steps, *spatial, channels) shape."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            input_steps=3,
            output_steps=2,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert element["target"].shape[0] == 2  # output_steps

    def test_many_steps_reduces_samples(self, pdebench_hdf5_file: Path) -> None:
        """More input+output steps = fewer valid sliding windows."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config_small = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            input_steps=1,
            output_steps=1,
        )
        config_large = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            input_steps=5,
            output_steps=5,
        )
        s_small = PDEBenchSource(config_small, rngs=nnx.Rngs(0))
        s_large = PDEBenchSource(config_large, rngs=nnx.Rngs(0))
        assert len(s_small) >= len(s_large)

    def test_darcy_no_time_dimension(self, pdebench_darcy_hdf5_file: Path) -> None:
        """Darcy Flow (no time dimension) should work with input_steps=1, output_steps=1."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_darcy_hdf5_file,
            dataset_name="2D_DarcyFlow",
            input_steps=1,
            output_steps=1,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        assert len(source) > 0
        element = source[0]
        assert "input" in element
        assert "target" in element


# =============================================================================
# PDEBenchSource Normalization Tests
# =============================================================================


class TestPDEBenchNormalization:
    """Normalization is a composable datarax MapOperator, not baked into the source arrays."""

    def test_normalize_operator_is_map_operator(self, pdebench_hdf5_file: Path) -> None:
        """normalize=True yields a datarax MapOperator; normalize=False yields None."""
        from datarax.operators.map_operator import MapOperator

        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        on = PDEBenchSource(
            PDEBenchConfig(file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", normalize=True),
            rngs=nnx.Rngs(0),
        )
        off = PDEBenchSource(
            PDEBenchConfig(
                file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", normalize=False
            ),
            rngs=nnx.Rngs(0),
        )
        assert isinstance(on.normalize_operator(rngs=nnx.Rngs(0)), MapOperator)
        assert off.normalize_operator(rngs=nnx.Rngs(0)) is None

    def test_loader_normalizes_to_unit_range(self, pdebench_hdf5_file: Path) -> None:
        """A normalize=True loader rescales input/target into [0, 1] (per-channel min-max)."""
        from opifex.data.sources.scientific import PDEBenchConfig

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", normalize=True
        )
        loader = create_pdebench_loader(config, batch_size=4)
        batch = loader.step()
        for field in ("input", "target"):
            values = np.array(batch[field])
            assert np.isfinite(values).all()
            assert values.min() >= -1e-5
            assert values.max() <= 1.0 + 1e-5

    def test_raw_source_arrays_are_unnormalized(self, pdebench_hdf5_file: Path) -> None:
        """The source itself holds raw data — normalization happens in the pipeline stage."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", normalize=True
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        # Raw standard-normal data has negative values; min-max would not.
        assert float(np.array(source.inputs).min()) < 0.0


# =============================================================================
# PDEBenchSource Random Access Tests (datarax pattern)
# =============================================================================


class TestPDEBenchRandomAccess:
    """Random access via __getitem__ — follows datarax __getitem__ pattern."""

    def test_getitem_returns_dict(self, pdebench_hdf5_file: Path) -> None:
        """__getitem__ should return a dict with 'input' and 'target'."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert isinstance(element, dict)
        assert "input" in element
        assert "target" in element

    def test_getitem_out_of_bounds(self, pdebench_hdf5_file: Path) -> None:
        """Out-of-bounds index should raise IndexError."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        with pytest.raises(IndexError):
            source[99999]

    def test_getitem_negative_index(self, pdebench_hdf5_file: Path) -> None:
        """Negative index should work (Python convention)."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[-1]
        assert "input" in element

    def test_getitem_all_elements_accessible(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """Every index from 0..len-1 should be accessible."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        for i in range(len(source)):
            element = source[i]
            assert element["input"].shape[0] == config.input_steps

    def test_getitem_returns_jax_arrays(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """Return values should be JAX arrays on default device."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert isinstance(element["input"], jax.Array)
        assert isinstance(element["target"], jax.Array)


# =============================================================================
# PDEBenchSource Batch Retrieval Tests (datarax pattern)
# =============================================================================


class TestPDEBenchBatch:
    """Datarax contract: stateless get_records + element_spec (Pipeline-drivable)."""

    def test_get_records_returns_the_records_at_the_indices(self, pdebench_hdf5_file: Path) -> None:
        """get_records gathers exactly the records named, in the order named."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(file_path=pdebench_hdf5_file, dataset_name="1D_Burgers")
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        indices = jnp.array([3, 0, 5], dtype=jnp.int32)
        batch = source.get_records(indices)
        assert isinstance(batch["input"], jax.Array)
        for row, index in enumerate(indices.tolist()):
            np.testing.assert_array_equal(np.asarray(batch["input"][row]), source[index]["input"])
            np.testing.assert_array_equal(np.asarray(batch["target"][row]), source[index]["target"])

    def test_get_records_traceable_under_jit(self, pdebench_hdf5_file: Path) -> None:
        """get_records accepts traced indices (composes with nnx.scan / jit)."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(file_path=pdebench_hdf5_file, dataset_name="1D_Burgers")
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(source)

        @jax.jit
        def fetch(state: nnx.State, indices: jax.Array) -> jax.Array:
            return nnx.merge(graphdef, state).get_records(indices)["input"]

        out = fetch(state, jnp.array([1, 2], dtype=jnp.int32))
        assert out.shape[0] == 2

    def test_shuffle_is_a_keyed_bijection(self, pdebench_hdf5_file: Path) -> None:
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", shuffle=True
        )
        _assert_shuffle_is_a_keyed_bijection(PDEBenchSource(config, rngs=nnx.Rngs(0)))

    def test_element_spec(self, pdebench_hdf5_file: Path) -> None:
        """element_spec describes one element's input/target shapes and dtypes."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            input_steps=3,
            output_steps=2,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        spec = source.element_spec()
        assert spec["input"].shape[0] == 3
        assert spec["target"].shape[0] == 2
        assert spec["input"].dtype == source.inputs.dtype


# =============================================================================
# PDEBenchSource Iteration Tests (datarax pattern)
# =============================================================================


class TestPDEBenchIteration:
    """Iteration tests — follows datarax __iter__ pattern."""

    def test_iteration_yields_all_elements(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """Iterating should yield exactly len(source) elements."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        elements = list(source)
        assert len(elements) == len(source)

    def test_iteration_elements_are_dicts(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """Each iterated element should be a dict."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        for element in source:
            assert isinstance(element, dict)
            assert "input" in element
            assert "target" in element
            break  # just check first

    def test_iteration_is_stateless_and_repeatable(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """Iteration carries no internal counter — re-iterating yields the same elements."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        first_pass = [np.array(e["input"]) for e in source]
        second_pass = [np.array(e["input"]) for e in source]
        assert len(first_pass) == len(second_pass)
        np.testing.assert_array_equal(first_pass[0], second_pass[0])


# =============================================================================
# PDEBenchSource Coordinates Tests
# =============================================================================


class TestPDEBenchCoordinates:
    """Coordinate grids are domain metadata, exposed via the ``coordinates`` attribute."""

    def test_coordinates_loaded(self, pdebench_hdf5_file: Path) -> None:
        """When /x and /t exist in HDF5, they are accessible on source.coordinates."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        assert source.coordinates is not None
        assert "x" in source.coordinates
        assert "t" in source.coordinates


class TestPDEBenchLoader:
    """create_pdebench_loader builds a datarax Pipeline over the source."""

    def test_loader_returns_pipeline(self, pdebench_hdf5_file: Path) -> None:
        """The loader returns a datarax Pipeline whose step() yields batched dicts."""
        from opifex.data.sources.scientific import PDEBenchConfig

        config = PDEBenchConfig(file_path=pdebench_hdf5_file, dataset_name="1D_Burgers")
        loader = create_pdebench_loader(config, batch_size=4)
        assert isinstance(loader, Pipeline)
        batch = loader.step()
        assert batch["input"].shape[0] == 4
        assert batch["target"].shape[0] == 4

    def test_loader_iterates_full_dataset(self, pdebench_hdf5_file: Path) -> None:
        """Successive step() calls advance through the dataset (pipeline-driven position)."""
        from opifex.data.sources.scientific import PDEBenchConfig

        config = PDEBenchConfig(file_path=pdebench_hdf5_file, dataset_name="1D_Burgers")
        loader = create_pdebench_loader(config, batch_size=4)
        b0 = loader.step()
        b1 = loader.step()
        assert b0["input"].shape == b1["input"].shape

    def test_iterating_the_loader_matches_step(self, pdebench_hdf5_file: Path) -> None:
        """A normalizing loader iterates one epoch equal to successive step() calls."""
        from opifex.data.sources.scientific import PDEBenchConfig

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", normalize=True
        )
        _assert_epoch_matches_step(lambda: create_pdebench_loader(config, batch_size=4))

    def test_a_shuffled_loader_serves_each_record_once_per_epoch(
        self, pdebench_hdf5_file: Path
    ) -> None:
        """A shuffled epoch holds every record once, in an order other than the stored one."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        # Unnormalized, so the served rows are the stored records themselves.
        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file, dataset_name="1D_Burgers", shuffle=True, normalize=False
        )
        stored = PDEBenchSource(config, rngs=nnx.Rngs(0)).inputs
        served = np.concatenate(
            [np.asarray(batch["input"]) for batch in create_pdebench_loader(config, batch_size=4)]
        )
        rows = [row.tobytes() for row in np.asarray(stored)]
        assert sorted(row.tobytes() for row in served) == sorted(rows)
        assert [row.tobytes() for row in served] != rows


# =============================================================================
# PDEBenchSource Multifield Tests
# =============================================================================


class TestPDEBenchMultifield:
    """Tests for datasets with separate field keys (e.g. 1D_CFD)."""

    def test_multifield_loading(
        self,
        pdebench_multifield_hdf5_file: Path,
    ) -> None:
        """Source should load and stack separate field datasets."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_multifield_hdf5_file,
            dataset_name="1D_CFD",
            field_keys=("Vx", "density", "pressure"),
            input_steps=1,
            output_steps=1,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        # Stacked fields should give channels = 3
        assert element["input"].shape[-1] == 3


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
        loader = create_vtk_mesh_loader(config, batch_size=2)
        assert isinstance(loader, Pipeline)
        batch = loader.step()
        assert batch["node_positions"].shape[0] == 2

    def test_iterating_the_loader_matches_step(self, vtu_directory: Path) -> None:
        """A mesh loader iterates one epoch equal to successive step() calls."""
        from opifex.data.sources.scientific import create_vtk_mesh_loader, VTKMeshConfig

        config = VTKMeshConfig(directory=vtu_directory, node_features=("velocity",))
        _assert_epoch_matches_step(lambda: create_vtk_mesh_loader(config, batch_size=2))
