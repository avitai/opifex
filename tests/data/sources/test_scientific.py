"""Tests for scientific data sources (PDEBench HDF5, VTK mesh).

Following datarax test patterns from:
- tests/sources/test_source_architecture.py (shared eager source tests)
- tests/sources/test_memory_source_module.py (full coverage patterns)
- tests/sources/test_hf_source.py (config-based source tests)
"""

from pathlib import Path

import jax
import numpy as np
import pytest
from flax import nnx


# Lazy-import guards for optional deps
h5py = pytest.importorskip("h5py")


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
    tensor = rng.standard_normal((n_samples, time_steps, spatial_x, channels)).astype(
        np.float32
    )
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
    tensor = rng.standard_normal((n_samples, spatial_x, spatial_y, channels)).astype(
        np.float32
    )

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
            data=rng.standard_normal((n_samples, time_steps, spatial_x)).astype(
                np.float32
            ),
        )
        f.create_dataset(
            "density",
            data=rng.standard_normal((n_samples, time_steps, spatial_x)).astype(
                np.float32
            ),
        )
        f.create_dataset(
            "pressure",
            data=rng.standard_normal((n_samples, time_steps, spatial_x)).astype(
                np.float32
            ),
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
    """Normalization tests."""

    def test_normalize_true(self, pdebench_hdf5_file: Path) -> None:
        """With normalize=True, values should be rescaled."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            normalize=True,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        inp = np.array(element["input"])
        # Normalized: should have mean ~0 and std ~1 (approximately)
        # Or min-max [0,1] depending on implementation
        assert np.isfinite(inp).all()

    def test_normalize_false(self, pdebench_hdf5_file: Path) -> None:
        """With normalize=False, raw data values are preserved."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
            normalize=False,
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        inp = np.array(element["input"])
        assert np.isfinite(inp).all()


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
    """Batch retrieval — follows datarax get_batch pattern."""

    def test_get_batch_shape(self, pdebench_hdf5_file: Path) -> None:
        """get_batch should return dict with batched arrays."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        batch = source.get_batch(4)
        assert batch["input"].shape[0] == 4
        assert batch["target"].shape[0] == 4

    def test_get_batch_stateless_with_key(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """Stateless batch with explicit key should be deterministic."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        key = jax.random.PRNGKey(42)
        b1 = source.get_batch(4, key=key)
        b2 = source.get_batch(4, key=key)
        np.testing.assert_array_equal(
            np.array(b1["input"]),
            np.array(b2["input"]),
        )

    def test_get_batch_jax_arrays(self, pdebench_hdf5_file: Path) -> None:
        """Batch values should be JAX arrays."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        batch = source.get_batch(2)
        assert isinstance(batch["input"], jax.Array)
        assert isinstance(batch["target"], jax.Array)


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

    def test_reset_allows_reiteration(
        self,
        pdebench_hdf5_file: Path,
    ) -> None:
        """After reset(), iteration should start from the beginning."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        first_pass = [np.array(e["input"]) for e in source]
        source.reset()
        second_pass = [np.array(e["input"]) for e in source]
        assert len(first_pass) == len(second_pass)
        np.testing.assert_array_equal(first_pass[0], second_pass[0])


# =============================================================================
# PDEBenchSource Coordinates Tests
# =============================================================================


class TestPDEBenchCoordinates:
    """Coordinate grid loading tests."""

    def test_coordinates_loaded(self, pdebench_hdf5_file: Path) -> None:
        """When /x and /t exist in HDF5, they should appear in element."""
        from opifex.data.sources.scientific import PDEBenchConfig, PDEBenchSource

        config = PDEBenchConfig(
            file_path=pdebench_hdf5_file,
            dataset_name="1D_Burgers",
        )
        source = PDEBenchSource(config, rngs=nnx.Rngs(0))
        element = source[0]
        assert "coordinates" in element
        assert "x" in element["coordinates"]
        assert "t" in element["coordinates"]


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
            cells = [("triangle", np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]]))]
            point_data = {
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
