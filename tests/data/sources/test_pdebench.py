"""PDEBench files are read as PDEBench's own loader reads them.

The fixtures are small files in PDEBench's published layouts: the 1D Burgers file holds
``tensor`` of shape ``(N, T, X)`` with ``x-coordinate`` and ``t-coordinate`` (``T + 1`` long),
the Darcy flow file ``nu`` ``(N, X, Y)`` and ``tensor`` ``(N, 1, X, Y)`` with ``x-`` and
``y-coordinate``, a compressible-flow file one dataset per field, and a grouped ``.h5`` file one
group per sample with ``data`` ``(T, x..., V)`` and ``grid/{x, y, t}``. Every expectation is the
rule of ``pdebench/models/fno/utils.py`` (``FNODatasetSingle``), stated in the test.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from opifex.data.sources import create_pdebench_loader, load_pdebench, PDEBenchConfig


N, T, X, Y = 10, 6, 8, 4


def _write_burgers(path: Path) -> np.ndarray:
    tensor = np.arange(N * T * X, dtype=np.float32).reshape(N, T, X)
    with h5py.File(path, "w") as f:
        f["tensor"] = tensor
        f["x-coordinate"] = np.linspace(0.0, 1.0, X, dtype=np.float32)
        f["t-coordinate"] = np.linspace(0.0, 1.0, T + 1, dtype=np.float32)
    return tensor


def _write_darcy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    nu = rng.random((N, X, Y), dtype=np.float32)
    tensor = rng.random((N, 1, X, Y), dtype=np.float32)
    with h5py.File(path, "w") as f:
        f["nu"] = nu
        f["tensor"] = tensor
        f["x-coordinate"] = np.linspace(0.0, 1.0, X, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0.0, 1.0, Y, dtype=np.float32)
    return nu, tensor


def _write_cfd_2d(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    fields = {
        name: rng.random((N, T, X, Y), dtype=np.float32)
        for name in ("Vx", "Vy", "density", "pressure")
    }
    with h5py.File(path, "w") as f:
        for name, value in fields.items():
            f[name] = value
        f["x-coordinate"] = np.linspace(0.0, 1.0, X, dtype=np.float32)
        f["y-coordinate"] = np.linspace(0.0, 1.0, Y, dtype=np.float32)
        f["t-coordinate"] = np.linspace(0.0, 1.0, T + 1, dtype=np.float32)
    return fields


def _write_grouped(path: Path, spatial: tuple[int, ...], channels: int) -> list[np.ndarray]:
    rng = np.random.default_rng(2)
    samples = []
    with h5py.File(path, "w") as f:
        for index in range(N):
            data = rng.random((T, *spatial, channels), dtype=np.float32)
            group = f.create_group(f"{index:04d}")
            group["data"] = data
            for axis, size in zip(("x", "y"), spatial, strict=False):
                group[f"grid/{axis}"] = np.linspace(0.0, 1.0, size, dtype=np.float32)
            group["grid/t"] = np.linspace(0.0, 1.0, T, dtype=np.float32)
            samples.append(data)
    return samples


def _config(**fields) -> PDEBenchConfig:
    """A config for these six-step fixtures, two of them input unless a test says otherwise.

    PDEBench's default ``initial_step`` of 10 exceeds the fixtures' time steps.
    """
    return PDEBenchConfig(**{"initial_step": 2, **fields})


class TestLayouts:
    def test_a_scalar_field_is_space_then_time_then_channel(self, tmp_path):
        """``tensor`` ``(N, T, X)`` becomes ``[N, X, T, 1]`` (utils.py: "convert to [x1, ..., xd, t, v]")."""
        tensor = _write_burgers(tmp_path / "1D_Burgers_Sols_Nu1.0.hdf5")
        data = load_pdebench(
            _config(file_path=tmp_path / "1D_Burgers_Sols_Nu1.0.hdf5", split="train")
        )
        # The test split is the first int(10 * 0.1) = 1 sample, so the train split starts at row 1.
        assert data.targets.shape == (N - 1, X, T, 1)
        np.testing.assert_array_equal(data.targets[0, :, :, 0], tensor[1].T)
        np.testing.assert_array_equal(data.inputs, data.targets[..., :2, :])
        assert data.grid.shape == (X, 1)

    def test_darcy_flow_stacks_the_coefficient_and_the_solution_along_time(self, tmp_path):
        nu, tensor = _write_darcy(tmp_path / "2D_DarcyFlow_beta1.0_Train.hdf5")
        data = load_pdebench(
            _config(file_path=tmp_path / "2D_DarcyFlow_beta1.0_Train.hdf5", initial_step=1)
        )
        assert data.targets.shape == (N - 1, X, Y, 2, 1)
        np.testing.assert_array_equal(data.targets[:, :, :, 0, 0], nu[1:])
        np.testing.assert_array_equal(data.targets[:, :, :, 1, 0], tensor[1:, 0])
        np.testing.assert_array_equal(data.inputs[..., 0, 0], nu[1:])
        assert data.grid.shape == (X, Y, 2)

    def test_compressible_flow_channels_are_density_pressure_then_velocity(self, tmp_path):
        fields = _write_cfd_2d(
            tmp_path / "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
        )
        data = load_pdebench(
            _config(
                file_path=tmp_path / "2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5"
            )
        )
        assert data.targets.shape == (N - 1, X, Y, T, 4)
        for channel, name in enumerate(("density", "pressure", "Vx", "Vy")):
            np.testing.assert_array_equal(
                data.targets[..., channel], np.moveaxis(fields[name][1:], 1, -1)
            )

    @pytest.mark.parametrize(("spatial", "channels"), [((X, Y), 3), ((X,), 1)])
    def test_a_grouped_file_stacks_each_samples_data(self, tmp_path, spatial, channels):
        """``data`` ``(T, x..., V)`` per group becomes ``[N, x..., T, V]``, groups in sorted order."""
        samples = _write_grouped(tmp_path / "2D_rdb_NA_NA.h5", spatial, channels)
        data = load_pdebench(_config(file_path=tmp_path / "2D_rdb_NA_NA.h5", initial_step=2))
        assert data.targets.shape == (N - 1, *spatial, T, channels)
        np.testing.assert_array_equal(data.targets[0], np.moveaxis(samples[1], 0, -2))
        assert data.grid.shape == (*spatial, len(spatial))

    def test_a_file_with_no_pdebench_layout_is_refused(self, tmp_path):
        path = tmp_path / "ns_incom_inhom_2d_512-0.h5"
        with h5py.File(path, "w") as f:
            f["velocity"] = np.zeros((1, T, X, X, 2), np.float32)
        with pytest.raises(ValueError, match="incompressible Navier-Stokes"):
            load_pdebench(_config(file_path=path))


class TestSplitsAndStrides:
    def test_the_test_split_is_the_first_test_ratio_of_the_samples(self, tmp_path):
        tensor = _write_burgers(tmp_path / "burgers.hdf5")
        test = load_pdebench(
            _config(file_path=tmp_path / "burgers.hdf5", split="test", test_ratio=0.3)
        )
        train = load_pdebench(
            _config(file_path=tmp_path / "burgers.hdf5", split="train", test_ratio=0.3)
        )
        np.testing.assert_array_equal(test.targets[..., 0], np.moveaxis(tensor[:3], 1, -1))
        np.testing.assert_array_equal(train.targets[..., 0], np.moveaxis(tensor[3:], 1, -1))

    def test_the_sample_cap_applies_after_the_batch_stride(self, tmp_path):
        """Rows 0, 2, 4, ... are kept, the first six of them, and the test split is their first."""
        tensor = _write_burgers(tmp_path / "burgers.hdf5")
        config = {
            "file_path": tmp_path / "burgers.hdf5",
            "reduced_batch": 2,
            "num_samples_max": 4,
            "test_ratio": 0.25,
        }
        test = load_pdebench(_config(**config, split="test"))
        train = load_pdebench(_config(**config, split="train"))
        np.testing.assert_array_equal(test.targets[..., 0], np.moveaxis(tensor[[0]], 1, -1))
        np.testing.assert_array_equal(train.targets[..., 0], np.moveaxis(tensor[[2, 4, 6]], 1, -1))

    def test_strides_reduce_the_time_and_space_resolution(self, tmp_path):
        tensor = _write_burgers(tmp_path / "burgers.hdf5")
        data = load_pdebench(
            _config(
                file_path=tmp_path / "burgers.hdf5",
                reduced_resolution=2,
                reduced_resolution_t=3,
                initial_step=1,
            )
        )
        np.testing.assert_array_equal(
            data.targets[..., 0], np.moveaxis(tensor[1:, ::3, ::2], 1, -1)
        )
        assert data.grid.shape == (X // 2, 1)

    def test_darcy_flow_never_strides_its_two_steps(self, tmp_path):
        _write_darcy(tmp_path / "darcy.hdf5")
        data = load_pdebench(
            _config(file_path=tmp_path / "darcy.hdf5", initial_step=1, reduced_resolution_t=2)
        )
        assert data.targets.shape[-2] == 2

    def test_a_grouped_files_test_split_is_its_first_groups(self, tmp_path):
        samples = _write_grouped(tmp_path / "swe.h5", (X, Y), 1)
        test = load_pdebench(
            _config(file_path=tmp_path / "swe.h5", split="test", test_ratio=0.2, initial_step=1)
        )
        np.testing.assert_array_equal(
            test.targets, np.stack([np.moveaxis(s, 0, -2) for s in samples[:2]])
        )


class TestRefusals:
    def test_more_input_steps_than_the_file_holds_are_refused(self, tmp_path):
        _write_darcy(tmp_path / "darcy.hdf5")
        with pytest.raises(ValueError, match="initial_step 2 leaves no step to predict of the 2"):
            load_pdebench(_config(file_path=tmp_path / "darcy.hdf5", initial_step=2))

    @pytest.mark.parametrize(
        "bad",
        [
            {"split": "val"},
            {"test_ratio": 1.0},
            {"initial_step": 0},
            {"reduced_batch": 0},
            {"num_samples_max": 0},
        ],
    )
    def test_a_config_that_selects_no_well_defined_data_is_refused(self, tmp_path, bad):
        (field,) = bad
        with pytest.raises(ValueError, match=field):
            PDEBenchConfig(file_path=tmp_path / "x.hdf5", **bad)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pdebench(_config(file_path=tmp_path / "missing.hdf5"))


class TestLoader:
    def test_the_loader_serves_input_and_target_records(self, tmp_path):
        _write_burgers(tmp_path / "burgers.hdf5")
        loader = create_pdebench_loader(
            _config(file_path=tmp_path / "burgers.hdf5", initial_step=2),
            batch_size=3,
            seed=0,
        )
        batch = next(iter(loader))
        assert set(batch) == {"input", "target"}
        assert batch["input"].shape == (3, X, 2, 1)
        assert batch["target"].shape == (3, X, T, 1)
