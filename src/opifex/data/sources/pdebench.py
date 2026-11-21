"""PDEBench HDF5 data, read as PDEBench lays it out and splits it.

An optional integration: it imports ``h5py``, which the ``scientific-data`` and ``pdebench``
extras install, at the top, and ``opifex.data.sources`` loads this module only when one of its
names is first used.

The rules follow PDEBench's own loader (``pdebench/models/fno/utils.py``, ``FNODatasetSingle``;
that file is under a noncommercial licence, so its rules are restated here, not its code):

- A file's layout decides how it is read. A single ``.hdf5`` file holds either ``tensor``,
  ``(N, T, x...)`` for a scalar equation, with ``nu`` beside a ``(N, 1, x, y)`` ``tensor`` for
  Darcy flow, or one dataset per field for compressible flow (``density``, ``pressure``,
  ``Vx``[, ``Vy``, ``Vz``], in that channel order). A grouped ``.h5`` file holds one group per
  sample (``"0000"``, ...), each with ``data`` of shape ``(T, x..., V)`` and ``grid/{x, y, t}``.
- Every field is returned as ``[N, x..., t, V]``: space, then time, then channels. Darcy
  flow's ``nu`` and solution are stacked along that time axis, so its ``initial_step`` is 1.
- ``reduced_batch``, ``reduced_resolution_t`` and ``reduced_resolution`` stride the sample,
  time and space axes (Darcy flow's two "time" steps are never strided). ``num_samples_max``
  caps the strided samples; the test split is the first ``int(num_samples * test_ratio)`` of
  them and the train split the rest.
- A record is ``(target[..., :initial_step, :], target)``: the input steps and the whole
  trajectory, with the grid shared by every record.

Strides are applied through h5py's hyperslab selection, so only the requested rows and points
are read, where PDEBench reads each file whole (the 1D Burgers file is 8.2 GB). PDEBench applies
no normalisation; fit :class:`opifex.core.normalization.GaussianNormalizer` on the train split
and reuse it for the test split.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from substrax.rng import rngs_from_seed


_CFD_FIELDS: tuple[str, ...] = ("density", "pressure", "Vx", "Vy", "Vz")
"""The compressible-flow fields in PDEBench's channel order; a d-dimensional file holds 2 + d."""

_COORDINATES: tuple[str, ...] = ("x-coordinate", "y-coordinate", "z-coordinate")
_GROUP_AXES: tuple[str, ...] = ("x", "y", "z")
_SPLITS: tuple[str, ...] = ("train", "test")


@dataclass(frozen=True, kw_only=True)
class PDEBenchConfig:
    """Which PDEBench file to read, at which resolution, and which split.

    Attributes:
        file_path: The ``.hdf5`` or ``.h5`` file, named as PDEBench publishes it.
        split: ``"train"`` or ``"test"``.
        initial_step: Time steps given as input (PDEBench's default is 10; Darcy flow needs 1).
        test_ratio: Share of the samples in the test split (PDEBench's default, 0.1).
        num_samples_max: Cap on the samples, after ``reduced_batch``; -1 keeps every sample.
        reduced_batch: Stride over the samples.
        reduced_resolution: Stride over each spatial axis.
        reduced_resolution_t: Stride over the time axis.
    """

    file_path: Path
    split: str = "train"
    initial_step: int = 10
    test_ratio: float = 0.1
    num_samples_max: int = -1
    reduced_batch: int = 1
    reduced_resolution: int = 1
    reduced_resolution_t: int = 1

    def __post_init__(self) -> None:
        """Refuse a split, ratio, step or stride that selects no well-defined data.

        Raises:
            ValueError: If ``split`` is not ``"train"`` or ``"test"``, ``test_ratio`` is outside
                ``[0, 1)``, ``initial_step`` or a stride is below one, or ``num_samples_max`` is
                neither -1 nor positive.
        """
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {self.split!r}")
        if not 0.0 <= self.test_ratio < 1.0:
            raise ValueError(f"test_ratio must lie in [0, 1), got {self.test_ratio}")
        for name in ("initial_step", "reduced_batch", "reduced_resolution", "reduced_resolution_t"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        if self.num_samples_max == 0 or self.num_samples_max < -1:
            raise ValueError(f"num_samples_max must be -1 or positive, got {self.num_samples_max}")


@dataclass(frozen=True, kw_only=True)
class PDEBenchData:
    """One split of a PDEBench file.

    Attributes:
        inputs: ``(n, x..., initial_step, V)``, the first ``initial_step`` steps of each target.
        targets: ``(n, x..., t, V)``, each sample's whole trajectory.
        grid: ``(x..., d)``, the spatial coordinates shared by every sample.
    """

    inputs: np.ndarray
    targets: np.ndarray
    grid: np.ndarray


def _dataset(node: h5py.Group, name: str) -> h5py.Dataset:
    """Return ``node[name]``, refusing an entry that is not a dataset."""
    item = node[name]
    if not isinstance(item, h5py.Dataset):
        raise TypeError(f"{node.name}/{name} is a {type(item).__name__}, not a dataset")
    return item


def _group(node: h5py.Group, name: str) -> h5py.Group:
    """Return ``node[name]``, refusing an entry that is not a group."""
    item = node[name]
    if not isinstance(item, h5py.Group):
        raise TypeError(f"{node.name}/{name} is a {type(item).__name__}, not a group")
    return item


def _split_rows(n_rows: int, config: PDEBenchConfig) -> slice:
    """Return the file rows a split reads: stride, then cap, then the first share is test."""
    n_strided = len(range(0, n_rows, config.reduced_batch))
    n_samples = n_strided if config.num_samples_max < 0 else min(config.num_samples_max, n_strided)
    test_count = int(n_samples * config.test_ratio)
    first, stop = (0, test_count) if config.split == "test" else (test_count, n_samples)
    step = config.reduced_batch
    return slice(first * step, stop * step, step)


def _space(stride: int, spatial_ndim: int) -> tuple[slice, ...]:
    """Return one strided slice per spatial axis."""
    return (slice(None, None, stride),) * spatial_ndim


def _coordinate_grid(node: h5py.Group, names: tuple[str, ...], stride: int) -> np.ndarray:
    """Return the strided meshgrid of the named 1-D coordinate datasets, stacked last."""
    axes = [np.asarray(_dataset(node, name)[::stride], dtype=np.float32) for name in names]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _read_tensor(
    file: h5py.File, rows: slice, config: PDEBenchConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Read a scalar equation's ``tensor``, or Darcy flow's ``nu`` and ``tensor``."""
    tensor = _dataset(file, "tensor")
    spatial_ndim = tensor.ndim - 2
    space = _space(config.reduced_resolution, spatial_ndim)
    if "nu" in file:
        solution = np.moveaxis(np.asarray(tensor[(rows, slice(None), *space)]), 1, -1)
        coefficient = np.asarray(_dataset(file, "nu")[(rows, *space)])[..., None]
        field = np.concatenate([coefficient, solution], axis=-1)
    else:
        time = slice(None, None, config.reduced_resolution_t)
        field = np.moveaxis(np.asarray(tensor[(rows, time, *space)]), 1, -1)
    grid = _coordinate_grid(file, _COORDINATES[:spatial_ndim], config.reduced_resolution)
    return field[..., None].astype(np.float32), grid


def _read_fields(
    file: h5py.File, rows: slice, config: PDEBenchConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Read a compressible-flow file's per-field datasets as channels."""
    spatial_ndim = _dataset(file, "density").ndim - 2
    selection = (
        rows,
        slice(None, None, config.reduced_resolution_t),
        *_space(config.reduced_resolution, spatial_ndim),
    )
    channels = [
        np.moveaxis(np.asarray(_dataset(file, name)[selection]), 1, -1)
        for name in _CFD_FIELDS[: 2 + spatial_ndim]
    ]
    grid = _coordinate_grid(file, _COORDINATES[:spatial_ndim], config.reduced_resolution)
    return np.stack(channels, axis=-1).astype(np.float32), grid


def _read_groups(
    file: h5py.File, path: Path, config: PDEBenchConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Read a grouped ``.h5`` file: one ``data`` of shape ``(T, x..., V)`` per sample group."""
    keys = sorted(file.keys())
    first = file[keys[0]] if keys else None
    if not isinstance(first, h5py.Group) or "data" not in first:
        raise ValueError(
            f"{path.name}: a grouped .h5 file holds a 'data' dataset per sample group, and this "
            f"one does not (first groups: {keys[:3]}). PDEBench's incompressible Navier-Stokes "
            "files have another layout, which PDEBench's loaders do not read either."
        )
    spatial_ndim = _dataset(first, "data").ndim - 2
    selection = (
        slice(None, None, config.reduced_resolution_t),
        *_space(config.reduced_resolution, spatial_ndim),
        slice(None),
    )

    def sample(key: str) -> np.ndarray:
        return np.moveaxis(np.asarray(_dataset(_group(file, key), "data")[selection]), 0, -2)

    selected = keys[_split_rows(len(keys), config)]
    field = (
        np.stack([sample(key) for key in selected])
        if selected
        else np.zeros((0, *sample(keys[0]).shape), np.float32)
    )
    grid = _coordinate_grid(
        _group(first, "grid"), _GROUP_AXES[:spatial_ndim], config.reduced_resolution
    )
    return field.astype(np.float32), grid


def load_pdebench(config: PDEBenchConfig) -> PDEBenchData:
    """Read one split of a PDEBench file as PDEBench's loader lays it out.

    Args:
        config: The file, split, resolution and input steps.

    Returns:
        The split's inputs, targets and grid.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has none of PDEBench's layouts, or ``initial_step`` leaves no
            time step to predict.
    """
    path = Path(config.file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDEBench file not found: {path}")
    with h5py.File(path, "r") as file:
        if path.suffix == ".h5":
            field, grid = _read_groups(file, path, config)
        elif "tensor" in file:
            field, grid = _read_tensor(
                file, _split_rows(_dataset(file, "tensor").shape[0], config), config
            )
        elif "density" in file:
            field, grid = _read_fields(
                file, _split_rows(_dataset(file, "density").shape[0], config), config
            )
        else:
            raise ValueError(
                f"{path.name}: no PDEBench layout found (keys: {sorted(file.keys())}); a "
                "PDEBench .hdf5 file holds 'tensor' or the compressible-flow fields"
            )
    time_steps = field.shape[-2]
    if config.initial_step >= time_steps:
        # PDEBench slices the input steps without a check, so an initial_step of at least the
        # time steps read makes the input the whole target: for Darcy flow (two steps, nu then
        # the solution) any initial_step above 1 puts the solution in the input.
        raise ValueError(
            f"{path.name}: initial_step {config.initial_step} leaves no step to predict of the "
            f"{time_steps} time steps read (Darcy flow holds 2: nu, then the solution)"
        )
    return PDEBenchData(inputs=field[..., : config.initial_step, :], targets=field, grid=grid)


def create_pdebench_loader(config: PDEBenchConfig, *, batch_size: int, seed: int) -> Pipeline:
    """Build a datarax ``Pipeline`` over one split of a PDEBench file.

    The split's ``{"input", "target"}`` records are served by a ``MemorySource``, as opifex's
    other loaders serve theirs; the train split is shuffled from the seed's ``shuffle`` stream.

    Args:
        config: The file, split, resolution and input steps.
        batch_size: Records per batch.
        seed: Seed of the shuffle stream and of the pipeline's streams.

    Returns:
        The pipeline; the grid every record shares is :func:`load_pdebench`'s.
    """
    data = load_pdebench(config)
    source = MemorySource(
        MemorySourceConfig(shuffle=config.split == "train"),
        data={"input": data.inputs, "target": data.targets},
        rngs=rngs_from_seed(seed, ("shuffle",)),
    )
    return Pipeline(source=source, stages=[], batch_size=batch_size, rngs=rngs_from_seed(seed))
