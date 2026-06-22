"""rMD17 molecular dynamics dataset loader, built on datarax.

The revised MD17 dataset (rMD17) recomputes the MD17 molecular-dynamics
trajectories at a consistent PBE/def2-SVP level of theory with tight SCF
convergence, removing the noise that made the original MD17 energies/forces
unsuitable as a strict ML benchmark.

Reference:
    A. S. Christensen and O. A. von Lilienfeld, "On the role of gradients
    for machine learning of molecular energies and forces", Mach. Learn.:
    Sci. Technol. 1, 045018 (2020). Data: figshare DOI
    10.6084/m9.figshare.12672038.

Archive layout (resolved live from the figshare article metadata):
    - one ``rmd17_<molecule>.npz`` per molecule, with keys
      ``coords`` (n, n_atoms, 3), ``energies`` (n,),
      ``forces`` (n, n_atoms, 3), ``nuclear_charges`` (n_atoms,),
      plus ``old_indices`` / ``old_energies`` / ``old_forces`` provenance.
    - ``index_train_0{1..5}.csv`` / ``index_test_0{1..5}.csv``: the five
      canonical 1000-configuration train/test splits (0-indexed, one
      integer per line).

Units:
    energies   kcal/mol
    forces     kcal/mol/Angstrom
    coords     Angstrom
    1 kcal/mol = 43.364 meV (``KCAL_PER_MOL_IN_MEV``).

The loader downloads + caches a single molecule once (idempotent), parses
it, splits it into train/validation sets, and wraps each split in a datarax
``MemorySource`` driven by a datarax ``Pipeline`` for batched, shuffled
iteration. No download happens at import time.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datarax.pipeline import Pipeline
from datarax.sources import MemorySource, MemorySourceConfig
from flax import nnx
from numpy.typing import DTypeLike  # noqa: TC002


logger = logging.getLogger(__name__)

# 1 kcal/mol expressed in meV (for energy-unit documentation / conversion).
KCAL_PER_MOL_IN_MEV: float = 43.364

# figshare article id behind DOI 10.6084/m9.figshare.12672038 (rMD17).
_FIGSHARE_ARTICLE_ID: int = 12672038
_FIGSHARE_ARTICLE_API: str = f"https://api.figshare.com/v2/articles/{_FIGSHARE_ARTICLE_ID}"

# Molecules shipped as ``rmd17_<molecule>.npz`` in the figshare archive.
KNOWN_MOLECULES: tuple[str, ...] = (
    "aspirin",
    "azobenzene",
    "benzene",
    "ethanol",
    "malonaldehyde",
    "naphthalene",
    "paracetamol",
    "salicylic",
    "toluene",
    "uracil",
)

_DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "opifex" / "rmd17"
_DOWNLOAD_TIMEOUT_SECONDS: int = 600
_DOWNLOAD_CHUNK_BYTES: int = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class RMD17Config:
    """Immutable configuration for an rMD17 loader.

    Attributes:
        molecule: rMD17 molecule name (must be in ``KNOWN_MOLECULES``).
        n_train: Number of training configurations.
        n_val: Number of validation configurations.
        batch_size: Records emitted per pipeline batch.
        seed: Seed controlling the split and shuffle order.
        shuffle: Whether to shuffle the training pipeline.
        data_dir: Cache directory for the downloaded npz/CSV files.
        split_index: Which canonical figshare split (1..5) to prefer when
            building the train/val sets from the archive's index CSVs.
    """

    molecule: str = "aspirin"
    n_train: int = 1000
    n_val: int = 1000
    batch_size: int = 32
    seed: int = 42
    shuffle: bool = True
    data_dir: Path | None = None
    split_index: int = 1

    def __post_init__(self) -> None:
        """Validate the molecule name and split index, failing fast."""
        if self.molecule not in KNOWN_MOLECULES:
            raise ValueError(
                f"Unknown molecule {self.molecule!r}. "
                f"Available rMD17 molecules: {', '.join(KNOWN_MOLECULES)}."
            )
        if not 1 <= self.split_index <= 5:
            raise ValueError(f"split_index must be in 1..5, got {self.split_index}.")
        if self.n_train < 1 or self.n_val < 1:
            raise ValueError(
                f"n_train and n_val must be >= 1, got n_train={self.n_train}, n_val={self.n_val}."
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")


@dataclass(frozen=True)
class RMD17Data:
    """Parsed, unit-annotated rMD17 arrays for a single molecule.

    Attributes:
        positions: Atomic coordinates, shape (n, n_atoms, 3), Angstrom.
        energy: Potential energies, shape (n,), kcal/mol.
        forces: Atomic forces, shape (n, n_atoms, 3), kcal/mol/Angstrom.
        atomic_numbers: Nuclear charges, shape (n_atoms,) (fixed per molecule).
        length_unit: Unit string for positions.
        energy_unit: Unit string for energy.
        force_unit: Unit string for forces.
    """

    positions: np.ndarray
    energy: np.ndarray
    forces: np.ndarray
    atomic_numbers: np.ndarray
    length_unit: str = "Angstrom"
    energy_unit: str = "kcal/mol"
    force_unit: str = "kcal/mol/Angstrom"

    @property
    def n_configs(self) -> int:
        """Number of molecular configurations in this molecule's dataset."""
        return int(self.positions.shape[0])

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return int(self.atomic_numbers.shape[0])


@dataclass(frozen=True)
class RMD17Loaders:
    """datarax pipelines plus metadata returned by :func:`create_rmd17_loader`.

    Attributes:
        train: datarax ``Pipeline`` over the training split (shuffled).
        val: datarax ``Pipeline`` over the validation split (sequential).
        atomic_numbers: Nuclear charges, shape (n_atoms,), fixed per molecule.
        train_indices: Indices into the parsed dataset used for training.
        val_indices: Indices into the parsed dataset used for validation.
        energy_mean: Mean training energy (kcal/mol) for normalization hooks.
        energy_std: Std of training energy (kcal/mol) for normalization hooks.
        units: Mapping of quantity name to its unit string.
    """

    train: Pipeline
    val: Pipeline
    atomic_numbers: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    energy_mean: float
    energy_std: float
    units: dict[str, str]

    @property
    def n_train(self) -> int:
        """Number of training configurations."""
        return int(self.train_indices.shape[0])

    @property
    def n_val(self) -> int:
        """Number of validation configurations."""
        return int(self.val_indices.shape[0])

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return int(self.atomic_numbers.shape[0])


# =============================================================================
# Parsing
# =============================================================================


def parse_rmd17_npz(npz_path: Path, *, dtype: DTypeLike = np.float32) -> RMD17Data:
    """Parse an ``rmd17_<molecule>.npz`` file into an :class:`RMD17Data`.

    Casts coordinates/energies/forces to ``dtype`` and nuclear charges to
    ``int32`` while preserving the documented physical units. The native
    figshare archive stores floats in ``float64``; pass ``dtype=np.float64`` to
    retain full double precision (e.g. for an ``x64`` training run whose force
    gradients would otherwise be capped), or keep the ``float32`` default for
    the standard single-precision path.

    Args:
        npz_path: Path to a (real or synthetic) rMD17 npz file.
        dtype: Floating dtype for coordinates/energies/forces (default
            ``float32``).

    Returns:
        Parsed, unit-annotated arrays.

    Raises:
        FileNotFoundError: If ``npz_path`` does not exist.
        KeyError: If a required array key is missing from the archive.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"rMD17 npz not found: {npz_path}")

    with np.load(npz_path) as archive:
        required = ("coords", "energies", "forces", "nuclear_charges")
        missing = [key for key in required if key not in archive]
        if missing:
            raise KeyError(f"rMD17 npz {npz_path} missing keys: {missing}")
        positions = np.asarray(archive["coords"], dtype=dtype)
        energy = np.asarray(archive["energies"], dtype=dtype)
        forces = np.asarray(archive["forces"], dtype=dtype)
        atomic_numbers = np.asarray(archive["nuclear_charges"], dtype=np.int32)

    return RMD17Data(
        positions=positions,
        energy=energy,
        forces=forces,
        atomic_numbers=atomic_numbers,
    )


# =============================================================================
# Download + cache
# =============================================================================


def _figshare_file_index() -> dict[str, str]:
    """Resolve figshare file names to download URLs in a single request.

    Returns:
        Mapping from file name to its (currently valid) download URL.

    Raises:
        OSError: If the figshare article metadata cannot be retrieved.
    """
    try:
        with urllib.request.urlopen(  # noqa: S310  # nosec B310 (fixed https figshare API)
            _FIGSHARE_ARTICLE_API, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            metadata = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OSError(f"Could not resolve rMD17 figshare metadata: {exc}") from exc

    return {entry["name"]: entry["download_url"] for entry in metadata.get("files", [])}


def _download_file(url: str, destination: Path) -> None:
    """Stream ``url`` to ``destination`` (atomic via a temp file).

    Args:
        url: HTTPS download URL.
        destination: Final cache path.

    Raises:
        OSError: If the download fails.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(  # noqa: S310  # nosec B310 (figshare download URL)
                url, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
            tmp_path.open("wb") as handle,
        ):
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
    except (urllib.error.URLError, TimeoutError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise OSError(f"Failed to download {url}: {exc}") from exc
    tmp_path.replace(destination)


def download_rmd17_molecule(molecule: str, cache_dir: Path) -> Path:
    """Download (or reuse) a single rMD17 molecule npz, idempotently.

    Resolves the figshare article metadata once, then fetches only the
    requested ``rmd17_<molecule>.npz`` if it is not already cached.

    Args:
        molecule: Molecule name (must be in ``KNOWN_MOLECULES``).
        cache_dir: Directory to cache the npz file in.

    Returns:
        Path to the cached npz file.

    Raises:
        ValueError: If ``molecule`` is unknown.
        OSError: If the file is absent from figshare or the download fails.
    """
    if molecule not in KNOWN_MOLECULES:
        raise ValueError(f"Unknown molecule {molecule!r}. Available: {', '.join(KNOWN_MOLECULES)}.")

    file_name = f"rmd17_{molecule}.npz"
    destination = cache_dir / file_name
    if destination.exists():
        logger.info("Using cached rMD17 npz: %s", destination)
        return destination

    file_index = _figshare_file_index()
    if file_name not in file_index:
        raise OSError(
            f"rMD17 file {file_name!r} not present in figshare article "
            f"{_FIGSHARE_ARTICLE_ID}. Available: {sorted(file_index)}."
        )

    logger.info("Downloading rMD17 %s to %s", molecule, destination)
    _download_file(file_index[file_name], destination)
    return destination


def _download_split_indices(cache_dir: Path, split_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Download the canonical figshare train/test index CSVs for a split.

    Args:
        cache_dir: Cache directory for the CSV files.
        split_index: Split number (1..5).

    Returns:
        Tuple of (train_indices, test_indices) as 0-indexed int arrays.

    Raises:
        OSError: If the CSVs are unavailable.
    """
    file_index = _figshare_file_index()
    out: list[np.ndarray] = []
    for kind in ("train", "test"):
        file_name = f"index_{kind}_0{split_index}.csv"
        destination = cache_dir / file_name
        if not destination.exists():
            if file_name not in file_index:
                raise OSError(f"rMD17 split file {file_name!r} not on figshare.")
            _download_file(file_index[file_name], destination)
        out.append(np.loadtxt(destination, dtype=np.int64))
    return out[0], out[1]


# =============================================================================
# Splitting + pipeline construction
# =============================================================================


def _resolve_split_indices(
    data: RMD17Data,
    config: RMD17Config,
    *,
    cache_dir: Path | None,
    use_archive_splits: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose disjoint train/val index arrays of the requested sizes.

    Prefers the archive's canonical index CSVs (when downloading a real
    molecule); otherwise falls back to a seeded random permutation.

    Args:
        data: Parsed molecule data.
        config: Loader configuration (sizes, seed, split index).
        cache_dir: Cache directory (only used for the archive-split path).
        use_archive_splits: Whether to attempt the figshare index CSVs.

    Returns:
        Tuple of (train_indices, val_indices), disjoint, requested sizes.

    Raises:
        ValueError: If the requested split exceeds available configurations.
    """
    n_configs = data.n_configs
    if config.n_train + config.n_val > n_configs:
        raise ValueError(
            f"Requested split (n_train={config.n_train} + n_val={config.n_val}) "
            f"exceeds available configurations ({n_configs}) for molecule "
            f"{config.molecule!r}."
        )

    if use_archive_splits and cache_dir is not None:
        train_pool, test_pool = _download_split_indices(cache_dir, config.split_index)
        train_indices = train_pool[: config.n_train]
        val_indices = test_pool[: config.n_val]
        return train_indices.astype(np.int64), val_indices.astype(np.int64)

    rng = np.random.default_rng(config.seed)
    permutation = rng.permutation(n_configs)
    train_indices = permutation[: config.n_train]
    val_indices = permutation[config.n_train : config.n_train + config.n_val]
    return train_indices.astype(np.int64), val_indices.astype(np.int64)


def _build_pipeline(
    data: RMD17Data,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Pipeline:
    """Wrap a subset of the parsed arrays in a datarax MemorySource + Pipeline.

    The datarax ``Pipeline`` drives iteration via the source's
    ``get_batch_at(position, batch_size, key)`` contract; shuffling is a
    property of the ``MemorySource`` (per-epoch permutation seeded by the
    pipeline's ``nnx.Rngs``). No transform stages are attached — the records
    are emitted as ``{"positions", "energy", "forces"}`` batches.

    Args:
        data: Parsed molecule data.
        indices: Configuration indices for this split.
        batch_size: Records per batch.
        shuffle: Whether to shuffle iteration order.
        seed: Seed for the source's shuffle stream and the pipeline rngs.

    Returns:
        A configured datarax ``Pipeline``.
    """
    subset = {
        "positions": data.positions[indices],
        "energy": data.energy[indices],
        "forces": data.forces[indices],
    }
    source = MemorySource(
        MemorySourceConfig(shuffle=shuffle),
        data=subset,
        rngs=nnx.Rngs(shuffle=seed),
    )
    return Pipeline(
        source=source,
        stages=[],
        batch_size=batch_size,
        rngs=nnx.Rngs(seed),
    )


def create_rmd17_loader(
    *,
    molecule: str = "aspirin",
    n_train: int = 1000,
    n_val: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
    shuffle: bool = True,
    data_dir: Path | None = None,
    split_index: int = 1,
    npz_path: Path | None = None,
    dtype: DTypeLike = np.float32,
) -> RMD17Loaders:
    """Create batched train/val datarax pipelines for an rMD17 molecule.

    Downloads + caches the molecule on first use (unless ``npz_path`` is
    given), parses it, builds disjoint train/val splits, and wraps each in a
    datarax ``MemorySource`` driven by a datarax ``Pipeline``.

    Args:
        molecule: rMD17 molecule name (default ``"aspirin"``).
        n_train: Number of training configurations (default 1000).
        n_val: Number of validation configurations (default 1000).
        batch_size: Records per pipeline batch.
        seed: Seed for the split and shuffle order.
        shuffle: Whether to shuffle the training pipeline.
        data_dir: Cache directory (default ``~/.cache/opifex/rmd17``).
        split_index: Canonical figshare split (1..5) for real downloads.
        npz_path: Optional path to a pre-existing rMD17 npz. When provided,
            no download occurs and a seeded random split is used. This is the
            constructor hook used by network-free tests.
        dtype: Floating dtype for coordinates/energies/forces (default
            ``float32``). Pass ``np.float64`` to keep the archive's native
            double precision for an ``x64`` run.

    Returns:
        An :class:`RMD17Loaders` bundle (train/val pipelines + metadata).

    Raises:
        ValueError: For an unknown molecule or an oversized split request.
        OSError: If a required figshare file cannot be downloaded.
    """
    config = RMD17Config(
        molecule=molecule,
        n_train=n_train,
        n_val=n_val,
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
        data_dir=data_dir,
        split_index=split_index,
    )

    if npz_path is not None:
        resolved_npz = npz_path
        cache_dir: Path | None = None
        use_archive_splits = False
    else:
        cache_dir = config.data_dir or _DEFAULT_CACHE_DIR
        resolved_npz = download_rmd17_molecule(config.molecule, cache_dir)
        use_archive_splits = True

    data = parse_rmd17_npz(resolved_npz, dtype=dtype)
    train_indices, val_indices = _resolve_split_indices(
        data, config, cache_dir=cache_dir, use_archive_splits=use_archive_splits
    )

    train_energy = data.energy[train_indices]
    energy_mean = float(train_energy.mean())
    energy_std = float(train_energy.std())

    train_pipeline = _build_pipeline(
        data, train_indices, batch_size=config.batch_size, shuffle=config.shuffle, seed=config.seed
    )
    val_pipeline = _build_pipeline(
        data, val_indices, batch_size=config.batch_size, shuffle=False, seed=config.seed
    )

    return RMD17Loaders(
        train=train_pipeline,
        val=val_pipeline,
        atomic_numbers=data.atomic_numbers,
        train_indices=train_indices,
        val_indices=val_indices,
        energy_mean=energy_mean,
        energy_std=energy_std,
        units={
            "positions": data.length_unit,
            "energy": data.energy_unit,
            "forces": data.force_unit,
        },
    )


__all__ = [
    "KCAL_PER_MOL_IN_MEV",
    "KNOWN_MOLECULES",
    "RMD17Config",
    "RMD17Data",
    "RMD17Loaders",
    "create_rmd17_loader",
    "download_rmd17_molecule",
    "parse_rmd17_npz",
]
