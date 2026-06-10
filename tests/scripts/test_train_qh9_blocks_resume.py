"""Resume behaviour for the QH9 block-form training driver.

The full benchmark run is multi-hour, so a crash must not throw away completed
epochs. These tests cover the ``--resume`` path of ``scripts/train_qh9_blocks.py``:
checkpoint discovery (prefer the highest completed epoch), in-place restoration of
the predictor parameters, and continuity of ``train.log`` + ``metrics.json`` across
a relaunch. A tiny synthetic QH9 sqlite db and a minimal model keep the end-to-end
run small enough for the unit suite.
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType  # noqa: TC003

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER_PATH = _REPO_ROOT / "scripts" / "train_qh9_blocks.py"

_H2O_ATOMS = np.array([8, 1, 1], dtype=np.int32)
_HCNO_ATOMS = np.array([1, 6, 7, 8], dtype=np.int32)
_H2_ATOMS = np.array([1, 1], dtype=np.int32)


def _load_driver() -> ModuleType:
    """Import ``scripts/train_qh9_blocks.py`` as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("train_qh9_blocks", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the driver's @dataclass field resolution can look the
    # module up via ``cls.__module__`` (it is loaded by path, not as a package).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver() -> ModuleType:
    """The imported training-driver module."""
    return _load_driver()


def _native_ao(atoms: np.ndarray) -> int:
    """QH9-native total AO count (5 per H/He, 14 otherwise)."""
    return int(sum(5 if int(z) <= 2 else 14 for z in atoms))


def _make_row(molecule_id: int, atoms: np.ndarray, seed: int) -> tuple:
    """Build one synthetic QH9 row ``(id, N, Z, pos, Ham)``."""
    rng = np.random.default_rng(seed)
    positions = rng.standard_normal((len(atoms), 3)).astype(np.float64)
    native = rng.standard_normal((_native_ao(atoms), _native_ao(atoms))).astype(np.float64)
    native = native + native.T
    return (molecule_id, len(atoms), atoms.tobytes(), positions.tobytes(), native.tobytes())


@pytest.fixture
def synthetic_qh9_db(tmp_path: Path) -> Path:
    """Write a tiny synthetic QH9-Stable sqlite db with non-empty 0.8/0.1/0.1 splits."""
    cycle = (_H2O_ATOMS, _HCNO_ATOMS, _H2_ATOMS)
    specs = [(cycle[i % len(cycle)], i + 1) for i in range(15)]
    db_path = tmp_path / "QH9Stable.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE data (id INTEGER, N INTEGER, Z BLOB, pos BLOB, Ham BLOB)")
        connection.executemany(
            "INSERT INTO data VALUES (?, ?, ?, ?, ?)",
            [_make_row(i, atoms, seed) for i, (atoms, seed) in enumerate(specs)],
        )
    return db_path


def _tiny_argv(db_path: Path, out_dir: Path, *, epochs: int, resume: bool) -> list[str]:
    """A minimal-model argv covering the synthetic compositions."""
    argv = [
        "--db",
        str(db_path),
        "--out",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        "2",
        "--max-atoms",
        "6",
        "--max-edges",
        "30",
        "--hidden",
        "8x0e + 4x1o",
        "--sh-lmax",
        "1",
        "--num-interactions",
        "1",
    ]
    if resume:
        argv.append("--resume")
    return argv


# ---------------------------------------------------------------------------
# Checkpoint discovery (fast, pure)
# ---------------------------------------------------------------------------
def test_latest_resume_checkpoint_none_when_empty(driver: ModuleType, tmp_path: Path) -> None:
    """No checkpoint directory (or an empty one) yields ``None``."""
    assert driver._latest_resume_checkpoint(tmp_path / "missing") is None
    (tmp_path / "checkpoints").mkdir()
    assert driver._latest_resume_checkpoint(tmp_path / "checkpoints") is None


def test_latest_resume_checkpoint_picks_highest_epoch(driver: ModuleType, tmp_path: Path) -> None:
    """The highest completed epoch wins across ``best_epoch_*`` and ``last_epoch_*``."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "best_epoch_2").mkdir()
    (checkpoint_dir / "last_epoch_5").mkdir()
    (checkpoint_dir / "best_epoch_4").mkdir()
    found = driver._latest_resume_checkpoint(checkpoint_dir)
    assert found is not None
    path, epoch = found
    assert epoch == 5
    assert path.name == "last_epoch_5"


# ---------------------------------------------------------------------------
# End-to-end resume continuity
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_resume_continues_epochs_and_preserves_log(
    driver: ModuleType, synthetic_qh9_db: Path, tmp_path: Path
) -> None:
    """A relaunch with ``--resume`` extends the run instead of restarting it."""
    out_dir = tmp_path / "run"

    driver.main(_tiny_argv(synthetic_qh9_db, out_dir, epochs=1, resume=False))

    first_metrics = json.loads((out_dir / "metrics.json").read_text())
    assert [record["epoch"] for record in first_metrics["epochs"]] == [1]
    log_after_first = (out_dir / "train.log").read_text()
    assert "epoch 1/1" in log_after_first
    assert (out_dir / "checkpoints").exists()

    driver.main(_tiny_argv(synthetic_qh9_db, out_dir, epochs=2, resume=True))

    second_metrics = json.loads((out_dir / "metrics.json").read_text())
    # Epoch 1 is carried forward; epoch 2 is the only newly trained epoch.
    assert [record["epoch"] for record in second_metrics["epochs"]] == [1, 2]
    log_after_resume = (out_dir / "train.log").read_text()
    assert "epoch 1/1" in log_after_resume  # original log not truncated
    assert "resuming from epoch 2" in log_after_resume


@pytest.mark.slow
def test_resume_without_checkpoint_starts_fresh(
    driver: ModuleType, synthetic_qh9_db: Path, tmp_path: Path
) -> None:
    """``--resume`` against an empty output directory trains from epoch 1."""
    out_dir = tmp_path / "fresh"

    driver.main(_tiny_argv(synthetic_qh9_db, out_dir, epochs=1, resume=True))

    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert [record["epoch"] for record in metrics["epochs"]] == [1]
    assert "no checkpoint" in (out_dir / "train.log").read_text().lower()
