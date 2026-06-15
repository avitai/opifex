r"""Tests for the QH9 checkpoint-evaluation CLI ``scripts/eval_qh9_blocks.py``.

These cover the script's *pure* orchestration logic -- µHa reporting, checkpoint /
database resolution and argument parsing -- without a GPU, the QH9 database or
PySCF. The heavy end-to-end metric computation lives in (and is tested through)
:mod:`opifex.neural.quantum.hamiltonian.qh9_eval`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType  # noqa: TC003

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER_PATH = _REPO_ROOT / "scripts" / "eval_qh9_blocks.py"


def _load_driver() -> ModuleType:
    """Import ``scripts/eval_qh9_blocks.py`` as a module (it is not a package)."""
    spec = importlib.util.spec_from_file_location("eval_qh9_blocks", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the driver's @dataclass(slots=True) field resolution
    # can look the module up via ``cls.__module__`` (loaded by path, not a package).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_DRIVER = _load_driver()


def _args(**overrides: object) -> object:
    """Build an ``EvalArgs`` with sensible defaults, overridable per test."""
    defaults: dict[str, object] = {
        "dataset": "stable",
        "db": None,
        "run_dir": None,
        "checkpoint": None,
        "limit": None,
        "hidden_irreps": "8x0e + 8x1o",
        "sh_lmax": 2,
        "num_interactions": 3,
        "start_refinement_layer": 0,
        "bottleneck_multiplicity": 8,
        "out": None,
    }
    defaults.update(overrides)
    return _DRIVER.EvalArgs(**defaults)


class TestReport:
    def test_micro_hartree_conversion(self) -> None:
        """The four Hartree MAEs gain ``*_micro_hartree`` keys scaled by 1e6."""
        raw: dict[str, float | int] = {
            "n_molecules": 5,
            "hamiltonian_mae": 3.15e-5,
            "orbital_energy_mae": 1.0e-4,
            "orbital_energy_mae_occ": 5.0e-5,
            "homo_lumo_gap_mae": 2.0e-4,
            "coefficient_similarity": 0.987,
        }
        report = _DRIVER._report(raw)
        assert report["hamiltonian_mae_micro_hartree"] == pytest.approx(31.5)
        assert report["orbital_energy_mae_micro_hartree"] == pytest.approx(100.0)
        assert report["homo_lumo_gap_mae_micro_hartree"] == pytest.approx(200.0)
        # Non-MAE fields are preserved and not converted.
        assert report["coefficient_similarity"] == 0.987
        assert report["n_molecules"] == 5
        assert "coefficient_similarity_micro_hartree" not in report


class TestCheckpointResolution:
    def test_explicit_checkpoint_wins(self, tmp_path: Path) -> None:
        explicit = tmp_path / "ckpt"
        assert _DRIVER._resolve_checkpoint(_args(checkpoint=explicit)) == explicit

    def test_run_dir_picks_highest_epoch(self, tmp_path: Path) -> None:
        checkpoints = tmp_path / "checkpoints"
        checkpoints.mkdir()
        for epoch in (2, 10, 7):  # out of order; selection is by epoch number, not mtime
            (checkpoints / f"best_epoch_{epoch}").mkdir()
        resolved = _DRIVER._resolve_checkpoint(_args(run_dir=tmp_path))
        assert resolved.name == "best_epoch_10"

    def test_missing_checkpoint_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _DRIVER._resolve_checkpoint(_args(run_dir=tmp_path))

    def test_no_source_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint or --run-dir"):
            _DRIVER._resolve_checkpoint(_args())


class TestDatabaseResolution:
    def test_explicit_db_wins(self, tmp_path: Path) -> None:
        explicit = tmp_path / "my.db"
        assert _DRIVER._resolve_db_path(_args(db=explicit)) == explicit

    def test_dataset_default(self) -> None:
        assert _DRIVER._resolve_db_path(_args()).name == "QH9Stable.db"


class TestArgParsing:
    def test_maps_flags_to_config_fields(self) -> None:
        parsed = _DRIVER._parse_args(
            [
                "--run-dir",
                "runs/exp1",
                "--hidden",
                "16x0e + 16x1o",
                "--sh-lmax",
                "4",
                "--num-interactions",
                "5",
                "--start-refinement-layer",
                "2",
                "--bottleneck-mul",
                "32",
                "--limit",
                "100",
            ]
        )
        assert parsed.hidden_irreps == "16x0e + 16x1o"
        assert parsed.sh_lmax == 4
        assert parsed.num_interactions == 5
        assert parsed.start_refinement_layer == 2
        assert parsed.bottleneck_multiplicity == 32
        assert parsed.limit == 100
        assert parsed.run_dir == Path("runs/exp1")
