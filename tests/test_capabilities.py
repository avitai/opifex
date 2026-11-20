"""The UQ capability catalogue: every declaration opifex makes, read-only and assembled once.

Declarations stay with the packages they describe; the catalogue collects them from each
package's public namespace. Nothing registers at import time, so importing a package
changes no global state, and a name declared twice is refused when the catalogue is built.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType

import pytest

from opifex.capabilities import assemble, require_uq_capability, UQ_CAPABILITIES
from opifex.core.capabilities import DefaultStrategy, UQCapability
from opifex.discovery import DISCOVERY_CAPABILITIES
from opifex.neural.operators import OPERATOR_CAPABILITIES
from opifex.uncertainty.sbi import SBI_CAPABILITIES


class TestTheCatalogue:
    """It holds every declaration, once, and cannot be changed."""

    def test_it_holds_every_table(self) -> None:
        for table in (DISCOVERY_CAPABILITIES, OPERATOR_CAPABILITIES, SBI_CAPABILITIES):
            for name, capability in table.items():
                assert UQ_CAPABILITIES[name] is capability

    def test_it_is_read_only(self) -> None:
        assert isinstance(UQ_CAPABILITIES, MappingProxyType)
        with pytest.raises(TypeError):
            UQ_CAPABILITIES["new"] = UQCapability()  # pyright: ignore[reportIndexIssue]

    def test_a_name_declared_twice_is_refused(self) -> None:
        first = {"model:X": UQCapability()}
        second = {"model:X": UQCapability(default_strategy=DefaultStrategy.BAYESIAN)}
        with pytest.raises(ValueError, match="model:X"):
            assemble(first, second)


class TestLookup:
    """An unknown name is refused with the names that exist."""

    def test_a_known_name(self) -> None:
        name = next(iter(DISCOVERY_CAPABILITIES))
        assert require_uq_capability(name) is DISCOVERY_CAPABILITIES[name]

    def test_an_unknown_name_lists_the_known_ones(self) -> None:
        with pytest.raises(KeyError) as raised:
            require_uq_capability("model:DoesNotExist")
        message = str(raised.value)
        assert "model:DoesNotExist" in message
        assert next(iter(DISCOVERY_CAPABILITIES)) in message


CHECK_EVERY_MODULE_IMPORTS = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_every_module_imports.py"
)


def test_importing_every_package_registers_nothing() -> None:
    """No import mutates global state: there is no registry left to mutate.

    **The corpus is the modules importable in this environment, and that is stated
    rather than assumed.** The walk's reach depends on which extras are installed, not on
    the source tree: CI's unit-test leg runs ``uv sync --locked --extra dev --extra test``,
    which installs neither ``ase`` nor ``pyscf``, while a local environment carrying
    ``--extra neural-dft`` has both. Measured by blocking those two roots on
    ``sys.meta_path``, exactly one of 601 modules becomes unimportable:
    ``opifex.deployment.ase_calculator``.

    A module that cannot be imported cannot register anything either, so skipping it
    serves this test's claim. The walk is ``scripts/check_every_module_imports.py``, which
    the clean-environment build check runs over the installed wheel too; it skips a module
    only for a missing optional root, so a ``ModuleNotFoundError`` naming an ``opifex``
    module, or any other import failure, still fails.
    """
    completed = subprocess.run(
        [sys.executable, str(CHECK_EVERY_MODULE_IMPORTS)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[-1].startswith("walked "), completed.stdout
    assert not hasattr(importlib.import_module("opifex.uncertainty"), "UQRegistry")
