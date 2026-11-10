"""The UQ capability catalogue: every declaration opifex makes, read-only and assembled once.

Declarations stay with the packages they describe; the catalogue collects them from each
package's public namespace. Nothing registers at import time, so importing a package
changes no global state, and a name declared twice is refused when the catalogue is built.
"""

import subprocess
import sys
import textwrap
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


#: Third-party roots a module may import at module level while only some extras install
#: them. Today that is ``ase`` alone, for ``opifex.deployment.ase_calculator``, which
#: subclasses ``ase.calculators.calculator.Calculator`` and so cannot defer the import.
#: ``pyscf`` is not here because ``qh9_eval`` imports it inside a function, which no
#: import-time walk reaches.
#:
#: **The list is deliberately this short.** A module that newly imports an optional
#: dependency at module level will fail this test rather than be skipped, which is the
#: point: the skip is keyed to the name that was missing, never to the module that failed.
OPTIONAL_THIRD_PARTY_ROOTS = ("ase",)

#: A walk returning fewer modules than this is not walking the tree, and the test would
#: pass by importing almost nothing. A chosen round number well under the 601 modules
#: measured, not a snapshot of that count -- it exists to catch a broken walk, not to
#: track the package's size.
MINIMUM_MODULES_WALKED = 100


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
    serves this test's claim. The skip is keyed to the **missing root**, so a
    ``ModuleNotFoundError`` naming an ``opifex`` module, or any other import failure, still
    fails -- and so does a missing third-party root outside
    :data:`OPTIONAL_THIRD_PARTY_ROOTS`.
    """
    snippet = textwrap.dedent(
        f"""
        import importlib, pkgutil, opifex
        optional = {OPTIONAL_THIRD_PARTY_ROOTS!r}
        walked, imported, skipped = 0, 0, []
        for info in pkgutil.walk_packages(opifex.__path__, "opifex."):
            walked += 1
            try:
                importlib.import_module(info.name)
            except ModuleNotFoundError as missing:
                root = (missing.name or "").split(".")[0]
                if root not in optional:
                    raise
                skipped.append(info.name + " <- " + root)
                continue
            imported += 1
        assert not hasattr(importlib.import_module("opifex.uncertainty"), "UQRegistry")
        print("walked", walked, "imported", imported, "skipped", ",".join(skipped) or "-")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=True, timeout=600
    )

    report = completed.stdout.split()
    walked, imported, skipped = int(report[1]), int(report[3]), report[5]
    assert walked >= MINIMUM_MODULES_WALKED, (
        f"the walk reached only {walked} modules, so it is not measuring the package; "
        "a walk that imports almost nothing passes this test for the wrong reason"
    )
    assert imported + (0 if skipped == "-" else len(skipped.split(","))) == walked, (
        f"{walked} walked but {imported} imported and {skipped} skipped do not account "
        "for each other"
    )
