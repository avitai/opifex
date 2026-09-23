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


def test_importing_every_package_registers_nothing() -> None:
    """No import mutates global state: there is no registry left to mutate."""
    snippet = textwrap.dedent(
        """
        import importlib, pkgutil, opifex
        for info in pkgutil.walk_packages(opifex.__path__, "opifex."):
            importlib.import_module(info.name)
        assert not hasattr(importlib.import_module("opifex.uncertainty"), "UQRegistry")
        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", snippet], capture_output=True, text=True, check=True, timeout=600
    )
    assert completed.stdout.strip() == "ok"
