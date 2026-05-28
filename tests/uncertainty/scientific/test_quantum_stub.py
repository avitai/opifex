"""Tests for the Step-10 quantum-chemistry stubs (Task 8.6)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.quantum import (
    ChemicalAccuracyCoverageStub,
    DensityUncertaintyStub,
    EnergyUncertaintyStub,
    ExchangeCorrelationUncertaintyStub,
)


_STEP_10_MESSAGE = r"Step 10 stub: see audit Migration Step 10"


def test_energy_uncertainty_stub_validates_arguments() -> None:
    stub = EnergyUncertaintyStub(method="dft")
    assert stub.method == "dft"
    assert stub.units == "hartree"

    EnergyUncertaintyStub(method="dft", units="kcal_per_mol")
    EnergyUncertaintyStub(method="dft", units="ev")

    with pytest.raises(ValueError, match="method"):
        EnergyUncertaintyStub(method="")
    with pytest.raises(ValueError, match="units"):
        EnergyUncertaintyStub(method="dft", units="joule")


def test_energy_uncertainty_stub_call_raises_step10() -> None:
    stub = EnergyUncertaintyStub(method="dft")
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub(jnp.zeros(8))


def test_density_uncertainty_stub_validates_grid_axes() -> None:
    stub = DensityUncertaintyStub(grid_axes=("x", "y", "z"))
    assert stub.grid_axes == ("x", "y", "z")
    with pytest.raises(ValueError, match="grid_axes"):
        DensityUncertaintyStub(grid_axes=())


def test_density_uncertainty_stub_call_raises_step10() -> None:
    stub = DensityUncertaintyStub(grid_axes=("x",))
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub(jnp.zeros((4, 4)))


def test_xc_uncertainty_stub_validates_functional_family() -> None:
    stub = ExchangeCorrelationUncertaintyStub(functional_family="GGA")
    assert stub.functional_family == "GGA"
    with pytest.raises(ValueError, match="functional_family"):
        ExchangeCorrelationUncertaintyStub(functional_family="")


def test_xc_uncertainty_stub_call_raises_step10() -> None:
    stub = ExchangeCorrelationUncertaintyStub(functional_family="LDA")
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub(jnp.zeros((4, 1)))


def test_chemical_accuracy_coverage_stub_validates_tolerance() -> None:
    stub = ChemicalAccuracyCoverageStub(tolerance_hartree=0.0016)
    assert stub.tolerance_hartree == 0.0016
    with pytest.raises(ValueError, match="tolerance_hartree"):
        ChemicalAccuracyCoverageStub(tolerance_hartree=0.0)


def test_chemical_accuracy_coverage_stub_call_raises_step10() -> None:
    stub = ChemicalAccuracyCoverageStub(tolerance_hartree=0.0016)
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub(jnp.zeros(4), jnp.zeros(4))


def test_quantum_module_imports_cleanly() -> None:
    import opifex.uncertainty.scientific.quantum as mod  # noqa: F401, PLC0415

    assert hasattr(mod, "EnergyUncertaintyStub")
    assert hasattr(mod, "DensityUncertaintyStub")
    assert hasattr(mod, "ExchangeCorrelationUncertaintyStub")
    assert hasattr(mod, "ChemicalAccuracyCoverageStub")
