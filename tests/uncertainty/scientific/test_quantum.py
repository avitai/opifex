r"""Quantum-chemistry UQ surface contracts (Phase-13 Feature F5).

Replaces the Step-10 stub contracts (``test_quantum_stub.py``). The three
uncertainty surfaces aggregate an *ensemble* of quantum-chemistry predictions
(model / sample estimates stacked along axis 0) into a
:class:`PredictiveDistribution` — the predictive mean and epistemic variance
over the ensemble axis, mirroring Opifex's deep-ensemble member aggregation
(:func:`opifex.uncertainty._predictive.ensemble_predictive`). The coverage
surface is the standard empirical chemical-accuracy coverage: the fraction of
predictions within the chemical-accuracy band ``|prediction - reference| <=
tolerance`` (1 kcal/mol ≈ 0.0015936 Ha — Pople, "Quantum Chemical Models",
Nobel Lecture, Rev. Mod. Phys. 71, 1267 (1999); the long-standing DFT
"chemical accuracy" target).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.quantum import (
    CHEMICAL_ACCURACY_HARTREE,
    ChemicalAccuracyCoverage,
    DensityUncertainty,
    EnergyUncertainty,
    ExchangeCorrelationUncertainty,
)
from opifex.uncertainty.types import PredictiveDistribution


# ---------------------------------------------------------------------------
# Module constant + citation
# ---------------------------------------------------------------------------


def test_chemical_accuracy_constant_is_one_kcal_per_mol_in_hartree() -> None:
    """1 kcal/mol expressed in Hartree (the chemical-accuracy band)."""
    assert pytest.approx(0.0015936, abs=1e-7) == CHEMICAL_ACCURACY_HARTREE


# ---------------------------------------------------------------------------
# EnergyUncertainty — ensemble aggregation over axis 0
# ---------------------------------------------------------------------------


def test_energy_uncertainty_validates_arguments() -> None:
    surface = EnergyUncertainty(method="dft")
    assert surface.method == "dft"
    assert surface.units == "hartree"

    EnergyUncertainty(method="dft", units="kcal_per_mol")
    EnergyUncertainty(method="dft", units="ev")

    with pytest.raises(ValueError, match="method"):
        EnergyUncertainty(method="")
    with pytest.raises(ValueError, match="units"):
        EnergyUncertainty(method="dft", units="joule")


def test_energy_uncertainty_aggregates_ensemble_over_axis_zero() -> None:
    """Mean / epistemic variance taken over the leading member axis."""
    # 3-member ensemble, batch of 4 energies (shape (3, 4)).
    energies = jnp.asarray(
        [
            [-1.0, -2.0, -3.0, -4.0],
            [-1.2, -1.8, -3.1, -3.9],
            [-0.8, -2.2, -2.9, -4.1],
        ]
    )
    surface = EnergyUncertainty(method="dft")
    dist = surface(energies)

    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean.shape == (4,)
    assert jnp.array_equal(dist.mean, energies.mean(axis=0))

    assert dist.epistemic is not None
    assert jnp.array_equal(dist.epistemic, energies.var(axis=0))
    assert bool(jnp.all(dist.epistemic >= 0.0))

    assert dist.variance is not None
    assert dist.total_uncertainty is not None
    assert jnp.array_equal(dist.total_uncertainty, dist.epistemic)
    assert jnp.array_equal(dist.variance, dist.total_uncertainty)

    assert dist.aleatoric is not None
    assert jnp.array_equal(dist.aleatoric, jnp.zeros_like(dist.mean))

    assert dist.samples is not None
    assert dist.samples.shape == (3, 4)
    assert jnp.array_equal(dist.samples, energies)

    # Variance-additivity invariant must hold.
    dist.validate()


def test_energy_uncertainty_records_method_units_and_member_count() -> None:
    energies = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    dist = EnergyUncertainty(method="ccsd_t", units="kcal_per_mol")(energies)
    meta = dist.metadata_dict()
    assert meta["method"] == "ccsd_t"
    assert meta["units"] == "kcal_per_mol"
    assert int(meta["num_members"]) == 3


# ---------------------------------------------------------------------------
# DensityUncertainty — per-grid-point ensemble aggregation
# ---------------------------------------------------------------------------


def test_density_uncertainty_validates_grid_axes() -> None:
    surface = DensityUncertainty(grid_axes=("x", "y", "z"))
    assert surface.grid_axes == ("x", "y", "z")
    with pytest.raises(ValueError, match="grid_axes"):
        DensityUncertainty(grid_axes=())


def test_density_uncertainty_aggregates_per_grid_point() -> None:
    """Per-element mean / variance over a 4-member density ensemble."""
    # 4 members on a 2x3 grid -> shape (4, 2, 3).
    key = jax.random.key(0)
    density = jax.random.uniform(key, (4, 2, 3))
    dist = DensityUncertainty(grid_axes=("x", "y"))(density)

    assert dist.mean.shape == (2, 3)
    assert jnp.array_equal(dist.mean, density.mean(axis=0))
    assert dist.epistemic is not None
    assert jnp.array_equal(dist.epistemic, density.var(axis=0))
    assert dist.variance is not None
    assert jnp.array_equal(dist.variance, dist.epistemic)
    assert dist.samples is not None
    assert dist.samples.shape == (4, 2, 3)

    meta = dist.metadata_dict()
    assert meta["grid_axes"] == ("x", "y")
    dist.validate()


# ---------------------------------------------------------------------------
# ExchangeCorrelationUncertainty — same aggregation, functional-family tag
# ---------------------------------------------------------------------------


def test_xc_uncertainty_validates_functional_family() -> None:
    surface = ExchangeCorrelationUncertainty(functional_family="GGA")
    assert surface.functional_family == "GGA"
    with pytest.raises(ValueError, match="functional_family"):
        ExchangeCorrelationUncertainty(functional_family="")


def test_xc_uncertainty_aggregates_per_element() -> None:
    """Per-element mean / variance over a 3-member XC-output ensemble."""
    xc_predictions = jnp.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.12, 0.18, 0.33],
            [0.08, 0.22, 0.27],
        ]
    )
    dist = ExchangeCorrelationUncertainty(functional_family="meta_gga")(xc_predictions)

    assert dist.mean.shape == (3,)
    assert jnp.array_equal(dist.mean, xc_predictions.mean(axis=0))
    assert dist.epistemic is not None
    assert jnp.array_equal(dist.epistemic, xc_predictions.var(axis=0))
    assert dist.variance is not None
    assert jnp.array_equal(dist.variance, dist.epistemic)
    assert dist.samples is not None
    assert dist.samples.shape == (3, 3)

    meta = dist.metadata_dict()
    assert meta["functional_family"] == "meta_gga"
    dist.validate()


# ---------------------------------------------------------------------------
# ChemicalAccuracyCoverage — empirical band coverage
# ---------------------------------------------------------------------------


def test_chemical_accuracy_coverage_validates_tolerance() -> None:
    cov = ChemicalAccuracyCoverage(tolerance_hartree=0.0016)
    assert cov.tolerance_hartree == 0.0016
    with pytest.raises(ValueError, match="tolerance_hartree"):
        ChemicalAccuracyCoverage(tolerance_hartree=0.0)


def test_chemical_accuracy_coverage_fraction_within_band() -> None:
    """3 of 4 predictions within tolerance -> 0.75."""
    references = jnp.asarray([0.0, 0.0, 0.0, 0.0])
    # Errors: 0.001 (in), 0.0015 (in), 0.0010 (in), 0.010 (out) at tol 0.0016.
    predictions = jnp.asarray([0.001, 0.0015, -0.0010, 0.010])
    cov = ChemicalAccuracyCoverage(tolerance_hartree=0.0016)
    result = cov(predictions, references)
    assert isinstance(result, float)
    assert result == pytest.approx(0.75)


def test_chemical_accuracy_coverage_all_within_and_none_within() -> None:
    references = jnp.zeros(3)
    cov = ChemicalAccuracyCoverage(tolerance_hartree=0.0016)
    assert cov(jnp.zeros(3), references) == pytest.approx(1.0)
    assert cov(jnp.full((3,), 1.0), references) == pytest.approx(0.0)


def test_chemical_accuracy_classmethod_uses_one_kcal_per_mol() -> None:
    """The ``chemical_accuracy()`` constructor pins the 1 kcal/mol band."""
    cov = ChemicalAccuracyCoverage.chemical_accuracy()
    assert cov.tolerance_hartree == pytest.approx(CHEMICAL_ACCURACY_HARTREE)
    # A 0.0015 Ha error is within 1 kcal/mol; a 0.0020 Ha error is outside.
    references = jnp.zeros(2)
    predictions = jnp.asarray([0.0015, 0.0020])
    assert cov(predictions, references) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# JAX/NNX transform compatibility (REQUIRED): jit / grad / vmap
# ---------------------------------------------------------------------------


def test_energy_uncertainty_is_jit_compatible() -> None:
    surface = EnergyUncertainty(method="dft")

    @jax.jit
    def aggregate(energies: jax.Array) -> jax.Array:
        dist = surface(energies)
        variance = dist.variance
        assert variance is not None
        return dist.mean.sum() + variance.sum()

    out = aggregate(jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert jnp.isfinite(out)


def test_energy_uncertainty_is_grad_compatible() -> None:
    surface = EnergyUncertainty(method="dft")

    def scalar(energies: jax.Array) -> jax.Array:
        return surface(energies).mean.sum()

    grad = jax.grad(scalar)(jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    # mean over 3 members -> d(mean.sum)/d(member entry) == 1/3 everywhere.
    assert jnp.allclose(grad, jnp.full((3, 2), 1.0 / 3.0))


def test_density_uncertainty_is_vmap_compatible() -> None:
    """vmap over a batch of independent density ensembles."""
    surface = DensityUncertainty(grid_axes=("x",))
    key = jax.random.key(1)
    # Outer batch of 5, each a 3-member ensemble over a length-2 grid.
    batched = jax.random.uniform(key, (5, 3, 2))
    means = jax.vmap(lambda ens: surface(ens).mean)(batched)
    assert means.shape == (5, 2)
    assert jnp.allclose(means, batched.mean(axis=1))
