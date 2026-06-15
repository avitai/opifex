r"""Quantum-chemistry uncertainty-quantification surfaces (Feature F5).

Four callables for quantum-chemistry UQ:

* :class:`EnergyUncertainty` — predictive uncertainty over total / per-state
  energies.
* :class:`DensityUncertainty` — predictive uncertainty over electron-density
  fields.
* :class:`ExchangeCorrelationUncertainty` — predictive uncertainty over an
  exchange-correlation functional's outputs.
* :class:`ChemicalAccuracyCoverage` — empirical fraction of predictions inside
  the chemical-accuracy band.

The three uncertainty surfaces aggregate an *ensemble* of quantum-chemistry
predictions — multiple model / sample estimates stacked along axis ``0`` — into
a :class:`PredictiveDistribution`. The predictive mean and the across-member
(sample) variance form the mean and the *epistemic* variance, mirroring
Opifex's deep-ensemble member aggregation. The construction is delegated to the
shared :func:`opifex.uncertainty._predictive.ensemble_predictive` factory so the
"stack-of-predictions → epistemic-decomposed predictive" reduction lives in one
place (Rule 1 — DRY) and stays identical to the ensemble model adapters.

``ChemicalAccuracyCoverage`` reports the empirical fraction of predictions
within ``tolerance_hartree`` of the references. The conventional target is
*chemical accuracy* — 1 kcal/mol — long used as the DFT / quantum-chemistry
benchmark threshold (Pople, "Quantum Chemical Models", Nobel Lecture,
Rev. Mod. Phys. 71, 1267 (1999)). In atomic units 1 kcal/mol equals
:data:`CHEMICAL_ACCURACY_HARTREE` (≈ 0.0015936 Ha).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opifex.uncertainty._predictive import ensemble_predictive
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001


if TYPE_CHECKING:
    import jax

import jax.numpy as jnp


# Chemical accuracy: 1 kcal/mol expressed in Hartree (atomic units).
# 1 kcal/mol = 1 / 627.509474... Ha ≈ 0.0015936 Ha. The 1 kcal/mol
# "chemical accuracy" target is the long-standing quantum-chemistry /
# DFT benchmark threshold (Pople, Rev. Mod. Phys. 71, 1267 (1999)).
CHEMICAL_ACCURACY_HARTREE: float = 0.0015936

# Provenance source-package tag stamped into the emitted predictive metadata.
_SOURCE_PACKAGE = "opifex.uncertainty.scientific.quantum"


class EnergyUncertainty:
    """Predictive uncertainty over total / per-state energies.

    Aggregates an *ensemble* of energy predictions into a
    :class:`PredictiveDistribution`. The ensemble members live along axis ``0``
    (shape ``(num_members, ...)``); the predictive mean is ``energies.mean(0)``
    and the epistemic variance is ``energies.var(0)``. Reports ``units`` so
    downstream consumers can render bands in Hartree / kcal/mol / eV without
    re-converting.
    """

    def __init__(self, *, method: str, units: str = "hartree") -> None:
        """Configure the electronic-structure method and reporting energy units."""
        if not method:
            raise ValueError("method must be a non-empty string.")
        if units not in {"hartree", "kcal_per_mol", "ev"}:
            raise ValueError(
                f"units must be one of 'hartree' / 'kcal_per_mol' / 'ev'; got {units!r}."
            )
        self.method = method
        self.units = units

    def __call__(self, energies: jax.Array) -> PredictiveDistribution:
        """Aggregate an energy ensemble into a :class:`PredictiveDistribution`.

        Args:
            energies: Ensemble of energy predictions with members stacked along
                axis ``0`` (shape ``(num_members, ...)``).

        Returns:
            A :class:`PredictiveDistribution` whose ``mean`` is the across-member
            mean and whose ``epistemic`` / ``total_uncertainty`` / ``variance``
            are the across-member variance; ``aleatoric`` is identically zero and
            the raw ensemble is stored on ``samples``.
        """
        return ensemble_predictive(
            energies,
            method=self.method,
            source_package=_SOURCE_PACKAGE,
            extra_metadata=(("units", self.units), ("num_members", int(energies.shape[0]))),
            include_zero_aleatoric=True,
        )


class DensityUncertainty:
    """Predictive uncertainty over electron-density fields.

    Aggregates an *ensemble* of density tensors on the ``grid_axes`` mesh into a
    per-grid-point :class:`PredictiveDistribution`. The ensemble members live
    along axis ``0`` (shape ``(num_members, *grid_shape)``); the predictive mean
    and epistemic variance are taken per grid point across the member axis.
    """

    def __init__(self, *, grid_axes: tuple[str, ...]) -> None:
        """Configure the named spatial-grid axes of the electron-density field."""
        if len(grid_axes) == 0:
            raise ValueError("grid_axes must contain at least one axis name.")
        self.grid_axes = grid_axes

    def __call__(self, density: jax.Array) -> PredictiveDistribution:
        """Aggregate a density ensemble into a per-grid-point predictive.

        Args:
            density: Ensemble of density fields with members stacked along axis
                ``0`` (shape ``(num_members, *grid_shape)``).

        Returns:
            A per-grid-point :class:`PredictiveDistribution` (mean + epistemic
            variance over the member axis), tagged with ``grid_axes``.
        """
        return ensemble_predictive(
            density,
            method="density_ensemble",
            source_package=_SOURCE_PACKAGE,
            extra_metadata=(
                ("grid_axes", self.grid_axes),
                ("num_members", int(density.shape[0])),
            ),
            include_zero_aleatoric=True,
        )


class ExchangeCorrelationUncertainty:
    """Predictive uncertainty over an exchange-correlation functional's outputs.

    Aggregates an *ensemble* of XC-functional predictions into a
    :class:`PredictiveDistribution` tagged with the ``functional_family``
    (e.g. LDA / GGA / meta-GGA / hybrid / neural). The ensemble members live
    along axis ``0`` (shape ``(num_members, ...)``); the predictive mean and
    epistemic variance are taken per element across the member axis.
    """

    def __init__(self, *, functional_family: str) -> None:
        """Configure the exchange-correlation functional family under study."""
        if not functional_family:
            raise ValueError("functional_family must be a non-empty string.")
        self.functional_family = functional_family

    def __call__(self, xc_predictions: jax.Array) -> PredictiveDistribution:
        """Aggregate an XC-output ensemble into a :class:`PredictiveDistribution`.

        Args:
            xc_predictions: Ensemble of XC-functional outputs with members
                stacked along axis ``0`` (shape ``(num_members, ...)``).

        Returns:
            A :class:`PredictiveDistribution` (mean + epistemic variance over the
            member axis), tagged with ``functional_family``.
        """
        return ensemble_predictive(
            xc_predictions,
            method="xc_ensemble",
            source_package=_SOURCE_PACKAGE,
            extra_metadata=(
                ("functional_family", self.functional_family),
                ("num_members", int(xc_predictions.shape[0])),
            ),
            include_zero_aleatoric=True,
        )


class ChemicalAccuracyCoverage:
    """Empirical fraction of predictions within the chemical-accuracy band.

    Returns ``mean(|prediction - reference| <= tolerance_hartree)``. The
    conventional band is *chemical accuracy* — 1 kcal/mol — the long-standing
    quantum-chemistry / DFT benchmark threshold (Pople, Rev. Mod. Phys. 71,
    1267 (1999)); use :meth:`chemical_accuracy` to pin that 1 kcal/mol
    tolerance.
    """

    def __init__(self, *, tolerance_hartree: float) -> None:
        """Configure the energy tolerance (in Hartree) defining chemical accuracy."""
        if tolerance_hartree <= 0.0:
            raise ValueError(f"tolerance_hartree must be > 0; got {tolerance_hartree}.")
        self.tolerance_hartree = tolerance_hartree

    @classmethod
    def chemical_accuracy(cls) -> ChemicalAccuracyCoverage:
        """Construct with the 1 kcal/mol chemical-accuracy tolerance.

        Pins ``tolerance_hartree`` to :data:`CHEMICAL_ACCURACY_HARTREE`
        (1 kcal/mol ≈ 0.0015936 Ha) — the conventional quantum-chemistry
        benchmark threshold (Pople, Rev. Mod. Phys. 71, 1267 (1999)).
        """
        return cls(tolerance_hartree=CHEMICAL_ACCURACY_HARTREE)

    def __call__(self, predictions: jax.Array, references: jax.Array) -> float:
        """Return the empirical fraction of predictions within the band.

        Args:
            predictions: Predicted energies (any broadcastable shape).
            references: Reference energies, broadcastable with ``predictions``.

        Returns:
            ``mean(|predictions - references| <= tolerance_hartree)`` as a
            Python ``float``.
        """
        return float(jnp.mean(jnp.abs(predictions - references) <= self.tolerance_hartree))


__all__ = [
    "CHEMICAL_ACCURACY_HARTREE",
    "ChemicalAccuracyCoverage",
    "DensityUncertainty",
    "EnergyUncertainty",
    "ExchangeCorrelationUncertainty",
]
