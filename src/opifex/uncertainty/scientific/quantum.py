"""Step-10 stubs for quantum-chemistry UQ surfaces.

Audit Migration Step 10 requires named interfaces so the Phase 7
capability registries can advertise quantum-chemistry UQ truthfully
without overclaiming behaviour. Every operational method raises
``NotImplementedError`` per Rule 6 (fail-fast). Constructors only
validate argument types / ranges.

Future implementers: see the audit's Migration Step 10 description
for the contract each stub captures (per-energy / per-density /
per-XC predictive distributions and chemical-accuracy coverage
diagnostics).
"""

from __future__ import annotations

import jax

from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001


_CANONICAL_MESSAGE = "Step 10 stub: see audit Migration Step 10"


class EnergyUncertaintyStub:
    """Stub: predictive uncertainty over total / per-state energies.

    Audit contract: callable that takes a batch of energy predictions
    and returns the matching :class:`PredictiveDistribution`. Reports
    ``units`` so downstream consumers can render bands in either
    Hartree or kcal/mol without re-converting.
    """

    def __init__(self, *, method: str, units: str = "hartree") -> None:
        if not method:
            raise ValueError("method must be a non-empty string.")
        if units not in {"hartree", "kcal_per_mol", "ev"}:
            raise ValueError(
                f"units must be one of 'hartree' / 'kcal_per_mol' / 'ev'; got {units!r}."
            )
        self.method = method
        self.units = units

    def __call__(self, energies: jax.Array) -> PredictiveDistribution:
        """Compute the per-energy predictive distribution; not yet implemented."""
        del energies
        raise NotImplementedError(_CANONICAL_MESSAGE)


class DensityUncertaintyStub:
    """Stub: predictive uncertainty over electron density fields.

    Audit contract: callable that takes a density tensor on the
    ``grid_axes`` mesh and returns the per-grid-point
    :class:`PredictiveDistribution`.
    """

    def __init__(self, *, grid_axes: tuple[str, ...]) -> None:
        if len(grid_axes) == 0:
            raise ValueError("grid_axes must contain at least one axis name.")
        self.grid_axes = grid_axes

    def __call__(self, density: jax.Array) -> PredictiveDistribution:
        """Compute the density predictive distribution; not yet implemented."""
        del density
        raise NotImplementedError(_CANONICAL_MESSAGE)


class ExchangeCorrelationUncertaintyStub:
    """Stub: predictive uncertainty over an XC functional's outputs.

    Audit contract: callable that takes XC-functional predictions and
    returns a :class:`PredictiveDistribution` tagged with the
    ``functional_family`` (LDA / GGA / meta-GGA / hybrid / NEURAL).
    """

    def __init__(self, *, functional_family: str) -> None:
        if not functional_family:
            raise ValueError("functional_family must be a non-empty string.")
        self.functional_family = functional_family

    def __call__(self, xc_predictions: jax.Array) -> PredictiveDistribution:
        """Compute the XC predictive distribution; not yet implemented."""
        del xc_predictions
        raise NotImplementedError(_CANONICAL_MESSAGE)


class ChemicalAccuracyCoverageStub:
    """Stub: fraction of predictions within the chemical-accuracy band.

    Audit contract: callable that returns the empirical fraction of
    ``|prediction - reference| <= tolerance_hartree``.
    """

    def __init__(self, *, tolerance_hartree: float) -> None:
        if tolerance_hartree <= 0.0:
            raise ValueError(
                f"tolerance_hartree must be > 0; got {tolerance_hartree}."
            )
        self.tolerance_hartree = tolerance_hartree

    def __call__(self, predictions: jax.Array, references: jax.Array) -> float:
        """Compute the coverage fraction; not yet implemented."""
        del predictions, references
        raise NotImplementedError(_CANONICAL_MESSAGE)


__all__ = [
    "ChemicalAccuracyCoverageStub",
    "DensityUncertaintyStub",
    "EnergyUncertaintyStub",
    "ExchangeCorrelationUncertaintyStub",
]
